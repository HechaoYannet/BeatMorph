"""M3：生成主干的 G1-G4 门禁（**标 slow，不进默认 CI，但必须实际跑过一次**）。

判据全部取 `beatmorph/infra/sanity.py` 的默认值（plan 04 §6.1："统一调用，判据取该模块默认值"）；
训练预算（步数 / 初始强度）由本文件显式给出并**在下方逐条说明理由**，因为泊松 NLL 的
下界**不是 0**（完美拟合时 loss ≈ 事件数 N），"末步 <= 0.1 x 首步" 只有在首步远离最优时
才是一个有意义的判据。

## 合成任务必须是**可学**的（否则 G2 会以另一种方式失效）

G2 抓的是「输入对目标零信息」。若合成任务本身超出当前架构能学到的范围，
真实臂与打乱臂会**同时停在边际解**上、loss 逐位相同，门禁于是变成空转。
本文件因此在任务设计上做了三轮实测筛选（结论见 plan 04 §9-17 / §9-18）：

- **G1** 用真实训练路径（`token_block + dilute` 遮盖，见 `TestTask.masked_batch`）：
  在 4 个样本上必须能被打穿。
- **G2** 用「把可见场复制到输出」的最小条件任务：真实臂能从输入读出每个 token 的
  事件格（实测 200 步达到理论最优），打乱臂把目标按格置换后**无法从输入读出**
  （只能背 16 个样本的映射）。训练预算取 100 步——实测步数加大后打乱臂开始**背样本**，
  两臂差距会缩到 1.05 门限以下（这是微型合成任务的固有性质，不是判据放宽）。
- **G3** 常数基线取 plan 03 的闭式 `N * (1 + log(|Omega| / N))`。
- **G4** 同时校验音频帧轴（MERT config 派生）与 tau 轴（BPM 与基本格派生）。

⚠️ 门禁的**第二个价值**已在本次落地中兑现：它当场抓出了遮盖通道的信息泄漏
（`mask == 1` 几乎等价于「此处有事件」，补全任务退化成照抄 mask）——
见 `OcclusionStats.mask_leak` 与 `beatmorph/generation/masks.py` 的模块 docstring。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT, NoteType
from beatmorph.core.contracts.tensors import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.field.grid import BEAT_SUBDIVISION, SECONDS_PER_MINUTE
from beatmorph.field.integrate import omega
from beatmorph.field.loss import constant_baseline_nll
from beatmorph.field.target import CHANNEL_INDEX
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from beatmorph.infra.sanity import (
    GateResult,
    constant_baseline_gate,
    frame_rate_gate,
    overfit_single_batch,
    shuffled_target_control,
    summarize,
)
from tests.unit.generation._builders import (
    TEST_AUDIO_DIM,
    TEST_BPM,
    audio_frames_for,
    make_grid,
    make_line_mask,
    make_tracks,
)

pytestmark = pytest.mark.slow

T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 2
#: G1 的样本数（plan 04 §6.2 M3：1-4 样本）
G1_SAMPLES = 4
#: G2 的样本数（越大越难「背样本」）
G2_SAMPLES = 16
#: G1 的优化预算（步数与首步强度都显式声明，理由见模块 docstring）
G1_STEPS = 300
G1_INITIAL_SCALE = 40.0
#: G2 的优化预算：100 步内真实臂已到理论最优，而打乱臂来不及背样本
G2_STEPS = 100
G2_INITIAL_SCALE = 10.0
LEARNING_RATE = 0.1

MODEL_CONFIG = ModelConfig(
    d_model=32,
    n_heads=2,
    n_layers=2,
    window=4,
    global_period=2,
    k_max=8,
    audio_dim=TEST_AUDIO_DIM,
)


@dataclass(frozen=True, slots=True)
class SyntheticTask:
    """可学合成任务的构造器（条件在所有样本间共享，杜绝「背音频指纹」的捷径）。"""

    samples: int
    seed: int

    def _shared_inputs(self, grid) -> tuple[torch.Tensor, torch.Tensor, int]:
        frames = audio_frames_for(grid)
        audio = (
            torch.randn(1, frames, TEST_AUDIO_DIM, generator=torch.Generator().manual_seed(3))
            .expand(self.samples, -1, -1)
            .contiguous()
        )
        tracks = (
            make_tracks(batch=1, k=K_LINES, t_line=grid.t_bins, seed=1)
            .expand(self.samples, -1, -1, -1)
            .contiguous()
        )
        return audio, tracks, frames

    def counts(self, grid, *, shuffle: bool) -> torch.Tensor:
        """每个 token 恰好一个事件，落在**随机**的 x 桶上（只能从可见场读出来）。"""
        shape = (self.samples, K_LINES, grid.t_bins, X_BINS, grid.sides, grid.channels)
        generator = torch.Generator().manual_seed(self.seed)
        counts = torch.zeros(shape, dtype=torch.int16)
        for batch in range(self.samples):
            for line in range(K_LINES):
                for tau in range(grid.t_bins):
                    x_index = int(torch.randint(0, X_BINS, (1,), generator=generator).item())
                    counts[batch, line, tau, x_index, 0, CHANNEL_INDEX[NoteType.TAP]] = 1
        if not shuffle:
            return counts
        # 打乱对照：把目标按格整体置换（输入与目标的关系被破坏，且无法从输入恢复）
        generator = torch.Generator().manual_seed(self.seed + 999)
        flat = counts.reshape(self.samples, -1)
        for batch in range(self.samples):
            order = torch.randperm(int(flat.shape[1]), generator=generator)
            flat[batch] = flat[batch][order]
        return flat.reshape(shape)

    def batch(self, grid, *, shuffle: bool = False, masked: bool = False) -> FieldBatch:
        """组装 batch；`masked=True` 时走真实的遮盖训练路径（G1 用）。"""
        audio, tracks, frames = self._shared_inputs(grid)
        counts = self.counts(grid, shuffle=shuffle)
        line_mask = make_line_mask(batch=self.samples, k=K_LINES)
        occlusion = None
        if masked:
            occlusion, _ = build_occlusion_batch(
                counts,
                ratio=0.5,
                seed=self.seed + 7,
                line_mask=line_mask,
            )
        return FieldBatch(
            audio_emb=audio,
            frame_rate=frames / max(grid.total_seconds, 1e-9),
            line_tracks=tracks,
            line_mask=line_mask,
            difficulty=torch.full((self.samples,), 15.0),
            grid=grid,
            counts=counts,
            occlusion=occlusion,
        )


def _fresh_model(grid, *, seed: int, scale: float) -> MaskedFieldModel:
    """固定初始化 + 抬高初始强度（G2 的两臂必须**逐位同起点**）。"""
    torch.manual_seed(seed)
    model = MaskedFieldModel(MODEL_CONFIG, grid)
    with torch.no_grad():
        model.head.cum_head.bias.fill_(scale)
    return model


def _runner(model: MaskedFieldModel, batch: FieldBatch) -> Callable[[], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def step() -> float:
        optimizer.zero_grad()
        output = model(batch)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()
        return float(output.loss.detach())

    return step


def test_g1_g4_gates_all_pass(capsys: pytest.CaptureFixture[str]) -> None:
    """M3：四道门禁全绿；`summarize()` 的输出可直接进训练日志。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    results: list[GateResult] = []

    # G1 单 batch 过拟合（真实训练路径：遮盖补全 + HT 重标定）
    task = SyntheticTask(samples=G1_SAMPLES, seed=5)
    g1_batch = task.batch(grid, masked=True)
    results.append(
        overfit_single_batch(
            _runner(_fresh_model(grid, seed=1, scale=G1_INITIAL_SCALE), g1_batch),
            steps=G1_STEPS,
        ),
    )

    # G2 打乱标签对照（同起点、同预算）
    g2_task = SyntheticTask(samples=G2_SAMPLES, seed=11)
    real_batch = g2_task.batch(grid)
    shuffled_batch = g2_task.batch(grid, shuffle=True)
    results.append(
        shuffled_target_control(
            _runner(_fresh_model(grid, seed=2, scale=G2_INITIAL_SCALE), real_batch),
            _runner(_fresh_model(grid, seed=2, scale=G2_INITIAL_SCALE), shuffled_batch),
            steps=G2_STEPS,
        ),
    )

    # G3 常数基线（闭式 N * (1 + log(|Omega| / N))）
    trained = _fresh_model(grid, seed=3, scale=G2_INITIAL_SCALE)
    step = _runner(trained, real_batch)
    model_loss = step()
    for _ in range(G2_STEPS - 1):
        model_loss = step()
    n_events = float(real_batch.counts.sum().item())
    baseline = constant_baseline_nll(n_events, omega(grid, n_lines=K_LINES))
    results.append(constant_baseline_gate(model_loss, baseline))

    # G4 合约：音频帧轴（MERT config 派生）与 tau 轴（BPM 与基本格派生）
    frames = int(real_batch.audio_emb.shape[1])
    duration = grid.total_seconds
    frame_rate = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
    results.append(frame_rate_gate(frames, duration, frame_rate))
    results.append(
        frame_rate_gate(grid.t_bins, duration, BEAT_SUBDIVISION * TEST_BPM / SECONDS_PER_MINUTE),
    )

    summary = summarize(results)
    with capsys.disabled():
        print(summary)
    assert all(result.passed for result in results), summary


def test_mask_leak_is_low_on_the_contract_path() -> None:
    """遮盖通道不得替模型回答问题：`mask_leak_tokens` 必须远低于字面口径的 1.0。

    这条断言是 G2 之所以有意义的前提（见 `OcclusionStats` 与 plan 04 §9-18）：
    若 `mask == 1` 几乎等价于「此处有事件」，真实臂与打乱臂会给出**逐位相同**的 loss。
    """
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    counts = SyntheticTask(samples=G2_SAMPLES, seed=11).counts(grid, shuffle=False)
    _, contract = build_occlusion_batch(counts, ratio=0.5, seed=7)
    _, literal = build_occlusion_batch(counts, ratio=0.5, seed=7, token_block=False, dilute=False)
    assert literal.mask_leak == pytest.approx(1.0)
    assert contract.mask_leak < 0.05
    assert contract.ratio == pytest.approx(0.5, abs=0.05)
    assert literal.ratio == pytest.approx(0.5, abs=0.05)


def test_shuffled_target_really_destroys_the_pairing() -> None:
    """G2 的数据侧前提：打乱后事件总数不变、但输入对目标不再有信息。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    task = SyntheticTask(samples=G2_SAMPLES, seed=11)
    real = task.counts(grid, shuffle=False)
    shuffled = task.counts(grid, shuffle=True)
    assert int(real.sum()) == int(shuffled.sum())
    assert not torch.equal(real, shuffled)
    # 事件**落点**的重合率：两者都稀疏，直接比 (real == shuffled) 会被空格淹没，
    # 因此只在「双方都有事件」的格子上统计重合。
    n_events = int(real.sum().item())
    overlap = int(((real > 0) & (shuffled > 0)).sum().item())
    assert overlap / n_events < 0.2


def test_poisson_floor_is_below_the_g3_baseline() -> None:
    """G3 立论的数值前提：完美拟合的下界必须显著低于常数基线。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    counts = SyntheticTask(samples=G2_SAMPLES, seed=11).counts(grid, shuffle=False)
    n_events = float(counts.sum().item())
    cell_volume = float(grid.cell_volumes()[0])
    floor = n_events * (1.0 + math.log(cell_volume))
    baseline = constant_baseline_nll(n_events, omega(grid, n_lines=K_LINES))
    assert floor < 0.9 * baseline
    assert baseline > 0.0
