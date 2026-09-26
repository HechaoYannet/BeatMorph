> 状态：🔵 实施中（M7.1–M7.8 代码与默认 CI 测试已落地；逐条见 §6 的「实施状态」列，**Lightning 后端尚未在装齐 train extra 的环境实跑**）｜ 阶段：Phase 1（环境自检）/ Phase 2（训练栈与门禁）｜ 负责：基础设施组
> 对应代码：`beatmorph/infra/`、`configs/`、`beatmorph/cli/train.py` ｜ 对应奠基章节：§5、§9

# Plan 07 — 训练基础设施（Lightning / Hydra / G1-G4 门禁 / 环境自检）

## 1. 目标与范围

### 1.1 交付什么

1. **训练栈**：PyTorch Lightning 训练循环骨架（`LightningModule` 基类、`Trainer` 构造、bf16 混合精度、梯度裁剪、单卡起步 / 多卡 FSDP 预留）。
2. **Hydra 组合式配置**：`configs/` 的 structured config、实验入口组合、**派生量的单一事实源**校验。
3. **G1-G4 门禁的执行契约**：`beatmorph/infra/sanity.py` **已存在**（实读），本 plan 规定它**如何被强制接入**训练流程——**任何新训练目标/损失，在扩大数据规模之前，G1-G4 必须全绿，且结果写入训练日志**。
4. **环境自检（env doctor）**：把「环境可用性」变成一条可失败的质量门禁。
5. **checkpoint / 实验追踪 / 日志规范**：产物目录、恢复语义、门禁文本落盘。
6. **特征缓存元数据校验**：`{rate, sample_rate, layer, model_rev, duration_s}` 随缓存落盘并在加载时校验（RFC-0029 §7-2）。
7. **数据来源与用途的强制记录**（数据合规硬约束 ③，2026-08-05 裁定）：训练配置中 `data.provenance` 为**必填**（来源 / 用途 / 脚本版本 / 获取时间），缺失或为空即**启动失败**；解析后的 provenance 随实验产物落盘（§3.2）。

### 1.2 不交付什么

- 不实现任何模型与损失（Plan 01/03/04）；本 plan 只提供**承载它们的骨架与门禁**；
- 不实现评估算法（Plan 06），只负责把评估产物写进实验目录与日志；
- 不定义 CLI 的用户语义（Plan 08），只提供 `beatmorph-train` 的现有入口延续；
- **不决定**任何帧率/单位数值：一切由特征提取器 config 派生（红线 7）。

## 2. 与奠基文档对应

| 本 plan 内容 | 奠基出处 | 关系 |
| --- | --- | --- |
| Python 3.11 / uv / PyTorch 2.5+ / Lightning / Hydra / TensorBoard（`wandb.entity=null`） | BasePlan §5 表格 | 直接落实 |
| G1-G4 四道门禁的判据与用途 | BasePlan §9、POSTMORTEM §6 | 直接落实（代码已在 `infra/sanity.py`） |
| 门禁结果写入训练日志、扩数据前必须全绿 | RFC-0029 §7-3、CLAUDE.md §5.8 | 硬约束（本 plan 的执行契约） |
| 环境可用性是质量门禁的一部分 | BasePlan §5「环境纪律（2026-08-05 教训）」 | 直接落实（历史事故：`.venv` 指向已删除的 conda 环境 → 整体不可用 → 唯一能证伪帧率的测试长期无法运行） |
| 帧率必须由 config 派生并断言，不得硬编码 | CLAUDE.md 红线 7、RFC-0029 §7-1、POSTMORTEM §2.2/§7-1 | 硬约束 |
| 特征缓存元数据 + 加载校验 | RFC-0029 §7-2、POSTMORTEM §7-2 | 硬约束 |
| 测试中的 mock 不得固化物理常量 | RFC-0029 §7-6、AGENTS.md §3.3 | 硬约束 |
| ✅ **数据合规已裁定（2026-08-05）：训练可启动**；硬约束 = ① 最终不发布权重 ② 数据不入库 ③ **脚本记录来源与用途** ④ 发布权重前重新裁定 | BasePlan §4.4、CLAUDE.md 红线 5 附注、RFC-0029 §7-7/§8.3 Q11b | 训练入口的守卫**不再是合规闸门**，改为「provenance 必填」；不提供任何权重发布路径 |

**偏离声明**：无。本 plan 未改变任何模型/损失/契约；仅把已有门禁与已有教训固化成可执行流程。

## 3. 接口契约

### 3.1 门禁 API（`beatmorph/infra/sanity.py`，**已存在，签名冻结**）

```
StepFn = Callable[[], float]        # 调用方自己跑一步优化并返回标量 loss

@dataclass(frozen=True)
GateResult:  name: str; passed: bool; detail: str

overfit_single_batch(step_fn, *, steps=300, target_loss=0.05, target_ratio=0.1) -> GateResult
shuffled_target_control(step_fn_real, step_fn_shuffled, *, steps=300, min_gap_ratio=0.05) -> GateResult
constant_baseline_gate(model_loss, baseline_loss, *, min_improvement=0.1) -> GateResult
frame_rate_gate(frames, duration_s, frame_rate, *, tol_frames=2) -> GateResult
summarize(results: list[GateResult]) -> str      # 可直接贴进训练日志的多行文本
```

- **设计约束（必须保持）**：`sanity.py` **范式中立**——不 import torch、不认识任何模型/数据集/tokenizer，只吃调用方给的 `step_fn`。新范式（泊松 NLL / 掩码补全 / 各对照臂）接入时**只写 `step_fn`，不得改本模块**（改签名须先改本 plan，必要时走 RFC）。
- **默认值可覆盖但不可忽略**：`target_loss` / `min_improvement` 等阈值对 `step_fn` 返回量的**量纲与标度**敏感（泊松 NLL 的量级随场体积变化）→ 每次使用必须显式记录实际生效的阈值（§9-2）。

### 3.2 训练入口契约

```
beatmorph-train --config-name <stage> [overrides...]
  --gates            # 强制先跑 G1-G4，结果落盘并在 FAIL 时以非 0 退出码中止
  --gates-only       # 只跑门禁，不进入正式训练
```

- **fail-closed 语义**：当配置 `infra.gates.required=true` 时，若实验目录缺少 `gates.txt` 或其中存在 FAIL，训练**必须拒绝以「扩数据规模」的配置启动**（退出码非 0）。「扩数据规模」的判定基准写入配置（如 `data.max_samples` 超过冒烟规模即视为扩大），避免成为口头约定。
- **数据来源与用途（必填，硬约束 ③）**：`data.provenance` 是 structured config 的**必填字段**——`source`（来源：数据源与清单快照/版本）、`purpose`（用途：训练 / 消融 / 冒烟）、`script_rev`（获取与处理脚本标识 + 版本）、`acquired_at`（获取时间）；缺字段或为空 → **启动即失败**，解析后的 provenance 落 `data_provenance.json`，使任一实验都可追溯到数据来源与用途。
- `runs/<experiment>/<timestamp>/` 产物（**固定清单，缺一即视为实验不可信**）：
  - `config.yaml`（Hydra 解析后的完整配置，含 override **与 provenance**）；
  - `gates.txt`（`summarize()` 原文）；
  - `data_provenance.json`（数据来源与用途的解析后快照）；
  - `checkpoints/`；
  - `logs/`（TB event 文件 + 文本日志）；
  - `metrics.json`（评估产物，格式见 Plan 06）。

### 3.3 派生常量的单一事实源

- 帧率等物理量**只允许**在 `beatmorph/core/contracts` 定义一次，并以**派生式**表达：
  `MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`；
  `field.dx = RPE_STAGE_WIDTH / field.x_bins`。
- `infra/` **不得**复制第二份帧率/单位常量。**判据**：全仓 grep 不允许出现第二处字面量帧率（POSTMORTEM §2.2 的事故正是同一常量被独立复制了三份）。
- 启动时打印派生量的取值并断言（红线 7 的「派生 + 断言」两半都要）。

### 3.4 特征缓存元数据

```
FeatureCacheMeta:  # RFC-0029 §7-2 原文要求
  rate: float; sample_rate: float; layer: int; model_rev: str; duration_s: float
```

- 加载缓存时**校验** `rate` 与当前 config 派生值一致；不一致**直接报错**（不是 warning）——否则「秒→帧」会在下游静默错位。

## 4. 内部设计

### 4.1 训练栈

- `LightningModule` 基类：统一 `training_step` / `validation_step` / `configure_optimizers` 的接口，统一 log 口径（哪些量进 TB、什么频率）；
- `build_trainer(cfg)`：从 Hydra 配置构造 `Trainer`（precision `bf16-mixed`、gradient clipping、devices/strategy）；单卡起步，多卡转 FSDP（BasePlan §5）；
- **损失与优化器不进基类**：各 Stage 自带，避免把某个范式的假设写进基础设施。

**实施说明（2026-09-27）**：`beatmorph/infra/train_loop.py` 提供 **torch 参考循环**（默认后端
`run.backend=torch`，无额外依赖、可在默认 CI 跑），`beatmorph/infra/lightning_module.py` 提供
**Lightning 目标栈**（`run.backend=lightning`，缺依赖时抛 `MissingTrainingDependency` 并给出安装命令，
**不静默回落**）。二者的损失与优化器都取自本仓实现（`generation` 的泊松 NLL + `AdamW`），
基类里没有范式假设。

### 4.2 Hydra 配置

- 组合式：`configs/{stage}/`、`configs/model/`、`configs/data/`、`configs/train/`、`configs/infra/`；入口以 `--config-name <stage>` 选择；
- 使用 **structured config（dataclass schema）**：缺字段/类型错在**启动时**失败，而不是跑到第 300 步才 KeyError；
- 配置中出现的派生量必须写成表达式并在启动时断言（§3.3）；
- **旧配置处置**：`configs/` 中 v2.x（osu!mania 时代）的配置随范式迁移退役，**由 infra-agent 在迁移时统一处置**（新增/归档），其它 agent 不并行改动 `configs/`（AGENTS.md §3.4）。

### 4.3 门禁执行器

```
run_gates(*, step_fns: GateStepFns, out_path: Path, cfg) -> list[GateResult]
  # 1) 组装四个 GateResult（G1-G4）
  # 2) 写 out_path（summarize() 原文 + 生效阈值 + git rev + data rev）
  # 3) 写 TB 标量（gate/G1_pass 等）供曲线面板对照
  # 4) 任一 FAIL -> 抛出并中止（退出码非 0）
```

四道门禁在**新范式**下的具体接法（判据不变，只换 `step_fn`）：

| 门禁 | 判据（BasePlan §9） | 在新范式下的接入 |
| --- | --- | --- |
| G1 单 batch 过拟合 | 1-4 个样本上 loss 打到接近 0 | `step_fn` = 对同一 batch 反复 backward/step；**必须能打穿**，打不穿说明通路坏（与数据量无关） |
| G2 打乱标签对照 | shuffle 目标后 loss **必须显著变差** | 提供**同模型同输入**的两个 `step_fn`（真标签 / 打乱标签）；对本范式，打乱的对象是**强度场目标**而非音频 |
| G3 常数基线 | 模型 loss 显著优于 `λ = N/abs(Ω)` | **基线不是 `λ ≡ 0`**（后者泊松 NLL = +∞，应写成契约断言）；`abs(Ω) = k * t_bins * x_bins * sides * channels`（Plan 00 §3.7），`k` 随谱变化 → 标度是 per-chart 的（§9-3） |
| G4 契约断言 | 帧率/形状由 config 派生并断言 | 直接 `frame_rate_gate(frames, duration_s, frame_rate)`，`frame_rate` **由 config 派生传入**；并含**「场网格 ↔ 秒」往返无损**（多 BPM 段，Plan 03 M12）——该断言进默认 CI、不依赖权重 |

**触发条件（写死，不靠记性）**：① 新增/修改任何训练目标或损失；② 任何一次把数据规模扩到超过冒烟规模。二者任一发生时，`gates.txt` 必须先存在且全绿。

**实施说明**：「冒烟规模」的判据写进配置（`gates.smoke_max_samples`，默认 8），
`data.max_samples` 为 `null`（全量）或超过它即视为扩大规模；未跑门禁的冒烟运行会在
`gates.txt` 里留下**明确不可判定**的记录（不含 `[PASS]`/`[FAIL]` 行），因此「文件存在」
永远不会被误读成「门禁通过」。合成批次来源 `infra/smoke.py` 走**同一条**目标构建路径
（`PhigrosChart` → `build_target`），音频帧数由契约帧率派生，所以 G4 在冒烟路径上同样有内容。

### 4.4 环境自检（env doctor）

动机（BasePlan §5 环境纪律）：`.venv` 曾因指向已删除的 conda 环境而**整体不可用**，导致「唯一能证伪帧率的测试」长期无法运行——**环境不可用会让所有其它门禁静默失效**。

检查项（全部可离线、可在无 GPU 环境运行）：

| # | 检查 | 失败判据 |
| --- | --- | --- |
| E1 | 当前解释器是否为项目 `.venv`（`sys.executable` vs `.venv` 路径） | 不一致 → FAIL（防止用系统 Python 跑出「另一个环境」的结论） |
| E2 | `.venv/pyvenv.cfg` 的 `home` / 基础解释器**是否存在** | 指向不存在的路径 → FAIL（**历史事故的根因**） |
| E3 | `uv.lock` 与 `pyproject.toml` 是否一致 | 不一致 → FAIL（可复现性） |
| E4 | 关键依赖 import 探针（`numpy` 必需；`torch` 可选） | 必需项缺失 → FAIL；可选项缺失 → **UNKNOWN**（不得报 PASS） |
| E5 | 契约常量可导入且派生式成立 | 失败 → FAIL |

- **三态结果**：PASS / FAIL / **UNKNOWN**。不确定一律 UNKNOWN，**不得**默认成 PASS（本项目的通则：不允许把「不知道」写成「没问题」）。
- 作为 `make test-fast` 之前的轻量检查（或 CI 步骤）运行，退出码非 0 表示环境不可信。

### 4.5 checkpoint 与恢复

- 保存策略（步数/时间）由配置决定；保留最近 N 个 + 最优 N 个；
- 恢复时必须校验：`config.yaml` 哈希、`gates.txt` 存在性、`data_rev`；不一致则**拒绝恢复并说明原因**（否则会用「环境已变」的实验续训出不可信结论）；
- 大文件不入库（红线 5）：`runs/` 在 `.gitignore` 中。

### 4.6 日志

- 生产路径一律 `from beatmorph.core.logging import get_logger`，**禁止裸 print**（CLAUDE.md §4.6）；
- TB 标量命名规范：`train/*`、`val/*`、`gate/*`；门禁的布尔值与 detail 同时落 `gates.txt`；
- 训练日志（`docs/TRAINING_LOG_*.md` 模板）**必须包含 `summarize()` 的原文**（RFC-0029 §7-3「结果写入训练日志」的落地形态）。

## 5. 依赖关系

| 方向 | 模块 | 内容 |
| --- | --- | --- |
| 上游 | Plan 00 `core/contracts` | 派生常量与契约断言（`MERT_SAMPLE_RATE_HZ` / `MERT_CONV_STRIDE_PRODUCT` / `RPE_STAGE_WIDTH`） |
| 上游 | Plan 01 `audio/` | `output_frame_rate()` 由主干 config 派生 |
| 上游 | Plan 02 `data/` | 数据集与特征缓存（本 plan 提供元数据校验） |
| 消费者 | Plan 03/04 | 模型与损失接入 G1-G4（**必须**） |
| 消费者 | Plan 06 `eval/` | `metrics.json` 落盘与实验目录 |
| 消费者 | Plan 08 `cli/` | `beatmorph-train` 入口与 `--gates` 转发 |
| 外部 | `pytorch-lightning` / `hydra-core` / `omegaconf` / `tensorboard` / `wandb` | 均已在 `pyproject.toml`（实读确认） |
| 外部 | `uv` | 环境与锁文件 |

## 6. 里程碑与验收

> **门禁硬性要求（本 plan 的核心交付）**：**任何新增训练目标或损失**（B1 focal、B2 泊松 NLL、B3/B4 的交叉熵、B5 掩码离散扩散训练目标，以及任何后续新增），在**扩大数据规模之前**必须通过 **G1-G4**（`beatmorph/infra/sanity.py`）且 `summarize()` 输出**写入训练日志**；实现该门禁的强制接入是本 plan 的 M7.2。**门禁未绿不得扩大数据规模**（BasePlan §9、CLAUDE.md §5.8、RFC-0029 §7-3、AGENTS.md §4）。

| # | 里程碑 | 可量化验收 | 实施状态 |
| --- | --- | --- | --- |
| **M7.1** | **环境自检**（最高优先，因历史事故） | 两个场景都必须可复现：① 正常环境 → E1-E5 全 PASS，退出码 0；② **人为把 `.venv/pyvenv.cfg` 的基础解释器指向不存在的路径** → E2 判 FAIL，退出码非 0，且诊断文本指出具体路径。检查本身**不依赖 GPU / 权重**，在默认 CI 中运行（不得被标记跳过） | ✅ `infra/env_doctor.py` + `tests/unit/infra/test_env_doctor.py`（含场景②故障注入）；CLI `python -m beatmorph.infra.env_doctor`，退出码 0/1/2 = 全 PASS / 有 FAIL / 有 UNKNOWN |
| **M7.2** | **门禁执行器接线**（核心） | `beatmorph-train --gates` 产出 `gates.txt`，内容含四个 `GateResult` 的 name/passed/detail 与生效阈值；**负例回归测试**：用一个故意断梯度的 `step_fn`（如 `loss.detach()`）使 G1 FAIL，训练**被中止**且退出码非 0 | ✅ `infra/gates.py` + `cli/train.py`（`--gates` / `--gates-only`）；`tests/unit/infra/test_gates.py`、`tests/integration/test_train_entry.py`（退出码 5） |
| **M7.3** | Hydra 配置骨架 | structured config 缺字段/类型错 → **启动即失败**（在 1 秒内，不进入训练循环）；**`data.provenance`（来源/用途/脚本版本/获取时间）缺失即视为缺字段 → 启动失败**；配置中的派生量断言在启动时执行；`config.yaml` 完整落盘（含 override） | ✅ `infra/config/`（schema + 严格加载）；`configs/smoke.yaml`、`configs/phigros_masked.yaml`；provenance 与 plan 02 的 `Provenance` 同 schema（§9-7 已定） |
| **M7.4** | 派生量单一事实源 | 全仓 grep 断言：字面量帧率只出现在 `core/contracts` 一处（可写成 CI 检查）；启动日志打印派生帧率与 `dx` | ✅ `infra/derive.py`（`format_derived_banner` + `scan_derived_literals` 全仓扫描，进默认 CI） |
| **M7.5** | checkpoint 与恢复 | 正常恢复：loss 曲线连续；**篡改 `config.yaml` 后恢复被拒绝**并给出差异字段（负例测试） | ✅ `infra/checkpoint.py`：配置指纹 + 逐字段 diff + 门禁全绿 + data_rev 三重校验 |
| **M7.6** | 特征缓存元数据校验 | 构造两份缓存：`rate` 与 config 派生值一致 → 通过；不一致 → **报错**（不是 warning）；单测**不依赖权重**（用手写的迷你缓存文件） | ✅ `infra/feature_cache.py`：配置期望帧率必须等于契约派生值，否则直接报错 |
| **M7.7** | 实验产物与日志 | `runs/<exp>/<ts>/` **六件套**齐全（§3.2，含 `data_provenance.json`）；TB 中可见 `gate/*` 标量；训练日志模板含 `summarize()` 原文与 provenance 摘要 | ✅ `infra/artifacts.py` 六件套 + `RunArtifacts.assert_complete()`；tensorboard 缺失时只告警（gates.txt 仍是权威记录） |
| **M7.8** | 复现性 | 固定 seed 下同配置两次运行的前 2 步 loss 逐位一致（断言）；`runs/` 路径不覆盖历史实验 | ✅ 时间戳目录 + 后缀顺延；`tests/integration/test_train_entry.py` 另断言两次运行的前 2 步 loss 逐位一致 |

## 7. 风险与缓解

| 风险 | 来源 | 本模块的缓解 |
| --- | --- | --- |
| **R-7** 物理常量/单位漂移（25 Hz 类） | BasePlan §6、POSTMORTEM | 派生式单一事实源（M7.4）+ 启动打印与断言 + G4 契约测试进默认 CI（不依赖权重）+ mock 不得固化常量（RFC-0029 §7-6） |
| **R-4** 稀疏目标塌陷 / 训练不收敛 | BasePlan §6 | G1-G4 强制接入（M7.2）；G3 用 `λ = N/abs(Ω)` 而非 0；`λ ≡ 0` → NLL `inf` 写成契约断言 |
| **R-2** 数据合规（**已裁定**，残余风险 = 不得发布权重） | BasePlan §4.4、CLAUDE.md 红线 5 附注 | 训练入口的守卫改为 **`data.provenance` 必填**（M7.3）+ provenance 落盘（M7.7）；**不提供任何权重发布/分发路径**；发布权重前必须重新裁定 |
| **R-1** MERT 表征不足 | BasePlan §6 | 本模块不解决，但保证 Adapter 可训练与 config 可切换（不写死主干） |
| **环境不可用导致所有门禁静默失效** | BasePlan §5 环境纪律（历史事故） | M7.1 环境自检（三态结果，UNKNOWN 不等于 PASS）；**建议**由决策者以 RFC 将「环境可用性」补为 BasePlan §6 的 **R-9**（见 §9-6） |
| 门禁沦为口头约定 | 本 plan 的制度风险 | fail-closed 语义 + 触发条件写死 + 负例回归测试（M7.2）+ 产物**六件套**缺一即视为实验不可信 |

## 8. 测试策略

**单元（默认 CI，无权重、无 GPU）**

- `sanity.py` 自身：四个门禁的**边界行为**（刚好达标 / 刚好不达标 / steps 合法性与 `steps < 1` 抛错）；`summarize()` 的 FAIL 计数；
- env doctor：E1-E5 的纯函数部分 + 场景 ② 的故障注入（临时目录构造坏 `pyvenv.cfg`）；
- 配置校验：缺字段/类型错/派生式断言失败的负例；
- 缓存元数据校验：一致/不一致两向；
- **mock 纪律**：测试的假 `step_fn` 可以用常数损失，但**不得**在 fixture 中固化帧率/坐标字面量；需要帧数时必须引用契约常量（AGENTS.md §3.3、RFC-0029 §7-6）。

**集成**：`beatmorph-train --gates-only` 在 dummy 模型上跑通并落盘（可 CI）；训练入口的 fail-closed 行为（缺 `gates.txt` 时拒绝启动）；**缺 `data.provenance` 时启动被拒**（负例，可 CI）。

**e2e（`slow` / `gpu`）**：真实 MERT + 真实数据的小规模冒烟（G4 帧率契约在**有**权重时也必须成立）。

**CI 契约（不可跳过）**：帧率契约测试必须进默认 CI 且**不依赖权重或 GPU**——「只有拿到权重才能跑」的断言等于没有断言（CLAUDE.md §4 测试约定）。

## 9. 开放问题

1. **`.venv` 当前实际状态未查证**：历史事故（指向已删除的 conda 环境）是否已彻底消除、是否仍存在 conda/uv 混用残留，需在 M7.1 落地时实测并记录。
2. **G1 的 `target_loss` 与 G3 的 `min_improvement` 对泊松 NLL 是否适用**：`sanity.py` 现有默认值（`steps=300`、`target_loss=0.05`、`min_improvement=0.1`）是在旧的回归/MSE 语境下定的；泊松 NLL 是**正量且随场体积 `abs(Ω)` 缩放**，绝对阈值可能永远无法达标或过于宽松。需为点过程定义**相对判据**（如相对首步的下降比例、相对 G3 基线的改进比例）并记录生效阈值。
3. **`abs(Ω)`（场体积）的标度影响（定义已由 Plan 00 §3.7 给出）**：`abs(Ω) = k * t_bins * x_bins * sides * channels`，其中 `k`（判定线数）**随谱变化**（实测中位 30、范围 2–82）→ NLL 与 G3 基线的绝对标度是 **per-chart** 的。这直接冲击 §9-2 的绝对阈值问题：门禁需要**归一化口径**（如 NLL 除以事件数、或相对基线的改进比例）。
4. **门禁样本数与步数**：G1 用 1-4 个样本（BasePlan §9），G2/G3 在同一批上跑还是用更大子集，未定。
5. **多卡 FSDP 的引入时机**：单卡 ≥ 16 GB 起步（BasePlan §5），何时需要多卡未定。
6. **是否把「环境可用性」补为 BasePlan §6 的 R-9**：本 plan 只能缓解，改风险表须走 RFC（建议由决策者裁定）。
7. ~~**`data.allow_training` 守卫的具体形式**~~ **已随裁定落地（2026-08-05）**：合规不再是训练的前置闸门；守卫改为 **`data.provenance` 必填校验**（§3.2 / M7.3）。
   **✅ 已定（实施期，2026-09-27）**：训练配置的 `data.provenance` **直接采用 plan 02 `beatmorph.data.phira.client.Provenance` 的 schema**（`source` / `query` / `fetched_at` / `purpose` / `script` / `script_version` / `chart_id_min` / `chart_id_max`），配置侧只做一个**字段逐字对应的镜像** `DataProvenanceConfig`，并由 `tests/unit/infra/test_config_schema.py::test_provenance_schema_matches_data_side` 断言两侧不漂移。此即本 plan §3.2 想要的 `source` / `purpose` / `script_rev`（= `script` + `script_version`）/ `acquired_at`（= `fetched_at`）。实验**运行**用途（train / ablation / smoke）另设 `run.purpose`，与数据**获取**用途（`ManifestPurpose`）是两层，两者都落盘。
8. **训练日志的载体**：`docs/TRAINING_LOG_*.md`（入库的文档）与 `runs/*/gates.txt`（不入库的产物）二者关系——建议「产物为原始证据、文档为摘要 + 链接」，但需与仓库文档习惯对齐。
9. **`configs/` 中 v2.x 旧配置的处置**（删除 vs 归档分支），属 infra-agent 职责范围，但需与 RFC-0029 §4.3 的归档策略一致。
   **部分处置（实施期）**：新增 `configs/smoke.yaml`（合成数据 / 门禁冒烟）与 `configs/phigros_masked.yaml`（真实数据全量）；v2.x 的 `configs/model/mert.yaml` 仍保留（MERT Stage 0 配置本身没过时），但其字段是否要并入 structured schema 未定。

**实施期新增（2026-09-27）**：

10. **默认训练后端是 torch 参考循环，Lightning 是可选目标栈**：`run.backend=torch` 为默认，`lightning` 需要 `uv sync --extra train`。理由：门禁与契约级断言必须进默认 CI，而 `pytorch-lightning` 属可选依赖（CLAUDE.md §4）。代价：**Lightning 路径目前只有「缺依赖即给可操作报错」的行为被测试覆盖，真机未实跑**（本机未安装 train extra，见 §9-11）。是否把它提升为「门禁必须跑的后端」需决策者裁定。
11. **train extra 的安装未完成（环境事实）**：本机 `uv sync --extra train` 因网络原因未完成，因此 env doctor 的 E4 恒为 **UNKNOWN**（缺 `pytorch-lightning` / `tensorboard`），TensorBoard 标量也只写不进去（代码只告警）。这不影响门禁结论（权威记录是 `gates.txt`），但意味着「TB 曲线」这条证据链尚未实跑过。
12. **G2 的样本数是判据的一部分**：样本太少时打乱臂可以直接背样本、让对照退化成空转（实施期实测：1 个样本时打乱臂 loss **低于**真实臂）。已把 `gates.shuffle_samples`（默认 16）写进配置并落进 `gates.txt` 的生效阈值。真实数据上要多少样本才够，需要实测曲线。
13. **跨谱批次受「一个 FieldBatch 只带一个 FieldGrid」约束**：不同 BPM 结构的窗口不能混批（否则测度 `J` 与积分项会静默错掉）。
    当前 `collate_field_batch` 的做法是**不一致即抛**（fail-closed），训练配置默认 `batch_size=1`。
    **已落地缓释（2026-09-27）**：`ManifestBatchSource` 按 `ChartPairDataset.grid_key(i)` 的网格身份**分桶并轮转组批**
    （桶内同 `bpm_eff` ⟹ 同批逐格 J 相同），回归测试见
    `tests/integration/test_train_entry.py::test_manifest_source_batches_by_grid_identity`（夹具自带 120/180 两段 BPM）。
    仍**未解决**的是「同一批里混合不同 BPM 段」——那需要「给 `FieldBatch` 加 per-sample 网格」或「在 loss 里按样本取 J」，
    属契约级问题，须开 RFC（plan 04 §9 相关）。
14. **门禁预算在真实数据上是否够用未验证**：G1 的判据实际生效的是**相对**判据（`target_ratio × 首步`，因为泊松 NLL 的下界是事件数而不是 0，§9-2 的预判已被证实），但「真实数据上 120-300 步能否打穿」只能等 plan 02 的真实特征缓存就绪后实测。
15. **真实数据上的门禁内存墙（本轮实测，2026-09-27）**：G2 的对照臂一次要 collate `shuffle_samples` 个窗口，
    而真实窗口的每批元素数 ≈ `samples × K × T × X × S × C`（本配置 `x_bins=128`、`t_window=192`；
    实测批次 K=63）。`shuffle_samples=16`（合成数据下的默认值）时，门禁装配在 model forward 上申请
    **5.4 GB 连续内存**直接失败（`DefaultCPUAllocator: not enough memory`，本机 31 GB RAM）。
    处置：`configs/phigros_masked.yaml` 显式写 `shuffle_samples: 4` 并注明代价——
    G2 的样本数变少 ⇒「打乱臂直接背样本」的风险回升（§9-12）。
    **待复核**：真实数据上 G2 需要多少样本才够、以及门禁是否应支持把 G2 分批（而不是一次性 collate 全部样本）。
16. **train extra 已装齐（本轮更新 §9-11）**：`uv sync --extra audio --extra train --extra data --extra viz` 在走代理后完成，
    env doctor 的 **E4 由 UNKNOWN 变为 PASS**（`pytorch_lightning` / `tensorboard` 可导入）。
    仍未做的是**Lightning 后端真机实跑**（`run.backend=lightning`）与 TB 标量真实写入；权威记录仍是 `gates.txt`。
17. **装齐 train extra 之后 `mypy --strict` 才暴露的 3 处第三方无类型调用**（`mert.py`：`AutoFeatureExtractor.from_pretrained` ×2、
    `get_peft_model` ×1）：此前依赖未安装 ⇒ mypy 视为 Any ⇒ 静默通过。已按「窄 `type: ignore` + 理由注释」处理。
    教训：**「类型检查全绿」依赖环境快照**——CI 与本地必须装同一组 extras，否则门禁是环境相关的。
