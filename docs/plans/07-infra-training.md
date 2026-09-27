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
    **⚠️ 2026-09-27 第三轮补充**：样本数只有在「两臂同批同损失」时才是判据的一部分——见 §9-19：接线曾让两臂**不同批也不同损失**，那一版的所有 G2 数字（含真实数据实跑）都不可引用。
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
    处置（第一版）：`configs/phigros_masked.yaml` 显式写 `shuffle_samples: 4` 并注明代价——
    G2 的样本数变少 ⇒「打乱臂直接背样本」的风险回升（§9-12）。
    **✅ 已解法（2026-09-27 第三轮，见 §9-20/§9-21）**：`make_step_fn(chunks=k)` 按样本维分段
    前向/反传，内存回到「一段样本」量级而不改判据；`gates.shuffle_chunks`（默认 4）落进
    `gates.txt` 的生效阈值。真实数据上「多少样本才够」的曲线**仍未实测**（CPU 上 16 样本 ×
    100 步 × 2 臂约 10 小时，须用 `--device cuda` 跑，见 §9-22）。
    **2026-09-27 第三轮实测（真实 200 行切片，第一轮真跑出门禁结论）**：
    沿用「训练 lr=3e-4 + 100 步」时四道结果是 G1 PASS、**G2 FAIL（真实 47335.7 vs 打乱 47678.0，差距 0.72% < 5%）**、
    **G3 FAIL（模型 305050 vs 常数基线 4461，差 68 倍）**、G4 PASS ⇒ **门禁预算本身没在真实配置上校准**：
    `configs/smoke.yaml` 是**专门校准过**的（lr=0.05 / bias=20 / 16 样本 / 100 步），而 `phigros_masked.yaml` 只给了训练 lr，
    100 步内两臂都停在初始点附近。把 lr 抬到 smoke 的 0.05 在真实场上**直接 NaN**（场的尺度不同，不能照搬）⇒ 只能加**步数预算**。
    ⚠️ **G3 的注意点**：`gates.initial_head_bias=20` 是为 **G1 的相对判据**服务的（首步必须远离最优），G3 臂复用它会先花大量步数
    把一个人为抬高的积分项压回去 ⇒ 若单纯加步数也不绿，**正确的修法是给 G3 臂单独的初值**（去掉一个属于 G1 的人为初值），
    **不是**放宽判据、也不是无上限加步数。

    **chunks 的内存/功耗实测（同一预算）**：`shuffle_chunks=4` → 显存 7.8/8.1 GB、GPU 利用率 100% 但**只有 28 W**、
    G3 预计算 100 步 **107 min 未完成**；`shuffle_chunks=16`（=1 样本/段）→ 峰值 **5.3 GB / 94 W**，同预算分钟级完成。
    ⇒ 与「每样本 O((K·T)²)、超线性来自内存与分配器抖动」的机制解释一致。
16. **train extra 已装齐（本轮更新 §9-11）**：`uv sync --extra audio --extra train --extra data --extra viz` 在走代理后完成，
    env doctor 的 **E4 由 UNKNOWN 变为 PASS**（`pytorch_lightning` / `tensorboard` 可导入）。
    仍未做的是**Lightning 后端真机实跑**（`run.backend=lightning`）与 TB 标量真实写入；权威记录仍是 `gates.txt`。
17. **装齐 train extra 之后 `mypy --strict` 才暴露的 3 处第三方无类型调用**（`mert.py`：`AutoFeatureExtractor.from_pretrained` ×2、
    `get_peft_model` ×1）：此前依赖未安装 ⇒ mypy 视为 Any ⇒ 静默通过。已按「窄 `type: ignore` + 理由注释」处理。
    教训：**「类型检查全绿」依赖环境快照**——CI 与本地必须装同一组 extras，否则门禁是环境相关的。

**实施期新增（2026-09-27 第三轮：门禁预算与 G2 接线）**

18. **合成夹具的条件必须携带目标信息（本轮，与 §9-19 配套）**：`SmokeBatchSource` 原先在
    **随机噪声**上撒事件——事件与条件独立 ⇒ 任何形式的 G2 都是空转（真实臂与打乱臂都停在边际解）。
    现把合成谱的**事件密度**写进 `audio_emb` 第 0 维（取自打乱**之前**的 counts），
    于是「真实臂能定位、打乱臂拿到与目标不符的条件」。这与 plan 04 §9-15 的方法学一致：
    **G2 的合成任务必须条件可学、打乱后不可学**。

19. **G2 的接线缺陷（本轮发现并修复，历史数字作废）**：`build_gate_inputs` 曾把 **G1 的遮盖补全臂**
    当作 G2 的真实臂——那是「`optim.batch_size` 个样本 + 重标定 `1/r` + 只监督被遮盖事件」，
    而打乱臂是「`gates.shuffle_samples` 个样本 + 全事件目标」。两者**既不同批也不同损失**，
    比值由批大小与重标定系数决定，对照失去意义（违反 §4.3 的「同模型同输入」）。
    **⇒ 此前所有 `gates.txt` 里的 G2 数字（含 2026-09-27 的真实数据实跑「605.2 vs 644.6」）一律不得再引用。**

    **修复分两步，第二步才是关键**：

    - 第一步（配对）：`GateInputs` 拆出 `step_fn_g2_real` / `step_fn_g2_shuffled`（G1 仍用
      `step_fn_real`），G3 另给一个**无遮盖**批（它的基线是全事件口径的闭式常数基线）。
    - 第二步（**输入必须真的钉住**）：两臂改为**同一个遮盖批**，打乱发生在
      `shuffle_hidden_counts` 里——只把**被遮盖格子内**的计数在遮盖集合内置换，
      于是可见场 `counts * ~occlusion` **逐位不变**，差异恰好是「上下文能否预测被遮盖事件」。
      为什么不能用无遮盖批：无遮盖时 `observed_counts()` 把 counts 原样喂回去，**输入就是目标**，
      打乱目标会连输入一起打乱，对照退化成「拟合真场 vs 拟合乱场」，测不出「输入对目标有没有信息」。
    - 实测（合成夹具，16 样本 / 100 步）：配对后 **真实 2972.8 vs 打乱 4309.2**（差距 **45%**），
      而旧接线给的是 1.065 的比例（由批大小与重标定系数决定）。

20. **G2/G3 的分批前向（§9-15 内存墙的解法）**：`make_step_fn(..., chunks=k)` 把批次按样本维切段，
    逐段前向+反传后**只走一次优化器**。等价性口径写在函数 docstring 里：损失按样本求和时
    （`full_poisson_loss` / `masked_poisson_loss` 的默认 `reduction="sum"`）梯度与标量 loss
    与不分段一致——实测 loss 相对差 2.3e-8、梯度最大绝对差 1.5e-5（≈ 尺度的 1e-7），只是 float32 求和顺序。
    `FieldBatch.slice_samples` / `split_samples` 提供切分（grid / 帧率 / 定义域**原样保留**）。
    回归测试：`tests/unit/infra/test_gate_chunking.py`。

21. **真实批次的每样本成本（实测，CPU）**：`B=1, K=24, T=192, X=128, S=2, C=5`（真实窗口，
    `d_model=256` / 6 层）⇒ counts 22.5 MB；**前向 6.9 s（峰值 +0.87 GB）**、
    **前向+反传 11.6 s（峰值 1.88 GB）**。⇒ 16 个样本一次性前向 ≈ 30 GB，正是本机 31 GB 内存上
    「5.4 GB 连续分配失败 / 换页」的成因；`chunks=4` 把它压回 ~8 GB。
    **代价是时间**：CPU 上「16 样本 × 100 步 × 2 臂」≈ 10 小时 ⇒ 真实预算曲线必须在 GPU 上跑。

22. **门禁的设备参数（本轮新增）**：`build_gate_inputs(..., device=...)` + `FieldBatch.to(device)` +
    CLI `--device` 转发，使真实批次的门禁可以跑在 GPU 上（默认仍是 `cpu`，行为不变）。
    ⚠️ 另发现**未修**的既有缺口：训练循环 `train()` 只对模型做 `.to(device)`、**不搬批次**，
    因此 `--device cuda` 在训练路径上会直接失败（当前默认 cpu，未被触发）。
    **✅ 已修（2026-09-27，训练 subagent）**：`train()` 现在同时搬运批次与模型（否则任务 C 的
    `--device cuda` 必崩）。

22b. **真实数据上的门禁预算被 global 层主导（本轮实测，详见 plan 04 §9-22）**：`shuffle_chunks=4`
    （4 样本/段）时一个 G3 预计算步（100 步）跑 **107 min** 未完成、整轮门禁外推 **6.7 h**；
    提到 `shuffle_chunks=16`（1 样本/段）后数据见下。原因是 global 层把 token 拉平成
    `(B, K*T, d)` 做全自注意力（**每样本** O((K·T)²)），**不是**跨样本耦合——
    样本间不互相注意（`DecoderLayer` 是 `batch_first=True`，注意力沿 L 维逐样本做）。
    **实测补充（2026-09-27 训练 subagent，RTX 5070 Laptop 8 GB / 55 W 限功耗）**：

    - **真实批形状**：G2/G3 批 `K=12, T=192, X=128, S=2, C=5` ⇒ **L = K·T = 2304**；
      G1 批（`optim.batch_size=1`）取到 `K=2`（该窗口线数极少，且 `g1_events=0`）。
    - **峰值显存**：`chunks=16` = **5.27 GiB**（`torch.cuda.max_memory_allocated`）；
      `chunks=4` 时落到 **7.85/8.15 GB** 的分配器边缘 —— GPU 利用率 100% 但功耗只有 **28 W**
      （分配器抖动/等待），提到 16 后 **~94 W**。⇒ `chunks=4 → 16` 的大幅提速来自**峰值内存**，
      **FLOPs 不变**（各样本的 L² 注意力本来就各算各的）。
    - **单步墙钟**（`chunks=16`，即 16 段 × 1 样本/段）：G2 两臂各 **1.39 s/优化步**
      （300 步 = 417.7 s / 417.8 s，实测）。
    - **整轮门禁耗时**（含约 3-5 min 的数据集索引）：`shuffle_steps=100` → **1190 s ≈ 20 min**；
      `shuffle_steps=300` → **3136 s ≈ 52 min**。⇒ 原「85 min」的估算是高估，实测见上。
    - **遮盖口径未变**：修完 hold 配对闭包后，G2 批的实际 `r = 0.554`（逐窗 0.50–0.75，
      `event_tokens=246` / `masked_event_tokens=133`）⇒ 闭包**没有**过度遮盖。
    - 想再压下来要么降 `data.t_window`，要么改 global 层（架构级，须开 RFC）。

23. **G2 的「遮盖集合内置换」控制口径在**真实数据**上结构性失效（本轮实测，RFC-ready，未实现修改）**：
    现象（2026-09-27 第四轮，真实 200 行切片、chunks=16、同一遮盖批、同一起点）——
    最终生效口径 = `contrast_initial_head_bias=0` + `shuffle_steps=100` + 符号稳健判据；
    权威 `gates.txt`：G1 PASS / **G2 FAIL** / G3 PASS / G4 PASS（EXIT=5，1183.8 s）。

    | 口径 | REAL 末步 | SHUF 末步 | shuf/real | 判定 |
    |---|---|---|---|---|
    | 抬高初值 bias=20（旧口径，G2/G3 也吃 G1 的初值）/ 300 步 | 734.81 | 491.68 | 0.67 | FAIL（打乱臂**更容易**） |
    | 自然初始化 bias=0 / 300 步 | −326.33 | −327.25 | **1.0028** | 旧判据**假绿**；新判据 FAIL |
    | **权威最终**（contrast bias=0，100 步，符号稳健判据） | **567.31** | **474.07** | **0.835** | **FAIL（结构性）** |

    - **G3 轨迹**（同一最终口径，contrast bias=0 / 100 步；`chunks=16`，G3 批 `K=29, T=192`
      ⇒ `L=5568`）：**81055.15 → (50) 2161.34 → (100) 343.56**，常数基线 **4461.12**，
      末步/基线 = **0.077**（权威 `gates.txt` 那一轮报 429.38/4461.12 = 0.096——两轮 dropout
      RNG 不同，均远低于 0.9 门限）⇒ **G3 是收敛中，不是结构性不达标**；而这正是因为对照臂
      改用自然初始化（`initial_head_bias=20` 时它停在 305050 vs 基线 4461）。

    - **机制**：`shuffle_hidden_counts` 在遮盖集合内置换被遮计数 ⇒ 打乱臂的标签从「少数格子上的
      峰值结构」变成「同一批计数摊在大片遮盖格子上」。对**平滑**的强度场，后者**更容易**拟合
      （真实臂要付出事件项代价去定位，打乱臂只要在遮盖区摊开）。这与 §9-16「G2 合成任务与训练
      预算」是同一类方法学问题：合成夹具的条件携带事件密度，真实数据的条件（MERT 特征）学不会
      这种定位，于是对照退化成「哪种标签分布更好拟合」。
    - **判据实现的假绿已修（收紧，非放宽）**：旧式 `need = real_last * (1 + min_gap_ratio)` 在
      `real_last < 0` 时乘出**更负**（= 更好）的阈值，于是 `shuf = 1.0028 x real` 也会 PASS。
      现口径 `need = real_last + min_gap_ratio * |real_last|`：`real > 0` 时与旧式**逐位相同**，
      `real < 0` 时要求打乱臂**更大（更差）**才通过。回归测试
      `tests/unit/infra/test_sanity.py::TestG2ShuffledTargetControl::test_negative_losses_do_not_false_green`
      与 `::test_positive_losses_keep_the_relative_formula`。
      ⚠️ 关闭假绿之后，**真实数据 G2 为结构性 FAIL**（不是预算问题）——`gates.required=true` 的
      fail-closed 因此**拦住**真实数据训练（CLAUDE.md 红线 7）。这是本轮**未完成**的一环。
    - **候选控制设计（只登记，未实现；须 RFC 裁定）**，三条都必须仍满足 plan 07 §4.3 的
      「同模型、同输入、只换标签」：
      - (a) **跨样本置换被遮内容**：把样本 i 的可见场保留，把样本 i 的被遮计数换成同批样本 j 的
        （同网格桶内，形状/测度一致）。**为什么更难**：可见场仍属于样本 i，而监督标签来自与 i
        无关的样本 ⇒ 打乱臂只能回到边缘分布；真实臂若能用 i 的上下文定位就应显著更好。
      - (b) **保边际、打断时间/线结构**：例如把被遮计数沿 τ 轴整体平移一个随机量，或按判定线
        整体置换。**为什么更难**：保留了逐 (k,x,s,c) 的计数边际，却破坏了可见上下文与隐藏布局
        的时间/线对齐关系。**风险**：τ 平移仍保留「同一模式平移后可见」的线索，可能不够狠。
      - (c) **零信息上界臂**：用「与输入无关、同边际」的随机场做监督（例如按批的边缘密度独立
        重采样）。**为什么更难**：按构造标签对输入零信息 ⇒ 任何真正利用了输入的模型都应更好；
        若真实臂打不过它，FAIL 是**有内容**的。**这不是**把 `shuffle` 换成「无遮盖批」——
        可见场仍是遮盖后的样本 i。
    - **批形状**（权威那轮）：`g1_events=0`（G1 批只取到 `K=2` 且无事件 ⇒ G1 是弱门禁，未修，
      登记待办）；G2 批 `K=12`（`L=2304`）；G3 批 `K=29`。实际遮盖 `r=0.554`（逐窗 0.50–0.75）。
    - 判据/配对**未改**；上表、G3 轨迹与 G2 轨迹（`runs/_g2_trace*.log`、`runs/_g3_trace_final.log`、
      `runs/_g2_trace.json`）是本条的证据，但都是**本地临时产物**（不入库）。
    - **未做**：真实数据训练（`data.max_samples=200` 属扩大规模，G2 未绿 ⇒ fail-closed 拦住）。
      训练通路/六件套改由 `configs/smoke.yaml optim.max_steps=2000` 的合成 run 验证（见 §9-24）。


24. **训练通路 / 六件套 / checkpoint 的 2000 步验证（合成夹具，2026-09-27 第四轮）**：因为真实数据
    被 G2 的 fail-closed 拦住，训练通路改由 `configs/smoke.yaml`（合成谱，G1–G4 合法全绿）验证一次
    **较长** run：`beatmorph-train --config-name smoke --gates --skip-env-doctor optim.max_steps=2000
    run.save_every=500` ⇒ **EXIT=0，218.8 s**。产物六件套齐全（`config.yaml` / `gates.txt` /
    `data_provenance.json` / `checkpoints/` / `logs/` / `metrics.json`），checkpoint 落
    step-500/1000/1500/2000，loss **531.97 → 176.95（best 172.33）**。
    ⚠️ **这是通路验证，不是训练结果**（合成夹具、不代表真实数据上的收敛），不得作为任何模型结论引用。

**实施期实测（2026-09-27 第五轮：G2 的真实数据结论被推翻，且门禁在真实切片上首次全绿）**

25. **`§9-23` 的「G2 在真实数据上结构性失效」是**数据缺陷的假象**——根因是 τ 轴终点**（plan 02 §9 第四轮）。
    真实 200 行切片、同一配置（`shuffle_steps=100` / `samples=16` / `chunks=16` / `contrast_initial_head_bias=0`），
    只改了「τ 轴终点」与「门禁批必须非空」两处，结果从**结构性 FAIL 变成全绿**：

    | 轮次 | G1 | G2（真实 vs 打乱） | G3（模型 vs 基线） | G4 | EXIT |
    |---|---|---|---|---|---|
    | 第四轮（旧） | PASS **但 g1_events=0（空过）** | **FAIL 567.31 vs 474.07（打乱更小）** | PASS 429.38 vs 4461.12 | PASS | 5 |
    | **第五轮（本轮）** | PASS 47455.37 → 65.13（`g1_events=20`、`g1_lines=12`） | **PASS 1672.99 vs 3875.21（打乱 **2.32×** 更差）** | PASS 107.77 vs 2680.19（0.040×） | PASS 215 帧 | **0** |

    权威 `gates.txt`：`runs/phigros_masked/20260927-043828/gates.txt`（EXIT=0，2305.8 s ≈ 38 min，
    其中数据集索引**命中落盘缓存**；`g2_events=764` / `g2_hidden_events=544` / `g2_lines=29`，`L=5568`）。

    - **机制**：旧口径下 G2 的 16 个窗**几乎全是空窗**（无事件、音频整段越界补零），
      于是「把被遮计数在遮盖集合内摊开」比「定位稀疏峰值」更容易 —— 对照测的是**哪种标签分布更好拟合**，
      而不是「输入对目标有没有信息」。这正是 `§9-16`/`plan 04 §9-15` 早就点出的方法学病症
      （**合成任务必须条件可学、打乱后不可学**），只是它在真实数据上由数据缺陷**伪造**出来了。
    - ⇒ **`§9-23` 的三条候选控制设计（a/b/c）不需要再实现**：现有口径（同一遮盖批 + 仅在遮盖集合内置换）
      在真实数据上给出 2.32× 的差距，是**有内容的证据**。
    - **教训（写进方法学）**：门禁 FAIL 时，先证明**输入批不是退化的**（事件数、音频补零率、K/L 形状），
      再谈「判据/控制设计」。`gates.txt` 现在把 `g1_events`/`g1_lines`/`g2_events`/`g2_hidden_events`/`g2_lines`/
      `g3_events`/`g3_lines`/`gate_min_batch_events` 全部落盘，就是为了让这件事**当场可判**。

26. **门禁批必须非空（`gates.batch_min_events`，默认 1）**：`BatchSource.batch(..., min_events=k)` 会重抽
    直到批内事件数达标（最多 `ManifestBatchSource.max_event_draws=64` 次），取不到即抛（fail-closed）。
    `SmokeBatchSource` 侧只做断言（合成谱本就密集）。
    为什么必须是**默认**而不是可选项：第四轮 `g1_events=0` 时 G1 报了 `7819.5 -> 0.0014` 的 PASS，
    而那个批里**没有任何事件可过拟合**——假绿。`gates.txt` 的上下文现在含 `gate_min_batch_events`。

27. **门禁装配的显存卫生（本轮定位的「GPU 滑进 Windows 共享内存」）**：四个批 + 四个模型同时在场时，
    本机 8 GB 卡实测 `nvidia-smi` **7.88 / 8.15 GB**、功耗从 94 W 掉到 **88 W**，
    同一个 G3 步从 **0.57 s 放大到 8.8 s（15×）**，整轮门禁迟迟不结束（本轮先杀掉了一次这样的运行）。
    处置：**G3 的预计算先跑、跑完立即 `del` 模型/批/优化器并 `torch.cuda.empty_cache()`**
    （G3 只需 `model_loss` / 基线 / 帧数 / 时长四个标量），之后才建 G1/G2 的模型。
    实测峰值回到 **6.0-6.6 GB**、功耗 74-94 W，整轮 **38 min** 完成。
    ⚠️ 这不是「优化」：不这么做时门禁**根本跑不完**（本机 8 GB）。

28. **真实切片上的门禁预算（本轮实测，回答 §9-14 的一部分）**：`chunks=16`（1 样本/段）下，
    G1 批 `K=12, L=2304` 单步 0.10 s；G2 批 `K=29, L=5568` 单步 ~7 s；G3 批 `K=33, L=6336` 单步 ~9 s
    （**步时 ∝ K²**，因为 global 层对 `L=K·T` 做全自注意力）。整轮 38 min（其中 G2 两臂 100 步 ≈ 23 min）。
    ⇒ 「多少样本才够」这条曲线**优先度下降**：16 样本已经给出 2.32× 的差距（门限 1.05×），
    而**更大的样本数会被 `K²` 的步时成本劝退**；真要继续压，应先处理 global 层的 `O(L²)`（架构级，须 RFC）。

**实施期落地（2026-09-27 第六轮：长跑运维 —— 断点续训 / 旋转 / 增量标量 / 巡检脚本）**

29. **断点续训（`--resume [RUN_DIR|latest]`）**：`train(resume_from=...)` 经 `load_checkpoint` 恢复
    **模型 + 优化器（AdamW 动量）+ 步号**，从 `step+1` 继续；`load_checkpoint` 的三项校验（配置指纹 /
    门禁全绿 / `data_rev`）任一不符即**拒绝恢复**。CLI 侧 `latest` 会**逐个往前找**「有 checkpoint 的
    目录」——最近的目录可能只跑过 `--gates-only`（没有 checkpoint）。
    `RunArtifacts.open_existing()` 保证续训落在**原来那个目录**（另起目录会把同一次实验劈成两半）。
    ⚠️ **随机数状态没有保存**：数据顺序与遮盖种子由 index 与 `optim.seed` 派生（同一 index 内容一致），
    但 dropout 的随机流与不中断时不同——这是「续训 ≈ 继续」而非「逐位等价」的边界，已如实登记。
30. **恢复指纹只覆盖语义字段（本轮口径，收紧/放宽都说清楚）**：指纹与 diff 先经 `_resume_relevant()`
    去掉 `RESUME_IGNORED_KEYS`（`optim.max_steps`、`run.save_every`/`log_every`/`keep_last`/`keep_best`/
    `log_level`/`runs_dir`/`experiment`）。理由：这些是**预算与落盘策略**，不是实验语义——不放行它们，
    「想多跑 5000 步」或「想加密 checkpoint」就必须另起实验目录，而那会把同一次训练劈成两个目录、
    曲线断在两处。模型 / 数据 / 优化器超参 / 门禁阈值 / seed **仍然**参与指纹（改 lr 照样拒绝）。
31. **checkpoint 旋转（`run.keep_last` / `run.keep_best`，此前只声明未实现）**：`rotate_checkpoints()`
    只保留最新 `keep_last` 个 `step-*.pt`（外加 `best.pt`），**先写后删**（新 checkpoint 落盘成功才删旧的）。
    `best.pt` = 存盘时刻训练损失最低的那一个；**val 路径尚未实现，故它不是模型选择依据**，只是崩溃恢复的
    第二个候选点。
32. **增量标量（长跑必须能在线看进度）**：此前的 `write_scalars()` 只在**训练结束后**写一次 TB ⇒
    人力监控在整轮训练期间**看不到任何曲线**。现改为每 `run.log_every`（默认 50）步：
    ① 追加 `logs/loss_history.jsonl`（权威、可脚本读、**续训后是同一条曲线**）；
    ② `writer.add_scalar` + `flush()` 到 TB（`train/loss` / `train/step_time_s` / `train/grad_norm` /
    `train/lr` / `sys/peak_vram_gib`）。TB 缺失只告警，不影响训练结论。
33. **`scripts/training_health.py`（巡检脚本，本轮新增）**：把「在跑吗 / 降速了吗 / 显存贴顶了吗 /
    checkpoint 落后了吗 / ETA 多久」压成一行结论 + 退出码（0 健康 / 1 告警 / 2 无数据），
    支持 `--watch 7200`（每 2 小时自动巡检）。预警规则：步时后半窗比前半窗慢 1.3× 以上、
    峰值显存 ≥ 7.5 GiB、标量 30 分钟没更新、checkpoint 落后于当前步。
    **它不依赖 GPU/驱动**（GB 实况是可选的 `--gpu`）——正好用于「训练在跑、agent 只读文件」的巡检。
34. **硬件安全（事故记录，写给下一个 session）**：一次「扫 K vs 显存」的 GPU 探测
    （`sdpa_kernel([...])` 强制后端 + 无界的连续前向/反传）把本机 8 GB 卡拖到驱动层 TDR，
    **Windows 被重启**。规则写进 `docs/TRAINING.md` §7.5：不强制注意力后端、不跑无界 GPU 扫批、
    同一时刻只跑一个 GPU 作业、全程盯功耗与显存。另记：硬重启会弄坏 `.mypy_cache`
    （`INTERNAL ERROR ... database disk image is malformed`）——删缓存重跑即可，不是代码问题。
35. **显存墙实测（2026-09-27 第六轮；回答 §9-28 与 plan 04 R-04-4 中可测的部分）**：
    用有守卫的单步探针（`scripts/local_vram_wall.py`：不强制注意力后端、每步前后查 `mem_get_info`、
    按 `(K·T)²` 外推超预算即停、每点只跑一个前向 + 反传 + 一步优化）在**生产配置**上量出曲线
    （B=1、d_model=256、heads=4、n_layers=6、window=32、global_period=4、`t_window=192`、
    `x_bins=128`、**fp32**——`train()` 里没有 autocast，见 §9-36）：

    | K | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
    |---|---|---|---|---|---|---|---|
    | 峰值显存 GiB | 0.81 | 1.24 | 1.84 | 2.46 | 3.30 | 4.13 | **5.05** |
    | 步时 s | 0.066 | 0.109 | 0.167 | 0.249 | 0.311 | 0.424 | **0.550** |

    拟合 `峰值 ≈ 1.05 + 0.00391·K² GiB`（K=24/28 定参、**K=32 独立校核：预测 5.05 / 实测 5.05**），
    步时 `≈ 0.00054·K² s`——与 §9-28 门禁实测的「K=29 → 约 7 s / 16 段 = 0.44 s/样本、
    K=33 → 0.56 s/样本」**独立吻合**。
    **⇒ 本机 8 GB（可用 6.85 GiB）的安全上限 K ≈ 28-31，物理上限 K ≈ 38。**
    而 train split **36.4% 的谱面 K > 31、27.9% 的 K > 38** ⇒ **`data.k_max=128` 不是保护**：
    按当前配置直接开全量训练，会在最初几个样本上就 OOM 或滑进 Windows 共享内存。

    **⛔ 文档里「global 层是 `O((K·T)²)` 的墙」这句话是错的（本轮实测）**：把 6 层全部翻成
    局部层（原地改 `DecoderLayer.kind`，**参数完全相同**）在 K=20 时峰值 **2.92 GiB**，
    全部翻成全局层只有 **1.22 GiB** ⇒ **局部层贵 2.4×**；生产排布（4 局部 + 2 全局）2.46 GiB。
    **机制已定位（第二个对照，§H）**：把 `line_tracks` 的时间轴从 192 格压到 **1 格**（只改 K/V 长度、
    纯输入侧），全局部层从 **2.92 → 1.29 GiB**、步时 1.169 → **0.064 s（18×）**；
    生产排布 2.47 → 1.26 GiB、0.365 → 0.080 s（4.6×）；而全全局层本来就是 1.22 GiB。
    ⇒ **局部层的代价几乎全部来自轨道 cross-attention，其 key 轴长度是 `K · T_line`**
    （`tracks.repeat_interleave(n_lines, dim=0)`）——也就是说**「局部层」在判定线轴上其实是全连通的**：
    每条线的 token 会 attend 到**全部 K 条线**的事件轨。这与模型 docstring 的
    「局部层是滑动窗口自注意力 / 全局层才同时看到全部 K 条线的场」不符。
    **这是本轮最贵的一处**：若局部层只用自己的那条线的轨（key 轴 = `T_line`），K=20 的峰值会从
    2.47 落到约 1.3 GiB，`K²` 项几乎消失。**是有意设计还是漏改，须在 RFC 里裁定**——
    本节只报测量，不动模型。

    **T 轴（K=24 固定）**：T=192 → 3.47 GiB / 0.300 s；128 → 2.28 / 0.184；96 → 1.77 / 0.147；
    64 → 1.27 / 0.085；48 → 1.02 / 0.066。拟合 `峰值 ≈ 0.86 + 1.23e-7·L²`（L = K·T）。
    ⇒ `t_window` 192→96 把上限从 K≈28 抬到 **K≈53**（**不是 4×**，因为有常数项），单步快 2.04×。

    **真实批校核**：从 `ManifestBatchSource` 取真实窗口（K=29、T=192、音频 750 帧），
    单步在 K=24 时 3.47 GiB，与合成探针同 K 的 3.30 GiB 一致 ⇒ 合成探针可代表生产形状。
    **刻意没做**：没有把显存压到 OOM（曲线由 7 个点 + 1 个独立校核点确定）。
36. **`optim.precision: bf16-mixed` 声明了但从未生效（本轮实测，可当场验证）**：
    全仓 grep `autocast` / `GradScaler` 在 `beatmorph/` 下**零命中**——`train()` 直接 fp32 前向，
    `schema.py` 的 `optim.precision` 只被声明、无人读。补测 bf16 autocast（同配置）：
    K=32 峰值 **5.05 → 2.92 GiB（1.73×）**、步时 **0.550 → 0.269 s（2.05×）**；
    K=40 → 3.94 GiB / 0.443 s。拟合 `峰值 ≈ 1.11 + 0.00177·K²` ⇒ 上限 **K ≈ 42-47**。
    ⚠️ bf16 的**首步**要 1.4-3.0 s（一次性内核编译 / autotune），只看首步会误判成「慢 23×」。
    **⇒ 打开一个配置里早已声明、且不需要任何架构改动的开关，就能把 K 上限抬约一半、单步砍半。**
    是否打开属 infra-agent 的配置口径修正（不动契约），但**上线前必须重跑 G1-G4 门禁**
    （bf16 改数值路径，G1-G4 是唯一能证明它没改坏东西的证据）。
37. **本线轨落地 + bf16 接线 + 真实数据实跑（2026-09-27 第七轮；RFC-0032 已采纳）**：
    ① `MaskedFieldModel.decode` 的局部层改用**本线轨**（reshape 而非 `repeat_interleave`），
    跨线仍只在全局层 —— 显存 ∝`K²` → ∝`K`（K=20: 2.46→1.31、K=32: 5.05→1.86、K=64: 3.41、
    配 bf16 时 K=128 → 5.10 GiB）；
    ② `optim.precision` **接上 autocast**（新增 `autocast_context` / `autocast_dtype`：只在 CUDA 且
    bf16 家族时启用；CPU 保持 fp32 ⇒ 默认 CI 数值不变；fp16 在 `validate_config` 里被拒）。
    **门禁的 `make_step_fn` 与 `train()` 用同一个上下文**，否则门禁与训练跑的不是同一套数值；
    ③ `train()` 的在线标量新增 **`batch_n_lines` / `batch_events`** —— K 与「空批」两个老问题都能在线看见；
    ④ **护栏**：`tests/unit/generation/test_local_own_line_tracks.py`（3 项）直接钉住 cross_tracks 的
    K/V 长度（局部 = `T_line`、全局 = `K*T_line`）——改回旧写法**不报错**，只会让显存重新变成 O(K²)。

    **实跑验证（真实数据 + CUDA，非合成）**：`data.max_samples=8 optim.max_steps=24` ⇒ 24 步内
    **4 次命中 K=100**、峰值显存 **4.00 GiB**、步时 1.3–2.2 s（K=7 → 0.16 s、K=24 → 0.33 s、K=48 → 0.60 s），
    loss 4758.6 → 353.2。**旧写法在 K=100 需要约 40 GiB**。证据
    `runs/ownline_check/20260927-082835/logs/loss_history.jsonl`。默认 CI **1009 passed**、ruff / mypy 干净。
    **遗留**：`--gates` 需在真机重跑一轮（随启动顺带，不额外占 GPU）；装饰线旁路按决策者口径
    登记为**后续扩展**（plan 04 §9-24）。

38. **采样器覆盖率缺陷与 epoch 定义（2026-09-27 第八轮；RFC-0033 已采纳）**：
    起因是排查「GPU 长期空转（利用率 **21.4%**、功耗中位 7 W）」。剖面定位到 `line_tracks_at`
    占每窗口 CPU **52.5%**（见下一条），但**顺带暴露了一个严重得多的缺陷**：
    `ManifestBatchSource._draw` 只有**一个全局共享游标**，且每步都被写成「按**当前桶**长度取模」
    `(start + take) % len(bucket)`；桶长极不均（min **1**、中位 66、max 35,751），
    **撞到长度 1 的桶就把游标清零** ⇒ 它永远在「每个桶的前几个窗口」之间震荡。
    用**仓库里那份 `_draw` 本身**（只把样本构建换成 stub）实测：

    | 步数 | 抽到的不同窗口 | 不同谱面 |
    |---|---|---|
    | 1,000 | **993** | **674** |
    | 3,000 | **993（不变）** | **674（不变）** |
    | 20,000 | **993（不变）** | **674（不变）** |

    ⇒ **`max_steps` 加到多少都不见新数据**；「全量大规模训练」实际只在 **0.156% 的窗口 /
    10.2% 的谱面**上训练，而 loss 曲线完全正常——与 [POSTMORTEM-2026-08-05] 的 25 Hz 事故同类。

    **修法**：① 每桶**独立的**抽签顺序（桶内按 (seed, epoch) 派生种子洗牌）+ 每桶独立游标；
    ② **按剩余窗口数加权选桶**（简单轮转让小桶被抽干、大桶只走 0.06%；而桶按 `bpm_eff` 分，
    BPM 常见的大桶恰好装着最多谱面——轮转 20,000 步只覆盖 47.2% 谱面，加权后 **91.3%**，
    与「按窗口均匀」的理论值 91.1% 一致）；③ `want`（批大小）优先，避免把 16 样本的门禁批切成 1。
    **epoch 定义**：一个 epoch = 全库窗口的一个**排列**（train split = **634,952** 步）；
    新增覆盖率在线标量 `epoch` / `coverage/windows_seen` / `coverage/charts_seen`（jsonl + TB）。
    「chart_epoch（≈6,614 步）」被否掉：每谱窗口数分布极不均（均值 96 / min 1 / 中位 90 / **max 922**），
    「每谱一个窗口」不是一致的量，直接报**谱面覆盖率**才诚实。

    **预算含义（修复后）**：6,614 步 → 谱面覆盖 59.9%；**20,000 步（当前配置）→ 91.3%**；
    50,000 步 → 99.0%；全量 epoch 634,952 步 ≈ **88 h @0.5 s/步**（8 GB 笔记本上不可行）。
    ⇒ 「epoch」在本机只能当**可观测指标**，不能当日常预算单位。

    **护栏**：`tests/unit/infra/test_sampler_coverage.py`（6 项）钉死「epoch 内每个窗口恰好一次 /
    覆盖率只增不减 / 同 seed 可复现 / 选桶按剩余加权 / 同批同网格」。**旧写法不报任何错**——
    没有这条护栏，这个缺陷能一直活到训练结束。
    证据脚本：`scripts/local_draw_verify.py`、`local_coverage_check.py`、`local_chart_lru_sim.py`、
    `local_windows_per_chart.py`。

    **顺带否掉两个猜想**：① 调大 `data.chart_cache_size` 不划算（同一取批顺序下 LRU 命中率
    cap=8 → 18.6%、cap=32 → 26.1%、cap=64 → 28.7%，而同谱复用间隔中位 **991** 步）；
    ② 「空事件窗口」不是瓶颈（它只是少了监督信号，且门禁另有 `batch_min_events` 兜底）。

39. **事件轨求值的向量化：每窗口 CPU 1.218 → 0.582 s（逐位等价，32.9x）**：
    剖面（`scripts/local_profile_dataset.py`，cProfile，24 个真实窗口）显示每窗口 1.206 s 里
    **52.5%** 花在 `beatmorph/data/tracks.py::line_tracks_at`：它按 `tau` 逐点调
    `JudgeLine.sum_track`，后者又线性扫过整条轨的关键帧并在扫描中反复调 `Beat.to_beats()`
    —— 实测 **1.46e6 次 `track_value` / 3.97e7 次 `to_beats`**。

    修法：① 每条 (轨, 层) 只算一次 `starts` / `ends`（`to_beats` 调用降到 1.3e6）；
    ② 用 `searchsorted(starts, "right") - 1` 与 `searchsorted(prefix_max(ends), "left")`
    向量化定位 **列表序首个命中的关键帧**（证明写进 `_first_matching_keyframe` 的 docstring，
    **不假设关键帧不重叠**：重叠时契约取「先出现的那个」，该式同样成立）；
    ③ 取值**仍然调用契约层的 `EventKeyframe.numeric_at`**（快路径只覆盖「Linear + 默认切割 +
    两端 float」并逐字复刻 `value_at` 的分支顺序），因此**数值逐位不变**。

    实测：该段 **32.9x**（21.97 s → 0.67 s：12 张真实谱面 × ~660 个 tau，含端点与边界扰动），
    每窗口 **1.218 → 0.582 s**，函数调用数 7.97e7 → 2.22e7；
    等值判据是 `torch.equal`（**不是** `allclose`），护栏
    `tests/unit/data/test_tracks_vectorized.py`（4 项：空隙 / 重叠 / 贝塞尔 / 非默认切割 /
    多层 / 端点 / 空轨 / 通道序）。**剩下的大头变成谱面解析（46%）**——那是下一轮的结构性优化
    （每窗口重解析同一张谱；见 §9-38 的 chart LRU 结论）。

40. **本轮默认 CI 口径**：`ruff` / `mypy --strict` 干净；`pytest -m "not slow and not gpu and not e2e"`
    **1005 passed**（17 deselected）。新增护栏 10 项（采样器 6 + 事件轨 4）。

41. **门禁成本的真相与「慢 vs 卡死」的可观测性（2026-09-27 第八轮）**：
    采样器修好后门禁从 **5 min 涨到 ~40 min**，但**这不是回归**：旧 sampler 的坏游标让门禁永远
    只取「最小 `bpm_eff` 桶的前几个窗口」，实测批是 **K=29/11/6**；修好后取的是**代表性**桶
    （本轮实测：200 行切片最大桶 K=**39**、中位桶 K=**40**；全量切片最大桶也是 K=39）。
    门禁单步成本 ∝ `(K·T)²`，且 `shuffle_chunks=16` 使 G2/G3 的**一步 = 16 次串行单样本前向**
    ⇒ G3 从 K=6 → 40 就是 **~44x**。**第五轮记录 2305.8 s ≈ 38 min 才是正常 K 下的预算**，
    r1 的 5 min 是「抽到退化批」的运气。

    - `gates.gate_samples=200` 限住**数据池**，但**限不住 K**（两个切片的最大桶 K 都是 39）
      ⇒ 不要指望它把门禁变快。
    - **教训（本轮的真实代价）**：我据此把两个**正在正常运行**的门禁误判成卡死并掐掉
      （r2 26 min、r3 16 min），浪费约 **40 min GPU**。根因是 `execute_gates`
      **整轮跑完才写第一行日志** ⇒ 外部只能靠功耗 / util 猜。
      已修（commit `d234f55`）：`run_gates(..., progress=)` 逐门禁上报「开始 / PASS|FAIL（耗时）」，
      `build_gate_inputs` 打印三个批的 K / samples / events 与 **G3 预计算的起止**
      ——G3 的 100 步在 `build_gate_inputs` 里跑，正是门禁里最大的一块静默时间。
      护栏：`tests/unit/infra/test_gate_progress.py`（2 项）。
    - **判据修正**：门禁期间 `memory.used ≈ 7.7 GiB / 8.15 GiB` 在当前 batch 口径下是**常态**，
      **不是**共享内存回退的特征（util 99-100%、功耗 47-92 W、连续 26 min 无 OOM）。
      **先看逐门禁日志，再看功耗**；不要仅凭显存贴顶就判卡死。

[POSTMORTEM-2026-08-05]: ../POSTMORTEM-2026-08-05-frame-rate-misalignment.md
