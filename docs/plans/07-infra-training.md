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
    多层 / 端点 / 空轨 / 通道序）。**剩下的大头变成谱面解析（46%）**——它的量化、瓶颈归属
    与修法已立案：见 §9-42 与 [RFC-0034](../decisions/RFC-0034-data-supply-throughput.md)
    （每窗口重解析同一张谱；§9-38 的 chart LRU 结论说明「调大缓存」不是解法）。

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

42. **GPU 中位利用率 5% 的真相：训练路径里没有 worker（2026-09-27 第九轮）**：
    决策者问「GPU 平均占用率依然不算高，是不是 workers 太小了」。实测（**训练进行中**，
    step 1850/20000，RTX 5070 Laptop 8 GB / 功耗上限 115 W）：GPU 利用率（1 Hz × 60 采样）
    **均值 35.7% / 中位 5.0% / p90 100% / 其中 29 个采样 ≤1%** —— **双峰**：要么满载、要么空转，
    中位 5% 说明一半以上时间 GPU 什么都没干；功耗 **均值 21.4 W = 上限的 18.6%**（峰值 78.3 W）；
    训练进程占 **6.45 / 24** 逻辑核（峰值 14.89），系统 CPU 均值 30.1%；显存 7862/8151 MiB。
    步时（1850 步）：中位 **0.539 s** / 均值 0.871 / p90 1.721 / p99 4.52 / max 6.83，
    **25.8%（478/1850）的步 ≥1 s**；K 中位 26 / p90 67 / max 128 ⇒ 长尾来自 ∝(K·T)² 的注意力。

    - **「workers」这个旋钮当前不存在**：全仓 `num_workers` 只出现在**数据准备**脚本
      （`scripts/fetch_phira.py` / `scripts/extract_features.py`）；训练侧唯一的 torch
      `DataLoader` 在 `beatmorph/infra/lightning_module.py`，且 `batch_size=None`、
      `num_workers` 取默认 **0**。真实训练走 `ManifestBatchSource.batch()`，它在**训练进程
      主线程里同步**完成「选桶 → 取窗口 → 解析谱面 → 建场 → collate」，而 `step_time_s`
      的计时起点就在这一句**之前** ⇒ 数据构建时间**本来就混在步时里**。
    - 根因是**每窗口重解析整张谱面**（一张谱平均 **96** 个窗口）：§9-38 修掉事件轨后它就是
      每窗口 CPU 的第一大项（**≈46%**）。把 `chart_cache_size` 调大**已实测无效**
      （同谱复用间隔中位 **991 步**，LRU 命中率封顶 18.6%）；而直接开 `num_workers`
      会让**每进程各持一份 LRU 缓存**，并把**覆盖率记账**变成各副本 —— 正是 RFC-0033
      刚修掉的那类静默失效（worker 只并行 `__getitem__`，**不并行「选桶」这个串行决策点**）。
    - **已落地（本轮）**：`step_time_s` 拆成 `data_time_s` + `compute_time_s`
      （逐行和**恒等于**步时），TB 增 `train/data_time_s` / `train/compute_time_s` /
      **`perf/data_share`**；`scripts/training_health.py` 打印拆分并在
      **数据占比 ≥50% 时告警**（老曲线无这两个字段则不误报）。同时修掉巡检脚本的一个
      **恒真告警**：旧口径 `last_step > checkpoints[-1]` 在 `save_every=2000` 时
      1999/2000 的步都命中（实测 step 2250 / ckpt 2000 被报「落后」）⇒ 退出码恒为 1、
      把 go/no-go 信号作废；现改为**只在真漏存（间隙 > save_every）时告警**。护栏
      `tests/unit/infra/test_step_time_split.py`（6 项）。**动机**：§9-41 的误判之所以发生，
      正是因为一个数把「GPU 在等数据」与「计算本身慢」混在了一起，只能靠功耗反推。
    - **架构决策待裁定**：[RFC-0034](../decisions/RFC-0034-data-supply-throughput.md)
      **v2：计划 / 物化 / 搬运三层分离**——顺序与覆盖率搬到主进程的**纯函数计划层**，
      样本构造交给 DataLoader worker（纯函数），主进程只做 H2D 与计算 ⇒ GPU 与下一批构造重叠。
      「谱面解析一次」不再是独立方案：桶内按谱聚簇后它是**取批顺序的副产物**。
      该 RFC 同时论证了 **0 次门禁重跑、0 次索引重建**（顺序是 (seed, epoch) 的纯函数；
      `data.workers` 语义无关 ⇒ 进 `RESUME_IGNORED_KEYS`）。

43. **遮盖退化（r==0）的在线可见性（2026-09-27 第九轮，本轮顺带发现，未修）**：
    本轮训练日志 47 行里有 **30 行**是
    `chart_id=… 窗口 N：8 次重掷种子后仍无法让事件可见（r<1），退化为无遮盖样本（r=0；losses 的 r==0 契约分支）`，
    即 ≈**1.6%**（30/1850 步）的窗口落到 `r == 0`。按 losses 契约 `r == 0` ⇒
    `masked_poisson_loss == full_poisson_loss`，**不是空批、也不是零梯度**（已核对：1900 步里
    `loss == 0.0` 的行数为 **0**）。但这些步**没有在练掩码补全目标**，而是在练全事件目标
    ⇒ 目标在 ~1.6% 的步上**漂移**，而 loss 曲线看不出来（两者量纲接近）。
    登记为**未决项**：待 val 路径接通后，按「掩码目标实际占比」评估是否要改遮盖策略
    （例如换遮盖单位 / 放宽重掷上限 / 把退化窗口在采样时降权），而不是继续静默兜底。

    ⚠️ **第十轮复核（口径纠正，实测）**：上面那个 **1.6%** 是从一段 1850 步的日志摘录里数的。
    把两份完整训练日志数一遍：`r6.log` **42 行 / 4000 步 = 1.05%**、`r5.log` 123 行 / ≈12,200 步
    ≈ 1.0% ⇒ **真实值是「约 1.0% 的**全部**步」**；该段空窗占 42.1% ⇒ 换成非空批口径是 **≈1.8%**。
    更重要的是一条**之前完全没写进记录**的事实：`dataset.py` 明文规定 **「窗口没有事件时 `r == 0`」
    是合法且不告警的**（`unit_count == 0` 分支）⇒ 本轮的语料里 **43.8% 的步天然就是 `r == 0`**，
    它们同样走 `masked_poisson_loss == full_poisson_loss` 分支。**所以「`r == 0` 的步有多少」
    这个问法本身要带口径**（全部 / 非空），否则数字相差 40 倍以上。
    `r` 未落盘 ⇒ 目前**只能数日志告警**，不能在 jsonl 上校验（见 §9-46 的仪表建议）。

44. **取批计划层 + DataLoader 并行（RFC-0034 S1–S4，2026-09-27 第九轮，已落地）**：
    提交 `cc0e166`。**顺序与覆盖率搬出采样器**（新增 `beatmorph/data/plan.py`）：一个 epoch 的
    取批顺序一次性物化为纯数据，覆盖率变成「顺序前缀」的 `searchsorted` 查询（O(log n)）
    ⇒ 采样器不再有状态，才能开 `data.workers` 而不破坏可复现性与记账。

    - **S3**：`train()` 改为流式 `next(stream)`；`data.workers>0` 走 DataLoader
      （`persistent_workers` + `prefetch_factor=2`，`pin_memory` 仅在 CUDA 可用时）。
      **批次切分仍在主进程**，槽位终点由 `batch_sampler` 上报 —— 记账**不依赖 collate 产物**。
      `data.workers` 进 `RESUME_IGNORED_KEYS`（语义无关字段）；门禁路径**强制 workers=0**。
    - **S4 的两版**（这条值得单独记）：第一版按**窗口数**分槽位，真实 train split 实测
      20,000 步只覆盖 **52.6%** 的谱面（旧均匀撒点 91.3%）——「窗口多、谱面少」的桶反复访问
      同一张谱，同时饿死窗口少的桶。改为**全局按轮次发牌**后：**6,600 步 71.8% /
      13,200 步 100.0%**，**严格优于**修复前。代价：`chunk=1` 下同谱连续段中位长度为 1，
      **没有**摊薄解析（那 46% 仍在），S4 这一轮拿到的是覆盖率不是解析摊薄。
    - 计划物化实测：索引缓存命中 1.3 s + 计划 1.8 s（991 桶 / 634,952 窗），一次性成本。
    - 护栏：`tests/unit/data/test_plan.py`（9）+ `tests/unit/infra/test_plan_batches.py`（6，
      含 **workers=0/2 逐位一致**的真实 spawn 等价性）+ 集成 pickle 往返。默认 CI **1033 passed**。
    - **未做**：S5 巡检；worker 侧 `r == 0` 计数的旁路汇总（docstring 已写明是已知缺口）；
      验收判据 3/5/6。

    **验收已实测（2026-09-27 19:55，`--resume latest` 从 step 4000 起，`OMP_NUM_THREADS=2` +
    `data.workers=3`）**：索引缓存命中、**未跑门禁**、指纹校验通过（RFC §5 三条论证在真实路径上成立）。
    步时中位 **0.539 → 0.099 s（5.4×）**；峰值显存 4.87 GiB（无回退，但注意
    `max_memory_allocated` 是**进程内**高水位，不是受控对比）；K 中位 25 / p90 58（旧 26 / 66）
    ⇒ 取批顺序变更**没有**偏斜 K。

    ⚠️ **同日同段还有一句话是错的，第十轮实测改正**：当时报「`perf/data_share` 中位 **5.0%**
    ⇒ 数据侧已不是瓶颈，剩余空转在主进程串行段（collate + 同步 H2D），下一步是 H2D 重叠、
    **不要再加 worker**」。**方向完全反了。** 该分布**极端重尾**：中位 5.0% 与**均值 45.0%** 同时
    为真，而决定吞吐的是均值；瓶颈恰恰就是数据供给，解药恰恰就是加 worker。完整量法与改正见 §9-45。

45. **GPU 利用率到底是多少 / 瓶颈在哪（2026-09-27 第十轮，已修）**：上一轮把 `perf/data_share` 的
    **中位**当成「数据侧不是瓶颈」的证据，并据此把下一步指向「H2D 与计算重叠、不要再加 worker」。
    **该读法被三种互相独立的量具推翻。**

    - **量具一（看分布，不看中位）**：新管线 10,900 步里 `data_time_s` **均值 0.125 s = 墙钟 45.0%**
      （中位仅 0.0043 s）；**前 10% 的步吃掉 77.4% 的数据时间，前 1% 吃掉 23.6%**。
    - **量具二（nvidia-smi 200 ms × 750 样本）**：`utilization.gpu` **均值 44.3% / 中位 47.0%**，
      **强烈双峰**：**42.7% 的采样 ≤5%（完全空转）**、23.3% ≥90%（满载）；功耗均值 44.4 W。
      ⚠️ 采样必须用 Python `Popen` 逐行读 `--loop-ms`：**PowerShell 管道会缓冲原生命令输出**，
      上一版脚本因此一条样本都没落盘（量具坏了，读数再漂亮也没用）。
    - **量具三（活进程快照）**：`py-spy dump` 主进程主线程 70 次 ⇒ **45.7% 停在
      `wait (threading.py:331)`**，回溯到 `_try_get_data`（DataLoader worker 队列）
      ⇒ **GPU 在饿死，不是在算**。⚠️ **`py-spy record` 在 Windows 上静默丢样本**
      （80 s@100 Hz 只拿到 4501 个），其叶帧统计把数据等待报成 **0.2%**，与 dump 的 45.7% 直接冲突
      —— 长跑诊断只能用 dump。
    - **逐项排除（都不是瓶颈）**：盘（原始读 **2006 MiB/s**）；解压（npz 压缩比仅 **1.09×**、
      222 MiB/s ⇒ 每窗口 106 ms）；`pin_memory`（开/关 6.33 vs 6.34 窗口/s）；主进程 CPU
      （**0.05 核**）；BLAS 线程（OMP=1 与 =2 给 6.31 vs 6.33）；Defender 实时防护（本就关闭）。
    - **标定曲线**（`scripts/local_worker_scaling.py`；真实 train split、只改 `data.workers`、
      逐臂断言索引缓存命中）：供给能力 **2.51 / 4.42 / 6.33 / 5.75 / 6.28 窗口/s @ N=3/6/8/10/12**
      ⇒ **N≥8 起不再上涨**，而训练需求约 11.5 窗口/s。单进程每窗口 **0.85 s**。
    - **根因是结构性的**：worker 剖面 = 谱面解析 ≈**44%**（JSON + pydantic）+ 建场 ≈**33%** +
      特征解压 ≈**12%**。计划层是**跨 991 个桶的轮转发牌**，一张谱的 ~96 个窗口摊在相隔 ~991 个
      槽位上 ⇒ `chart_cache_size=8` / `feature_cache_size=2` 的**命中率必然为 0**，同一张谱每个
      epoch 被完整重解析 96 次。**这就是 RFC-0033/0034 拿到覆盖率所付的代价**，调参数绕不开。
    - **已修（N=3 → 8，纯配置 + `OMP_NUM_THREADS=1`）**，同分布段间实测（`--resume latest` 从
      step 16000 起）：

      | 口径 | workers=3（14000–16000） | **workers=8（16201+）** |
      |---|---|---|
      | GPU 利用率 均值 / 中位 | 44.3% / 47.0% | **75.5% / 80.0%** |
      | GPU 完全空转（≤5%）占比 | 42.7% | **9.5%** |
      | `perf/data_share`（按和） | **46.4%** | **22.4%** |
      | 步时 mean / p90 | 0.296 / 0.732 s | **0.195 / 0.347 s** |
      | data 等待 p90 | 0.451 s | **0.0118 s（38×）** |

      ✅ 判据 3b（GPU 利用率 >50%）**首次达标**；**跑完全段（steps 16201–20000，n=4050）后 `perf/data_share` = 19.0% ⇒ 判据 3a（<20%）也达标**（上面表里的 22.4% 是 16201–17800 的部分段；全段步时 mean 0.1994 s、K 中位同为 26 ⇒ 不是 K 分布变简单）。
      续训照旧：索引缓存命中、**未跑门禁**、指纹通过。
    - **下一步（按性价比排序）**：① `data_share` 还剩 22.4%，尾部仍在（p99 1.68 s；一个 worker
      卡一次能拖穿 16 个预取槽）⇒ 加深预取（`prefetch_factor`，代价是主机内存）或**削掉整谱重复**；
      ② `commit charge` 已达 **38.1/40.1 GiB（95%）** ⇒ N=8 同时是**内存上限**，再加 worker 会踩页文件；
      ③ 削整谱重复（谱面级紧凑缓存 / 在顺序里加局部性）**会动取批顺序语义** ⇒ **必须先开 RFC**，
      不得在配置里悄悄改；

46. **训练期仪表全面体检：现有 2 万步参数答不了「模型在学吗」（2026-09-27 第十轮调研，未修）**：
    由 subagent 做的只读调研（报告与复现脚本在**未入库**的 @@runs/_research/q2_*@@；本条是它的结论落盘）。

    - **「模型在学习吗」= 用现有材料答不了。** 唯一可比的量是「非空批 loss vs **同批**最优常数场基线」
      （@@L = N·(1 + log(|Ω|/N))@@，@@|Ω| = K·3_240_000/bpm_eff@@；用项目自己的 @@field.integrate.omega@@
      校验到 **1e-16**）。在**已验证可还原**的 step 4001–20000（16,000 步，靠离线重放 @@WindowPlan@@
      补 @@bpm_eff@@、再用日志 @@charts_seen@@ 逐行校验一致率 **100%**）里，这个比值**完全平坦**：
      中位 **0.834**（模型 90.48 / 基线 107.99），四段改善 16.0 → 20.7 → 17.2 → **13.1%**（末段最差），
      全段 @@Spearman(step, 比值) = **−0.120**。⇒ 只能说「**明显优于平凡常数场约 16%，但该优势在
      16,000 步里没有扩大**」；「在学 / 不在学」都推不出来。
    - **loss 是双峰，不是一条曲线**：**43.8% 空窗（中位 6.5e-4）/ 56.2% 非空（中位 96）**，差 5 个数量级；
      @@loss < 1e-3@@ 的行**全部**是空窗（非空批最小值 **4.88**，无一行恰为 0）。空窗的 loss **恒等于积分项** @@∫λdV@@。
      ⇒ @@metrics.json@@ 里的「0.002140 → 17.453615」是**双峰上各抽一个样本**，不携带趋势信息；
      **@@best.pt@@ 按训练损失选 = 按「谁抽到最空的窗」选**（实测最小 loss 在 step 6234，events=0、K=1）。
    - **@@grad_norm@@ 不是独立量具**：与 loss 的 log-log Pearson **0.996**，只是 loss 的影子；
      且 @@optim.grad_clip_norm=1.0@@ ⇒ **每一步都被裁剪**（10–1000×），AdamW 对全局缩放近似不变
      ⇒ 该裁剪近乎安慰剂。@@raw_grad_norm@@ 未落盘 ⇒ 裁剪比看不见。
    - **架构**：G1 证**通路**、G3 证**容量**（16 样本跑 100 步 ⇒ 是训练损失不是泛化）、G2 证「可见上下文含
      目标信息」——但 **G2 用同线未遮盖的事件轨就能过，不必用音频**。TB 的全部 tag 里
      **没有任何「音频条件是否被用上」的证据**（所有比较都是「条件在场 vs 条件在场」）。
    - **数据卫生（分析 jsonl 前必须知道）**：21,100 行 / 20,000 step，**1100 个 step 有两条副本**——
      4001–4650（两条**数据不同**：旧管线无 @@data_time_s@@）与 **16001–16450**（两条 @@lr/K/events/
      windows_seen/charts_seen/epoch@@ **逐位相同**、非空批 loss 中位相对差 6.9% ⇒ **同数据、权重已分叉的
      两条轨迹**，不能当独立样本）。正确口径 = 按 step 去重 + **整段丢弃 1–4000**（旧采样器不可还原，
      @@charts_seen@@ 一致率仅 0.9%）。**注意 1–4000 段不可与后段合并比较。**
    - **待接线的仪表（都有现成实现，只是没接）**：① @@output.diagnostics@@（@@occlusion_ratio@@ /
      @@lam_max@@，@@model.py:525@@ 已算好）+ @@nll_decomposition@@（事件/积分项）+ @@batch/bpm_eff@@ /
      @@batch/omega@@ ⇒ **loss 与常数基线只用在线标量就能算**（现在缺 @@bpm_eff@@ 这一个逐窗标量）；
      ② @@raw_grad_norm/clip@@ 裁剪比；③ **音频条件干预三元组** @@cond/audio_zero_delta@@ /
      @@cond/audio_perm_delta@@ / @@cond/track_zero_delta@@（同批同权重 @@no_grad@@ 前向）——**唯一**能回答
      「音频有没有被用上」的手段；④ 接通 @@val_every@@（见下）；⑤ @@eval/paired_delta@@：每 2000 步在
      **同一批固定窗口**上重算 ⇒ 否则「曲线看不出学习」永远无法与「数据换了」区分开。
    - **存疑**：@@1/r@@ 重标定给单步 loss 带来 ±(p10–p90 0.52–1.34) 的噪声，且 @@train_loop.py:357-360@@
      **明文禁止**把 masked loss 与全事件常数基线配对（本次靠 HT 期望等式近似跨过，只能作**方向性**证据）——
      **这条直接约束 val 的设计**（见 §9-47）；「可达到的最优 loss」未知 ⇒ 0.834 是「学得少」还是
      「本来就难」**判不了**；未打开任何 @@.pt@@、未做显著性检验。

47. **val 要不要加、怎么加（2026-09-27 第十轮分析，待裁定）**：结论是**要加**，而且是当前最高价值的一项——
    不是因为「有个 val 更规范」，是因为**现有的模型选择判据是坏的**：§9-46 实测 loss 是双峰，
    @@best.pt@@ 实际选中的是 step 6234 那个 @@events=0, K=1@@ 的空窗。**先修判据，再谈调参。**

    **A. 固定验证集（最关键的一条，其余都是次要的）**
    @@split_val@@（814 张，按曲目切分，是真正的留出集）→ @@plan_epoch(val_dataset, seed, epoch=0)@@ 的
    **前 N 个槽位**，每 @@val_every@@ 步**重放同一批**：① 计划层是纯函数（RFC-0034 §5）⇒ 续训后逐位一致；
    ② 遮盖种子 @@_window_seed(seed, row_index, window_index)@@ 也是确定性的 ⇒ **连遮盖都一样**；
    ③ @@model.eval()@@（@@dropout=0.1@@ 必须关）+ @@torch.no_grad()@@。
    ⇒ 得到**确定性、跨步可比**的指标，彻底摆脱「这一步恰好抽到空窗」的污染。**没有这一条，其余设计都会退回成噪声。**

    **B. 对照臂必须换到同一个测度上（这一条是本轮被调研纠正的，我原来的写法是错的）**
    我最初提的 @@val/nll_gap = val/nll − val/nll_constant@@ **不成立**：@@train_loop.py:357-359@@ 明文写着
    G3 的基线是**全事件**口径的闭式解，而遮盖路径的 loss 是「只监督被遮盖事件 + 重标定 @@1/r@@」，
    **两者不在同一测度上**（那里还专门写了「同 §4.3 的配对纪律，别再犯 G2 那个错」）。
    ⇒ 正确做法：**在与 val 相同的遮盖测度上重算常数基线**（常数场 λ=c 在同一 @@masked_poisson_loss@@ 下
    的最优值），再比。**主判据** = @@val/ratio = val/nll_masked / val/nll_masked_constant@@。
    全事件口径的数字可以**附带**报，但要显式标注「不可与常规常数基线直接比」，而且无遮盖批上
    @@observed_counts()@@ 会把 counts 原样喂回去（**输入就是目标**，@@train_loop.py:350-352@@）⇒ 那个数只能当参考。

    **C. 归一化与分解（否则只是把一个不可比的数换成另一个）**
    @@val/nll@@ 用 @@reduction="mean"@@（按有效线数，不然 K 一变数字就跳）；再拆 @@val/nll_empty@@ /
    @@val/nll_nonempty@@（前者只反映 ∫λ 压得多低，**后者才反映事件放得对不对**）；报 @@val/integral@@（Σ∫λdV）与
    @@val/events@@（Σn）⇒ **校准比 @@val/pred_over_true@@**；报 @@val/r0_share@@ 与 @@val/empty_share@@。

    **D. 对照臂清单**（同一批、同一权重、@@no_grad@@）
    | 指标 | 对照什么 | 现成实现 |
    |---|---|---|
    | @@val/nll_shuffled@@ | 只置换被遮盖格内的标签（可见场逐位不变） | G2 的 @@shuffle_hidden_counts@@ |
    | @@val/nll_masked_constant@@ | 常数场在**同一遮盖测度**下的最优值 | 需新写（§B） |
    | @@cond/audio_zero_delta@@ / @@audio_perm_delta@@ / @@track_zero_delta@@ | **音频条件是否被用上** | 需新写（§9-46 ③） |
    | **@@val/ratio@@** | **「学到东西」的头号判据** | |
    单看 @@val/nll@@ 的绝对值**无法区分「学会了」与「把 λ 整体压低」**；只有同一测度上的相对判据才是。

    **E. 成本与参数（给算式，不拍脑袋）**
    单窗口 ≈ 数据 0.038 s + 纯前向 ~0.05 s ≈ **0.09 s**。注意 **val 可以用 worker 路径**——RFC-0034 §5 已证明
    worker 数与样本序列无关且有 workers=0/2 逐位一致的测试，不必像门禁那样强制 workers=0；否则单窗口数据
    成本从 0.038 s 涨到 **0.85 s（贵 20 倍）**。约束 @@val_windows × 0.09 ≤ 5% × val_every × 0.199@@ ⇒
    **@@val_every=1000@@ + @@val_windows=128@@ ≈ 11.5 s / 199 s ≈ 5.8% 开销**，2 万步拿到 20 个点。

    **F. 落点**：@@optim.val_windows@@ 进 schema（现在只有 @@val_every@@ 且**空转**）→ @@train()@@ 在
    @@_flush_scalars@@ 后调 @@_evaluate@@，指标进 TB 与 @@loss_history.jsonl@@ 的 @@val_*@@ 字段 →
    **@@run.keep_best@@ 改按 @@val/ratio@@ 选**（不再按训练 loss）→ 与离线 @@beatmorph-eval@@ **共用指标函数**。

    **G. 坑**：① val split 的索引是**另一份缓存**，首次要解析 814 张谱（按 train 的 36.5 min / 6750 张外推
    **≈4-5 min**，一次性）——必须提前知道，否则看起来像卡死；② @@initial_head_bias@@ **只在门禁路径生效**
    （训练路径 bias=0）⇒ **val 的数字与 @@gates.txt@@ 不可直接比**；③ val 沿用训练的 @@bf16-mixed@@ 口径
    （可比；bf16 舍入是确定性的，不影响跨步比较）；④ **val 不得变成训练信号**（先不按 val 调 LR、不早停）；
    ⑤ 固定集里空窗占比是**常数**，影响指标**水平**不影响**趋势**，报告时要连 @@empty_share@@ 一起报；
    ⑥ **存疑**：@@val_windows=128@@ 是否足够代表 814 张谱——这是**固定的系统偏差而非噪声**，建议先跑一次
    把「128 窗」与「全量 814 张」比一次来标定。**这一条我还没测。**

    **H. val 回答不了什么**：回答不了「架构对不对」（那是 plan 04 消融臂与 plan 06 的 F1——**NLL 好 ≠ 谱面可玩**）；
    回答不了「难度条件用上了吗」（需要一个难度置换对照，可顺手塞进 D 的清单）；回答不了无条件生成的质量。

    **I. 顺带一条几乎免费的**（不进 val、进训练）：@@train/integral_per_event = Σ∫λdV / Σn@@ ——loss 里本来就
    算过 ∫λ，**零额外前向**，能在 val 接线之前直接看出「λ 是不是被整体压低来刷 loss」。

48. **取批顺序改「谱面序」的实测代价与收益（2026-09-27 第十轮，决策者提案，待裁定）**：
    决策者提案：**优先从一个桶内取批，按谱面的顺序填进队列**（从「同时训数十张谱」改成「按谱面顺序」），
    理由是需要跑多个完整 epoch、速度优先。**实测结论：对 worker 有效（3.74×），但总量被另一个
    共享上限卡住（1.27×）⇒ 单靠改顺序拿不到想要的速度。**

    **实现面很小**：@@plan_epoch@@ 的产物只差一个排序键——现状是 @@sorted(blocks)@@（按 @@(轮次, 桶偏移)@@
    ⇒ 跨 991 桶轮转），提案是按 @@(桶, 谱面)@@ 排。因为同一 @@(桶,谱面)@@ 的窗口在计划里的出现次序
    **就是**窗口顺序，所以**对现有 WindowPlan 做一次稳定排序即可**，不需要 window_index。工作集 = 1 张谱。

    **收益（纯 CPU 实测，脚本 @@scripts/local_order_worker.py@@ / @@local_order_throughput.py@@）**：

    | 口径 | 现状(轮转) | 提案(谱面序) | 比 |
    |---|---|---|---|
    | **单个 worker 走整 epoch 计划的真实跨步序列** | **1.81 窗口/s** | **6.77 窗口/s** | **3.74×** |
    | 同上，p99 延迟 | 2165 ms | **1217 ms** | 减半 |
    | **DataLoader 聚合吞吐（W=8）** | **5.72 窗口/s** | **7.25 窗口/s** | **仅 1.27×** |
    | DataLoader 聚合（W=2 / W=4） | 2.03 / 3.50 | 2.81 / 4.77 | 1.38× / 1.36× |

    ⚠️ **关键推论：存在一个约 7 窗口/s 的共享上限，它不是 worker 数、不是顺序、不是 @@pin_memory@@、
    也不是主进程 CPU。** 证据：单 worker 的物化能力被改顺序抬到 6.77/s（8 worker 理论上百），
    而聚合只到 7.25/s ⇒ **worker 在等交接，不是在算**。这与前面「N=8/10/12 都给 6.3/s」是同一件事。
    ~~**首要嫌疑：@@FieldBatch@@ 经 multiprocessing 共享内存的搬运成本**~~ ⇒ **已被 §9-49 的实验 A 否掉**
    （真实批次的 `ForkingPickler.dumps` 只有 **1.1 KB / 3 ms**，句柄共享零拷贝）。
    **真正的原因是 DataLoader 的 `in_order=True` 默认值**（§9-49）；而**「改顺序」也确实不是解药**
    （§9-49 的 `same_set` 对照：同批同 payload、只换顺序 = **1.07×**）。

    **代价（同一批 16,000 个窗口，只换顺序，实测）**：监督信号的 **lag-1 自相关 +0.024 → +0.426**；
    相邻步同谱 0.2% → **94.6%**；空窗连续段**最长 17 → 31**（中位仍是 2）；相邻事件数差均值 7.4 → 5.0。
    ⇒ **每一步之间不再近似独立**，这是要拿模型质量去换的东西，且**目前没有 val 可以量化它**（§9-47）。
    **覆盖率**：改后 N 步覆盖的谱面数 = N/96 ⇒ 20,000 步只有 **208/6614 = 3.1%**（现状 100%）。
    这**只有在真的跑多个完整 epoch 时才可接受**：一个 epoch = 634,952 步，按当前 0.199 s/步是 **35 h**。

    **落地要求**（按 CLAUDE.md §3.1，改采样语义**必须先开 RFC**）：
    ① 开 RFC-0036 记录本裁定与上面的实测；② 实现成 @@data.order: round_robin | chart_local@@ 旋钮
    （**默认不改**，门禁路径保持 round_robin）；③ 必须保住：每窗口每 epoch 恰好一次、顺序是
    @@(seed, epoch, 索引)@@ 的纯函数、@@data.workers@@ 续训中性、覆盖率标量语义；
    ④ **章程要求「每 epoch 重新洗牌谱面顺序」**，否则固定前缀偏差会被反复放大。

49. **供给吞吐共享上限已定位：DataLoader 的 `in_order=True` 默认值（2026-09-27 第十轮，决定性）**：
    由 subagent 做的只读定位实验（报告 `runs/_research/q3_supply_ceiling.md` + 探针原始 JSON）。
    **本节推翻 §9-48 的两条结论**（下详）。

    **① 实验 A（决定性）：把 `ChartPairDataset.__getitem__` 换成平凡样本，其余逐字不动**

    | arm | 每窗 payload | W=1 | W=8 |
    |---|---|---|---|
    | real | 22-24 MiB | 1.07 | **5.66 窗口/s** |
    | zeros（真实体积、全零） | 18.28 MiB | **111.0** | **229.0**（= **4.2 GB/s**） |
    | tiny | 2 KB | 1838 | **2862** 窗口/s |

    ⇒ **搬运/队列/共享内存/记账/主进程这条共享路径的容量比 7/s 高 30-400 倍，瓶颈在物化侧。**
    （实验里还抓到一次真事故并修掉：patch 的口径赋值晚于首个 `next()` ⇒ worker 仍是 real、读数无效；
    修后另用 batch 内形状证据确认 patch 真的在 worker 进程生效。）

    **② 根因**：torch 2.13 的 `DataLoader` **默认 `in_order=True`**，而 `_batches_parallel` 没有传这个参数。
    读 torch 源码：乱序先到的结果只被存进 `_task_info`，**`_try_put_index` 只在「按序交付」时被调用一次**
    ⇒ **队头一条慢窗口就让 8 个 worker 干完手上的活集体等索引**。

    | 口径（real，W=8） | `in_order=True`（现状） | **`in_order=False`** |
    |---|---|---|
    | 聚合吞吐 | 6.85 窗口/s | **9.85-10.08** |
    | worker CPU 占用 | 4.8-5.2 核（60-65%） | **8.05-8.15 核（100%）** |
    | 交付 gap | 415-523 ms | **9.4-10.9 ms** |

    `prefetch_factor=8` **不解决**（5.97）。`py-spy dump`：`in_order=True` 时 worker 或在算、或停在
    `_worker_loop` 的 `index_queue.get()`，**无一卡在输出侧**；`in_order=False` 三轮 **8/8 全在算**。
    **重尾是致命项**：每项耗时 p50 568 / p90 2303 / p99 5891 ms。
    推算吻合：`8 / 0.756 s = 10.6`（W=8 物化上限）× 65% 占用 = **6.9/s**，实测 6.85（1% 内）。

    **搬运层被否**：真实批次 17-29 MiB，但 `ForkingPickler.dumps` 只有 **1.1 KB / 3 ms**（file_system 句柄共享、
    零拷贝、`%TEMP%` 新增文件 0）；盘并发实测 **1587-1759 MiB/s**，需求 ≈490 MiB/s。

    **物化分相（单线程，100% 对账）**：**谱面解析 296 ms（37%）+ 遮盖构造 274 ms（34%）+
    特征 npz 116 ms（14%）** + 子谱/建场/事件轨/校验 116 ms。

    **③ 「改成谱面序」能不能突破它？——否。**
    默认管线下谱面序 5.89 vs 轮转 5.66-6.85（**噪声内**）；管线修好后 12.88 vs 9.88（1.30×，
    只是把每窗口 805→606 ms 兑现）。**新增 `same_set` 对照**（同一批 480 个窗口、payload 差 0.15%、
    只改顺序、`in_order=False`）：谱面序 **12.64** vs 交错 **11.77** = **1.07×** ⇒ 谱面序的收益
    **几乎全部来自消灭「跨 991 桶的整谱重复解析」**（解析 296→81、特征 116→18），**同一桶内部再排序没有额外收益**；
    而且它把**遮盖构造成本抬高 47%**（274→404，因窗口构成不同）。
    ⚠️ 先前 §9-48 的 3.74×/1.27× **有固有混淆**：两种顺序在有限窗口内访问的不是同一批窗口
    （谱面序 mean 51.9 MiB/窗 vs 轮转 22.4）⇒ **以 `same_set` 的 1.07× 为准。**

    **④ 顺带修正两条既有结论**：
    - **「N≥8 封顶、8 是拐点」是 `in_order` 的产物**：`in_order=False` 下 W=12 仍到 **12.21**、
      W=10 = **11.55**（**首次超过需求 ~11.5**）⇒ `configs/phigros_masked.yaml` 里那段标定注释需要重写；
    - §9-48 的「首要嫌疑 = 共享内存搬运」**已被实验 A 否掉**（已就地标注）。

    **⑤ 下一步改哪一层**：
    - **第 1 层（收益最大）**：`_batches_parallel` 传 `in_order=False`。**但它不是纯配置项**：
      `for batch in loader: self._cursor = pending.popleft()` **假定按序交付**，乱序会让「槽位 ↔ 批」错位，
      **覆盖率、续训定位、RFC-0034 §5 的「顺序是纯函数」全部失真**。正确改法 = **让批自带槽位号**
      （collate 返回 `(slot_stop, FieldBatch)` 之类），主进程按批自带槽位记账；护栏需同步更新。
    - **第 2 层**：物化层 `_build_sample` —— 谱面解析 + 特征读取（55%）可被谱面序砍掉 73%/85%（已实测）；
      **遮盖构造 274-404 ms 是本轮唯一看不懂的大头**（含重掷种子循环，约 1.0% 的步走满 8 次），
      建议**先向下剖析再动**（它还决定 `r==0` 退化率，语义敏感）。
    - **不建议动**：搬运层（1.1 KB/3 ms，余量 18×）、盘（余量 3-30×）、**配置加 worker**
      （`commit` 已 35.8/37.0 GiB，页文件上限自己在长到 45 GiB；W=12 时可用物理最低 7.4 GiB）
      —— **`in_order` 修好之前加 worker 只吃内存不涨吞吐**。

    **存疑（报告 §11 共 11 条）**：P/E-core 放置未测（CPU 总量饱和已排除：全系统 9-11/24 核）；
    无 `Memory\Pages/sec`，分页是否为共同因素判不了；0.480 s vs 0.85 s 的单窗口冷成本差异**未复现**
    （本次实测 0.76-0.92 s）；`same_set` 只覆盖 1 个桶的 480 个窗口。
      ④ **巡检脚本的「步时退化」告警对 K 不敏感 ⇒ 会误报**：`workers=8` 段内 16200→17800 步，
      `data_share` 稳定在 15–26%（数据侧没有退化），但步时 mean 从 0.189 漂到 0.224——涨的是
      **计算侧**（compute mean 0.139 → 0.178），与 K p90 从 53 升到 62 一致（取批顺序推进到
      更大的桶）。告警因此在 step ≈17800 触发一次 `1.50x`。这是与 §9-42 那个恒真告警**同一类**
      的量具缺陷（读数不控制混杂变量）⇒ 下一步应改成**按 K 分层比较**，而不是全局均值比。

50. **§9-49 的第一层已修：`DataLoader(in_order=True)` 的队头阻塞（2026-09-27，已落地）**：
    §9-49 把根因与收益都量出来了，但没有落地。本节是它第 ⑤ 条第 1 层的实现与实测。

    **修法**（`beatmorph/infra/train_loop.py`；**不动取批顺序、不动契约、不动配置语义**）：
    ① `DataLoader(..., in_order=False)` —— 每交付一批就补发一个索引，worker 不再集体等队头；
    ② 但 `in_order=False` 让**交付顺序变成任意排列**，而 `_cursor` 是「计划前缀长度」一个标量，
       覆盖率在线标量 / 续训 O(1) 定位 / `data.workers` 语义中性（RFC-0034 §5）全由它保证
       ⇒ **每批自带槽位终点标签**：`_indices()` 产出的 index 列表首元素是负编码 `-stop-1`，
       `_SlotTaggedDataset` 在 worker 侧翻成 `SlotTag`，`_collate_with_slot`（经
       `functools.partial` 传递，因此测试里 monkeypatch 的 stub collate 仍能在 worker 里还原）摘下；
    ③ 主进程用 `dict` 缓冲按槽位**重排回计划顺序**再产出 ⇒ 并发拿满，而产出顺序**严格等于**
       `workers=0`。缓冲上界 = DataLoader 在飞批次上界（`prefetch_factor × workers`），与
       `in_order=True` 时 `_task_info` 自己攒的乱序批次**同一量级**，不新增内存风险。

    **实测**（`scripts/local_inorder_ab.py`，真实 train split、`workers=8`、**ABBA 交错**）。
    为什么必须交错：首版按固定次序跑两臂，把「`in_order` 的收益」与「后一臂吃前臂留下的暖
    文件缓存」混在了一起；交错后混淆被对称抵消。60 s × 2 臂：

    | 臂 | 窗口/s | 交付 p50 / p90 / p99 | 主进程核数 |
    |---|---|---|---|
    | `ordered`（旧管线，`in_order=True`） | 6.33 / 6.46 → **6.39** | 0.5 / 561-575 / 2072-2334 ms | 0.04-0.05 |
    | `reordered`（新管线 = 生产路径） | 8.76 / 8.59 → **8.67** | 0.1 / 15-139 / 2809-2861 ms | 0.15-0.16 |
    | `raw`（不重排的裸 `in_order=False`，**上限**） | 8.68 / 8.75 → **8.71** | 88 / 247-252 / 484-495 ms | 0.03-0.04 |

    ⇒ **1.36×**；重排相对「裸乱序」只花 **0.5%**（25 s 窗口下曾读到 9%，那是重排缓冲**首次
    填满的启动瞬态**被算进了短窗口，60 s 窗口下消失）。

    **「语义零变更」的实证**：两臂 `windows_seen` 轨迹**逐位一致**（60 s 对齐 **389 步**，
    25 s 对齐 160 步）—— 吞吐变了，模型看到的东西一个字节没变。
    新增 6 项护栏 `tests/unit/infra/test_inorder_pipeline.py`：用**假 `DataLoader`** 把「整段
    倒序交付」这个**最坏情形**变成确定性输入（真实调度下的乱序是概率性的，那样写的测试会
    时灵时不灵），断言重排后与 `workers=0` 逐位一致、`in_order` 确实传成了 `False`、A/B 开关
    能退回 `True`、以及**丢批必须抛**（静默少训会让 `_cursor` 停在错误槽位）。

    ⚠️ **重排把尾部等到了队头**：`reordered` 的 p99 反而更长（2.8 s vs 2.1-2.3 s）——乱序交付时
    消费者拿到的是**最快完成**的那一条，重排后必须等**计划里的那一条**。总吞吐净赚 1.36×，
    但这条尾巴是真代价，记在这里不粉饰。

    **仍然不够，而且不该靠加 worker 补**：需求约 11.5 窗口/s ⇒ 8.67 仍差约 25%。但 ① 内存
    （`commit` 曾到 38.1/40.1 GiB）；② **端到端剩余空间本来就只有约 1.29×**（compute mean
    0.151 s vs step mean 0.195 s，RFC-0035 §5）。下一层是**物化侧的整谱重复**（每窗口 91.3%
    的 CPU 花在「一张谱只该做一次」的事上），即 RFC-0035 的 M0/M1/M2；§9-49 第 ⑤ 条第 2 层
    （遮盖构造 274-404 ms）仍**未被解释**，动它之前要先向下剖析。


51. **val 路径落地 + 「单步 loss ↔ 窗内事件数」解耦（2026-09-27，已实现）**：§9-47 的 A–I 逐条落地，
    并按任务书补上 §9-46 / §9-47 I 的训练侧仪表。**判据与损失的数学一个字未改**（B3 /
    CLAUDE.md §3.1）——本轮修的是**量具与选择判据**，不是目标函数。

    **① 逐条对账（§9-47 → 落点）**

    | §9-47 | 落点 |
    |---|---|
    | A / A1 固定验证集 | `ManifestValSource`（`train_loop.py`）：`plan_epoch(split_val, seed, epoch=0)` 的
    | | **前 `optim.val_windows` 个槽位**，每次调用**原样重放**（同窗口 + 同遮盖种子，因为遮盖种子由
    | | `_window_seed(seed, row_index, window_index)` 派生 ⇒ 续训后逐位一致） |
    | A2 eval/no_grad | `evaluate_val`：`model.eval()` + `torch.no_grad()`，`finally` 里**无条件**恢复原状态 |
    | A3 同测度基线 | `eval/val_metrics.py` 的 `constant_field_solution` / `constant_field_nll`：常数场在
    | | **同一遮盖测度**下由 `c* = s·N_sup/\|Ω\|` 求最优，再用 `masked_poisson_loss` 本身求值 |
    | A4 归一化与分解 | `masked_readout`：`reduction="mean"`（按有效线数）、`nll_empty`/`nll_nonempty`、
    | | `integral`/`events`/`pred_over_true`/`r0_share`/`empty_share` |
    | A5 对照臂 | `val/nll_shuffled`（复用 G2 的 `shuffle_hidden_counts`，只在遮盖格内置换）+
    | | `cond/audio_zero_delta` / `cond/audio_perm_delta` / `cond/track_zero_delta`（同批同权重前向差分） |
    | A6 schema | `optim.val_windows`（默认 128）+ `optim.val_every`；成本算式写在两份注释里 |
    | A7 worker 路径 | `ManifestValSource.batches()` 走与训练**同一套** SlotTag/重排 DataLoader |
    | A8 keep_best | `_save_best`：判据改为 `val/ratio`，且在**测出更优的那一步**写盘（不随 save_every） |
    | A9 落盘 | val 读数并进该步的训练标量行，走 `_flush_scalars`（TB 标签表 `SCALAR_TAGS` 一处定义） |
    | A10 共用指标 | 全部口径在 `beatmorph/eval/val_metrics.py`；全事件口径走离线评估的
    | | `calibration.poisson_nll_float`（**不复制一份**） |
    | B1 积分比 | `train/integral_per_event = Σ∫λdV / Σn`（∫λ 走同一个 `integral_term`，零额外前向） |
    | B2 分层 | `stratified_step_loss`：`train/loss_empty` / `train/loss_nonempty` + 两个占比；
    | | 用的是损失**本身**（`reduction="none"` 逐窗口求和），不改梯度路径 |
    | G-① 索引坑 | `ManifestValSource._ensure` 在构建**之前**先打「首次要解析该切分每张谱面」的日志 |
    | G-② 与 gates 不可比 | 训练开始处显式打日志：`initial_head_bias` 只在门禁路径生效（训练路径 bias=0） |
    | G-③ 精度 | val 沿用 `optim.precision`（bf16 舍入确定性，不影响跨步比较），日志里说明 |
    | G-④ 不得成为训练信号 | `no_grad` + 不按 val 调 LR / 不早停；护栏断言「带 val 与不带 val 的
    | | 训练损失序列逐位一致」 |
    | G-⑤ 空窗占比 | `val/empty_share` 与 `val/nll` 同行落盘（**水平**受它影响、趋势不受） |

    **② 一条推导出来的恒等式（纠正 §9-47 B 的读法）**

    在 `hidden` 重标定口径下 `s·N_sup = (1/r)·N_h = N_h/(N_h/N) = N`——**重标定把「被遮盖事件」
    精确还原成「全部事件」** ⇒ 常数场在遮盖测度下的最优值**数值上等于**全事件闭式基线
    `N(1 + log(|Ω|/N))`。也就是说 §9-46 那份调研用全事件基线算出的 **0.834 与 `val/ratio` 是同一个量**
    （差别只在批与归约）。**这不是「绕过了配对纪律」**：相等是 Horvitz-Thompson 重标定在常数场上的
    推论，而实现仍然是在**同一测度里**求值（`tests/unit/eval/test_val_metrics.py` 用暴力扫描把
    「报告值 = `masked_poisson_loss` 的极小值」钉死）。⚠️ 该恒等式**不是恒真的**：`hidden_doc` 与
    `none` 口径下两个数不相等 ⇒ 不能据此认为「随便拿个基线都行」。

    **③ 成本：§9-47 E 的算式少算了三件事（实测 + 推导）**

    CPU-only 探针（`runs/_scratch/val_cost_probe.py` / `val_composition_probe.py`，**未入库**；跑的时候
    父 agent 的两个 CPU 重活正在并发 ⇒ **时间读数是上界**）：

    | 量 | 实测 | 对照 |
    |---|---|---|
    | val split 规模 | **74,889 窗 / 814 行**（索引缓存 1.5 MB） | §9-47 A 说 814 张（**吻合**） |
    | val 索引**首次**构建 | **301.9 s ≈ 5.0 min**（缓存命中重载 0.2 s） | G-① 估计 4-5 min（**吻合**） |
    | 前 128 槽位 | 128 窗 / **93 张谱** / **128 个网格桶** | ⇒ 每窗各自成桶 ⇒ **B=1 的前向 ×128** |
    | workers=0 单窗 | **1.07 s** | E 说 0.85 s |
    | workers=8 单窗 | **0.276 s**（128 窗 35.3 s）；512 窗 ⇒ 0.184 s/窗 | E 的算式用 **0.038 s** |
    | 空窗占比 | 前缀 128：**51.6%**；前缀 512：**59.0%** | G-⑤：它是常数（**但见 ⑥**） |
    | K（线数）中位/均值/p90 | 30.5 / 46.4 / **106** | 训练流 K 中位 25-26 / p90 53-67 |
    | 退化 r==0 占比 | 1.56%（512 窗 1.37%） | §9-43 的训练侧 1.0-1.8%（**同量级**） |

    **推导**（用 §7.5 的 `步时 ≈ 5.4e-4·K² s`、前向约占 1/3 ⇒ `≈ 1.8e-4·K²`）：val 前缀的 K 分布尾部很重
    （p90 = 106 ⇒ 单窗前向 ~0.6 s），逐窗前向均值约 **0.2-0.4 s**。于是**一次 val**：

    - 5 个臂（打乱 1 + 干预 3 + 模型 1）：`5 × 128 × 0.2~0.4 s + 15~37 s(数据) ≈ **143-293 s**`，
      而 1000 步的训练预算只有 `1000 × 0.199 = 199 s` ⇒ **0.7-1.5×**，不是 5.8%；
    - 即便只跑单臂（E 的口径）：`≈ 41-88 s` ⇒ **21-44%**，仍远高于 5.8%。

    三个原因：① **对照臂把前向次数 ×5**（E 只算了 1 次前向，而 D 列的臂是**强制**的）；
    ② 数据侧单窗不是 0.038 s 而是 0.12-0.28 s（`in_order` 修好后的训练侧实测是 8.67 窗/s ⇒ 0.115 s/窗）；
    ③ val 前缀的 K 分布比训练流重（p90 106 vs 53-67）⇒ 前向更贵。
    **所以 `val_every=1000 + val_windows=128 + 5 臂` 的真实开销与训练预算同量级。**
    把 val 压回 5% 量级的三条出路（**需决策者裁定，本轮不改默认**）：`val_windows≈32`、
    `val_every≈4000`、或**对照臂降频**（例如每 4 次 val 跑一次干预三元组）。

    **④ 验收（全部在默认 CI 内，无权重 / 无 GPU / 无真实语料）**

    - `tests/unit/infra/test_val_path.py`（19 项）：**T1** val 批两次调用逐位一致 + `data.workers=0/2`
      **逐位一致**（真实 spawn）+ 前缀 == plan 前缀 + 跨实例可复现 + 空 split 降级为告警；
      **T3** 「一空一空非」批上 `loss_empty`/`loss_nonempty` 落桶正确（空窗损失**恒等于积分项**）而
      混合的单步 loss 落两者之间；**T4** 剧本化 `val/ratio` ⇒ `best.pt` 落在最优那一步（且与训练损失
      最优步**不同**，防恒真）；**val 不改变训练**（带/不带 val 的损失序列逐位一致）；
      产出键 ⊂ `SCALAR_TAGS`（jsonl 与 TB 不许漂）；`val_every`/`val_windows` 不进续训指纹。
    - `tests/unit/eval/test_val_metrics.py`（17 项）：**T2** 四档重标定口径下**暴力扫描**复核常数场
      最优值 + HT 恒等式 + r==0 退化分支 + 空窗损失 ≡ 积分项 + 跨批**按有效线加权** + 缺失不填 0 +
      全事件口径确实走离线入口。
    - **T5**：`ruff check .`（**仅剩父 agent 并行在写的 `tests/unit/data/test_window_cache.py` 的 1 项
      PT018，与本轮改动无关**）、`mypy beatmorph`（81 files 无问题）、
      `pytest -m "not slow and not gpu and not e2e"` ⇒ **1081 passed / 17 deselected**（基线 1039；
      其中 36 项是本轮新增，其余为同一工作区里并行 agent 新增的测试）。

    **⑤ 存疑清单（我们不知道什么）**

    1. **`val_windows=128` 的构成没有收敛**（G-⑥ 的**可测部分**，本轮实测）：空窗占比从 128 窗的
       **51.6%** 变到 512 窗的 **59.0%**（-7.4 pt）、K 中位从 30.5 变到 26.0 ⇒ 这是**固定系统偏差**
       而不是噪声，128 个窗口的指标**水平**会跟着偏。**指标差本身未测**（需要 GPU 前向，本任务禁止）。
    2. **val 与 gates.txt 不可比**（G-②，已按字面要求打日志）：门禁路径 `initial_head_bias=20`、训练
       路径 `bias=0` ⇒ `val/nll` 与 gates.txt 里的数字不是同一初值下的读数，**只能各自内部比较**。
    3. **一次 val 的真实墙钟未测**：只有数据侧实测（35.3 s / 128 窗，争用下）与按 K² 律的**推导**；
       前向侧需要 GPU，本任务明确禁止。上表的 143-293 s 是推导值，不是实测值。
    4. **对照臂的统计效力未知**：`cond/*_delta` 在固定 128 窗上是确定性数字，但「Δ 多大才算音频被
       用上了」没有基线（合成夹具上可以造，真实模型上没有对照）。§9-46 说 G2 用同线事件轨就能过
       ⇒ 这三个 Δ 的**符号**可能长期为正而音频贡献很小。
    5. **`val/ratio` 回答不了「架构对不对」**（§9-47 H 原文）：NLL 好 ≠ 谱面可玩；难度条件是否被用上
       需要一个**难度置换**对照（本轮未实现，§9-47 H 提到「可顺手塞进 D 的清单」）；无条件生成的质量
       完全不在 val 的覆盖范围内。
    6. **val 的 K 分布比训练流重**（p90 106 vs 53-67）——这是采样的偶然还是 val split 的系统性质
       （按曲目切分的 814 张谱比 train 更难/线更多），**未查证**；它同时影响 val 的成本与读数水平。
    7. **`data.workers` 在 val 路径上的正确性只在 CI 的 stub 上验过**（真实 spawn + 逐位一致），
       真实语料上的 val worker 路径**没有实跑过**（需要真实数据 + 时间；本轮只跑了 CPU 探针）。

**§9-54 真实门禁 G2 回归：门禁批的抽取方式被取批计划层改掉（2026-09-28 本轮实测）**

`beatmorph-train --config-name phigros_masked --gates --device cuda` 本轮**未通过**：G1/G3/G4 全绿，
**G2 FAIL**（真实 `182.856` vs 打乱 `-31.199`，需打乱 `>= 192.0`），退出码 5 ⇒ 训练被 fail-closed 拒绝。
权威记录：`runs/phigros_masked/20260927-180550/gates.txt`（git `8796ff8`）。

**① 不是模型 / 损失 / 数据侧的改动**（逐文件核对 `git log --oneline b96b12f..HEAD -- <file>`）：
`generation/model.py`、`generation/losses.py`、`generation/masks.py`、`infra/sanity.py`（G1-G4 判据本体）
**自 `b96b12f` 起未被触碰**；`infra/smoke.py` 的改动只有 `batches()` / `coverage()` 两个协议方法，
G2 的打乱实现 `shuffle_hidden_counts` **逐字未变**；`data/tracks.py` 的向量化在 `45a46cd`（上一次全绿）
之前就已在场 ⇒ 也不是它。本轮唯一改到门禁输入的是 **`cc0e166` 的取批计划层**
（`ManifestBatchSource._draw` / `_consume` 改走 `plan_epoch`），`gates.py` 的同批改动只有 `workers = 0`。

**② 门禁批退化成了「某张谱的开场」**：`_consume(16)` 从计划**头部**取 16 个连续槽位；实测
`plan.order[:24] = [2154..2169, 8050..8057, ...]`，前 16 个窗口是**行 80/67 开头的 16 个连续窗口**，
即一张谱的开场。开场系统性偏空、偏单线（全库均值约 17.4 事件/窗，门禁批只有 3.9）。实测：

| 批（抽取方式） | K | 事件 | 被遮盖 | 用到的线 | G2 |
|---|---|---|---|---|---|
| 门禁批＝计划头部 | 24 | 63 | 32 | **1** | **FAIL**（4 次独立复跑，real 162.6~186.5 / shuf -53.5~-15.1，全 FAIL） |
| `plan@0.50` | 37 | 132 | 73 | 5 | FAIL |
| 计划第 3 批 | 24 | 154 | 83 | 3 | FAIL |
| 计划第 9 批 | 7 | 119 | 87 | 1 | FAIL |
| 计划第 10 批 | 10 | 248 | 158 | 1 | FAIL |
| 计划第 1 批 | 28 | 174 | 106 | 7 | **PASS**（real -69.5 / shuf +17.6） |
| 上次全绿 `45a46cd` | 33 | 249 | 179 | 61 | PASS |
| 全绿 `b96b12f`（bf16 那一次） | — | 777 | 568 | 11 | PASS |
| 全绿 `e4cda09` | 33 | 764 | 544 | 29 | PASS |

**③ 为什么不能「调配置追绿」**：改动**只有门禁批的抽取方式**，而 G2 的结论**随批剧烈变化**
（本轮 6 个批 5 FAIL / 1 PASS），且**没有任何单一量能把两者分开**——线数 5 FAIL / 7 PASS、
事件数 **248 仍 FAIL**、事件/线 71 PASS 而 26 FAIL。把 `gates.batch_min_events` 抬到刚好命中
通过的那一批，就是**拿结果反推判据**（CLAUDE.md 红线 7 的反面）。本轮的处置：**不调门禁、
不跑全量训练**，把结论留给决策者。

**④ 需要裁定的两个问题（属方法论，须 RFC）**：

1. **门禁批的抽取口径**：门禁是「验证训练目标 / 模型」的夹具，**不应继承训练采样器的顺序**
   （它现在会继承，于是门禁结论跟着采样器改版漂移）。至少应规定：不得只取计划头部、必须跨桶
   跨谱取、且**批构成要落盘可复查**（现在只落了 `g2_events` / `g2_lines`）。
2. **G2 判据的统计效力**：G2 比的是「同一批上 100 步后的**训练**损失」，它天然随批变化
   （本轮 6 批 5 FAIL，而上一次全绿是 PASS）。要么改成**多批平均 + 方差**判据，要么显式
   声明它是「单批抽样」并规定抽取方式与置信度。
   ⚠️ 这一条同时意味着：**此前所有 G2 的 PASS/FAIL 都不可复现**（批不同）——与「2026-09-27 门禁
   口径修复」那次「此前 G2 数字作废」是同一类问题，只是那次是判据错，这次是**抽样错**。

**⑤ 本轮同时测到的另一件事（与门禁无关）**：窗口预切缓存的**构建速率有硬上限**——
`--jobs 12` 与 `--jobs 20` 的**聚合吞吐相同**（**13.0 / 13.19 窗口/s**，`shard_windows=128`），
即 **≈0.92 CPU-s/窗**已是这台机器的上限（约 12 核当量的吞吐）；加进程只吃内存
（`--jobs 20` 时可用物理内存 2.0-2.5 GiB、commit 余量 <1 GiB，而聚合吞吐**一点没涨**）。
⇒ 全库 train **634 952 窗 ≈ 13.4 h**（**不是** §9-52 的 18.5 h，那是 8 进程线性外推）、
val 74 889 窗 ≈ 1.6 h；磁盘 339 GiB。半成品目录已清掉。

**⑤-补（同日稍后，已启动）**：**全量构建已于本轮启动**（`--jobs 12 --shard-windows 512`，train → val 串行）。
**⑥ 门禁批并不「明显不具代表性」——这一点必须写清楚**：训练批是 `optim.batch_size=1`
（跨谱 batching 需要同一网格身份），训练流按**计划顺序**逐步消费 ⇒ 训练在 step 1 看到的**就是**
计划头部的那一个窗口。因此「门禁批＝计划头部」与「训练流的批分布」**是一致的**
（差别只在门禁一次要 16 个同桶窗口）。⇒ **不能**简单地说「门禁抽错了批」；更准确的读法是：
**G2 的结论随批变化，而门禁只用单批**。本轮实测的逐批通过率 = **1/6**
（第 1 批 PASS，其余 5 个 FAIL，**含事件数 248 的那一批**）。
⇒ 这把问题指向二者之一：**目标函数在典型真实窗口上赢不过打乱对照**，或
**G2 的单批判据统计效力不足**。两者都要决策者裁定，都不是 agent 可以自行调参解决的。

**⑦ 已有的全量产物（本次盘点）**：`runs/phigros_masked/20260927-100604` 是本仓库**唯一**
跑完全程的实验目录——`gates.txt` 全绿、`metrics.json.gates_passed=true`、`loss_history` 覆盖
step 16001→20000（即 `--resume` 从 16000 续跑到 `optim.max_steps=20000`，3 个 checkpoint）。
但它是 **RFC-0034 之前**的采样器下跑出来的 ⇒ 它只证明「流水线能跑完全程」，
**不证明**当前代码能过门禁。其余 22 个实验目录全部 `ckpts=0`。
**⑧ 门禁的真正缺陷：判据的噪声底 ≈180%（bf16），而判据余量只有 5%**

本轮实测（同一批、同一初值种子、同一超参，**只重复同一个臂**）：

| 运行 | step 1 | step 20 | step 50 | step 100 |
|---|---|---|---|---|
| bf16 `real_run1` | 75197.97 | 298.78 | 175.92 | **139.13** |
| bf16 `real_run2` | **75197.97**（逐位相同） | 300.41 | 208.54 | **386.62** |
| bf16 `shuffled_target` | 75211.73 | 364.96 | 244.04 | **19.93** |
| fp32 `real_run1` | 75120.06 | 318.5 | 213.54 | **168.34** |
| fp32 `real_run2` | 75120.06 | 318.5 | 213.54 | **178.75** |
| fp32 `real_run3` | 75120.06 | 318.5 | 213.54 | **178.75** |
| fp32 `shuffled_target` | 75132.41 | 365.12 | 204.66 | **-26.49** |

1. **bf16**：两个**逐位相同**的臂（step 1 都是 75197.97）在 step 100 差到 **139.13 vs 386.62 = 2.8×**。
   而判据余量是 **5%** ⇒ **在 bf16 下 G2 判的是数值噪声，不是模型**。同一次运行的输入干预臂
   （`audio_zero` 148.18 / `audio_perm` 127.66 / `track_zero` 136.24）与真臂 139.13 **无法区分**，
   却都优于 `real_run2` 的 386.62 ⇒ **单次运行的臂间比较不可用**。（此前一次 bf16 剖面上看到的
   「34 < 52 < 68 < 166」的干净序是**运气**，不是证据。）
2. **fp32**：step 1/20/50 **逐位相同**，step 100 = 168.34 / 178.75 / 178.75 ⇒ **重复运行可复现**
   （噪声底 ≈0-6%，只剩首跑的一个 6% 偏差）。
3. **但 G2 的 FAIL 不是 bf16 造成的**：fp32 下 `shuffled_target` = **-26.49** vs 真臂 **168-179**
   ⇒ 仍然 FAIL，且差距约 **200**。

⇒ 这**收窄了 RFC-0036 的选项**：**至少必须**把门禁的对照臂跑在 fp32（或确定性算法）下——
这是**不改任何判据**的工程修复；做完之后 G2 的裁决才可信。
**⑨ fp32 下的输入干预对照：仍然不干净（同一批、同一目标、只改输入）**

fp32（可复现：`real_run1` = `real_run2` = 178.75）：

| 臂 | 输入 | 最终 loss | 相对真臂 |
|---|---|---|---|
| real | 真输入 | 178.75 | — |
| `audio_perm` | 音频**帧轴置换** | **270.22** | 更差 ✓ |
| `audio_zero` | 音频**置零** | **121.78** | **更好** ✗ |
| `track_zero` | 本线轨**置零** | **131.65** | **更好** ✗ |

⇒ 在**这条门禁批**上三个输入干预**互相矛盾**：置换音频让它变差（说明音频时序被用上了），
但**置零音频 / 置零本线轨反而让拟合更好**。⇒ 这条批（谱面开场、32 个遮盖事件、只用 1 条线）
**不适合做任何控制实验**；同时说明「把判据换成输入置换」这条捷径**本轮并不成立**。
⇒ 现状：门禁 FAIL 的**成因仍未分离**（模型 / 目标 / 批三者），但**前置已确定**：
先把对照臂跑在 **fp32** 下（见 ⑧），否则任何裁决都不可解释。
**⑩ fp32 重测：§1.3 的 bf16 表不可用，以本表为准（G2 的 FAIL 是真的）**

fp32（可复现）逐批实测，5 批，同一 `shuffle_steps=100`、同一初值种子：

| 批 | K | 事件 | 被遮盖 | 用到的线 | real | shuffled | 判据 |
|---|---|---|---|---|---|---|---|
| 0（＝门禁批） | 24 | 63 | 32 | **1** | 178.75 | **-26.49** | FAIL |
| 10 | 10 | **248** | 158 | **1** | 102.81 | 78.92 | FAIL |
| 3 | 24 | 154 | 83 | **3** | 256.85 | 3.74 | FAIL |
| 6 | 31 | 131 | 80 | **6** | 37.58 | 32.03 | FAIL（gap **5.55**，差 1.9） |
| 1 | 28 | 174 | 106 | **7** | -74.56 | +70.04 | **PASS** |

- **4 FAIL / 1 PASS，且在 fp32 下可复现 ⇒ G2 的 FAIL 不是数值噪声。**
- **事件数不是判别量**（第 10 批 248 个事件仍 FAIL）；**线数**是唯一能把两者分开的量，
  且 gap 随线数收窄（1 线 205 → 3 线 253 → **6 线 5.55** → 7 线 -145，正负翻转在 6-7 线之间）。
  ⚠️ 只有 **5 个点**，**相关不等于机制**。
- 机制假说（**未证实**）：真实事件全压在一条线上时模型要单线打尖峰，而打乱把同样多的事件摊到
  24 条线 ⇒ 打乱目标更易拟合 ⇒ **单线批上「置换目标」这个控制无效**。
**⑪ 根因定位：G2 的控制改了「每条线的事件分布」（nuisance）——改成线内置换后同批 PASS**

fp32、**同一进程**、就是这条门禁批，三臂同批同初值：

| 臂 | 用到的线 | final loss | 判据 |
|---|---|---|---|
| real | **1** | 166.74 | — |
| 置换但**保持每条线的事件数**（线内置换） | **1** | **257.61** | **PASS**（需 >= 175.08） |
| 置换（现状：全局 `shuffle_hidden_counts`） | **17** | **-26.49** | **FAIL** |

⇒ `shuffle_hidden_counts` 在**全部被遮盖格子**上置换 ⇒ 它**同时**破坏了两件事：
「位置 ↔ 上下文对应」（想要的）与「**每条线的事件数分布**」（不该动的 nuisance）。
门禁批只有 **1 条线**承载事件，全局置换把它摊到 **17 条线**；在一条线上打 32 个尖峰
vs 摊到 17 条线上各打几个——**后者更容易** ⇒ 打乱臂更优 ⇒ 假 FAIL。
⇒ **修复（RFC-0036-P1）：把 G2 的置换改成线内置换**（不改阈值 / 算式 / 指标口径；
是把控制**做强**：只破坏目标变量，不动 nuisance）。与 **P0（fp32）** 合起来，本批 G2 即 **PASS**。
**复测（fp32，每批 3 臂同进程）——线内置换 3/3 全 PASS**：

| 批 | 用到的线 | real | **线内置换** | 全局置换（现状） |
|---|---|---|---|---|
| 0（＝门禁批） | 1 | 166.74 | **257.61 → PASS** | -26.49 → FAIL |
| 10 | 1 | 248.86 | **550.46 → PASS** | 78.92 → FAIL |
| 6 | 6 | 63.37 | **275.95 → PASS** | 96.90 → PASS※ |

※ **第 6 批的全局置换在另一个进程里给 32.03（FAIL），本进程给 96.90（PASS）** ⇒
**fp32 同进程可复现、跨进程仍翻转**（CUDA 核选择 / 分配器状态）。⇒ 即便 P0+P1 都做了，
**单批判据仍然脆弱**——这是 RFC-0036 §3.B（多批口径）仍然必要的直接理由。
**⑫ P1 的覆盖面：5 PASS / 1 FAIL（不是万能药）**

再测 3 条批（fp32，同进程，real 与线内置换两臂）：

| 批 | 用到的线 | real | 线内置换 | 比值 | 判据 |
|---|---|---|---|---|---|
| 5 | 1 | 350.84 | 218.73 | 0.62 | **FAIL** |
| 9 | 1 | 162.68 | 284.33 | 1.75 | PASS |
| 11 | 2 | -74.37 | -25.61 | 0.34 | PASS※ |

⇒ 线内置换合计 **5 PASS / 1 FAIL**（0/10/6/9/11 PASS，**5 FAIL**）⇒ **P1 修掉了 nuisance 但不足以保证判据成立**。
⇒ 「只做 P0+P1 就开训练」**不安全**；RFC-0036 §3.B（多批口径）是**必需**而非可选。

**⑬ ~~第三个缺陷（P2）~~【更正 2026-09-28：P2 **不成立**，撤回（RFC-0036 §2.6 已同步更正）】**

原文论证判据 `shuffled >= real*1.05` 在 real < 0 时会放行「打乱臂反而更优」。**复核**：生产判据
`sanity.py::shuffled_target_control` **自 `013f347`（第四轮）起已是符号安全式**
`need = real + min_gap_ratio*|real|`（real > 0 时与 `real*1.05` 逐位相同；real < 0 时要求打乱臂更差；
当初就是为修「−326.33 / −327.25 假绿」而改的），失败 run `8796ff8` 的 gates.txt 与该实现逐字一致。
符号敏感式出自本轮 `local_g2_*` **探针脚本自行实现的简式**，不是仓库判据。
按生产公式复核已记录的负 real 批：第 1 批（−74.56 / +70.04，need −70.83）PASS、
第 11 批（−74.37 / −25.61，need −70.65）PASS——**verdict 全部不变**；其余批 real > 0，两式相同
⇒ **①–⑫ 的全部数字与结论不受影响**。⇒ **P2 撤回：无需裁定、无需改任何判据代码**。
⚠️ 教训（AGENTS.md §3 同类）：**探针不得重新实现判据，必须复用生产比较器**——本轮连 RFC 提出者
都把探针简式误当成生产判据，写进了 RFC / plan / README 三处文档（均已更正）。

**⑭ 窗口预切缓存：当前代码上的有界排练复核（第十三轮·数据侧）**

第十二轮的逐位一致验收（4581/4581）做在 `cc0e166`（RFC-0034 计划层）之前的代码上；本轮在**当前代码**上
重建有界切片复核（`scripts/local_cache_validate.py --rows 20` ⇒ 2 154 窗，按步长抽 100 窗）：

- 数据集**接受**该缓存（指纹匹配，`_open_window_cache()` 非 None）；
- **逐位一致 100/100 OK**（6 张量字段 + 4 标量 + grid 逐字段 `torch.equal`）；
- 读取吞吐 25.8 窗/s（38.7 ms/窗）——在 12 进程全量构建抢盘的干扰下测得，低于第十二轮单独实测的
  104.7 窗/s，方向符合预期（构建期盘被占满）；
- 同轮修掉 `beatmorph/cli/build_windows.py` 的 `--shard-windows` 守卫死代码（`or` 把 `0` 洗成默认值，
  校验分支永不可达；一次 `--shard-windows 0` 的测试因此真的启动过全库构建）+ 4 项 CLI 单测
  （`tests/unit/cli/test_build_windows.py`，默认 CI 1082 ⇒ 1086）。
- ⚠️ **探针陷阱**：stdout 重定向到文件时仍走 GBK，`⇒`（U+21D2）让打印结果一步 `UnicodeEncodeError`
  崩溃（构建已完成、结果被吞）⇒ 长跑重定向探针的输出**只用 ASCII 箭头 `=>`**。

结论：builder + reader 在当前代码上仍逐位一致。全库缓存（train 634 952 + val 74 889 窗）建成后的终验
= 同一脚本 `--rows 0 --reuse --root data/processed/window_cache --sample 400`（val 指纹应为
`fac0448693925c86`）。

**⑮ 全库窗口缓存建成 + 终验通过（第十三轮·数据侧收官）**

`--jobs 12 --shard-windows 512 --chart-cache-size 2 --feature-cache-size 1`，train → val 顺序单作业跑完
（`train exit=0` / `val exit=0`）：

| split | 目录（指纹前 16 位） | n_windows（index.json） | 体积 | 分片文件数 | 墙钟 | 均速 |
|---|---|---|---|---|---|---|
| train | `train/aa677031c17e3a3d` | **634 952**（= 计划值，精确一致） | 353.6 GB | 12 411 | 03:23→14:46 ≈ **11.4 h** | 15.5 窗/s |
| val | `val/fac0448693925c86` | **74 889**（= 计划值，精确一致） | 41.7 GB | 1 471 | 14:50→16:02 ≈ **1.2 h** | 17.3 窗/s |

- 指纹与事前计算**完全一致**（train `aa677031c17e3a3d…`、val `fac0448693925c86…`）；`*.partial` 残留 = 0；
  index.json `version=1`、`t_bins=192`、`x_bins=128`、`feature_dim=1024`。
- **终验**（`scripts/local_cache_validate.py --rows 0 --reuse --root data/processed/window_cache`）：
  train 缓存被数据集接受、**逐位一致 400/400**、读取 **95.2 窗/s**（10.5 ms/窗 = 原路径 81×）；
  val 接受、**逐位一致 100/100**、**76.2 窗/s**（13.1 ms/窗 = 65×）。两个 exit=0。
- 与第十二轮 50 行切片实测（559.8 KiB/窗、104.7 窗/s）同量级；全库合计 **395.3 GB**，D: 余 460 GB。
- ⇒ **数据处理（全量窗口预切缓存 train+val）完成并验收**。全量训练（`max_samples=null`）的数据侧前置已清；
  训练本身仍被 **G2/RFC-0036 裁定**挡住（见 §9-54 ①–⑬，红线 7：门禁未绿不得扩大数据规模）。

### §9-55 第十三轮②：损失归约口径查证 + 20k 步曲线波动归因 + G2 有效性复核（决策者质疑，2026-09-28）

决策者提出两项质疑：① G2 判据是否真的有效（「G2 根本不能反映有效信息」）；② 最近一次
20 000 步训练里 loss 与 grad_norm 上蹿下跳，是否训练不稳定——并要求查证损失的归约口径
（对窗口平均还是求和？对 mask 数平均还是窗口数平均？）。以下为**代码级 + 数据级**查证。

**① 训练损失的归约口径：纯求和，没有任何平均**

- 训练步：`model.forward(compute_loss=True)` → `masked_poisson_loss(output, batch)`
  （`generation/model.py:538`，**默认 `reduction="sum"`**）。逐**有效线**的损失 =
  `(1/r)·Σ_{被遮盖格} −n·log λ`（事件项，对被遮盖格**求和**，不除以 mask 数）
  `+ Σ_{全域 T×X×S×C} λ·dV`（积分项，对全网格**求和**），再对**有效线求和**。
  `optim.batch_size=1` ⇒ 每步一个窗口，**也不存在对窗口数的平均**。
- `reduction="mean"` 的分母是**有效线数**（`losses.py::_reduced(active=line_active)`），
  不是 mask 数 / 事件数 / 窗口数；训练路径不用它，**val 用它**（`val_metrics`：
  `val_nll = Σ损失/Σ有效线`，`val/ratio = val_nll / val_nll_constant` 无量纲）。
- 日志口径：`loss` = 该步 sum；`loss_empty`/`loss_nonempty` = 逐窗口 sum 按类平均
  （`stratified_step_loss`，train_loop:501-512）；`integral_per_event` = Σ∫λ/Σn；
  `grad_norm` = **裁剪前**总范数（`clip_grad_norm_` 的返回值，train_loop:1200-1202）。
  ⚠️ 更正 §9-46 的「`raw_grad_norm` 未落盘」——**该说法错误**：jsonl 里的 `grad_norm`
  就是裁剪前的原始值（裁剪后的值在触发时恒为 1.0，无需落盘）。

**② 20 000 步 run（`20260927-100604`，jsonl 20 000 行去重）的波动归因**（`scripts/local_loss_stability.py`）

- **量级由批构成支配**：log-log 回归 `loss ≈ 2.24·K^0.055·E^0.93`（R² = **0.744**）——
  单步损失量级 ~**线性于事件数 E**、对 K 几乎不敏感；corr(loss,E)=+0.51、corr(loss,K)=+0.06。
  加上 **43.8% 空窗**的双峰（loss_empty ~1e-3 vs loss_nonempty ~90，五个数量级），
  **原始逐步曲线的锯齿是测度的结构性质，不是稳定性信号**。
- **无经典不稳定证据**：NaN/Inf **0**；对 (K,E) 回归后的未解释尖峰 **4/11 248 非空步
  （0.04%）**，最大残差在 step 1（初值伪影）；grad_norm（裁剪前）分段中位 **279→349 平稳**、
  无爆炸无塌缩；corr(grad,loss)=+0.68 而 corr(grad,K)=+0.03、corr(grad,E)=+0.05
  ⇒ **grad_norm 的跳动跟随 loss（即批构成），不是独立的失稳信号**。
- **两个真实信号（不是「不稳定」，但必须记录）**：
  (a) `grad_clip_norm=1.0` 在 **56.4%** 的步触发，非空步裁剪前中位范数 ≈ **316**
      （step 1 为 3.4e5）⇒ 触发步的有效更新 ≈ **单位方向 × lr**（归一化梯度下降形态）。
      它压住了爆炸，但也让 lr 与梯度幅度脱钩、且掩盖「损失量级 ∝ 批构成」对步长的影响
      （§9-46 已登记，此处补上定量）。
  (b) `loss_nonempty/E` 在 ~4 000 步后**平台并轻微上行**（8.4 → 10.2），但分段非空占比
      漂移极大（89% → 31% → 63%，计划层按轮发牌的产物）⇒ 段间构成不可比；且该 run
      **没有 val/ratio**（val 第十二轮才落地）⇒ **真实训练趋势无法从该 run 的训练曲线判定**。
      首次全量 run 必须以 `val/ratio` 为主判据（已落地，`val_every=1000`）。
- 候选日志改进（随门禁改动一并裁定）：每步补 `loss_per_event = loss/E`（非空步）与
  `loss_per_line = loss/K`，让原始曲线可读；不动优化目标本身。

**③ G2 有效性复核** ⇒ 结论与新增选项 D/E 已写入 **RFC-0036 §2.8 / §3**（决策者裁定）。
要点：现行 G2 =「单批、100 步、未归一化 sum 损失的速度赛」，量级 ∝ E^0.93 而判据余量 5%；
真实批上判别力**未被证明**且反证充分（逐批 verdict 翻转、P1 nuisance、输入干预自相矛盾、
跨进程漂移）；唯一正面证据是合成解耦夹具（45% 差距）。**健全替代已在仓库里**：val 路径的
`val/nll_shuffled_delta`（固定模型、128 held-out 窗、遮盖内置换、同测度、线数归一）+
`val/ratio` + 输入干预 contrasts。

[POSTMORTEM-2026-08-05]: ../POSTMORTEM-2026-08-05-frame-rate-misalignment.md
