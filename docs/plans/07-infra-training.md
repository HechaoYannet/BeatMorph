> 状态：🔵 实施中（M7.1–M7.8 代码与默认 CI 测试已落地；逐条见 §6 的「实施状态」列，**Lightning 后端尚未在装齐 train extra 的环境实跑**）｜ 阶段：Phase 1（环境自检）/ Phase 2（训练栈与门禁）｜ 负责：基础设施组
> 对应代码：`beatmorph/infra/`、`configs/`、`beatmorph/cli/train.py` ｜ 对应奠基章节：§5、§9

# Plan 07 — 训练基础设施（Lightning / Hydra / 门禁 G1/G3/G4（G2 已随 RFC-0037 删除）/ 环境自检）

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
  --gates            # 强制先跑门禁（G1/G3/G4），结果落盘并在 FAIL 时以非 0 退出码中止
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
  # 1) 组装三个 GateResult（G1/G3/G4；G2 已随 RFC-0037 删除）
  # 2) 写 out_path（summarize() 原文 + 生效阈值 + git rev + data rev）
  # 3) 写 TB 标量（gate/G1_pass 等）供曲线面板对照
  # 4) 任一 FAIL -> 抛出并中止（退出码非 0）
```

三道门禁在**新范式**下的具体接法（判据不变，只换 `step_fn`；RFC-0037 起损失一律
per-event 归一、门禁臂固定 fp32）：

| 门禁 | 判据（BasePlan §9） | 在新范式下的接入 |
| --- | --- | --- |
| G1 单 batch 过拟合 | 1-4 个样本上 loss 打到接近 0 | `step_fn` = 对同一 batch 反复 backward/step；**必须能打穿**，打不穿说明通路坏（与数据量无关） |
| G3 常数基线 | 模型 loss 显著优于 `λ = N/abs(Ω)` | **基线不是 `λ ≡ 0`**（后者泊松 NLL = +∞，应写成契约断言）；`abs(Ω) = k * t_bins * x_bins * sides * channels`（Plan 00 §3.7），`k` 随谱变化 → 标度是 per-chart 的（§9-3）；基线与模型臂**同除** `max(E,1)`（RFC-0037 §2.3） |
| G4 契约断言 | 帧率/形状由 config 派生并断言 | 直接 `frame_rate_gate(frames, duration_s, frame_rate)`，`frame_rate` **由 config 派生传入**；并含**「场网格 ↔ 秒」往返无损**（多 BPM 段，Plan 03 M12）——该断言进默认 CI、不依赖权重 |

> **原 G2（打乱标签对照）已删除**（RFC-0037，2026-09-28 决策者裁定；证据 RFC-0036 §1-§2、
> §9-54/§9-55）。「输入对目标零信息」改由 val 的 held-out 在线对照承担：
> `val/nll_shuffled_delta`（**线内置换**）+ `val/ratio` + `cond_*_delta`；判读红线见
> docs/TRAINING.md §7.1 与 RFC-0037 R1。

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

### §9-56 第十三轮③：RFC-0037 落地——删除 G2 + 训练损失 per-event 归一化（2026-09-28，已裁定即实施）

决策者裁定（原话要点见 RFC-0037 §0）：**G2 的设计本身是错误，直接删除并贯彻到 BasePlan**；
**训练损失必须按被监督事件归一**，并配一组可读的附带指标。实施记录：

**① 损失归一化（R2/R3）**

- `losses.py`：`Reduction` 增 `"per_event"`；`event_normalizer(batch)` = 有效线事件总数、下限 1；
  `masked/full_poisson_loss` 支持；`model.forward(compute_loss=True)` 一律 per_event。
  **整式相除**——只除事件项会把最优强度缩小 D 倍（RFC-0037 §2.2 推导，已写进 docstring）。
- `make_step_fn` 分段前向加**除子修正** `D_part/D_full`（否则分段梯度按 1/D_part 加权 ≠ 整批）；
  门禁分段等价性测试（`test_gate_chunking.py`）在该修正下继续通过。
- G3 基线**同除一个 D**（`constant_baseline_for`），判据不等式逐位等价（§2.3）。
- 新标量（jsonl + TB）：`loss_sum_raw`（旧 sum 口径对照）、`loss_nonempty_per_event`
  （主趋势）、`clip_active`（裁剪触发 0/1）。`grad_norm` 本就是裁剪前值（§9-46 记录已更正）。
- 护栏：`tests/unit/generation/test_loss_per_event.py`（5 项：尺度恒等式、**梯度恰差 1/D**
  （argmin 不变的可执行形式）、空批 D=1、r==0 契约、normalizer 只数有效线）。

**② G2 删除（R1）+ P0/P1（R4/R5）**

- 删除面：`sanity.shuffled_target_control`；`gates.GateInputs.step_fn_g2_*` 与 G2 stage、
  `thresholds_of` 的 `g2_*`（新增 `g3_steps/g3_samples/g3_chunks`）；`build_gate_inputs` 的
  G2 双臂装配；`smoke.shuffled_counts` / `shuffle_hidden_counts` / `batch(shuffled=)`；
  `BatchSource.batch(shuffled)` 协议参数与 `_shuffle_targets`；schema `GatesConfig.shuffle_*`
  四字段（→ `baseline_steps/baseline_samples/baseline_chunks`，语义不变、归 G3 独用）。
- **门禁臂固定 fp32**（原 P0）：`build_gate_inputs` 内 `gate_precision = "fp32"`，
  G1/G3 的 `make_step_fn` 都用它；训练仍 `bf16-mixed`。stats 落 `gate_precision` 进 gates.txt。
- **val 置换改线内**（原 P1）：新增 `smoke.shuffle_counts_within_line`（独立 Generator、
  每 (样本,线) 的遮盖集合内置换），`evaluate_val` 改用它。护栏
  `tests/unit/infra/test_shuffle_within_line.py`（RFC-0037 §2.4 五契约 + 无遮盖批报错）。
- 测试改造：删 `test_g2_pairing.py`（G3 装配/lr/初值三案迁至 `test_gate_assembly.py`，新增
  fp32 装配断言）；`test_sanity.py`、两个 `test_sanity_gates.py`（slow，实跑全绿：
  G1 40.21→2.85、G3 1.578 vs 3.173、G4 双轴）、`test_gates.py`、`test_gate_progress.py`、
  `test_gate_min_events.py`、`test_train_entry.py`（六件套断言 4→3 行且显式断言无 G2）。

**③ 修宪与文档（R1 的文档面）**

BasePlan §9 门禁表（四道→三道 + RFC-0037 修订注记）、§7 Phase2-7、风险表 R-4；
CLAUDE.md 红线 7（含判读红线全文）、§2 拓扑、§5-8、§6 新条目；AGENTS.md §4；
docs/TRAINING.md 约束二、§7.1（表 + 预算 ~10-15 min + 手工片段 + val 判读红线）、§7.2、
§8.1/§8.2、§9；plan 04 §3.3/§4.3；decisions/README：RFC-0036 → **废弃**（证据保留）、
RFC-0037 → **已裁定（采纳）**、下一编号 0038。

**④ 边界与未做**

- 损失语义变更 ⇒ **不与旧 checkpoint / 旧 gates.txt 可比**，不跨此变更 `--resume`
  （GatesConfig 字段变化本身会令续训指纹 fail-closed，属预期）。
- `grad_clip_norm=1.0` 是否随归一化调整：**未动**，先由 `clip_active` 率观察（RFC-0037 §6-1）。
- 判读红线（`val/ratio >= 1` 或 `val/nll_shuffled_delta <= 0` ⇒ 扩规模结论作废）目前靠
  人工/巡检盯 val 曲线，**未**做成自动中止（避免误杀首个 val 点前的暖机段）。

### §9-57 第十三轮④：首次全量训练的**显存事故**与处置（2026-09-28，RFC-0037 落地后首跑）

**事故 1（真实故障）：驱动侧滑进 Windows 共享显存，step 951 卡死。**
run `runs/phigros_masked/20260928-142108`：门禁 G1/G3/G4 全绿（3.5 min），训练跑到
**step 950** 正常（`data_time_s` 0.001-0.007 s ⇒ 窗口缓存生效），**step 951 起 60 s 零新行**。
实测：驱动侧 VRAM 峰值 **7874 / 8151 MiB（96.6%）**，功耗 **102 W → 31 W** 而 util 100%
（决策者同时观测到**共享 GPU 内存 13.2 GB**）——即文档记录的「滑进共享内存」特征。
本 run 的 K：中位 26 / p90 63 / **max 128（0.11% 的步）**；torch 分配器峰值 **5.227 GiB**。

⇒ **更正第七轮的口径**：「bf16 K=128 → 5.10 GiB 亦可训」是 **torch 分配器口径**，未计
CUDA 上下文 / cuBLAS·cuDNN workspace / **分配器碎片**，也未计长期运行；在 Windows/WDDM 上
驱动装不下就**静默回退共享内存**（系统内存），8 GB 卡上「接近上限」= 灾难。**该结论不适用于
全量长跑**，以本节实测为准。

**事故 2（工具误伤）：看门狗 v1 误杀了一次健康运行。** 重试（`20260928-143418`）带了
`expandable_segments:True`：门禁期 mem 1942 MiB / 功耗 ~100 W，训练到 step 50 时驱动侧
**5828 MiB**、torch 峰值 **4.598 GiB**（比事故 1 同期更低，说明 `expandable_segments` 在起作用），
却被我自写的看门狗判成「停滞 241 s」掐掉——**根因**：它拿**上一个 run 的 950 行 jsonl**
当步号基准，新 run 只有 50 行 ⇒ 行数「不前进」。**教训：跨 run 的进度基准必须绑定本次运行的
文件（mtime 晚于启动时刻）**；这条已写进脚本 v2（但该脚本随后按决策者要求撤下，见下）。

**已落地的处置（都不改训练语义）**

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —— ⚠️ **实测在 Windows 上无效**：
   torch 2.13 启动即警告 `expandable_segments not supported on this platform`（no-op）。
   因此「重试期驱动峰值更低（5828 vs 7716 MiB）」**不能**归因于它——那只是步构成不同。
   保留该设置仅因 Linux 上有效、代价为零；**本机真正生效的只有第 2 条的显存卫生**。
2. **显存卫生** `optim.vram_hygiene_gib`（默认 1.0）：每步 backward+step 后，若
   `memory_reserved() − memory_allocated() > 阈值` 就 `torch.cuda.empty_cache()`，把「保留但
   空闲」的块还给驱动——这正是事故的直接机制（峰值保留量长期占位）。护栏
   `tests/unit/infra/test_vram_hygiene.py`（4 项，monkeypatch，无需 CUDA）；
3. 遥测：新增 `vram_reserved_gib`（jsonl + TB `sys/vram_reserved_gib`），与 `peak_vram_gib` 并列；
4. `optim.val_batch: 8 → 4`（留显存余量；val 指标语义不变，仅组批数增多）；
5. **看门狗撤下**（决策者 2026-09-28 指示：由他人工盯并**主动通知**故障，不要自动杀进程）。
   脚本 v2 已删除；本节保留 v1 的误杀教训。

**仍存的风险与下一层杠杆（未决）**：K≈128 的批在 8 GB 卡上本身就贴着驱动上限；卫生回收
把「空闲占位」去掉后**预计**有 ~1 GiB 余量，但**未证**。若再次回退：
(a) **给批 K 设上限**（跳过 K > 预算的窗口）——**改变数据分布**，须决策者裁定/开 RFC；
(b) 检查 gap 判断的显存开销（为后续可能的
    梯度检查点/分块全局注意力留位置）；
(c) 换更大显存的卡。首跑的三条观察点见 README 交接件（`clip_active` 率 / `val/ratio` /
`val/nll_shuffled_delta`）。

### §9-58 第十三轮⑤：全量训练前 1500 步实测（事故修复生效 + 首个 val 的双红线）

run `runs/phigros_masked/20260928-144342`（`vram_hygiene_gib=1.0`、`val_batch=4`，无看门狗）：

- **事故修复生效**：跨过 §9-57 的 step 951，跑到 **step 1500** 健康（驱动侧峰值 7480 MiB，
  但功耗 **78-92 W 未塌陷**）；`vram_reserved_gib` 最大 **1.53 GiB** 而 torch 峰值 **6.79 GiB**
  ⇒ **卫生回收在起作用**（保留量不随历史峰值累积）——这是本机唯一真正生效的显存杠杆。
- **首个 val（step 1000）两条判读红线双双满足**（RFC-0037 R1 首次实测）：
  `val/ratio = 0.7521 < 1`；`val/nll_shuffled_delta = +0.7351`（shuffled 1.690 vs model 0.955）
  ⇒ **「输入对目标有信息」这一命题首次拿到 held-out 上的统计有效证据**——这正是 G2 想测而
  从未测成的命题（G2 时代的所有 verdict 都不可复现，见 §9-54/§9-55）。
- 条件干预（探索性）：`audio_perm_delta = −0.0015`（≈0）、`audio_zero_delta = +0.0111`、
  `track_zero_delta = +0.2484` ⇒ 该 val 批上**判定线轨被用得比音频多**，音频时间对齐几乎没被用
  （一个值得后续追的线索，不是本轮结论）。
- **val 真实墙钟 = 100.52 s**：此前推导值 **33 s**（§9-53）**miss 3×**；步时中位 0.097 s ⇒ 1000 步
  训练 ≈97 s ⇒ `val_every=1000` 让**运行时长大致翻倍（≈50% 开销，不是 16.7%）**。
  建议（下一个长跑）：`val_every` 1000 → **4000**（≈12.5%）或压 `val_windows`；本 run 保持 1000
  以便在线看趋势（val/ratio 是主判据）。
  ⚠️ **就地更正（见 §9-59）**：该 miss 的**根因是 val 跨墙（内存路径），不是算力**——`val_every` 只压
  **频率**、不减**单次**成本。step 2000 的 val 已涨到 **167.58 s**（输入构成与 step 1000 **逐字段相同**）。
- **`clip_active` = 100%**（797/797 非空步）⇒ **per-event 归一化没有降低裁剪触发率**：
  Adam 对损失整体缩放不变，梯度范数依旧远超 `grad_clip_norm=1.0`；有效更新 ≈ 归一化梯度形式。
  这是 RFC-0037 §6-1 的首个答案（阈值是否该调，待专门裁定）。
- `data_time_s` 中位 **2.7 ms**（窗口缓存生效；原路径 0.230 s ⇒ **85×**）。

### §9-59 第十三轮⑥：**更正——val 确实跨墙**（共享显存 5.6 GB，未崩溃，只是变慢）（2026-09-28）

**决策者观测（权威）**：本 run 某次 val 期间，Windows 任务管理器「共享 GPU 内存」最高 **5.6 GB**；
**训练没有崩溃**，val 只是比预期慢。

**本节实测（同一 run `20260928-144342`，两次 val 的输入构成逐字段相同）**：

| val 点 | `val_windows` | `val_lines` | `val_events` | `val_time_s` |
|---|---|---|---|---|
| step 1000 | 128 | 5081 | 498 | **100.52 s** |
| step 2000 | 128 | 5081 | 498 | **167.58 s** |
| step 3000 | 128 | 5081 | 498 | **120.81 s** |

⇒ 三次 val 的**计算量与被读窗口完全相同**（连 `val_empty_share = 0.578` 都一样），墙钟却在
**100.52 / 167.58 / 120.81 s** 之间摆动（幅度约 ±33%）。这**不能**由数据或算力解释（它们逐位固定），
只能来自**内存路径**（驱动换页 / 主机内存争用 / GPU 时钟回退三者之一或叠加）——与「val 跨墙、
权重与激活被换到共享显存」一致。⚠️ **这个摆动本身就是证据**：若是纯算力，三个数应当几乎相等。

**为什么先撞上墙的是 val 而不是训练**：训练是 `batch_size=1`（单样本的 K 就是显存上限，§9-35/§9-57），
而 val 按网格桶成组、每批 `val_batch` 个**同桶（同 K 量级）**窗口（§9-53）⇒ **一个 val 批 ≈ 至多
`val_batch` 个训练单步**。`val_batch: 8 → 4` 只把倍数减半，**没有把最大那几个桶压到墙以下**
（最大桶 K=128，正是 §9-57 事故里 K≈128 的单样本口径）⇒ val 期间跨墙是**结构性**的，不是偶发。

**就地更正 §9-57 的绝对化表述：跨墙 ≠ 必然卡死。**
- step 951 的跨墙（共享 13.2 GB、功耗 102→**31 W 且不回来**）= **卡死**；
- val 的跨墙（共享 5.6 GB、功耗在 **30-65 W 间摆动且会回来**）= **只变慢**。

两者都在 8 GB 卡上、都跨墙，**仅凭共享显存量区分不了**；能区分的是**是否恢复**。

**现场佐证（2026-09-28 23:01，val@3000 进行中）**：`nvidia-smi` 读到 `memory.used = 7824 / 8151 MiB`、
`util = 100%`、`power.draw = 30.62 W`（9 s 后同状态 **64.94 W**），同时 jsonl 停在 step 2950
已 **≥108 s**（val 期间不写训练行）⇒ **val 正在跨墙**。主机侧 `commit charge = 61.26 / 63.23 GiB（97%）`、
可用物理内存 3.53 GB ⇒ 被换出的共享显存与 worker / 窗口缓存**争同一块系统内存**（与 §9-54 的 95% 同源，且更紧）。

**仪器缺口（上一轮判错的直接原因）**：`sys/peak_vram_gib` 是**训练步在 val 之前**记的
`torch.cuda.max_memory_allocated()`（全局高水位，见 `train_loop.py` 的 row 组装顺序），
**val 段没有任何独立显存遥测** ⇒ 跨墙在 jsonl / TB 里**完全不可见**，只能靠决策者的任务管理器读数。
**必须补**：`val_peak_vram_gib` + val 起止日志行。

**对处置的影响（建议，本轮未改代码）**

1. **val 3× miss 的根因是内存，不是算力** ⇒ `val_every` 1000→4000 只压**频率**、不减**单次**成本；
   更直接的是**按显存预算拆大 K 桶**（把 `val_batch` 从「固定条数」改成「受 `max K` / `ΣK²` 预算约束」）。
   它**不改变** val 的 128 窗集合与指标定义，但会改归约顺序 ⇒ 数值可能末位不同（同 §9-53 的成组）。
   **须裁定**（属 val 组批口径 + 代价取舍）。
   ⚠️ 与「跳过 K > 预算的窗口」**不同**——**后者改变 val 分布，本条不改**。
2. **停机判据在 val 段会误报**：`memory.used > ~7.4 GiB` 与 `power.draw < 45 W` 在 val 期间**本来就成立**
   （本轮现场：7824 MiB / 30.6 W）⇒ 人工盯盘必须先确认「当前是否在 val 段」，否则会掐掉健康运行。
   **这就是为什么不要自动看门狗**（§9-57 事故 2 的教训再确认一次）。
3. **长跑预算**：按 167.58 s/次与 `val_every=1000`，**一次 val 的墙钟已超过 1000 步训练本身**
   （≈97-100 s）⇒ 下一个长跑必须**同时**调频率与单次成本，只调一个不够。

**存疑（我们不知道什么）**

- 167.58 − 100.52 = **67 s 的增量**里，驱动换页 / 主机争用 / GPU 时钟回退**各占多少，未分离**（缺 val 侧遥测）；
- 决策者读到的 5.6 GB 发生在**哪一次 val、哪个桶，未记录**，无法与 100.52 / 167.58 一一对应；
- 最大桶（K=128）在一次 val 里被前向几次、每批**实际**批大小与 K，**未落盘**（`val_batch` 是上限，桶不够时更小）；
- val 段的停机阈值**没有定义**（只知 val 会跨墙且会恢复，恢复时间 100-170 s 也是实测值，不是保证）。

### §9-60 第十三轮⑦：**val@3000 判读红线被破**（`val/ratio = 1.0033`）——该 run 扩规模结论作废（2026-09-28）

权威记录：`runs/phigros_masked/20260928-144342/logs/loss_history.jsonl`（step 1000/2000/3000）。

| val 点 | `val_nll` | 常数基线 | **`val_ratio`** | `val_nll_shuffled_delta` | `val_pred_over_true` | 判定 |
|---|---|---|---|---|---|---|
| 1000 | 0.9550 | 1.2697 | 0.7521 | +0.7351 | 0.345 | PASS |
| 2000 | 0.9259 | 1.2697 | 0.7292 | +0.7686 | 0.539 | PASS |
| 3000 | **1.2740** | 1.2697 | **1.0033** | +0.6228 | **0.0816** | **FAIL（红线破）** |

- `val/ratio = 1.0033 >= 1` ⇒ 按 **RFC-0037 R1**：**该 run 的扩规模结论作废**。
  README 的既定处置是「红线破 ⇒ 停并开新 RFC，**不得**继续扩规模」——**处置待决策者裁定**（训练仍在推进）。
- `val/nll_shuffled_delta = +0.6228 > 0` 仍成立 ⇒「输入对目标有信息」这条**没有被推翻**；
  被推翻的是「**模型优于常数场**」。
- **同一现象在训练侧可见**（不是 val 读数的孤例）：非空窗 per-event 训练损失 ~7-8（step 200-2000）
  → **15.69（step 2800）**，`grad_norm` 在 step 2000 达 **157.7**（`clip_active` 长期 100%）。
- **签名 = 强度场塌缩**：`val_integral` 268.4 → **40.6**、`val_pred_over_true` 0.539 → **0.0816**、
  `val_nll_nonempty` 1.785 → **2.456** —— 模型把 `λ` 整体压小（空窗积分项奖励它这么做，而 43.8%
  的步天然是空窗），代价是非空窗的事件拟合崩塌。
- 条件干预同步变差：`cond_audio_zero_delta = −0.0619`（**变负**：把音频置零反而降低损失）、
  `cond_audio_perm_delta = −0.0030`（≈0）、`cond_track_zero_delta = +0.1304`。
- `best.pt` 仍停在 **step 2000**（`val/ratio = 0.7292`）——step 3000 更差，未覆盖它
  ⇒ **线上可用的最好权重是 step 2000**。

**待裁定（不自行处置）**

1. **是否立即停这次 run**（既定规则是「红线破 ⇒ 停并开新 RFC」；训练当前在 step 3350+ 继续推进）。
2. **根因方向**（**未查证，不要先下结论**）：① 空窗积分项主导（43.8% 空窗 + `E_total` 归一不区分空/非空）
   ⇒ 梯度把 λ 拉向 0；② `clip_active` 近 100% 使有效更新近似「每步归一化」，可能放大振荡；
   ③ `reweight="hidden"` 的 `1/r` 重标定在 `r` 小时方差大。
3. **与 §9-59 的关系要分开记**：**跨墙不是数值变差的原因**（val 读数是确定性的，训练损失同向变坏）。
   两件事（成本异常 / 指标退化）各自成立，不要互相归因。

**存疑**

- 只有 3 个 val 点，**分不清单调退化还是振荡**（step 3200 的非空窗损失 10.94 又低于 2800 的 15.69）；
- `val/ratio` 破 1 是**单点**事件还是已经越过后不再回来，**未知**（下一个 val 在 step 4000）；
- 「空窗积分项主导」这条假设**没有任何消融**（要为它开 RFC 才能动手改损失）。

### §9-61 第十三轮⑧：**全量大训练首跑（20000 步）完成——无卡死，但 20k 步零学习进展**（2026-09-28/29，exit 0）

权威记录：run `runs/phigros_masked/20260928-144342`（`logs/loss_history.jsonl` 每步一行、`metrics.json`、
`gates.txt`、`checkpoints/`）。**这是目录里第一个跑到 `max_steps` 并正常退出的全量 run**。

**运行事实**

| 项 | 实测 |
|---|---|
| 墙钟 / 退出码 | 22:43:39 → 00:21:00 = **5841 s（97.3 min）**，**exit 0**，无卡死、无 NaN |
| 门禁（3.5 min） | **G1/G3/G4 全绿**：G1 4640.195 → 2.560（需 ≤464.02）；G3 模型 −0.734 vs 常数 10.042（需 ≤9.038）；G4 123.3 帧 vs 实测 124（容差 2） |
| 步时 | Σ `step_time_s` = **3252.9 s**；中位 **0.098 s**、均值 **0.163 s**、p95 0.551、max 11.20（step 2，首步） |
| val | 20 个点、Σ `val_time_s` = **2234.9 s**（单次 73.0–190.3 s）⇒ **占墙钟 38.3%**，是步时总和的 **0.687×** |
| 其余 | 门禁 + 建/读索引 + 20 次 checkpoint ≈ 350 s（6%） |
| 覆盖 | `charts_seen` **6614/6614（100%）**、`windows_seen` 20000/634952（**3.15%**，`epoch` 0.031）⇒ RFC-0034 S4 的全局发牌在长跑里保持 100% 谱面覆盖 |
| 权重 | `best.pt` @ **step 2000**；保留 step-16000/18000/20000 + best（旋转按 `save_every=2000` 工作正常） |

**① 训练侧：step ~1000 之后完全没有进展**（本次最重要的实测）

按 2000 步分块的**非空窗 per-event 损失**（`loss_sum_raw / batch_events`，跨步同口径）：

| 块 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 中位 | 9.52 | 10.24 | 10.21 | 10.61 | 9.88 | 9.51 | 10.05 | 10.21 | 10.47 | 10.24 |
| 均值 | 10.54 | 10.87 | 11.16 | 11.52 | 10.87 | 10.55 | 11.30 | 11.33 | 11.57 | 11.66 |

⇒ 中位**无趋势**、均值 19000 步只涨 11%（在块间噪声内）。**另外 19000 步买到的是 0**：
损失在 step ~1000 就触底，之后是**平台 + 振荡**。这同时回答了 §9-60 存疑②：step 2800 的 15.69
**不是退化趋势**，是同一条振荡线上的一个样本（块中位/均值全程 9.5–11.7）。
唯一仍在动的量是**空窗积分项**（`loss_empty` 中位 1.20e-3 → 3.7e-4，前 8000 步降 3×，之后平）——
即模型学会了「空窗不要乱放强度」，但**没有学会把非空窗的 λ 变尖**。

**② val 侧：单点破线 + 全程平（不是单调退化）**

| step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| `val_ratio` | 0.7521 | 0.7292 | **1.0033** | 0.7649 | 0.8258 | 0.8055 | 0.7957 | 0.8592 | 0.8635 | 0.7402 |
| `shuf_delta` | +0.7351 | +0.7686 | +0.6228 | +0.6535 | +0.6506 | +0.6148 | +0.6424 | +0.6516 | +0.4827 | +0.7007 |
| `pred/true` | 0.345 | 0.539 | **0.082** | 1.249 | 0.933 | 0.515 | 0.521 | 0.319 | 0.775 | 0.669 |

| step | 11000 | 12000 | 13000 | 14000 | 15000 | 16000 | 17000 | 18000 | 19000 | 20000 |
|---|---|---|---|---|---|---|---|---|---|---|
| `val_ratio` | 0.7576 | 0.8202 | 0.8326 | 0.8405 | 0.9303 | 0.7988 | 0.8235 | 0.9698 | 0.8812 | 0.8867 |
| `shuf_delta` | +0.7344 | +0.7430 | +0.7057 | +0.6865 | +0.8255 | +0.7674 | +0.7315 | +0.7097 | +0.6412 | +0.6846 |
| `pred/true` | 0.999 | 0.959 | 0.479 | 0.458 | 0.363 | 0.567 | 0.379 | 0.282 | 1.221 | 0.424 |

最小二乘趋势（`n=20`，残差自由度 18）：

| 量 | 首 | 末 | 均值 ± sd | 斜率/1000 步 | t | 残差 sd |
|---|---|---|---|---|---|---|
| `val_ratio` | 0.7521 | 0.8867 | 0.834 ± 0.072 | +0.00473 ± 0.00273 | **+1.73** | 0.0667 |
| `val_nll` | 0.9550 | 1.1258 | 1.059 ± 0.092 | +0.00601 ± 0.00346 | **+1.73** | 0.0847 |
| `shuf_delta` | +0.7351 | +0.6846 | +0.688 ± 0.071 | +0.00283 ± 0.00284 | +1.00 | 0.0695 |
| `pred/true` | 0.345 | 0.424 | 0.604 ± 0.312 | +0.00047 ± 0.01274 | **+0.04** | 0.3118 |

- **判读红线（RFC-0037 R1）**：20 个点里 **`val_ratio >= 1` 恰好 1 个（step 3000）** ⇒
  **该 run 的扩规模结论作废**（按既定规则执行，非本次新裁定）；**`val/nll_shuffled_delta <= 0` 0 个**
  ⇒「输入对目标有信息」在 20/20 个点上成立，未被推翻。
- **诚实的限度（新）**：`val_ratio` 的整体漂移（+0.13）**t = 1.73，未达 5% 显著**，而它自己的
  残差 sd 就是 **0.067**——**漂移量与量具噪声同量级**。§9-60 存疑①「分不清单调退化还是振荡」
  现在有答案：**是振荡**（step 3000 之后 17 个点全部 < 1，最大 0.970）。但**「没有变好」是确定的**：
  最优值停在 step 2000，之后 18000 步一个点都没超过它。
- `pred/true` 的**振荡幅度**（0.082–1.249，sd 0.31、无趋势）才是真正的现象：模型每 1000 步把
  总强度甩来甩去而损失几乎不动 ⇒ **损失曲面在「强度尺度」方向很平**，这与 ① 的「平台」是同一件事。

**③ 新证据：音频条件对损失几乎没有贡献**（§9-46 ③ 三元组的首个跨 20 点读数）

`cond_*` 三臂是**同批、同权重**的 no_grad 前向差分（Δ = NLL(干预) − NLL(基线)，>0 = 该条件被用上）：

| 臂 | 20 点的范围 | 相对 `val_nll`（均值 1.06） |
|---|---|---|
| `cond_audio_perm_delta`（音频**帧轴**置换） | −0.0030 … **+0.0059** | **≤ 0.6%** |
| `cond_audio_zero_delta`（音频置零） | −0.0619 … +0.0590 | ≤ 6% |
| `cond_track_zero_delta`（事件轨置零） | +0.0486 … +0.3186 | 4.6% … 30% |

⇒ **在遮盖测度下，模型几乎不看音频**（置换时间轴——分布内、只破坏对齐——只值 0.6%），
看事件轨多一些（最高 30%）。**方法学前提已核查**：`audio_zero` 的 Δ 不是 0 而是有符号的
~0.03 ⇒ 缓存里的音频**不是退化常量**（常量会给逐位 Δ=0）。⚠️ **不要把它读成「音频没用」**：
① 可见场（50% 未遮盖计数）本身携带极大信息，掩盖了音频的边际价值；② 这个读数只说明
**当前训练目标没有奖励模型使用音频**，而生成时没有可见场 ⇒ 这正是「平台」的候选根因。**未消融，勿下结论。**

**④ 四个量具/工程缺陷（本次新登记）**

1. **val 检验力不足（新的顶格未决项）**：128 窗 / **5081 线 / 498 事件**，其中 **57.8% 的窗口是空窗**
   而 `val_nll_empty ≡ 0.0000`（空窗无事件 ⇒ 分子分母都只有积分项）⇒ **主判据实际只由 ~42% 的窗口
   承载**；`val_ratio` 的自噪声 sd ≈ **0.067**，与被判读的漂移（0.13）同量级。**「红线判据」目前是
   低检验力量具**：单点破 1 既可真是退化、也可只是 1.5σ 抖动（step 3000 的 `pred/true`=0.082 说明
   那次确实是模型侧事件，但**下一个这样的点会不会被误判**取决于量具，而不是规则）。
2. **val 单次成本高 8×**：val 每窗 0.785 s / 128 窗，而**训练**每步（含反传）0.098 s ⇒ 5 个前向臂
   （模型 + 置换 + 3 条件）是主因，其次才是跨墙。**20 个 val 点里没有任何一个改变了处置**
   ⇒ 频率与臂数都有可压空间（建议见 RFC-0038）。
3. **val 之后的第一步稳定慢 6–70×**（实测 20/20：step 6001 = **6.84 s**、19001 = 3.28、17001 = 1.75，
   其余 0.56–1.25，而全程中位 0.098）⇒ val 的显存/页缓存扰动会把紧随其后的一步拖慢；
   这是**成本**不是事故（1 步内恢复，不触发停机判据）。
4. `metrics.json` 的 `best_loss = 8.65e-6` 是**误导性统计**：它是全程**单步 per-event 损失的最小值**，
   落在某个空窗步上（同类问题见 §9-47 B2）。训练摘要日志里的 "best 0.000009" 同源
   ⇒ **不要用它选权重**（`best.pt` 按 `val/ratio` 选，那条路径是对的）。

**⑤ 这份实测对既有结论的更正**

- §9-60「非空窗 per-event 训练损失 ~7-8 → 15.69（step 2800）」的**趋势读法作废**：那是振荡，不是退化；
- §9-60「val/ratio 破 1 是单点还是越过不回来」⇒ **单点**（后续 17 点全部 < 1）；
- 事故侧结论**保持不变**：`vram_hygiene_gib` 生效（`vram_reserved_gib` 中位 1.23 GiB、max 1.53 GiB），
  20000 步**零卡死**；驱动侧贴顶仍只发生在 val 段（§9-59）；
- `clip_active` = **非空步 100%**（20 个块全部），与 RFC-0037 §6-1 一致；
- `data_share` 中位 **2.9%**（均值 3.6%）⇒ 窗口缓存把数据侧彻底移出关键路径（对照：原路径 46%）。

**结论（一句话）**：**管线跑通了（20000 步无卡死、100% 谱面覆盖、缓存 85×），但这一版的目标/优化器
在 step ~1000 就饱和——之后 19000 步既没让训练损失下降，也没让 val 变好；音频条件的边际价值 ≈ 0。**
按 RFC-0037 R1 该 run 的扩规模结论**作废**；处置与下一步**待决策者裁定**（提案见
[RFC-0038](../decisions/RFC-0038-training-saturation-and-budget.md)）。

**存疑（我们不知道什么）**

- **平台是「学完了」还是「学不动」**：可见场已携带大部分信息 ⇒ 平台可能是**不可约熵**，也可能是**优化器
  走不动**。**两者不可由现有数据区分**（缺：关掉可见场的对照、固定批上的过拟合速度、`lr` 扫描）；
- **音频为何不被用**：是 LoRA/适配器没更新、是条件注入路径弱、还是目标函数不需要它——**未分离**；
- **`cond_*` 三臂的 20 个点全部耦合同一个 val 集**（498 事件），**没有独立样本** ⇒ 表里的「范围」
  不等于「不确定度」；
- **val 段跨墙对判据的影响未度量**：val 读数在 `no_grad` 且 bf16 下确定性，但换页是否引入
  **非确定性**（如 CUDA kernel 选择变化）**未查**；
- 单次 val 73–190 s 的 **2.6× 摆动**仍未归因（§9-59 存疑延续）。

### §9-62 第十四轮①：**根因排查——模型停在「mask + 位置」条件边际解；音频无法与 τ 轴对齐**（2026-09-29）

**背景**：§9-61 把 20000 步的平台如实记为「零进展」，留下「学完了 vs 学不动」的存疑。本轮用六个
**一次性探针**（`runs/_diag_*.py`，不入库）把它拆开。**决策者裁定：在举措被证明有效之前不得开 RFC**
——本节只记录事实与已启动的实验。

**① 度量有巨大空间 ⇒ 平台不是「学完了」**

同一批 128 个 val 窗口、**同一遮盖测度**下的「不用模型」基线梯子：

| 基线 | `val_nll` | `val_ratio` |
|---|---|---|
| `constant(c*)` | 1.2697 | 1.0000 |
| **step-20000 模型** | **1.1258** | **0.8867** |
| `token_oracle_hidden`（作弊：知道哪些 (k,τ) 有被遮盖事件，token 内均匀） | 0.7125 | 0.5611 |
| `token_oracle_all` | 0.6877 | 0.5416 |
| `oracle_full`（作弊：知道被遮盖事件的**确切格子**） | −0.1324 | −0.1043 |

⇒ 模型只吃到「常数 → 作弊上界」的约 **10%**；即便对「只知道 token」的弱作弊上界也还差 **0.41 nats/line**。
**「平台 = 不可约熵」被否证**（格分辨率确有 `log 1280 = 7.16` nats/事件的软下界，但模型连
「该在哪个 token 放质量」这一层都没做满）。

**② 输入侧的真实信息量（更正 §9-61③ 的前提）**

- `mask_leak_tokens = 0.00079`：被遮盖 token 有 **306 512** 个（占全部 1 207 296 的 **25%**），
  其中含事件的只有 **0.079%** ⇒ 遮盖通道**几乎不泄漏**（信息中性目标 0.0334%，富集仅 **2.4×**）；
  `uniform@mask=1` 的 nll = **1.3423（比常数还差）** ⇒ 它不构成捷径。
- **可见场在 token 级极稀疏**：val 每窗口约 **3.9 事件**、被遮盖 65% ⇒ 每窗口只有 **≈1.4 个 token**
  携带可见证据（全场 9 432 个 token）。因此「模型不看可见场」**不是缺陷**——几乎没有可看的。

**③ 模型对输入的依赖：25% 看 mask、≈0% 看可见场**（场级实测，`_diag_lamsens.py`，不是损失级）

| 变体 | 随机初始化 `‖Δλ‖/‖λ‖` | step-20000 |
|---|---|---|
| blank（可见场清零） | 0.00021 | 0.00010 |
| x-roll / τ-roll（可见场打乱） | 0.00022 / 0.00022 | 0.00141 / 0.00567 |
| mask-off（遮盖通道清零） | 0.0223 | **0.25000** |

⇒ 训练后的模型**确实是 mask 驱动的**（λ 变 25%；`max/mean` 对比度 37 923 → 18 733），
对可见场几乎无反应；随机初始化也已有 2e-4 级响应 ⇒ **通路没坏**。

**④ 质量去向**：`mass@hidden-event cells = 0.0002`（99.99% 的 ∫λ 落在空格）⇒ 模型在「被遮盖的 τ 切片」内
**没把质量放到事件格**，空间分布近似「token 内均匀」甚至更差。

**⑤ 根因：τ 轴与音频轴不在同一时间基，而模型拿不到换算因子**

- 窗口网格把 τ 轴重标定为**窗口等效 BPM**（`bpm_eff = 60/J(τ_start)`，`data/dataset.py` 的 `_window_grid`）
  ⇒ 「1 个 τ 格 = 多少秒」**逐窗口不同**（实测窗口 `bpm_points = [(0, 150.0)]`、192 格 = 1.596 s）；
- 音频帧的位置编码以**秒**为基准（`encode_audio`），场 token 的位置编码用 **τ 格下标**；
- **模型从未拿到 J（BPM）**：`beatmorph/generation/model.py` 全文无任何 bpm/tempo 输入。
⇒ 音频与场 token 相差一个**逐窗口变化的未知因子**，cross-attention 无法对齐——这正解释
`cond_audio_perm_delta ≈ 0`（0.6%）与「音频时间对齐几乎没被用」。**plan 04 §5 早已把它登记为残留问题**
（「位置编码与条件注入在 τ 轴上的具体形式……与音频帧轴对齐的插值方式」），本轮首次给出**可观测后果**。

**⑥ 已启动的处置（实验臂，未开 RFC）**

1. **臂 A：`model.seconds_position=true`**（新增开关，默认 off）——给场 token **额外**加一条以秒为基准的
   位置编码，使两条轴落在同一时间基。换算仍只在 `field/` 实现（新增 `FieldGrid.tau_seconds()`；
   生成侧只调它，红线 7 的源码门禁 `tests/unit/generation/test_source_guards.py` 已把首次写法拦下并已改正）。
   守卫：`tests/unit/generation` + `tests/unit/field` + `test_config_schema.py` ⇒ **350 passed**。
2. **臂 B：`optim.lr=1e-3`**——检验「停在浅盆地、步长不足」这一竞争假设。
3. 两臂各 **8000 步**、`run.experiment=probe_*`，**同 seed / 同数据顺序**（计划层是纯函数）⇒ 与基线
   20000 步 run 的前 8000 步**逐块可比**。判据：`loss_nonempty_per_event` 的分块中位（基线
   blk1-4 = 9.52 / 10.24 / 10.21 / 10.61）与 `val_ratio`@8000（基线 0.8592），以及
   **`cond_audio_perm_delta` 是否从 ≈0 抬起**（对齐修复的直接指征）。

**⑦ 已启动实验臂的结果（各 8000 步，2026-09-29）——两个假设均被否证**

| run | blk1 | blk2 | blk3 | blk4 | `val_ratio`@8000 | `cond_audio_perm`@8000 |
|---|---|---|---|---|---|---|
| 基线（lr=3e-4） | 9.515 | 10.241 | 10.206 | 10.610 | 0.8592 | +0.0043 |
| **臂 A：`seconds_position=true`** | 9.622 | 10.158 | 10.300 | 10.666 | 0.9343 | **−0.0017** |
| **臂 B：`lr=1e-3`** | 13.671 | 13.974 | — | — | — | — |

- **臂 A 否证「音画未对齐是平台主因」**：分块中位与基线**逐块同水平**、val 略差（0.9343 vs 0.8592），
  且 `cond_audio_perm_delta` **仍在 0 附近**（−0.0017）⇒ 把两条轴放到同一时间基，并没有让模型开始用音频。
  （实验臂记录：`runs/probe_secpos/20260928-183412`。）
- **臂 B 否证「步长不足」**：lr=1e-3 前两块中位 **13.7 / 14.0 ≫ 基线 9.5 / 10.2**，并在 **step 2794
  直接 λ=NaN** 崩掉（`assert_lambda_valid` 抛 `AssertionError: lambda 出现 NaN`）
  ⇒ **3e-4 已在稳定边界附近，不是"太小"**。（记录：`runs/probe_lr1e3/20260928-190204`。）

**⑧ per-event 损失分布——不是「少数灾难」，是「普遍平庸」（`_diag_eventloss.py`）**

324 个被监督事件：p00 **4.34** / p25 6.64 / **p50 10.08** / p75 13.30 / p99 22.71 / max 26.53；
**损失最大的 10% 事件只占 18.7% 的总损失** ⇒ 没有离群主导。事件处 λ 跨五个数量级
（p10 **4.1e-8** / p50 **4.2e-5** / p90 **4.3e-3**）⇒ 约 10% 的事件模型**完全没命中**，其余也只是勉强。
连**最好的**事件（4.34）离作弊上界（≈ −1.35 nats/事件）都还有 ~5.7 nats ⇒ **没有任何一个事件是"真的会做"的**。

**⑨ 决定性探测：信息在音频里，但模型一点没用上（`_diag_audio_probe.py`，CPU）**

问题：平台是「输入里没有这个信息」还是「模型没学会用」？做法：按 **τ→秒** 对齐取出 MERT 帧
（并附**帧差分**作为粗糙 onset 特征），用**线性 logistic** 预测「这个 τ 切片有没有音符（任意线）」，
train/val 分开口径（train 384 窗拟合、val 128 窗检验）：

| 探针 | train AUC | **val AUC** |
|---|---|---|
| 音频帧 + 帧差分（BPM 对齐） | 0.9255（in-sample，偏乐观） | **0.6398** |
| 仅位置（秒） | 0.5201 | 0.5420 |
| 随机分数 | — | 0.5077 |

⇒ **音频确实携带「哪里该有音符」的信息**（val AUC 0.64，显著高于仅位置 0.54），而模型的
`cond_audio_perm_delta ≈ 0`（0.6%，且**两个实验臂都没改变它**）说明它**完全没用上这条路**。
⇒ 平台**不是**「输入里没有信息」，而是**条件通路没学出来**（cross-attention 那条路：
§9-61③ 与 §9-62⑤ 都把矛头指向它）。

**⑩ 由此启动的臂 C（未开 RFC）**：`model.audio_align=true`——**绕开 cross-attention**，
把「该 τ 切片对应时刻的音频帧 + 帧差分」投影后**直接加到 token 嵌入**上
（τ→秒仍只走 `FieldGrid.tau_seconds()`；配置门控、默认 off）。
判据：① `cond_audio_perm_delta` 是否从 ≈0 抬起（对齐/通路是否真的接上）；
② 分块中位是否低于基线（9.515 / 10.241 / 10.206 / 10.610）；③ `val_ratio`@8000 vs 基线 0.8592。
记录：`runs/probe_audioalign/`。

**⑪ 把模型自己的输出在 token 内摊平 ⇒ 损失**变好**（`_diag_flat.py`，step-20000，128 val 窗）**

| 变体 | `val_nll` | `val_ratio` |
|---|---|---|
| 模型原样 | 1.1258 | 0.8867 |
| **模型自己的 τ/线分布 + token 内均匀** | **0.9334** | **0.7351** |
| **只把 x 轴摊平（保留 side/type）** | **0.8503** | **0.6697** |
| 同总量、全域均匀 | 1.2983 | 1.0225 |

⇒ 模型的 `(x,s,c)` 空间分布**比均匀还差**，代价 **0.19–0.28 nats/line**——比音频（0.003）、
对齐（0）、LR 等本轮所有已测效应**大两个数量级**。⇒ 它的 **side/type** 边际有用（只摊平 x 更好），
坏的是 **x** 那一维。

**⑫ 边缘分布对照（`_diag_marginals.py`，在被遮盖集合上）**

side corr **+1.000**；type **+0.980**（但模型 Tap 0.527 vs 真实 0.673、HoldEnd 0.155 vs 0.043 ⇒
**过估 HoldEnd**）；x **+0.583**（镜像对照 +0.311 ⇒ **不是坐标错配**）但**峰度是真实的 2.5×**
（与均匀的最大偏差 0.229 vs 0.091）；τ 的跨窗口聚合相关性 +0.077——⚠️ **该读数作废**，理由见 ⑬。

**⑬ τ 位置**确实被奖励**（`_diag_tau_reward.py`）**：把模型的 τ 轴随机置换 ⇒ `val_nll` **1.1258 → 3.2229**
（ratio 0.887 → **2.538**）；`roll(−24)` ⇒ 3.2905（2.592）。⇒ **τ 位置是损失最敏感的维度之一**，
⑫ 的 τ 聚合相关性（同一批事件在不同窗口的绝对 τ 分布被平均掉）是**量具假象**，就地作废。

**⑭ 由此启动的臂 D（未开 RFC）**：新增 `optim.cell_entropy_weight`（默认 0 = 关闭）——空间 softmax 的
**熵正则**，惩罚项与事件项**同结构**：`w · (1/r) · Σ_被遮盖 n · H(p_token) / max(E_total,1)`
（H = token 内 (x,s,c) 分布的熵）⇒ `w` 与 `−log λ` 同量纲。**剂量-反应两档 w = 1.0 / 0.25**（各 8000 步）。
判据：分块中位 + `val_ratio`@8000（基线 0.8592）。**若它把 val_ratio 推向 ⑪ 的 0.735–0.670 区间，
则「空间分布过尖且错」这条根因成立**；若纹丝不动，则该缺陷与优化器无关，须回到输出头参数化（`cell_skip` 的常数输入）。

**⑮ 臂 D 结果（熵正则 w = 1.0 / 0.25，各 8000 步）——也失败，但方式有信息量**

| run | blk1 | blk2 | blk3 | blk4 | `val_ratio`@8000 |
|---|---|---|---|---|---|
| 基线 w=0 | 9.515 | 10.241 | 10.206 | 10.610 | **0.8592** |
| w=0.25 | 10.012 | 10.592 | 10.643 | 11.093 | 0.8970 |
| w=1.0 | 11.628 | 12.021 | 12.180 | 12.652 | 0.9042 |

训练损失随 w **近似线性上升**（≈ `w·H`）⇒ **模型只是把惩罚付掉了、熵并没有变**；而事件项对「尖」的
偏好强于熵项对「平」的偏好。⇒ **「事后摊平能获益」≠「训练时加熵正则就能拿到」**：⑪ 的收益是
**事后校正**性质，不是优化轨迹上的可达点。（记录：`runs/probe_entropy/20260928-195345`（w=1.0）、
`.../20260928-202344`（w=0.25）。）

**⑯ 空间分布的实测形状（`_diag_anti.py`）**：token 内空间分布平均熵 **1.88 nats**（上限 log1280 = **7.155**）
⇒ **近似 one-hot**；被遮盖事件格的 λ 高于同 token 均值的比例 **0.3673（< 0.5）**；
`λ(event)/token-均值` 的 p10 / p50 / p90 = **0.0025 / 0.215 / 26.6** ⇒ **多数事件落在模型排名偏低的位置**。
x 轴位移扫描最优 **k = 0**（corr +0.583）⇒ **不是坐标错位**（⑫ 的镜像对照亦同向）。

**⑰ 决定性否证：这不是过拟合 ⇒「数据量不够」对这条缺陷不成立（`_diag_overfit.py`）**

| split | 命中率 | 事件数 | token 熵 |
|---|---|---|---|
| train（**已见过**的前 128 窗） | **0.3729** | 421 | 2.23 |
| val（留出谱面） | **0.3673** | 324 | 1.88 |

⇒ **train 与 val 的命中率相同**：模型在**自己训练过的窗口上一样差** ⇒ **加数据修不好它**。
（决策者提出的「模型没跑完整个数据集」这一假设，**对这条缺陷被实测否证**；
它对其余部分——例如 τ/线定位是否还能随数据量改善——**尚未被检验**。）

**⑱ 由此启动的臂 E（未开 RFC）**：`model.head_skip=false`（新增开关，默认 true = 原行为）。
对**被遮盖 token**，直连 skip 的输入恒为常数 `[0…0, 1…1]` ⇒ 它**只能提供与 token 无关的空间先验**，
而实测该先验与真实事件格不符（⑯）。关掉它 = 逼 `p(x,s,c)` 只能来自 trunk。
判据：分块中位 + `val_ratio`@8000（基线 0.8592）+ 命中率（⑯ 的 0.367）。**若 G1 当场失败（门禁 fail-closed），
那本身就是「skip 是 G1 依据」的复现**（plan 04 §9-17），须另寻出路。

**⑲ 臂 E 结果（`head_skip=false`，8000 步）——本轮唯一有效的改动，且同时修好了空间缺陷**

| 指标 | 基线（skip=on） | **臂 E（skip=off）** |
|---|---|---|
| 分块中位 blk1-4 | 9.515 / 10.241 / 10.206 / 10.610 | **8.299 / 9.272 / 8.691 / 9.210** |
| `val_ratio`@8000 | 0.8592 | 0.8546 |
| `val_nll` | 1.0909 | 1.0851 |
| `val_pred_over_true` | 0.3193 | **0.5050** |
| `cond_track_zero_delta` | 0.0486 | **0.3431** |
| `cond_audio_perm_delta` | +0.0043 | −0.0057 |
| **命中率**（事件格 λ > token 均值） | **0.3673** | **0.7500** |
| λ(event)/token-均值 p10/p50/p90 | 0.0025 / 0.215 / 26.6 | **0.162 / 7.124 / 39.6** |
| token 内空间熵 | 1.88 nats | **3.10 nats** |
| x 边缘 corr | +0.583 | **+0.801** |

**机制链（完整）**：输出头的**直连 skip**（`cell_skip`/`cum_skip`，plan 04 §9-17 为**合成**任务引入）
在真实任务上是**有害**的——对**被遮盖 token**，其输入恒为常数 `[0…0, 1…1]`，于是
① 它**只能给出与 token 无关的空间先验**（实测与真实事件格**反相关**：命中率 0.367）；
② 这条「容易的路」让 **trunk 学不动**（权位移：trunk 5–20%、head 1100–2900%）；
③ 因此**一切条件通路形同虚设**——音频 0.6%、对齐修复无效、直连音频注入无效、LR 无效。
**这一条解释了本轮此前全部零结果**（⑮⑯⑰ 与臂 A/B/C/D）。

**顺带更正 plan 04 §9-17**：G1 在 `skip=off` 下**仍然 PASS**（4646.7 → 2.46）⇒「没有 skip 就会停在
只学边际分布的盆地」在**真实任务**上不成立（至少「300 步单批过拟合」这一层不成立）。

**仍未解决**：`val_ratio`@8000 **没有跟着训练损失改善**（0.8546 vs 0.8592）——训练侧 −1.2 nats/事件而
val 侧纹丝不动 ⇒ **泛化尚未跟上**（或该 run 的 val 在 8000 步时仍处在「trunk 刚被解放」的起点）。
臂 F（`head_skip=false`、**24000 步**、val 每 4000 步）正在检验「val 随后是否跟上」。

**⑳ 臂 F 结果（`head_skip=false`、**24000 步**、val 每 4000）——训练侧持续下降，val 无显著改善**

| step | 4000 | 8000 | 12000 | 16000 | 20000 | 24000 |
|---|---|---|---|---|---|---|
| 基线（20k，skip=on） | 0.7649 | 0.8592 | 0.8202 | 0.7988 | 0.8867 | — |
| **臂 F（skip=off）** | 0.8121 | 0.8574 | **0.7892** | **0.7579** | **0.7737** | 0.9272 |

配对差（n=5，4000-20000）：均值 **−0.028**、sd 0.058 ⇒ **t = −1.07，不显著**。
而**训练侧**：非空窗 per-event 中位 **10.04 → 8.85**（前 1000 步 9.81→10.09 **上升** 的基线 vs
臂 E 的 8.85→8.85 **持平**），24000 步末 **7.94**。
⇒ **改动确实修好了"学不动"，但 24000 步内没有把它转成可判读的 val 收益。**

**㉑ 记忆化检验（`_diag_memorize.py`）——设计有缺陷，结论作废，但暴露了更要紧的东西**

| checkpoint | train 前 128 窗 nll | val 128 窗 nll | train ratio | val ratio |
|---|---|---|---|---|
| baseline@20000 | 2.1184 | 1.1258 | **1.1184** | 0.8867 |
| armF@24000 | 1.6740 | 1.1900 | 0.9913 | 0.9372 |

两组的**难度不可比**（见 ㉒）⇒ 该对比不能判"记忆化"（train 组不是随机样本）。但顺带暴露：
**模型在训练窗上 ratio = 1.12（比常数场还差）**，而在 val 窗上 0.887 ⇒ **val 比训练流容易得多**。

**㉒ val 集与训练流的"难度刻度"差 3.5×（`_diag_val_repr.py`，不需要模型）**

| 窗口组 | 常数基线 nll/line | 空窗占比 | 事件/窗（中位） | K 中位 |
|---|---|---|---|---|
| **A) val 的 128 窗** | **1.2697** | **57.8%** | **0.0** | 28 |
| B) train 计划 0–127 槽位 | 1.8942 | 48.4% | 2.5 | 25 |
| C) train 计划 300000.. | **4.4645** | **5.5%** | **12.0** | 25 |

⇒ ① **`val_ratio` 是在一个比训练流稀疏得多的子总体上算的**；
② **训练流本身沿计划位置非平稳**：计划是「按桶（=按 bpm）轮转发牌」，随着轮次推进，
小桶先被抽空，剩下的偏密 ⇒ 事件/窗从 2.5 升到 12.0（**5×**）。
一个「跑得更久」的 run 会在**越来越难**的窗口上训练，而 val 固定在稀疏端。

**㉓ val 选取偏置假设被否证（`_diag_val_bias.py`）**：曾怀疑 `_slots()` 取 `sorted(bucket)` 前 32 个桶
= **最低 BPM 的 32 个桶**；实测**不成立**——`bucket_id` 是「计划里首次出现的顺序」而非 grid_key 排序，
选中的 128 窗 bpm 中位 164 vs 全 split 174（min 69 vs 31.4）⇒ **无实质 BPM 偏置**。

**㉔ 臂 G 结果（`head_skip=false` + `audio_align=true`，24000 步，与臂 F **同 seed/同数据顺序/同 val 排程**）——本轮最好的一档**

| step | 4000 | 8000 | 12000 | 16000 | 20000 | 24000 | 均值 |
|---|---|---|---|---|---|---|---|
| 基线（skip=on，20k） | 0.7649 | 0.8592 | 0.8202 | 0.7988 | 0.8867 | — | 0.826 (n=5) |
| 臂 F（skip=off） | 0.8121 | 0.8574 | 0.7892 | 0.7579 | 0.7737 | 0.9272 | 0.798 (n=5) |
| **臂 G（skip=off + audio_align）** | **0.7058** | **0.7607** | 0.8154 | 0.8014 | **0.6920** | **0.7841** | **0.760 (n=6)** |

配对差（臂 G − 基线，同 step，n=5）：均值 **−0.071**、sd 0.078 ⇒ **t = −2.03**（尚不达 5% 显著，但方向一致）。
**训练侧**：末步 `loss` 基线 ~10（20k 步平台）／臂 F **7.94**／**臂 G 5.58**。

**关键机制读数（臂 G 末次 val）——音频终于接上了，但只作为"内容"而不是"时间"**

| 读数 | 基线 | 臂 G |
|---|---|---|
| `cond_audio_zero_delta`（把音频**置零**） | 0.012 | **4.971** |
| `cond_audio_perm_delta`（把音频**时间轴置换**） | +0.004 | −0.003 |
| `cond_track_zero_delta` | 0.049 | **0.495** |
| `val_pred_over_true` | 0.319 | **0.628** |
| `val_nll_empty` | 1.2e-5 | 0.024 |

⇒ ① 模型现在**极度依赖音频存在**（置零代价 4.97 nats/line，是总 NLL 的 5 倍）——
**说明 `head_skip` 就是此前"音频接不上"的那道闸**（臂 A/C 在 skip=on 下全部无效，见 ⑲）；
② 但它仍然**不依赖音频的时间对齐**（置换 ≈ 0）⇒ 现在用上的是**全局内容/风格**，还不是"哪一拍有音符"；
③ `val_nll_empty` 从 1e-5 升到 0.024 ⇒ 模型不再把空窗的 λ 压到 0（与 §9-61① 的"只学会空窗压强度"相反）。

**㉕ 高功效配对评估（512 窗，139 批，同一批窗口）——`audio_align` 是真正的放大器**

| checkpoint | `val_nll` | **`val_ratio`** |
|---|---|---|
| baseline@20000（skip=on） | 1.6201 | 0.9608 |
| armF@24000（skip=off） | 1.5959 | 0.9465 |
| **armG@24000（skip=off + audio_align）** | **1.3827** | **0.8201** |

配对差（armG − baseline，n=139 批）：**+0.627 nats/批，t = +2.44**（p ≈ 0.016）；
受控比较（**同 24000 步**、只差 `audio_align`）：**armF 0.9465 → armG 0.8201**。
⇒ ① `head_skip=false` 单独把训练损失打到 7.94、val 略降；
② **再叠加 `audio_align=true` 才是真正的跃升**（训练损失 **5.58**、512 窗 val **0.82**）——
而**同一开关在 skip=on 下完全无效**（臂 C，⑲）⇒ **两道闸是串联的：先拆 skip，音频通路才谈得上**。

⚠️ **量具更正**：128 窗 val 给出 0.887（基线）/0.855（臂 F），512 窗给出 **0.961 / 0.947**
⇒ **128 窗那批是系统性更容易的子集**（常数基线 1.27 vs 512 窗上更高），此前的 val 读数**系统性乐观**。

**㉖ 臂 H：把解法放大到 40000 步（`head_skip=false` + `audio_align=true`，`val_windows=512`）——解法成立**

| step | 8000 | 16000 | 24000 | 32000 | 40000 | 均值 |
|---|---|---|---|---|---|---|
| **臂 H `val_ratio`（512 窗）** | **0.7234** | 0.8285 | 0.7813 | 0.8047 | **0.7455** | **0.7767** |
| `val_nll` | 1.2198 | 1.3970 | 1.3173 | 1.3569 | 1.2569 | 1.3096 |

**对照（同一批 512 窗、同一测度）**：基线@20000 = **0.9608**；`head_skip=false` 单独@24000 = 0.9465。
⇒ **`val_ratio` 从 0.961 稳定压到 0.72–0.83（≈ −19%）**，5 个点全部远低于基线，且**在 8000 步就已经达标**
（说明收益来自"能学"而不是"学得久"）。训练侧末步 `loss` = **0.0150**（基线 ~10、臂 F 7.94、臂 G 5.58）。

**本轮结论（可复述给决策者）**

1. **真正的原因**：输出头的**直连 skip**。它不是"调参问题"——它把 trunk 的学习通道掐断了：
   被遮盖 token 的 skip 输入恒为常数 ⇒ 只能给一个**与 token 无关**的空间先验（实测与真实事件格**反相关**，命中率 0.367），
   而这条捷径同时让 trunk 停在初始化附近（权位移 5–20% vs head 1100–2900%）。
2. **真正有效的解法**：`model.head_skip=false` **+** `model.audio_align=true`（**串联**：单独任一个都拿到大部分收益的只有前者的一小部分）。
   三条独立证据线一致：**训练损失 10 → 0.015**；**512 窗 val ratio 0.961 → 0.777**；**机制读数**（空间命中率 0.367 → 0.750、
   音频置零代价 0.012 → 4.97、事件轨置零代价 0.049 → 0.495）。
3. **仍未解决**：音频只当"内容"用（帧轴置换仍 ≈ 0）⇒ 落点的时间定位还没让音频参与；
   以及 val 集合代表性 / 两个开关的默认值（须裁定）。

**㉗ 最终配对验证（512 窗 / 139 批 / 同一批窗口）——解法确认**

| checkpoint | `val_nll` | **`val_ratio`** | 配对差 vs 基线 | t |
|---|---|---|---|---|
| baseline@20000（skip=on） | 1.6201 | 0.9608 | — | — |
| armF@24000（skip=off） | 1.5959 | 0.9465 | +0.223 | +1.16 |
| armG@24000（skip=off+audio） | 1.3827 | 0.8201 | +0.627 | +2.44 |
| **armH@40000（skip=off+audio）** | **1.2568** | **0.7454** | **+0.860** | **+3.03** |

⇒ **`val_ratio` 0.9608 → 0.7454（−22%），配对 t = +3.03（n=139，p ≈ 0.003）**；
且 armH 的**在训 val**（step 40000：ratio 0.74546 / nll 1.2569）与**事后独立复评**（0.7454 / 1.2568）**逐位吻合**
——评测脚本与训练循环互为对照，无口径漂移。

**目标达成判定（goal：寻找模型学习失败的真正原因与真正有效的解决方案）**

- **原因**：输出头直连 skip（4 条独立证据：空间命中率 0.367 < 0.5；trunk/head 权位移 5–20% vs 1100–2900%；
  拆掉后 trunk 立刻开始学习、命中率 0.750；音频置零代价 0.012 → 4.97）。
- **解法**：`model.head_skip=false` + `model.audio_align=true`（串联）。三条证据线一致：
  **训练损失 10 → 0.015**、**512 窗 val_ratio 0.961 → 0.745（t = +3.03）**、**机制读数全部翻正**。
- **遗留（下一步，不属本目标）**：① 音频只当"内容"用（帧轴置换 ≈ 0）⇒ 时间定位仍未借助音频；
  ② 两个开关的默认值须决策者裁定（**"举措有效"的前提已由本目标满足**）；③ val 集合代表性（128 窗偏乐观、训练流非平稳）。

---

### §9-63 第十四轮②：**RFC-0039 R2/R3 实现 + 真实数据复评**（2026-09-29）

本节是 R2（val 重选）/ R3（端到端产物）的**实现记录与第一批评读数**。所有数字都来自实跑
（`runs/_verify_r2r3.py`，armH = `probe_fix_long/20260929-042940/step-0040000.pt`，
baseline = `phigros_masked/20260928-144342/step-0020000.pt`；读数日志 `runs/_verify_*.log`）。

**① R2：密度扫描是廉价的（这是它可行的前提）**

| 量 | 实测 |
|---|---|
| val split 窗口数 | **74 889**（索引 0.3-0.5 s，走落盘索引缓存） |
| 逐窗事件数扫描（`WindowCacheReader.event_count`） | **0.9-1.3 s** |

⇒ 「按事件密度分层」不再是一个昂贵的想法：只读缓存里的稀疏计数（`counts_ptr` / `counts_val`），
不物化稠密场、不读音频（无缓存路径是 0.85 s/窗 ⇒ 全 split 要 17 h）。

**② R2：新 val 集合的构成（与旧口径的对比就是「为什么必须换」）**

| 层（事件数区间） | 0 `[0,0]` | 1 `[1,9]` | 2 `[10,13]` | 3 `[14,17]` | 4 `[18,inf]` |
|---|---|---|---|---|---|
| 总体（74 889 窗） | 8 093 | 19 078 | 16 620 | 15 549 | 15 549 |
| 选中（512 窗） | 55 | 131 | 114 | 106 | 106 |

- **总体的空窗占比 = 10.8%**；而**旧的 128 窗 val 实测 57.8% 空窗、事件/窗中位 0**
  （§9-62 ⑪㉒）⇒ 旧量具的系统性偏差被这条对比钉死。
- 512 窗成 **141 批（≤4/批）**，集合指纹 `f9d9f7d1068c`（同 seed 逐位可复现）。
- 空窗层的配额只有 55/512 ⇒ 主判据不再被「空窗步」主导。

**③ R2：一次 val 的真实开销（RFC-0039 写的是「8-10 min」，实测更好）**

| 项 | 实测 |
|---|---|
| 512 窗 × 5 臂（模型 / 线内置换 / 音频置零 / 音频帧轴置换 / 事件轨置零） | **205.8 / 210.6 s**（两次，3.4-3.5 min） |
| 相对预算（`val_every=10000` ≈ 33 min @0.2 s/步） | ≈ **10%** |

⇒ 生产配置取 `val_windows: 512` + `val_every: 10000` 是**可负担的**；
RFC-0039 §R2 里「≈4×」的推断方向正确但绝对值偏悲观（当时按 128 窗 33 s 外推）。

**④ R2 的复评：在**新量具**上重跑两个关键 checkpoint（这是 R2 存在的意义）**

| checkpoint | `val_nll` | 常数基线 | **`val_ratio`** | `pred/true` | `cond_audio_zero` | `cond_audio_perm` | `cond_track_zero` |
|---|---|---|---|---|---|---|---|
| baseline@20000（skip=on） | 3.936 | — | **0.9987** | 0.303 | +0.014 | +0.004 | +0.471 |
| **armH@40000（skip=off + audio_align）** | **3.003** | 3.941 | **0.7620** | 0.522 | **−0.033** | +0.006 | **+1.664** |

- **解法在代表集上更强**：`val_ratio` **0.9987 → 0.7620（−24%）**。
  ⚠️ 注意 baseline 在新集合上**贴着红线**（0.9987 ≈ 1.0）——旧集合给它的 0.9608 是**偏乐观**的，
  新量具把「旧配置基本等于常数场」这件事说清楚了。
- `val_nll_shuffled_delta = +1.478 > 0`（RFC-0037 红线满足）。

**⑤ ★ 新量具推翻了一条上一轮的结论（下一轮必须知道）**

- 旧（128 窗，57.8% 空窗）：armG 的 `cond_audio_zero_delta = **+4.97**` ⇒ 当时判为「音频终于被依赖」。
- 新（512 窗，10.8% 空窗，代表集）：armH 的 `cond_audio_zero_delta = **−0.033**`（置零音频**不增加损失**）。
  ⇒ **在代表性窗口上，当前模型对音频没有可测的依赖**（连「当内容用」都没有）；
  而 `cond_track_zero_delta = +1.664` ⇒ 判定线事件轨是**真的被依赖**的。
  ⇒ 下一轮的核心目标（「修音频只当内容用」）**比上一轮记录的更严重**：
  旧读数是**稀疏集合的性质**，不是模型的普遍行为。R2 的第一次使用就改了结论——这正是它该做的事。

**⑥ R3：端到端产物的第一份真实产物**

- 产物：`tests/e2e-val/outputs/20260929-214209-step0/`（`chart.json` 2.9 MB + `meta.json` + `notes.txt`）。
- 配置：145 窗 × 8 步迭代并行解码；K=34（**条件来自清单里该曲的真谱** `manifest:test`，
  chart_id=15831《赴大荒》）；τ 轴 212.24 s、丢弃尾部 6 格；墙钟 **44.8 s**。
- 读数：解码候选 **760**（模型自己的期望事件数 ∫λdV = **721**）⇒ 665 note、**合法**、跨线冲突 0。
- **对照真谱**：1378 note ⇒ 模型给出 **48%** 的音符量（与 `val_pred_over_true = 0.522` 一致，互证）。

**⑦ R3 实现中暴露并修掉的一个真实缺陷（值得记：它是「产物机制」的第一个功用）**

第一版逐窗循环把**整首歌**的场事件全留在内存里：

| 现象 | 实测 |
|---|---|
| D1（peaks）α=1 的**单窗**场事件数 | **8.7-15.7 万**（该窗模型期望 **0.02-2.96**） |
| 逐窗 CPU（decode + pair） | 1.3-2.4 s（而 GPU 前向只有 0.2-2 s） |
| 进程常驻内存 | **21 GB**（145 窗 × ≈12 万 `DecodedEvent`） |
| 表现 | 单核 100%、GPU 0-10%、物理内存只剩 858 MB ⇒ **看起来像卡死** |

修法（都已落地）：① **流式 + 事件预算闸**（`MAX_DECODED_EVENTS = 60 000`，超限即中止并记账）；
② **逐窗解码统计不再丢弃**（`d1_*` / `d2_*` 的求和与 min/max）；
③ 新增 **`model_expected_events`（∫λdV）**——判断「解出多少事件算合理」的唯一基准；
④ 解码臂可声明（`method: thinning|peaks`），默认 **thinning**；
⑤ 每 20 窗打一行进度日志。
⇒ 同一首歌从「15+ 分钟未完成」变成 **45 s 完成**（145/145 窗）。

**⑧ 交给 plan 05 的两条实测发现（本轮的产物机制顺带产出的）**

1. **D1（peaks）的默认阈值 α=1 在真实训练场上完全没标定**：阈值取 `α·λ_0`（场的**平均**强度），
   而实测场的事件数是模型期望的 **10^4 量级**倍；α 从 1 扫到 32 只把事件数从 15.7 万降到 1.5 万
   （**每翻倍只降 15-20%**，因为峰计数对全局标度近似不变）。⇒ M5.7 的阈值标定是**必需项**，
   不是装饰；在标定之前，`peaks` 臂的产物不可审阅（`method=thinning` 才可用于人工审阅）。
2. **D2（thinning）与模型的期望事件数自洽**（760 候选 vs 期望 721 ✓），但它**不会自己配对 Hold**：
   实测 665 个 hold 起点只有 **2** 个配上终点、106 个退化成零时长 Hold ⇒ plan 05 需要一条
   Hold 端点策略（同一纤维内的成对抽样），否则产物的 Hold 全部是退化的。

**存疑清单（本轮新增）**

1. `val_windows=512` 的**抽样精度**仍未标定：新集合的代表性只由「与总体的密度分布一致」保证，
   没有做「512 vs 全 split」的偏差标定（成本 = 749 次 val）。
2. `val_every=10000` 是**按 10% 预算**定的，未做 A/B（更密 vs 更疏对 `best.pt` 的影响）。
3. e2e 产物的**人工可玩性**没有任何自动判据（RFC-0039 R3 明文「不是门禁」）；
   本轮只做到「合法性后处理为空违规」这一步。

---

### §9-64 第十五轮：**「模型学不会按音乐出谱」的根因 —— τ→秒表被冻结 + 音频通路被优化器抛弃**（2026-09-30）

**背景**：上一轮交接件（README 下一步 1）把「让音频真正参与」定为核心目标。§9-62 的解法
（`head_skip=false` + `audio_align=true`）在**代表性 512 窗**上把 `val_ratio` 压到 **0.7620**，
但 `cond_audio_zero_delta = −0.033`（置零音频**不增加**损失）⇒ 模型对音频没有可测依赖。
**决策者指令**：找到「无法根据音乐特征生成谱面」的根因与有效解法，**在解法被证明有效之前
不得开 RFC**。本节是实测记录，不是 RFC。

#### ① 代码级根因：`_field_seconds` 的缓存键漏掉了时间基

`MaskedFieldModel._field_seconds` 的缓存键原本是 `(t_bins, device)`，而 `grid.tau_seconds()`
由 `(t_bins, bpm_points)` **完全**决定：窗口网格的 `bpm_eff = 60/J(τ_start)` **逐窗口不同**，
`t_bins` 却是全库常数（`data.t_window`）⇒ 这张表在**第一次前向时被冻结**成「第一个窗口的
BPM」，此后**每个**窗口的 τ→秒 都被同一个**逐窗口变化**的因子缩放。
同一张表同时供 `audio_align`（帧下标）与 `seconds_position`（场 token 秒轴）使用。

工程上值得记的一点：这个键**看起来**是「按形状缓存一个派生量」的常规写法，
而它实际缓存的是**时间基**——`(t_bins, device)` 里没有任何一个量能表达 BPM。
复核脚本 `runs/_diag_seconds_cache.py`（CPU，10 行）：两张 `t_bins` 相同、BPM 差 2 倍的网格
返回**同一个张量对象**（`torch.equal` 为 True，比值 1.0 而非 2.0）。

#### ② 数据级实测：错位的规模（`runs/_diag_align_audit.py`）

| 量 | 实测 |
|---|---|
| 冻结口径来源：train 计划的**第一个**窗口 | `bpm_eff=301.5`、60 音频帧、0.794 s |
| val 512 窗的 `bpm_eff` | min 69 / p25 128 / **中位 162** / p75 210 / max 480 |
| 错位窗口占比 | **100.0%**（没有一窗的 BPM 等于冻结值） |
| 冻结下标只覆盖窗口**前段**的窗口 | **98.2%**（`bpm_true < bpm_frozen`） |
| 冻结下标越界被 clamp 到末帧 | 1.8% 的窗口（最多 40.6% 的格） |
| 前 4 个 val 窗（150 BPM、120 帧、真实跨度 1.596 s） | 只取到 **0.794 s** 的音频（前 50%） |

⇒ `audio_align` 注入的音频**不在该 token 的时刻上**，而是被一个**逐窗口未知**的因子压缩/
截断过的版本。模型拿不到 BPM，也就无法反解这个缩放（这正是 §9-62 ⑤ 提出的假设，
本轮把它从「可疑」钉成了「已实测」）。

#### ③ 推理层因果分解：那条通路是**死的**（`runs/_diag_audio_paths.py`，armH@40000、512 窗新口径）

基线 **val_nll 3.003145 / val_ratio 0.761963**（与 §9-63 ④ 的 3.003 / 0.7620 逐位吻合）。
同一批窗口上的 11 个干预臂（Δ = 干预后 NLL − 基线，>0 表示该条件被用上）：

| 干预 | Δ（nats/line） | 读法 |
|---|---|---|
| `both_zero`（音频整体置零） | **−0.0327** | 置零**反而更好** |
| `both_perm`（音频帧轴置换） | −0.0012 | 时间结构没被用 |
| `align_zero`（只关掉 audio_align 注入） | **+0.000002** | 该通路贡献 = 0 |
| `align_perm` / `align_roll8` / `align_swap` | +0.00011 / +0.000045 / −0.000092 | 对 τ 的对应关系无感 |
| **`align_true`（换成「正确对齐」的帧）** | **+0.000033** | 换对的也没有任何变化 |
| `align_x4`（注入幅度 ×4） | +0.000022 | 放大 4 倍也还是 0 |
| `cross_zero`（只关掉 cross-attention 的 K/V） | **+0.1318** | 这条路**有**作用 |
| `cross_perm` / `cross_swap`（帧轴置换 / 跨窗换音频） | −0.000046 / −0.005767 | 但只对「有音频」敏感 |

λ 级直测（`runs/_diag_align_path.py`）：注入向量逐元素 sd **0.766**、**93.6%** 的能量沿 τ 轴变化，
而把它置零 / ×4 / 换成同量级随机噪声，对 λ 的相对改变量分别是 1.1e-2 / 1.0e-2 / 1.0e-2
——**三者同量级**：token 流上「随机方向』的扰动与「整段音频注入」的扰动一样大。
⇒ 结论不是「这条通路被削弱」，而是**它在损失上根本不可见**（优化器把它压成了常数偏移）。

#### ④ 信息层：冻结把音频的 τ 信号抹到了随机水平（`runs/_probe_audio_info2.py` + `_probe_audio_within.py`）

探针口径（两个方法陷阱已就地记录）：**全局** AUC 会被「窗 / 线密度」的同义反复主导
（`vis_win` / `line_vis` 对同窗同线的全部 token 相同）⇒ 必须看**组内** AUC
（按 `(窗口, 线)` 分组，只在组内排序，即「**哪一拍**有音符」这一问）。
train 512 窗（计划等距抽样）/ val 512 窗（全分层集），186 万被遮盖 token、2945 正例。

| 特征组 | 全局 AUC | **组内 AUC** |
|---|---|---|
| τ 位置先验 | 0.5241 | 0.5533 |
| τ + 可见场 | 0.9491 | 0.5974 |
| 仅音频（对齐 = 修复后） | 0.5429 | **0.5337** |
| 仅音频（冻结 = 修复前） | 0.5305 | **0.5051** |
| τ+可见场+音频（对齐） | 0.9482 | 0.5874 |
| τ+可见场+音频（冻结） | 0.9451 | 0.5836 |
| 随机分数 | 0.5036 | 0.4984 |

两条读法：

1. **冻结的代价被钉死在信息层**：仅音频的组内 AUC **0.5337 → 0.5051**（随机 = 0.4984）
   ⇒ 错位把这条特征里「哪一拍有音符」的信息**基本抹平**。这是「模型学不会」的输入侧证据。
2. **音频的增量信息本来就小**：`τ+可见场` 已经到 0.5974，再叠加音频**没有增益**
   （0.5874，低于不加）⇒ 在**当前目标**（遮盖补全，50% 事件可见）下，音乐对「τ 定位」
   的边际贡献接近 0。这一条与下面 ⑦ 的配对实验结果一致，是下一轮真正的课题。

#### ⑤ 修法：删掉缓存（修根因，不是修键）

`beatmorph/generation/model.py`：`_field_seconds` 改为**每次直接求值**，删掉 `_seconds_cache`。
为什么不是「把 key 改成 `(t_bins, bpm_points)`」：那样键里必须重新表达一次 `field/` 已经拥有的
换算语义（红线 7 的边界），而且**下一次仍会有人忘掉一个自由变量**；而调用频率是**每批 1-2 次**
（不是每 token），`tau_seconds()` 是 192 元素级 numpy 运算 ⇒ 开销在噪声里。
护栏：`tests/unit/generation/test_audio_alignment.py`（4 项，**旧码下 2 项必红**，已复核）：
秒表随网格而非首次调用、帧下标落在该 token 的时刻上（量化 ≤ 半帧）、跨 BPM 不串表、
注入随批次网格变化。

#### ⑥ 配对训练验证：修复让音频从「被抛弃」变成「载荷项」（`runs/probe_alignctl2` vs `probe_alignfix2`）

两臂**同 seed / 同数据顺序 / 同配置 / 同 `data.workers=0` / 各 8000 步**，唯一差别是上面那处代码。
（`data.workers` 只影响取批实现，不改样本序列——RFC-0034 §5；此处两支都用 0 以避开
Windows 上 val 期新建 DataLoader 的 spawn 崩溃：对照组第一版正是**死在 8000 步的 val** 上。）

| 读数 | 对照臂 @4000 | **修复臂 @4000** | 对照臂 @8000 | **修复臂 @8000** |
|---|---|---|---|---|
| `val_ratio` | 0.7740 | 0.8137 | 0.8586 | 0.9030 |
| `cond_audio_zero_delta` | +0.5476 | **+1.5470** | **−0.0748** | **+1.0334** |
| `cond_audio_perm_delta` | −0.0026 | −0.0028 | +0.0005 | +0.0041 |
| `cond_track_zero_delta` | +1.731 | +1.616 | +1.142 | +0.238 |
| `val_pred_over_true` | 0.627 | 0.495 | 0.423 | 0.212 |

* **音频确实变成载荷项**：对照臂到 8000 步已把音频**彻底抛弃**（置零**不增加**损失，−0.075，
  与 armH@40000 的 −0.033 同一量级）；修复臂仍是 **+1.03 nats/line**（step 4000 时 +1.55）。
  这就是「模型无法根据音乐出谱」在**模型行为**上的直接读数被翻正的证据。
* **但谱面质量没有跟上**：`val_ratio` 两处都**略差**（+0.040 / +0.044），且 `pred_over_true` 掉到 0.21。
  ⇒ **「音频被用上」已验证有效，「谱面变好」未验证**——按决策者指令，此时**不开 RFC**。
* `cond_audio_perm` 两臂都 ≈ 0：模型用的是音频的**存在/整体内容**，仍不是**时间对齐**。

#### ⑦ 量具更正：`训练损失 10 → 0.015` 是**空窗步**的读数

§9-62 ⑲/㉖ 与 `configs/phigros_masked.yaml` 的注释都用「训练损失 10 → 0.015」作为解法成立的
一条证据。逐文件复核（`runs/_trend_new.py`）后更正：`0.015` 出现在 **`batch_events=0` 的空窗步**
（`loss_empty` = 纯积分项），它**不是**「拟合训练数据」的读数。同一口径的 `loss_nonempty_per_event`
分块中位：

| run | 非空窗 per-event 损失（4 分块中位） |
|---|---|
| baseline（skip=on，20k） | 9.909 / 9.876 / 10.066 / 10.358 |
| 对照（旧码，8k） | 9.607 / 9.285 / 10.090 / 10.217 |
| **修复（8k）** | 9.619 / 9.286 / 9.902 / 10.138 |
| armH（skip=off+audio，40k） | 8.563 / 8.887 / 8.354 / 8.240 |

⇒ `head_skip=false` 的真实训练侧收益约 **20%**（10.1 → 8.2，且到 40k 才显出来），**不是 660×**。
`val_ratio` 那条证据（0.9987 → 0.7620，配对 t=+3.03）不受影响——它是独立读数。
**config 里那句「训练损失 ~10 → 0.015」应作废**（本轮只登记，不改那条注释，避免越界改配置）。

#### 存疑清单（本轮新增）

1. **修复臂 `val_ratio` 为什么略差**：只有 1 组配对 × 2 个步点，而单 run 的步间摆幅本身有 0.08
   （对照臂 4000→8000 是 0.774 → 0.859）⇒ 这个差**未达可判读水平**，不能反过来说「修复有害」。
   要判它需要 24k 步的配对 + 多点，本轮预算不够。
2. **音频的增量信息为什么这么小**：④ 的组内 AUC 只做了**线性**读出（32 维随机投影），
   没做非线性探针；也没控制「可见场邻近事件」的计数口径（±8 格）。
3. **`seconds_position` 未测**：它同样被这处 bug 冻结过，修好后是否值得开（cross-attention 的
   两条轴这才真正落在同一时间基上）没有跑过任何臂。
4. **`cond_audio_perm ≈ 0` 的机制未定**：是「音频的时间结构信息本来就不够」（④ 支持），
   还是「cross-attention 那条路的对齐仍学不出来」（§9-62 ⑤ 的旧怀疑）。两者**未被区分**。
5. 两臂都用了 `data.workers=0`（避开 spawn 崩溃）。RFC-0034 只保证**样本序列**一致，
   不保证 bf16 下的逐位数值一致——实测对照组 workers=8 与 0 在 step 4000 **就分叉了**
   （val_ratio 0.7980 vs 0.7740，`cond_audio_zero` +0.350 vs +0.548）⇒ 跨 workers 设置的读数
   **不可直接比**，后续配对实验必须固定这一项。

### §9-65 判定线资格闸门：不可见 / 装饰线上不再产生 note（决策者实测报告，2026-09-30）

**报告**：e2e 产物里部分 note 由**已经不可见的判定线**判定，装饰 / 表演线也会拿到 note。
复核脚本 `runs/_audit_e2e_lines.py`（读产物 + 条件模板，逐 note 求 `pose_at().alpha`）：

| 读数 | 修复前（`20260929-214209-step0`，armH） |
|---|---|
| 生成 note | **665** |
| 落在**模板里零 note 的线**（装饰 / 表演线）上 | **149（22.4%）** |
| 落在 note 时刻**完全不透明 = 0** 的线上 | **238（35.8%）** |
| 典型线 | line 0（模板 1 note）拿到 103 个 note，alpha 中位 **0.000**；line 24（模板 0 note）71 个 |

根因：解码器只保证 `line_id` 在 `[0, K)` 内（`legality.py` 的既有判据），**对「这条线此刻
能不能判」零判据**；而模型在 K 条线上出强度场，K 条线里 22 条是装饰线。

**修法**（`beatmorph/decoder/events.py::filter_field_events_by_line` + `infra/e2e.py` 的 `LineFilter`）：
在 `pair_events` **之前**（Hold 起止同纤维，先配对后过滤会制造孤儿端点）丢掉两类场事件：
① 该线在**该事件的绝对时刻**不透明度 ≤ 阈值（经 `JudgeLine.pose_at`，跨层求和 + 父线递归）；
② `allowed_lines`（= 条件谱面里承载 note 的线集合）之外的线。
记账进 `decode_stats` / 产物 `meta.json`：`line_filter_{allowed_lines,kept_events,
dropped_empty_line,dropped_invisible}`。

**一条必须留意的语义**：RPE 的默认 alpha 语义是「没有 alphaEvents 的层求和为 0 ⇒ 不可见」，
而解码器自建的**合成模板**根本没有 alpha 轨——那是「没给信息」，不是「不可见」。
因此闸门只在**该线（含父线链）确实带 alpha 轨**时才判可见性；否则放行。
（把两者混为一谈会把无条件生成路径的 note **一次清空且不报错**。）

**复验**（同一 checkpoint armH@40000、同 seed、同 8 步采样，只改上述一处）：

| 读数 | 修复前 | **修复后**（`20260930-002922-step0`） |
|---|---|---|
| note 总数 | 665 | 411 |
| 装饰线上的 note | 149（22.4%） | **0** |
| 不可见线上的 note | 238（35.8%） | **0** |
| 合法性后处理 | 合法 | 合法（仍可导出） |
| 主判定线 line 30 | 336 note | 336 note（未动） |

护栏：`tests/unit/decoder/test_line_eligibility.py`（6 项：不可见判据逐事件求值、父线链、
缺 alpha 轨 = 未知而非不可见、装饰线集合、无集合时不筛装饰线、记账键完整）。
⚠️ 实现时被这组测试抓到一个真 bug：帧下标缓存键写成 `int(tau)` 会把一个 4 拍窗里 τ<1 的
**全部格塌成同一个键**（整窗只按第一个事件判一次可见性）⇒ 已改用 `tau_bin_index`（权威实现）。

**同轮顺带的产物字段审计**（`runs/_audit_e2e_fields.py`，修后产物 411 note）：

| 字段 | 实测 | 读法 |
|---|---|---|
| `alpha` / `visible_time` / `is_fake` / `size` | 255 / 999999.0 / False / 1.0（411/411） | 没有退化字段 |
| `type` 分布 | Tap 267 / Drag 64 / Flick 60 / Hold 20 | 与 `pred_over_true=0.52` 一致的稀疏解 |
| note 时刻范围 | 0 – 211.07 s（谱面 212.24 s）、`t<0` 0 个、`tau<0` 0 个 | 无轴外事件 |
| 同线同刻 >=2 个 note | 5 组（最大 2） | 不同 x/type，不是重复事件 |
| **Hold** | 64 个里 **63 个 `hold_time <= 0`**（均值 0.022 s） | 已知项：plan 05 M5.7 的 Hold 端点策略 |

**已登记的口径限制**：闸门只在 note 的**起始时刻**判可见性。一个「起点可见、终点落在不可见段」
（或反之）的 Hold 目前**不被拦**——当前产物里 63/64 个 Hold 是零时长所以看不出来，
但 plan 05 修好 Hold 端点配对之后必须一并处理（判据应取 Hold 的整个区间）。

---

### §9-66 第十六轮：**「模型学不会音乐」的判据侧诊断 —— 目标不需要音乐、判据与目标脱节、输入本身封顶**（2026-09-30）

**背景**：§9-64 修掉了「音频从来没在对的时刻进模型」（冻结的 τ→秒表），配对实验证明音频**确实被用上了**
（`cond_audio_zero` −0.075 → +1.033 nats/line），但 `val_ratio` **没有变好**。目标仍是
「寻找模型学习音乐失败的真正原因与真正有效的解决方案」。本轮不再改模型，先把**判据侧**量清楚：
增益从哪来、生成侧到底差多少、输入里还有多少可用的定位信息。

#### ① 收益阶梯：**4 个可见场计数特征就等价于那个 40M 模型**（`runs/_probe_gain_ladder.py`）

做法：λ_token = exp(w·x)，token 内空间分布用**训练集的空间边缘** p(x,s,c)（138 万 token / 3183 事件），
损失的求值**走同一个 `masked_poisson_loss`**（256 train / **512 val**（全分层集）窗）。

| 特征组 | 维度 | val_nll | **val_ratio** |
|---|---|---|---|
| 常数场（判据分母） | 0 | 3.915719 | 1.000000 |
| 偏置（只有水平） | 1 | 3.691881 | 0.942836 |
| τ 位置先验 | 6 | 3.694120 | 0.943408 |
| **可见场（邻域 / 本线 / 全窗 三个计数）** | **4** | 3.017791 | **0.770686** |
| **τ + 可见场** | **9** | 3.003105 | **0.766936** |
| τ+可见场+事件轨 | 14 | 3.794039 | 0.968925 ⚠️ 拟合不稳定，**作废** |
| + 音频（对齐） | 46 | 2.969900 | 0.758456 ⚠️ 同上，**作废** |
| + 音频（冻结） | 46 | 2.976048 | 0.760026 ⚠️ 同上，**作废** |
| 仅音频（对齐） | 33 | 3.800765 | 0.970643 |
| 仅音频（冻结） | 33 | 3.851436 | 0.983583 |
| **armH@40000 模型（40M 参数）** | ~40M | 3.003 | **0.7620** |

**读数**：一个 **9 维线性读出**（τ 的 5 个傅里叶项 + 3 个可见场计数 + 偏置）拿到 **0.7669**，
而 40M 参数的模型是 **0.7620**——**差 0.5%**。也就是说：**在遮盖补全的判据下，模型的全部增益
基本可以由「抄可见场的密度 + 一个空间边缘先验」解释完**，音乐连一个可见的份额都没有。
（可见场与**被遮盖事件**的统计关联是真实存在的——两者来自同一片稠密区——所以这不是泄漏，
而是这个判据本身的性质：r=0.5 时可见密度几乎是隐藏密度的充分统计量。）

#### ② 定位探针：音乐是**最强单特征**，而冻结的表把它抹到随机（`runs/_probe_localization.py`）

同一套探针，但目标换成**全部 token**（不做遮盖），指标是 **(窗口, 线) 组内 AUC**
（全局 AUC 会被「窗/线的密度同义反复」主导；512 val 窗、354 万 token、4903 正例）：

| 特征组 | 全局 AUC | **组内 AUC** |
|---|---|---|
| τ 位置先验 | 0.5196 | 0.5528 |
| 可见场（**生成时根本没有**） | 0.9627 | **0.4812**（组内反而低于随机） |
| 条件事件轨（生成时有） | 0.9163 | 0.5369 |
| τ + 条件事件轨 | 0.8835 | **0.5682** |
| **仅音频（对齐 = 修复后）** | 0.5662 | **0.5600** |
| **仅音频（冻结 = 修复前）** | 0.5552 | **0.5166** |
| 事件轨 + 音频（对齐） | 0.8848 | 0.5619 |
| 随机分数 | 0.5018 | 0.5019 |

两条：① **音频是「哪一拍有音符」的最强单特征（0.5600）**，比条件事件轨（0.5369）还高，
而**冻结那张表把它砍到 0.5166**——§9-64 修的正是信息量最大的那条线索；
② 可见场在组内是**反信息**的（0.4812）：它只告诉你「这一窗/这条线有多少音符」，
不告诉你「哪一拍」——而这恰好是生成任务唯一需要的东西。

#### ③ 生成侧判据：`val_ratio 0.762` 与 `event F1 0.002` 同时为真（`runs/_eval_generation.py`）

R3 的产物第一次有了数：用 `beatmorph/eval` 的权威实现（plan 06）把产物与**该曲真谱**比：

| 产物 | note | timing F1 @20/50ms | **event F1 @20/50ms** | position_x MAE |
|---|---|---|---|---|
| armH@40000（修复前，含非法线） | 665 | — | 0.0010 / 0.0010 | 307 |
| **armH@40000 + 判定线闸门** | 411 | 0.0794 / **0.1923** | 0.0011 / **0.0022** | 299 |
| 修复臂 @8000（非盲） | 96 | 0.0217 / 0.0488 | 0.0000 / 0.0000 | 333 |
| 生成制度臂 @8000（盲） | 23 | 0.0043 / 0.0086 | 0.0000 / 0.0000 | 459 |

（真谱 1378 note；`position_x` 的舞台半宽约 617 ⇒ MAE 299 意味着 x 基本随机。）

**这是本轮最重的一条**：模型在它自己的判据上是「比常数场好 24%」，在目标上却是
**与人类谱面几乎无关**（event F1 0.002；timing F1@50ms 0.19 ≈ 只对上了 12.5% 的音符时刻）。
⇒ **判据与目标脱节**：`val_ratio` 量的是「密度的遮盖补全」，而目标是「这首歌该在什么时候出什么键」。

#### ④ 输入天花板：最强线索是**条件事件轨**（`runs/_probe_ceiling.py`，口径有保留）

做法：特征按 (窗口, 线) **组内标准化**（只留组内排序信息），线性读出后取分数最高的 N 个 token
（N = 该组真实正例数）作为 τ 定位预测，统计命中率（token = 1/48 拍 = 164BPM 下 7.6ms ⇒ 等价于 timing F1 的上界）。
128 val 窗、4591 组、376 个正 token：

| 特征组 | TP/376 | F1（τ 定位上界） | **组内 AUC** | 随机 F1 |
|---|---|---|---|---|
| **条件事件轨（生成时有）** | 80 | **0.2128** | **0.6431** | 0.0319 |
| 事件轨 + 音频（对齐） | 32 | 0.0851 | 0.6289 | 0.0346 |
| 事件轨 + 音频（冻结） | 52 | 0.1383 | 0.6410 | 0.0293 |
| τ + 可见场（补全制度） | 76 | 0.2021 | 0.5831 | 0.0213 |
| 仅音频（对齐） | 22 | 0.0585 | 0.5762 | 0.0133 |

**只能取组内 AUC 那一列**：top-N F1 在只有 376 个正例时对**排序头部**极不稳定（同一组内 AUC 0.629 与
0.641 的两个拟合，F1 可以差 1.6×）⇒ 该列**登记为未解决**，不作为结论。可复述的是：

* 组内标准化后**条件事件轨是最强的 τ 定位线索（0.643）**，音频（对齐）第二（0.576），
  冻结的音频掉到 0.52（与 §9-64 一致）；
* 模型在真谱上的 **timing F1@50ms = 0.192** 与「只用事件轨的线性上界 ≈ 0.21」同量级
  ⇒ **模型已经把手里最好的那条线索用掉了**，τ 定位的瓶颈在**输入**，不在容量或目标函数；
* 音频与事件轨高度冗余（联合 0.629 < 事件轨单独 0.643）⇒ **在「事件轨已给」的条件下，音乐几乎不提供额外的 τ 信息**。

#### ⑤ 生成制度臂（`model.visible_input=false`）：第一步实验，8k 步**尚无定论**

新开关（`ModelConfig.visible_input`，schema 同步；护栏 `tests/unit/generation/test_blind_regime.py` 4 项）：
把**训练输入**换成**推理第一步**的输入——可见场全 0、遮盖通道全 1（迭代解码的第 1 步就是这样），
损失与遮盖测度**一个字不改**（事件项仍只监督被遮盖事件、仍 1/r 重标定 ⇒ 仍是对全谱密度 MLE 的无偏估计）。

| 臂（8000 步，同 seed / 同数据 / 同 `data.workers=0`） | val_ratio | cond_audio_zero | cond_track_zero | 生成 note | timing F1@50ms |
|---|---|---|---|---|---|
| 对照（旧码，冻结表） | 0.8586 | **−0.075** | 1.142 | — | — |
| 修复（表按窗口） | 0.9030 | **+1.033** | 0.238 | 96 | 0.0488 |
| **生成制度（盲）** | **0.8326** | +0.272 | 1.039 | 23 | 0.0086 |

* 盲臂的规划判据**最好**（0.833 vs 0.903 / 0.859），但它把依赖几乎全压在**事件轨**上（1.039），
  音频只剩 +0.272；生成侧反而**更差**（23 note）。
* 机制上的一个已知代价：盲模型对可见场/遮盖通道**完全无感** ⇒ 迭代并行解码的后续步与第一步
  输入相同，**8 步迭代退化为一步**。因此「盲 vs 非盲」在生成侧同时混入了「迭代失效」，
  这一步实验**不能单独宣判制度假设**。24 000 步的盲臂正在跑（`runs/probe_genregime24k`）。

**⑥ 盲臂的音频依赖**同样在衰减**（本轮交付时的在跑读数，`runs/probe_genregime24k`）**：

| 步 | val_ratio | cond_audio_zero | cond_track_zero |
|---|---|---|---|
| 4000 | 0.8176 | +0.300 | +0.912 |
| 8000 | 0.8358 | **+0.071** | **+1.833** |

⇒ 即使**把可见场彻底拿掉**（模型只能从「音频 + 条件事件轨 + τ/线先验」出谱），训练过程仍然把依赖
**几乎全部压到条件事件轨上**，音频边际掉到 ≈0。这与 ④ 的信息读法一致：**事件轨与音乐高度冗余**
（组内 0.629 < 事件轨单独 0.643），而事件轨在 e2e 里是**从该曲真谱复制**来的（R3 的条件解析）。
**推论（登记为假设，未做验证）**：只要条件里给了真谱的判定线事件轨，「从音乐出谱」这件事在信息上
就被提前解决了大半；要真正检验「模型能不能靠音乐出谱」，条件必须**去掉事件轨**（无条件生成）
或者让事件轨**由模型自己产出**——这正是 RFC-0029/RFC-0032 一直挂着「无条件生成时由谁产出随之裁定」
的那个未决项，本轮第一次给出了它的量化依据。

#### 本轮结论（可复述）

1. **判据侧的真原因**：在**遮盖补全**这个判据下，音乐是不必要的——**4 个可见场计数特征 + 空间边缘**
   就打平了 40M 模型（0.767 vs 0.762）。梯度里没有让模型去听音乐的力。
2. **判据与目标脱节**：同一个模型 `val_ratio 0.762`、`event F1 0.002`。要判「会不会写谱」必须看
   生成侧读数（本轮已把这条读数建起来）。
3. **输入侧封顶**：生成时手里最强的 τ 线索是**条件事件轨**（组内 0.643），音频其次（0.576，
   冻结时 0.52）；模型已经把事件轨用到了与线性上界同量级的水平 ⇒ 想再上一个台阶，得先有
   **更能定位时间轴的音频特征**，而不是继续调主干。
4. **仍未验证「有效解法」**：§9-64 的对齐修复是**必要**的（它把音频从 0.52 拉回 0.58，并让模型
   真的依赖音频），但**不足以**让 `val_ratio` 变好；制度侧的第一步（盲训练）8k 步未分胜负。
   按决策者指令：**此时不开 RFC**。

**存疑清单（本轮新增）**

1. **top-N F1 上界不稳**（376 正例 + 排序头部）⇒ ④ 的 F1 列**只作登记**；要拿到可判读的上界
   需要更大的 val 集（或直接在整个 e2e 曲目上做）。
2. **「4 个计数特征打平 40M 模型」只在这一条判据（512 窗分层 val）上验证过**；换 val 集合、
   换 r 都还没做（r 变则窗口缓存指纹变，代价见 §9-52）。
3. **音频特征本身没被质疑过**：本轮的探针一律用「单帧 + 1 步差分」的 32 维随机投影。
   多帧上下文 / onset 强度 / 对数梅尔 等更强的读出是否能把 0.58 抬上去，**未测**。
4. **盲训练的迭代退化**（8 步 → 1 步）未解：要真做「推理制度训练」，遮盖通道必须按推理的
   schedule 变化（数据侧改动 ⇒ 窗口缓存重建），而不是恒 1。
5. 24 000 步盲臂的结果**尚未回来**（本轮交付时仍在跑）。

---

### §9-67 第十七轮：**τ 定位的信息来源与上限 —— 更正 §9-66 的一条读数，并把「信息 vs 读出」分开**（2026-09-30）

**背景**：§9-66 ④ 曾写「条件事件轨是最强的 τ 定位线索（组内 AUC 0.643），音频第二（0.576）」。
本轮复核发现那条读数的**样本口径错了**，同时把「是不是特征太弱」这一问分离出来。

#### ① 更正：0.643 来自**非代表集**（val 的前 128 窗）

`runs/_probe_ceiling.py` 当时用了 `val_windows()[:128]`——而分层集的**前 55 个是空窗层**、其后 73 个
是最稀疏的一层（事件 1-9）⇒ 那是**整个 split 最稀疏的一端**，不是代表集。全 512 窗（代表集）上重测，
同一套特征与拟合（组内标准化 + 无类别权重）：

| 特征组 | **组内 AUC（512 窗，代表集）** |
|---|---|
| τ 位置先验 | 0.5555 |
| **音频（对齐 = 修复后）** | **0.5551** |
| 音频（对齐）+ τ | **0.5716**（全场最好） |
| **音频（冻结 = 修复前）** | **0.4983**（= 随机） |
| 轨 move_x / move_y / rotate / alpha / speed（单通道） | 0.5057 / **0.5401** / 0.5044 / 0.5191 / 0.4778 |
| 轨：全部 5 通道 | 0.5085 |
| 轨：move+rotate（坐标系） | 0.5199 |
| 轨：alpha+speed（可读性） | 0.4915 |
| 音频 + move+rotate | 0.5571 |
| 可见场（生成时没有） | 0.4812（§9-66 ②，同口径） |

**更正两条推论**：

1. **事件轨不是 τ 定位线索**（代表集 0.51，与随机无异）。它在损失上依然值钱（`cond_track_zero` 0.9-1.8），
   但那份价值在**窗/线密度**上——正是组内 AUC 刻意扣掉的那一维。所以 §9-66 的「音乐与事件轨高度冗余」
   与「模型已把事件轨用到线性上界」两条**不成立**，就地作废。
2. **冻结那张表把音频的 τ 信息抹到随机**：0.4983 vs 对齐 0.5551（这是第三次独立测到同一件事，
   且这次是「与随机不可区分」的强表述）。§9-64 的修复由此在信息层有了**最干净的一条证据**。

#### ② 信息 vs 读出：换更强的音频前端**抬不动**（`runs/_probe_audio_frontend.py`）

同一套探针（256 train 窗 / **256 等距** val 窗、组内标准化、线性读出），只换音频特征：

| 音频读出 | 组内 AUC |
|---|---|
| 现状：单帧 + 1 步差分的 32 维投影（= `audio_align` 用的那一路） | **0.5506** |
| 多尺度起始包络 \|a[f]-a[f-d]\|，d=1/2/4/8 | **0.4997**（随机） |
| 能量 + 能量一阶差分 | 0.5531 |
| 包络 + 能量 + 投影（三路并联） | 0.5518 |
| τ + 现状音频 | **0.5724** |
| τ + 包络 + 能量 + 投影 | 0.5638 |
| （对照）冻结口径的投影 / 包络 | 0.5243 / 0.5192 |

⇒ **手写的更强读出没有增益**：最好的一档仍是「现状 + τ」（0.5724），而 τ 先验单独就有 0.5583。
也就是说 **MERT 帧本身对「哪一拍有音符」的判别力就这么大**——瓶颈在**信息**，不在读出形式。
（限定：这仍是**线性**读出的结论；非线性读出是否更高，要由模型自己来回答，见 ③。）

#### ③ 待测：模型自己在这条线上在哪？（本轮未完成）

把 armH@40000 的 λ 按 (窗口,线) 组内 AUC 与探针的 0.57 比：若模型 ≈0.57 ⇒ 它已经吃满了输入能给的
τ 定位信息（那「学不会音乐」的根因就在**输入**）；若显著低于 0.57 ⇒ 还有可训练的空间（根因在优化/目标）。
**这条读数本轮没跑完**（GPU 让给了音乐-only 臂），列进下一轮第一件事。

#### ④ 音乐-only 臂（`model.visible_input=false` + `model.track_input=false`）

**动机**：把两条「捷径」同时拿掉——可见场（§9-66 ① 的密度插值）与判定线事件轨内容（本轮 ①；
它虽然不定位 τ，但提供了窗/线密度）。剩下的只有**音频 + 线身份 + τ 轴 + 定数**。这才是
「只靠音乐出谱」的制度。新开关 `ModelConfig.track_input`（只清空轨**内容**，保留位置编码与 line embedding，
因此模型仍知道「哪条线、第几拍」），护栏 `tests/unit/generation/test_track_input.py`（2 项）。

臂配置：`--experiment probe_musiconly --max-steps 8000`（`data.workers=0`，门禁 G1/G3/G4 已全绿）。
**本轮交付时仍在跑**（约 3.4k/8k 步）；判据：`cond_audio_zero`（音乐是否是唯一来源）+ 生成侧
timing/event F1 + 线闸门统计（不知道可见性时会有一批 note 落在不可见线上，这条要如实报）。
#### ⑤ 显式起始前端：**唯一一个把 τ 定位抬上去**的改动（`runs/_probe_onset.py`）

动机：§9-67 ② 的结论只在「MERT 帧」这一族里成立——MERT 是语义编码器，不是起始检测器。
于是直接从**原始音频**（`data/processed/audio/<feature_key>.*`）算：STFT（n_fft=1024、hop=24000/75=320，
与 MERT 帧**同栅格**）→ 谱通量（正半波差分的频带和）、局部极大、对数能量。
同一套探针（256 train / 256 等距 val 窗，组内标准化，线性读出）：

| 特征组 | 组内 AUC |
|---|---|
| τ 先验 | 0.5583 |
| MERT 现状读出（单帧 + 1 步差分投影） | 0.5506 |
| **原始音频：谱通量（单帧）** | **0.5792** |
| 原始音频：起始包络（通量 + 局部极大 + 能量） | 0.5744 |
| τ + MERT 现状 | 0.5724 |
| **τ + 起始包络** | **0.5818** ← 目前全场最好 |
| τ + 起始包络 + MERT | 0.5762 |

⇒ **显式 onset 前端比模型现在吃的那一路（MERT 帧差分）高约 +0.03 AUC**（0.579 vs 0.551；带 τ 时
0.582 vs 0.572）。这是本轮唯一**抬动了天花板**的改动，方向明确：**给模型加一条 onset 通道**。
⚠️ 幅度不大（+0.03 AUC），而且有**真实成本**：onset 包络必须从原始音频重算，等于给全库
（8551 首）做一次新的特征提取（与 MERT 提取同量级的工程），不能只改模型。

⚠️ **同表里「可见场」那一类读数要小心**：本轮探针在**全部 token** 上评估，而可见 token 的
`visible > 0` **恒等**于「它有事件」⇒ 该特征对可见 token 是同义反复（实测因此能到 0.71）。
音频/事件轨/onset 各列不读 counts，**不受此影响**；可见场只作对照。

#### ⑥ 音乐-only 臂：消融**精确**成立，模型确实转向音乐（`runs/probe_musiconly`）

配置：`model.visible_input=false` + `model.track_input=false`（门禁 G1/G3/G4 全绿）。
**因此条件里只剩**：音频、线身份 + 位置编码、定数、以及恒为 1 的遮盖通道。

| 步 | val_ratio | `cond_audio_zero` | `cond_track_zero` |
|---|---|---|---|
| 4000 | 0.8754 | **+0.5426** | **9.37e-09** |
| 8000 | 0.9116 | +0.1589 | 6.5e-08 |

⚠️ 音频依赖**又在衰减**（+0.54 → +0.16），而 `val_ratio` 同时在变好（0.875 → 0.912）、

两条读数：① `cond_track_zero = 9.4e-9` ⇒ 消融**逐位成立**（模型对事件轨完全无感，
与 `tests/unit/generation/test_track_input.py` 的断言一致）；② 在这种「只剩音乐」的制度下
模型**确实依赖音频**（置零 +0.54 nats/line），而 `val_ratio` 仍优于常数场 12.5%。
生成侧（e2e + 真谱 F1）**尚未跑**（本轮时间用在了 ⑦ 上）；注意它必然受一个真实约束：
不知道可见性时会有一批 note 落在不可见线上，会被 §9-65 的闸门丢掉，这条要如实报。
#### ⑦ ★ 模型自己在这条线上在哪：**τ 定位 = 随机**（`runs/_probe_model_within.py`，512 窗全 val）

把模型的 λ 折成「每个 token 的总强度」，在 **(窗口, 线) 组内**排序（同一度量、同一 val 集）：

| 分数 | 组内 AUC |
|---|---|
| **模型 λ（armH@40000）逐 token** | **0.5009** |
| 模型 λ 的 **τ 边缘**（在 K 条线上求和后广播回每条线） | **0.4985** |
| 真值计数（度量自检） | 1.0000 ✓ |
| 随机 / 常数（度量自检） | 0.5289 / 0.5000 ✓（合成夹具上另测） |
| **可达（线性探针）**：τ 先验 / MERT 现状 / onset / τ+onset | **0.5583 / 0.5506 / 0.5792 / 0.5818** |

**两种制度下都是随机**：音乐-only@8000 的模型逐 token **0.5021**、τ 边缘 **0.5004**。

⇒ **模型学到了「谱面长什么样」（密度、x/类型的边缘），但没学到「这首歌的音符落在哪一拍」**：
它在同一条线内对 τ 的排序与随机不可区分（0.50），而输入里**可达**的水平是 0.5555-0.5818。
这条与 §9-66 ①/③ 完全自洽：`val_ratio` 的增益来自「可见场密度 + 空间边缘」，与本条测的
「时间落点」是**两件不同的事**——而 R3 产物的 `event F1 = 0.002`（x 也随机）+ `timing F1 = 0.19`
正是这件事的下游后果。

⚠️ 但这**不代表**「模型做得差」：那条可达信号本身很弱（0.58 AUC ⇒ 192 个 token 里挑中那一个的概率
只是从 0.5% 抬到 ~2%），对损失的贡献很小 ⇒ 模型「不学」它可能是**理性的**。要判成
「优化失败」还是「信号太弱」，需要把 τ 定位**单独**做成判据（例如给定 token 的时间组只算 τ 那一维）。
这条列进下一轮。
**生成侧（补记，`20260930-033137-step0`）**：音乐-only@8000 的产物 = **34 note**（模型自报期望 399），
真谱 1378 ⇒ recall 1.2%；但 **timing precision@50ms = 0.500**（17/34），是迄今**最高的一档**
（对照：armH@40000 0.418 / 修复@8000 0.375）。⇒ 只靠音乐时模型**出的音符少但更准**——
「音乐确实带一点时间信息，但模型只能很保守地用它」；样本很小（34 个），只作登记。
（⚠️ `line_filter_kept_events` 与 `model_expected_events` 的量纲提示 `_accumulate_decode_stats`
的过滤键可能不是累加口径，下一轮顺手核一下；不影响上面的 F1 读数，它来自产物本身。）
---

### §9-68 第十八轮：**生产配置把模型的「时间轴」整个丢掉了 —— τ 置换代价 −7e-6（老基线 +2.148）**（2026-09-30）

**背景**：§9-67 ⑦ 量到「模型 λ 在 (窗口,线) 组内对 τ 的排序 = 0.5009（随机）」，而输入里可达 0.55-0.58。
本轮先问**为什么**，答案不是「学不好」，而是**它根本没有时间轴**。

#### ① 决定性读数：置换输出的整条轴（`runs/_probe_tau_reward.py`，同批 val、同口径）

把模型输出的 λ **整体置换 τ 轴**（所有线、所有格同一置换）再算 `masked_poisson_loss`，
对照置换 x 轴：

| checkpoint | **τ 置换 Δ** | x 置换 Δ |
|---|---|---|
| baseline@20000（`head_skip=true`，**旧配置**） | **+2.148** | +0.444 |
| **armH@40000（`head_skip=false`，当前生产配置）** | **−0.000007** | +0.357 |
| 修复臂@8000（同生产配置） | +0.000008 | +0.298 |
| 对照臂@8000 | +0.000038 | +0.283 |
| 音乐-only@8000 | +0.000060 | +0.283 |

⇒ **所有 `head_skip=false` 的模型对 τ 完全无感**（Δ ≤ 6e-5），而老的 `head_skip=true` 基线是 **+2.148**。
x 轴在两种配置下都在 +0.28…+0.44 ⇒ 「空间还在、时间没了」。

#### ② 互证：λ 沿 τ 近似**均匀**（`runs/_probe_tau_profile.py`，70 批 / 220 万 token）

* armH@40000：质量最多的 25% τ 格只占 **0.250** 的质量（均匀 = 0.25）⇒ **精确均匀**；
  `corr(λ 质量, 真值事件数)` 按 192 格 = **−0.071**（只取有事件的 111 格 = −0.081）；
* baseline@20000：0.267（略尖）、corr = **+0.019 / +0.054**；两者的组内 τ AUC 都是 0.47-0.51。
⇒ 两条独立证据（置换代价 = 0、剖面 = 均匀）指向同一件事，且**旧配置也一样弱**——
差别只在「有没有那条短路径」。

#### ③ 机制：**用空间轴换掉了时间轴**（一句话能复述的因果链）

1. 老配置里 `FieldHead` 有两条直连 skip：`cell_skip`（→ 空间 softmax p）与 `cum_skip`（→ 标量 δ(τ)）；
2. §9-62 实测指控的是 **`cell_skip`**：对被遮盖 token 它的输入恒为常数 `[0…0, 1…1]` ⇒ 只能给**与 token
   无关的空间先验**，而该先验与真值反相关（命中率 0.367）、token 内熵只有 1.88/7.16；
3. 但 `cum_skip` 那条路**顺手提供了 τ 形状**：它读的是该 token 的可见场（哪些 τ 上有可见证据），
   而可见事件与被遮盖事件同处一片区域 ⇒ δ(τ) 天然带时间结构（这正是 +2.148 的来源）；
4. 当时把**两条一起关掉** ⇒ `val_ratio` 0.9987 → 0.762（空间大赚）、**τ 置换代价 2.148 → 0**
   （时间全亏）——而**没有一条判据能看见后者**（`val_ratio`、`loss_*`、命中率、熵都只测空间与密度）。

⇒ 这是本轮对目标的第一个**结构性结论**：模型「学不会音乐」的一个重要原因是**它连时间都不建模**，
而生成的谱面之所以「像谱面但不对」正是因为落点在时间维上是均匀的。

#### ④ 修法：把两条 skip 拆开，**只要回 τ、不要回空间捷径**

`ModelConfig` 新增 `head_cell_skip` / `head_cum_skip`（`None` = 跟随 `head_skip`，**向后兼容**；
两个 `Linear` 模块本来就一直在场 ⇒ **checkpoint 形状不变**，老的权重照旧能载入）。
实验臂 = `head_skip=false` + **`head_cum_skip=true`**：`cum_skip` 只作用于每 token **一个标量 δ**，
给不出「与 token 无关的空间先验」⇒ 不复活 §9-62 指控的那条路。
护栏 `tests/unit/generation/test_head_skip_split.py`（5 项：解析语义 + 三条通路的**梯度归属**：
cum-only 臂只有 `cum_skip` 拿到梯度、`cell_skip` 必须为 0）。

#### ⑤ 预登记判据（**跑之前写死**，防止事后挑读数）

1. **τ 置换 Δ 明显 > 0**（对照：同配置的修复臂是 +8e-6，老基线是 +2.148）；
2. λ 的 (窗口,线) **组内 τ AUC > 0.50**（朝向探针可达的 0.55-0.58）；
3. `val_ratio` **不低于**同预算（8000 步）的修复臂 **0.9030**；
4. 生成侧 timing F1 > 同预算修复臂的 **0.0488**。

臂：`--experiment probe_cumskip --max-steps 8000`（`data.workers=0`，门禁 G1/G3/G4 已全绿）。
**本轮交付时仍在跑**（约 0.15k/8k 步）。
---

### §9-69 第十九轮：**目标函数确实奖励 τ（占常数基线 42.5%），而模型连「每条线几个音符」都没打赢**（2026-09-30）

**背景**：§9-68 量到「所有 `head_skip=false` 的模型对 τ 无感」。本轮先问**目标函数本身**在 τ 上给不给奖励——
若不给，那模型平坦是理性的，一切改模型的动作都白做。

#### ① 目标函数的轴分解（族最优版，`runs/_probe_objective_tau.py`，**不需要模型**）

遮盖测度下的损失逐格可分：`L(λ) = Σ_j [ -s·n⁰_j·log λ_j + λ_j·dV_j ]`（s = 1/r，n⁰ = 被遮盖计数）。
因此「沿某轴取常数」的约束族**最优成员有闭式解**（该轴上 n⁰ 与 dV 各自求和再相除）。
下表每行都是**该族在损失意义下的最优场**，且全部用**权威损失实现**求值：

| 族（oracle 场） | NLL | 除以 const | 含义 |
|---|---|---|---|
| `const` | 3.94133 | **1.0000** | val_ratio 的分母 |
| `line_only` | 2.83622 | **0.7196** | 只知道每条线**总共**几个音符 |
| `space_flat` | 1.64383 | **0.4171** | τ 全对、空间无结构 |
| `tau_flat` | 1.00877 | **0.2559** | 空间全对、τ 无结构 |
| `oracle` | −0.66486 | **−0.1687** | 无约束最优（完美模型） |

⇒ **τ 轴值 42.46% 的常数基线、空间轴值 58.58%、线轴值 88.83%**（`gap_tau/gap_space = 0.725`）。
**「目标函数不奖励 τ」被否定**：丢掉 τ 结构要付 1.67 nats/线 的代价，是全部分数空间的一大块。

#### ② 而模型在哪：**比「只知道每条线几个音符」还差**

同一测度下模型的 `val_ratio`：

| 臂 | 步 | val_ratio |
|---|---|---|
| 修复臂 `probe_alignfix2` | 4000 / 8000 | 0.8137 / 0.9030 |
| 对照臂 `probe_alignctl2` | 4000 / 8000 | 0.7740 / 0.8586 |
| 生成制度 `probe_genregime` | 4000 / 8000 | 0.8787 / 0.8326 |
| 音乐-only `probe_musiconly` | 4000 / 8000 | 0.8754 / 0.9116 |
| cum-skip `probe_cumskip` | 4000 / 8000 | **0.7636** / **1.0029** |

对照 `line_only` = **0.7196** ⇒ **最好的模型也没打赢「每条线一个常数」**（且它是 oracle 场，模型还得从可见场推）。
⇒ 训练出来的 λ 在 (τ, x, s, c) 上**没有任何净有益的偏离**；这不是「时间学不好」，是**输出几乎只剩线密度**。

#### ③ 机制：架构是 `λ = ΔΛ(t)·p_t(x,s,c)/dV`，**τ 边缘 ≡ ΔΛ(t)**

`field/integrate.factorized_lambda` 把因子化头写死成 λ_j = ΔΛ_{k,t}·p_j/dV_j（Σ_{x,s,c} p = 1）。
于是模型能表达的时间结构只有两种：(i) 每 τ 格的质量 `ΔΛ(t)`；(ii) 空间分布 `p_t` 随 τ 怎么变。
§9-68 量到 (i) **精确均匀**（质量最多的 25% τ 格只占 0.250）。
⇒ 模型「不建模时间」在参数化上是**说得通的**，问题在**它为什么学不出 (i)(ii)**。
`FieldHead` 的实测记录（plan 04 §9-17）已经给出了答案的方向：
**没有那条直连 skip 时，联合训练会停在「只学边际分布」的盆地**（在「把可见位置复制到输出」
这个最小任务上全参训 800 步仍停在边际解，而冻结解码器只训头部立刻到达下界）。
而 §9-62 把两条 skip **一起关掉**正是因为 `cell_skip` 给出的**无条件常数空间先验**有害——
于是模型被留在那个被记录过的盆地里，本轮的四条读数（平坦 τ 剖面、τ 置换代价 ≈ 0、
组内 τ AUC 0.50、val_ratio 打不赢 line-only）**全部是该盆地的签名**。

#### ④ 判据侧的新缺陷：**val_ratio 主要在抽「总质量」这签**

`val_pred_over_true = Σ∫λdV / Σn`（1.0 = 遮盖测度下的最优校准）。全量 20k run 的 20 个 val 点：

```
0.34 0.54 0.08 1.25 0.93 0.51 0.52 0.32 0.78 0.67
1.00 0.96 0.48 0.46 0.36 0.57 0.38 0.28 1.22 0.42
```

`val_ratio` 与之同步（`pred=0.0816 → ratio 1.0033`；`pred=0.9992 → ratio 0.7576`）⇒ **判据读数的方差
主要由一个标量（总质量）贡献**，而总质量在 1000 步内能摆 15 倍。同轮把 11 个 checkpoint 放在**同一批
40 个 val 批**上复测（`runs/_probe_mass.py`）：

| checkpoint | 质量 | NLL | ratio | τ 置换 Δ |
|---|---|---|---|---|
| cumskip best/4000 | 0.0015 | 1.3701 | 1.1765 | +0.000004 |
| cumskip 6000 | 0.0013 | 1.3733 | 1.1792 | +0.000049 |
| cumskip 8000 | **0.0003** | 1.6115 | 1.3837 | +0.000111 |
| 修复臂 4000/best | 0.9551 | 0.9046 | 0.7767 | −0.000052 |
| 修复臂 8000 | 0.5085 | 0.9769 | 0.8388 | +0.000019 |
| 长跑 best（8k 快照） | 0.6281 | 0.7831 | 0.6724 | +0.210015 |

**cum-skip 臂在 8000 步把质量压到 0.055（全 val 口径）、并把 `val_ratio` 顶到 1.0029**——
**破了 `val/ratio < 1` 的红线**（RFC-0037 R1）⇒ 该臂的扩规模结论作废（本轮按红线处理，不当作结论）。
同一现象也出现在别的臂上（修复臂 4000→8000 质量 0.955 → 0.509），只是没到破线。
⇒ **`best.pt` 由 `val_ratio` 选 = 在一个高度不稳的标量上抽签**；这是判据侧的独立缺陷，
与模型侧的问题叠加，使「哪条臂更好」在 4000 步尺度上**不可判**。

#### ⑤ 本轮的结论（可复述）

1. **目标函数没问题**：τ 值 42.5%、空间 58.6%、线 88.8% 的分数空间；
2. **模型没在用它**：四条 `head_skip=false` 臂的实际水平 ≤ `line_only`（0.7196），
   即输出只剩线密度，**时间与空间都没有净收益**；
3. **机制是「直连 skip 关掉后落回边际盆地」**（plan 04 §9-17 已记录过这个盆地），
   而关它的理由（`cell_skip` 的无条件常数先验）**早于 τ→秒对齐修复**（§9-64），从未在修复后的代码上复测；
4. **判据侧**：`val_ratio` 的方差主要来自总质量，且质量可以崩到破红线 ⇒ 它现在还不是一个可信的判据。

#### ⑥ 预登记的下一步（**跑之前写死**）

在**修复后的代码**上补齐 2×2：`head_cell_skip` × `head_cum_skip`（cum-only 已跑，见上）。
新臂一律 8000 步、`data.workers=0`、同一 val_every=4000。判据：

1. `head_skip=true`（两条都开）与 `head_cell_skip=true`（只开空间）两臂中，**至少一条**在 step 4000/8000
   的 `val_ratio` **≤ 0.7196**（打赢 per-line oracle）——若打不赢，则「skip 是逃生通道」的假设被否定，
   下一步必须走「学到的门控」而不是「开关」；
2. 两臂的 `val_pred_over_true` 全程落在 **[0.5, 1.5]**——质量崩溃复现则记为判据侧缺陷的第二次复现；
3. τ 置换 Δ 在 (2) 成立的 checkpoint 上 **> 0.01**。

臂：`--experiment probe_skipall model.head_skip=true`、`--experiment probe_cellskip model.head_cell_skip=true`。
**本轮交付时已排队**。
#### ⑦ 结果（本轮交付前拿到的部分）

**(a) 预登记判据 1 被否定：重开 skip 没有救回来。** `probe_skipall`（`head_skip=true`，两条都开）@step 4000：
`val_ratio = 0.9734`、`val_pred_over_true = 0.314` —— **是全部臂在 4000 步最差的一条**（其余 0.7636–0.8787），
质量也跳出 `[0.5,1.5]`。⇒ **「skip 是逃生通道」的假设被否定**，且 §9-62「skip 有害」的结论在**修复后的代码上复现**。
`probe_cellskip`（只开空间 skip）本轮未跑完（每步 0.7 s × 8000 ≈ 2 h，排在其后）。

**(b) 可见场基线（`runs/_probe_visible_baseline.py`，与模型**同一批 40 个 val 批**、const = 1.16460）**：
「把可见计数抄成密度 + 5% 均匀底 + 族最优档位」= **0.7440**；
同批上模型：修复臂@4000 **0.7767**、长跑 best **0.6724**、cum-skip@4000 **1.1765**。
⇒ 一条**不需要任何音乐理解**的基线落在模型同一区间（略优于修复臂@4000，略差于长跑 best）。
⚠️ **仪器存疑（必须如实登记）**：表中 4 个 `sigma_x` 行的读数与 `no_smooth` **逐位相同**（0.86648），
而 `smooth()` 单独测试是正确的（σ=1 把单位脉冲摊成 0.0044/0.054/0.242/0.399…）⇒ **空间平滑那一轴未生效，
原因未定**；本轮的结论只用 `no_smooth` 那一行（它不经过平滑）。
---

### §9-70 第二十轮：**取批计划的「每轮每谱第 k 窗」把训练分布换成了前奏切片（前 8000 步 61.5% 空窗）**（2026-09-30）

#### ① 决定性读数：训练步级分布 ≠ 语料分布（**训练日志本身，8000-40000 步实测**）

| 口径 | 空窗占比 | 事件/窗均值 | 事件/窗中位 |
|---|---|---|---|
| **步级（训练日志，batch_size=1）** | **0.615** | **4.40** | 0.0 |
| 窗口级（计划均匀抽样 2000 窗） | 0.105 | 13.46 | 12.0 |

⇒ **模型看到的世界比语料稀 3 倍、空 6 倍**。12 个 run 的日志给出同一个数（0.615 / 4.40，逐位相同）。

#### ② 机制：计划是**按轮次发牌**，第 k 轮发的全是各谱的第 k 窗

`plan_epoch` 的 S4 设计（RFC-0034）把每个桶内每张谱的窗口切成块，**按轮次逐行发牌**：第 k 轮
每张谱发它的第 k 块。于是：

* `coverage_at`：6 614 槽位覆盖 72% 谱面、13 228 槽位覆盖 100%（= 每谱 2 窗）⇒ **前 6 614 步
  只有每张谱的「第 0 窗」**，而真实谱面的第 0 窗多是**前奏空窗**；
* 前缀画像（`runs/_probe_plan_prefix.py`，真实 train split 逐窗读）：

| 前缀长度 | 空窗占比 | 事件/窗均值 | 中位 |
|---|---|---|---|
| 500 | 0.476 | 5.20 | 2.0 |
| 8000 | **0.615** | **4.40** | 0.0 |
| 20000 | 0.526 | 5.38 | 0.0 |
| 60000 | 0.351 | 7.68 | 6.0 |
| 全库均匀 | 0.105 | 13.46 | 12.0 |

⇒ **整个 epoch 是 634 952 窗 = 96 轮**，而本项目跑过的最长 run 是 40 000 步 = **6.3% 个 epoch**
⇒ **至今没有任何一次训练跑在语料的无偏切片上**。空窗在遮盖测度下（r=0 退化支）的最优 λ 恰为 0，
所以「每步都在推着输出变小」既解释了 §9-69 的**质量下漂**（每条臂都在掉）、也解释了 τ 平坦与
「越训练越不听音乐」——模型收敛到的是**一个大部分为空的分布**的最优解，而不是谱面的最优解。

#### ③ 判据侧的独立证据：可见场里的 τ 信息**很便宜**（`runs/_probe_tau_ceiling.py`）

11 维线性回归（可见场的多尺度 τ 剖面：原始 + 7 个尺度的平滑 + 线总量 + 窗总量 + τ 位置 + 截距，
256 个 train 窗拟合、全 val 求值，**不训练任何神经网络**）：

| 读数 | 值 | 对照 |
|---|---|---|
| (窗口,线) **组内 τ AUC** | **0.5723**（642 组） | **模型 0.5009**、纯 τ 先验 0.5555 |
| `val_ratio`（形状来自回归、总量取真值） | **0.7294** | 模型 0.76–0.90；`line_only` 0.7196 |
| `val_ratio`（总量也来自回归） | 0.8711 | —— |

⇒ **一个 11 参数的线性映射在 τ 定位上打赢 40M 参数的模型**（+0.07 AUC），说明信息在那里、
而且**很便宜**；同时它的 `val_ratio` 只到 0.73 ⇒ τ 定位**本身**买不到多少损失（大头在空间轴，
`space_flat` 族最优 0.4171）。两条合起来把矛头完全指向**训练分布**而不是表达能力。

#### ④ 修法（已实现、已验、默认关闭）：`data.plan_window_shuffle`

**每一轮每张谱发它自己的随机一块**（该谱种子化置换），轮转结构与覆盖率**逐条不变**
（仍是「每轮每谱一块」⇒ 全部谱面在约「谱面数」步内各被访问一次），只是任何前缀都成为
全库的无偏切片。真实数据实测（前 2000 窗）：

| shuffle | 空窗占比 | 事件/窗均值 | 中位 |
|---|---|---|---|
| False（旧行为） | 0.496 | 5.85 | 2.0 |
| **True** | **0.114** | **17.39** | 11.0 |
| 全库均匀抽样 | 0.105 | 13.46 | 12.0 |

⇒ 前缀从「一半是空窗」变成「与语料一致」。护栏 `tests/unit/data/test_plan_window_shuffle.py`（6 项：
两种模式都是排列、覆盖率不变、旧行为=逐位不变、新行为前缀不再只有第 0 窗、确定性/种子相关）。
**默认 False**（生产行为不变，避免在验证前改变语义）；打开后**必须**重测，见 ⑤。

#### ⑤ 预登记判据（**跑之前写死**）

臂：`probe_shuffle`，8000 步、`data.workers=0`、`data.plan_window_shuffle=true`，其余与 §9-69 的六条臂同。

1. **质量不再下漂**：step 4000 与 8000 的 `val_pred_over_true` 都落在 **[0.5, 1.5]**，且 8000 的值
   **不低于** 4000 的 0.6 倍（旧行为：修复臂 0.955→0.509、cum-skip 0.845→0.055）；
2. **打赢 per-line oracle**：至少一个 val 点的 `val_ratio` **≤ 0.7196**；
3. **τ 不再平坦**：τ 置换 Δ > 0.01（旧行为 ≈1e-4）。

任一条不成立 ⇒ 「训练分布非平稳是主因」被削弱，下一轮转向「表示/优化」那一侧（§9-69 ⑥ 的冻结头部对照）。
#### ⑥ 结果：**修法让 val_ratio 首次随训练变好**（本轮交付时的实测）

| 读数 | 旧行为（同预算 8k） | **`plan_window_shuffle=true`** |
|---|---|---|
| 步级空窗占比 / 事件每窗 | 0.615 / 4.40 | **0.122 / 14.26**（全库 0.105 / 13.46） |
| `val_ratio` @4000 | 0.7636–0.8787 | **0.7857** |
| `val_ratio` @8000 | **0.8586–1.1316**（六条臂全部变差） | **0.7324**（**变好**） |
| `val_pred_over_true` 4k→8k | 0.955→0.509、0.845→0.055 等 | **1.147 → 0.523** |
| 谱面覆盖率 @8000 槽位 | 5843 / 6614（88.3%） | 5803 / 6614（87.7%，不变） |
| 门禁批事件数 | 63 | **304** |

**判据核对**（§9-70 ⑤）：① 质量全程落在 [0.5,1.5] **成立**（1.147 / 0.523），但「8k ≥ 0.6x 4k」
**不成立**（比值 0.46 —— 仍在下漂，只是没崩）；② `val_ratio <= 0.7196` **未达成**（8k = 0.7324，
差 1.8%）；③ τ 置换 Δ 见 `runs/probe_shuffle_out.txt`。
⇒ **方向成立、幅度差一点**：这是本项目**第一次**看到「训练越久 val 越好」，也是 8k 预算下的最好读数
（旧最好 0.8137）。**但「六条旧臂对比」现在全部作废**——它们训练在偏置切片上（§9-70 ①）。
#### ⑦ 判据 ③ 的实测与**结论的拆分**（同批 40 个 val 批，与 §9-69④ 完全同口径）

| checkpoint | 质量 | NLL | ratio | **τ 置换 Δ** |
|---|---|---|---|---|
| 修复臂@4000（旧采样） | 0.9551 | 0.9046 | 0.7767 | −0.000052 |
| 修复臂@8000（旧采样） | 0.5085 | 0.9769 | 0.8388 | +0.000019 |
| **shuffle@4000** | **1.5447** | 0.9158 | 0.7864 | −0.000240 |
| **shuffle@8000（=best）** | **1.1450** | **0.8117** | **0.6970** | **−0.000021** |

⇒ **判据 ③（τ 置换 Δ > 0.01）不成立**：训练分布修好之后模型**仍然对 τ 完全平坦**。
⇒ **本轮把「模型学不会音乐」拆成了两个独立的问题，并解决了其中一个**：

1. **质量/校准问题 —— 已解决，根因就是取批计划的发牌顺序**。证据链完整：训练分布 6x 空 / 3x 稀
   （①）⇒ 每条臂的质量下漂（§9-69④）⇒ 修好后质量 1.145（旧 0.509）、val_ratio 首次随训练变好、
   8k 最好读数 0.7324（旧最好 0.8137）。**这条是真正的「有效方案」，且已实测**。
2. **τ 定位问题 —— 仍未解决，且已排除训练分布这一项**。同样的数据、同样的预算，11 维线性回归
   能到组内 AUC **0.5723** / val_ratio 0.7294，而模型是 **0.5009 / 0.7324** ⇒ 剩下的候选只有
   **输入表示**与**优化路径**（§9-69⑥：冻结解码器 + 只训头部与输入投影的对照，下一轮第一件）。
---

### §9-72 第二十二轮：**遮盖信号在输入里是**完美可分**的（1.0000），是**解码器**把它变成随机（0.5057）**（2026-09-30）

#### ① 先更正上一轮：§9-71⑤ 的「崩塌在 embedding」**被本轮否定**

§9-71⑤ 的 L1.5 = 0.4790 是用**岭回归拟合稀有标签**（val 正例 0.03%）得到的，在 256 维上只有几百个正例
⇒ 读数可能是**欠拟合**而非「信息没了」。本轮换成**零参数**的检验（`runs/_probe_embedding_sep.py`）：

    方向 = mean(token | 全遮盖) - mean(token | 全可见)，只用**前一半** val 估；
    分数 = token·方向，在**后一半** val 上算 AUC（目标标签 = 「本 token 是否全遮盖」，29.97% 正例，统计充分）。

| 一级 | 同半边 AUC | **跨半边 AUC** |
|---|---|---|
| L1 输入特征 | 1.0000 | **1.0000** |
| E1 `visible` 分量 | 0.7274 | 0.4984 |
| E2 `occlusion` 分量 | 1.0000 | **1.0000** |
| E3 LayerNorm **之前** | 1.0000 | **1.0000** |
| **E4 LayerNorm + 音频之后（= 解码器输入）** | 1.0000 | **1.0000** |
| **L2 解码器输出** | 0.6393 | **0.5057** |
| L3 头输出（每 token λ 质量） | 0.7286 | 0.6546 |

⇒ **遮盖信号一路完美线性可分到解码器输入（1.0000），在解码器里掉到 0.5057（随机）**。
**崩塌在解码器，不在 embedding**；而且头输出仍保留 0.6546 的遮盖意识（说明解码器不是「删了」，
而是**非线性地混掉了**，训练过的头还能挤出一点）。

#### ② 输入表示的**尺度失衡**（本轮最有解释力的一条读数）

同一次前向里量到的各分量范数（全体 token 平均）：

| 分量 | 范数 |
|---|---|
| `visible` 分量（可见计数投影） | **0.2809**（且对全遮盖 token 它**恒等于偏置**） |
| `occlusion` 分量（遮盖通道投影） | **37.6518** |
| 位置 + 线嵌入 | 19.4230 |
| 音频项 | 12.1120 |
| LayerNorm 之前 / 之后 | 50.9255 / 19.5928 |

⇒ **承载「谱面内容」的那一支比遮盖通道小两个数量级**（0.28 vs 37.65）。模型在 token 里几乎只看得见
「这一段被遮住了没有」，而看不见「这一段里有哪些音符」。这解释了：

* 为什么模型只学到边际分布（§9-69④）——内容信号接近于噪声水平；
* 为什么 §9-62 的 `cell_skip` 对遮盖 token 只能给出常数（它的输入是 `visible≡偏置` + `occlusion≡常数`）；
* 为什么「重开 skip」三条臂全部失败（§9-69⑦）——那条短路的输入本身就没有内容。

#### ③ 判据侧：**最优 λ 在未遮盖 token 上恒为 0**

事件项只监督**被遮盖**处，而积分项对任何非零强度都是纯成本 ⇒ 一个只用遮盖通道的场
「λ = 档位 · 1[该 token 全遮盖]」是**不需要任何音乐理解**就能做到的第一步。它的读数见
`runs/_probe_mask_only.py`（本轮在跑）；这是「模型本该轻松拿到、却没拿到」的那部分分数。
#### ④ **方法学发现：同配置重跑的差值达 0.05 val_ratio**（本轮意外抓到，影响所有臂对比）

`probe_shuffle24k` 因卡死重跑了一次，两次的 **`config.yaml` 逐字段相同**（种子、数据顺序、
`plan_window_shuffle=true`、`workers=0`、`val_every=4000`、`max_steps=24000`），步级分布也一致
（空窗 0.12 / 事件 14.57）：

| 运行 | step 4000 `val_ratio` | 质量 |
|---|---|---|
| 第一次（20260929-235303） | **0.7492** | 0.4715 |
| 第二次（20260930-005123） | **0.8005** → step 8000 **0.7287** | 0.6841 → 0.5502 |

⇒ **同配置重跑在 4000 步处的差值是 0.051**（唯一可能的来源是 GPU 非确定性：bf16 autocast + SDPA 核）。
**这条改写了臂对比的判读口径**：

* 《§9-70⑥ 修法有效》的三条证据里，**分布与质量的差异是量级性的、可信的**（空窗 0.615 -> 0.122、
  质量从「腰斩/崩溃」变成「稳定」），但 **`val_ratio` 0.7324 vs 0.8586 这个具体差值被噪声污染**；
* 今后任何「A 比 B 好 0.0x」的结论**必须给出重复跑**（至少 2 次同配置），否则不予采信；
* 这也解释了历史上多次「同预算两条臂差 0.03-0.05」的不稳定观感（§9-64 只记了 `data.workers` 的可比性）。

#### ⑤ 判据侧（在跑）：遮盖感知场的读数

**最优 λ 在未被遮盖的 token 上恒为 0**（事件项只监督被遮盖处，积分项对任何非零都是纯成本）⇒
「照着遮盖通道放质量」是**不需要任何音乐理解**就能做到的第一步。
`runs/_probe_mask_only.py` 量它值多少：场 = 档位 ·（1[全遮盖 token] + 可见计数）——
**只用输入即可构造**，且在退化批（r=0，损失走全事件口径）上也覆盖被监督的位置。
本轮读数待该探针跑完（前两版分别栽在 τ 轴长度不齐与退化批 inf 上，均已修）。
**读数（`runs/_probe_mask_only.py`，档位用族最优闭式，**自检：`line_only` 复现 §9-69 的 0.7196 逐位一致**）**：

| 场（都不需要任何音乐理解） | NLL | 除以 const |
|---|---|---|
| `const` | 3.94133 | 1.0000 |
| `line_only`（只知每条线总量，**oracle**） | 2.83622 | **0.7196** |
| **`mask_plus_visible`**（1[全遮盖] + 可见计数） | 2.67727 | **0.6793** |
| **`mask_event`**（1[全遮盖]×(可见计数+1) + 可见计数） | 2.39105 | **0.6067** |

⇒ **一条只读自己遮盖通道、零音乐理解的场拿到 0.6067**，**打赢 per-line oracle（0.7196）**，
也**打赢每一个训练出来的模型**（0.73–0.90）。

**⇒ 本轮把链条闭合了**：
1. 遮盖信号在模型自己的输入里**完美线性可分**（跨半边 AUC **1.0000**）；
2. **只靠它**就能把 `val_ratio` 打到 **0.6067**（模型 0.73–0.90）；
3. 而它在**解码器**里被抹成随机（**0.5057**）。
---

### §9-73 第二十三轮：**只训头部（冻结解码器）能不能把遮盖通道用起来**（判据预登记）

#### ① 为什么要这一刀

§9-72 把现场钉在解码器：遮盖信号在**解码器输入**处跨半边 AUC **1.0000**，在**解码器输出**处 **0.5057**。
但「解码器表达不足」与「解码器没被优化到」是**两种**解释，而 plan 04 §9-17 早就记过同一现象：
**冻结解码器只训头部立刻到达下界，全参训却停在边际解**。把那个对照搬到真实数据上，就能二分：

* 头 + 直连 skip **能**学到 ≈ 遮盖基线 ⇒ 表达与头都没问题，问题在**联合优化**（解码器把输入混掉了）；
* 头 + skip **学不到** ⇒ 问题在头/损失本身，与解码器无关。

#### ② 新开关 `optim.train_head_only`（默认 False，语义字段）

冻结除 `model.head` 以外的一切参数，优化器只吃可训参数；**门禁走自己的优化器，不受影响**
（`gates` 的 `optimizer_for(model)` 一直是全参）。护栏 `tests/unit/infra/test_train_head_only.py`（3 项：
只有 head 可训 / 优化器只见 head 且参数变少 / head 占比 > 5%）。

#### ③ 预登记判据（**跑之前写死**）

臂：`probe_headonly`，2000 步，`optim.train_head_only=true` + `model.head_cell_skip=true` + `head_cum_skip=true`
（= 头直接读 `input_features`，其中含遮盖通道），`plan_window_shuffle=true`，`val_every=1000`。

1. **主判据**：`val_ratio` 在 1000/2000 步任一点 **≤ 0.70**（遮盖基线是 0.6067，per-line oracle 0.7196）；
2. `val_pred_over_true` 全程 ∈ **[0.5, 1.5]**（质量不崩）；
3. 若两条都成立 ⇒ 结论 =「头与 skip 有能力，联合优化是瓶颈」，下一轮改**两段式训练**（先冻结训头，再解冻微调）；
   若 1 不成立 ⇒ 结论 =「头/损失本身解不出」，下一轮改**损失与参数化**（例如直接预测遮盖指示器 × 密度）。
#### ④ 结果（第一部分）：**判据 1 不成立**，头没有「立刻」解出来

`probe_headcum`（`optim.train_head_only=true` + `model.head_cum_skip=true`，1500 步，一次 val）：

| 读数 | 值 | 判据 |
|---|---|---|
| `val_ratio` @1500 | **0.7665** | **≤ 0.70 ✗**（遮盖基线 0.6067、per-line oracle 0.7196） |
| `val_pred_over_true` | 0.7208 | ∈ [0.5,1.5] ✓ |

⇒ 头握着一条**到遮盖通道的直连线性通路**（`cum_skip: 2560 -> 1`，要学的只是「全遮盖 token 的 8 个遮挡通道之和」），
1500 步后只到 **0.7665** —— 既没到遮盖基线 0.6067，也没到 per-line oracle 0.7196。
**这否掉了「头一训就会」的天真版本**，但也可能是步数不够（该臂步时只有 0.031 s ⇒ 8000 步只要 4 分钟，
已加长重跑，见下）。

#### ⑤ 同轮的运行侧发现：**`head_cell_skip` 的 val 会跨墙**

第一版用 `head_cell_skip=true`（`cell_skip: 2560 -> 1280` 每 token）跑 2000 步：
训练步时正常（0.047 s），但 **step 1000 的 val 卡了 20 分钟以上**（GPU 100%、6.5 GiB、功耗 48.7 W，
既不报错也不前进）——正是 §9-59 记过的「val 跨墙只变慢不崩」。已强杀并改用**只开 `cum_skip` 的等价廉价变体**
（同一个「有没有把遮盖用起来」的问题，参数量少三个数量级）。**运维结论：诊断臂要挑参数矩阵最小的那条通路。**
---

### §9-74 第二十四轮：**模型的 λ 对遮盖通道毫无响应（44.4% vs 44.2%），而在被监督集合内 τ AUC = 0.5130**（2026-09-30）

#### ① 基线阶梯补齐：**完全无 oracle** 的那一条是 0.7594

`runs/_probe_mask_only.py`（档位用族最优闭式；自检 `line_only` 复现 §9-69 的 0.7196 逐位一致）：

| 场 | 形状来源 | 档位来源 | 除以 const |
|---|---|---|---|
| `mask_pure` | **只用输入**（1[全遮盖] + 可见计数） | **只用输入**（2×可见 + 0.05） | **0.7594** |
| `line_only` | 常数 | oracle 每线总量 | 0.7196 |
| `mask_plus_visible` | 只用输入 | oracle 每线总量 | 0.6793 |
| `mask_ratio` | 含遮盖内事件数（oracle） | 只用输入 | 0.6867 |
| `mask_event` | 含遮盖内事件数（oracle） | oracle | 0.6067 |

⇒ **一条两行的、只用输入构造的场拿到 0.7594**；而模型的**最好**读数是 0.7287（24k@8k）/ 0.7324（8k 臂）
⇒ 模型**只比这条启发式好一点点**，而离 oracle 族（0.42 / 0.26）仍然极远。
⚠️ 这条也修正了 §9-73⑥ 的口径警告：把 oracle 部分全部拿掉之后，「够得着的下界」是 **0.7594 而不是 0.6067**。

#### ② 决定性读数：λ 的质量**与 token 数成正比**（对遮盖零响应）

`runs/_probe_mask_response.py`（armH 的 8k 快照，48 个 val 批）：

* 质量分布：**被遮盖 token 44.4% / 未被遮盖 token 55.6%** —— 而 token 占比是 44.2% / 55.8%
  ⇒ **模型把质量按 token 数均匀撒开，对遮盖通道完全没有响应**；
* 最优解要求未被遮盖处 λ **恒为 0**（事件项只监督被遮盖处，积分项对任何非零都是纯成本）
  ⇒ 模型把 **55.6% 的质量花在了不收任何事件项回报的地方**；
* **在被遮盖 token 内部（= 真正被监督的子集）组内 τ AUC = 0.5130**（对照：未被遮盖内部 0.4853）
  ⇒ 这是「τ 平坦」的最锋利版本：把考察范围收窄到损失真正监督的那批 token，仍然是随机。

#### ③ 由 ② 直接推出的干预：`model.mask_lambda`（λ ← λ ⊙ occlusion）

既然最优解在未被遮盖处**恒为 0**，那就是一条**不引入任何新参数、也不改损失语义**的硬约束
（只是把「已知为 0 的那些格子」钉成 0）。护栏 `tests/unit/generation/test_mask_lambda.py`（3 项：
开关确实改输出 / 未被遮盖处恰为 0 且被遮盖处未被清零 / **全遮盖时必须是恒等**（否则推理第一步被改坏））。

**预登记判据**（臂 `probe_masklam`，8000 步，`plan_window_shuffle=true` + `model.mask_lambda=true`）：
1. `val_ratio` ≤ **0.72**（打赢 `line_only` 0.7196 附近；它已经是「不许浪费质量」的最低要求）；
2. 质量分布应变成「被遮盖 100%」（构造性保证，用 ② 的探针复核）；
3. 被遮盖内部组内 τ AUC **> 0.55**（若仍 ≈0.51，说明浪费质量只是**并发症**，不是主因）。
#### ⑤ 第一次跑就被门禁抓住：**全可见批会把事件项打成 +inf**

首版 `mask_lambda` 不带守卫，门禁直接 **退出码 7：`lambda 出现 NaN`**（`runs/probe_masklam/20260930-032246`）。
原因：G3 门禁用 `source.batch(masked=False)`（**全可见**批），损失走**全事件**口径 ⇒ 监督的是全部事件，
而 `lam * occlusion` 把所有格子清零 ⇒ `-n·log 0 = +inf` ⇒ 反传出 NaN。**门禁在这一步替我们拦住了语义错误**
（这正是 fail-closed 该有的样子）。修法：守卫 `occlusion.any()` —— 全可见批退回恒等（全事件口径下本来就该如此），
全遮盖批乘法是恒等（生成制度不受影响），两条边界都对称。护栏加到 **4 项**（新增「全可见批必须是恒等」）。
#### ⑥ `probe_masklam` 的结果：**判据 ① 达成（史无前例），判据 ③ 失败**

| step | `val_ratio` | `val_pred_over_true` |
|---|---|---|
| 4000 | **0.6890** | 0.7037 |
| 8000 | **0.6984** | 0.5925 |

对照（同一 val 口径）：本项目此前**最好**的模型读数 0.7287（24k@8k）/ 0.7324（8k 臂）；
`line_only` 0.7196；**完全无 oracle** 的 `mask_pure` 0.7594；oracle 族 0.4171 / 0.2559。

⇒ **判据 ①（≤0.72）达成，且是本项目历史上第一次「一条臂同时打赢全部模型与全部启发式基线」**；
两个 val 点稳定（0.689 / 0.698），质量都在 [0.5,1.5]（判据 ② 由构造保证）。

**判据 ③ 失败**：`runs/_probe_mask_response.py --mask-lambda` 在 `step-0008000.pt` 上量到
**被遮盖 token 内部组内 τ AUC = 0.4992**（原 0.5130，噪声范围内）⇒ **模型仍然分不出「哪个被遮盖的 token 有事件」**。
（该探针同时报「质量 73.3% 在被遮盖处 / 26.7% 在未被遮盖处」——后者不是矛盾：全可见批（空窗、退化窗）
被 `occlusion.any()` 守卫退回恒等，而该探针取的**前 48 个 val 批恰好是稀疏/空窗层**，这是口径产物，
不是 `mask_lambda` 失效。）

#### ⑦ 本轮结论（对目标的意义）

1. **`model.mask_lambda` 是一个真实有效的修法**（零参数、不改损失语义，把「最优解已知为 0 的格子」钉成 0），
   它是本项目**第一次**让一条臂同时打赢全部模型与全部输入启发式基线（0.689/0.698）。
2. **但它与「学会音乐」正交**：被监督集合内的 τ 定位**没有任何改善**（0.4992）。
   ⇒ **「浪费质量」是并发症，不是主因**（判据 ③ 的预登记分叉走的是这一支）。
3. 因此主因仍然指向：**模型无法在被遮盖的那批 token 内部把事件定位出来**——而输入里唯一的 τ 信号
   是遮盖指示器（§9-72①），它在解码器输出处已线性不可分（0.5057）。**下一轮应当直接攻这条链**：
   （a）给解码器加**跨 token 的显式通路**（例如让场编码器直接看到相邻 token 的可见计数与遮盖状态的
   局部聚合），（b）或改成「先预测每个被遮盖 token 的事件数、再预测其内部空间分布」的两段式参数化——
   两者都必须在**被遮盖内部 AUC** 上验收，而不是 `val_ratio`。
#### ⑥ 加长后的结果：**平台在 0.77，没有向 0.61 靠**

`probe_headcum8k`（同配置、8000 步预算、`val_every=2000`）：

| step | `val_ratio` | `val_pred_over_true` |
|---|---|---|
| 1500（另一条同配置短跑） | 0.7665 | 0.7208 |
| **2000** | **0.7712** | 0.5616 |

⇒ 头握着**到遮盖通道的直连线性通路**、解码器完全冻结（不干扰梯度），平台仍停在 **0.77**。

⚠️ **但必须把对照的口径说清（否则这条结论会被高估）**：§9-72⑤ 的 `mask_event = 0.6067` 用的是
**族最优档位**（`amp = s·N_supervised / Σshape`）——那个「每条线被遮盖总量」是 **oracle**。
头拿不到它，只能从可见量推 ⇒ **0.6067 与 0.7665 之间的差距里，有一部分是 oracle 档位贡献的**，
不能全部算成「头没学会」。**下一轮必须先量出「无 oracle 档位」的同形状场**（例如把档位固定成
`可见总量/(1-r)` 的闭式估计），那条读数才是模型真正够得着的基线。

⇒ 本轮的**稳健结论**只有两条：
1. 遮盖信号在输入里**完美可分**（1.0000），在解码器输出处**线性不可分**（0.5057）；
2. **只训头部 + 直连 skip 也到不了「早就可表达」的解**（0.77 vs 可表达的下界），
   即问题不止在解码器的梯度路径上——**损失/优化本身也没把可达解找出来**。

#### ⑦ 同一轮对 §9-72④ 的**更正**：24k 的 val 序列**不是单调改善**

第二次 24k 跑到 12 000 步：**4k 0.8005 → 8k 0.7287 → 12k 0.7890**。
⇒ 结合同配置重跑差值 **0.051**，正确说法是「**在 0.73–0.80 的带里波动**」，
而不是「越训越好」；对照组（旧采样八条臂）则是在 8k 处集体进入 **0.83–1.13** 并伴随质量崩溃。
**修法的证据仍然是量级性的**（分布与质量），但**趋势性结论必须收回**。
**加长跑跑完（全序列）**：`probe_headcum8k` —— 2000 **0.7712** / 4000 **0.7633** / 6000 **0.7457（最好）** /
8000 **0.7947**；`val_pred_over_true` 依次 **0.56 / 1.84 / 1.31 / 0.30**。
⇒ 正确说法不是「头解不出」，而是「**头这条路下降极慢、且质量会过冲**」：
    头单独训 6000 步只买到 0.03（0.7712→0.7457），而全参训同预算能到 0.73 上下。
    **这条读数不支持「换两段式训练就能解决」**，下一轮的量对照（无 oracle 档位）仍然必须先做。
    后段在 0.7457–0.7947 之间反弹、质量从 1.84 甩到 0.30 ⇒ **与全参训练同一种不稳**，只是更慢。

⇒ **模型连「照着遮盖通道放质量」这件不需要音乐的事都没做到，卡点在解码器。**
（⚠️ 遮盖是**协议产物**不是音乐：会用遮盖 ≠ 学会音乐；但它是**前置能力**，前置能力都拿不到时，
「学不会音乐」的现场就必然在这一层。）
---

### §9-71 第二十一轮：**τ 信息在输入里（AUC 0.97），到解码器输出就没了（0.498）**（2026-09-30）

#### ① 逐级读数（`runs/_probe_stage_auc.py`，同一批 train/val 窗口、同一目标、同一岭回归口径）

目标 = 每 (窗口,线,τ) token 的**被遮盖质量**；指标 = (窗口,线) **组内 τ AUC**；
L0/L1/L2 都是**在同一套 train 窗上拟合**的线性读出，L3 不需要拟合（就是模型自己的输出）。

| 一级 | 维度 | 组内 τ AUC |
|---|---|---|
| L0 输入：可见场多尺度 τ 剖面 | 11 | 0.5583 |
| **L1 token 输入特征（可见场 + 遮盖通道）** | 2560 | **0.9687** |
| **L2 解码器输出 token** | 256 | **0.4983** |
| **L3 模型头输出（每 token 的 λ 质量）** | 1 | **0.4970** |

⇒ **信息在输入里几乎是完备的（0.97），到解码器输出就掉到随机（0.498）**，头输出同样是随机（0.497）。
**这就是「模型学不会音乐」在本轮被定位到的那一级**：不是目标函数（§9-69 已排除）、不是训练分布
（§9-70 已排除）、也不是「可见场里没有信息」（0.97 证明有），而是**可见场 -> 解码器表示这一段把 τ 抹平了**。

⚠️ **口径说明（防止把 L1 读成「模型表示」）**：L1 是**原始输入特征**（稀疏计数 + 遮盖通道），
线性读出它等于在问「输入里有没有 τ 信息」；L2 才是**模型的表示**。因此结论应当读成
「嵌入 + 解码器这一段把 τ 丢掉了」，而不是「模型表示在 L1 上很强」。

#### ② 与既有读数的关系

* §9-68：模型的 λ 沿 τ **精确均匀**（质量最多的 25% τ 格只占 0.250）、τ 置换代价 ≈ 0；
* §9-69：目标函数在 τ 上值 **42.46%** 的常数基线（所以不是「不值得学」）；
* §9-70：修好训练分布后 τ 置换 Δ 仍是 −2e-5（所以不是「训练分布」）；
* 本轮：**输入有 0.97、表示只有 0.498** ⇒ 只剩「嵌入/解码器这一段」与「优化路径」两种解释，
  而两者的分界线就是**解码器之前/之后**这一个读数。

#### ③ 下一刀（**预登记**，本轮已排队的一部分）

1. 在 L1 与 L2 之间补一级 **L1.5 = 过完 `model.embedding`（+ 音频注入）之后的 token**：
   若 L1.5 已经 ≈ 0.50 ⇒ 崩塌发生在**输入投影/嵌入**（1280->256 的 `nn.Linear` + LayerNorm）；
   若 L1.5 仍 > 0.9 ⇒ 崩塌发生在**解码器栈**（局部/全局注意力 + 残差）。
   **这是唯一还需要的一次前向**，下一轮第一件。
2. 若崩塌在嵌入 ⇒ 直接把「输入投影的尺度」当嫌疑：可见场是**稀疏计数**（每 token 约 11/1280 个非零格），
   线性投影后的范数远小于位置编码/线嵌入 ⇒ 逐 token 的 L1 归一化（或改成多尺度密度输入）是候选修法。
3. 若崩塌在解码器 ⇒ 查局部层的 band mask 与全局层的 K*T 注意力是否把 τ 平均掉了。


#### ⑤ **更正（同日）：① 的 L1/L2 曾用错输入口径，本节数字以本修正版为准**

`MaskedFieldModel.forward` 用的是 `batch.observed_counts()`（= counts * ~occlusion），而 ① 的探针
把**原始 counts** 当可见场喂了进去 ⇒ 等于把答案送进输入，L1 的 0.9687 与 L2 的 0.4983 **作废**。
修正后（`--cpu --train 96 --val 60`，同一批口径；新增 L1.5 = 过完 `model.embedding` 之后的 token，
以及「本 token 是否全遮盖」这条平凡基线）：

| 一级 | 维度 | 组内 τ AUC |
|---|---|---|
| L0 输入：可见场多尺度 τ 剖面 | 11 | 0.5538 |
| L1 token 输入特征（可见场 + 遮盖通道） | 2560 | **0.7086** |
| **L1.5 过完 `model.embedding`（+音频注入）之后的 token** | 256 | **0.4790** |
| L2 解码器输出 token | 256 | 0.4935 |
| L3 模型头输出（每 token λ 质量） | 1 | 0.4849 |
| 平凡基线：**本 token 是否全遮盖**（1 维） | 1 | **0.7086** |

**两条结论（修正版）**：

1. **输入里唯一线性可读的 τ 信号就是「遮盖指示器」**：L1（2560 维）的 AUC **逐位等于**那条 1 维
   平凡基线（**0.7086 = 0.7086**），而可见场自身只有 0.5538（= τ 先验量级）⇒ 除了**遮盖协议本身的
   泄漏**，输入里没有别的 τ 信息。
2. **崩塌发生在 `model.embedding` 内部，不在解码器**：L1 0.7086 -> **L1.5 0.4790** -> L2 0.4935 ->
   L3 0.4849 ⇒ 过完嵌入（1280->256 的 `nn.Linear` + 遮盖通路 + 正弦位置编码 + 线嵌入 + LayerNorm）
   之后信息就没了，解码器只是**保持**了「没有信息」这个状态。

#### ⑥ 同轮的结构性事实：遮盖是**逐 τ token 全有全无**（这解释了 §9-62 的「常数先验」）

`runs/_probe_mask_protocol.py`（全 val、纯 CPU）：

* token 总数 4 225 728：**全遮盖 44.22% / 全可见 55.78% / 部分遮盖 0.0000%**；
* 有事件的 token 2 945 个**全部是全遮盖**；有可见事件的 token 1 958 个全部是全可见（**交集 0**）；
* `P(token 有事件 | 全遮盖) = 0.0016`，而全谱「事件 token / 全部 token」= 0.00116 ⇒ 泄漏比 **1.36x**
  （远不是退化，但确实不是 0）；「全遮盖」这条指示器单独就有 **AUC 0.7086**。

**为什么这是本轮最有用的一句话**：被遮盖 token 的场输入**恒为常数**（可见场全 0 + 遮盖通道全 1），
所以：(a) 该 slot 的**空间**信息在输入里根本不存在，模型只能靠上下文与先验；
(b) §9-62 量到的 `cell_skip`「对被遮盖 token 给出常数先验、与真值反相关」**由此完全解释**——
对全遮盖 token，skip 的输入就是一个常数向量，线性层只能吐出常数；
(c) 「模型只学到边际分布」也由此解释：对一个输入恒定的 token，最优策略本来就是输出边际。
⚠️ 该设计**是本项目已知并有意的**（`generation/masks.py` docstring 明写 `token_block=True` 是为了
抑制 `mask_leak==1.0` 的退化），本轮的增量是：**它在 τ 定位这个判据下的代价是「输入里再没有别的 τ 信息」**。

#### ④ 同轮的第二件事：修法下的长跑（判「质量那一半是否持续改善」）

`probe_shuffle24k`（24000 步，`plan_window_shuffle=true`，`val_every=4000`）本轮已排队开跑。
判据：`val_ratio` **持续下降**（8k 时 0.7324，旧六条臂在 8k 之后全部反弹到 0.83–1.13）。
#### ⑦ 长跑的结局（如实记账）

* `probe_shuffle24k` 的 **step 4000 val = `val_ratio 0.7492`、质量 0.4715** —— **本项目最好的 4000 步读数**
  （旧最好 cum-skip 0.7636；同族 8k 臂在 4k 是 0.7857）；
* 但它**在 step ~5826 卡死**：进程 CPU 增量 0/20 s、`nvidia-smi` 利用率 0%、功耗 3 W、显存仍占 1.8 GiB，
  自 08:17 起 20 分钟无新日志 ⇒ 已强杀（显存回 0）。**结论未取得，下一轮必须重跑**。
* ⚠️ **本轮新登记的运维教训**：那一刻我同时在跑一个**吃内存的 CPU 探针**（L1 的每批特征矩阵约 0.8 GiB，
  8 线程）⇒ 与训练的 8 GiB 叠加很可能触到 commit 上限。原有纪律只管「同一时刻一个 **GPU** 作业」，
  现补一条：**与训练并行的任何探针都必须显式限内存**（或干脆等训练结束）。
**(c) `probe_skipall` @step 8000：`val_ratio = 1.1316`、质量 **0.060** —— 又一条破红线的臂。**
至此 skip 的三种开关（全关 0.8137→0.9030、仅 τ 0.7636→**1.0029**、全开 0.9734→**1.1316**）**没有一条**在 4000 步
打到 `line_only` = 0.7196，而**质量在每条开了 skip 的臂里都崩到 0.06**（全关的臂只是腰斩到 0.51）。
⇒ **「skip 是逃生通道」与「某个开关拧错了」两条假设都被否定**；现象是**输出场的整体尺度随训练下漂**，
叠加在「模型只学到线密度」之上。`probe_cellskip`（只开空间 skip）本轮**主动停掉**（避免无人值守的 2 h GPU 作业，
见 docs/TRAINING.md §7.5），留作下一轮第一件。

#### ⑧ 矩阵（修复后的代码，8000 步，全部同一 val 口径 const = 3.94133）

| 臂 | skip | 4k ratio | 8k ratio | 质量 4k→8k | τ 置换 Δ |
|---|---|---|---|---|---|
| 修复臂 fix2 | 无 | 0.8137 | 0.9030 | 0.955→0.509 | ≈0 |
| 对照臂 ctl2 | 无（旧码） | 0.7740 | 0.8586 | 0.627→0.424 | ≈0 |
| 生成制度 | 无 + 盲 | 0.8787 | 0.8326 | 0.438→0.224 | ≈0 |
| 音乐-only | 无 + 盲 + 无轨 | 0.8754 | 0.9116 | 0.854→0.207 | ≈0 |
| cum-skip | 仅 τ | **0.7636** | **1.0029** | 0.845→0.055 | ≈0 |
| skip-all | 两条 | 0.9734 | **1.1316** | 0.314→0.060 | — |
| —— 参照：`line_only` 族最优 | — | **0.7196** | — | — | — |

⇒ **六种配置没有一条打到 per-line oracle**；**每一条臂的质量都随训练下漂**（8k 时 0.06–0.51）。
这就是「越训练越不听音乐」在判据上的形状：不是音频权重掉了，而是**整个输出场在缩**。

---

### §9-75 第二十五轮：**被遮盖内部的可达天花板只有 0.5753（纯 τ 先验就有 0.5555）**（2026-09-30）

#### ① 只用输入的线性读出能到多少（`runs/_probe_within_masked_ceiling.py`）

特征 = 任何模型都能从可见场算出来的 11 维：可见 token 计数沿 τ 的多尺度平滑（σ=1…64）+ 本线可见总量
+ 全窗可见总量 + τ 位置。岭回归在 256 个 train 窗上拟合（目标 = 被监督的被遮盖质量），在 val 上评估：

| 口径 | 组内 τ AUC |
|---|---|
| **被遮盖 token 内部（= 模型验收口径 §9-74）** | **0.5753**（638 组） |
| 全体 token（旧口径，对照） | 0.5729（642 组） |
| 模型（armH 8k / `mask_lambda` 8k） | 0.5130 / **0.4992** |
| **纯 τ 先验**（§9-67） | **0.5555** |
| onset 前端（§9-67，从原始音频重算） | 0.5792 |

⇒ 两条结论，方向相反但都重要：

1. **模型落在纯 τ 先验之下**（0.499 vs 0.5555）——它连「同一 (窗口,线) 内事件在 τ 上的**平均形状**」
   都没学出来，而那是**零信息**就能拿到的下限；
2. **可达天花板也只有 0.5753**，只比先验高 **+0.020**（onset 前端 0.5792 = +0.024，同量级）
   ⇒ **τ 定位这件事本身的信息量极小**：§9-69 里那 42.46% 的 `gap_tau` 是 **oracle 信息**换来的，
   **不是模型够得着的**。
#### ② 结构性的推论：**逐 token 遮盖把「空间分布」也一起拿走了**

把 §9-69 的族最优分解与 §9-71⑥ 的遮盖协议放在一起看：

* `space_flat`（**τ 边缘 oracle**、空间均匀）= **0.4171** ⇒ 光把「哪个 τ 有几个音」做对就值 0.30；
* `tau_flat`（**空间 oracle**、τ 均匀）= **0.2559** ⇒ 光把「落在哪个 x」做对也值 0.46；
* 而**被遮盖 token 的可见场恒为常数**（§9-71⑥ 实测：部分遮盖 0.0000%）⇒ 对模型被要求预测的那些 token，
  **token 内的空间分布在自己的输入里根本不存在**，只能来自先验；τ 侧的可达信息又只有 +0.020 AUC。

⇒ **本轮对目标的结论（新）**：`val_ratio` 的地板主要不是「模型学不会」，而是**输入表示在两件事上都不含信息**：
被遮盖 token 的**空间**分布（结构性缺失）与**时间**分布（天花板 0.575 vs 先验 0.5555）。
§9-69 里那些吓人的 oracle 数字（0.42 / 0.26）是**信息论上界**，本模型与任何模型都够不着。

#### ③ 下一刀（**预登记**）：把「空间补全」变成真任务 —— **块状遮盖**

现在的遮盖是「按事件选 50% → 扩张到整个 τ token」。它的副作用是：被遮盖 token 的空间信息**被自己抹掉了**。
候选改法：**按空间块遮盖**（在 (τ, x) 平面上随机取连续的块，块内**所有**格子——含空格——一起遮），于是：

* 块内仍有可见邻居（x 方向与 τ 方向的边缘）⇒ **空间插值重新变成可学的任务**；
* `mask_leak` 不再退化（块内既有事件也有空格，遮盖指示器不再等价于「这里有事件」）；
* 与现有 `granularity` 正交，可作为**新的 granularity 模式**加入 `generation/masks.py`。

**验收口径**：① `mask_leak` ≈ 全谱事件密度（信息中性）；② **被遮盖内部组内 τ AUC > 0.60**
（打破 §9-75① 的 0.5753 线性天花板说明表示变了）；③ `val_ratio` **< 0.68**（打赢 `mask_lambda` 的 0.6890）。
#### ④ 判据侧的可达上界（`runs/_probe_tau_ceiling.py`，同一批 val；自检 `oracle_tau` 复现 0.4171）

| 场 | 除以 const |
|---|---|
| `oracle_tau`（**τ 边缘 oracle** + 均匀空间 + 无遮盖） | 0.4171 |
| `mask_ridge_oracle`（岭回归 τ 形状 + **遮盖置零** + **oracle 每线总量**） | **0.6711** |
| `ridge_oracle`（同上但不带遮盖） | 0.7294 |
| **`mask_ridge_free`**（岭回归 τ 形状 + 遮盖置零 + **档位也只用输入**） | **0.7910** |
| **模型（`mask_lambda` @4k/8k）** | **0.6890 / 0.6984** |

⇒ **三条都可操作的读数**：

1. 模型**打赢了完全无 oracle 的读数上界**（0.689 vs 0.7910）——它不是「什么都没学到」；
2. 但离「**oracle 每线总量 + 遮盖**」这个上界还差 **0.018–0.027**，而那条上界只需要
   「每线总质量」这一个标量是已知的 ⇒ **剩下的差距就是总质量校准**（模型 `val_pred_over_true` 0.59–0.70）；
3. **0.671 就是当前输入下的地板**：它已经用掉了「τ 边缘 oracle」这个不该给的信息，模型却在同一量级 ⇒
   **`val_ratio` 这条判据已经被压得差不多了，再想往下走必须换输入表示（下一节的块状遮盖）**。
#### ⑤ ⚠️ 可行性约束（本轮顺手查到，**下一轮必须先看这一条**）：缓存把遮盖表示锁成了 token 位图

`beatmorph/data/window_cache.py` 存的是「稀疏计数 + **token 遮盖位图** + 轨道 + 音频」，并且在装配时
**断言**稠密遮盖恰好等于该 token 位图的广播（`_token_view` → `"遮盖不是 token 区块结构：本缓存的紧凑
表示不适用（不要静默退化，请检查 masks.py）"`，`WindowCacheError`）。

⇒ **块状遮盖（§9-75③）不是改 `masks.py` 一个函数就能上的**：

* 它要么**重建窗口缓存**（实测 train 353.6 GB / 11.4 h + val 41.7 GB / 1.2 h，见 §9-54⑮），
* 要么**改缓存格式/装配路径**：缓存里已有「遮盖种子（由 `(seed, row_index, window_index)` 派生）」
  与稀疏计数，而块状遮盖只影响 `occlusion` 张量、不影响 `counts` ⇒ 可以在**读窗口时按新方案重算遮盖**
  （代价：要给方案加版本号，并让“逐位一致”的验收换一套口径）。

**建议路线**：先走第二条（不动 353 GB 的计数缓存，只换遮盖的生成方案 + 版本号），在**小切片**上验收
（`data.max_samples=200` 的切片只需重建那一小段），确认 §9-75③ 的三条判据之后再谈全量。
---

### §9-76 第二十六轮：**块状遮盖（`granularity="block"`）——把「空间补全」变回真任务**（2026-09-30）

#### ① 为什么是它

§9-75② 的结构性结论：`event` 粒度是「先选事件、再扩张到**整个 (k,τ) token**」⇒ **被预测 token 的可见场恒为常数**，
于是「token 内部的空间分布」在自己的输入里根本不存在，而 `tau_flat`（空间 oracle）= 0.2559 说明那一维值 0.46。
§9-75⑤ 又说明：窗口缓存把遮盖压成 token 位图并**断言**区块结构 ⇒ 不能只改 `masks.py` 而不动缓存。

#### ② 实现（三处，全部带护栏）

1. **`granularity="block"`**（`beatmorph/generation/masks.py`）：把 (τ, x) 平面按 `BLOCK_TAU=4 × BLOCK_X=32`
   切成块，**块内所有格子（含空格）一起遮**；权重用一次分块求和预计算，只对选中的块做切片赋值
   （不能复用 `_units` 的平坦下标表示：真实窗口约 1.2e7 个格子，展开成 Python list 会爆）。
2. **Hold 配对收口**（**门禁当场抓到的**）：首版不带收口 ⇒ 门禁退出码 7「hold 配对点被拆散：起止平坦下标
   93122 / 185283 的遮盖状态不一致」。修法按**格子级**补遮（每个配对独立、只补那一个端点格），
   而**不是**扩张到整个 token —— 后者是 `event` 粒度的做法，而块状粒度的设计本来就是「token 内允许孤立被遮格」。
3. **缓存兼容的重遮盖**（`beatmorph/data/dataset.py::_remask`）：缓存里的**计数仍然有效**，只有遮盖需要重算
   ⇒ `__getitem__` 在非 `event` 粒度下从计数用**同一套派生种子**重新生成遮盖。这样训练与 val 走的是
   **同一条快速路径**（`window_cache` + `workers=0`），不必重建 353 GB、也不必换成 8 worker（那会破坏可比性）。
   字段：`data.occlusion_granularity`（语义字段，进续训指纹）。

**护栏** `tests/unit/generation/test_block_occlusion.py`（5 项），核心那条是结构性不变量：
**块状遮盖必须留下「部分遮盖」的 token**，且对照路径（`event`）**必须仍然是全有全无**——
一旦有人把它改回「先选事件再扩张」，实验会悄悄退回原状且不报错。

#### ③ 预登记判据（**跑之前写死**）

臂 `probe_block`：8000 步、`data.workers=0`、`plan_window_shuffle=true`、`model.mask_lambda=true`、
`data.occlusion_granularity=block`，其余与 `probe_masklam`（0.6890 / 0.6984）完全一致。

1. **结构性**：`mask_leak` 明显低于 1（不再等价于「这里有事件」），且存在部分遮盖 token（护栏已锁）；
2. **验收口径**：**被遮盖 token 内部组内 τ AUC > 0.60**（打破 §9-75① 的 0.5753 线性天花板 ⇒ 说明输入表示变了）；
3. **判据**：`val_ratio` **< 0.68**（打赢 `mask_lambda` 的 0.6890）。

任一条不成立 ⇒ 「逐 token 遮盖抹掉了空间信息」这个结论**不足以解释**模型的失败，下一轮转回表示/优化侧。
#### ④ 结果（step 4000）：**判据 ③ 差 0.004 未达成，但质量校准几乎完美**

| 臂 | `val_ratio` @4k | `val_pred_over_true` @4k |
|---|---|---|
| `probe_masklam`（token 遮盖 + mask_lambda） | 0.6890 | 0.7037 |
| **`probe_block`（块状遮盖 + mask_lambda）** | **0.6838** | **0.9797** |

⇒ **块状遮盖只买到 0.005（0.6890 → 0.6838），判据 ③（< 0.68）差 0.004 未达成**；
但**质量校准从 0.70 跳到 0.98**（几乎精确）⇒ 块状遮盖确实让「该放多少」变得可学，
**而对「放在哪里」几乎没有帮助**。这与「空间分布可从块内可见邻居推断」的预期**不符**。

⚠️ **本轮的一条重要限制**：块状遮盖**同时改变了两件事**——(i) 被预测 token 内部重新有可见邻居，
(ii) 遮盖指示器的含义（块内既有事件也有空格）。0.005 的净收益说明这两件事要么互相抵消、
要么都不是瓶颈。**在拿到判据 ②（被遮盖内部 τ AUC）之前不要下结论**——那条读数区分
「表示变了但判据不敏感」与「表示根本没变」。

**未完成项**：判据 ② 需要在块状臂的 checkpoint 上跑 `runs/_probe_mask_response.py --mask-lambda`
（本轮 GPU 被训练占着，未跑）；8000 步的 val 点本轮交付时仍在跑。**两者都是下一轮的第一件。**

### §9-77 第二十七轮：**τ 从来不是"天花板 0.58"——语料 τ 表值 0.93 AUC，是训练把三个 local 层推到"沿 τ 不变"**（2026-09-30）

> 本轮推翻了 §9-66/§9-67/§9-68/§9-75 里**整条「输入天花板 ≈0.58」的证据链**，把根因从「数据不给力」
> 换成「架构里 τ 只有一条通路，而训练把这条通路亲手压平」。以下每条都有可复跑的探针。

#### ① 谱面格栅吸附（`runs/_probe_grid_snap.py`，600 张 train 谱 / 782 902 事件，**不经数据管线**）

| 落在 1/d 拍上 | 1/1 | 1/2 | 1/3 | **1/4** | 1/6 | **1/8** | 1/12 | 1/16 | 1/24 | **1/48** |
|---|---|---|---|---|---|---|---|---|---|---|
| 事件占比 | 38.40% | 63.17% | 40.62% | **82.76%** | 66.53% | **89.64%** | 86.84% | 91.66% | 93.83% | **95.86%** |

窗口内 192 格直方图：CV **3.516**（均匀理论值 0.992），前 10 格
`0:10.38% 24:6.07% 48:9.33% 72:6.43% 96:9.66% 120:6.30% 144:9.50% 168:6.33%`
⇒ **理想先验 AUC = 0.9326；1/4 拍指示器 = 0.9557**。

#### ② 在**真实数据集样本**上对撞（`runs/_probe_window_phase.py`，与训练同一条装配路径）

* `tau_start` 拍相位：train 203/256、val 221/256 落在**整拍**（非零相位窗仅 20.7% / 13.7%）
  ⇒ 窗口 = 从 τ=0 起连续 192 格 = 4 拍，**管线 τ 轴与谱面拍轴一致**；
* 数据集内 192 格直方图与原始谱面同形（val 1/4 拍格占比 **72.90%**、CV **3.090**）；
* **以 train 256 窗的 τ 直方图（一张常数表）为唯一分数**，在 val 上按 (窗,线) 分组：

| 分数 | 组内 τ AUC | 组数 |
|---|---|---|
| **语料 τ 表（train 直方图）** | **0.9292** | 9472 |
| 同上，仅被遮盖格 | **0.9295** | 9472 |
| 1/4 拍指示器（12 格步长） | 0.8312 | — |
| val 自身直方图（自先验，上界参考） | 0.9491 | — |
| 均匀 | 0.5000 | — |

#### ③ 更正：§9-75① 的「0.5753 天花板 / 0.5555 先验」**作废**（探针特征写错周期）

`runs/_probe_audio_info2.py` 的 `pos`（被 §9-66/§9-67/§9-68/§9-69/§9-75 全线复用）是

```python
tau_f = cols / t_bins          # 列号 / 窗口长度 —— **窗口**占比，而窗口 = 4 拍
pos = [2*tau_f - 1, sin(2*pi*tau_f*4), cos(2*pi*tau_f*4),
                    sin(2*pi*tau_f*12), cos(2*pi*tau_f*12)]
```

`sin(2π·tau_f·4)` 的周期是 **1/4 个窗口 = 1 拍**，`·12` 的周期是 **1/3 拍**——
**这条特征族根本表达不了承载 82.8%/89.6% 事件的 1/4 拍与 1/8 拍**。
因此 0.5555 / 0.5538 / 0.5753 / 0.5583 / 0.5792 / 0.5818 **全部只说明「那条 5 维特征族不行」**，
不说明输入里没有 τ 信息。**「瓶颈在信息」这条结论随之作废**（§9-67/§9-68）。

#### ④ 免拟合定位：τ 是被**训练**抹平的，不是架构不能表达（`_probe_tau_flat_audit.py` + `_probe_tau_layer_trace.py`）

指标 = 沿 τ 的相对起伏 `rel = ‖x − mean_τ x‖_F / ‖mean_τ x‖_F`（逐线中位，1 批 K=104 T=192，纯 CPU、无拟合）：

| 一级 | 未训练 seed0 | 训练后（`probe_masklam` best.pt） |
|---|---|---|
| embedding 输出 tokens (+音频) | 7.564 | 7.218 |
| layer 00 local | 5.547 | **0.835** |
| layer 01 local | 4.257 | **0.0257** |
| layer 02 local | 3.270 | **0.00014** |
| layer 03 global | 2.868 | **0.00000** |
| layer 04 local / 05 global | 2.231 / 1.941 | 0.00000 / 0.00000 |
| head `delta = ΔΛ(t)` | 0.832 | 0.00001 |
| 空间 `prob(x,s,c|t)` | 1.193 | 0.00000 |

⇒ **架构能表达 τ（未训练 2.06–7.56），是优化把它推到「沿 τ 不变」**。
这一步同时解释了 §9-68 的两条读数：`Σ_{x,s,c} λ = ΔΛ(t)·(空间调和项)`，
ΔΛ 一旦沿 τ 恒定，**组内 τ 排序全部并列 ⇒ AUC 精确等于 0.5009**（不是"接近随机"）。

#### ⑤ 结构性的要害：头的 τ 通路只有解码器这一条

τ 位置编码是在 `FieldTokenEmbedding` 里加到**输入** token 上的；local 层一平均，位置信息在到达头之前就没了；
`head_skip=False` 时 `FieldHead` 只剩 `decode` 的输出，**没有第二条 τ 通路**。
于是「学那张值 42.46% 常数基线的 τ 表」（§9-69 `gap_tau`）在**当前装配下不可达**——不是优化没找到。

#### ⑥ 处置（决策者已授权改损失语义 / 层结构）

1. **给头一条直接的 τ 通路**：`FieldHead` 增加**可学习的 τ 偏置**，在 softplus/softmax **之前**
   加到 logits 上（`cum` 一条 192 维、`cell` 一条 192×1280）。最小版 = 只加 `cum` 的 192 个参数，
   它应当直接学出②里那张 AUC 0.93 的语料 τ 表。"无条件场先验"由头直接表达，条件调制仍由解码器给。
2. **验收（预登记，跑之前写死）**：被遮盖内部**组内 τ AUC > 0.85** 且 **`val_ratio` < 0.60**。
   若 τ AUC 上去而 `val_ratio` 不动 ⇒ τ 通道不是主瓶颈，下一轮转"空间"轴。
3. 若 1 无效 ⇒ 再动 local 层的残差/归一化（把 τ 起伏的每层衰减从 0.12 倍压到 ≈0.9 倍）。

#### ⑦ 同轮补记：§9-76 的三条判据全部落地

* 判据 ② 实测 **0.5198**（92 组，`_probe_mask_response.py --cpu`）⇒ **FAIL**（要求 > 0.60）；
* 判据 ③ `probe_block` **4k 0.6838 / 8k 0.7384**，4k 只赢 `mask_lambda` 0.005（同配置重跑差 0.051，§9-63）
  ⇒ 4k 的差异在噪声内，8k 反而更差。**块状遮盖的价值在"质量校准"（0.98 vs 0.70），不在定位。**
* ⇒ 「逐 token 遮盖抹掉空间信息」解释不了模型的失败（§9-76 预登记的退出条件），本轮已按该条件转回**表示/优化侧**。

#### ⑧ 运维事故（如实记账）

本轮期间发生 **GPU 显存过载 → 驱动重载 → Windows 重启**（决策者告知）。**触发器未取证**：
崩溃前同时存在过多后台作业（训练 + 多个 CPU 探针），运行时间线与 `nvidia-smi` 未留下采样。
新增纪律（并入 TRAINING.md §7.5）：
1. 任何 GPU 作业前**先读 `nvidia-smi memory.used`**，非 0 一律不得启动；
2. 探针**默认 `--device cpu` + `CUDA_VISIBLE_DEVICES=''`**，需要 GPU 时必须显式说明理由与批数；
3. 同一时刻**只允许一个 `runs/` 后台作业**，作业全部落日志、逐个核对结束码。
4. ⚠️ **探针不要一律丢到 CPU 上**（决策者指出，同轮实测复现）：真实 val 批的全局层注意力是
   `O((K·T)²)`（K=104、T=192 ⇒ L=19968），**fp32 / CPU 路径会物化 25.5 GB 的注意力矩阵**
   （实测报错 `DefaultCPUAllocator: not enough memory: 25518145536 bytes`），
   而同样的批在 **bf16-mixed + SDPA** 下只占几百 MB。把探针放 CPU 既慢、又要吃 20+ GB 内存，
   反而更容易把整机推到页面文件；而 **fp32 放 GPU 会溢出 8 GB 显存、掉进 Windows 共享显存**
   （现象：GPU 利用率几乎为 0、功耗 11 W，但任务"还在跑"）。
   **正解 = 与训练同一条口径**：`autocast_context(cfg.optim.precision, device)` + GPU + **单作业**。
   `runs/_probe_mask_response.py` 一直是这么写的（所以它从未出问题）；
   只有本轮临时写的 `runs/_smoke_tau_bias.py` 忘了加，一次就复现了两种失败。
5. **已登记的缓存失效（本轮实测）**：train 的窗口预切缓存在磁盘上是
   `window_cache/train/aa677031c17e3a3d`，而本轮 `plan_window_shuffle=true` 算出的指纹是
   `af2df05c3f900e30` ⇒ **缓存未命中、训练退回原路径**（步时约 0.5 s 而不是 0.2 s）。
   注意 `plan_window_shuffle` 只改**取窗顺序**、不改窗口内容，把它算进缓存指纹是**过保守**的；
   val 侧不受影响（val 不用计划层，指纹仍是 `fac0448693925c86`）。未修，先如实记账。


### §9-78 第二十八轮：**修法验证通过 —— 192 个参数的 τ 偏置把组内 τ AUC 从 0.5009 抬到 0.9522、`val_ratio` 从 0.6890 打到 0.6115**（2026-09-30）

#### ① 实现（提交 `d7f429b`）

* `ModelConfig.head_tau_bias` / `head_cell_tau_bias`（**默认 False、初始化为 0** ⇒ 开启时与关闭时逐位同构）；
* 在 `FieldHead` 的 `cum` / `cell` logits 上、softplus/softmax **之前**相加，
  形状 `(t_bins,)` / `(t_bins, cells)`；
* 护栏 5 项 `tests/unit/generation/test_head_tau_bias.py`，其中两条是**这次失败的形状**：
  「沿 τ 不变的 token ⇒ 场必须沿 τ 不变」（对照）与「加上偏置后必须沿 τ 变」（修法），
  外加「不同 (b,k) 的 τ **排序**必须一致」（防 reshape 广播到线轴）；
* 顺带修护栏：`test_source_guards` 的片段匹配从「子串包含」改为「按 `_` 整段匹配」——
  `head_tau_bias` 含子串 `d_tau`，不改规则就会被判成「自建 τ 格宽常量」（已加回归测试）。

#### ② 臂与预登记判据（判据见 §9-77⑥，**跑之前写死**）

`probe_taubias` = `data.plan_window_shuffle=true` + `model.mask_lambda=true` + `model.head_tau_bias=true`，8 000 步。

| 判据 | 预登记 | 实测 | 结论 |
|---|---|---|---|
| 被遮盖 token 内部组内 τ AUC | **> 0.85** | **0.9531 @4k / 0.9522 @8k / 0.9513 @16k** | ✅（历史 0.4992–0.5198；语料 τ 表本身 0.9292） |
| `val_ratio` | **< 0.60** | **0.6397 @4k / 0.6115 @8k / 0.5927 @12k / 0.5571 @16k** | ✅（历史最好只有 0.6838） |

**16k 延长跑（同轮完成，`--resume`，EXIT=0）**：`val_ratio` **四点单调下降**
0.6397 → 0.6115 → 0.5927 → **0.5571**，`best.pt` = step 16000；τ AUC 四点稳定在 **0.9513–0.9531**
⇒ τ 那一半**饱和了**，12k→16k 的继续下降（−0.036）来自**别的轴**（空间 / 质量校准）。
⚠️ 单跑；同配置重跑的本底差是 0.051（§9-72④），而 0.6890 → 0.5571 = **−0.132**（2.6 倍本底）
⇒ 量级可信，但按 §9-63 纪律应补一次同配置重复。

#### ③ 与历史最好值的对照（同一 val 口径，const = 3.94133）

| 臂 | 4k | 8k | 12k | 16k |
|---|---|---|---|---|
| `probe_masklam`（token 遮盖 + `mask_lambda`） | 0.6890 | 0.6984 | — | — |
| `probe_block`（块状遮盖） | 0.6838 | 0.7384 | — | — |
| **`probe_taubias`（+ 头 τ 偏置）** | **0.6397** | **0.6115** | **0.5927** | **0.5571** |

⇒ **`val_ratio` 首次在 4k→8k 继续下降，并把单调下降延续到 16k**（此前每条臂到 8k 都反弹到 0.83–1.13，或 `1.0033` 破红线）；
τ AUC 从「**精确并列**的 0.5009」抬到 **0.9522**，并且**高于语料 τ 表自己的 0.9292**
⇒ 模型学到的是**条件**排序，不是只背了一张全局表。

#### ④ 为什么这构成「真因 + 有效解」的闭合

1. **真因（§9-77④）**：τ 位置信息只加在**输入** token 上；训练把三个 local 自注意力层推到
   「沿 τ 不变」（未训练 rel = 2.064 → 训练后 0.00000），而 `head_skip=false` 时头只剩 decode 的输出
   ⇒ **无条件 τ 先验在装配上不可达**（不是优化没找到）。
2. **它值多少（§9-77②、§9-69）**：语料 τ 表在真实 val 上组内 AUC **0.9292**；τ 轴占常数基线 **42.46%**。
3. **修法的代价**：**192 个参数**、不动任何损失语义；条件调制仍全部由解码器提供。
4. **实测**：τ AUC 0.5009 → **0.9522**；`val_ratio` 0.6890 → **0.6115**（−0.077），且 4k→8k 继续下降。

#### ⑤ 未完成（下一轮第一件）

* **空间侧的同一件事**：`gap_space` = 58.58% > `gap_tau` = 42.46%（§9-69），而 `tau_flat`
  （完美空间 + τ 常数）= **0.2559** ⇒ **空间轴的净值比 τ 轴还大**，而本轮只修了 τ。
  `head_cell_tau_bias`（192×1280 的无条件联合先验）已实现但**尚未单独验证**——
  预登记判据应写成「`val_ratio` < 0.55」（在 0.6115 的基础上再要一成）加空间轴命中率。
* ~~16k 延长跑~~ **已完成**：0.5571 @16k、τ AUC 0.9513 @16k（见②）。
* **同配置重复**一次（§9-63 纪律）：本条是本项目历史最大的一次单臂改善（−0.132），
  但只有一跑；机制侧（τ AUC 0.5009 → 0.95）不需要重复，**数值侧（`val_ratio`）需要**。
* 「τ AUC 稳定在 0.95 而 `val_ratio` 还在降」⇒ 下一段的收益在**空间轴**上，
  这与 §9-69 的 `gap_space` = 58.58% > `gap_tau` = 42.46% 一致。

### §9-79 第二十九轮：**「我们喂给模型的数据有一半不是玩法」——去表演口径落地（决策者裁定）**（2026-09-30）

#### ① 问题的规模（新增 `runs/_probe_scorable_share.py`，与调研任务 research/kipphi-rpejson 同口径）

判定 = **命中时刻该线 `alphaEvents` 跨层求和 ≤ 0 ⇒ 玩家看不见、判不到**（prpr A 级语义）。

| | train（300 张 / 392 502 note） | val（300 张 / 395 943 note） |
|---|---|---|
| 假音符 `isFake=1` | 2.11%（dataset 已滤） | 2.81% |
| **命中时线不可见** | **22.81%** | **18.68%** |
| ↳ 其中在**真线**上 | **12.94%（50 789）** | 10.59%（41 911） |
| 可计分 | 75.08% | 78.50% |
| **可计分落在装饰线上** | **0** | **0** |

⇒ ① 占目标**两成**的事件是玩家看不到的表演 note（落点由演出设计决定、不由音乐决定）；
② 其中**一半在真线上** ⇒ **线级剥离拿不掉**，恰好混在玩法事件里；
③ 「可计分」与「装饰线」在真实语料里**零交集**，是管线把两类混在了一起。

**线轴**（`runs/_probe_line_strip.py`，各 300 张）：

| | 全部线（中位 / p90 / max） | 真线（中位 / p90） | k_max=128 超限行 |
|---|---|---|---|
| train | 26 / 68 / 240 | **5 / 12** | 2.0% → **0** |
| val | 25 / 62 / 434 | **5 / 13** | 2.7% → **0** |

⇒ **模型此前把绝大部分容量用在「这条线上 λ 恒为 0」**；祖先线代价极小（train 0.06 条/谱、4.7% 的谱需要）。

#### ② 落地（提交 `a1220d8` + `308675c`，决策者口径：**train 与 val 同口径**）

* `decoder.events.note_is_scorable / scorable_note_mask / gameplay_subchart`：
  **判据只有一份实现**，生成闸门（`scorable_lines` → e2e 的 allowed_lines）与训练目标共用；
* `gameplay_subchart` = **真线 ∪ 祖先线** + 只留可计分 note；线序保持相对顺序、
  `father` 与 `note.line_id` 一并重映射；**必须连带祖先**（`JudgeLine.pose_at` 会合成父线运动，
  摘掉父线 = 换输入而不是去表演）；无任何可计分 note ⇒ 抛错，调用方跳过该行；
* `unknown_alpha_is_visible`：生成闸门默认 True（合成模板没给 alpha 轨 = 「没这个信息」），
  训练目标显式 **False**（真实谱面没有 alphaEvents ⇒ 不可见）；**两个默认值故意相反，已写在现场注释里**；
* `data.scorable_target`（与 `window_cache_dir` 互斥、进索引指纹、`build_windows` 拒跑）；
* 过滤点只在 `_chart_for`（唯一谱面入口）⇒ `_note_bins` / `_hold_blocked_boundaries` /
  `_count_events_beyond_axis` / `build_target` 四条镜像口径自动一致；
* 台账 `notes_non_scorable_dropped` / `skipped_no_playable_lines` 进索引摘要（不静默丢弃）。

CI：ruff/mypy ✅、pytest **1194 passed**（新增 7 项）。护栏 `tests/unit/data/test_scorable_target.py`。

#### ③ 为什么这是「数据噪声」的正确刻画

不是随机噪声，而是**系统性的、与目标定义不符的内容**：目标函数把「演出装饰」当成「玩法事件」，
于是 ① 约两成的梯度在教模型**忽略音乐**（装饰的落点由演出决定）；② 线轴上 83% 的位置
正确答案是「什么都不放」，模型学到的「输出边际分布」在很大程度上是**最优反应**，
这解释了 §9-62 的「常数先验」、零模型基线能到 0.72–0.76、以及音频依赖在每个臂上衰减。
去表演之后，「零模型打平」这件事本身应当失效——**那正是本轮的判据**。

#### ④ 预登记判据与进行中的臂

`probe_noperf` = `plan_window_shuffle + mask_lambda + head_tau_bias + scorable_target`，8 000 步。
判据（跑之前写死）：
1. `val_ratio` **< 0.50**（在 `probe_taubias` 的 0.5571 上再要一成）；
2. **线轴利用率**：`line_only` 这条「每线一个常数」的零模型基线应当**明显变差**
   （去表演后线密度不再是主要信息来源）——这是「模型不再靠边际分布」的直接证据；
3. G1/G3/G4 全绿（已通过：G1 977.4→−2.09，G3 −1.21 vs 5.74）。

⚠️ 口径已变 ⇒ **损失数值不可与 `probe_taubias` 直接比较**，必须在**同一套 val 口径**下对照
（`runs/_probe_val_ratio.py` 已写好：固定 val 窗口、把任意 checkpoint 放到指定口径上复算 ratio）。
⚠️ **名债**：开关仍叫 `scorable_target`，但它现在同时剥离线轴；改名（`playable_only`）留待下一轮。

### §9-80 第三十轮：**「喂进去的数据有一半不是玩法」——口径已落地，但实测并不改善学习（负结果，如实记账）**（2026-09-30）

#### ① 臂

`probe_noperf` = `plan_window_shuffle + mask_lambda + head_tau_bias + scorable_target`（去表演：真线∪祖先 + 只留可计分 note），
8 000 → 16 000 步（`--resume`），窗口缓存关闭，索引并行 12 进程。

| 步 | 本臂自己 val 的 ratio | 备注 |
|---|---|---|
| 4 000 | 0.6893 | |
| 8 000 | 0.6242 | |
| 12 000 | **0.5808** | `best.pt` |
| 16 000 | **0.6815** | **反弹** |

⚠️ 本臂的 val 是**未分层**的（窗口缓存关闭 ⇒ 密度拿不到 ⇒ 按设计回退并打警告）：空窗 **0.566**、
常数基线 **7.02**（分层集 3.94）⇒ **绝对值不可与历史比**。原预登记的「< 0.50」在本口径下不可判读，
已就地更正为**等步数 + 同一套 val 窗口**的对照（下面）。

#### ② 等步数对照（`runs/_probe_val_ratio.py --scorable-target`，同一套 512 窗 / 2 276 事件）

| 步 | 旧口径臂（`probe_taubias`） | 去表演臂（`probe_noperf`） | Δ |
|---|---|---|---|
| 12 000 | 0.5872 | **0.5808** | **−0.0064** |
| 14 000 | 0.5971 | — | |
| 16 000 | **0.5703** | 0.6815 | +0.111 |

**判读**：等步数下相差 **0.0064**，而**同配置重跑的本底差是 0.051**（§9-72④）⇒ **去表演没有可测的改善**；
16k 那一臂反而**反弹**（旧口径臂持续下降）。

#### ③ 结论（这是一条**负结果**，与直觉相反但必须记下）

「目标里 19–23% 的事件是玩家看不见的表演 note」是**数据事实**（§9-79①），把它从目标里去掉在**定义上是对的**
（模型不该被要求预测不可判定的装饰）；但**它不构成「模型学不会音乐」的主因**：
去掉之后，等步数的 val 收益在噪声之内，稳定性还更差。

⇒ 到目前为止，**唯一被证据支持的「有效解」仍然只有 τ 通路那条**（§9-78：τ AUC 0.5009 → 0.9513、
`val_ratio` 0.6890 → 0.5571，同一口径同一 val 集、四点单调、效应 = 2.6 倍本底噪声）。

#### ④ 未决 / 下一刀

* 去表演臂**不是旧口径臂的严格消融**：线轴从 K≈26（83% 装饰线）变成 K≈5，输入分布本身变了。
  「去掉噪声但换了输入分布」与「只是换了输入分布」在本实验里不可分；要分开得再补一条
  「去表演但保留全部线轴（λ 在装饰线上恒为 0）」的臂。
* 16k 的反弹与 §9-71⑦ 的「val 序列非单调」一致 ⇒ 单点对照不可靠，`best.pt` 才是判读口径。

[POSTMORTEM-2026-08-05]: ../POSTMORTEM-2026-08-05-frame-rate-misalignment.md
