# 代码结构详解

> **文档版本**：v3.0（对齐 [CLAUDE.md](../CLAUDE.md) §2 目标模块拓扑与 [BasePlan.md](BasePlan.md) v3.0）
> 本文件同时描述**目标物理结构**与**当前实际结构**，并逐模块标出「现状 vs 目标」的差距。
> 技术决策见 [`BasePlan.md`](BasePlan.md)，实施计划见 [`plans/`](plans/README.md)，格式/单位事实见 [`knowledges/`](knowledges/)。

---

## 0. 先读这一段：迁移已开始（契约 / 数据 / 强度场已落地）

RFC-0029（2026-08-05 采纳）把目标游戏由 osu!mania 4K 改为 **Phigros**，生成范式改为
**判定线局部系多线标记点过程 + 掩码补全 + 非齐次泊松 NLL**，主路径格式锁定 **RPEJSON**。
**迁移自 2026-09-27 起已经开始**：`core/contracts`、`data/`（含 `dataset.py` / `tracks.py` 训练数据通路）、
`field/`、`generation/`（主干 M1–M4/M6）、`decoder/` 与 `io/formats/rpejson/` **写侧**（plan 05 M5.1–M5.6）、
以及 `infra/` **训练基础设施**（plan 07 M7.1–M7.8）与 `cli/train.py` 已按 v3.0 落地，
其余（消融臂 / 离散扩散、评估指标的部分细化）仍待建。
逐模块进度以 [plans/](plans/README.md) 各 plan 的里程碑状态列为准，本文件只描述结构。

| 标记 | 含义 |
|------|------|
| ✅ **符合目标** | 已存在且可直接复用 |
| 🟡 **【改造】** | 已存在但不是目标形态，需改写/改语义 |
| ⬜ **【待建】** | 目标拓扑要求，但目录/文件尚不存在 |
| 🗄️ **【归档】** | v2.x 退役资产，迁 `archive/osu-mania` 分支后不在主路径维护 |

**三类最关键的差距**：

1. ~~**契约里没有 Phigros**~~ ✅ **已解决**：`core/contracts/phigros.py` 定义了 `JudgeLine` / `PhigrosNote` /
   `PhigrosChart` / `BpmPoint` / `Side` / `NoteType`，`field.py` 定义了 `ChartFieldSpec` / `ChartField` /
   `ChartTargetField`，`tensors.py` 只留音频侧常量；`GameMode.PHIGROS` 已存在（plan 00 已交付）。
2. ~~**三个目标模块整体不存在**~~ 🟡 **大部分解决**：`field/`（plan 03）、`data/`（plan 02）、
   `generation/`（plan 04 主干）、`decoder/` + `io/formats/rpejson/` **写侧**（plan 05）已落地；
   `eval/` 仍待建（plan 06）。
3. ~~**生成主干尚未开始**~~ 🔵 **主路径已落地**：`generation/` 是掩码补全 Encoder-Decoder +
   非齐次泊松 NLL + 迭代并行解码（plan 04 M1–M4/M6，G1–G4 门禁实跑全绿）；
   消融臂 B1/B3/B4/B5 与离散 token 化待后续（plan 04 M7–M11）。

> ✅ **残留已清**（2026-08-05）：契约常量已是派生式 75 Hz
> （`core/contracts/tensors.py`：`MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`），
> 且 `configs/model/mert.yaml` 已同步修正（`feat: 1024`、`frame_rate: null` 并声明"派生量，勿写死"）。
> **其余 v2.x 配置组**（ar_transformer / planner / osu_50k / stage1_planner / stage2_ar / download）仍描述已退役模块，随 `archive/osu-mania` 迁移清理。
> 背景见 [POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md)。

---

## 1. 顶层布局

```
BeatMorph/
├── beatmorph/            # 主 Python 包（见 §2）
├── tests/                # 测试（unit / integration / e2e / fixtures）
├── configs/              # Hydra 配置（model / data / train 配置组）
├── docs/                 # BasePlan / plans / decisions / knowledges / 本文件等
│   ├── BasePlan.md               # 技术奠基（最高权威，v3.0）
│   ├── decisions/                # RFC 决策记录（RFC-0029 为当前范式权威）
│   ├── knowledges/               # 格式/单位/数据/文献事实库（Phigros RPEJSON 等）
│   ├── plans/                    # 各模块实施计划
│   ├── references/newplan.md     # 非权威思路参考
│   └── POSTMORTEM-2026-08-05-frame-rate-misalignment.md  # 门禁由来
├── scripts/              # 运维/验证脚本（当前仅 2 个：download_sayobot.py 已退役、verify_mert_frame_rate.py 仍有效）
├── data/                 # 数据（git 忽略；仅 fixtures / dev_sample 微型夹具入库）
├── models/               # 模型权重（git 忽略；models/pretrained/m-a-p/MERT-v1-330M 为本地缓存）
├── pyproject.toml        # 项目与工具配置（uv / ruff / pytest / mypy）
├── Makefile              # 常用命令快捷方式（类 Unix；Windows 用 uv 原生命令）
├── CLAUDE.md             # AI 协作宪法（v3.0，§2 模块拓扑）
├── AGENTS.md             # 子 agent 协作约定（v3.0）
└── LICENSE               # Apache-2.0
```

### 1.1 ⚠️ 工具链陷阱（易复发，勿回退）

- **排除模式必须锚定仓库根**：`pyproject.toml` 的 `[tool.ruff].extend-exclude` 与 `[tool.mypy].exclude`
  原为未锚定模式（裸写 `"data"`），会匹配**任意层级**的 `data/`，导致 `make lint` / `make typecheck`
  **静默跳过整个 `beatmorph/data/`**（mypy 少查 10 个文件：27 vs 37）。已改为锚定形式
  （ruff `"/data"`；mypy `"^(data|models|runs|outputs)/"`）。改动这些模式后请用「植入一个 F401 探针文件」
  的方式复验覆盖率——这类失效**不会报错**，只会让门禁空转（与 25 Hz 事件同类）。
- **物理常量防线**：`tests/unit/core/test_derived_constants.py` 用 AST 扫描 `beatmorph/` 下的**数值字面量**
  （注释与 docstring 中的说明性数字不会误报），黑名单含 1350 / 900 / 675 / 450 / 10.546875 / 0.83175 /
  75.0 / 320 / 24000 / (1÷48) / 120.23。新增物理量时改 `core/contracts/` 并在黑名单登记，**不要**在别处写字面量。

## 2. 目标包结构（`beatmorph/`）—— CLAUDE.md §2 的逐模块职责

```
beatmorph/
├── audio/encoder/        Stage 0  MERT-v1-330M + LoRA Adapter；帧率由模型 config 派生（= 75 Hz）
├── data/                 数据流水线：Phira 谱面获取 + RPEJSON 解析 + 质检 + MERT 特征离线提取
├── core/contracts/       ★ 跨模块契约：JudgeLine / PhigrosNote / PhigrosChart / ChartField
├── field/                强度场：λ_k(t,x,s,c) 的网格化、目标构建、∫λ 数值积分、可视化
├── generation/           Stage 1  掩码补全 Encoder-Decoder（主选）；离散扩散对照臂
├── decoder/              强度场 → 离散谱面（迭代/峰值解码）+ 合法性与可玩性后处理
├── io/formats/rpejson/   RPEJSON 读写（主路径唯一权威格式）
├── eval/                 事件级 F1 / 类型准确率 / 跨线合法性 / NLL 校准 / 人评协议
├── infra/                训练栈 + 健全性门禁 G1-G4（beatmorph/infra/sanity.py）
├── cli/  api/            命令行与服务接口
└── （alignment/ rag/）   Phase 3 待重估：DPO 偏好对齐 / 风格检索，RFC-0029 未涉及
```

| 模块 | 一句话职责 | 权威出处 |
|------|-----------|---------|
| `audio/encoder` | 把原始音频编码为帧级表征 `audio_emb [B,T,1024]`，帧率 = 派生量 75 Hz | BasePlan §3.1 |
| `data` | Phira 谱面包 → (audio, chart) 训练对；含获取、质检、特征离线提取 | BasePlan §4、[survey](knowledges/phira-dataset-survey.md) §9 |
| `core/contracts` | **唯一**跨模块通信面：判定线/音符/谱面/强度场的数据契约与张量形状 | CLAUDE.md 红线 2 |
| `field` | 强度场 λ 的网格定义、目标构建、两条 ∫λ 路径互校、可视化 | BasePlan §3.4、RFC-0029 §3.2 |
| `generation` | 掩码补全 Enc-Dec 主选；absorbing-state 离散扩散作对照臂 | BasePlan §3.3、RFC-0029 §3.3 |
| `decoder` | λ → 离散事件（D1 `find_peaks` 基线 / D2 Ogata thinning 原则解）+ 红线校验与**留痕**钳位 | BasePlan §3.6、[RFC-0030](decisions/RFC-0030-decoder-export-contract-ownership.md) |
| `io/formats/rpejson` | RPEJSON 读/写；不读 PEC（已淘汰），官谱 JSON 不在主路径 | RFC-0029 §8.1 Q3 |
| `eval` | 事件级 F1（±20ms / ±50ms 双报）、type/side 指标、跨线合法性、NLL 校准、人评 | RFC-0029 §5 |
| `infra` | 训练栈（Lightning / Hydra / ckpt）+ **G1-G4 门禁** | BasePlan §9 |
| `cli` / `api` | `beatmorph-train` 等命令行入口与推理服务（Phase 3） | BasePlan §5/§7 |

## 3. 现状 vs 目标（逐模块）

### 3.1 已存在但不是目标形态（🟡【改造】）

> 本轮（2026-09-27）已把 `core/contracts` 全量换成 v3.0 形态；该表其余行描述的是**尚未迁移**的模块。

| 路径 | 现状 | 目标形态 / 差距 |
|------|------|----------------|
| ~~`core/contracts/events.py`~~ | ✅ **已删除**（v2.x 的 `Note`/`Chart`/`Section`/`EventToken` 随 osu!mania 退役） | 由 `phigros.py` 取代；`GameMode` 现在只有 `PHIGROS`（红线 4：单一目标） |
| `core/contracts/tensors.py` | ✅ **已收敛**：只留 `MERT_SAMPLE_RATE_HZ` / `MERT_CONV_STRIDE_PRODUCT` / `MERT_FRAME_RATE_HZ`（派生）/ `MERT_DEFAULT_FEAT_DIM` 与 `AudioEmbedding` | 已完成；谱面侧常量移入 `phigros.py` / `field.py` |
| `audio/encoder/mert.py` | MERT + LoRA、滑窗拼接、`output_frame_rate()` 从 config 派生 ✅ | 基本可复用；仅需明确长音频窗与事件级对齐口径 |
| `audio/separation/demucs.py` | 四轨分离（可选增强） | v3.0 未列入主路径（BasePlan §3.1 不再提 Demucs），去留待定 |
| `data/parsers/osu_path.py` | `.osu` 解析 + 段落统计 | 换成 RPEJSON 解析（含 `father` 嵌套、事件跨层求和、补洞） |
| `data/pipeline/embed.py` | `PreprocessPipeline`（4 步）+ `extract_mert_embeddings`（按 `.osu` 父目录找音频） | 骨架可留，输入改为 Phira 谱面包 + `info.yml.chart`/`info.yml.music` 定位 |
| `data/manifest.py` | sayobot 集级元数据注入 + 质量过滤 | 换为 Phira 元数据（`difficulty`/`tags`/`stable`/`ranked`）；阈值须待全库统计后重定 |
| `data/datasets.py` | `PlannerDataset`（按 `mode == MANIA_4K` 过滤） | 换为强度场样本数据集（含 mask 通道） |
| ~~`generation/ar_transformer.py`~~ | ✅ **已删除**（v2.x 的 AR Transformer） | 主选改掩码补全 Enc-Dec；AR 保留为**对照臂 B3/B4**，将按 v3.0 的离散 token 化重写 |
| `generation/batch.py` | ✅ **已交付**：`FieldBatch` / `FieldOutput` 契约与形状断言（含「普通轨必须先跨层求和」的 5 vs 20 断言） | plan 04 §3.1/§3.2 |
| `generation/masks.py` | ✅ **已交付**：三 mask 语义、按事件遮盖（hold 配对同遮）、**防 mask 泄漏的 token 块遮盖 + 稀释**、抄邻居/泄漏诊断 | plan 04 §4.2/§4.5 |
| `generation/losses.py` | ✅ **已交付**：`full_poisson_loss` / `masked_poisson_loss`（HT 重标定）、排列敏感性与「无 line 分类损失」契约、B1/B5 的目标函数 | plan 04 §3.3/§4.3 |
| `generation/model.py` | ✅ **已交付**：掩码补全 Enc-Dec（滑动窗口 + 周期全局层、难度 AdaLN、轨道/音频 cross-attention、可变 K、因子化 λ 头 + **直连 skip**） | plan 04 §4.1/§4.2 |
| `generation/sampling.py` | ✅ **已交付**：迭代并行解码（`steps >= 2` 契约、单调 schedule、三种连续场置信度） | plan 04 §4.4 |
| ~~`decoder/postprocess/constraints.py`~~ | ✅ **已由 v3.0 形态取代** | 现为 `decoder/postprocess/legality.py`：Phigros 红线（越界**只统计**、同刻上限、Hold 区间、跨线几何冲突、重复事件），`EditKind` **无 clamp 成员**（红线 3 在类型层不可表达），口径见 [RFC-0030](decisions/RFC-0030-decoder-export-contract-ownership.md) |
| `io/formats/osu.py` / `sm.py` / `base.py` | `.osu`（Reader/Writer 完整）、`.sm`、抽象基类 | `base.py` 可复用；`osu.py`/`sm.py` 归档；新增 `rpejson/` |
| ~~`core/eval.py`~~ | ✅ **已删除**（v2.x tokenizer 往返度量，属退役范式） | `eval/`（plan 06）重建为事件级 F1 / 校准 / 合法性 |
| `infra/train_loop.py` | ✅ **已交付**：torch 参考训练循环 + 门禁输入装配（G1-G4 的 step_fn 接线）+ 真实清单批次来源 | plan 07 §4.1/§4.3 |
| `infra/lightning_module.py` | 🟡 **已交付但未实跑**：`build_trainer` + LightningModule 封装（可选依赖；缺失即抛可操作报错，不静默回落） | plan 07 §4.1 / §9-10 |
| `infra/config/` | ✅ **已交付**：structured schema（缺字段/类型错启动即失败）+ OmegaConf 严格合并 + `data.provenance` 必填 | plan 07 §4.2 / M7.3 |
| `infra/`（门禁执行 / 产物 / 恢复 / 自检 / 派生量） | ✅ **已交付**：`gates.py`（fail-closed）、`artifacts.py`（六件套）、`checkpoint.py`（恢复校验）、`feature_cache.py`、`env_doctor.py`（三态）、`derive.py`（派生量扫描）、`smoke.py`（合成批次） | plan 07 M7.1–M7.8 |
| `cli/train.py` | ✅ **已交付**：`beatmorph-train --config-name <name> [overrides] [--gates|--gates-only]`，退出码语义见 TRAINING.md §7.2 | plan 07 §3.2 |
| `cli/generate.py` / `api/app.py` | 占位（Phase 3） | 延后，不阻塞主路径 |
| `configs/model/mert.yaml` | ✅ **已修正**：`feat: 1024`、`frame_rate: null`（派生量） | 无需再动；其余 v2.x 配置随迁移清理 |
| `scripts/verify_mert_frame_rate.py` | ✅ 三条证据链复算帧率（config 推导 / torch 实搭 / checkpoint 交叉核对） | **保留**，作为 G4 自检工具（见 [TRAINING.md](TRAINING.md) §1） |

### 3.2 目标拓扑要求但尚不存在（⬜【待建】）

| 路径 | 要交付什么 | 依据 |
|------|-----------|------|
| ~~`core/contracts/`（Phigros 部分）~~ | ✅ **已交付**：`phigros.py` + `field.py` + 不变量断言（plan 00） | RFC-0029 §2.3/§3.1 |
| ~~`io/formats/rpejson/`~~ | ✅ **写侧已交付**（`writer.py`：秒→beat 三元组、导出门禁、字节幂等）；**读侧在 `beatmorph/data/parsers/rpejson.py`**（plan 02，独立实现，只读 prpr/phichain 行为规范） | RFC-0029 §4.2、plan 05 §4.4 |
| ~~`field/`~~ | ✅ **已交付**：网格 / 目标构建 / 两条 ∫λ 路径互校 / 共格碰撞 / 可视化（plan 03） | BasePlan §3.4 |
| `eval/` | 事件级 F1 / MAE / side 与 type 准确率 / 合法性 / NLL 校准 / 人评协议 | RFC-0029 §5.1 |
| ~~`generation/`（掩码补全）~~ | ✅ **已交付**：Enc-Dec + 显式 mask 通道 + 按事件遮盖（plan 04 M1–M4/M6；G1–G4 门禁全绿） | BasePlan §3.3 |
| ~~数据获取脚本~~ | ✅ **已交付**：Phira API 枚举 + Range 预筛 + 选择性下载 + RPEJSON 解析 + 质检（plan 02） | [survey](knowledges/phira-dataset-survey.md) §9 |
| ~~v3.0 配置~~ | ✅ **已交付**：`configs/smoke.yaml`（合成门禁冒烟）与 `configs/phigros_masked.yaml`（真实数据全量） | CLAUDE.md §4（`configs/` 由 infra-agent 维护） |

### 3.3 v2.x 退役资产（🗄️【归档】→ `archive/osu-mania` 分支）

| 路径 | 退役理由 |
|------|---------|
| `planner/density.py` | 段级均值池化是**结构性信息瓶颈**，且 Phigros 不需要"密度规划"这一层（RFC-0029 §4.3） |
| `tokenizer/vqvae.py` | VQ-VAE 小节码本 = "小节级最近邻检索重组"，信息损失不可逆（BasePlan §3.2.1） |
| `tokenizer/events.py` + `bpe.py` | RFC-0028 产物，**保留存档**；其 "POS+NUDGE 无损离散化" 思想对 `positionX` 量化仍适用（RFC-0029 §4.3） |
| `rag/retriever.py` / `alignment/dpo.py` | Phase 3 待重估，RFC-0029 未涉及（CLAUDE.md §2） |
| `io/formats/osu.py` / `sm.py`、`scripts/download_sayobot.py`、相关单测 | `.osu` 仍是通用格式，**保留可复用**（RFC-0029 §4.3） |

> 归档执行的**红线**：`core/contracts` 与 `infra/sanity.py` 不属于归档范围 —— 前者是被扩展的共享边界，
> 后者是**范式中立**的门禁基座（只吃调用方的 `step_fn`，不 import torch、不认识任何模型）。

## 4. 数据流（Phigros 版，对应 BasePlan §2）

```
输入（极简）: ① 音频(WAV/MP3)  ② 难度  ③ 判定线事件轨（可选，条件输入）
   │
   ├─[audio/encoder] MERT-v1-330M（冻结，24kHz）+ LoRA Adapter
   │      └──► audio_emb [B, T, 1024]，帧率 = 24000/320 = 75 Hz（派生量，禁止硬编码）
   │
   ├─[data]      Phira 谱面包 ──(info.yml.chart 定位、按内容判型)──► RPEJSON
   ├─[io/formats/rpejson] RPEJSON ──► PhigrosChart { JudgeLine[1..K], PhigrosNote[*] }
   │      └─ beat 三元组 → 秒（BPMList 分段积分）；事件跨层求和 + 补洞
   │
   ├─[field]     网格化 ──► λ_k(t, x, s, c)  +  目标构建（按事件遮盖）
   │      └─ ∫λ 两条路径互校：网格数值积分 / 累积强度 Λ(t) 参数化
   │
   ├─[generation] Stage 1 掩码补全 Enc-Dec ──► 补全后的 λ_k
   │      └─ 条件：难度嵌入 + 判定线事件轨（cross-attention）+ 音频
   │
   ├─[decoder]   λ → 离散事件（v0: find_peaks；原则解: Ogata thinning）
   │      └─ 合法性/可玩性后处理：同刻按键上限 / Hold 区间 / 越界 / 跨线几何冲突
   │
   └─[io/formats/rpejson] write ──► RPEJSON（可玩谱面）

[eval] 事件级 F1 @±20ms 与 @±50ms（双报）/ positionX MAE / side 与 type 准确率 /
       跨线合法率 / 泊松 NLL（仅作校准，不作质量分）/ 人评协议
```

> **必须在原始时间域（秒）评估**，不得在帧索引域评估（RFC-0029 §7 硬约束 5）。

## 5. 测试结构（`tests/`）

```
tests/
├── conftest.py                     # 共享 fixtures（fixtures_dir / require_gpu）
├── unit/                           # 与 beatmorph/ 镜像
│   ├── core/test_contracts.py      # ★ 契约冒烟测试（默认 CI 内跑）
│   ├── core/ audio/ data/ io/ tokenizer/ planner/ generation/ decoder/ rag/ alignment/ infra/
│   └── infra/test_sanity.py        # ★ G1-G4 门禁单测
├── integration/                    # 多模块串联（audio→plan、pipeline→download、chart↔token）
├── e2e/                            # 音频→谱面端到端（当前仅占位 __init__.py）
└── fixtures/                       # 小型合法样本（可入库；仅微型夹具）
```

- **测试标记**：`slow` / `gpu` / `integration` / `e2e`（见 `pyproject.toml [tool.pytest.ini_options]`）。
  CI 默认跑「非 slow、非 gpu、非 e2e」（`make test-fast`）。
- ⚠️ **契约级测试不得依赖权重或 GPU**（CLAUDE.md §4）：凡"只有拿到权重才能跑"的断言等于没有断言。
- ⚠️ **mock / fixture 不得固化物理常量**：若 mock 必须产生帧数/坐标，须引用契约常量
  （RFC-0029 §7 硬约束 6 —— 25 Hz 之所以存活到万级数据规模，正是 mock 把它洗成了绿灯）。
- **已落地测试目录**：`tests/unit/{core,audio,data,field,generation,decoder,io,infra}/`、
  `tests/integration/{test_field_pipeline,test_time_conversion_seam,test_decode_to_rpejson,test_train_entry,generation/test_train_step}.py`；
  评估指标测试见 `tests/unit/eval/`（plan 06）。
  夹具按 [survey](knowledges/phira-dataset-survey.md) §9.3 建议取 **`chart/1000`（标准 RPE）** 与
  **`chart/7039`（伪装成 `.json` 的 PEC）**，但**必须裁成微缩样本**再入库（红线 5）。

## 6. 配置结构（`configs/`）

v3.0 的配置**不是** Hydra 的 group 组合目录，而是「一份完整的 structured config」：

```
configs/
├── smoke.yaml            # 合成数据：门禁冒烟 / CI（无权重、无网络、无 GPU）
├── phigros_masked.yaml   # 真实清单 + 特征缓存（max_samples: null = 全量 ⇒ 必须先有全绿 gates.txt）
└── model/mert.yaml       # Stage 0 MERT 的 group 配置（plan 01；不走 --config-name）
```

调用形态（可执行）：

```powershell
uv run beatmorph-train --config-name smoke --gates-only          # 数秒跑通 G1-G4 并落盘六件套
uv run beatmorph-train --config-name phigros_masked --gates      # 真实数据：先门禁，再训练
uv run beatmorph-train --config-name smoke optim.lr=1e-3 --gates # 点号 override（合并后仍过校验）
```

**严格性**：缺字段 / 类型错 / 多余字段 / `data.provenance` 为空 → **启动期失败**（1 秒内，不进入训练循环）；
配置里**不出现任何派生物理量**（帧率/桶宽只来自 `core/contracts`）。

> ✅ **已交付**：v3.0 的配置（`configs/smoke.yaml` 冒烟门禁 / `configs/phigros_masked.yaml` 真实数据全量）。
> `configs/` 由 infra-agent 维护，其它 agent 需新配置提需求（AGENTS.md §3.4）。
> ✅ `configs/model/mert.yaml` 已于 2026-08-05 修正（`frame_rate` 改为派生量、`feat` → 1024）。
> ✅ **已清理**：v2.x 的配置组（ar_transformer / planner / osu_50k / stage1_planner / stage2_ar / download / archive）已随 `archive/osu-mania` 迁移删除，主路径只剩上面三个文件。

## 7. 环境变量

| 变量 | 用途 | 出处 |
|------|------|------|
| `BEATMORPH_DATA_DIR` | 预处理产物 / 特征缓存根 | `.env.example`、各 `configs/*.yaml` |
| `BEATMORPH_RUNS_DIR` | 训练产出（ckpt / log）根 | `.env.example`、`configs/*.yaml` |
| `BEATMORPH_CACHE_DIR` | 通用缓存根 | `.env.example` |
| `BEATMORPH_MODELS_DIR` | 预训练权重根（`models/pretrained`） | `.env.example` |
| `BEATMORPH_RAW_DIR` | 原始谱面包落地根（v2.x 为 `.osu`+音频） | `configs/stage0_mert.yaml` |
| `HF_HOME` / `MODELSCOPE_CACHE` | MERT 权重缓存（二选一） | `.env.example`、[TRAINING.md](TRAINING.md) |
| `WANDB_PROJECT` / `WANDB_ENTITY` | 实验追踪（Phase 2 接入，Phase 1 默认关闭） | `.env.example` |
| `CUDA_VISIBLE_DEVICES` | 训练卡选择 | `.env.example` |

> 权重 / 原始音频 / 全量数据集**不入库**（CLAUDE.md 红线 5），走外部存储或各设备自行提取。

## 8. 迁移顺序（对齐 BasePlan §7 Phase 1-2）

1. ~~裁定数据合规~~ ✅ **已裁定（2026-08-05）**：风险由决策者承担，**硬约束 = 最终不发布权重**；训练可启动（CLAUDE.md 红线 5 附注）；
2. `core/contracts` 增 Phigros 契约 → 全模块以此为准（AGENTS.md §3.1）；
3. ~~`io/formats/rpejson/` 独立实现 + 契约断言测试~~ ✅ **读路径**（plan 02，落在 `data/parsers/rpejson.py`）+ **写路径**（plan 05，`io/formats/rpejson/writer.py`）；
4. 数据获取（Phira API 枚举 / Range 预筛 / 选择性下载）+ 质检；
5. MERT 特征离线提取（元数据随缓存落盘并在加载时校验）；
6. `field/` 两条 ∫λ 路径 + 一致性测试；
7. `generation/` 掩码补全骨架 + **G1-G4 全绿后才扩数据**；
8. ~~`decoder/` 双解码臂 + 合法性后处理~~ ✅（plan 05 M5.1–M5.6：D1 峰值 / D2 thinning / 合法性后处理 / RPEJSON 写路径，608 项默认 CI 内）；
   ⬜ 剩余：`eval/` 指标落地（plan 06）、真实模型 → 首版可玩 RPEJSON（M5.7 与 plan 08 联合验收）。

## 相关文档

- [BasePlan.md](BasePlan.md) — 技术奠基（最高权威，v3.0）
- [CLAUDE.md](../CLAUDE.md) — 宪法与红线（§2 模块拓扑、§3 七条红线）
- [decisions/RFC-0029](decisions/RFC-0029-phigros-continuous-chart-generation.md) — 当前范式权威
- [plans/](plans/README.md) — 模块实施计划（**当前索引仍为 v2.x 旧编号 00-09，待随 v3.0 重写**；
  目标编号见 [AGENTS.md](../AGENTS.md) §2：00 契约 / 01 音频 / 02 数据 / 03 场 / 04 生成 / 05 解码 / 06 评估 / 07-08 基础设施与接口）
- [knowledges/phigros-format.md](knowledges/phigros-format.md)、[phigros-units-and-geometry.md](knowledges/phigros-units-and-geometry.md)、
  [phira-dataset-survey.md](knowledges/phira-dataset-survey.md)、[chart-generation-literature.md](knowledges/chart-generation-literature.md)
- [POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) — 门禁与红线 7 的由来
- [TRAINING.md](TRAINING.md) — 训练操作手册 ｜ [glossary.md](glossary.md) — 术语表
