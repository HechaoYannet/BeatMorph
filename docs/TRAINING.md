# BeatMorph 训练操作手册

> **版本**：v3.0（Phigros 范式，对齐 [BasePlan v3.0](BasePlan.md) 与 [RFC-0029](decisions/RFC-0029-phigros-continuous-chart-generation.md)）
> 本文是 **「合规裁定 → 数据 → 解析 → 特征 → 强度场 → 门禁 → 训练」** 的端到端操作手册。
> 技术权威：[BasePlan.md](BasePlan.md) ｜ 格式事实：[knowledges/phigros-format.md](knowledges/phigros-format.md) ｜ 单位几何：[knowledges/phigros-units-and-geometry.md](knowledges/phigros-units-and-geometry.md) ｜ 数据源：[knowledges/phira-dataset-survey.md](knowledges/phira-dataset-survey.md)

---

## ⚠️ 0. 先读：两道**不可跳过**的顺序约束

> ### ✅ 约束一（已裁定 2026-08-05）：**数据合规由决策者承担风险，训练可启动**
>
> 风险事实留档：唯一可得的万级数据源（**Phira 官方 API，实测 9649 张**）其 ToU **未授予机器学习训练权利**，明示「**不保证用户拥有分发已上传内容的权利**」、禁止未经法律授权**创建衍生作品**；**代码许可证（GPL-3.0 / MIT）不覆盖用户上传内容**。
> **裁决**：风险由决策者自行承担（定位为对 Phira 社区的贡献与共创，最终善后由决策者合规完成）。
> **硬约束**：① **最终不发布模型权重**（裁决成立的前提，项目级承诺）② 数据可本地落盘但**不得入库** ③ 脚本须记录来源与用途 ④ 发布权重前必须重新裁定。
> 出处：[CLAUDE.md](../CLAUDE.md) 红线 5 附注、[RFC-0029 §8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md)、[survey](knowledges/phira-dataset-survey.md) §6。
>
> ### 🔴 约束二：**G1-G4 门禁必须在扩数据之前跑通**
>
> 任何新模型/新范式，**先在 1–4 个样本上把门禁跑绿，再谈扩数据规模**；
> 门禁实现在 [`beatmorph/infra/sanity.py`](../beatmorph/infra/sanity.py)，结果必须写入训练日志。
> 出处：CLAUDE.md 红线 7、BasePlan §9、RFC-0029 §7 硬约束 3。

**执行状态图例**（本文对每一步都标注）：

| 标记 | 含义 |
|------|------|
| ✅ **可执行** | 命令/路径**当前仓库已存在**，可直接跑 |
| 🟡 **部分可执行** | 骨架存在但仍是 osu!mania 版，需迁移（见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md) §3.1） |
| ⬜ **待建** | 目标组件尚未实现，本文只给**行为规范**，不给假命令 |

---

## 1. 环境准备 ✅

### 1.1 获取仓库与安装依赖

```powershell
git clone https://github.com/HechaoYannet/BeatMorph.git BeatMorph
cd BeatMorph

# 全量开发环境（audio + train + data + dev 组）
uv sync --extra audio --extra train --extra data --group dev
```

**依赖组说明**（`pyproject.toml [project.optional-dependencies]`）：

| extra / group | 含 | 何时需要 |
|-------|----|---------|
| `audio` | transformers / **modelscope** / peft / soundfile / librosa | MERT 权重与音频加载（Stage 0 / 特征提取） |
| `train` | pytorch-lightning / wandb / datasets / pyarrow / tensorboard | 训练栈、Parquet 读写 |
| `rag` | faiss-cpu | Phase 3 待重估（可不装） |
| `data` | httpx | 数据获取脚本的 HTTP 客户端 |
| `dev`（默认组） | ruff / pytest / mypy / pre-commit / respx | 开发与测试 |

> **包管理纪律**：统一 `uv`，**禁止直接 pip**（保证 `uv.lock` 可复现）。
> CUDA 轮子由 `pyproject.toml [tool.uv]` 的 `pytorch-cu130` index 选择；CUDA 版本不匹配时按 [PyTorch 官网](https://pytorch.org/) 调整 index。

### 1.2 环境变量

```powershell
$env:BEATMORPH_DATA_DIR  = "data/processed"   # 预处理产物 + 特征缓存根
$env:BEATMORPH_RUNS_DIR  = "runs"             # 训练产出（ckpt / log）根
$env:BEATMORPH_CACHE_DIR = ".cache"
$env:BEATMORPH_MODELS_DIR = "models/pretrained"
$env:BEATMORPH_RAW_DIR   = "data/raw"         # 原始谱面包落地根（v2.x 为 .osu + 音频）
$env:CUDA_VISIBLE_DEVICES = "0"

# MERT 权重缓存（二选一）
$env:HF_HOME = "$HOME/.cache/huggingface"
# 或（国内推荐）: $env:MODELSCOPE_CACHE = "$HOME/.cache/modelscope"
```

> 完整样例见 [`.env.example`](../.env.example)。**权重 / 原始音频 / 全量数据集绝不入库**（红线 5）。

### 1.3 MERT 权重就位 ✅（本机已缓存于 `models/pretrained/m-a-p/MERT-v1-330M`）

```powershell
# 方式 A：ModelScope（国内可达性最好）
uv run python -c "from modelscope import snapshot_download; print(snapshot_download('m-a-p/MERT-v1-330M'))"

# 方式 B：HuggingFace
uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)"
```

### 1.4 冒烟验证 ✅

```powershell
uv run pre-commit install                 # 提交钩子
make test-fast                            # 快测试（跳过 slow/gpu/e2e）：应全绿
uv run ruff check . ; uv run mypy beatmorph   # lint + 类型：应 0 error
```

### 1.5 ★ 帧率 / 单位自检（**每次动过音频侧代码后都要跑**）✅

这是 25 Hz 事件的收尾证据工具：只依赖 torch，直接按官方 `config.json` 的 `conv_kernel`/`conv_stride` 搭卷积栈实测帧数，并交叉核对真实 checkpoint 的卷积核形状。

```powershell
uv run python scripts/verify_mert_frame_rate.py --model-dir models/pretrained/m-a-p/MERT-v1-330M
```

**期望输出**（关键行）：

```
[config] conv_stride = [5, 2, 2, 2, 2, 2, 2]  -> prod = 320
[derive] frame rate  = 24000/320 = 75.0 Hz
[torch ]  1.0s ->    74 frames = 74.00 Hz
[torch ]  2.0s ->   149 frames = 74.50 Hz
[torch ]  5.0s ->   374 frames = 74.80 Hz
[assert] 5s 窗口应为 374 帧（75Hz），实测 374 帧 -> OK
[ckpt  ] conv_layers.0.weight: (512, 1, 10) ; encoder.layers.0.attention.k_proj.weight: (1024, 1024)  (hidden = 1024)
```

- ✅ **判据**：5 s → **374 帧**；帧率 **75 Hz**（`24000 / 320`，±1 帧来自卷积取整）。
- 契约侧对应测试（**不依赖权重与 GPU，默认 CI 内跑**）：`uv run pytest tests/unit/audio/test_frame_rate_contract.py -q`。
- 参考实现：[POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) §2.1。

---

## 2. 数据合规（**已由决策者裁定，2026-08-05**）✅

### 2.1 风险事实留档（不因裁定而消失；只陈述公开声明，不构成法律意见）

| 事实 | 出处 |
|------|------|
| Phira ToU：「对于用户上传内容，Phira **不保证用户拥有分发已上传内容的权利**」 | [survey](knowledges/phira-dataset-survey.md) §6.1 |
| ToU：用户「只有在获得明确授权的情况下才能使用此内容」，禁止未经法律授权**创建衍生作品** | 同上 |
| DMCA 页：平台自认「**may not be observed in some cases**」（存在未获授权的上传） | [survey](knowledges/phira-dataset-survey.md) §6.2 |
| 音频与曲绘**随谱 100% 捆绑分发**（196/196 实测）→ 版权风险与谱面作者授权是**两层** | [survey](knowledges/phira-dataset-survey.md) §4.3/§6.5 |
| 代码许可证（`TeamFlos/phira` GPL-3.0、`phira-web` MIT）**不覆盖用户上传内容** | [survey](knowledges/phira-dataset-survey.md) §6.4 |
| 解析器侧：prpr **GPL-3.0**、phichain **LGPL-3.0** → **只读行为规范、独立实现**，逐行移植有衍生作品风险 | RFC-0029 §4.2 |

### 2.2 裁决（2026-08-05）与由此产生的硬约束

**裁决**：由**决策者自行承担全部数据合规风险**——本项目定位为对 Phira 社区的**贡献与共创**行为，最终处理与善后由决策者以合规方式完成。**训练可启动。**

**项目级硬约束**（红线 5 附注 / RFC-0029 §7 硬约束 7）：

| # | 约束 |
|---|------|
| 1 | **最终不发布模型权重** —— 本裁决成立的前提条件，属**项目级承诺** |
| 2 | 谱面与音乐数据**允许本地落盘**，但**不得入库**（红线 5） |
| 3 | 获取与处理脚本须**记录来源与用途**，便于追溯与善后 |
| 4 | **发布权重前必须重新裁定** —— 本裁决不覆盖任何分发场景 |

> **对操作的影响**：第 3–7 节现在可以执行到"落音频 / 开训"。但**第 3 节的数据获取脚本必须实现约束 3 的来源记录**，且必须确认数据目录未被版本控制纳入（红线 5）。

---

## 3. 数据获取：Phira API ⬜（脚本待建）

> ⚠️ **当前 `scripts/` 下只有 `download_sayobot.py`（v2.x，已退役）与 `verify_mert_frame_rate.py`（§1.5 仍有效）**——
> **Phira 获取脚本尚不存在**。本节给的是**必须实现的行为规范**，不是可复制粘贴的命令。
> 实现依据：[survey](knowledges/phira-dataset-survey.md) §7.6 / §9。

### 3.1 元数据枚举（322 页）

- 入口：`GET https://api.phira.cn/chart`；**响应键是 `results`**（不是 C 级文档写的 `result`）。
- **`pageNum` 上限 30**（31 → HTTP 400）；`page` 从 1 到 **322**，末页 19 条 → 总数 **9649**。
- **没有按定数（`difficulty`）过滤的参数**，分层只能本地做；`level` 是自由文本，**只能用 `difficulty`（f32）做数值分层**（且比较前 round 到 0.1）。
- 每次请求之间加 sleep（0.3–1 s；未观测到限流，但**是否有限流未查证**）。
- 落盘字段：`id, name, level, difficulty, charter, composer, tags, created, updated, chartUpdated, file, preview, illustration`。
- **"上架/ranked"与"全部"必须显式决策**：`stable=true` 仅 627 张、`type=2`（unstable）9022 张 → 建议全用（`type=3`）但把 `stable/ranked` 作为**元数据特征**保留，便于做"只在高质子集上训练"的消融。

### 3.2 全库预筛（**必须先做，省约 90% 带宽**）

- 谱面包 CDN **支持 HTTP Range**（`Range: bytes=0-1023` → `206`）；全量直抓约 **76 GB**（推断）。
- 流程：① `HEAD`/`Range: bytes=0-0` 取总长 → ② 抓 zip **尾部 ~200 KB** 定位 EOCD（`PK\x05\x06`）读**中央目录** → ③ 读 `info.yml` 条目拿 `chart`（谱面文件名）与 `music`（音频名）→ ④ **只 `Range` 取谱面条目前 24 KB 压缩字节**做格式嗅探。
- **格式嗅探启发式**（实测有效但非权威）：出现 `"eventLayers"` → **RPE**；出现 `"notesAbove"`/`"notesBelow"`/`"formatVersion"` → **官谱 JSON**；文本且符合 PEC 行结构 → **PEC**；其余（PBC 等）**直接拒收并记账**。
- 产出直方图（格式 / 大小 / 是否有音频 / 判定线数量）**再决定下载策略**。
- 实测吞吐参考：单线程 ~30 s/张 → 8 线程 ~1.8 s/张；**务必自行限速**。

### 3.3 选择性下载

- ⚠️ **必须按 `info.yml.chart` 定位谱面文件**，**不要"取最大的 .json"**（实测有包内 `.json` 解压总量 294 MB，而谱面文件只有 3.25 MB）。
- ⚠️ **必须按 `info.yml.music` 定位音频**（音频文件名无规律），并按 **sha1 去重**（同曲多谱共享一份音频）。
- ⚠️ **必须重试**：200 张扫描中 7 张（3.5%）出现超时/连接重置 → 指数退避 + Range 断点续传。
- ⚠️ **落盘不要用原始文件名**（实测含全角字符），用 `<chart_id>/<chartfile>`；音频与谱面**分目录存放**，便于"只训结构 / 联合训练"切换。

### 3.4 三类静默陷阱（**错了不会报错，只会静默错位**）

| # | 陷阱 | 证据 |
|---|------|------|
| 1 | **文件后缀完全不可信**：`.json` 里可能是 PEC 文本 | 实测 `chart/7039` 的 `24432296.json` 内容是 PEC |
| 2 | **`info.yml.format` 实测恒为 `null`**，且 196/196 张的谱面文件都**不叫** `chart.json` → 必须按**内容**判型 + 读 `info.yml.chart` 定位 | [survey](knowledges/phira-dataset-survey.md) §4.2/§5.3 |
| 3 | **note `type` 数字在两套格式中含义不同**（RPE 1 Tap/2 Hold/3 Flick/4 Drag；官谱 2 Drag/3 Hold/4 Flick）→ 混用会**静默全员错位** | [phigros-format.md](knowledges/phigros-format.md) §5.2 |

---

## 4. RPEJSON 解析 → Chart IR ⬜（`io/formats/rpejson/` 待建）

> 目标模块 `beatmorph/io/formats/rpejson/` **当前不存在**（`io/formats/` 下只有 `base.py`/`osu.py`/`sm.py`）。
> **实现纪律**：只读 prpr / phichain 的**行为规范**，**独立实现**，不逐行移植（GPL-3.0 / LGPL-3.0 风险，RFC-0029 §4.2）。

### 4.1 解析必须处理的语义

| 项 | 规则 | 出处 |
|----|------|------|
| 时间 | `beat = [1]/[2] + [0]`；`秒 = 60/BPM × beat`；多 BPM 段按 `BPMList` 分段积分（须 `beat2sec(sec2beat(x)) == x` 单测） | [phigros-format.md](knowledges/phigros-format.md) §7.1 |
| 侧别 | **`above == 1 ? FRONT : BACK`**（不得当布尔）；实测取值 `{0,1,2}`，0 与 2 都是背面 | [phigros-format.md](knowledges/phigros-format.md) §5.1/§5.3 |
| 类型 | 按格式分派映射表，**RPE 与官谱不可共用** | [units](knowledges/phigros-units-and-geometry.md) §6.7 |
| 事件轨 | **跨层求和**（不是取最上层）；`null` 层 / 缺失字段 / 缺失 `eventLayers` **三者同一化**；事件间隙**补洞** | [phigros-format.md](knowledges/phigros-format.md) §4.1/§4.3 |
| 嵌套线 | `father != -1` 时位置 = 自身 + 父线（可嵌套）；实测 **26%** 的谱面含嵌套 → **不可跳过** | [phigros-format.md](knowledges/phigros-format.md) §6.3；[survey](knowledges/phira-dataset-survey.md) §7.3 |
| 越界 | `\|positionX\| > 675` **只统计不钳位**（钳位会改变落点分布，违红线 3） | [units](knowledges/phigros-units-and-geometry.md) §7.5 |

### 4.2 契约断言（Phase 1 必做，**默认 CI 内运行、不依赖权重**）

| 断言 | 依据 |
|------|------|
| `above ∈ {0,1,2}` | 实测两轮同时出现 0 与 2 |
| `type ∈ {1,2,3,4}` | RPE A 级源码 |
| `\|positionX\| ≤ 675` | 12674 个 note 实测极值 ±675.000 |
| `1 ≤ len(eventLayers) ≤ 5` | 实测层数直方图 `{1:506, 2:190, 3:9, 4:92, 5:60}` |
| 帧率派生：`T_seq == round(duration_s × rate)`，`rate` 由 config 派生 | 红线 7 / G4 |

### 4.3 微缩夹具

用 **`chart/1000`**（标准 RPE：71 条判定线 / 1659 note / `above` 含 2）与 **`chart/7039`**（伪装成 `.json` 的 PEC，完美负样本）。
⚠️ **必须裁成微缩样本再入库**（红线 5），夹具与 mock **不得固化物理常量**（RFC-0029 §7 硬约束 6）。

---

## 5. MERT 特征离线提取 🟡

**目标口径**（RFC-0029 §6 第 2 条 / §7 硬约束 2）：

- 帧率 **75 Hz**，由模型 config **派生并断言**（`MERTAdapter.output_frame_rate()`，不可读时回落契约常量并**告警**）；
- 缓存**必须带元数据** `{rate, sample_rate, layer, model_rev, duration_s}` 并在**加载时校验**；
- 音频统一重采样到 **24000 Hz**，并在元数据里记录**原始**采样率与时长；
- 长音频按 **5 s 窗 / 1 s 重叠**滑窗，重叠区**按帧率对齐**平均。

**当前状态**：`beatmorph/data/pipeline/embed.py` 的 `extract_mert_embeddings` **存在但仍是 .osu 版**（按 `.osu` 父目录解析谱面对音频），需迁移为"按 `info.yml.music` 定位 + 按 sha1 去重"。

```powershell
# 现状调用形态（v2.x 输入；迁移后输入将改为 Phira 谱面包，接口形态待 v3.0 plan 定稿）
uv run python -c "
from pathlib import Path
from beatmorph.data.pipeline.embed import PreprocessPipeline
PreprocessPipeline(Path('data/raw'), Path('data/processed')).extract_mert_embeddings(Path('data/raw'))
"
```

> 提取后**先做形状/帧率抽检**（`T_seq ≈ duration_s × 75`），再进入 §6。若跳过本步，下游数据集会把缺少特征的样本 skip 并计数入日志。

---

## 6. 强度场构建（`field/`）⬜（待建）

| 要点 | 规则 | 出处 |
|------|------|------|
| 网格 | `Δx = RPE_STAGE_WIDTH / N`，**默认 N = 128 → 10.546875 RPE-x 单位**；常量**必须派生**，不得写死 `10.546875` | RFC-0029 §3.1 |
| 共格碰撞 | 先统计「同线 + 同刻 + 同侧」note 对的**最小 \|ΔpositionX\|**，再定 N（否则 128 只是新魔数）；须补 **N ∈ {64,128,256,512} 消融** | [units](knowledges/phigros-units-and-geometry.md) §7.3 |
| 积分 | **两条路径互校**：① 与强度场**同网格**的数值积分；② **累积强度 Λ(t)** 参数化（Omi et al. 2019）——同一场上二者的 NLL 必须一致到给定容差 | BasePlan §3.4；RFC-0029 §3.2 |
| 目标构建 | 按**事件**遮盖（不是按帧）；**必须显式产出 mask 通道** | BasePlan §3.3 |
| 越界 | 生成侧 `\|x\| > 675` 区域的 λ 置 0（或加越界惩罚），使舞台外落点概率为 0 | [units](knowledges/phigros-units-and-geometry.md) §7.5 |
| G3 基线 | 常数基线是 **`λ = N/\|Ω\|`**，**不是 λ = 0**（后者泊松 NLL = +∞，应写成契约断言） | RFC-0029 §3.2 |

---

## 7. 训练 🟡

### 7.1 ★ 门禁 G1-G4 —— **扩数据之前必须全绿**

门禁模块 [`beatmorph/infra/sanity.py`](../beatmorph/infra/sanity.py) **范式中立**：只吃调用方给的 `step_fn`（跑一步优化并返回标量 loss），不 import torch、不认识任何模型/数据集，因此在最小环境下也能跑，**也就不会被"依赖缺失"跳过**（这正是 G4 当年失败的方式）。

**四道门禁与判据**：

| 门禁 | 函数 | 判据（默认参数） | 抓什么 |
|:---:|------|-----------------|--------|
| **G1** 单 batch 过拟合 | `overfit_single_batch(step_fn, steps=300, target_loss=0.05, target_ratio=0.1)` | 末步 loss `<= max(0.05, 0.1 × 首步)` | 通路断、梯度断、loss 用错 |
| **G2** 打乱标签对照 | `shuffled_target_control(step_fn_real, step_fn_shuffled, steps=300, min_gap_ratio=0.05)` | 打乱后末步 loss `>= 真实 × 1.05` | 输入对目标**零信息**（帧率/对齐类 bug 在此当场现形） |
| **G3** 常数基线 | `constant_baseline_gate(model_loss, baseline_loss, min_improvement=0.1)` | 模型 loss `<= 基线 × 0.9` | 模型其实什么都没学到（停在均值地板上） |
| **G4** 契约断言 | `frame_rate_gate(frames, duration_s, frame_rate, tol_frames=2)` | `frames ≈ duration_s × frame_rate`（`frame_rate` **必须由 config 派生**） | 单位/帧率/采样率漂移 |

**操作步骤（在 1–4 个样本上跑，任一失败都不得扩数据）**：

```python
from beatmorph.infra.sanity import (
    constant_baseline_gate,
    frame_rate_gate,
    overfit_single_batch,
    shuffled_target_control,
    summarize,
)

def step_real() -> float:
    """一步优化并返回标量 loss（调用方负责 backward/step）。"""
    loss = train_step(model, batch)
    return float(loss)

def step_shuffled() -> float:
    """同模型/同输入，但目标被 shuffle。"""
    return float(train_step(model, shuffle_targets(batch)))

results = [
    overfit_single_batch(step_real),                                      # G1
    shuffled_target_control(step_real, step_shuffled),                    # G2
    constant_baseline_gate(model_loss, baseline_loss),                    # G3：λ = N/|Ω|
    frame_rate_gate(emb.shape[1], duration_s, encoder.output_frame_rate()),  # G4
]

log = summarize(results)      # 可直接贴进训练日志的多行文本
assert all(results), results  # 未全绿 → 停止，不要扩数据
```

- 门禁本身的单测（✅ 可执行）：`uv run pytest tests/unit/infra/test_sanity.py -q`。
- **记录要求**：门禁结果必须**写入训练日志**（RFC-0029 §7 硬约束 3）；
  **门禁未绿而扩数据规模**是本项目已经付过一次代价的错误模式（见 §8.1）。

### 7.2 训练启动

| 项 | 状态 |
|----|------|
| ✅ 现有入口 | `uv run beatmorph-train --config-name stage1_planner experiment.max_steps=10000` —— **这是 v2.x 的 planner stage，不可用于 v3.0 训练** |
| 🟡 训练栈 | `infra/trainer.py`（Lightning + bf16-mixed + 梯度裁剪）与 Hydra 分发骨架**可复用** |
| ⬜ 待建 | v3.0 的模型配置与训练 stage（掩码补全 Enc-Dec + 泊松 NLL + mask 通道），须由 infra-agent 在 `configs/` 落地（AGENTS.md §3.4） |

**训练目标的硬约束**（实现时逐条对照，BasePlan §3.4 / RFC-0029 §3.2）：

1. `L = −Σ_n log λ_{k(n)}(e_n) + Σ_k ∫∫∫ λ_k dt dx ds`，`∫λ` **必须与强度场同网格**且分辨率**显式声明**；
2. 强度用 softplus / 指数参数化保证 **λ ≥ 0**；事件项与积分项**同量纲**（都是"计数"）；
3. **不要用 Monte-Carlo 估计 ∫λ**（Jensen 不等式引入偏差）；
4. **热图 focal 与泊松 NLL 不兼容**（目标 y 未归一化，`∫y ≠ 事件数`）→ B1 是**独立消融臂**，损失不得混用；
5. 评估与解码一律在**原始时间域（秒）**。

### 7.3 对照臂 B1-B6（全部必须实现）

| 臂 | 内容 |
|----|------|
| **B1** | 热图 + focal loss（文献主流，**正式消融臂而非稻草人**） |
| **B2** | **主线**：泊松 NLL 强度场 + 掩码补全 |
| **B3** | 离散 event token + 自回归（以 GOCT 配置为骨架） |
| **B4** | **自回归上界臂**（arXiv 2510.03289 对并行采样的质疑要求一个 AR 上界；**不可省**） |
| **B5** | 掩码**离散**扩散（absorbing-state） |
| **B6** | 解码策略消融：`find_peaks` vs **Ogata thinning** |

> 消融**必须指明对照层级**（训练目标 vs 采样/解码策略），否则不可解释（RFC-0029 §3.3）。

### 7.4 评估协议要点

- 事件级 F1 **双容差报告**（±20ms 对 DDC / ±50ms 对 GenéLive!）+ **单独报全谱相位偏移** + **单独报背面 recall**（`side` 强不平衡）；
- **NLL 只作校准指标，不得作质量分数**（ChartGenEval 实测 perplexity 在"常见图案重写"下下降 37%）；
- "响应音乐（能量相关性）"是**探索性指标**，不得作模型选择主判据；
- 贪心一对一匹配，**未匹配的生成事件留在分母**；按谱平均与 micro 平均都报；
- 合法性：同刻按键上限、Hold 区间合法性、**跨线几何冲突**、越界；
- 人评：MIREX 2026 三段式。详见 RFC-0029 §5。

---

## 8. 常见问题（排障 cheatsheet）

### 8.1 ⚠️ 警示案例：25 Hz 事件（**必读**）

| 项 | 内容 |
|----|------|
| **症状** | 119 样本时 loss 1.8 → 0.369，被判"收敛正常"；**扩到万级样本后训练集 loss 完全不可下降**（纹丝不动，不是不收敛也不是过拟合） |
| **根因** | MERT-v1-330M 真实帧率 **75 Hz**（`24000/320`），代码在**三个互相独立的位置**硬编码 **25 Hz**（契约常量、`mert.py` 滑窗拼接、`planner/density.py` 秒→帧换算），且提取路径**不做任何降采样** → "秒→帧"映射整体**错 3×** |
| **后果** | 120 s 的曲子落盘 9000 帧，代码以为时长 360 s：段落 [8,16] s 取到的帧真实时间是 [2.67,5.33] s，最后一段取到 [37.3,40.0] s → **模型在每个段落上看到的是歌曲前 1/3 的音频**，且错位随段落漂移 |
| **为什么没暴露** | ① 唯一能证伪的测试带 `@pytest.mark.gpu` + 权重守卫 → `make test-fast` **永远不跑它**；② mock/fixture 里 4 处**断言 25 Hz 是正确的**（`round(dur*25)`）→ **测试套件在断言这个 bug 是不变量**；③ 没有单 batch / 打乱标签 / 常数基线门禁 |
| **修复** | 契约改为派生式（`MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`），`MERTAdapter.output_frame_rate()` 从 config 推导，删除复制的第三份常量；**唯一能证伪的断言进默认 CI** |
| **制度化** | 新增 **G1-G4 门禁**（`infra/sanity.py`）与 **红线 7**（物理常量必须派生 + 断言；新范式必须过门禁）；**mock 不得固化物理常量** |
| **一句话** | **一个被 mock 覆盖了的物理常量，等价于一个被伪装成事实的假设。** |

出处：[POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md)。

### 8.2 其它常见问题

| 现象 | 原因 / 解决 |
|------|------------|
| `verify_mert_frame_rate.py` 报 `MISMATCH` | `--model-dir` 指向的 `config.json` 不是 MERT-v1-330M（`conv_stride` 累乘必须为 320）；核对权重目录 |
| 特征张量帧数 ≠ `时长 × 75` | 音频未重采样到 24 kHz、或滑窗拼接用了错误的帧率换算 → 跑 §1.5 自检 + 检查 `output_frame_rate()` 是否被绕过 |
| 缓存特征加载时报元数据不匹配 | 设计如此（RFC-0029 §7 硬约束 2）：缓存必须带 `{rate, sample_rate, layer, model_rev, duration_s}`，元数据不符即**拒绝加载**并重抽 |
| 训练 loss 完全不动（万级数据） | 先跑 **G2 打乱标签对照**：若打乱后 loss 不变 → 输入对目标零信息（对齐/帧率类 bug），**不要**先调学习率或换模型 |
| 训练 loss 迅速塌到"全 0" | 稀疏目标的经典塌陷 → 确认用的是**泊松 NLL**（积分项惩罚全 0），而不是朴素 BCE/MSE 热图；对照 **G3 常数基线** `λ = N/\|Ω\|` |
| 解析后 `type` 分布明显偏离实测（Tap 52–63%） | 大概率是**官谱/RPE 的 type 数字混用**（§3.4 陷阱 3）→ 检查是否按格式分派映射表 |
| `side` 全为正面 | `above` 被当布尔解析（`!= 0`）/ 或背面样本被过滤 → 必须写 `above == 1 ? FRONT : BACK`，并在评估中单独报背面 recall |
| 事件值在间隙期间为 0 / 跳变 | `eventLayers` 未补洞，或**取了最上层而不是跨层求和**（[phigros-format.md](knowledges/phigros-format.md) §4.3） |
| 判定线位置整体偏移 | `father != -1` 的嵌套线未合成父线位置（实测 26% 的谱面含嵌套） |
| 谱面文件读不出来 / JSON 解析失败 | 文件后缀不可信（`.json` 可能是 PEC）→ 改为按**内容**判型 + 读 `info.yml.chart` 定位 |
| `PreprocessPipeline` 报大量 "no audio" | 现状实现按 `.osu` 父目录找音频（v2.x 逻辑）；迁移后应按 `info.yml.music` 定位并做 sha1 去重 |
| `pytest` 里 MERT 相关用例全 skip | 需要权重 + CUDA（`@pytest.mark.gpu`）；但**契约级测试不得依赖权重或 GPU**，帧率断言应在默认 CI 内跑（§1.5） |
| 想跳过门禁直接扩数据 | **不允许**：CLAUDE.md 红线 7 —— 新范式必须先过 G1-G4，结果写入训练日志 |

---

## 9. 最小可跑序列（cheatsheet）

### 9.1 今天就能跑的部分 ✅（不需要合规裁定，不落音频、不训练）

```powershell
# 0. 环境
uv sync --extra audio --extra train --extra data --group dev
$env:BEATMORPH_DATA_DIR = "data/processed"; $env:BEATMORPH_RUNS_DIR = "runs"

# 1. 帧率 / 单位自检（期望 5s -> 374 帧 = 75Hz）
uv run python scripts/verify_mert_frame_rate.py --model-dir models/pretrained/m-a-p/MERT-v1-330M

# 2. 契约级测试（不依赖权重/GPU，默认 CI 内跑）
uv run pytest tests/unit/audio/test_frame_rate_contract.py tests/unit/infra/test_sanity.py -q

# 3. 全仓快测试 + lint + 类型
make test-fast
uv run ruff check . ; uv run mypy beatmorph
```

### 9.2 待建后才有（前置：合规裁定 + 组件落地）

```
[裁定数据合规 RFC]                      ← 🔴 在此之前不得训练
   ↓
[Phira 获取脚本]  元数据 322 页 → Range 预筛 → 格式嗅探 → 选择性下载   (⬜ 待建)
   ↓
[RPEJSON 解析器]  io/formats/rpejson/ + beat→秒 + 契约断言            (⬜ 待建)
   ↓
[MERT 特征离线提取]  75Hz 派生 + 缓存元数据校验                        (🟡 迁移中)
   ↓
[强度场构建]  field/：网格 + 两条 ∫λ 路径互校                          (⬜ 待建)
   ↓
[门禁 G1-G4 全绿]  ← 🔴 门禁未绿不得扩数据                             (✅ 模块就绪，待接入新范式)
   ↓
[训练]  掩码补全 Enc-Dec + 泊松 NLL → B1-B6 对照 → 评估                (⬜ 待建)
```

---

## 相关文档

- [BasePlan.md](BasePlan.md) §4 数据 / §5 基础设施 / §7 路线图 / §9 门禁 —— 技术权威
- [CLAUDE.md](../CLAUDE.md) §3 红线 / §6 当前状态
- [decisions/RFC-0029](decisions/RFC-0029-phigros-continuous-chart-generation.md) —— 范式权威（§6 路线 / §7 硬约束 / §8.3 Q11b 合规）
- [POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) —— 25 Hz 事件全文与门禁由来
- [knowledges/phira-dataset-survey.md](knowledges/phira-dataset-survey.md) —— 数据源实测（§6 合规 / §7 多线统计 / §9 流水线建议）
- [knowledges/phigros-format.md](knowledges/phigros-format.md) ｜ [knowledges/phigros-units-and-geometry.md](knowledges/phigros-units-and-geometry.md)
- [CODE_STRUCTURE.md](CODE_STRUCTURE.md) ｜ [glossary.md](glossary.md)
