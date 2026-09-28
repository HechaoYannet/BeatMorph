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
> ### 🔴 约束二：**门禁（G1/G3/G4）必须在扩数据之前跑通 + val 对照判读红线**
>
> 任何新模型/新范式，**先在 1–4 个样本上把门禁跑绿，再谈扩数据规模**；
> 门禁实现在 [`beatmorph/infra/sanity.py`](../beatmorph/infra/sanity.py)，结果必须写入训练日志。
> （RFC-0037，2026-09-28 裁定：原 G2 打乱标签对照已删除；「输入对目标零信息」改由
> **val held-out 对照**承担。**判读红线**：任一 val 点 `val/ratio >= 1` 或
> `val/nll_shuffled_delta <= 0` ⇒ 该 run 的扩大数据规模结论**作废**。）
> 出处：CLAUDE.md 红线 7、BasePlan §9、RFC-0029 §7 硬约束 3、RFC-0037。

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

## 3. 数据获取：Phira API ✅（`scripts/fetch_phira.py`）

驱动脚本已落地（2026-09-27）。三段式：**枚举** → **预筛**（HTTP Range 取中央目录 + 24 KB 前缀嗅探）
→ **选择性下载**（只取 `info.yml` 指名的谱面条目与音频条目）。数据落 `data/processed/**`
（`.gitignore` 全量忽略，**不入库**），每份清单都带 `provenance`（M8 硬约束③，逐张可追溯到 chart id）。

```powershell
uv run python scripts/fetch_phira.py meta                      # 322 页全量枚举 → meta.jsonl
uv run python scripts/fetch_phira.py fetch --limit 1000 --sample-seed 20260927 --workers 8
uv run python scripts/fetch_phira.py pairs                     # 按曲目切分 → pairs.json
uv run python scripts/fetch_phira.py stats                     # 语料统计 → stats.json
```

产出：`meta.jsonl / charts.jsonl / quarantine.jsonl / pairs.json / stats.json / fetch_report.json`
以及 `charts/<chart_id>/<normalized>`、`audio/<sha1>.<ext>`、`features/<sha1>.npz`。

**实测（2026-09-27，本机）**

- 枚举 `count = 9651`（调研基线 9649，+2——社区库仍在长：偏差写进交接件而不是当噪声）；
- 单张谱面包约 10 次 Range 往返（取长 / 尾部 / 中央目录 / `info.yml` / 谱面前缀 / 谱面条目 / 音频分块）；
- **8 并发 + 4 MB 分块 ≈ 20 MB/s**（约 8 GB / 6.5 min），0 网络失败；
- 拒收按 `sniff` / `parse` / `qc` 三类分别记账，**未入库的行留在 `quarantine.jsonl` 可追溯**。

### 3.1 四条硬规则（脚本已内置；改脚本时必须保留）

| # | 规则 | 反面教材（实测） |
|---|------|-----------------|
| R1 | **不得**「取最大的 json」 | id 45756：包内 json 解压总量 294 MB，谱面文件仅 3.25 MB |
| R2 | **不得**依赖默认名 `chart.json` | 196/196 张的谱面文件都不叫这个名字 |
| R3 | **不得**依赖 `info.yml.format` | 实测恒为 null ⇒ 只能按内容嗅探 |
| R4 | 落盘**不得**沿用原始文件名（含全角字符） | 形如 `＃53682.json` ⇒ 规范化为 `<chart_id>/<normalized>` |

### 3.2 三类静默陷阱（**错了不会报错，只会静默错位**）

| # | 陷阱 | 证据 |
|---|------|------|
| 1 | **文件后缀完全不可信**：`.json` 里可能是 PEC 文本 | 实测 `chart/7039` 的 `24432296.json` 内容是 PEC |
| 2 | **`info.yml.format` 实测恒为 `null`**，且 196/196 张的谱面文件都**不叫** `chart.json` → 必须按**内容**判型 + 读 `info.yml.chart` 定位 | [survey](knowledges/phira-dataset-survey.md) §4.2/§5.3 |
| 3 | **note `type` 数字在两套格式中含义不同**（RPE 1 Tap/2 Hold/3 Flick/4 Drag；官谱 2 Drag/3 Hold/4 Flick）→ 混用会**静默全员错位** | [phigros-format.md](knowledges/phigros-format.md) §5.2 |

### 3.3 真实数据暴露的四个坑（2026-09-27，全部已修 + 已加回归测试）

| 坑 | 现象 | 处置 |
|----|------|------|
| **`info.yml` 文本字段实测是 `null`** | `tip: null` / `level: null` 极常见 ⇒ 裸 `str` 声明让 pydantic 拒收整个 `info.yml`：首批 20 张里 **10 张**被误判为「包结构错误」 | `ChartInfo` 加 `mode="before"` 校验：文本字段 null → 空串（`format` 除外——它恒为 null 且只记录） |
| **PEC 语法族比调研样例大** | 旧谱面的 PEC 还有 `&` / `cf` / `cr` 三种符号命令（未识别行首频次 80198 / 49280 / 41644）⇒ 全被记成 `unknown`，污染格式占比统计 | `PEC_COMMANDS` 扩表（判错只影响标签：v1 本就拒收 PEC） |
| **`build_pairs` 把已解析路径写进清单** | `chart_path` 变成 `data/processed/charts/...`，`ChartPairDataset` 再拼一次 chart_dir ⇒ 20/20 行判「谱面缺失」，门禁装配直接失败 | 清单里保持**相对 chart_dir**；回归测试锁定 `build_pairs → load_pairs → 解析` 往返 |
| **token 扩张会拆散长 Hold 的配对** | Hold 起止落在不同 `(k, tau)` token；同 token 里另一个事件被选中时扩张只遮一端 ⇒ 数据集在 index=12 抛「hold 配对点被拆散」 | `close_hold_pairs`：扩张后按 token 收口（仅契约路径 `granularity="event"`） |

### 3.4 提速与排障（实测）

- **必须强制 IPv4**：`phira.5wyxi.com` 只有 A 记录（IPv4），而 httpx/httpcore 会**串行**尝试
  getaddrinfo 的每个地址；本机 IPv6 路由黑洞 ⇒ 每次连接先等满 connect 超时
  （实测 ConnectTimeout / ConnectError / 11.3 s 才拿到 2 MB，整体退化到 ~0.2 MB/s）。
  `build_http_client()` 用 `local_address="0.0.0.0"` 强制 IPv4 后同一 URL **0.10–0.27 s / 2 MB**。
- **代理只给 PyPI，不给 CDN**：本机 PyPI 直连 ~59 KB/s、走代理 ~3.3 MB/s；
  Phira CDN 反过来（直连 2.8–11 MB/s；代理 0–2.8 MB/s，三次里两次直接失败）。
- **失败分级**：传输失败退避重试（`--max-retries`；4xx 不重试）；连续 `--max-failures` 张失败即中止；
  `--workers` 上限 **8**（plan 02 §4 的限速纪律）。
- **可续跑**：已入库 + 已拒收的 chart_id 都会跳过（`--retry-quarantined` 可强制重试）。

### 3.5 尚未查证

- **限流**：未观测到 `X-RateLimit-*` 或 `Retry-After`，但「未观测到」不等于「没有」（plan 02 §9-Q2）。
- **PBC 的准入**：结构完全未查证 ⇒ 一律归 `UNKNOWN` 并记账（plan 02 §偏离 2）。

---

## 4. RPEJSON 解析 → Chart IR ✅（`beatmorph/data/parsers/rpejson.py`，独立实现）

> 目标模块 `beatmorph/io/formats/rpejson/` **当前不存在**（`io/formats/` 下只有 `base.py`/`osu.py`/`sm.py`）。


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

## 5. MERT 特征离线提取 ✅（`scripts/extract_features.py`）

```powershell
uv run python scripts/extract_features.py --dry-run      # 只报账（不加载权重、不写缓存）
uv run python scripts/extract_features.py --limit 24     # 冒烟
uv run python scripts/extract_features.py                # 全量（可续跑：已有缓存跳过）

# ★ 全量推荐：store 压缩（省掉 52% 的纯 CPU zlib）+ 3 分片（按缓存键 sha1 取模，键集不交）
uv run python scripts/extract_features.py --shard 0/3 --npz-compression store --log-every 200
uv run python scripts/extract_features.py --shard 1/3 --npz-compression store --log-every 200
uv run python scripts/extract_features.py --shard 2/3 --npz-compression store --log-every 200
```

> ⚠️ **不要在 GPU 被别的作业占着时开多个分片**：每分片各持一份模型（≈1.6 GB 显存），
> 显存不够会直接 CUDA OOM。分片记账各写 `features_report.shard{i}of{N}.json`，
> 规范口径的 `features_report.json` 由 `--dry-run` 重新生成。

- 音频由清单的 `audio_path` 定位（**内容 sha1** 命名）；**缓存键 = 音频内容 sha1**
  ⇒ 「同曲多谱只抽一次」是结构性的，不靠调用方记得去重。
- 提取前**重算 sha1 并与清单记录比对**，不一致即拒抽并记账（缓存挂到错音频上是静默故障）。
- 编码器默认 `adapter=none`（离线冻结直出）、`layer=12`、FP16，本地权重目录
  `models/pretrained/m-a-p/MERT-v1-330M` 优先（缺失才回落 HF id）。
- 缓存元数据六项校验（rate / sample_rate / layer / model_rev / duration→帧数 / dtype / adapter），
  其中 `rate` 必须 == `MERT_FRAME_RATE_HZ = 75`（派生量，红线 7）。
- **实测吞吐（2026-09-27，RTX 5070 Laptop 8 GB / 115 W 限功耗）**：

  | 配置 | 秒/首 | GPU 利用率 |
  |---|---|---|
  | 单进程 + 默认 deflate（旧口径） | 1.65 | 33–36%（峰谷） |
  | **3 分片 + `store`（生产推荐）** | **0.76** | **96%** |

  瓶颈**不是 GPU**：单首 1.65 s 里 `np.savez_compressed` 占 **52%**（deflate 对 fp16 特征
  只有 1.09× 压缩比，L1/L3/L6 无差别），GPU 前向只占 36%。`store` 把落盘降到 0.015 s
  （39–48×），代价是文件大约 +9%（数组与元数据逐字段一致，加载器校验照常通过）。
  `--pipeline`（预取 + 后台落盘）基准里 2.85×，但**生产上会挂住**（CPU 满载、GPU 0%、
  无输出）⇒ **暂不可用**，见 plan 02 §9 待办。
- 帧数全部满足 `T_seq == round(duration_s × 75)`（例：148.77 s → 11157 帧）。
- 报告落 `data/processed/features_report.json`（含 provenance、失败清单与 dry-run 计数）。

> 下游：`pairs.json` 只收**特征齐备**的行（`build_pairs(require_feature=True)`），
> 因此**全量抽完特征后必须重跑** `scripts/fetch_phira.py pairs`，否则训练清单只有冒烟子集。

---


## 6. 强度场构建（`field/`）✅（网格 / 目标 / 双积分路径 / 泊松 NLL / 碰撞 / 可视化）

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

### 7.1 ★ 门禁 G1/G3/G4 —— **扩数据之前必须全绿**（RFC-0037 起 G2 已删除）

门禁模块 [`beatmorph/infra/sanity.py`](../beatmorph/infra/sanity.py) **范式中立**：只吃调用方给的 `step_fn`（跑一步优化并返回标量 loss），不 import torch、不认识任何模型/数据集，因此在最小环境下也能跑，**也就不会被"依赖缺失"跳过**（这正是 G4 当年失败的方式）。

**三道门禁与判据**（损失一律是 **per-event 归一化**口径，RFC-0037 R2；门禁臂**固定 fp32**，R4）：

| 门禁 | 函数 | 判据（默认参数） | 抓什么 |
|:---:|------|-----------------|--------|
| **G1** 单 batch 过拟合 | `overfit_single_batch(step_fn, steps=300, target_loss=0.05, target_ratio=0.1)` | 末步 loss `<= max(0.05, 0.1 × 首步)` | 通路断、梯度断、loss 用错 |
| **G3** 常数基线 | `constant_baseline_gate(model_loss, baseline_loss, min_improvement=0.1)` | 模型 loss `<= 基线 × 0.9`（基线与模型臂**同除一个** `max(E,1)`） | 模型其实什么都没学到（停在均值地板上） |
| **G4** 契约断言 | `frame_rate_gate(frames, duration_s, frame_rate, tol_frames=2)` | `frames ≈ duration_s × frame_rate`（`frame_rate` **必须由 config 派生**） | 单位/帧率/采样率漂移 |

> **原 G2（打乱标签对照）已删除**（RFC-0037，2026-09-28 决策者裁定）：单批 100 步「速度赛」
> 在真实批上判别力未被证明（证据：RFC-0036 §1-§2、plan 07 §9-54/§9-55）。「输入对目标
> 零信息」改由 **val held-out 对照**在线承担：`val/nll_shuffled_delta`（线内置换、线数归一、
> 128 held-out 窗）+ `val/ratio` + 条件干预 `cond_*_delta`。**判读红线**：任一 val 点
> `val/ratio >= 1` 或 `val/nll_shuffled_delta <= 0` ⇒ 该 run 的扩规模结论作废。

**两种跑法**：① **命令行（推荐，会落盘并强制 fail-closed）**：`uv run beatmorph-train --gates`
（或 `--gates-only` 只跑门禁）；② 手工片段（下文），用于研究单个门禁的行为。

**真实数据上的门禁预算（2026-09-27 第五轮实测，务必先读）**：

- **数据侧先自证不是退化的**（本轮两次假结论都出在这里）：门禁 FAIL 时先看 `gates.txt` 上下文的
  `g1_events` / `g1_lines` / `g3_events` / `g3_lines`。
  - `g1_events = 0` ⇒ G1 是**空过**（曾经报到 0.0014 的「过拟合」）；现在由
    `gates.batch_min_events`（默认 1）兜底：取不到非空批就重抽，重抽不到即抛。
  - `τ 轴终点缺陷`（`chartTime` 虚高，52% 的语料）曾让批里几乎全是空窗，**伪造出**门禁假结论。
    详见 **RFC-0031** 与 plan 02 §9 第四轮；默认口径 `data.tau_end_policy=audio` 已落地，改回
    `chart` 即回旧行为。
- **8 GB 卡上不要让它滑进 Windows 共享内存**（本轮实测的坑）：多个批 + 多个模型同时在场时
  `nvidia-smi` 到 **7.88 / 8.15 GB**、功耗掉到 88 W，**同一个 G3 步从 0.57 s 变成 8.8 s**，
  整轮门禁跑不完（表现为「GPU 100% 但迟迟不结束」）。装配已改为 G3 预计算先跑、跑完立即释放
  （plan 07 §9-27）。判据：`nvidia-smi` 的 `power.draw` 应稳定在 **90 W 上下**；若长期 < 90 W
  且显存贴顶，先怀疑共享内存回退，**不要**盲目加大步数预算。
- **时间预算**：旧实测（含 G2）整轮 ~38-45 min；RFC-0037 删 G2 后预期 **~10-15 min**
  （`--device cuda`；G3 预计算 100 步 × 16 段是大头，门禁臂 fp32 略慢于 bf16）。
  **步时 ∝ (K·T)²**（global 层对 `K·T` 做全自注意力），而 `baseline_chunks=16` 让 G3 的
  **一步 = 16 次串行单样本前向** ⇒ 成本**强依赖 K**：`K=6` 是分钟级、`K≈40` 就是上面这个
  量级。CPU 上是小时级，必须给设备。

  ⚠️ **不要用「跑了几分钟」判断卡死**（2026-09-27 实测踩坑：把两个正在跑的门禁误判为卡死
  掐掉，浪费约 40 min GPU）。门禁现在有**逐门禁进度日志**（`门禁 G1 … 开始 / PASS（耗时）`
  与 `门禁 G3 预计算开始/完成`）——**先看日志，再看功耗**。
  门禁期间 `memory.used ≈ 7.7 / 8.15 GiB` 是当前批口径下的**常态**，不是共享内存回退的特征
  （那是「功耗塌陷 + 步时放大一个量级」）。见 plan 07 §9-41。

  **门禁的数据规模已与训练解耦**（`gates.gate_samples`，默认 200 行），无需再手工传覆盖：

  ```bash
  uv run beatmorph-train --config-name phigros_masked --gates-only --device cuda --skip-env-doctor
  ```

  注意：`gate_samples` 限住的是**数据池**，**限不住 K**（实测 200 行切片与全量切片的最大桶
  K 都是 39）——它让门禁可复现、可对照，但不会让它更快。

- **索引有落盘缓存**（plan 02 §9 ④）：全库（6750 行）首次重建 **36.5 min**，命中缓存后是**秒级**
  （日志会写「索引缓存命中」）。缓存指纹覆盖配置与文件 stat，改配置/换谱面会自动重建。

`gates.txt` 里除了 `summarize()` 原文，还会写**本次生效的阈值**（例如 `g1_steps`、`g3_samples`、
`g3_min_improvement`）与上下文（git rev / 数据来源 / 派生帧率）——因为门禁阈值对量纲敏感，
「默认值」不等于「本次用的值」（plan 07 §3.1 / §9-2）。

**操作步骤（在 1–4 个样本上跑，任一失败都不得扩数据）**：

```python
from beatmorph.infra.sanity import (
    constant_baseline_gate,
    frame_rate_gate,
    overfit_single_batch,
    summarize,
)

def step_real() -> float:
    """一步优化并返回标量 loss（调用方负责 backward/step）。"""
    loss = train_step(model, batch)
    return float(loss)

results = [
    overfit_single_batch(step_real),                                      # G1
    constant_baseline_gate(model_loss, baseline_loss),                    # G3：λ = N/|Ω|（同除 max(E,1)）
    frame_rate_gate(emb.shape[1], duration_s, encoder.output_frame_rate()),  # G4
]

log = summarize(results)      # 可直接贴进训练日志的多行文本
assert all(results), results  # 未全绿 → 停止，不要扩数据
# 扩规模后的持续判读（原 G2 的命题所在）：每个 val 点必须 val/ratio < 1
# 且 val/nll_shuffled_delta > 0，否则该 run 的扩规模结论作废（RFC-0037 R1）。
```

- 门禁本身的单测（✅ 可执行）：`uv run pytest tests/unit/infra/test_sanity.py -q`。
- **记录要求**：门禁结果必须**写入训练日志**（RFC-0029 §7 硬约束 3）；
  **门禁未绿而扩数据规模**是本项目已经付过一次代价的错误模式（见 §8.1）。

### 7.2 训练启动

| 项 | 状态 |
|----|------|
| ✅ **门禁冒烟（今天就能跑）** | `uv run beatmorph-train --config-name smoke --gates-only` —— 合成谱、无权重、无网络、无 GPU，**CPU 约 1-2 分钟**跑完 G1/G3/G4 并把六件套写进 `runs/smoke/<时间戳>/`（G2 已删除，RFC-0037） |
| ✅ **训练入口** | `uv run beatmorph-train --config-name phigros_masked --gates`（真实清单 + 特征缓存）。`--gates` 先跑 G1/G3/G4，**任一 FAIL 即以退出码 5 中止**；不跑门禁而数据规模超过冒烟上限时 **fail-closed 拒绝启动** |
| ✅ 训练栈 | `beatmorph/infra/train_loop.py`（torch 参考循环，默认 `run.backend=torch`）；`beatmorph/infra/lightning_module.py`（Lightning 目标栈，需 `uv sync --extra train`，缺失时给出安装命令而**不静默回落**） |
| ✅ 环境自检 | `uv run python -m beatmorph.infra.env_doctor`（退出码 0/1/2 = 全 PASS / 有 FAIL / 有 UNKNOWN）；训练入口默认先跑它，可用 `--skip-env-doctor` 跳过（仅测试/容器） |
| ⬜ 真实数据的清单与特征 | 仍待 plan 02 的 Phira 获取脚本 + 特征提取（§3–§5）：**没有它们就没有真实训练批次**，`data.source=manifest` 会在清单不存在时直接报错 |

**退出码语义**（`beatmorph/cli/train.py`）：0 成功｜2 参数错误｜3 配置错误（含 provenance 为空）｜
4 环境自检 FAIL｜5 门禁 FAIL / fail-closed 拒绝启动｜6 缺少可选依赖｜7 训练异常。

**实验产物（六件套，缺一即视为实验不可信）**：`config.yaml` / `gates.txt`（含**生效阈值**与
git/data rev）/ `data_provenance.json`（来源与用途）/ `checkpoints/` / `logs/` / `metrics.json`。

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

### 7.5 长跑运维（断点续训 / 巡检 / TB / 显存墙）

全量训练是**十小时级**的单个进程。下面五件事每一条都在短跑里看不出问题、只在长跑里咬人。

**① 存盘节奏与旋转**

- `run.save_every=2000`（= 每 4 轮，1 轮 = `optim.val_every`）+ `run.keep_last=3` + `run.keep_best=1`：
  崩溃最多丢一个间隔，磁盘只留最新 3 个步级 checkpoint 加一个 `best.pt`（**先写后删**）。
- 目录：`runs/<experiment>/<timestamp>/checkpoints/step-NNNNNNN.pt`（**文件名就是步号**，恢复靠它）。
- **`best.pt` 现在按 `val/ratio` 选**（2026-09-27 §9-51；此前按训练损失选，而实测那等于
  「按谁抽到最空的窗选」——最小 loss 落在 step 6234 的 `events=0, K=1` 空窗上）。
  新口径在**测出更优 `val/ratio` 的那一步**立即写盘，不随 `save_every`。无 val 时回落旧口径，
  且在日志里明说「不作为模型选择依据」。

> ⚠️ **开 val 之前必须读这一段（实测开销，不是估算）**：§9-47 E 原先算的一次 val ≈ `val_every`
> 预算的 **5.8%**，**实测是 0.7–1.5×**（即一次 val 的开销与它之间那 1000 步的训练**同量级**）。
> 三个原因，缺一不可地都被低估了：① 对照臂是 **5 次前向**（E 只算了 1 次）；② 数据侧实测
> **0.276 s/窗**（workers=8），不是 E 的 0.038 s；③ val 前缀抽到的 **K 比训练流重**
> （p90 **106** vs 训练流 53–67），而前向成本 ∝ K²。
> ⇒ 当前 `optim.val_every=1000 / val_windows=128` 会让训练**慢一倍以上**。要压回 5% 需在三者里
> 裁定：`val_windows≈32` / `val_every≈4000` / **对照臂降频**（条件干预三元组不必每次 val 都跑）。
> 详见 plan 07 §9-51 ③。

**② 断点续训**

```bash
# 从本实验最近一个「有 checkpoint」的目录续跑（不重跑门禁，但要求原目录 gates.txt 全绿）
uv run beatmorph-train --config-name phigros_masked --resume latest --device cuda --skip-env-doctor
# 指定目录
uv run beatmorph-train --config-name phigros_masked --resume 20260927-120000 --device cuda --skip-env-doctor
```

- 恢复前 `load_checkpoint` 逐项校验：**配置指纹 / 门禁全绿 / data_rev**，任一不符即**拒绝**（退出码 7）。
  指纹只覆盖**语义字段**；预算与落盘策略（`optim.max_steps`、`run.save_every`/`log_every`/`keep_*`）
  变更**不会**让旧 checkpoint 失效——否则「想多跑几步」就得另起一个实验目录，把同一次训练劈成两半。
- 优化器状态（AdamW 动量）一并恢复；**随机数状态没有保存**（数据顺序/遮盖种子由 index 与 seed 派生，
  因此同一 index 的样本内容仍然一致，但 dropout 的随机流不同）。

**③ 每 2 小时巡检**

```bash
uv run python scripts/training_health.py --experiment phigros_masked --gpu              # 只查一次
uv run python scripts/training_health.py --experiment phigros_masked --gpu --watch 7200 # 每 2 小时自动巡检
```

它把「有没有在跑 / 有没有降速 / 显存有没有贴顶 / checkpoint 有没有落后 / ETA /
**瓶颈在哪一侧**」压成一行结论：退出码 `0` 健康 / `1` 有告警 / `2` 无数据。
预警规则：步时后半窗比前半窗慢 1.3× 以上、**数据侧占比 ≥ 50%**、峰值显存 ≥ 7.5 GiB、
标量超过 30 分钟没更新、**checkpoint 真漏存**（间隙 > @@@@run.save_every@@@@；间隔内「当前步 >
最近 checkpoint」是**正常状态**，不告警）。

**③b 「GPU 利用率不高」怎么读（2026-09-27 第九轮，[RFC-0034](decisions/RFC-0034-data-supply-throughput.md)）**

先说结论：**训练路径里没有 worker 这个旋钮**。全仓 `num_workers` 只出现在数据准备脚本
（`scripts/fetch_phira.py` / `extract_features.py`）；训练侧唯一的 `DataLoader` 在
`beatmorph/infra/lightning_module.py` 且 `num_workers` 默认 **0**。真实取批由
`ManifestBatchSource.batch()` 在**训练进程主线程里同步**完成，而 `step_time_s` 的计时起点
就在它之前 ⇒ 数据构建时间**一直混在步时里**。

因此本轮把步时**拆开**，只看拆分，不要靠功耗猜：

| 标量 | 含义 |
|---|---|
| `train/data_time_s` | 选桶 → 取窗口 → **解析谱面** → 建场 → collate（**全在主线程**，此刻 GPU 是空的） |
| `train/compute_time_s` | H2D + 前向 + 反向 + 裁剪 + 优化器（`loss.detach()` 取值即同步点） |
| `perf/data_share` | `data_time_s / (data_time_s + compute_time_s)` |

判读（实测基线，step 1850/20000，RTX 5070 Laptop 8 GB / 115 W 上限）：

- **双峰是正常的**：GPU 利用率均值 35.7%、**中位 5.0%**、29/60 采样 ≤1%，功耗均值 21.4 W
  = 上限的 **18.6%**。中位 5% 表示**一半以上时间 GPU 什么都没干**，这与「功耗贴顶」是两回事。
- **长尾来自 K²，不是数据侧**：K 中位 26 / p90 67 / max 128，单步注意力 ∝(K·T)²；
  1850 步里 **25.8% 的步 ≥1 s**（p99 4.52 s / max 6.83 s），而中位步只有 0.539 s。
- **判据**：`data_share` 中位**持续 ≥50%** ⇒ 瓶颈在**数据管道**，加卡/加 batch 都不解决
  （巡检脚本会直接告警）；`data_share` 低而成长期步时高 ⇒ 那是 K 分布，属正常。

**④ TensorBoard（对接人力监控）**

```bash
uv run tensorboard --logdir runs/phigros_masked
```

标量每 `run.log_every` 步**增量**刷盘（`train/loss` / `train/step_time_s` /
**`train/data_time_s` / `train/compute_time_s` / `perf/data_share`** / `train/grad_norm` /
`train/lr` / `sys/peak_vram_gib` / `batch_n_lines` / `batch_events` / `coverage/*`），因此曲线是**在线**的；
同样的行同时追加进 `logs/loss_history.jsonl`（权威、可脚本读，TB 缺失也不影响训练）。

**④b 数据覆盖率（必看，2026-09-27 第八轮新增；[RFC-0033](decisions/RFC-0033-sampler-coverage-and-epoch.md)）**

同一条 jsonl 里还有 `epoch` / `windows_seen` / `charts_seen`（TB 里是
`coverage/epoch` / `coverage/windows_seen` / `coverage/charts_seen`）。**这三个数不是装饰**：

> ⚠️ 旧采样器（`ManifestBatchSource` 的全局共享游标）在 **1000 步后彻底饱和**——
> 之后无论 `max_steps` 加到多少，永远只抽同一批 **993 个窗口（全库 0.156%）/ 674 张谱面（10%）**。
> **loss 曲线完全看不出来**（它只是反复拟合同一小撮样本）。本轮修好后才有了覆盖率可看。

- **`epoch`**：一个 epoch = **走遍全库、每个窗口恰好一次** = `len(dataset)` 步
  （train split = **634,952** 步）。**本机跑不完一个 epoch**（≈88 h @0.5 s/步）。
- **`coverage/charts_seen`**：累计去重的谱面覆盖率（只增不减）。这才是「训练到底见了多少音乐」。
- **判据**：`max_steps=20000` 时 **≈91% 谱面覆盖**（≈3.15% window-epoch）。
  若这条曲线**平掉**（长时间不涨）⇒ 采样器又饱和了，立刻停并查
  `tests/unit/infra/test_sampler_coverage.py` 覆盖的几条不变量，**不要**靠加步数掩盖。

**⑤ 显存墙：上限是多少、怎么判降速（2026-09-27 第六轮实测）**

**先说结论：本机 8 GB（可用 6.85 GiB）在 `t_window=192`、fp32 下的安全上限是 `K ≈ 28-31`，
物理上限 `K ≈ 38`。而 train split 有 36.4% 的谱面 `K > 31`、27.9% 的 `K > 38` ⇒
`data.k_max=128` 不是保护，直接开全量训练会在最初几个样本上就 OOM。**

实测曲线（生产配置，B=1，`scripts/local_vram_wall.py` 单步探针）：

| K | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| 峰值显存 GiB | 0.81 | 1.24 | 1.84 | 2.46 | 3.30 | 4.13 | 5.05 |
| 步时 s | 0.066 | 0.109 | 0.167 | 0.249 | 0.311 | 0.424 | 0.550 |

- 拟合：`峰值 ≈ 1.05 + 0.00391·K² GiB`、`步时 ≈ 0.00054·K² s`（K=32 是独立校核点，预测/实测都是 5.05 GiB）。
  `K` = 该样本的判定线条数，`T = t_window`；**训练是 `batch_size=1`，所以单个样本的 K 就是上限**。
- ⚠️ **`(K·T)²` 不只在 global 层，大头在「局部」层**：6 层全翻局部比全翻全局贵 **2.4×**
  （K=20：2.92 vs 1.22 GiB）。机制已定位：局部层的轨道 cross-attention 的 K/V 是
  `tracks.repeat_interleave(K)`（key 轴 = `K · T_line`）⇒ **每条线的 token 会 attend 全部 K 条线的事件轨**，
  「局部层」在判定线轴上其实是全连通的，与模型 docstring 的表述不符。把轨的时间轴压到 1 格后，
  全局部层 2.92 → **1.29 GiB**、步时 1.169 → **0.064 s**（18×）。
  **是有意设计还是漏改，须开 RFC 裁定**（plan 07 §9-35）——先别按旧叙述做架构决定。
- **`t_window` 192→96** 把上限抬到 `K ≈ 53`（不是 4×，有常数项），单步快 2.04×——但**改的是任务口径**。
- **bf16 autocast 是现成的杠杆**：`optim.precision: bf16-mixed` 早已声明，但 `train()` 从未启用
  （全仓无 `autocast`）。补测：K=32 峰值 5.05 → **2.92 GiB**、步时 0.550 → **0.269 s** ⇒ 上限抬到 `K ≈ 42-47`。
  注意 bf16 **首步**要 1.4-3.0 s（内核编译），别把首步当稳态。启用前必须重跑门禁（G1/G3/G4）。⚠️ RFC-0037 R4：门禁臂**固定 fp32**（bf16 重复噪声 2.8× 会让读数不可解释），训练精度不受影响。

> ⚠️ **2026-09-28 实测更正（plan 07 §9-57）**：上表的「峰值显存」是 **torch 分配器口径
> （`max_memory_allocated`）**，**不含** CUDA 上下文 / cuBLAS·cuDNN workspace / **分配器碎片**。
> 首次全量训练实测：K=128 的批把 torch 峰值顶到 **5.23 GiB**，但**驱动侧总量**到
> **7874 / 8151 MiB**，下一步就滑进共享显存（决策者观测 **13.2 GB** 共享内存），step 951 卡死。
> ⇒ **判据必须是驱动侧**（`nvidia-smi memory.used`），不是 `sys/peak_vram_gib`。已落地的三个杠杆：
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`optim.vram_hygiene_gib`（保留量远超实际
> 分配时回收缓存，见 `sys/vram_reserved_gib`）、`optim.val_batch: 8 → 4`。

**降速怎么判（贴顶是静默的）**

- 显存到 ~7.9/8.15 GB 时驱动滑进 Windows 共享内存：利用率仍显示 100%，但 `power.draw` 从 ~100 W 掉到 **~31 W**
  （2026-09-28 实测），**步时放大一个量级**（实测 0.57 s → 8.8 s，最坏是**直接卡死**）。
- 判据：**`nvidia-smi memory.used` > ~7.4 GiB** 或 `power.draw` 长期 < 45 W 且显存 > 5 GiB ⇒ 立刻停。
  巡检脚本已把这个判据做成告警（`scripts/training_health.py`，峰值 ≥ 7.5 GiB / 步时后半窗慢 1.3×）。
  ⚠️ **不要靠自建看门狗自动杀**：2026-09-28 的 v1 版本因「跨 run 的步号基准」误杀过一次健康运行
  （plan 07 §9-57 事故 2）——决策者已指示**由人工盯**并主动通知。
- **改 `t_window` / 放开 `data.k_max` / 启用 bf16 之前**：先按上表外推，超过 5 GiB 就先停下来开 RFC 或调配置。

**⚠️ 硬件安全（2026-09-27 的事故，写给下一个 session）**

一次「扫 K vs 显存」的 GPU 探测把 8 GB 笔记本卡拖到驱动层 TDR，**Windows 被重启**。三条禁令：

1. **不要**用 `torch.nn.attention.sdpa_kernel([...])` 强制注意力后端（会绕过安全回退路径）；
2. **不要**跑无界的「连续前向+反传」扫批脚本；任何 GPU 探测都要有显存余量检查与单次上限；
3. 同一时刻**只跑一个** GPU 作业（不要与门禁/训练并行），并全程盯 `nvidia-smi` 的功耗与显存。

（另：硬重启会弄坏 `.mypy_cache`。若 `uv run mypy` 报 `INTERNAL ERROR ... database disk image is
malformed`，删掉 `.mypy_cache` 重跑即可——那是缓存损坏，不是代码问题。）

## 8. 常见问题（排障 cheatsheet）

### 8.1 ⚠️ 警示案例：25 Hz 事件（**必读**）

| 项 | 内容 |
|----|------|
| **症状** | 119 样本时 loss 1.8 → 0.369，被判"收敛正常"；**扩到万级样本后训练集 loss 完全不可下降**（纹丝不动，不是不收敛也不是过拟合） |
| **根因** | MERT-v1-330M 真实帧率 **75 Hz**（`24000/320`），代码在**三个互相独立的位置**硬编码 **25 Hz**（契约常量、`mert.py` 滑窗拼接、`planner/density.py` 秒→帧换算），且提取路径**不做任何降采样** → "秒→帧"映射整体**错 3×** |
| **后果** | 120 s 的曲子落盘 9000 帧，代码以为时长 360 s：段落 [8,16] s 取到的帧真实时间是 [2.67,5.33] s，最后一段取到 [37.3,40.0] s → **模型在每个段落上看到的是歌曲前 1/3 的音频**，且错位随段落漂移 |
| **为什么没暴露** | ① 唯一能证伪的测试带 `@pytest.mark.gpu` + 权重守卫 → `make test-fast` **永远不跑它**；② mock/fixture 里 4 处**断言 25 Hz 是正确的**（`round(dur*25)`）→ **测试套件在断言这个 bug 是不变量**；③ 没有单 batch / 打乱标签 / 常数基线门禁 |
| **修复** | 契约改为派生式（`MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`），`MERTAdapter.output_frame_rate()` 从 config 推导，删除复制的第三份常量；**唯一能证伪的断言进默认 CI** |
| **制度化** | 新增 **G1-G4 门禁**（`infra/sanity.py`；G2 后于 2026-09-28 随 RFC-0037 删除，现为 G1/G3/G4）与 **红线 7**（物理常量必须派生 + 断言；新范式必须过门禁）；**mock 不得固化物理常量** |
| **一句话** | **一个被 mock 覆盖了的物理常量，等价于一个被伪装成事实的假设。** |

出处：[POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md)。

### 8.2 其它常见问题

| 现象 | 原因 / 解决 |
|------|------------|
| `verify_mert_frame_rate.py` 报 `MISMATCH` | `--model-dir` 指向的 `config.json` 不是 MERT-v1-330M（`conv_stride` 累乘必须为 320）；核对权重目录 |
| 特征张量帧数 ≠ `时长 × 75` | 音频未重采样到 24 kHz、或滑窗拼接用了错误的帧率换算 → 跑 §1.5 自检 + 检查 `output_frame_rate()` 是否被绕过 |
| 缓存特征加载时报元数据不匹配 | 设计如此（RFC-0029 §7 硬约束 2）：缓存必须带 `{rate, sample_rate, layer, model_rev, duration_s}`，元数据不符即**拒绝加载**并重抽 |
| 训练 loss 完全不动（万级数据） | 先看 **val 对照**（RFC-0037 起）：`val/nll_shuffled_delta ≈ 0` 或 `cond_*_delta ≈ 0` → 输入对目标零信息（对齐/帧率类 bug），**不要**先调学习率或换模型；再核 `val/ratio` 是否 < 1 |
| 训练 loss 迅速塌到"全 0" | 稀疏目标的经典塌陷 → 确认用的是**泊松 NLL**（积分项惩罚全 0），而不是朴素 BCE/MSE 热图；对照 **G3 常数基线** `λ = N/\|Ω\|` |
| 解析后 `type` 分布明显偏离实测（Tap 52–63%） | 大概率是**官谱/RPE 的 type 数字混用**（§3.4 陷阱 3）→ 检查是否按格式分派映射表 |
| `side` 全为正面 | `above` 被当布尔解析（`!= 0`）/ 或背面样本被过滤 → 必须写 `above == 1 ? FRONT : BACK`，并在评估中单独报背面 recall |
| 事件值在间隙期间为 0 / 跳变 | `eventLayers` 未补洞，或**取了最上层而不是跨层求和**（[phigros-format.md](knowledges/phigros-format.md) §4.3） |
| 判定线位置整体偏移 | `father != -1` 的嵌套线未合成父线位置（实测 26% 的谱面含嵌套） |
| 谱面文件读不出来 / JSON 解析失败 | 文件后缀不可信（`.json` 可能是 PEC）→ 改为按**内容**判型 + 读 `info.yml.chart` 定位 |
| `PreprocessPipeline` 报大量 "no audio" | 现状实现按 `.osu` 父目录找音频（v2.x 逻辑）；迁移后应按 `info.yml.music` 定位并做 sha1 去重 |
| `pytest` 里 MERT 相关用例全 skip | 需要权重 + CUDA（`@pytest.mark.gpu`）；但**契约级测试不得依赖权重或 GPU**，帧率断言应在默认 CI 内跑（§1.5） |
| 想跳过门禁直接扩数据 | **不允许**：CLAUDE.md 红线 7 —— 新范式必须先过门禁（G1/G3/G4），结果写入训练日志；扩规模后还须守住 val 对照判读红线（RFC-0037） |

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

# 3. 环境自检（E1-E5；退出码 2 = 有 UNKNOWN，例如没装 train extra）
uv run python -m beatmorph.infra.env_doctor

# 4. 门禁冒烟：跑通 G1/G3/G4 并落盘六件套（合成数据，数秒）
uv run beatmorph-train --config-name smoke --gates-only

# 5. 全仓快测试 + lint + 类型
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
[门禁 G1/G3/G4 全绿] ← 🔴 门禁未绿不得扩数据      (✅ beatmorph-train --gates 已接线，fail-closed)
   ↓
[训练]  掩码补全 Enc-Dec + 泊松 NLL → B1-B6 对照 → 评估                (🟡 训练栈与数据通路已通；
        B1-B6 对照臂与评估仍待建；真实批次待 §3–§5 的脚本)
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
