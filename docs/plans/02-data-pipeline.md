# Plan 02 — 数据流水线：Phira 谱面获取 + RPEJSON 解析 + 质检 + 特征离线提取

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：数据组（data-agent）
> 对应代码：`beatmorph/data/`（`phira/client.py`、`phira/package.py`、`parsers/sniff.py`、`parsers/rpejson.py`、`qc.py`、`pipeline/embed.py`） ｜ 对应奠基章节：§4、§3.2.4、§9

## 1. 目标与范围

### 交付

1. **元数据枚举**：Phira 官方 API `GET https://api.phira.cn/chart` 全量分页（实测 `count = 9649`、322 页、`pageNum` 上限 30）→ 元数据表落盘。
2. **包内结构预筛**（不下载整包）：HTTP Range 抓 zip 中央目录 + 只取谱面条目前缀 → 得到谱面文件名、展平大小、格式嗅探结果，**省约 90% 带宽**。
3. **格式内容嗅探** `sniff_format()`：只按**内容**判定 `RPE | PEC | OFFICIAL | PBC | UNKNOWN`，**不看后缀、不看 `info.yml.format`**。
4. **RPEJSON 解析器（⭐ 独立实现）** → `PhigrosChart`（plan 00 契约），含 beat→秒 换算、事件跨层求和与补洞、`father` 嵌套、note 标记映射。
5. **质检** `quality_check()`：schema 级 + 单位级 + 分布级三层规则，产出 `qc_report` 与隔离区。
6. **特征离线提取**：复用 plan 01 的 `MertAudioEncoder` + 缓存契约（元数据 `{rate, sample_rate, layer, model_rev, duration_s}` 并在加载时校验），按音频 hash 去重。
7. **配对与切分**：~(audio, chart)` 对构建、**按曲目**切分 train/val/test、唯一曲目与同曲重复率统计。
8. **微缩夹具**：`chart/1000`（标准 RPE）与 `chart/7039`（伪装成 `.json` 的 PEC）。

### 不交付

- 模型训练（Adapter 训练归 plan 01；强度场与生成主干归 plan 03 / plan 04）；
- RPEJSON **写出**（plan 05，`io/formats/rpejson/` 写侧）——本模块只读；
- DEMUCS/分轨、官谱 JSON 与 PBC 的完整解析（v1 主路径锁定 RPEJSON，其余只做嗅探识别与记账）；
- 音频版权合规的**裁定本身**（**已由决策者裁定，2026-08-05**；本模块只落 §6-M8 的硬约束执行与追溯记录）。

### 价值

本模块是「自监督理解取代显式标注」的燃料入口：把 Phira 上 9649 张社区谱面加工成 **(audio, chart) 训练对**。它同时承载三个**静默**陷阱的防线（后缀、文件名、type 数字）——这三个错了不会报错，只会让整条流水线安静地训练出错误的东西。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| Phira API 全量枚举（9649 / 322 页 / `pageNum ≤ 30`） | [BasePlan §4.1](BasePlan.md)、[phira-dataset-survey.md §2/§3](knowledges/phira-dataset-survey.md) | — | 实测事实，直接采用 |
| 响应键取 `results`（非文档写的 `result`） | 调研 §2.1.1 | — | C 级文档与 A 级实测冲突时**以实测为准** |
| CDN 支持 HTTP Range → 结构预筛 | [BasePlan §4.1](BasePlan.md)「带宽优化」、调研 §7.6 | — | 全量直抓约 76 GB（推断），Range 预筛压到 ~10 KB/张 |
| 后缀不可信 / 必须读 `info.yml.chart` / type 两套数字 | [BasePlan §4.2](BasePlan.md)、[RFC-0029 §6](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | 三类静默陷阱，逐条进里程碑 |
| RPEJSON → Chart IR（`PhigrosChart`） + beat→秒 | [BasePlan §4.3](BasePlan.md) 流水线图 | — | 时间统一到**秒**（秒域评估，RFC-0029 §7-5） |
| 解析器**独立实现**（只读行为规范） | [BasePlan §4.4](BasePlan.md)、[RFC-0029 §4.2](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | prpr **GPL-3.0** / phichain **LGPL-3.0**，逐行移植有衍生作品风险 |
| 质检：区间校验、越界统计、量纲断言 | [BasePlan §4.3](BasePlan.md)、[phigros-units-and-geometry.md §7.5](knowledges/phigros-units-and-geometry.md) | — | **越界只统计不钳位**（红线 3） |
| MERT 特征离线提取（75 Hz，元数据随缓存落盘并校验） | [BasePlan §4.3/§7](BasePlan.md)、[RFC-0029 §6-2/§7-2](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | 帧率**派生量**；缓存元数据是防线 |
| 去重：同曲多谱为独立样本，但须统计唯一曲目数与重复率 | [BasePlan §4.5](BasePlan.md) | — | 真实 (audio, chart) 对数远小于 9649 |
| ✅ **数据合规已裁定（2026-08-05）：训练可启动**（原「裁定前不启动训练」的阻塞闸门解除）；**硬约束 =** ① 最终不发布模型权重（裁决前提）② 谱面/音乐可本地落盘但**不得入库** ③ 获取/处理脚本**记录来源与用途** ④ 发布权重前必须重新裁定 | [BasePlan §4.4](BasePlan.md)、[RFC-0029 §7-7/§8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md)、CLAUDE.md 红线 5 附注 | — | 项目级风险 R-2 由**决策者**承担；四条硬约束是本模块的工程落点（M8） |

**偏离 1（切分口径）**：BasePlan §4.5 只说「同一曲目多张谱视为独立样本」，未规定切分单位。本 plan 规定**按曲目（`name` + `composer` 近似）切分**，同曲多谱必须落在同一 split，否则 val 泄漏；并额外保留一个「同曲跨谱泛化」评测集（train 用某曲 IN 谱、test 用同曲 AT 谱）。依据：调研 §9.2 [8]。

**偏离 2（官谱 JSON / PBC 处置）**：v1 主路径锁定 RPEJSON。官谱 JSON 与 PBC **只做嗅探识别 + 计数记账**，不实现解析（官谱 type 映射仅 C 级来源；PBC 结构完全未查证）。拒收样本必须**显式记账**，不得静默丢弃。

**偏离 3（难度字段口径）**：BasePlan 未指定难度来源字段。本 plan 规定数值分层只用 `info.yml.difficulty`（f32，比较前 round 到 0.1），**禁止** regex 解析 `level` 自由文本（实测形如 `"AT  Lv.16"` / `"sweet"` / `"酔い"` / `"14.514"`）。依据：调研 §3.2。

## 3. 接口契约

### 3.1 常量（实测事实，非物理常量；物理量一律派生）

```python
PHIRA_API_BASE: str = "https://api.phira.cn"
CHART_LIST_ENDPOINT: str = "/chart"
CHART_LIST_RESULTS_KEY: str = "results"     # ⚠️ 非文档写的 "result"（实测）
CHART_PAGE_SIZE_MAX: int = 30                # 实测 pageNum=31 → HTTP 400
CHART_TOTAL_EXPECTED: int = 9649             # 2026-09-26 实测 count；作为枚举完整性断言，非硬编码语义
RANGE_PREFIX_BYTES: int = 24 * 1024          # 谱面条目嗅探用的压缩前缀（实测有效）
ZIP_TAIL_BYTES: int = 200 * 1024             # 中央目录读取窗口（实测有效）
REQUEST_INTERVAL_S: float = 0.5             # 自限速：未观测到限流 ≠ 无限流（调研 Q-2）
CHART_FILE_FIELD: str = "chart"              # info.yml 中定位谱面文件的字段（不得按名猜）
CHART_MUSIC_FIELD: str = "music"             # 音频文件名的唯一来源
```

### 3.2 客户端与包访问（`beatmorph/data/phira/client.py`、`package.py`）

```python
class PhiraClient:
    def iter_chart_meta(self, type: int = 3, sleep_s: float = REQUEST_INTERVAL_S) -> Iterator[ChartMeta]:
        """分页枚举全部谱面元数据。type=3 为 any（= 全部 9649）；每页 pageNum=CHART_PAGE_SIZE_MAX。

        断言：results 键存在；累计条数 == count（最后一页的 count 与首页一致）。
        """

    def fetch_zip_index(self, file_url: str) -> ZipIndex:
        """HTTP Range 取 zip 尾部 → 解析 EOCD/中央目录 → 条目表（名、压缩/解压大小、本地头偏移）。"""

    def fetch_prefix(self, file_url: str, entry: ZipEntry, n: int = RANGE_PREFIX_BYTES) -> bytes:
        """Range 取条目压缩前缀并经 zlib.decompressobj(-15) 部分解压（截断流可解出前缀）。"""

    def download_entry(self, file_url: str, entry: ZipEntry, dest: Path) -> Path:
        """只下载**单个条目**（如谱面文件）并校验解压大小；带指数退避重试 + Range 断点续传。"""
```

```python
class ChartPackage:
    """一个谱面包 = zip：info.yml + 谱面文件 + 音频 + 曲绘（+ 可选 extra.json/贴图/着色器）。"""
    chart_file: str        # **只来自** info.yml[CHART_FILE_FIELD]；不得按后缀或大小猜
    music_file: str        # **只来自** info.yml[CHART_MUSIC_FIELD]
    entries: dict[str, ZipEntry]
    def chart_bytes(self) -> bytes: ...
```

**硬规则（负样本测试的对象）**

| # | 规则 | 反面教材（实测） |
|---|------|-----------------|
| R1 | 包内 `.json` 总量可达数百 MB（特效资源），**不得「取最大的 json」** | id 45756：包内 json 解压总量 294 MB，而谱面文件仅 3.25 MB |
| R2 | **不得**依赖文件名 `chart.json`（默认名） | 196/196 张的谱面文件都**不叫** `chart.json` |
| R3 | **不得**依赖 `info.yml.format` | 实测恒为 `null` → 只能按内容嗅探 |
| R4 | 落盘**不得**沿用原始文件名（含全角字符） | 实测形如 `1817439042209534.json`、`＃53682.json` → 规范化为 `<chart_id>/<normalized>` |

### 3.3 格式嗅探（`beatmorph/data/parsers/sniff.py`）

```python
class ChartFormat(StrEnum):
    RPE = "rpe"; PEC = "pec"; OFFICIAL = "official"; PBC = "pbc"; UNKNOWN = "unknown"

def sniff_format(data: bytes) -> ChartFormat:
    """**只按内容**判定（第一非空字节 / 关键字段名 / 行结构）。签名不接受文件名或后缀。

    - 含 "eventLayers"                → RPE
    - 含 "notesAbove"/"notesBelow"/"formatVersion" → OFFICIAL
    - 文本且匹配 PEC 行结构（bp/cp/cm/n1..n4）      → PEC
    - 其余                              → UNKNOWN（记账，不抛给上层静默通过）
    """
```

### 3.4 RPEJSON 解析器（`beatmorph/data/parsers/rpejson.py`，⭐ 独立实现）

```python
def parse_rpejson(data: bytes, source: ChartSource) -> PhigrosChart:
    """RPEJSON → plan 00 的 PhigrosChart 契约。**只读行为规范，不移植任何 GPL/LGPL 源码。**"""

def beat_to_seconds(beats: float, bpm_points: Sequence[BpmPoint]) -> float:
    """**格式层**换算（RPE 原生 beat 三元组 → 契约秒）；分段积分：Σ (Δbeats × 60 / bpm)。
    断言 round-trip：beat2sec(sec2beat(x)) == x（容差内）。
    ⚠️ 与「场网格秒↔τ」的关系：数学同源（τ 即拍），红线 7 要求 beat-aligned 换算只在 field/ 内实现
    → 二者的接缝须裁定，本模块**不得**成为第二份独立实现（RFC-0029 §7-8、plan 03 §9-13）。"""

def seconds_to_beat(t_s: float, bpm_points: Sequence[BpmPoint]) -> float: ...
```

**必须实现的语义（逐条都有格式事实依据）**

1. **beat 三元组**：`beats = i + n / d`（A 级 `Triple(i32,u32,u32)`）；单 BPM 段内 `seconds = 60 / bpm * beats`；多段按 `BPMList` 分段积分。
2. **note 映射**：`type` 走 `note_type_from_rpe()`（1/2/3/4 = Tap/Hold/Flick/Drag，A 级源码确证）；`side` 走 `side_from_above()`（**`above == 1` 为正面，其余值为背面**）；`is_fake = (isFake == 1)`；三者均保留 `*_raw`。
3. **Hold**：`hold_time = endTime - startTime`（秒）；非 Hold 的 `endTime == startTime`；断言 `endTime >= startTime`（违约进隔离区）。
4. **eventLayers 三态归一**：`null` 层 / 字段缺失 / `eventLayers` 整体缺失 → 一律归约为「空轨列表」，长度**原样保留**（1..`RPE_MAX_EVENT_LAYERS`）。
5. **事件跨层求和**：`moveX/moveY/rotate/alpha` 的最终值 = 各层之和（**不是取最上层**）。
6. **补洞**：事件按 `startTime` 排序后，相邻事件之间若存在时间空隙，插入「保持前一事件终值」的常量事件；末尾追加一个足够长的常量事件。**不补洞则间隙内取值无定义**。
7. **缓动**：1..29 全表 + 归一化（非 int → 1，越界 → 末端值）+ `easingLeft/Right` 切割 + 贝塞尔（`bezier/bezierPoints`）；`speedEvents` 只有 5 个字段（无 bezier）。
8. **`father` 嵌套**：`father != -1` 时递归叠加父线位置（实测 26% 的谱面存在）；解析期检测成环。
9. **`bpm_factor`**：**存储但不参与换算**（prpr 标为 TODO，存疑 D4）；`!= 1.0` 必须计入隔离报告。
10. **`is_cover`**：`1 = 遮罩，其余 = 不遮罩`（同 `above` 类陷阱，禁止 `bool()`）。

### 3.5 质检（`beatmorph/data/qc.py`）

```python
@dataclass(frozen=True)
class QcReport:
    chart_id: int
    passed: bool
    n_lines: int; n_notes: int
    fmt: ChartFormat
    out_of_visible_range: int      # |position_x| > RPE_STAGE_HALF_WIDTH 的事件数（**只统计，不钳位**）
    out_of_audio_window: int       # 事件时刻超出音频时长的事件数
    errors: list[str]              # schema 级硬错误 → 隔离
    warnings: list[str]

def quality_check(chart: PhigrosChart, audio_duration_s: float | None) -> QcReport: ...
```

| 层级 | 规则 | 处置 |
|------|------|------|
| **schema** | `judgeLineList` 非空；每线有 `notes`；`type_raw ∈ {1,2,3,4}`；Hold `endTime >= startTime`；`father` 索引合法且无环 | 违约 **拒收 → 隔离区** |
| **单位** | 全部时间在**秒域**；`|position_x| <= RPE_STAGE_HALF_WIDTH`；断言 `RPE_STAGE_WIDTH` 派生一致 | 越界**只计入 `out_of_visible_range`**（哨兵：它同时是「我方解析单位错」的探测器） |
| **分布** | 每谱一行统计：线数、note 数、type/above 分布、每线 note 数（含熵）、时间跨度、BPM 区间 | 与调研 §7 实测基线比对，**离群进隔离区而非直接进训练集** |

### 3.6 特征提取与配对（`beatmorph/data/pipeline/embed.py`）

```python
def extract_features(audio_path: Path, out_dir: Path, encoder: "MertAudioEncoder") -> FeatureCacheMeta:
    """重采样到 MERT_SAMPLE_RATE_HZ → plan 01 encode → 写 `.npz` + `.meta.json`；
    元数据含 {rate, sample_rate, layer, model_rev, duration_s, original_sample_rate, feat_dim, dtype, adapter}。"""

def build_pairs(meta_table: Path, chart_dir: Path, feature_dir: Path) -> "datasets.Dataset":
    """按曲目归组后切分 train/val/test（同曲多谱同 split）；输出 (audio_emb, chart IR) 对。"""

def dataset_stats(table: Path) -> DatasetStats:
    """唯一曲目数、同曲重复率、格式分布、线数分布、type/above 分布、越界率、共格碰撞率。"""
```

**音频去重**：按音频内容 hash（sha1）落盘 `audio/<sha1>.<ext>`，谱面通过外键引用；同曲多谱只提取一次特征。**音频与谱面分开存放**，使「只训练谱面结构、不碰音频」与「联合训练」两种模式可自由切换（调研 §9.2 [6]）。

### 3.7 微缩夹具契约（`tests/fixtures/phigros/`）

| 夹具 | 来源 | 性质 | 用途 |
|------|------|------|------|
| `rpe_min.json` | 由 `chart/1000`（アイドル AT Lv.15，实测 71 条线 / 1659 note，`above` 含 2）**裁剪** | 标准 RPE | 正样本：多线、四类 type、背面 note、事件多层 |
| `pec_masquerade.json` | 由 `chart/7039`（`24432296.json` 实为 PEC 文本）**裁剪** | 伪装成 `.json` 的 PEC | 负样本：后缀陷阱 |
| `pkg_min/`（目录） | 手工构造的最小 zip 包 | `info.yml` + 谱面文件（文件名 ≠ `chart.json`）+ 一个**更大的**干扰 json | 陷阱 2/规则 R1/R2 |

**硬性约束**：夹具必须**微缩**（单文件 ≤ 32 KB，目录 ≤ 128 KB）、**不含音频与曲绘**、写入 `.gitattributes` 的 LF 规则；来源与裁剪方式写在夹具旁的 `README.md`（含原 chart id 与哈希）。

## 4. 内部设计

- **三段式获取**：枚举（API）→ 预筛（Range + 中央目录 + 24 KB 前缀嗅探）→ 选择性下载（只取谱面条目）。**先建直方图再决定下载策略**，禁止一上来全量拖包（调研 §9.2 [1]–[3]）。
- **内容优先原则**：格式判定、谱面文件定位、type 映射分派三处**都只以内容为依据**；`info.yml` 只用于**定位**（`chart` / `music` 字段）与元数据，不用于判型。
- **解析器独立实现**：只读格式文档与「参考实现的行为规范」，**不移植** prpr（GPL-3.0）/ phichain（LGPL-3.0）代码；实现过程中若发现文档未覆盖的行为，记入本 plan §9 而非去抄源码（红线与许可证双重约束）。
- **时间唯一口径**：解析后**立刻**转秒，IR 内不保留 beat 作为主字段（beat 只在 `meta` 溯源里保留原始三元组）；所有评估在秒域（RFC-0029 §7-5）。**`BPMList` 必须完整保留在 IR 中**——它是下游 `field/` 派生 `J(τ) = dt/dτ` 与秒↔τ 换算的**唯一**依据（红线 7、RFC-0029 §7-8）；解析器**不得**丢弃或规范化掉非整拍/非常见分母的 BPM 段。
- **错误分级**：schema 违约 = 拒收进隔离区；单位越界 = 统计 + 标记；分布离群 = 隔离但不删（保留全库统计的可追溯性）。
- **限速与重试**：`REQUEST_INTERVAL_S` 间隔、8 并发上限、指数退避 + Range 断点续传（实测 200 张扫描有 7 张网络失败）。
- **落盘布局**：`data/raw/<chart_id>/`（包元数据）、`data/audio/<sha1>`、`data/features/<sha1>.npz`、`data/manifests/*.parquet`；全部在 `.gitignore` 黑名单内（红线 5），只有 `tests/fixtures/**` 入库。
- **来源与用途留痕（合规硬约束 ③）**：获取与处理脚本必须为每份清单写入 `provenance`——**来源**（API 端点 / 查询条件 / 抓取时间 / chart id 范围）、**用途**（训练 / 评估 / 统计）、**脚本标识与版本**；随 manifest 落盘，可逐张追溯。
- **日志**：`from beatmorph.core.logging import get_logger`，禁止裸 `print`。

## 5. 依赖关系

- **上游**：Phira 官方 API 与谱面包 CDN（**外部**；合规风险已由决策者裁定承担，本模块的落点是 M8 的四条硬约束）；`core/contracts`（plan 00）提供 `PhigrosChart` / `ChartFieldSpec` / 单位派生常量。
- **下游**：plan 01（音频 → MERT 特征，本模块只负责调用与落盘）；`field/`（消费 `PhigrosChart` 构建目标场）；`generation/`（消费 (audio_emb, chart) 对）；`eval/`（消费同一 split 与统计）。
- **外部库**：`httpx` 或 `requests`（Range/流式）、`PyYAML`（info.yml）、标准库 `zipfile`/`zlib`（中央目录与前缀部分解压）、`datasets` / `pyarrow`（清单与配对落盘）、`librosa`/`torchaudio`（重采样）、`pydantic>=2.5`（契约校验）。
- **许可证纪律**：**不得**引入 prpr / phichain 的任何源码或二进制作为依赖；`pyproject.toml` 不新增 GPL/LGPL 依赖。

## 6. 里程碑与验收标准

| 里程碑 | 验收（可量化、可测试） |
|--------|----------------------|
| **M1 元数据枚举** | 322 页分页全部成功；累计条数 == `count` == 9649（与实测一致，偏差必须解释）；落盘 parquet 含 `id/name/level/difficulty/charter/composer/tags/created/updated/file`；**负样本单测（mock，不发真实请求）**：把响应键改成 `result` 必须报错；`pageNum = CHART_PAGE_SIZE_MAX + 1` 必须以 HTTP 400 语义失败并给出可读错误 |
| **M2 微缩夹具入库** | 三个夹具（§3.7）存在且体积达标（单文件 ≤ 32 KB）；`README.md` 记录来源 chart id / 裁剪方式 / 哈希；夹具**不含音频与曲绘** |
| **M3 ⚠️ 陷阱 1：后缀不可信** | `sniff_format(` 对 `pec_masquerade.json`（内容为 PEC、后缀为 `.json`）返回 `PEC`；`parse_chart_package()` 对其**拒收并记账**（`format=PEC`），**绝不**进入 RPE 解析路径；反向用例：把 RPE 内容存成 `.pec` 后缀也必须嗅探为 `RPE`；`sniff_format` 的签名不含文件名参数（签名级约束） |
| **M4 ⚠️ 陷阱 2：必须读 `info.yml.chart`** | 用 `pkg_min` 夹具（谱面文件**不叫** `chart.json`，且包内存在一个**更大的**干扰 json）断言：选中的是 `info.yml.chart` 指定的条目（R1/R2）；把 `info.yml.chart` 指向不存在条目 → 必须报错，**不得**回退到「取最大 json」或「取名为 chart.json 的文件」；引用实测基线：196/196 张的谱面文件都不叫 `chart.json` |
| **M5 ⚠️ 陷阱 3：note type 两套数字** | `note_type_from_rpe(2) is HOLD` 且 `note_type_from_official(2) is DRAG`（同一数字语义相反）；分派**只能由 `sniff_format` 的结果驱动**；构造一个「RPE 内容 + .json 后缀 + 数字 2」的用例，断言解析结果 type == HOLD（而不是被官谱分派成 DRAG） |
| **M6 结构语义** | ① 跨层求和：构造两层各给一半位移的夹具，断言最终位移 == 两层之和；② 补洞：事件间存在空隙时，空隙内取值 == 前一事件终值（而不是默认值）；③ 三态归一：`null` 层 / 缺字段 / 缺整段 `eventLayers` 三种输入产出同一 IR；④ `father` 递归：子线位置 == 自身 + 父线位置，成环输入被拒；⑤ beat→秒：`beat2sec(sec2beat(x)) == x` 与反向在 1e-9 容差内，多 BPM 段用例覆盖 |
| **M7 质检** | 对夹具与**本地落盘（不入库）**的样本跑 `quality_check`：schema 违约样本 100% 进隔离区；越界样本 **`out_of_visible_range > 0` 且 `position_x` 值未被修改**（逐字段比对原 JSON）；分布统计命中调研 §7 的量级（线数中位 30 / 背面 2.4–3.0% / Tap 52–63%）区间内才算通过 |
| **M8 ✅ 数据合规硬约束（已裁定，2026-08-05）** | **训练可启动**——原「`compliance_gate` 默认关闭 + 训练入口拒绝启动」的**阻塞闸门已解除**，改为执行四条硬约束：① **最终不发布模型权重**（项目级承诺，本模块不产出任何分发物）；② 谱面/音频**允许本地落盘、不得入库**（`.gitignore` 覆盖 `data/**`，仅 `tests/fixtures/**` 入库）；③ 获取与处理脚本**记录来源与用途**（`provenance` 随 manifest 落盘，可追溯到 chart id）；④ **发布权重前必须重新裁定**。验收：① 负样本测试——`data/` 下任何音频/谱面文件试图入库时被 `.gitignore` 拦截（CI 可验）；② 每份 manifest 的 `provenance` 字段非空且通过 schema 校验，缺失即报错；③ 源码级断言：不存在任何「权重发布/分发」代码路径。依据 [BasePlan §4.4](BasePlan.md)、[RFC-0029 §7-7/§8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md)、CLAUDE.md 红线 5 附注 |
| **M9 特征离线提取** | 在**本地落盘（不入库，硬约束 ②）**的音频子集上：缓存文件数 == 音频数；抽样加载校验六项全过（plan 01 §3.3）；`meta.rate == MERT_FRAME_RATE_HZ`（派生量，非字面量）；篡改任一元数据字段后加载必须抛错；唯一曲目数（按音频 sha1）与同曲重复率写入 `dataset_stats` |
| **M10 配对与切分** | 同曲多谱进同一 split（用含同曲 2 张谱的样本断言）；产出 train/val/test 三份清单，**曲目集合两两不相交**；另产出一份「同曲跨谱泛化」评测集（同曲不同难度）；报告真实 (audio, chart) 对数（< 9649） |

> **G1–G4 门禁义务说明**：本模块**不引入训练目标或损失**，故无 G1–G4 全绿义务；它是 **G4 的数据侧落点**——帧率、单位、形状三项派生断言在 M7/M9 内以**默认 CI 契约测试**形式落地。任何消费本模块数据的新训练目标（`field/` / `generation/` 的 plan）在扩大数据规模之前必须先跑通 G1–G4（`beatmorph/infra/sanity.py`，[BasePlan §9](BasePlan.md)）。

## 7. 风险与缓解

| 风险 | 编号 | 缓解 |
|------|------|------|
| **数据合规**：唯一万级数据源未授予训练权 | **R-2** | **已裁定（2026-08-05）**：风险由决策者承担，**训练可启动**。缓解 = 四条硬约束的工程落点（M8）：不发布权重、数据不入库、来源与用途留痕、发布前重新裁定；产出物只保留微缩夹具 |
| 社区谱面质量参差，模型学到坏习惯 | **R-3** | M7 三层质检 + 隔离区；`stable/ranked` 作为**元数据特征**保留，便于做「只在高质子集上训练」的消融（**不预设阈值**，先做全库统计再定） |
| 三类静默陷阱（后缀 / 文件名 / type 数字） | **R-7** | 逐条落成 M3/M4/M5 里程碑 + 签名级约束（嗅探函数不接受文件名）；三者都是「不报错但全错」型故障 |
| 物理常量/帧率漂移 | **R-7** | 特征缓存元数据六项校验（复用 plan 01）；时间统一秒域；坐标只统计不钳位 |
| 解析器许可证风险（GPL-3.0 / LGPL-3.0） | **R-2 附注** | 独立实现；**不引入** prpr/phichain 源码或依赖；只读行为规范 |
| 网络不稳定 / 限流 | 派生 | 限速 `REQUEST_INTERVAL_S` + 指数退避 + Range 断点续传；失败样本记账后可重跑（实测失败率 ~3.5%） |
| 全库统计缺失导致阈值拍脑袋 | 派生 | M1/M9/M10 先产全库分布（格式占比、线数、唯一曲目率、共格碰撞率），**再**定过滤阈值；在此之前不做质量过滤 |
| 官谱 JSON / PBC 混入造成 type 错位 | 派生 | 只识别 + 记账、不解析（偏离 2）；`UNKNOWN` 一律拒收并计数 |

## 8. 测试策略

- **单元（默认 CI，无网络、无权重、无 GPU）**
  - `tests/unit/data/test_sniff_format.py` — M3：后缀与内容不一致的四种组合；`UNKNOWN` 的记账行为。
  - `tests/unit/data/test_package_locate.py` — M4：`info.yml.chart` 定位、干扰大 json、缺字段报错（不发网络请求，用本地微缩 zip）。
  - `tests/unit/data/test_note_mapping.py` — M5：两套 type 映射、`above ∈ {0,1,2}`、`isFake`、`isCover`。
  - `tests/unit/data/test_event_layers.py` — M6：跨层求和、补洞、三态归一、缓动归一化。
  - `tests/unit/data/test_beat_time.py` — M6⑤：多 BPM 段 round-trip、边界（0 拍、段边界、负时间）。
  - `tests/unit/data/test_qc.py` — M7：schema 违约进隔离、越界只统计不修改原值。
  - `tests/unit/data/test_phira_client.py` — `results` 键、分页边界、`pageNum` 上限、限速参数；**全部用 mock 响应**，默认 CI 不发真实网络请求。
  - `tests/unit/data/test_provenance.py` — M8：`provenance`（来源/用途/脚本/时间）必填且 schema 校验，缺失即报错；`data/**` 路径不出现在可入库集合中；无「权重发布/分发」代码路径。
- **禁止事项**：mock 响应与夹具**不得固化物理常量**——凡需要帧数/坐标，必须引用 `MERT_FRAME_RATE_HZ` / `RPE_STAGE_HALF_WIDTH` / `RPE_X_GRID_DX`（RFC-0029 §7-6）。
- **集成（本地夹具，无网络）**：`tests/integration/test_pipeline_min.py` — 微缩包 → 嗅探 → 解析 → 质检 → （假编码器）特征 → 配对，端到端产出清单；断言 IR 契约往返、split 按曲目不泄漏。
- **e2e（标记 `@pytest.mark.e2e` + `slow`，不进默认 CI）**：全库预筛（322 页 + 9649 次 Range）产出格式/大小/线数分布；对**本地落盘（不入库）**的真实音频做特征提取抽样与在线推理比对。

## 9. 开放问题

1. **Q-1 全库精确格式占比**：现有 283 张抽样为 RPE 97.2% / PEC 2.5%，且 PEC 在旧谱面上占比更高（4.4%）。全库值只能由 M1 + 预筛给出——**在拿到全库数字之前，不得据抽样比例设过滤规则**。
2. **Q-3 唯一曲目数与同曲重复率**：完全未统计，直接决定真实 (audio, chart) 对数。M9 用音频 sha1 去重给出；该数字会影响「全量 9649」这一规模叙事，须回写 [phira-dataset-survey.md](knowledges/phira-dataset-survey.md)。
3. **Q-2 限流策略**：未观测到 `X-RateLimit-*` 或 `Retry-After`，但「未观测到」不等于「没有」。`REQUEST_INTERVAL_S = 0.5` 只是自保值，是否足够未查证；大规模抓取前应做小规模速率探测。
4. **Q-16 同刻跨线并发上限**：未统计。它是「跨线合法性」规则（[RFC-0029 §5](decisions/RFC-0029-phigros-continuous-chart-generation.md) / plan 05 §9-1）的输入，也决定 `ChartField` 的桶内计数是否需要「同桶多事件」的显式通道。M7 的分布统计应把它一并产出。
5. **共格碰撞率决定 `RPE_X_GRID_BINS`**：同线 + 同刻 + 同侧的最小 |ΔpositionX|（实测 min gap 0.32，手工谱常见网格 1350/60 与 1350/120）尚未统计。**该统计出来之前，128 只是 RFC 给的默认值而非结论**（plan 00 §9-9）。
6. **`Q-8`（5 层 vs 4 普通 + extended）** 与 **Q-15（`father` 字段名/默认值三处写法不一）** 仍未裁定：解析器按「原样保留 + 兼容两种拼写」实现，但**语义分支**（哪些层参与求和）待裁定后回写 plan 00 与本 plan。
7. **`Q-5` 官谱 / PBC 的准入**：官谱 type 映射仅 C 级、PBC 结构未查证 → v1 拒收。若日后要并入，必须先验证映射（偏离 2）。
8. **夹具的合规性**：微缩夹具改编自 Phira 用户上传内容（谱面作者授权不明，音频/曲绘另有版权）。本 plan 只入库**无音频无曲绘**的最小结构片段；Q11b 已裁定由决策者承担风险（训练可启动），但**「改编片段是否算衍生作品」这一注意义务不因裁定而消失**——若日后需要对外分发任何夹具，须回到硬约束 ④（重新裁定）；纯手工构造的 `pkg_min`（§3.7）是现成退路。
9. **`META.offset` 的符号与作用点**（Q11 / 存疑 D10）：RPE 为毫秒、官谱为秒，且 RPE 文档表述语义缠绕；prpr 在本轮读到的源文件中未使用它。本 plan 暂**只记录不使用**，是否参与音频对齐须与 eval 的 plan 联合裁定。
10. **RPEJSON 读 / 写的归属边界**：`docs/plans/README.md` 把 `beatmorph/io/formats/rpejson/` 的**读**归本 plan、**写**归 plan 05。两侧必须共用同一 IR 与同一份 schema 校验，且 `parse → write → parse` 的**往返等价测试**应由两份 plan 共同维护；具体测试落点待与 plan 05 联合确认。
11. **格式层 beat↔秒 与场网格秒↔τ 的接缝（须裁定）**：`beat_to_seconds` / `seconds_to_beat`（本模块，格式层，契约层 `PhigrosNote.t` 用秒这一点不变）与 `field/` 的 τ 换算在数学上**同源**（τ 即拍）。CLAUDE.md 红线 7 要求「beat-aligned 的时间换算只在 `field/` 内实现」→ 二者是否必须共用同一实现、依赖方向如何，**未裁定**（plan 03 §9-13）。裁定前只保留本模块这一处格式层换算，**不得**出现第三处。
