# Plan 02 — 数据流水线：Phira 谱面获取 + RPEJSON 解析 + 质检 + 特征离线提取

> 状态：🟢 **主路径已跑通真实数据**（M1–M12 全部落地；驱动脚本 `scripts/fetch_phira.py` / `scripts/extract_features.py` 已对真实 Phira 库执行全量枚举与选择性下载，产出带 provenance 的清单 + 真实 MERT 特征缓存；逐条证据见 §6 与 §9「实施期裁定」） ｜ 阶段：Phase 1 ｜ 负责：数据组（data-agent）
> 对应代码：`beatmorph/data/`（`phira/client.py`、`phira/package.py`、`parsers/sniff.py`、`parsers/rpejson.py`、`qc.py`、`pipeline/embed.py`、**`tracks.py` / `dataset.py`**） ｜ 对应奠基章节：§4、§3.2.4、§9

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
CHART_LIST_RESULTS_KEY: str = "results"  # ⚠️ 非文档写的 "result"（实测）
CHART_PAGE_SIZE_MAX: int = 30  # 实测 pageNum=31 → HTTP 400
CHART_TOTAL_EXPECTED: int = 9649  # 2026-09-26 实测 count；作为枚举完整性断言，非硬编码语义
RANGE_PREFIX_BYTES: int = 24 * 1024  # 谱面条目嗅探用的压缩前缀（实测有效）
ZIP_TAIL_BYTES: int = 200 * 1024  # 中央目录读取窗口（实测有效）
REQUEST_INTERVAL_S: float = 0.5  # 自限速：未观测到限流 ≠ 无限流（调研 Q-2）
CHART_FILE_FIELD: str = "chart"  # info.yml 中定位谱面文件的字段（不得按名猜）
CHART_MUSIC_FIELD: str = "music"  # 音频文件名的唯一来源
```

### 3.2 客户端与包访问（`beatmorph/data/phira/client.py`、`package.py`）

```python
class PhiraClient:
    def iter_chart_meta(
        self, type: int = 3, sleep_s: float = REQUEST_INTERVAL_S
    ) -> Iterator[ChartMeta]:
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

    chart_file: str  # **只来自** info.yml[CHART_FILE_FIELD]；不得按后缀或大小猜
    music_file: str  # **只来自** info.yml[CHART_MUSIC_FIELD]
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
    RPE = "rpe"
    PEC = "pec"
    OFFICIAL = "official"
    PBC = "pbc"
    UNKNOWN = "unknown"


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
    n_lines: int
    n_notes: int
    fmt: ChartFormat
    out_of_visible_range: int  # |position_x| > RPE_STAGE_HALF_WIDTH 的事件数（**只统计，不钳位**）
    out_of_audio_window: int  # 事件时刻超出音频时长的事件数
    errors: list[str]  # schema 级硬错误 → 隔离
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
def extract_features(
    audio_path: Path, out_dir: Path, encoder: "MertAudioEncoder"
) -> FeatureCacheMeta:
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

| 里程碑 | 验收（可量化、可测试） | 实施状态 |
|--------|----------------------|---------|
| **M1 元数据枚举** | 322 页分页全部成功；累计条数 == `count` == 9649（与实测一致，偏差必须解释）；落盘 parquet 含 `id/name/level/difficulty/charter/composer/tags/created/updated/file`；**负样本单测（mock，不发真实请求）**：把响应键改成 `result` 必须报错；`pageNum = CHART_PAGE_SIZE_MAX + 1` 必须以 HTTP 400 语义失败并给出可读错误 | ✅ 代码 + mock 单测；**2026-09-27 全库实跑 322 页**：`count = 9651`（基线 9649，+2 已解释为社区库仍在长），落盘 `data/processed/meta.jsonl`。分页路径本轮补上指数退避重试——实跑中第 54 / 87 / 286 页各命中一次传输超时，**没有重试时整轮枚举直接报废**（实测） |
| **M2 微缩夹具入库** | 三个夹具（§3.7）存在且体积达标（单文件 ≤ 32 KB）；`README.md` 记录来源 chart id / 裁剪方式 / 哈希；夹具**不含音频与曲绘** | ✅ 三夹具齐备（`rpe_min.json` / `pec_masquerade.json` / `pkg_min`），单文件 ≤ 11.5 KB、目录 24.6 KB；夹具**手工构造**、`pkg_min` 由文本部件确定性打包（不提交二进制） |
| **M3 ⚠️ 陷阱 1：后缀不可信** | `sniff_format(` 对 `pec_masquerade.json`（内容为 PEC、后缀为 `.json`）返回 `PEC`；`parse_chart_package()` 对其**拒收并记账**（`format=PEC`），**绝不**进入 RPE 解析路径；反向用例：把 RPE 内容存成 `.pec` 后缀也必须嗅探为 `RPE`；`sniff_format` 的签名不含文件名参数（签名级约束） | ✅ `test_sniff_format.py`（四种后缀×内容组合 + `inspect` 签名断言）；PBC 不产出判定（结构未查证） |
| **M4 ⚠️ 陷阱 2：必须读 `info.yml.chart`** | 用 `pkg_min` 夹具（谱面文件**不叫** `chart.json`，且包内存在一个**更大的**干扰 json）断言：选中的是 `info.yml.chart` 指定的条目（R1/R2）；把 `info.yml.chart` 指向不存在条目 → 必须报错，**不得**回退到「取最大 json」或「取名为 chart.json 的文件」；引用实测基线：196/196 张的谱面文件都不叫 `chart.json` | ✅ `test_package_locate.py`（含 decoy `chart.json` 与更大的干扰 json；指向不存在条目必报错） |
| **M5 ⚠️ 陷阱 3：note type 两套数字** | `note_type_from_rpe(2) is HOLD` 且 `note_type_from_official(2) is DRAG`（同一数字语义相反）；分派**只能由 `sniff_format` 的结果驱动**；构造一个「RPE 内容 + .json 后缀 + 数字 2」的用例，断言解析结果 type == HOLD（而不是被官谱分派成 DRAG） | ✅ `test_note_mapping.py`（含 `above ∈ {0,1,2}` / `isFake` / `isCover`） |
| **M6 结构语义** | ① 跨层求和：构造两层各给一半位移的夹具，断言最终位移 == 两层之和；② 补洞：事件间存在空隙时，空隙内取值 == 前一事件终值（而不是默认值）；③ 三态归一：`null` 层 / 缺字段 / 缺整段 `eventLayers` 三种输入产出同一 IR；④ `father` 递归：子线位置 == 自身 + 父线位置，成环输入被拒；⑤ beat→秒：`beat2sec(sec2beat(x)) == x` 与反向在 1e-9 容差内，多 BPM 段用例覆盖 | ✅ `test_event_layers.py` + `test_beat_time.py`（往返 1e-9，独立复算比对） |
| **M7 质检** | 对夹具与**本地落盘（不入库）**的样本跑 `quality_check`：schema 违约样本 100% 进隔离区；越界样本 **`out_of_visible_range > 0` 且 `position_x` 值未被修改**（逐字段比对原 JSON）；分布统计命中调研 §7 的量级（线数中位 30 / 背面 2.4–3.0% / Tap 52–63%）区间内才算通过 | ✅ `test_qc.py`（越界只统计不钳位、逐字段比对原 JSON）；语料级基线区间断言落在 `dataset_stats().corpus_outliers()`，用合成数据单测 |
| **M8 ✅ 数据合规硬约束（已裁定，2026-08-05）** | **训练可启动**——原「`compliance_gate` 默认关闭 + 训练入口拒绝启动」的**阻塞闸门已解除**，改为执行四条硬约束：① **最终不发布模型权重**（项目级承诺，本模块不产出任何分发物）；② 谱面/音频**允许本地落盘、不得入库**（`.gitignore` 覆盖 `data/**`，仅 `tests/fixtures/**` 入库）；③ 获取与处理脚本**记录来源与用途**（`provenance` 随 manifest 落盘，可追溯到 chart id）；④ **发布权重前必须重新裁定**。验收：① 负样本测试——`data/` 下任何音频/谱面文件试图入库时被 `.gitignore` 拦截（CI 可验）；② 每份 manifest 的 `provenance` 字段非空且通过 schema 校验，缺失即报错；③ 源码级断言：不存在任何「权重发布/分发」代码路径。依据 [BasePlan §4.4](BasePlan.md)、[RFC-0029 §7-7/§8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md)、CLAUDE.md 红线 5 附注 | ✅ ①②③ 均有默认 CI 测试（`test_provenance.py`）；`data/**` 已被 `.gitignore` 覆盖（含 features / manifests / audio），`assert_local_only` 守卫对域外路径仍会拦下 |
| **M9 特征离线提取** | 在**本地落盘（不入库，硬约束 ②）**的音频子集上：缓存文件数 == 音频数；抽样加载校验六项全过（plan 01 §3.3）；`meta.rate == MERT_FRAME_RATE_HZ`（派生量，非字面量）；篡改任一元数据字段后加载必须抛错；唯一曲目数（按音频 sha1）与同曲重复率写入 `dataset_stats` | ✅ `test_embed_features.py`（假编码器，六项校验各有负样本）；**2026-09-27 真实权重实跑**：24 首曲目共 48 s（RTX 5070 Laptop 8 GB / 115 W 限功耗），帧数全部满足 `T_seq == round(duration_s × 75)`（例 148.77 s → 11157 帧） |
| **M10 配对与切分** | 同曲多谱进同一 split（用含同曲 2 张谱的样本断言）；产出 train/val/test 三份清单，**曲目集合两两不相交**；另产出一份「同曲跨谱泛化」评测集（同曲不同难度）；报告真实 (audio, chart) 对数（< 9649） | ✅ `test_pairs_split.py` + `tests/integration/test_pipeline_min.py`（同曲同 split、曲目集合两两不相交、同曲跨谱泛化集）；**2026-09-27 真实清单实跑**：语料级 `stats.json` 已产出（唯一曲目 / 同曲重复率 / 判定线分布 / 越界率，见 §9）。⚠️ 本轮修掉一个**跨模块往返 bug**：`build_pairs` 曾把**已解析**的 `chart_path` 写进清单，`ChartPairDataset` 再拼一次 chart_dir ⇒ 20/20 行判「谱面缺失」，门禁装配直接失败（回归测试已锁定该往返） |

| **M11 训练数据通路** | 「谱面 + 特征缓存」→ `FieldBatch`：窗口化（每窗 `PairSample`）、`collate_field_batch` 通过 `FieldBatch.assert_shapes()`、网格身份不一致**抛错**（`GridMismatchError`）、**绝不发出 r == 1 的样本**、同 index 取两次逐位一致、窗口边界不切断 Hold、音频切片帧数由契约帧率派生 | ✅ `beatmorph/data/dataset.py` + `tracks.py`；`tests/unit/data/test_dataset.py`（27 项）+ `tests/integration/test_dataset_to_generation.py`（真实窗口 → 前向 → 反传）；主会话口径见 §9「实施期裁定」 |

| **M12 驱动脚本 + 真实拉取**（本轮新增） | ① 枚举 → 预筛 → 选择性下载 → 解析 → 质检 → 清单 全链路可由**一条命令**重跑且可续跑；② 每份清单带 provenance（来源 / 查询 / 时间 / 用途 / 脚本与版本）；③ 拒收（sniff / parse / qc）与网络失败**分别记账**；④ 音频按内容 sha1 去重；⑤ 真实库实跑：格式分布 / 唯一曲目 / 同曲重复率 / 越界率 / 共格最小间距 / 同刻并发上限 | ✅ `scripts/fetch_phira.py`（`meta` / `fetch` / `pairs` / `stats` / `all`）+ `scripts/extract_features.py`；`tests/unit/scripts/`（42 项，全部离线：假 Phira 客户端用本地 zip 提供 Range 语义）；**实跑**见 §9「实施期裁定」 |

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
4. ~~**Q-16 同刻跨线并发上限**：未统计~~ **✅ 已统计（2026-09-27 第三轮，见下方「离群值口径复核」）**：全库中位 4 / p99 39；头部值 12012 是 6 张「音符墙」退化谱面造成的，**不得**用作并发上限。
5. ~~**共格碰撞率决定 `RPE_X_GRID_BINS`**：同线 + 同刻 + 同侧的最小 |ΔpositionX| 尚未统计~~ **✅ 已统计（同上）**：72.5% 的谱面最小间距为 0.0 ⇒「不共格所需 N」不可达；且现口径**漏了 type 通道**，把碰撞率夸大约 2.3 倍。N 只能按碰撞率–N 曲线选（plan 03 §9-7）。
6. **`Q-8`（5 层 vs 4 普通 + extended）** 与 **Q-15（`father` 字段名/默认值三处写法不一）** 仍未裁定：解析器按「原样保留 + 兼容两种拼写」实现，但**语义分支**（哪些层参与求和）待裁定后回写 plan 00 与本 plan。
7. **`Q-5` 官谱 / PBC 的准入**：官谱 type 映射仅 C 级、PBC 结构未查证 → v1 拒收。若日后要并入，必须先验证映射（偏离 2）。
8. **夹具的合规性**：微缩夹具改编自 Phira 用户上传内容（谱面作者授权不明，音频/曲绘另有版权）。本 plan 只入库**无音频无曲绘**的最小结构片段；Q11b 已裁定由决策者承担风险（训练可启动），但**「改编片段是否算衍生作品」这一注意义务不因裁定而消失**——若日后需要对外分发任何夹具，须回到硬约束 ④（重新裁定）；纯手工构造的 `pkg_min`（§3.7）是现成退路。
9. **`META.offset` 的符号与作用点**（Q11 / 存疑 D10）：RPE 为毫秒、官谱为秒，且 RPE 文档表述语义缠绕；prpr 在本轮读到的源文件中未使用它。本 plan 暂**只记录不使用**，是否参与音频对齐须与 eval 的 plan 联合裁定。
10. **RPEJSON 读 / 写的归属边界**：`docs/plans/README.md` 把 `beatmorph/io/formats/rpejson/` 的**读**归本 plan、**写**归 plan 05。两侧必须共用同一 IR 与同一份 schema 校验，且 `parse → write → parse` 的**往返等价测试**应由两份 plan 共同维护；具体测试落点待与 plan 05 联合确认。
11. **格式层 beat↔秒 与场网格秒↔τ 的接缝（须裁定）**：`beat_to_seconds` / `seconds_to_beat`（本模块，格式层，契约层 `PhigrosNote.t` 用秒这一点不变）与 `field/` 的 τ 换算在数学上**同源**（τ 即拍）。CLAUDE.md 红线 7 要求「beat-aligned 的时间换算只在 `field/` 内实现」→ 二者是否必须共用同一实现、依赖方向如何，**未裁定**（plan 03 §9-13）。裁定前只保留本模块这一处格式层换算，**不得**出现第三处。
   > **临时裁定（2026-09-27，实施期）**：接缝**实测确有分歧**，但只在一个畸形情形下：BPMList 首段起点 > 0 拍时，
   > 本模块以首段为时间原点（与 Phira 官方参考实现 `beat2sec` 一致），而 `field/` 把 τ=0 当作谱面原点并在前面外推一段，
   > 两条路径相差一个常量偏移（实测首段在第 4 拍时相差 2 秒）。RPE 规范首段是 `[0,0,1]`，故该情形只可能来自畸形谱面。
   > **处置**：不静默二选一，而是把「首段必须起于 0 拍」提升为 `qc.py` 的 **schema 违约 → 隔离区**；
   > 首段起于 0 拍时两条路径在默认 CI 的 `tests/integration/test_time_conversion_seam.py` 中逐点一致（含多 BPM 段、段界、往返无损、改写 BPMList 必变）。
   > 仍需 RFC 决定的是**统一实现**（依赖方向 `data → field`？或把分段积分上提到 `core/contracts`）——在裁定前不得出现第三处换算。

### 实施期裁定（主会话，2026-09-27）

**M11 训练数据通路已落地**：`beatmorph/data/dataset.py`（窗口化 `ChartPairDataset` + `collate_field_batch`）
与 `beatmorph/data/tracks.py`（事件轨在 τ 轴上求值）把「谱面 + 特征缓存」变成 generation 主干可直接消费的
`FieldBatch`。三条口径由主会话裁定：

1. **窗口局部网格**（实现方案，接受）：`PairSample.grid` 的 τ 轴是窗口局部的，`bpm_points` 收成单段
   `[(0, bpm_eff)]`，`bpm_eff = 60 / J(τ_start)` 由 `field.grid.jacobian_at` 派生；绝对 τ 位置由
   `PairSample.tau_start` 携带。好处：`FieldBatch` 只带一个 grid 的约束下，同批样本的逐格 J **必然相同**。
   代价：跨 BPM 变更点的窗口被跳过并记账（不静默丢弃）。
2. **绝不发出 r == 1 的样本**（接受）：`generation.losses.masked_poisson_loss` 在 r == 1 时**拒绝训练**，
   因此数据集先排除「遮盖单位数 <= 1」的窗口（与种子无关），再对残余情形重掷种子（上限 8 次），
   仍不行则以「无遮盖」发出并记账告警。
3. **事件 token 数 == 1 的窗口跳过并记账**（接受）。另一种口径是以全 False 遮盖发出（不丢数据），
   但当前选择与主会话指令一致，且不影响 M10 的记账纪律。

**新登记（需 RFC 或契约变更）**：

- **`FieldBatch` 缺 `time_mask`**：批内音频长度不一致时 `collate_field_batch` 只能补零，
  补零帧是**伪造的静音特征**。由于默认 `batch_size=1`，当前不触发；但一旦要跨谱批训练，
  必须给音频 padding 一个显式掩码（改 `generation/batch.py` 的契约）。
- **跨谱 batching**：与 plan 07 §9-13 同一问题（按网格分组采样 vs per-sample 网格）。

### 实施期裁定（主会话，2026-09-27 第二轮：真实数据全库拉取）

**M12 驱动脚本 + 全库实跑已完成**（`scripts/fetch_phira.py` / `scripts/extract_features.py`）。全库口径：

| 项 | 实测 |
|----|------|
| 枚举 | 9651 张（`meta.jsonl`；基线 9649，+2 解释为社区库仍在长） |
| 入库 | **8551** 张 RPE（`charts.jsonl`，带 provenance） |
| 拒收 | **1099**：`sniff` 886（PEC **872** / 官谱 14）、`parse` 191、`qc` 22（`quarantine.jsonl` 逐条留痕） |
| 未解 | 1 张（网络失败，可重跑） |
| 体量 | 谱面 34.5 GB + 音频 42.8 GB = **77 GB**（与调研「全量直抓 ≈76 GB」吻合） |
| 唯一曲目 | **6807**（同曲重复率 **20.4%**）⇒ 真实 (audio, chart) 对的数量级 |
| 唯一音频 | **8084**（按内容 sha1；音频层重复率 5.5%） |
| 判定线 | 中位 **25**（P25 24 / P75 42，范围 1–770）；note 计 11 077 712 |
| 分布 | Tap 55.7% / 背面 3.11%（调研基线 2.4–3.0%，`corpus_outliers` 标为轻微离群）/ 越界 0.13% |

**四条与调研基线冲突的实测（都已落成代码或待办）**

1. **PEC 占 9.0%**（872/9651），远高于调研的 2.5%（§9-Q1 得到答案）——原因是抽样偏近期；且旧谱面的 PEC 语法族更宽（`&` / `cf` / `cr`），调研样例的 10 个命令不足以判型（`sniff.py` 已扩表）。
2. **`info.yml` 文本字段是 `null` 或标量**（`tip` / `level` / `charter` / `composer`）——裸 `str` 声明会把整张谱面判成「结构错误」（首批 20 张里 10 张；修完 `null` 后全库还剩 5 张是数字/布尔，也已归一）。
3. **`build_pairs` 的 `chart_path` 往返 bug**：写回已解析路径 ⇒ `ChartPairDataset` 再拼一次 chart_dir ⇒ 全部行判「谱面缺失」。**这是「契约写了相对路径、实现写了绝对路径」的典型静默失效**，已加回归测试。
4. **`max_simultaneous_onsets = 12012`** 与 **`min_same_line_same_time_gap_x = 0.0`**：两者都是 §9-Q16 / §9-5 的输入，但这两个数值**先要复核口径**（0.0 间距与 1.2 万同刻事件更可能是统计口径/异常谱面问题，而不是真实创作习惯）。

### 实施期实测（主会话，2026-09-27 第三轮：两个离群值的口径复核）

§9-Q16 与 §9-5 的输入值此前都是「可疑数字」（`max_simultaneous_onsets = 12012` /
`min_same_line_same_time_gap_x = 0.0`）。本轮用全库清单（8551 张，行级已带这两个统计）
加 150 张抽样重解析复核，**结论：两者都不是创作习惯——一个是少数退化谱面，一个是统计口径**。

**① `max_simultaneous_onsets = 12012` 是退化谱面，不是并发上限**

- 该值来自 chart **44144**「Hello (BPM) 2023」：**71.5%** 的 note（12012 / 16794）落在**同一时刻**
  t = 157.4296 s，且全部是同一条线、同一 `side`、`positionX` 全为 0.0、`type = 4`（Flick）、`above = 1`。
- 同型谱面共 **6 张**（单一时刻 note 占比 ≥ 40% 且 note 总数 ≥ 32）：77392（77.1%）、44144（71.5%）、
  50922（53.1%）、54540（73.5%，两个时刻各 4346 / 3000）、78218（42.9%）、33204（26.4%，未过 40% 线）。
  它们都是把 note 堆在单一时刻的「音符墙」，不构成可玩谱面。
- **剔除这 6 张后**（8545 张）同刻并发分布：中位 **4**、p90 11.6、p99 **39**、p99.9 **192**、最大 **795**；
  绝对计数：≥16 的 497 张、≥32 的 125 张、≥64 的 40 张。
- ⇒ **Q-16 口径**：报数必须写「剔除音符墙后」，头部值 12012 **不得**用于定「同刻跨线并发上限」。
  退化谱面可用于 **QC 告警（不作红线）**，判据：单一时刻 note 占比 ≥ 40% 且 note 总数 ≥ 32。

**② `min_same_line_same_time_gap_x = 0.0` 是结构性现象，且现口径夸大了碰撞**

- 全库 8517 张有该统计（34 张无同线同刻对）：**6176 张（72.5%）最小间距恰为 0.0**
  ⇒「同线 + 同刻 + 同侧 + 同 `positionX`」的 note 在真实谱面里是**常态**。
- 抽样 150 张（`seed = 20260927`）复核两种分组口径：

  | 分组 | 最小间距 == 0 的谱面 | N = 128 下同桶对 / 同刻对 |
  |---|---|---|
  | 现口径 `(line, t, side)` | 66.7% | 6.79% |
  | **含 type 通道** `(line, t, side, type)` | **21.3%** | **2.98%** |

  场张量是 `(K, T, X, S, C)`，`C` 是 note 类型通道：**类型不同的两个 note 落在同一个 x 桶里并不共格**。
  现口径因此把碰撞率夸大约 **2.3 倍** ⇒ 统计函数应带上 type 维度（实现待改）。
  复现：`scripts/local_collision_grouping.py`（**本地脚本，被 `.gitignore` 的 `scripts/local_*` 覆盖**；
  方法 = 两种分组键 + 150 张抽样、`seed=20260927`、`N=128` 下统计同桶对/同刻对）。
  同型复现脚本：`scripts/local_onsets_anomaly.py`（离群谱定位与退化判据）。
- 重合组的性质抽样：既有「不同类型同位置」（如 Drag + Hold 叠放，谱面里可区分，属合法写法），
  也有「签名完全相同」的真重复（如 chart 41051 的 Hold 重叠）——两种都存在，不可一概而论。
- ⇒ **N 的选型口径改写**：`N ≥ RPE_STAGE_WIDTH / min|Δx|`（「不共格所需 N」）在**任何 N 下都不可达**
  （min|Δx| 可为 0；排除 0 后 p1 ≈ 0.0028 ⇒ N ≥ 48.6 万）。N 只能按 **碰撞率–N 曲线 + positionX MAE**
  选，不能按「消除碰撞」选（plan 03 §9-7 / plan 00 §9-9）。

**③ 特征提取的吞吐瓶颈 = zlib，不是 GPU（本轮实测；GPU 从 36% → 96%）**

单进程串行的 1.65 s/首分解（12 首样本、平均时长 150.8 s）：

| 阶段 | 秒/首 | 占比 |
|---|---|---|
| `np.savez_compressed`（deflate） | **1.034** | **52.3%** |
| GPU 前向（MERT fp16） | 0.720 | 36.4% |
| 解码 / 重采样 / H2D+D2H | 0.217 | 11.0% |

**根因**：MERT fp16 特征的 deflate 压缩比只有 **1.09×**（18.08 → 16.61 MB），
却要花 0.6–0.8 s 的单线程 CPU；`np.savez(store)` 只要 **0.015 s（39–48×）**。
deflate L1/L3/L6 = 0.551/0.627/0.624 s ⇒ **调级别无用**，是格式选择问题。

**实测提速（生产配置）**：`--npz-compression store` + `--shard i/N`（按缓存键 sha1 取模，
各分片键集**两两不交**；进程各持一份模型 ≈1.6 GB 显存）：
**1.65 s/首 → 0.759 s/首（3 分片，2.17×）**，GPU 利用率 **36% → 96%**（104 W）。
`--pipeline`（预取 + 后台落盘）基准里实测 2.85×，但**首次上生产就挂住**（CPU 满载 1 核、
GPU 0%、数分钟无任何输出）。**已定位并修复**（主会话，同一轮）：

- `_AsyncDecoder.prefetch` 的回压循环 `while len(self._order) >= self._depth: self._futures[self._order[0]].exception()`
  **从不把最老的一项移出窗口** ⇒ 条件恒真、死循环空转（这就是「CPU 100% + GPU 0%」）；
- 同一函数里的补窗下标写成 `position + depth`（`position` 从 1 起）⇒ 索引恰为 `depth` 的那一项
  永远不会被预取，`take()` 会 `KeyError`。改为 `position + depth - 1`。

修完 `tests/unit/data/test_extract_features_pipeline.py` **10/10 通过、1.7 s**（修复前该文件
**永久挂住**，还会把默认 CI 一起挂死）。⚠️ **注意 GPU 已经打满时流水线不再有收益**：
3 分片 + `store` 的聚合已是 `max(GPU 0.72 s/首, CPU/3)`，实测 0.76 s/首、GPU 96%。

代价：`store` 的文件比 `deflate` 大约 **9%**（~15 GB/全库），数组与元数据**逐字段一致**，
`load_feature_cache` 的六项校验照常通过（已实测 5 个新写缓存）。同一缓存目录里
两种压缩并存**不影响加载**。

**④ 真实取批成本 = 训练吞吐的前置（本轮实测，非门禁问题）**

`ChartPairDataset` 的索引构建是 **O(split 行数)** 的谱面解析（`_build_plan` 逐行
`_read_chart_or_skip`），且 `__getitem__` **每次取窗口都会重新解析该谱面**（无缓存）。
实测（1130 行 train split、本机 CPU、与特征提取抢机器）：

- 索引构建 + 首个批次：**≈15 min 墙钟 / ≈900 s 单线程 CPU**（≈ 0.8 s/行，1130 行）；
- 外推：全库 8551 行 ⇒ **≈2 h** 才轮到第一个优化步。

⇒ GPU 训练会被数据侧**饿死**；这是 M12「首版可玩谱面」的前置，不是门禁问题。
候选修法（都要保持「窗口数含尾巴口径」不变）：① 索引落盘缓存（清单 + 参数 → 计划快照）；
② `__getitem__` 的逐行谱面缓存（同一谱面的窗口相邻取样，命中率高）；③ 桶序（`grid_key`）一并缓存。

**⑤ 门禁/训练装配路径在真实批上报「hold 配对点被拆散」（本轮新发现，未修）**

复现（2026-09-27 04:05，GPU 已空出、`data.max_samples=200`）：

```
uv run beatmorph-train --config-name phigros_masked --gates-only --device cuda --skip-env-doctor data.max_samples=200
→ 门禁输入装配失败（退出码 7）：hold 配对点被拆散：起止平坦下标 138562 / 153923 的遮盖状态不一致
```

（原始日志：`logs/real_gates_final.log`。）

- 单窗口路径（数据集的 `__getitem__` 自带 `close_hold_pairs` / 重掷种子）与合成夹具都正常，
  出问题的是**装配路径**：`ManifestBatchSource.batch(masked=True, samples=N)` 把 N 个窗口拼成一批后
  再断言一次 hold 配对 ⇒ 两处检查口径存在接缝。
- **这是训练的前置**：`--gates` 与训练都走这条装配路径。
- 待办：定位是「数据集发出的窗口本身违约」还是「collate 改了计数/遮盖」，给出默认 CI 的回归测试。

### 实施期实测（主会话，2026-09-27 第四轮：τ 轴终点缺陷 —— 语料级，且是本轮最大的数据侧缺陷）

**结论：`META.chartTime` 在真实语料里大面积不可信，而 `PhigrosChart.duration_s()` 无条件信任它**
（`max(最后一事件, chart_time_s)`）。这不是个别脏数据，而是**多数谱面**：

| 指标（全库 8551 张，`charts.jsonl` 的 `time_span_s` / `audio_duration_s`） | 实测 |
|---|---|
| `time_span_s / audio_duration_s > 10` | **4452 张（52.2%）** |
| 该比值：中位 / p75 / p99 / 最大 | **52.8 / 382 / 2575 / 3 407 201** |
| `time_span_s`：中位 / p75 / p99 / 最大 | **7766 s（2.2 h） / 59 896 s / 421 513 s / 7.49e8 s** |
| `audio_duration_s`：中位 / p99 / 最大 | 151 s / 369 s / 1282 s |
| 分布形状 | **双峰**：47% 落在音频 1.1× 以内（`chartTime` 缺省或为 0），53% 远超音频 |
| `out_of_audio_window > 0` | 228 张 |

典型样本：chart 27751 `chartTime=104 512.82` 而最后一个 note 在**第 444 拍（220 BPM ⇒ 121 s）**；
chart 77337 `chartTime=40 634.53` 而最后 note 在 174.9 s；chart 26102 `chartTime=7.49e8`（23 年）而音频 220 s。
⇒ `chartTime` 既不是秒、也不是任何一致的单位，**不能作为 τ 轴长度的事实源**。

**后果（本轮实测，200 行真实切片，`t_window=192`）**

| 口径 | 窗口数 | 索引耗时 | 抽样窗口事件数（12 个） | 说明 |
|---|---|---|---|---|
| 旧（`policy="chart"`，即 `duration_s()`） | **6 744 925**（≈3.4 万/行） | 200 s | 中位 **0**，12/12 为零事件 | 空窗 + 音频整段越界补零 |
| 新（`policy="audio"`，默认） | **17 836**（≈91/行） | 51 s | 中位 **12**，1/12 为零事件 | 与正常谱面（50–110 窗）同量级 |

- 这正是「全库索引要 2.3 h 才轮到第一个优化步」与「G1 抽到的批 K=2 / 0 事件」的**同一个根因**；
  `README` 里「1108 行 → 3361 万窗口」的规模叙事也是它造成的（正常口径下应约 780 万→**约 78 万**）。
- 记账（`DatasetIndexStats`，进训练日志）：`tau_end_truncated_rows=105`、
  `tau_end_truncated_seconds=933 万秒（2592 小时空窗）`、`events_beyond_tau_end=98`、`tau_end_fallback_rows=0`。

**处置（`beatmorph/data/dataset.py`，默认口径，可回退）**

- 新增 `DatasetConfig.tau_end_policy`：`"audio"`（默认）= `min(chart.duration_s(), 特征缓存元数据的 audio 时长)`；
  `"chart"` = 旧行为，保留以便对照（`tests/unit/data/test_dataset_tau_end.py` 逐条钉死两者差异）。
- 音频时长取自**特征缓存的 meta JSON**（`FeatureCacheMeta.duration_s`，几 KB，索引期只读一次）——
  不新增清单字段、不重跑 `pairs`、不读特征数组。
- 截断**只砍尾巴**：同一窗口在两种口径下逐格计数相同（有测试）；落在轴外的 note **显式计数**
  （`events_beyond_tau_end`），不静默丢事件。
- ⚠️ **终点口径本身仍是 plan 03 §9-14 的未裁定项**（`chartTime` / 最后一事件 / 音频时长三选一）。
  本轮只做「**消除不可信输入**」这一件事，未替该裁选定论：默认取「音频时长」是因为它是**唯一有独立
  证据（音频文件本身）的上界**，且 228 张有越界 note 的谱面说明音频之外的事件本就不该进目标。

**全库实跑（`scripts/local_index_full.py train`，2026-09-27 第五轮）**

| 项 | 旧口径（`duration_s()` 信任 `chartTime`） | 新口径（`tau_end_policy="audio"`） |
|---|---|---|
| train split 窗口数 | 3361 万（上一轮记录） | **634 952**（94.1 窗/行，6750 行） |
| 索引重建一次 | 2.3 h（外推） | **36.5 min**（0.325 s/行，实测 2193.1 s） |
| 被截断的行 | — | **3580 行（53.0%）** |
| 因此省下的空窗 | — | **307 896.8 小时**（≈35 年） |
| 轴外事件（口径丢弃，显式计数） | — | 4554 |
| 行级跳过 | — | 线数超 `k_max` 136 行 |

⇒ 「3361 万窗口」这个规模叙事的 **98% 是 `chartTime` 造出来的空窗**；真实可用规模是 **约 63 万窗**。
另：索引构建完成后**已落盘缓存**，下一次开训命中缓存只需秒级。

**同时落地的两处训练吞吐修复（plan 02 §9 ④ 的修法①②③）**

1. **索引落盘缓存**（`_plan_fingerprint` / `save_plan_cache` / `load_plan_cache`）：
   指纹覆盖 `口径版本 + 配置 + 行身份 + 谱面/特征 meta 的 (size, mtime_ns)`，因此「改了配置却复用旧计划」
   不可能发生；缓存损坏 / 指纹不符一律回退重建（缓存是加速器，不是第二个事实源），写入为原子替换。
   全库 8551 行重建一次约 36 min（0.25 s/行，本轮实测）⇒ 命中后开训不再等半小时。
2. **行级 LRU**：`chart_cache_size`（默认 8，索引期解析出的谱面留在缓存里）+ `feature_cache_size`（默认 2），
   于是同一行的多个窗口不再重复解析/重复读 40–50 MB 特征；`0` = 关闭（对照用）。

**新增门禁装配纪律（同轮，plan 07 §9-23 的第二条）**：门禁批必须**非空**（`gates.batch_min_events=1`，
取不到就重抽、重抽上限内仍取不到即抛）。真实切片上 G1 曾抽到 0 事件窗口并报了
「`loss 7819.5 -> 0.0014`」的**空过** PASS。

**待办**

- §9-Q1（格式占比）✅、§9-Q3（唯一曲目/重复率）✅、§9-Q16 ✅、共格碰撞率 ✅；
- **⑤ 的 hold 配对缺陷**（训练前置，最高优先）；
- 取批成本的缓存化（③④ 两条修法）；
- 把「含 type 通道」的碰撞统计**落进代码**（`qc.min_same_line_same_time_gap_x` 的分组键）与
  退化谱面的 QC 告警项——两者都改 `qc.py` 的统计口径，**须与 plan 03 §9-7 的 N 选型联动**。
