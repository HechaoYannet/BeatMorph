# RFC-0024 — sayobot 数据源与 §4.3 质量过滤激活

- 状态：采纳 ｜ 提出日期：2026-07-31 ｜ 决定日期：2026-07-31
- 提出者：数据组
- 影响模块：plan 08（`beatmorph/data/*`、`scripts/download_sayobot.py`）、`beatmorph/data/pipeline/embed.py`、`beatmorph/data/manifest.py`、`pyproject.toml`、`configs/data/download.yaml`

## 背景

奠基文档 §4.3 规定训练数据须经质量过滤「`≥3 星 且 play_count > 500`」，Plan 08 §3.1 把该过滤落在 `PreprocessPipeline._quality_filter`（常量 `MIN_STARS=3.0` / `MIN_PLAY_COUNT=500`），从 `chart.meta["difficulty_rating"]` / `["playcount"]` 取值。但：

- `.osu` 文件本身**不含** star rating 与 play_count——这两个是 osu! 服务端的集级统计量。
- 当前流水线因此**永远走 "Phase 1 relaxed pass"**（宽松通过），§4.3 过滤名存实亡（见 `beatmorph/data/pipeline/embed.py::_quality_filter` 的 `if rating is None` 分支）。
- Plan 08 §1 把「抓取与版权合规流程本身」声明为**工程外流程**，仓库此前只有解析侧（`OsuManiaReader` / `PreprocessPipeline`），**没有数据获取侧**。
- Phase 1 里程碑（奠基 §7）需要 10K 真实 `.osu`+音频 + 质量 filtering，缺数据获取侧无法起步。

因此本 RFC 把数据获取侧纳入仓库（**偏离 plan 08 §1「工程外流程」声明**，特此登记），并定义一套机制让 §4.3 过滤真正生效。

## 提议

### 1. 数据源：osu.sayobot.cn 镜像（无鉴权）

逆向自 HAR 抓包（`osu.sayobot.cn`）：

- **列表/搜索**：`POST https://api.sayobot.cn/?post`
  - `Content-Type: text/plain`（**非常规**：body 是 JSON 字符串，非 `application/json`；用 `content=` 而非 `json=`）
  - body：`{"cmd":"beatmaplist","limit":25,"offset":0,"type":"search","keyword":"","mode":8,"class":31,"subtype":63,"genre":1535,"language":4095}`
    - `mode:8` = osu!mania（位掩码 bit3）
  - 响应：`{"data":[{sid,title,artist,creator,play_count,order,approved,modes,favourite_count,lastupdate,...}, ...]}`
    - `sid` = 谱面集 id；`order` = 星级 float（未 Ranked/Pending 时 `0.0`；`approved:1`=Ranked / `:3`=Pending）；`play_count` = 集级总 play。
- **下载**：`GET https://txy1.sayobot.cn/beatmaps/download/full/{sid}?server=auto` → `302` 跳 `https://cmcc.sayobot.cn:25225/beatmaps/{前3位}/{余}/full?filename=...` → `200 application/octet-stream`，body 即 `.osz`。
  - `.osz` = ZIP（内含多难度 `.osu` + 引用的音频 `audio.mp3`/`.ogg`）。
  - 需浏览器 `User-Agent` + `Referer: https://osu.sayobot.cn/`。
  - `cmcc.sayobot.cn:25225` 证书链 httpx 默认无法验证 → 遇 `SSLCertVerificationError` 降级 `verify=False` 并 warn（公开数据下载可接受，但须显式记录）。

### 2. 下载脚本：`scripts/download_sayobot.py`

search/list → 分页 → 下载 `.osz` → `zipfile` 解压到 `data/raw/{sid}/` → 追加写一行到 `data/raw/manifest.jsonl`（JSONL，断点续传友好）。策略：4K-only 过滤、`--skip-unranked`（默认跳 `order=0.0` 或 `approved=3`）、`--rate-delay` 礼貌限流、连接级重试、已存在 sid 跳过（续传）。

> 落地目录 `data/raw/` 已被 `.gitignore` 覆盖，且全局 `*.osz`/`*.mp3`/`*.wav` 均被忽略，下载产物**永不入库**（CLAUDE.md §3 红线 5）。`data/fixtures/**` 仍为唯一 audio 例外。

### 3. manifest 与元数据注入：`beatmorph/data/manifest.py`

- `load_manifest(path)` 把 JSONL → `{sid: SetRecord}`（跳坏行）。
- `ManifestInjector.apply(chart_meta)` 按 `chart.meta["beatmap_set_id"]`（`OsuManiaReader` 已存为 int）查 manifest，**只加不覆盖**地注入：
  - `difficulty_rating` (float) ← SetRecord.stars（=API `order`）
  - `playcount` (int) ← SetRecord.play_count
  - `license` (str) ← `"academic"`（奠基 §4.3 / plan 08 §6 字面写作 `Chart.meta={"license":"academic"}`，本 RFC 纠偏为 **`setdefault` 合并**，避免误删 `beatmap_set_id`/`creator` 等既有键）
  - `approved` (int) ← 审计用
- **不改 `Chart.meta: dict[str, str|int|float]` 契约类型** → 无需 contract RFC。

### 4. 管线对接：`PreprocessPipeline`

- 构造增可选 `manifest_path: Path | None = None`；命中则惰性构造 `ManifestInjector`，不传/不存在 → `None`（**向后兼容**，既有无 manifest 调用零回归）。
- `run()`：`glob("*.osu")`→`rglob("*.osu")`（下载器按 `{sid}` 子目录落盘，必须递归；rglob 仍能找到扁平 fixture，既有测试不变）；在 `_quality_filter` **前**调用 `injector.apply`，再 `chart.meta.setdefault("license","academic")`。
- **`_quality_filter` 逻辑不变**——激活全靠 meta 注入。注入后 `difficulty_rating`/`playcount` 非空 → 走真实过滤分支，「Phase 1 relaxed」日志不再出现。
- 未 Ranked（`order=0.0`）注入 `difficulty_rating=0.0` → `0.0 < 3.0` 自动剔除，符合 §4.3「community-validated」语义。

### 5. 依赖：新增 `data` 可选 extra

`pyproject.toml` 增 `[project.optional-dependencies] data = ["httpx>=0.27"]`（仅下载脚本需要，不污染运行时 base deps，与 `pyarrow`∈train、`faiss`∈rag 模式一致）；`dev` 组增 `httpx` + `respx`（httpx mock，离线单测）。

### 6. 配置与环境变量

`configs/data/download.yaml` 用 `${oc.env:BEATMORPH_RAW_DIR,data/raw}` 解析器；新增环境变量 `BEATMORPH_RAW_DIR`（默认 `data/raw`，与既有 `BEATMORPH_DATA_DIR` 同口径）。

## 备选方案

1. **osu! 官方 API**：需 API key、限流更紧、TOS 更严，且官方对批量镜像抓取有顾虑。放弃。
2. **Bloodcat / Chimu 等其它镜像**：域名长期不稳、覆盖率波动大。sayobot.cn 在国内可达性最好、API 无鉴权。放弃。
3. **管线运行期联网查 per-diff 星级**：在 `PreprocessPipeline` 内调用 osu!API 查每个 `beatmap_id` 的精确星级。代价：**训练管线引入网络依赖**，变得脆弱、不可离线复现。本 RFC 选择**下载期注入**，管线全程离线。放弃。
4. **扁平 `.osu` + sid 前缀命名**：避免 rglob 改动。但 `.osu` 内的 `AudioFilename` 仍指向 `audio.mp3`，扁平目录里多 set 音频名撞车，需重命名音频并改写 `.osu` 内字段——**变更源数据**，不可接受。放弃。
5. **Parquet manifest**：列式，append 需整表重写，断点续传不友好。JSONL 追加即写、易 `grep` 调试。放弃。

## 后果

- **plan 08 §1 偏离**：「工程外流程」现有了仓库内实现（下载脚本）。plan 08 §1/§4.3 表述须注明「获取侧见 RFC-0024 + `scripts/download_sayobot.py`」。
- **§4.3 过滤真正生效**：传 `manifest_path` 后 `passed_filter` 显著低于宽松通过数，parquet 内每行 `difficulty_rating≥3.0 且 playcount>500`。不传 manifest 行为不变（向后兼容）。
- **集级 stars 近似 diff 级**：列表 API 每 set 一个 `order`，注入到该集所有 diff。等价于把 §4.3 当作**集级质量闸**（与 §4.3「community-validated good maps」的集级语义一致）。代价：同集不同难度（如 2 星 Easy / 7 星 Insane）会被同一 `order` 过滤，可能误放行低难度 diff 或误拒高难度 diff。Phase 1 可接受；若需 diff 级精度，后续可开 RFC 在下载期按 `beatmap_id` 查 osu!API 缓存精确星级（仍保持管线离线）。
- **`order=0.0` 语义**：未 Ranked/Pending 无社区评分，注入 `0.0` 后被过滤剔除。默认 `--skip-unranked` 更在下载期即省带宽。
- **新增依赖 `httpx`（data extra）+ `respx`（dev）**：`uv sync --extra data` 安装。
- **TLS 兜底**：`cmcc:25225` 默认 `verify=True`，遇证书错降级 `verify=False` + warn。公开数据下载可接受，但作为已知风险记录（见下）。
- **音频路径解析约定**：`Chart.audio_path` 仅为 `AudioFilename` 的 basename（如 `audio.mp3`），相对 `.osu` 父目录解析。plan 08 Step3（MERT 提取，当前 NotImplementedError）落地时须据此相对 set 目录解析，**不可相对 `raw_dir`**。

## 风险与开放问题

- **sayobot TOS / 限流**：非官方镜像。已设 `rate_delay≥1.0s` + 单线程 + 重试 + 续传。10K+ 规模须实测其容忍度，若出现 429/封禁则降速或换源。
- **cmcc:25225 TLS**：能否 pin CA bundle 显式校验待查；暂以 `verify=False` 兜底 + 告警。
- **集级 vs diff 级 stars**：见后果。后续 per-diff 精确星级为可选增强 RFC。
- **`integration` 未被 `test-fast` 排除**：Makefile `test-fast = not slow and not gpu and not e2e`，联网集成测试须同时标 `@pytest.mark.slow` + `BEATMORPH_LIVE_NETWORK=1` 守卫，否则 CI 触网。此为既有隐患，建议后续 Makefile 改 `not integration`（非本 RFC 范围）。

## 关联

- 奠基：§4.3（质量过滤）、§4.2（预处理流水线）、§7 Phase 1（10K MERT 离线提取前置）。
- plan 08：§1（工程外流程边界）、§3.1（`PreprocessPipeline` 契约）、§4.3（过滤口径）、§6（license 标记）。
- RFC-0020（解析产出统一到 `Chart` IR，本 RFC 复用其 `meta` 字段）。
- `docs/knowledges/osu-file.md`（`.osu` 格式，`AudioFilename` 字段；本 RFC 补充 `.osz`=ZIP 这一格式文档未覆盖的事实）。
