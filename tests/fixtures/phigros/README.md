# Phigros 微缩夹具（`tests/fixtures/phigros/`）

> 对应：plan 02 §3.7 / M2 ｜ 合规约束：CLAUDE.md 红线 5 附注（数据不入库、只保留微缩夹具）

## 0. 一句话结论

**本目录所有谱面内容均为手工构造（非真实谱面），不是从任何真实 Phira 谱面裁剪的片段。**
夹具**不含音频、不含曲绘**，全部是纯文本 + 一个由文本部件确定性打包出来的 zip。

## 1. 文件清单

| 文件 | 体积 | 用途 |
|------|------|------|
| `rpe_min.json` | 5 814 B | 标准 RPE 正样本：4 条判定线、四类 note type、背面 note（`above` ∈ {0,1,2}）、事件多层、跨层求和、事件间空隙（补洞）、`father` 嵌套、`isCover=2`、越界 `positionX` |
| `pec_masquerade.json` | 198 B | **内容为 PEC 文本、后缀为 `.json`**（陷阱 1：后缀不可信） |
| `pkg_min/` | 14 315 B | 最小谱面包的**文本部件**：`info.yml` + 谱面文件（`min_chart.json`，**不叫** `chart.json`）+ 干扰 json（`resources/effects.json`，比谱面文件**更大**）+ decoy `chart.json` |
| `__init__.py` | 4 245 B | 夹具读取与 `build_pkg_zip()`（确定性打包） |

合计 24 572 B（上限 128 KB）；单文件最大 11 481 B（上限 32 KB）。

## 2. 构造方式

### 2.1 `rpe_min.json` / `pec_masquerade.json`

**手工构造**：字段名、默认值、单位取自 `docs/knowledges/phigros-format.md` 与
`docs/knowledges/phigros-units-and-geometry.md` 的 A 级事实（prpr 源码 / Phira 官方文档），
内容与任何真实谱面无关（曲名/谱师名一律写成 `fixture-*`）。

- `rpe_min.json` 的 BPM 段：120 BPM 起于 0 拍、180 BPM 起于第 8 拍 → 覆盖多 BPM 段积分。
- 事件轨刻意留出**空隙**（第一层 `moveXEvents` 在第 2 拍结束、下一条从第 6 拍起），
  用于验证「补洞后空隙内取值 == 前一事件终值」。
- 两层各贡献一部分位移（60 + 40 量级），用于验证**跨层求和**（不是取最上层）。
- 第 2 条线（`child-line`）的 `father = 1`，覆盖 `father` 递归。
- 第 4 条线**没有** `eventLayers` 键，第 2 条线是 `[null]` → 覆盖三态归一。
- 越界 note：`positionX = 720.0`（> 正负半宽），用于「越界只统计不钳位」用例。
  ⚠️ 该值是**夹具数据**，不是物理常量；测试里的越界坐标一律由
  `RPE_STAGE_HALF_WIDTH * 系数` 现算（AGENTS.md §3.3）。
- `pec_masquerade.json` 的 PEC 行结构参考 `docs/knowledges/phira-dataset-survey.md`
  §4.4 的实测样例（首行版本号 + `bp`/`cp`/`cm`/`ca`/`cv`/`n1..n4` 行）。

### 2.2 `pkg_min/`（**不提交二进制 zip**）

`pkg_min/` 目录里放的是**文本部件**；zip 由 `tests/fixtures/phigros/__init__.py` 的
`build_pkg_zip(dest, info_text=None)` 用标准库 `zipfile` **确定性**构造：

1. 固定条目顺序：`info.yml` → `min_chart.json` → `chart.json` → `resources/effects.json`；
2. 固定 `date_time = (1980, 1, 1, 0, 0, 0)`、`compress_type = ZIP_DEFLATED`、
   `external_attr = 0o644 << 16`；
3. 测试把它物化到 `tmp_path`（同一进程内构造两次字节完全一致，见
   `test_pkg_zip_is_deterministic`）。

这样做的理由：夹具保持**纯文本、可 diff、可 review**，且不往仓库里塞二进制；
同时能满足「必须能被测试稳定读取」的要求。

包内三个陷阱：

| 陷阱 | 落点 |
|------|------|
| R2：谱面文件**不叫** `chart.json` | `info.yml.chart = min_chart.json`；另放一个 decoy `chart.json`（内容是另一个 RPE，`META.name = decoy-chart`） |
| R1：**不得取最大的 json** | `resources/effects.json` 比谱面文件大一个数量级（11 481 B vs 1 503 B） |
| R3：**不得依赖 `info.yml.format`** | `info.yml` 里 `format: null`（实测全库恒为 null） |

另含负样本入口：`build_pkg_zip(info_text=<把 chart 指向不存在条目>)` →
`ChartPackage` 必须报错，**不得**回退到「取最大 json」或「取 `chart.json`」。

### 2.3 不含音频与曲绘

`info.yml` 里声明了 `music: min.ogg` 与 `illustration: min.png`，但**故意不入包**
（合规：夹具不得含音频与曲绘）。`ChartPackage.missing_music` 会记录该状态并告警——
音频缺失在真实数据里是异常，在夹具里是**有意为之**。

## 3. sha256（信息性记录，非断言）

| 文件 | sha256 |
|------|--------|
| `__init__.py` | `f81dc1c2e919feaa24ad63ba24d3a945d1fd4823a672710815b625d08c9871ee` |
| `pec_masquerade.json` | `ad58e96bd0715c8202ecb63f9415d07bab8f1a23c40537f5882544b07e89a36a` |
| `rpe_min.json` | `195c164515ca6cca813d56988b9d7c052a63d94669a6ebe5158d49aaf285c1e7` |
| `pkg_min/chart.json` | `17a76dc18b1c88e1b298d9d425e42b1140124837ffbb48d838873b138a7eca2a` |
| `pkg_min/info.yml` | `a6e2a2ac6b1ea0f036e81de64c23355f3df872f0a12262c56702a25841bbd40a` |
| `pkg_min/min_chart.json` | `446d5698b30260e1d884fc5ea05457cbcdc4d22c9ce9fdd9f4ee68efb36519ba` |
| `pkg_min/resources/effects.json` | `5457224f10fb647364c8c02c71bcca4f824775e85647d125fc2bda75b08af0d4` |

（哈希用 `Get-FileHash -Algorithm SHA256` 于提交前计算；测试**不断言**具体哈希值——
跨平台/跨 zlib 版本的字节级稳定性不作保证，断言的是「构造确定」与「体积达标」。）

## 4. 合规说明（重要）

- 本目录内容**不是**任何真实谱面的片段，不含音频、不含曲绘，不包含真实 Phira 用户内容。
- 若日后需要对外分发任何夹具，须回到 plan 02 §6-M8 硬约束④（**发布权重/数据前必须重新裁定**）。
- `pkg_min` 的纯手工构造版本是可对外分发的现成退路（plan 02 §9-8）。
