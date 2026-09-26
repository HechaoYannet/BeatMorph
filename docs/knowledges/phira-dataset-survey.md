# Phira / Phigros 社区谱面数据集与获取途径调研

> 状态：信息准备 B（数据源侧）｜ 编制日期：2026-09-26 ｜ 适用：RFC-0029（Phigros 多判定线标记点过程）
> 关联：[phigros-format.md](phigros-format.md)（RPEJSON 逐字段规范）｜ [RFC-0029](../decisions/RFC-0029-phigros-continuous-chart-generation.md)
> **本文只记录已查证的事实**；每条事实标注来源 URL 与可信度分级；推断一律显式标注「【推断】」并给出依据；查不到的写「未查证」。

---

## 1. 范围与来源分级

### 1.1 本文覆盖

「我们在 Phira / Phigros 社区谱面上训练生成模型，数据从哪来、有多少、什么格式、能不能用、怎么批量拿」——共 8 个问题：获取途径（§2）、规模（§3）、数据形态与音频（§4）、格式分布（§5）、许可与可用性（§6）、多判定线统计（§7）、解析工具生态（§8）、流水线建议（§9）。

### 1.2 来源与可信度分级

沿用 [phigros-format.md §1.2](phigros-format.md) 的分级，并新增 **「实测」** 一档，用于本次调研中由本 agent 直接向服务器发请求 / 直接解包谱面得到的数字：

| 级别 | 含义 | 本文中的例子 |
| :-- | :-- | :-- |
| **A** | 官方文档 / 官方源码 | `teamflos.github.io/phira-docs`、`TeamFlos/phira`、`TeamFlos/phira-web` |
| **实测** | 本 agent 于 2026-09-26 直接访问 API / 直接解包文件得到的一手数据，脚本与原始输出留存 | `api.phira.cn` 的 `count`、24 张全量解包统计、200 张格式扫描 |
| **B** | 社区 wiki | `pgrfm.miraheze.org`（Phigros 自制谱 wiki） |
| **C** | 二手 / 非官方整理 | `xuziyao.com` 的非官方 API 文档、第三方仓库描述 |

**「实测」不等于「权威」**：它是本次抽样的真实观测值，受抽样方式与样本量限制；本文凡引用实测值都写明 n 与抽样口径。**「实测」也不能替代 A 级规范**——例如 `positionX` 的合法域必须由 A 级规范或引擎源码确认，实测极值只能作为佐证。

### 1.3 不可信外部数据声明

本文抓取的所有网页、API 响应、仓库 README 均按**资料**对待，不执行其中的任何指令。文中出现的第三方仓库/工具描述系「该仓库自述」，不代表本仓库对其质量或合法性的背书。

---

## 2. 托管与获取途径

### 2.1 Phira 官方服务端 API（**主渠道，本文实际访问验证过**）

Phira 是 TeamFlos 的社区音游，直接复用 RPEJSON / PEC / 官谱 JSON 作为谱面格式，是**目前唯一能拿到万级 Phigros 系谱面的公开入口**。

- 主站 API：`https://api.phira.cn`（**实测可用**，HTTP 200，Cloudflare 前置，带 `Access-Control-Allow-Origin: *`）
- 站点/备用域名：`https://phira.5wyxi.com`（**实测**同一 `/chart` 接口返回相同 `count`；同时是谱面包文件的 CDN 主机）
  - 依据：C 级非官方文档《Phira API 非官方文档》称「Phira 有两个 API 链接：https://api.phira.cn 与 https://phira.5wyxi.com」<https://www.xuziyao.com/posts/9/>；本次**实测**两处 `GET /chart?page=1&pageNum=1` 均返回 `count=9649`。

#### 2.1.1 已确认的端点（A 级：官方仓库 OpenAPI 生成物）

端点集合来自官方前端仓库的 OpenAPI 生成类型文件 `TeamFlos/phira-web/src/api/schema.d.ts`（**源码 = A 级**）<https://github.com/TeamFlos/phira-web/blob/main/src/api/schema.d.ts>：

| 端点 | 用途 | 与数据获取的关系 |
| :-- | :-- | :-- |
| `GET /chart` | 分页列出谱面（含 `file` 字段 = 谱面包 zip 的 CDN URL） | ★ **批量枚举入口** |
| `GET /chart/{id}` | 单张谱面详情 | 补齐元数据 |
| `GET /chart/multi-get?ids=` | 批量取详情 | ★ 批量补齐元数据（`ids` 为字符串） |
| `GET /chart/{id}/versions` | 该谱面的版本列表 | 可用于取历史版本（未实测） |
| `GET /chart/{id}/version/{version_id}` | 单版本 | 未实测 |
| `GET /chart/{id}/stabilize-status` | 评议状态 | — |
| `GET /collection`、`GET /collection/{id}` | 谱面合集 | ★ 另一个枚举维度（未实测） |
| `GET /event`、`GET /event/{id}` | 活动 | — |
| `GET /user`、`GET /user/{id}` | 用户 | 可按 uploader 过滤 |

> ⚠️ **C 级文档与 A 级源码的一处冲突（以源码为准）**：非官方文档把列表响应写成 `{"count":…, "result":[…], …}`，但**实测**返回的键是 **`results`（复数）**，且字段 `preview`、`file`、`illustration`、`rating`、`ratingCount` 与文档一致。解析时请以实测为准并按 `results` 取值。

#### 2.1.2 `GET /chart` 的查询参数（A 级源码）

来自 `schema.d.ts` 的 `chart_list` 操作定义（**实测**：`page`/`pageNum`/`division`/`type` 均生效）：

| 参数 | 类型 | 说明（源码注释直译） |
| :-- | :-- | :-- |
| `page` | number | 页码，从 1 起 |
| `pageNum` | number | 每页条数。**实测上限 = 30**：`pageNum=31` 返回 HTTP 400，`pageNum=30` 正常 |
| `search` | string | 关键词 |
| `order` | string | 排序 |
| `tags` | string | 标签过滤 |
| `uploader` | number | 按上传者用户 ID 过滤 |
| `reviewed` / `stable` / `ranked` / `stableRequest` | boolean | 按审核 / 上架 / rks 计入 / 上架申请状态过滤 |
| `division` | string | 分区标签（`regular` / `troll` / `plain` / `visual`） |
| `rating` | string | 评分区间，形如 `"0.3,1.0"`（**注意：这是社区评分，不是定数 difficulty**） |
| `type` | number | 谱面类别：`0` ranked / `1` special / `2` unstable / `3` any |

> ⚠️ **源码中没有按「定数 `difficulty`」过滤的参数**（A 级：`chart_list` 参数表已完整列出）。若需要按定数分层抽样，只能本地过滤。
> ⚠️ **`division` 语义有歧义（实测）**：`division=regular` 返回的 `count` 与不加该参数时**完全一致**（均 9649），而 `troll`/`plain`/`visual` 分别返回 657/3326/169。**推断**：`regular` 表示「无分区标签」而服务端未实现该分支的过滤；**不要**把 `division=regular` 的结果当作 regular 子集大小。此项**未查证**（无源码级依据）。

#### 2.1.3 分页可达性（实测）

- `page=322&pageNum=30` → 返回 19 条（`322*30-30+19 = 9649`），说明**全量 9649 张可被完整分页枚举**，共 322 页。
- `page=300&pageNum=30` → 正常返回 30 条。
- 未观测到限流响应头（无 `X-RateLimit-*`，无 `Retry-After`）；**是否有限流策略未查证**。批量抓取时应自行限速。

#### 2.1.4 谱面包下载（实测）

`/chart` 与 `/chart/{id}` 响应中的 **`file` 字段**就是谱面包 zip 的直链，托管在 `https://phira.5wyxi.com/files/<uuid>`：

- **实测**：`GET` 该 URL 返回完整 zip；服务端**支持 HTTP Range**（`Range: bytes=0-1023` → `206`，`Content-Range: bytes 0-1023/8201243`）。
- 这一点很关键：**可以只抓 zip 的中央目录 + 单个条目的字节区间**（见 §9.2），把「先探格式再决定是否全量下载」的成本压到 ~10 KB/张。
- `preview` 字段是试听音频直链，`illustration` 是曲绘直链——**两者都是独立于 zip 的 CDN 文件**，可作为音频缺失时的备选（但 preview 通常只是 15 秒切片，见 §4.3）。

#### 2.1.5 前端仓库里的一处**版权审查**线索（A 级）

`schema.d.ts` 同级目录存在 `src/components/review/CopyrightResults.vue`（5131 B）与 `src/review/metadata.ts`，说明 Phira 审核流程里**有明确的版权检查环节**（[源码树](https://github.com/TeamFlos/phira-web/tree/main/src/components/review)）。这不是数据集许可依据，但说明上传侧被要求声明权利（见 §6）。

### 2.2 GitHub 上的谱面仓库（**都是小规模，非万级替代品**）

以下均为 **A 级（仓库元数据来自 GitHub API，本次实测查询）**，但**内容规模都很小**，且**多数没有声明许可证**：

| 仓库 | 内容 | 规模（GitHub `size` 字段） | 许可证 | 最近推送 | 能否当训练数据 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [yuameshi/phigros-charts-repo](https://github.com/yuameshi/phigros-charts-repo) | **官方谱**（供 PhiCommunity / phigros-html5 使用），根下有 `official.json`/`fanmade.json`/`special.json` 清单，约 20 个曲目目录 | 111754 KB（≈109 MB） | **未声明** | 2026-01-21 | 可作**官谱格式**与几何核对的参考；不是社区谱规模来源 |
| [yuameshi/PhiCommunity-Charts-Repo](https://github.com/yuameshi/PhiCommunity-Charts-Repo) | **社区自制谱**（供 PhiCommunity），约 24 个曲目目录 | 141207 KB（≈138 MB） | **未声明** | 2024-06-25 | 可作少量 PEC/官谱 JSON 样本；不是主来源 |
| [zwtwz/PhigrosChartTransformer](https://github.com/zwtwz/PhigrosChartTransformer) | Python，「把 phigros 官方谱面自动转换为 phira 的 pez 谱面」 | 82735 KB（≈81 MB） | **未声明** | 2025-11-06 | 内含官谱转换产物，可作格式对照 |
| [swordalt/phigros-chart-downloader](https://github.com/swordalt/phigros-chart-downloader) | TS，导出/下载**官方** Phigros 歌曲资源与可玩谱面 | 2241 KB | **MIT** | 2026-09-24 | 面向官谱资源，非社区谱；工具思路可参考 |
| [7aGiven/Phigros_Resource](https://github.com/7aGiven/Phigros_Resource) | 「Phigros apk 资源提取」 | — | — | — | 音频/曲绘侧，非谱面 |
| [Klrohias/nFast-Pkg](https://github.com/Klrohias/nFast-Pkg) | 「a player app and editor app for fanmade Phigros chart」 | 73171 KB | **未声明** | 2026-07-03 | 未展开核实其内置谱面 |

> 结论：**GitHub 不是 Phigros 社区谱面的规模来源。** 搜索 GitHub `topic:phigros` 共 69 个仓库（**实测**），按星数排序的前列全是查分器/模拟器/资源提取/播放器，谱面仓库合计不足 50 首曲目量级。
> 「Phigros 自制谱论坛」：**未查证**。本次检索未找到可作为批量数据源的官方或半官方论坛；社区发布似乎以 QQ 群 / B 站 / Phira 站内上传为主，**没有可下载的公开档案库**。

### 2.3 其它播放器 / 平台（潜在二级来源）

| 项目 | 语言 | 许可证 | 与数据获取的关系 |
| :-- | :-- | :-- | :-- |
| [PhiZone/player](https://github.com/PhiZone/player) | TypeScript | MPL-2.0 | HTML5 Phigros 谱面播放器/模拟器（91★，2026-09-26 仍在推送）。PhiZone 是另一个 Phigros 社区平台；**其谱面托管 API 是否公开、是否可批量拉取——未查证** |
| [TeamFlos/phira-mp](https://github.com/TeamFlos/phira-mp) | Rust | Apache-2.0 | 多人游戏服务端与客户端库，**不托管谱面** |
| [liquidhelium/phirs](https://github.com/liquidhelium/phirs) | Rust | 未声明 | 第三方 Phigros 播放器（2022 年停更） |
| [qaqFei/phispler](https://github.com/qaqFei/phispler) | Python | MIT | Phigros 模拟器，可作渲染/校验参考 |

---

## 3. 规模

### 3.1 总量（**实测，A 级数据源**）

`GET https://api.phira.cn/chart?page=1&pageNum=1` → `{"count": 9649, "results":[…]}`（2026-09-26 实测）。

**Phira 主站共 9649 张谱面**。这是本次调研中**唯一可靠的规模数字**。

> 口径说明：`count` 是服务端对当前查询条件的命中总数。以下分项也是同一接口的 `count`（**实测**）：

| 查询 | `count` | 占比 |
| :-- | --: | --: |
| 无条件（全部） | **9649** | 100% |
| `type=0`（ranked） | 577 | 6.0% |
| `type=1`（special） | 50 | 0.5% |
| `type=2`（unstable） | 9022 | 93.5% |
| `type=3`（any） | 9649 | 100% |
| `stable=true`（已上架） | 627 | 6.5% |
| `ranked=true`（计入 rks） | 583 | 6.0% |
| `reviewed=true`（已过审） | 9360 | 97.0% |
| `division=troll` | 657 | 6.8% |
| `division=plain` | 3326 | 34.5% |
| `division=visual` | 169 | 1.8% |
| `division=regular` | 9649（**过滤未生效，见 §2.1.2**） | — |

一致性检验：`type` 三分项 577 + 50 + 9022 = 9649，与总数吻合（**实测**），说明 `type` 是完备划分。

### 3.2 按定数（difficulty）分布

- **API 不提供按定数过滤的参数**（A 级：`chart_list` 全参数表，见 §2.1.2），只能本地统计。
- 本次 24 张全量抽样中 `difficulty` 范围 **9.9 – 18.5，中位数 15.6**（**实测**，n=24，抽样偏向近期更新，见 §3.4 偏差说明）。
- `level` 字段是**自由文本**，格式极不统一。实测出现：`"AT  Lv.16"`、`"IN.14"`、`"IN 15.7"`、`"IN Lv.14"`、`"Lv.16(16.1）"`、`"Love  Lv.9.9"`、`"sweet"`、`"酔い"`、`"14.514"`（**实测**，n=24）。
  → **解析时必须用 `difficulty`（f32）做数值分层，绝不能 regex 解析 `level` 字符串。** 且 `difficulty` 存在浮点误差（实测 `14.900001`、`18.000004`），比较前应 round 到 0.1。

### 3.3 总量与存储估算

| 指标 | 值 | 依据 |
| :-- | :-- | :-- |
| 谱面总数 | **9649** | 实测 API `count` |
| 单包 zip 大小（n=193 实测） | min 1.77 MB / **中位 6.56 MB** / 均值 7.86 MB / max 31.4 MB | 200 张抽样扫描 |
| 谱面 JSON 解压后大小（n=193） | min 57 KB / 中位 3.74 MB / max 49 MB | 同上（zip 中央目录 `uncompressed size`） |
| 【推断】全量 zip 体积 | **≈ 76 GB**（9649 × 7.86 MB） | 依据：均值外推；**未查证真实总量**，仅作容量规划量级参考 |
| 【推断】其中音频占比 | 约 40–60%（单包 .mp3/.ogg/.wav 合计常 2–7 MB） | 依据：24 张全量抽样的扩展名字节分布 |

### 3.4 抽样偏差声明（**重要**）

本次两轮抽样（24 张全量解包、200 张格式扫描）的候选池都来自 `GET /chart` 的**默认排序前若干页**（200 张那轮取了前约 400 条元数据）。

- 实测首页返回的谱面 `created` 为 2026-09-25，说明**默认排序近似「最近更新优先」**。
- 因此样本**偏向近期上传/更新的谱面**，对「已淘汰格式（PEC）占比」存在**低估**风险。
- 为对冲该偏差，另做了一轮**深页抽样**（页码 210–325，即列表尾部/最旧一端），结果见 §5.2。
- **全库无偏统计未做**（需要 322 页 × 30 条枚举 + 全量下载，约 76 GB）。

---

## 4. 数据形态与音频

### 4.1 一个 Phira 谱面包 = 一个 zip（**实测**）

**实测解包** `GET /chart/1000`（アイドル，AT Lv.15）完整目录：

```
AT15.json            6 509 000 B     ← 谱面文件（RPEJSON）
idol-48k-0.8.ogg     7 417 237 B     ← 音频（随谱分发）
illustration.jpg       579 045 B     ← 曲绘
info.yml                   809 B     ← ChartInfo（YAML）
info.txt                   163 B     ← 旧式文本元信息（PhiEditer 遗留）
extra.json              87 882 B     ← 扩展特性（Effects / 视频背景）
ai1.png / ai1_mh.png / ai2.png / ai2_mh.png   ← 特效贴图
Bloom.fs                   702 B     ← 着色器
```

**实测** `GET /chart/7039`（AT Lv.18）完整目录：`24432296.json` (248 KB) + `56695308.mp3` (4.8 MB) + `56695308.jpg` + `info.yml` + `info.txt`。

两次实测都呈现同一结构：**info.yml + 谱面文件 + 音频 + 曲绘（+ 可选 extra.json / 贴图 / 着色器）**。与 A 级文档一致：「一个 Phira 谱面包 = `info.yml` + 谱面文件（默认 `chart.json`）+ 音乐 + 插图（+ 可选 `extra.json`）」（[Phira 文档·谱面文件格式](https://teamflos.github.io/phira-docs/chart-standard/chart-format/index.html)），也与 [phigros-format.md §11](phigros-format.md) 相符。

### 4.2 包内条目统计（**实测**，n=196 张成功读出 zip 中央目录的谱面包）

| 条目 | 出现率 | 备注 |
| :-- | --: | :-- |
| `info.yml` | **196/196 (100%)** | ChartInfo；全量解包的 24 张中其 `format` 字段**全部为 `null`**，必须按内容嗅探格式 |
| 音频（`.mp3`/`.ogg`/`.wav`/`.m4a`/`.flac`/`.opus`/`.aac`） | **196/196 (100%)** | 见 §4.3 |
| 曲绘（`.jpg`/`.jpeg`/`.png`/`.webp`） | **196/196 (100%)** | — |
| `info.txt` | 191/196 (97.4%) | 旧式元信息，可忽略 |
| `extra.json` | 9/196 (4.6%) | 特效 / 视频背景 |
| `unlock.mp4` | 5/196 (2.6%) | 解锁动画视频 |
| **谱面文件名为默认 `chart.json`** | **0/196 (0.0%)** | ⚠️ 见下 |
| 包内条目数 | min 4 / **中位 5** / max 15 | — |

⚠️ **实测反直觉发现**：A 级文档说「RPE 生成的谱面通常为 `chart.json`，PE 生成的谱面通常为 `xxx.pec`」（[Phira 文档·谱面信息](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html)），但本次 **196 张样本中没有任何一张的谱面文件叫 `chart.json`**——实测文件名多为 `<数字ID>.json`（如 `1817439042209534.json`、`5120398229067517.json`），偶见 `AT15.json`、`drd_IN.pec`。
→ **解析器绝不能按文件名（含默认名 `chart.json`）定位谱面文件，必须读 `info.yml.chart`。**

**注意**：单个谱面包的 `.json` 解压总量最高见过 **294 MB**（191 个条目，含大量特效/粒子资源，见样本 id 45756），但那与「谱面文件」无关——谱面文件本身只有 3.25 MB。**扫描时务必按 `info.yml.chart` 精确定位谱面文件，不要「取最大的 json」。**

### 4.3 音频是否随谱面分发？—— **是，且近乎 100%**（**实测**）

- 本次 200 张抽样中，**能读出 zip 中央目录的 196 张，全部含有音频扩展名条目（196/196 = 100%）**。
- 音频扩展名实测分布：`.mp3`、`.ogg`、`.wav`（另有少量 `.m4a`/`.flac`/`.opus` 兜底匹配）。
- 音频文件名由 `info.yml.music` 指定，**实测形如** `1817439042209534.mp3`、`idol-48k-0.8.ogg`、`56695308.mp3`——**文件名无规律，必须读 `info.yml.music`，不能按扩展名猜**。

**这意味着音频版权随谱面包一起被分发**（见 §6）。

**若音频缺失/不可用，替代途径**（按可行性排序，均有代价）：

1. **同曲复用**：Phira 上同一首歌常有多张谱面（不同谱师/难度）。可按 `name` + `composer` 聚合，从任一含音频的包取音频，**仅需下载一次**。这是最省流量的做法，且完全在 Phira 数据内。
   - 依据：**未查证**同曲重复率的精确数字；但 `/chart?search=<曲名>` 可按曲名检索（A 级参数），聚合可行。
2. **`preview` 字段**：`/chart/{id}` 返回的 `preview` 是试听音频 CDN 直链（**实测**存在）。⚠️ 但 `info.yml` 的 `previewStart`/`previewEnd`（实测 56.0 → 68.3 秒）表明它通常只是**一段十几秒的切片**，**不足以训练**。
3. **游戏本体资源提取**：多个社区项目提取官方 Phigros 的音频（如 [7aGiven/Phigros_Resource](https://github.com/7aGiven/Phigros_Resource)「Phigros apk 资源提取」、[HoshinoUnreal/Phigros-Assets](https://github.com/HoshinoUnreal/Phigros-Assets)）。⚠️ 这是**从官方包体提取受版权保护的商业音频**，与社区谱面的授权问题**叠加**，风险更高。
4. **自采音频**：完全规避版权，但失去「音频↔谱面」的真实配对，只能做合成/迁移实验。

### 4.4 `info.yml` 关键字段（A 级 + 实测）

A 级规范：[Phira 文档·谱面信息（ChartInfo）](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html)（逐字段表见 [phigros-format.md §11](phigros-format.md)）。**实测** `chart/1000` 的 `info.yml` 原文关键行：

```yaml
name: アイドル
difficulty: 15.4
level: AT Lv.15
charter: K Production
composer: YOASOBI
illustrator: 【推しの子】製作委員会
chart: AT15.json        # ← 谱面文件名（务必用它定位谱面文件）
format: null            # ← 实测全库样本中恒为 null → 必须内容嗅探
music: idol-48k-0.8.ogg # ← 音频文件名
illustration: illustration.jpg
previewStart: 56.0
previewEnd: 68.3
aspectRatio: 1.7777778
offset: 0.0
lineLength: 6.0
tags: [art, ornament, fun, gimmicky, regular]
```

⚠️ **`chart` 字段名与文件内容不可互推**：**实测** `chart/7039` 的 `info.yml.chart` 是 `24432296.json`，但该文件**内容是 PEC 文本**（首行 `175`，随后 `bp 0.000 270.000` / `cp 0 0.000 1024.00 700.00` / `cm …`）。这正是 A 级文档「**忽略谱面文件的后缀名**」的实际后果。见 §5.3。

---

## 5. 格式分布

### 5.1 Phira 支持的格式（A 级）

官方文档原文：「目前支持的谱面文件格式包括：**RPE 格式**… **PEC 格式**… **PBC 格式**（文档待完善）。格式的推断通过 `info.yml` 中的 `format` 字段进行，若为空则通过文件内容进行推断。**注意：忽略谱面文件的后缀名**」
—— [Phira 文档·谱面文件格式](https://teamflos.github.io/phira-docs/chart-standard/chart-format/index.html)

官方 Rust 客户端的解析器目录**恰好三分**，与文档吻合（A 级源码）：
`prpr/src/parse/rpe.rs`（36 KB）、`prpr/src/parse/pgr.rs`（9.8 KB，官谱 JSON）、`prpr/src/parse/pec.rs`（14 KB），外加 `extra.rs`
—— <https://github.com/TeamFlos/phira/tree/main/prpr/src/parse>

### 5.2 实测占比（抽样，非全库）

> **抽样口径**：从 `GET /chart` 的元数据列表按固定随机种子抽 200 张，用 zip 中央目录 + HTTP Range 只抓谱面条目的**前 24 KB 压缩字节**并部分解压，按内容嗅探判型。**不下载完整包**（见 §9.2）。脚本与原始 JSONL 在调研过程中留存于临时目录，未入库。

| 轮次 | 抽样范围 | n（成功） | RPE | PEC | PBC | 网络失败 |
| :-- | :-- | --: | --: | --: | --: | --: |
| 第一轮（近期偏置） | 列表前 ~400 条 | 193 | **190 (98.4%)** | 3 (1.6%) | 0 | 7 |
| 第二轮（深页 / 最旧端） | 页码 210–325 | 90 | **85 (94.4%)** | 4 (4.4%) | 0（另有 1 张文本型未归类） | 0 |
| **两轮合计** | — | **283** | **275 (97.2%)** | **7 (2.5%)** | **0** | 7 |
| 全量解包抽样 | 随机 24 张 | 24 | 23 | 1（`.json` 后缀但内容是 PEC） | 0 | 0 |

**抽样偏差实测检验（重要）**：两轮的 `created` 年份分布截然不同——第一轮 188/193 是 2026 年上传，第二轮 **90/90 全部是 2024 年**。这证实了 §3.4 的担忧：默认排序确实「近期优先」，且 **PEC 占比在旧谱面上明显更高（4.4% vs 1.6%）**。因此全库 PEC 占比应更接近（或略高于）两轮合计的 2.5%，而不是第一轮的 1.6%。**全库精确值仍需枚举 322 页才能确定。**

**判据**（本轮脚本使用的启发式，**实测有效但非权威**）：解压前缀中出现 `"eventLayers"` → RPE；出现 `"notesAbove"`/`"notesBelow"`/`"formatVersion"` → 官谱 JSON；前缀为文本且能匹配 PEC 的行结构 → PEC。190 张 RPE 的根键顺序**实测**高度一致：`BPMList, bpm, startTime, META, RPEVersion, background, …`。

3 张被判定为 PEC 的样本（**实测**）：id 73890（`drd_IN.pec`）、id 16069（`AutoSave_Goats' Gifts.pec`）、id 73581（`Inevitability.pec`）——**后缀即 `.pec`**，且包内音频齐全。

**官谱 JSON 与 PBC 在抽样中占 0**。这与常识一致（官谱不公开、PBC 是新格式），但**不能因此断言全库为 0**——样本仅覆盖 200/9649。

> **【推断】** 社区谱面的格式构成大致为 **RPE ≫ PEC ≫ 官谱 JSON/PBC**，RPE 占比在 95%+ 量级。依据：两轮抽样一致指向 98%+；且 A 级 `info.yml` 文档已注明「**RPE 生成的谱面通常为 chart.json，PE 生成的谱面通常为 xxx.pec**」，PEC 属旧工具链产物。
> **但要强调：全库精确占比未查证。** 要拿到精确数字需要枚举 9649 条并把每张谱面文件完整下载解析（每个谱面文件解压后中位 3.7 MB，累计数十 GB）。

### 5.3 解析侧必须注意的混用陷阱（含实测证据）

| # | 陷阱 | 证据 |
| :-- | :-- | :-- |
| 1 | **文件后缀完全不可信**。`.json` 里可能是 PEC 文本，`.pec` 里可能是 JSON | **实测** id 7039：`24432296.json` 内容为 PEC。A 级文档明写「忽略谱面文件的后缀名」 |
| 2 | **`info.yml.format` 实测恒为 `null`**（全量解包的 24 张样本中 24/24；另有 196/196 张确认含 `info.yml`，但未逐张读取其 `format` 值），**不能依赖它**分派解析器，必须按内容嗅探 | 实测 + A 级文档（「若为空则通过文件内容进行推断」） |
| 3 | **note `type` 数字含义在 RPE 与官谱之间不同**：RPE 为 1=Tap/2=Hold/3=Flick/4=Drag；官谱 JSON 为 2=Drag/3=Hold/4=Flick（**该官谱映射为 C 级来源**，见 [phigros-format.md §5.2](phigros-format.md)） | [phigros-format.md §5.2](phigros-format.md)、[RFC-0029 §6](../decisions/RFC-0029-phigros-continuous-chart-generation.md) |
| 4 | **`above` 取值域实测为 `{0,1,2}`**，语义是「`==1` 为正面，其余为背面」 | **实测**：24 张样本合计 1→25698 / 2→456 / 0→168；10 张样本合计 1→12299 / 2→327 / 0→48。两轮都同时出现 0 和 2 → **Q3「是否只允许 {0,1,2}」得到实测支持**（0 与 2 都在真实数据中出现，都按背面处理） |
| 5 | **`eventLayers` 长度实测为 1–5** | **实测**：23 张样本的层数直方图 `{1:506, 2:190, 3:9, 4:92, 5:60}` → 出现 5 层，支持 A/B 级「最大五个」的说法；**Q18 的「4 普通 + 1 extended」与「5 个普通层」之争仍未由实测裁定**（`extended` 是独立字段，**实测** 23/23 张都有非空 `extended`） |
| 6 | **事件跨层求和**，不是取最上层 | A 级 [phigros-format.md §4.3](phigros-format.md) |
| 7 | 同一首歌可能有多张谱面（同一 `name`）；**按曲名聚合时务必去重**，否则同曲过采样且 (audio, chart) 对会重复 | **实测**：`info.yml` 无全局唯一歌曲 ID 字段（A 级字段表），只能靠 `name`+`composer` 近似 |

---

## 6. 许可与可用性

> ⚠️ 本节**只陈述公开可查的声明原文**，不构成任何法律意见，也不对训练用途的合法性作判断。

### 6.1 Phira 使用条款（**A 级：官方前端仓库源码**）

来源：`TeamFlos/phira-web/src/TermsOfUse.vue`（中英双语）<https://github.com/TeamFlos/phira-web/blob/main/src/TermsOfUse.vue>。关键原文：

- **服务描述**：「Phira 同时提供了托管用户上传内容的方式……**对于用户上传内容，Phira 不保证用户拥有分发已上传内容的权利，并不对由此造成的后果承担责任。**」（英文原文：*We assume no responsibility over whether users have the rights to distribute uploaded content.*）
- **用户上传内容与内容删除**：上传者「确认、声明或保证：您拥有或具备必要的许可、权利、同意和权限……」，且「**不会提交受版权保护……的材料，除非您是这些权利的所有者，或者已获得其合法所有者的许可**」。
- **内容的专有权利**：「用户**只有在获得明确授权的情况下才能使用此内容**。用户不得未经法律授权，对内容进行任何可能侵害版权方利益的行为，**包括从这些内容中创建衍生作品**。Phira 保留追责的权利。」
- **适用法律**：中华人民共和国法律。

**这意味着**：平台条款把权利保证的举证责任放在上传者身上，同时明确禁止未获授权地使用站内内容（含「创建衍生作品」）。**社区谱面的授权状态是逐张不确定的**，平台层面**没有**提供「可用于训练」的授权声明。

### 6.2 Phira DMCA / 侵权处理（**A 级**）

来源：`TeamFlos/phira-web/src/DMCA.vue` <https://github.com/TeamFlos/phira-web/blob/main/src/DMCA.vue>。要点原文：

- 「Phira is a community-driven rhythm game, **devoid of commercial intent**. We do not generate any profit either directly or indirectly. We operate solely on user donations.」
- 「Although we **demand users to obtain legal license of used assets before uploading levels, it may not be observed in some cases.**」← **平台自己承认存在未获授权的上传**
- 提供 DMCA 通知流程与指定版权代理（含线下地址）；「Phira will take whatever action, in its sole discretion, it deems appropriate, including removal of the challenged content from the Site.」

### 6.3 谱师自述授权（**实测**：来自 `info.yml.intro`）

`chart/1000` 的 `intro` 字段原文（**实测**）：

> 「自制谱与Phigros官方或Phira官方无关，歌曲以及图片仅为个人使用 配置由@kkkkkkkywy 完成 特效由@Kevin_2106 完成…… **未经允许不得自行录制Autoplay并发布至任何平台** 如要发布手元，在b站同时@我和kkkkkkkywy的情况下，不需要经过同意」

→ 这是**谱师在简介里自行声明的使用条件**，且显然不是数据/训练授权；不同谱师的 `intro` 内容各异（`tip` 字段同理）。**实测 24 张样本中未见任何一张声明「可用于机器学习/训练」。**

### 6.4 仓库许可证 vs 内容许可证（**实测**）

- `TeamFlos/phira` 是 **GPL-3.0**、`TeamFlos/phira-web` 是 **MIT**、`TeamFlos/phira-docs` 是 **CC-BY-4.0**（GitHub API 实测）。
- ⚠️ **这些许可证只覆盖代码/文档本身，不覆盖用户上传的谱面内容。** 不要把「Phira 客户端是 GPL-3.0」误解为「Phira 上的谱面可用于训练」。
- 两个社区谱面仓库（`yuameshi/phigros-charts-repo`、`yuameshi/PhiCommunity-Charts-Repo`）**均未声明许可证**（GitHub API 实测 `license=null`）→ 默认「保留所有权利」。

### 6.5 训练用途的常见约定与风险（**只陈述可查到的公开声明**）

本次调研**没有**在 Phira 官方文档、条款或 DMCA 页面中找到任何关于「机器学习训练」的明确许可或禁止条款——**这一点本身就是结论：未查证到明确授权**。

可查到的相关事实：
- Phira 条款禁止「未获明确授权使用站内内容」与「创建衍生作品」（§6.1）。
- Phira 自认非商业、靠捐赠运营，并要求上传者有合法许可，但承认「may not be observed in some cases」（§6.2）。
- 谱面里通常**捆绑分发音频与曲绘**，二者版权属于**曲师/画师/厂牌**，与谱面作者的授权是**两层**（§4.3）。
- 社区惯例中常见的做法（**这些是观察到的社区实践模式，不是授权依据**）：仅用于个人研究/非商业、不重新分发原始音频、公开成果时标注来源与谱师。

**风险提示（非法律意见）**：把 Phira 全量谱面（含音频）下载到本地并用于训练，会同时触及「谱面作者授权不明」与「音频/曲绘第三方版权」两层不确定性；平台条款中的「禁止创建衍生作品」措辞是否覆盖模型训练，**需决策者自行判断或寻求专业意见**，本文不作结论。

---

## 7. 多判定线统计（**RFC-0029 最关键的一节**）

### 7.1 有多少条判定线？（**实测，n=23 张 RPE 全量解析**）

`judgeLineList` 长度分布：

| 统计量 | 值 |
| :-- | --: |
| n | 23 |
| min | **2** |
| p25 | 24 |
| **中位** | **30** |
| p75 | 52 |
| max | **76** |
| 均值 | 37.3 |

10 张补充样本的 `n_lines`（**实测**）：9、18、24、24、25、30、30、33、56、82。

另一张实测：`chart/1000`（アイドル AT Lv.15）**71 条判定线**、1659 个 note。

**结论**：社区 RPE 谱面的判定线数量**中位数约 25–30 条，长尾到 80+ 条**。**RFC-0029 的「v1 引入多判定线」在数据侧完全成立**，且不是「两三条线」量级——单线假设会丢掉 95%+ 的结构。

> **推断**：判定线数量与谱面难度正相关（AT 17 的样本常见 60–80 条，HD/IN 低难常见 9–25 条）。依据：样本中 `AT Lv.17`/`AT Lv.18` 的 `n_lines` 为 71/76/82，而 `HD Lv.5` 为 9。**n 小，相关性强弱未查证。**

### 7.2 note 如何在多条线之间分配？（**实测，这是最反直觉的一条**）

**判定线多 ≠ note 均匀分布在多条线上。** 实测的两组数据都指向「**少数线承载绝大多数 note**」：

24 张全量样本（**实测**）：

| 指标 | min | 中位 | 均值 | max |
| :-- | --: | --: | --: | --: |
| 至少承载 1 个 note 的线数 / 总线数 | 0.17 | **0.58** | 0.58 | 1.00 |
| 最忙的一条线承载的 note 占比 | 0.34 | **0.73** | 0.74 | 1.00 |

即：**中位情况下只有 ~58% 的判定线真的带 note，而最忙的一条线独占 ~73% 的 note。**

10 张逐线 note 数（按条数降序取前 6，**实测**）：

| chart id | 线数 | 总 note | 各线 note 数（降序前 6） |
| :-- | --: | --: | :-- |
| 43121 | 33 | 1449 | 981, 110, 109, 58, 26, 26 |
| 51200 | 56 | 988 | 231, 184, 102, 74, 62, 60 |
| 60518 | 18 | 1011 | 698, 104, 39, 24, 24, 20 |
| 43191 | 24 | 1036 | 687, 161, 66, 44, 19, 18 |
| 48952 | 30 | 1367 | 567, 273, 62, 59, 58, 48 |
| 60451 | 24 | 1351 | 727, 279, 149, 144, 31, 11 |
| 68257 | 30 | 1600 | 912, 134, 133, 100, 74, 74 |
| 60360 | 25 | 1578 | **1578**, 0, 0, 0, 0, 0 |
| 25137 | 82 | 2124 | 1457, 146, 101, 99, 94, 63 |
| 43135 | 9 | 170 | 158, 8, 1, 1, 1, 1 |

**两种截然不同的形态都真实存在**：
- **「一条主线 + 大量装饰/演出线」**（60360：25 条线，全部 1578 个 note 在 1 条线上；43135：9 条线，158/170 在 1 条线上）。
- **「真·多线分摊」**（51200：56 条线，最忙的只有 231/988 = 23%，前 6 条线合计 ~70%）。

→ **对建模的直接含义（推断，供决策参考）**：note 的 `line_id` 分布是**高度长尾且双峰**的，不是均匀分类问题。直接做「`line_id` 的 softmax 分类」会在「一条线吃掉全部」的样本上梯度极小、在「真多线」样本上极难收敛。更稳的做法可能是：**先预测「主判定线」（承载绝大多数 note 的那条）+ 一个稀疏的次级线集合**，或对 `line_id` 施加**频率先验/长尾平滑**（如按线内 note 数分桶）。**该结论是推断，未做实验验证。**

### 7.3 note 标记的其它统计（**实测**）

**类型分布**（RPE `type`：1 Tap / 2 Hold / 3 Flick / 4 Drag）：

| 抽样 | n(note) | Tap | Drag | Hold | Flick |
| :-- | --: | --: | --: | --: | --: |
| 24 张全量 | 26 322 | **62.9%** | 20.2% | 11.0% | 6.0% |
| 10 张补充 | 12 674 | **52.0%** | 30.9% | 9.8% | 7.3% |

→ **Tap 约 52–63%，Drag 20–31%，Hold 10–11%，Flick 6–7%**。两轮抽样构成不同（10 张那轮难曲更多、Drag 更多），说明**类型比例对曲目/难度敏感**，不要用单点估计当全库常量。

**侧别 `above`**：`{1: 97.0–97.6%, 2: ~1.7–1.8%, 0: ~0.5–0.6%}` → **背面（above≠1）只占 2.4–3.0%**。
→ **对建模的直接影响**：`side` 是强不平衡的二分类。若直接预测 `side`，全预测「正面」即有 ~97.5% 准确率——**评估时必须报告背面类的 recall/F1，不能只看总准确率**。

**判定线嵌套**：23 张样本中 **6 张（26%）存在非 `-1` 的 `father` 引用**（**实测**）→ `rotateWithFather` / 嵌套判定线在真实数据里是普遍存在的，不是边角情况。解析器必须处理 `father` 字段（[phigros-format.md §12-Q4](phigros-format.md) 指出其字段名/默认值本身还有歧义）。

**`extended` 层**：23/23 张样本都有**非空 `extended`**（实测；实测 `chart/1000` 的 `extended` 只含 `inclineEvents`）→ 第 5 层事件不可忽略。

### 7.4 `positionX` 的实测取值范围（**对 RFC-0029 §8.2-Q8 的强证据**）

**实测**：10 张谱面、12674 个 note：

| 统计量 | 值 |
| :-- | --: |
| min | **−675.000** |
| p1 | −506.250 |
| p25 | −202.500 |
| **中位** | 0.000 |
| p75 | 225.000 |
| p99 | 540.000 |
| max | **+675.000** |
| **max \|positionX\|** | **675.000** |

**推断（依据充分，但仍需 A 级确认）**：RPE 的 `positionX` 合法域为 **\([-675, +675]\)**，即**判定线全长为 1350**，相对于判定线中心。依据：
1. 12674 个真实 note 的极值**恰好对称地命中 ±675.000**——若是无界自由浮点，两侧同时精确命中同一整数的概率极低；更像是**边界钳位值**。
2. p1 = −506.25、p75 = 225.0、p99 = 540.0 也全部是 0.125 的整数倍，符合「编辑器按网格放置、边界为 675」的图景。
3. 这与 [phigros-format.md §6.4](phigros-format.md) 中「**同刻度、1350 全长**」的【推断】**独立吻合**（该处推断来自 Phira `info.yml.lineLength` 默认 6.0 与官谱 X 单位的对照）。

**✅ 后续交叉验证（2026-09-26 同日，另一路调研）：本节的实测结论已被 A 级源码证据独立确认。** 姊妹文档 [phigros-units-and-geometry.md](phigros-units-and-geometry.md) 从 prpr 源码中取出常量 **`RPE_WIDTH = 1350.0`、`RPE_HEIGHT = 900.0`**（<https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs>，**A**），并给出归一化映射 `sx = (x + 675)/1350`（Phira 官方 Python 参考实现，**A**）。
→ 两条独立证据链（本文的**数据实测** ±675 + 该文的**源码常量** 1350）**完全吻合**：`positionX` 的单位是 **RPE 舞台系 x 坐标（1 单位 = 舞台宽度 1/1350）**，语义可见范围 **[−675, 675]**。
⚠️ 但 **RFC-0029 §8.2-Q8 的「桶宽」仍需决策者裁定**——单位已定，桶宽是建模选择；本文 §7.4.1 给出的量化粒度实测（手工谱常见 22.5 = 1350/60 网格）是该选择的直接输入。

> ⚠️ **诚实标注**：本文 §7.4 的实测在编制时**尚不知道** prpr 有 `RPE_WIDTH` 常量（该常量由同日另一路调研发现）。保留原始推断链与「确证前不得写死桶宽」的告诫，是为了保留证据独立性——**结论一致不等于推导过程可以省略**。

#### 7.4.1 `positionX` 的量化粒度（实测，n=3 张谱）

对 3 张谱面统计**去重后的 `positionX` 取值个数与最小间隔**：

| chart id | note 数 | 去重取值数 | min | max | 最小的几个相邻间隔 |
| :-- | --: | --: | --: | --: | :-- |
| 43121 | 1449 | **24** | −675.0 | 675.0 | 22.5, 45.0, 67.5, 135.0 |
| 43191 | 1036 | 28 | −506.25 | 506.25 | 11.25, 16.875, 22.5, 28.125, 33.75 |
| 51200 | 988 | 45 | −472.5 | 472.5 | **0.324639, 0.496754**, 3.75, 6.0, 6.75 |

**三条实测事实**：
1. **手工谱的 `positionX` 落在很粗的网格上**：43121 全谱只用 **24 个不同取值**装下 1449 个 note，间隔全是 **22.5 = 1350/60** 的整数倍；43191 的最小间隔 **11.25 = 1350/120**（其倍数 5.625 = 1350/240）。（**实测**）
2. **但不是全局固定网格**：51200 出现了 0.324639 / 0.496754 这类**任意浮点**取值（分母 1802、3889）→ 该谱的位置不是画在网格上，**推断**为自动转换/程序计算产物。（**实测** + 【推断】）
3. 因此 **不能用「`positionX` 是 × 的整数倍」当校验规则**；只能做区间断言（\(|positionX| \le 675\)）。

**对 RFC-0029 §8.2-Q8「桶宽」的量化参考（推断）**：若在 \([-675, 675]\) 上开 128 桶，桶宽 ≈ **10.55**（1350/128）——**比手工谱常见的 22.5 网格更细**，即 128 桶不会在手工谱上产生系统性量化冲突（但会在 51200 那类任意浮点谱上产生连续化误差）。**这是数值推断，不是规范结论；Q8 仍需决策者裁定。**

### 7.5 有没有现成的多判定线统计？

**没有。** 本次检索未找到任何公开的 Phira/Phigros 谱面「判定线数量分布」或「note 跨线分配」统计报告（**未查证到，非不存在**）。上表全部是本次自己跑出来的。

### 7.6 怎么用脚本统计（可直接复用的思路）

**思路 A：不下载谱面文件，先看结构（最快，本调研用的就是这个）**

1. 翻 `GET /chart?page=P&pageNum=30`（`pageNum` 上限 30），P 从 1 到 322，收集全部 9649 条的 `id` / `file` / `difficulty` / `tags` / `created`。
2. 对每个 `file` 发 **`HEAD`** 或 `Range: bytes=0-0` 取 `Content-Range` 里的总长度（**实测可用**）。
3. 拉取 zip **尾部** 最多 ~200 KB，定位 EOCD（`PK\x05\x06`），读中央目录 → 得到每个**条目的名字、压缩大小、解压大小、本地头偏移**。
4. 用 `info.yml` 条目做两件事：读 `chart` 字段定位谱面文件；统计包内是否含音频扩展名。
5. 若要判格式：对谱面条目做一次 **`Range` 取前 24 KB 压缩字节**，用 `zlib.decompressobj(-15)` **部分解压**（截断的 deflate 流可解出前缀），在前缀里嗅探 `eventLayers` / `notesAbove` / `formatVersion` / PEC 行结构。
6. 批量时用 **线程池（8 并发足够）**。本调研实测：单线程 ~30 s/张 → 8 线程 ~1.8 s/张。**务必自行限速**，不要打满对方服务。

**思路 B：统计判定线数量分布（必须解析谱面文件，但不必装音频）**

1. 同上拿到谱面文件名与展平大小。
2. **只 `Range` 下载谱面文件那一个条目**（用中央目录的 `local header offset` + 读本地头 30 字节拿 `name_len`/`extra_len` 定位数据段），**跳过音频与曲绘**——这一步能把每张谱的成本从 ~8 MB 降到 ~1–4 MB。
3. `json.loads` → `len(root["judgeLineList"])` 得线数。
4. 逐线 `len(line["notes"])` 得**每线 note 数向量**；衍生统计：
   - `lines_with_notes = sum(1 for c in counts if c > 0)`
   - `busiest_share = max(counts) / sum(counts)`
   - **基尼系数 / 熵**：`H = -Σ p_i log p_i`，`p_i = c_i / Σc`——比 `max/sum` 更能刻画「长尾 vs 均匀」。建议同时报 `H / log(n_lines)`（归一化熵）。
   - **note 的跨线时间重叠**：对每个 note 求绝对时间（RPE beat 三元组 → 秒，见 [phigros-format.md §7.1](phigros-format.md)），检查**同一时刻不同线上的 note** 是否并发——这直接决定「同刻按键上限」合法性规则。
5. 别忘了 `type` / `above` / `isFake` / `speed` / `endTime` 的分布，以及 `father` 引用率、`eventLayers` 长度直方图。

**思路 C：音频侧**

- 用 `info.yml.music` 定位音频条目，只下载该条目（同样用中央目录 + Range），得到 **sha1 哈希** → 按哈希去重，可得「**唯一曲目数**」与「同曲多谱」的重复率。**这是本次未查证但很值得补的一个数字。**

---

## 8. 解析工具生态

> 「能当解析器用」优先。所有仓库元数据（语言/许可证/推送时间）来自 GitHub API（**实测**）。

| 项目 | 语言 | 许可证 | 成熟度 | 支持格式 | 能否直接产出我们需要的字段 | 评级 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **`TeamFlos/phira` 内的 `prpr`** <br><https://github.com/TeamFlos/phira/tree/main/prpr> | **Rust** | **GPL-3.0** | ★★★★ 官方客户端实际使用的解析器（2676★，2026-09 仍活跃） | **RPE**（`parse/rpe.rs` 36 KB）、**官谱 JSON**（`parse/pgr.rs`）、**PEC**（`parse/pec.rs`）**三者齐全** | ✅ 是「事实标准」。解析后是内部 `JudgeLine`/`Note` 模型，含时间、位置、类型、事件轨、`father`。⚠️ 但它是**渲染器导向**的：会把 RPE 语义**规约**成统一的内部模型（如缓动映射表 `RPE_TWEEN_MAP`、`process_lines` 做同刻 note 标记），**保真度取舍需自行核对** | **首选（但要读源码）** |
| **`Ivan-1F/phichain`** <br><https://github.com/Ivan-1F/phichain> | **Rust** | LGPL-3.0 | ★★★☆ 活跃（47★，2026-09-09 推送）；「Phigros charting toolchain」 | workspace 内含 **`phichain-format`**，其 `src/` 下**只有 `rpe/` 与 `official/` 两个格式模块**（**实测**，无 PEC） | ✅ 拆成了独立 crate（`phichain-chart` 数据模型 / `phichain-format` 读写 / `phichain-converter` 转换 / `phichain-compiler`），**比 prpr 更适合当库用**（读写双向、不只读） | **推荐（库形态最好）** |
| **`Mivik/prpr`**（独立仓） <https://github.com/Mivik/prpr> | Rust | GPL-3.0 | ★★★ 117★，但 **2023-04 后停更** | 同上（是 Phira 内置版的上游） | ⚠️ 已过时，**用 Phira 内的 vendored 版**而不是这个 | 参考 |
| **`ChenWuwei404/py-phi-editor`** <br><https://github.com/ChenWuwei404/py-phi-editor> | **Python** | GPL-3.0 | ★☆ 5★，最后推送 2025-05；是个 **pygame 编辑器 GUI**（`editor.py`/`main.py`/`widgets/`），不是库 | **未逐一核实**其解析器覆盖范围 | ⚠️ **不是库**，是把 GUI 程序拆出解析逻辑。好处是 Python 同栈，可读代码抄逻辑；坏处是无 API、无测试、维护弱 | 参考（抄逻辑） |
| **`PhiZone/player`** <br><https://github.com/PhiZone/player> | **TypeScript** | MPL-2.0 | ★★★★ 91★，2026-09-26 活跃 | 官谱/RPE（HTML5 播放器） | 可读，但 TS + 浏览器渲染导向，不适合当离线批处理解析器 | 参考 |
| **`zwtwz/PhigrosChartTransformer`** | Python | 未声明 | ★★ 15★，2025-11 | 官谱 JSON → Phira `.pez` | 可参考**官谱→RPE 的字段映射**（正是我们最怕搞错的那部分） | 参考 |
| Phira 客户端本体（GUI） | Rust | GPL-3.0 | — | 同上 | ❌ GUI 程序，不能当批处理解析器；但**可用作黄金对照**（把我们的解析结果与 Phira 实际渲染比对） | 对照 |

### 8.1 选型建议（推断）

- **主路径：`phichain-format`（Rust, LGPL-3.0）当参考实现读源码，`prpr/src/parse/rpe.rs` 当语义权威。**
- BeatMorph 是 Python 栈（CLAUDE.md §4），**不建议**为了解析引入 Rust 依赖链。更现实的方案：**用 Python 自己写 RPEJSON 解析器，但以 `rpe.rs` + `phichain-format/rpe/` 为逐字段对照，并用 prpr/Phira 的实际行为做验收**。
- ⚠️ **许可证注意（事实陈述，非法律意见）**：`prpr`/`phira` 是 **GPL-3.0**，`phichain` 是 **LGPL-3.0**。**逐行翻译/移植源码**可能构成衍生作品；**只读规范文档 + 独立实现**更干净。若决定移植，需先开 RFC 讨论许可证影响（AGENTS.md §1「不擅自越权」）。

---

## 9. 对 BeatMorph 数据流水线的建议

> 前提：RFC-0029 已锁定 **RPEJSON** 主路径、v1 引入多判定线。下面按「现在就要开工」给一条可执行链路。

### 9.1 总体链路

```
[0] 元数据枚举 (API, 322 页)      → charts_meta.parquet  (9649 行)
        │
[1] 结构预筛 (Range + 中央目录)   → 每张谱：条目表 / 大小 / 是否有音频
        │
[2] 格式嗅探 (Range + 部分解压)   → format ∈ {RPE, PEC, OFFICIAL, PBC, ?}
        │
[3] 选择性下载 (只取谱面文件条目) → charts_raw/<id>/<chartfile>
        │
[4] 解析 (RPEJSON → 契约类型)     → ChartRecord (notes + line event tracks)
        │
[5] 质检 (schema/单位/合法性)     → 通过/隔离，写 qc_report
        │
[6] 音频下载 (按 info.yml.music)  → audio/<sha1>.ogg   （按哈希去重）
        │
[7] 特征提取 (MERT, 75Hz)         → features/<sha1>.npz + meta
        │
[8] 配对与切分                    → (audio, chart) 对 + train/val/test 划分
```

### 9.2 每一步的要点与坑

**[0] 元数据枚举**
- `pageNum` **上限 30**（实测 31 → HTTP 400）；`page` 到 322。
- 响应键是 `results`（**不是** C 级文档写的 `result`）。
- 每次请求之间**加 sleep**（未观测到限流，但不代表没有）。322 次请求，建议 0.3–1 s 间隔。
- 落盘：`id, name, level, difficulty, charter, composer, tags, created, updated, chartUpdated, file, preview, illustration`。
- **区分「上架/ranked」与「全部」**：`stable=true` 只有 627 张，`type=2`（unstable）有 9022 张。**训练用哪一档必须显式决策**——建议全用（`type=3`）但把 `stable/ranked` 作为**元数据特征**保留，便于做「只在高质子集上训练」的消融。

**[1]–[2] 预筛（**强烈建议先做，能省 90% 带宽**）**
- **实测** CDN 支持 Range（`206`）。脚本思路见 §7.6 思路 A。
- 预筛产出：`n_entries`、条目名列表、`info.yml.chart`、`info.yml.music`、谱面条目解压大小、`format`。
- 用预筛结果**先建直方图**（格式、大小、是否有音频）→ 再决定下载策略。**不要一上来就全量拖 76 GB。**

**[3] 选择性下载**
- ⚠️ **必须按 `info.yml.chart` 定位谱面文件，不要「取最大的 .json」**——实测有一张谱面包里 `.json` 解压总量 294 MB（特效资源），谱面文件只有 3.25 MB。
- ⚠️ **必须重试**：本次 200 张扫描有 7 张（3.5%）出现 `TimeoutError`/`URLError`/`ConnectionResetError`。带指数退避重试 + 断点续传（Range 续传）。
- ⚠️ **落盘时不要用原始文件名**（实测形如 `1817439042209534.json`、`＃53682.json`——含全角字符）。用 `<chart_id>/<chartfile>` 或直接规范化。

**[4] 解析**
- **不要信后缀名，也不要信 `info.yml.format`**（实测全 `null`）。按内容嗅探：第一非空字节是 `{` → JSON 类；否则文本 → 尝试 PEC。
- JSON 类内部再分：含 `eventLayers` → RPE；含 `notesAbove`/`notesBelow`/`formatVersion` → 官谱 JSON。（PBC 未查证其结构，建议先直接拒收并记账。）
- ⚠️ **`type` 映射按格式分派**：RPE 1/2/3/4 = Tap/Hold/Flick/Drag；官谱 2/3/4 = Drag/Hold/Flick（**该官谱映射仅 C 级**）。**这一条错了会「静默全员错位」**——RFC-0029 §6 已点名。
- 时间：RPE 是 `[beat, num, den]` 三元组 + `BPMList`；**必须先写出并单测 beat→秒 的换算**（[phigros-format.md §7.1](phigros-format.md)，注意 Q7 的 `bpmfactor` 乘除歧义）。
- `above`：**实现必须写 `above == 1 ? FRONT : BACK`**，不得当布尔解析（实测 0 与 2 都出现）。
- `eventLayers`：**跨层求和**；且事件间隙要**补洞**（[phigros-format.md §8.4](phigros-format.md) 三类静默陷阱）。
- `father` 嵌套：实测 26% 的谱面有，**必须处理**，不能跳过。

**[5] 质检（建议先定这几条硬规则）**
- schema 级：`judgeLineList` 非空；每条线有 `notes` 数组；`type ∈ {1,2,3,4}`；`Hold(type=2)` 的 `endTime ≥ startTime`。
- 单位级（RFC-0029 §7 硬约束）：所有时间在**秒域**计算并断言；`positionX` 记录 min/max，**越出 [−675, 675] 的样本单独标记**（这正是发现「我方解析单位错」的哨兵）。
- 分布级：每张谱落一行统计（线数、note 数、type/above 分布、每线 note 数熵、时间跨度、BPM 区间），与 §7 的实测基线比对，**离群样本进隔离区而非直接进训练集**。
- **可玩性不在此阶段判定**（生成侧才需要），但「note 数 / 时长」比、同刻并发数是很好的质量信号。

**[6] 音频**
- 音频**随包分发且 100% 存在**（实测 196/196），所以「音频缺失」不是主要问题；**主要问题是版权**（§6）。
- **按 sha1 去重**：同一首歌会有多张谱面，重复下载音频是纯浪费。
- 建议把音频与谱面**分开存放**（`audio/<sha1>` ↔ `chart → audio_sha1` 外键），这样「只训练谱面结构、不碰音频」与「音频+谱面联合训练」两种模式可以自由切换，也便于在授权不明时**只保留谱面、不落音频**。
- ⚠️ **音频文件名无语义**，只能读 `info.yml.music`。

**[7] 特征提取**
- MERT-v1-330M 帧率 **75 Hz**，由 config 派生并断言（CLAUDE.md 红线 7 / RFC-0029 §7）。
- 缓存必须带 `{rate, sample_rate, layer, model_rev, duration_s}` 元数据并在加载时校验。
- 音频采样率实测多样（`.mp3`/`.ogg`/`.wav`），**统一重采样到模型要求的采样率**，并在缓存元数据里记录**原始**采样率与时长。

**[8] 配对与切分**
- ⚠️ **切分必须按「曲目」而不是按「谱面」**：同一首歌的多张谱面进同一个 split，否则 val 会泄漏。
- 建议同时保留一个「**同曲跨谱泛化**」的评测集（train 用某曲的 IN 谱、test 用同曲的 AT 谱）——这比随机切分更能反映模型是否学到了音乐↔谱面的真关系。
- **本次调研未查证同曲重复率**，建议在 [0]/[6] 阶段顺手统计（见 §7.6 思路 C）。

### 9.3 最该先做的三件事（按 ROI 排序）

1. **跑一次全库预筛**（§9.2 [0]-[2]）：322 页元数据 + 9649 次「尾部 200 KB + 谱面前缀 24 KB」的 Range 抓取。产出全库的**格式分布、大小分布、音频存在率、判定线数量分布**——这正是本文最大的空白（§3.4 / §5.2 / §7）。
2. **写 RPEJSON 解析器 + beat→秒换算，并用 `chart/1000` 与 `chart/7039` 两个实测样本做夹具**：前者是标准 RPE（71 线 / 1659 note / 4 类 type / `above` 含 2），后者是**伪装成 `.json` 的 PEC**（完美的负样本夹具）。
   - ⚠️ 按 CLAUDE.md §3.5，夹具只能是**微缩**的；把这两张谱裁成极小样本再入库，别把 6.5 MB 的 JSON 提交上去。
3. **把 §7 的实测基线写成契约测试**：`above ∈ {0,1,2}`、`type ∈ {1,2,3,4}`、`|positionX| ≤ 675`、`1 ≤ len(eventLayers) ≤ 5`。这几条正是「能挡住静默错位」的断言（CLAUDE.md 红线 7：物理常量必须派生 + 断言）。

---

## 10. 存疑清单

> 与 [phigros-format.md §12](phigros-format.md) 的 19 项互补；编号前缀 `Q-` 为本篇新增。

| # | 问题 | 目前证据状态 |
| :-- | :-- | :-- |
| **Q-1** | Phira 全库的**精确格式占比**（RPE / PEC / 官谱 JSON / PBC）？ | 仅有 200 张抽样的 **RPE 190 / PEC 3**（实测）。全库需枚举 9649 张并逐个嗅探；**未查证**。抽样偏向近期更新，PEC 占比可能被低估。 |
| **Q-2** | Phira 是否对 API 有**限流**？批量抓取的安全速率？ | 未观测到限流响应头，官方文档也未记载；**未查证**。批量脚本必须自行限速。 |
| **Q-3** | 全库**唯一曲目数**与同曲多谱的重复率？ | **完全未统计**。可通过音频 sha1 去重得到（§7.6 思路 C）。这个数字直接决定「真实 (audio, chart) 对数」远小于 9649 的哪一档。 |
| **Q-4** | `positionX` 单位与 `info.yml.lineLength`（默认 6.0）的换算？1350 与屏幕宽度的关系？ | **✅ 已由姊妹文档裁定（A 级）**：prpr 源码常量 `RPE_WIDTH = 1350.0`，`positionX` 单位 = RPE 舞台 x 坐标（1 单位 = 舞台宽 1/1350），范围 [−675, 675] —— 见 [phigros-units-and-geometry.md](phigros-units-and-geometry.md) §3。本文**独立**由数据实测得到同一区间（n=12674，§7.4），两条证据链吻合。**仍未查证**的是 `lineLength=6.0` 与 1350 的换算（该文另有讨论）。 |
| **Q-6** | 若在 [−675, 675] 上开 128 桶，桶宽是否合适？ | **未裁定（属 RFC-0029 §8.2-Q8，建模选择）**。单位已定（Q-4）；数值上 128 桶 → 10.55 单位/桶，**细于**手工谱实测的 22.5 网格（§7.4.1）→ 【推断】不会在手工谱上引入系统性冲突；对任意浮点谱则等价于连续化。**桶宽选定属决策者职权。** |
| **Q-5** | `positionX` 的真实量化粒度（编辑器网格）？ | **部分查证（实测 n=3）**：手工谱落在 1350/60（间隔 22.5）或 1350/120（间隔 11.25）级别的**粗网格**上，存在只用 24 个不同取值承载 1449 个 note 的谱面；但也存在携带**任意浮点**（最小相邻间隔 0.32）的谱面。**不存在全库统一的固定网格** → 只能做区间校验，不能做「整除」校验。详见 §7.4.1。 |
| **Q-6** | 若在 [−675, 675] 上开 128 桶，桶宽是否合适？ | **未裁定（属 RFC-0029 §8.2-Q8）**。数值上 128 桶 → 10.55 单位/桶，**细于**手工谱实测的 22.5 网格（§7.4.1）→ 【推断】不会在手工谱上引入系统性冲突；对任意浮点谱则等价于连续化。**仍需 A 级单位定义后才能定案。** |
| **Q-7** | 官谱 JSON 的 note `type` 映射（2=Drag/3=Hold/4=Flick）？ | 仍只有 **C 级来源**（Lchzh Docs）；RPE 侧由 A+B 双重确认。若将来要混入官谱数据，**必须先验证**。 |
| **Q-8** | `eventLayers` 的 5 层到底是「5 普通」还是「4 普通 + 1 特殊」？ | 实测出现长度为 5 的 `eventLayers`（**实测** 23 张中 60 条线），且 `extended` 是**独立字段**、23/23 张非空 → 支持「**5 普通层 + `extended` 独立**」的读法，但 A/B 级文档互相矛盾（[phigros-format.md §12-Q18](phigros-format.md)）。**未最终裁定。** |
| **Q-9** | PBC 格式的结构？Phira 文档只写「文档待完善」 | **完全未查证**。抽样中 0 例，可暂时直接拒收。 |
| **Q-10** | PEC 格式在 Phira 的现存数量？ | 抽样 200 张中 3 张（**实测**）；**全库未查证**。本次另在 24 张全量样本中遇到 1 张「`.json` 后缀、PEC 内容」的样本（id 7039）——说明 PEC 无法靠后缀识别。 |
| **Q-11** | Phira 站内内容可否用于**机器学习训练**？ | **未查证到任何明确许可或明确禁止条款。** 条款有「禁止未获授权使用站内内容 / 创建衍生作品」的措辞（§6.1），DMCA 承认存在未授权上传（§6.2）。**需决策者判断**，本文不给结论。 |
| **Q-12** | 谱面捆绑的**音频/曲绘**的版权状态？ | 归曲师/画师/厂牌，与谱面作者授权是**两层**，且平台不保证上传者有权分发（§6.1）。**逐张不可判定，未查证。** |
| **Q-13** | PhiZone 等其它 Phigros 社区平台是否有可批量拉取的谱面 API？ | **未查证**。仅确认 [PhiZone/player](https://github.com/PhiZone/player) 存在且活跃。 |
| **Q-14** | 「Phigros 自制谱论坛」这一说法对应哪个具体站点？ | **未查证**。本次检索未找到可作批量来源的论坛；社区分发似以 QQ 群/B 站/Phira 站内为主。 |
| **Q-15** | `father`/`rotateWithFather` 的字段名与默认值？ | 实测 26% 的谱面存在非 `-1` 的 `father` 引用（证明该字段真实在用），但字段名三处写法不一（[phigros-format.md §12-Q4](phigros-format.md)）。**未查证。** |
| **Q-16** | 同一时刻跨多条线的 note 并发上限？ | **未统计**。这是 RFC-0029 要求的「跨线合法性」规则的输入（[RFC-0029 §5](../decisions/RFC-0029-phigros-continuous-chart-generation.md)）。统计方法见 §7.6 思路 B 第 4 步。 |

### 10.1 本次调研**没有**做到的事（诚实边界）

1. **没有全库统计**。判定线数量分布（§7.1，n=23）、每线 note 分配（§7.2，n=23+10）、type/above 比例（§7.3，n=24+10）、格式占比（§5.2，n=283）、`positionX` 分布（§7.4，n=10 张 / 12674 note；§7.4.1，n=3 张）**全部是抽样结果**。**没有任何一个是全库数字。**
2. **没有统计唯一曲目数与同曲重复率**（Q-3）——这直接影响「真实 (audio, chart) 对有多少」。
3. **没有验证 PBC 格式**（Q-9）、**没有查到 Phira 的限流策略**（Q-2）、**没有找到任何训练授权声明**（Q-11）。
4. **没有做法律判断**。§6 只复述公开声明。
5. 抽样脚本临时存放于本机 `%TEMP%`，**未入库**（遵守 CLAUDE.md §3.5 大文件不入库）；复现方法见 §7.6。

---

## 11. 来源清单

### A 级（官方文档 / 官方源码 / 官方 API）

| 标题 | URL | 本次是否直接访问 |
| :-- | :-- | :-- |
| Phira 谱面文件格式（RPE / PEC / PBC） | <https://teamflos.github.io/phira-docs/chart-standard/chart-format/index.html> | ✅ |
| Phira 谱面信息（ChartInfo / info.yml） | <https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html> | 转引 [phigros-format.md](phigros-format.md) |
| Phira 官方 API（谱面列表/详情/多取） | <https://api.phira.cn/chart>、`/chart/{id}`、`/chart/multi-get` | ✅ **实测** |
| Phira CDN 谱面包托管 | <https://phira.5wyxi.com/files/> | ✅ **实测**（含 Range 支持） |
| Phira Web 前端 OpenAPI 生成类型（端点与参数权威） | <https://github.com/TeamFlos/phira-web/blob/main/src/api/schema.d.ts> | ✅ |
| Phira 使用条款 | <https://github.com/TeamFlos/phira-web/blob/main/src/TermsOfUse.vue> | ✅ |
| Phira DMCA 政策 | <https://github.com/TeamFlos/phira-web/blob/main/src/DMCA.vue> | ✅ |
| Phira 客户端源码（`prpr` 解析器三分：rpe/pgr/pec） | <https://github.com/TeamFlos/phira>、<https://github.com/TeamFlos/phira/tree/main/prpr/src/parse> | ✅ |
| Phira 多人服务端库 | <https://github.com/TeamFlos/phira-mp> | ✅（元数据） |
| phichain 工具链（`phichain-format` 仅 rpe + official） | <https://github.com/Ivan-1F/phichain> | ✅（元数据 + 目录树） |
| PhiZone 播放器 | <https://github.com/PhiZone/player> | ✅（元数据） |

### C 级（二手 / 非官方）

| 标题 | URL | 备注 |
| :-- | :-- | :-- |
| Phira API 非官方文档（大松） | <https://www.xuziyao.com/posts/9/> | 端点清单与字段说明大体可用；**响应键 `result` 与实测 `results` 不符**，且未覆盖官方 schema 中的 `/chart/{id}/versions`、`/collection`、`type` 等 |
| Lchzh Docs·Phigros 谱面格式说明 | <https://docs.lchzh.top/learning/phigros/> | 官谱 note `type` 映射的唯一来源（Q-7） |
| GitHub `topic:phigros` 检索 | <https://github.com/topics/phigros> | 69 个仓库（**实测**） |

### B 级（社区 wiki）

沿用 [phigros-format.md §13](phigros-format.md) 的 `pgrfm.miraheze.org`（Phigros 自制谱 wiki）系列条目，本篇不重复列出。

### 实测数据（本 agent 于 2026-09-26 生成）

| 数据 | n | 说明 |
| :-- | --: | :-- |
| `GET /chart` 的 `count` 与筛选分项 | 12 次查询 | §3.1 |
| 全量解包统计（线数 / note / type / above / eventLayers / father / 包结构） | 24 张 | §4、§5、§7 |
| 格式扫描（Range + 部分解压嗅探） | 第一轮 200 张（193 成功）+ 第二轮 90 张 | §5.2 |
| 逐线 note 数与 `positionX` 分布 | 10 张 / 12674 note | §7.2、§7.4 |
| `positionX` 去重取值与量化粒度 | 3 张 / 3473 note | §7.4.1 |
| 格式扫描第二轮（最旧端，2024 年谱面） | 90 张 | §5.2 偏差检验 |
| 两个完整谱面包的目录清单 | 2 张（id 1000、7039） | §4.1、§5.3 |

> 原始脚本与 JSONL 输出留存于本机临时目录（未入库，遵守 CLAUDE.md §3.5）；如需复现，按 §7.6 的思路重跑即可。
