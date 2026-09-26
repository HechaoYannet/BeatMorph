> 状态：🟡 草案 ｜ 阶段：Phase 2（CLI）/ Phase 3（服务接口）｜ 负责：基础设施组
> 对应代码：`beatmorph/cli/`、`beatmorph/api/` ｜ 对应奠基章节：§2、§3.6、§7（Phase 3）

# Plan 08 — CLI / API / 端到端入口

## 1. 目标与范围

### 1.1 交付什么

1. **生成 CLI**（Phase 2）：

   ```
   beatmorph-generate --audio song.mp3 --difficulty 12 --lines <trajectory.json> -o out.json
   ```

   把「音频 + 难度 + 判定线事件轨」串成完整推理管线（Stage 0 → Stage 1 → 解码 → 后处理 → RPEJSON），并保证**只有零违规项时才写出**。
2. **训练入口的延续**（Phase 2）：`beatmorph-train` 已存在（实读 `pyproject.toml` 的 `[project.scripts]` 与 `beatmorph/cli/train.py`），本 plan 只负责与 Plan 07 的门禁开关（`--gates`）对齐。
3. **评估入口**（Phase 2，提案）：`beatmorph-eval`（消费 Plan 06 的 `EvalReport`）。
4. **服务接口**（Phase 3）：`beatmorph/api/` 的推理服务（现有 `create_app()` 为占位实现，实读确认）。

### 1.2 不交付什么

- 不实现任何模型/解码/评估逻辑（Plan 01/03/04/05/06）；
- **不做 Web 前端**（BasePlan §7-13「API / Demo」的 UI 部分另计）；
- 不追求推理 < 2 s/首：BasePlan §3.7 明确「先要质量，再要速度」；
- **不拷贝、不落盘音频到仓库或输出目录**（红线 5、BasePlan §4.5）；**不提供模型权重的发布/分发入口**（数据合规裁决的前提条件：最终不发布权重——见 §7 R-2、§9-7）；
- 不定义判定线事件轨的**数据契约**（见 §9-1：新增跨模块类型须先开 RFC，红线 2）。

## 2. 与奠基文档对应

| 本 plan 内容 | 奠基出处 | 关系 |
| --- | --- | --- |
| 输入层三件套（音频 / 难度 / 判定线事件轨） | BasePlan §2「输入层（极简）」、§1.1 | 直接落实 |
| 管线顺序 Stage0 → Stage1 → 解码 → 后处理 → 写 RPEJSON | BasePlan §2 架构图、§3.6 | 直接落实 |
| 判定线事件轨是**条件输入**（v1 不做联合生成） | RFC-0029 §2.2 决议 **Q2**、§3.7 | 约束 |
| 难度用**数值定数**而非 `level` 文本 | 数据集实测（survey §3.2：`level` 是自由文本，实测出现 `"AT  Lv.16"` / `"sweet"` / `"酔い"`；`difficulty` 为 f32 且有浮点误差） | 硬要求 |
| 100% 合法方可导出 | CLAUDE.md 红线 6、BasePlan §3.7 | 硬约束（退出码设计） |
| 音频不落盘/不分发；**不发布模型权重**（数据合规裁决的前提，2026-08-05） | CLAUDE.md 红线 5 附注、BasePlan §4.5、RFC-0029 §7-7 | 硬约束 |
| 一切单位派生 | CLAUDE.md 红线 7 | 硬约束 |
| 服务接口延后到 Phase 3 | BasePlan §7 Phase 3-13、§3.7 | 阶段约束 |
| 不做联合生成线事件轨 | RFC-0029 §3.7、§2.2 | 范围约束 |

**偏离声明**：无。

## 3. 接口契约

### 3.1 CLI 参数表

| 参数 | 必填 | 取值 | 语义 | 依据 |
| --- | --- | --- | --- | --- |
| `--audio` | 是 | 路径（WAV/MP3） | 输入音频（24000 Hz 由 preprocessor 派生，不由用户指定） | BasePlan §3.1 |
| `--difficulty` | 是 | f32 数值 | 难度定数；**比较/分档前 round 到 0.1** | survey §3.2 |
| `--lines` | 否 | 路径（JSON） | 判定线事件轨条件输入（schema 见 §9-1） | RFC-0029 Q2 |
| `-o` / `--out` | 是 | 路径 | 输出 RPEJSON 文件或谱面包目录 | 本 plan |
| `--checkpoint` | 条件 | 路径 | 模型权重；缺省时使用配置中的默认实验 | Plan 07 |
| `--decode` | 否 | `peaks` / `thinning` | 解码臂选择（B6 对照臂的入口） | RFC-0029 §5.2 B6 |
| `--seed` | 否 | int | 固定解码随机性（thinning 是随机算法） | Plan 05 §4.2-5 |
| `--device` | 否 | `cpu` / `cuda` | 推理设备 | — |
| `--validate-only` | 否 | flag | 只跑合法性校验，不写文件 | 红线 6 |
| `--dry-run` | 否 | flag | 跑通管线但不落盘（CI 用） | — |
| `--gates` | 否 | flag | 转发给训练入口（`beatmorph-train` 专用） | Plan 07 §3.2 |

**实现约束**：`pyproject.toml` 当前**不含 `click` / `typer`**（实读确认）→ 用标准库 `argparse`，**不引入新的 CLI 依赖**；`[project.scripts]` 中已注释的 `beatmorph-generate` 需要在 M8.1 启用（改 `pyproject.toml` 属 infra-agent 范围内）。

### 3.2 退出码表（**契约，测试须逐条覆盖**）

| 码 | 含义 | 触发例 |
| --- | --- | --- |
| 0 | 成功（且已写出合法谱面） | 正常完成 |
| 2 | 参数错误（argparse 约定） | 缺 `--audio`；`--difficulty` 非数值 |
| 3 | **合法性校验未通过 → 拒绝导出** | `LegalityReport.violations` 非空（红线 6） |
| 4 | 输入不可用 | 文件不存在；音频无法解码；`--lines` schema 不符 |
| 5 | 内部错误 | 权重与配置不匹配；未知异常 |

- 「拒绝导出」必须**不留下半成品文件**（先写临时文件再原子移动，或在校验通过后才打开目标文件）。

### 3.3 输出布局

```
<out>/                      # 目录形态（推荐，便于被 Phira 直接读取）
  <chart>.json              # RPEJSON 谱面本体
  info.yml                  # 谱面包元数据（读写接口归 Plan 02；本 plan 只调用）
  report/
    legality.json           # LegalityReport（violations / stats / edits）
    decode.json             # 解码元数据（decode arm / seed / x_bins / frame_rate 派生值）
```

- **输出目录内不得出现音频文件**（红线 5）：`META.song` / `info.yml.music` 只写**文件名引用**，不复制媒体。
- `report/decode.json` 是**可复现性证据**：记录 `frame_rate` 的派生值与来源、`x_bins`、`dx`、解码臂、seed。

## 4. 内部设计

### 4.1 管线编排

```
load audio ─► Stage0 (MERT encoder, 帧率由 config 派生)
           ─► Stage1 (条件: difficulty / line tracks / audio_emb) ─► ChartField
           ─► decoder.decode(field, arm=..., seed=...)          ─► DecodedEvent[]  (秒域)
           ─► postprocess.validate + clamp                      ─► LegalityReport + events
           ─► rpejson.write(...)                                ─► 文件
```

- 每一步都是 Plan 01/03/04/05 的**公共入口**，CLI 内**不写任何领域逻辑**（避免出现「第二份实现」——POSTMORTEM §2.2 的教训是同一条常量被复制三份）；
- CLI 层只做：参数解析 → 配置组合（Hydra compose，可选）→ 调用 → 错误翻译 → 退出码。

### 4.2 错误分类

- **用户错误**（退出码 2/4）：参数与输入问题，信息必须包含「哪个参数/哪个文件/期望什么」；
- **业务失败**（退出码 3）：合法性问题——输出必须列出**前若干条违规项**（含检查项名与位置），便于用户/开发者定位；
- **内部错误**（退出码 5）：打印完整堆栈到 `stderr`，且**不得吞掉**异常（禁止 `except Exception: pass`）。

### 4.3 日志与进度

- 生产路径用 `get_logger`（禁裸 print，CLAUDE.md §4.6）；`--quiet` / `--verbose` 控制级别；
- 长音频的 Stage 0 提取应打印进度（`tqdm` 已在依赖中）；
- **不得**在日志里输出音频内容或大段谱面 JSON。

### 4.4 服务接口（Phase 3）

- 形态：`create_app()` 的 HTTP 服务（现有占位见 `beatmorph/api/app.py`）；
- 语义与 CLI **共用同一条管线函数**（不允许两份实现）；
- 请求体：音频（上传或引用）+ difficulty + 可选线事件轨 + 解码参数；响应体：谱面 JSON + 合法性报告；
- **不落盘音频**、不提供音频下载（红线 5/BasePlan §4.5）；
- 具体框架选型（FastAPI 等）与部署形态**未定**，见 §9-5。

## 5. 依赖关系

| 方向 | 模块 | 内容 |
| --- | --- | --- |
| 上游 | Plan 00 `core/contracts` | 全部跨模块类型（红线 2） |
| 上游 | Plan 01 `audio/` | MERT 编码入口与派生帧率 |
| 上游 | Plan 02 `data/`、`io/formats/rpejson/` | RPEJSON schema、`info.yml` 读写 |
| 上游 | Plan 03 `field/` + Plan 04 `generation/` | 强度场生成 |
| 上游 | Plan 05 `decoder/` | 解码、后处理、写路径与 `LegalityReport` |
| 上游 | Plan 06 `eval/` | `beatmorph-eval` 的实现 |
| 上游 | Plan 07 `infra/` | 配置、checkpoint、门禁开关转发 |
| 外部 | `argparse`（标准库） | **不引入 click/typer**（实读 `pyproject.toml` 无此类依赖） |
| 外部 | `tqdm` / `rich` | 已在前述依赖中 |

## 6. 里程碑与验收

> **门禁硬性要求**：本 plan **不引入任何训练目标或损失**。若在实现过程中出现任何新的可学习组件或损失（例如为 CLI 增加的辅助头），必须先通过 **G1-G4**（`beatmorph/infra/sanity.py`）并把结果写入训练日志，**门禁未绿不得扩大数据规模**（BasePlan §9、CLAUDE.md §5.8）。`--gates` 的语义必须与 Plan 07 §3.2 的 fail-closed 语义一致。

| # | 里程碑 | 可量化验收 |
| --- | --- | --- |
| **M8.1** | CLI 骨架与参数契约 | `beatmorph-generate --help` 退出码 **0**；缺 `--audio` → 退出码 **2** 且 stderr 含参数名；`--difficulty abc` → 退出码 **2**；参数解析单测**不加载模型、不依赖 GPU**，整套 < 1 s |
| **M8.2** | 端到端跑通（单曲） | 用微型夹具音频 + 微型权重跑完整管线：产出 RPEJSON，`LegalityReport.violations == []`（红线 6），退出码 **0** |
| **M8.3** | 可复现性 | 同 `--seed` 同输入两次运行：输出谱面文件**逐字节一致**（thinning 的随机性必须被 seed 完全决定） |
| **M8.4** | 红线 5 断言 | 输出目录扫描：**零音频文件**；`META.song` 为文件名引用（含负例测试：断言实现不会调用 `shutil.copy` 音频） |
| **M8.5** | 拒绝导出语义 | 注入非法谱（构造触发 `violations` 非空的用例）→ 退出码 **3**，stderr 列出违规项，且**输出路径不存在或为空**（无半成品） |
| **M8.6** | 批量模式（Phase 2 末） | `--audio-dir` 批量：单曲失败不影响其它曲；汇总报告含成功/失败计数与失败原因分类 |
| **M8.7** | 服务接口（Phase 3） | `create_app()` 起服务后，与 CLI **同一输入产出同一谱面**（同 seed 逐字节一致，证明共用管线）；接口不返回音频 |

## 7. 风险与缓解

| 风险 | 来源 | 本模块的缓解 |
| --- | --- | --- |
| **R-6** 解码精度 | BasePlan §6 | CLI 暴露 `--decode`，使 B6 对照臂可一键复现；`report/decode.json` 记录解码元数据 |
| **R-8** 跨线几何冲突不可玩 | BasePlan §6 | 导出前强制 `validate()`（退出码 3 拒绝导出）；冲突计数进 `report/legality.json` |
| **R-7** 单位漂移 | BasePlan §6 | CLI **不做任何单位换算**（不写「秒→帧」）；帧率只从 config 派生并在 `decode.json` 留证；用户不可指定采样率 |
| **R-2** 数据合规（**已裁定**，2026-08-05） | BasePlan §4.4、CLAUDE.md 红线 5 附注 | 不落盘/不转发音频；服务接口不提供音频下载；**不提供权重下载/分发入口**（裁决前提 = 最终不发布权重）；训练入口的 provenance 守卫见 Plan 07 §3.2 |
| **R-1 / R-4 / R-5** | BasePlan §6 | 不属 CLI 职责，但 CLI 不得掩盖：非法率、越界占比、跨线冲突数必须原样出现在报告里 |
| 难度条件被误用 | survey §3.2 实测（`level` 是自由文本） | `--difficulty` 只接受**数值**；不提供 `--level` 参数（从入口杜绝 regex 解析） |
| 用户误以为「越界会被自动修正」 | 红线 3 | `report/legality.json` 显式给出越界**计数**与「未做任何钳位」的说明字段 |
| CLI 变成第二份实现 | POSTMORTEM §2.2 同型风险 | CLI 内零领域逻辑（§4.1），并以「与 API 同 seed 逐字节一致」的测试（M8.7）反向约束 |

## 8. 测试策略

**单元（默认 CI，无权重、无 GPU）**

- 参数解析：必填缺失、类型错误、路径不存在、`--decode` 取值域；退出码表逐条覆盖（§3.2）；
- 输出布局：目录/文件形态、`report/*.json` 的 schema 冻结测试；
- 红线 5 断言：输出目录扫描无音频；
- 错误翻译：把内部异常映射到正确退出码（表驱动）；

**集成**：以 fake 模块注入管线各段（不加载真实权重），验证编排顺序、配置透传、`--validate-only` / `--dry-run` 的副作用边界（不写文件）。

**e2e（`slow`）**：真实权重 + 夹具音频 → 合法 RPEJSON（M8.2）；同 seed 复现（M8.3）。

**服务接口（Phase 3，`slow`）**：与 CLI 的等价性测试（M8.7）。

**mock 纪律**：fake 模块**不得**固化帧率/坐标字面量（AGENTS.md §3.3）；需要帧数时引用契约常量。

## 9. 开放问题

1. **`--lines <trajectory.json>` 的 schema 未定义**：判定线事件轨（`moveX/moveY/rotate/alpha/speed` + `extended` 第 5 层，事件字段见 `phigros-format.md` §4.2/§4.3，且**事件必须跨层求和**、必须有补洞预处理）需要成为**跨模块契约类型**——按 CLAUDE.md 红线 2，**新增跨模块类型前必须先开 RFC**（由 contracts-agent 落 `core/contracts`）。本 plan 只声明入口形态，不定义字段。
2. **`line_tracks` 的时间表示**：用 beat 三元组（格式原生）还是秒（评估/解码统一口径）？两者需要一个**单一权威转换点**（倾向：契约内统一为秒，写回时转 beat），未决。
3. **难度的取值范围与语义**：Phira 的 `difficulty` 是 f32 定数，实测范围 9.9–18.5 但样本仅 n = 24 且偏近期更新（survey §3.4 有明确偏差声明）→ 是否限定区间、越界如何提示，需全库统计后定。
4. **`info.yml` 的写归属**：谱面包元数据的读写属 Plan 02 的 `io/formats/rpejson/`，还是本 plan 的输出布局职责，需与 data-agent 对齐（与 Plan 05 §9-9 是同一处边界）。
5. **服务接口的技术选型**：HTTP 框架、异步任务编排（长音频推理）、并发与超时、鉴权与配额，全部未定；BasePlan 只把 API/Demo 放在 Phase 3。
6. **`beatmorph-eval` 的入口形态**：独立 CLI 还是 `beatmorph-generate --eval`？涉及 Plan 06 的产物约定。
7. **权重的本地获取方式**：checkpoint 走外部存储（红线 5），CLI 如何取得（本地路径 / 下载工具）未定。**发布/分发不在范围内**——数据合规裁决（2026-08-05）以「最终不发布模型权重」为前提，任何对外分发场景必须先回到该裁决重新裁定。
8. **`--lines` 缺省时的行为**：BasePlan §2 把线事件轨标为「可选」，但 RFC-0029 §2.2/§3.7 明确「不生成线事件轨」——缺省时是用**一条静止线**（能力边界，RFC §2.2 的 (c) 方案）还是拒绝运行？**必须显式裁定**，因为它直接决定「产物是不是 Phigros 谱面」（RFC §2.2 警告：静止线的产物是退化的无轨道下落式谱面）。
