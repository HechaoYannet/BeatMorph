# AGENTS.md — 子 Agent 协作约定

> 本文件约束在 BeatMorph 仓库中工作的多个 AI agent（subagent / Claude Code 会话）如何分工与防冲突。
> **v3.0（2026-08-05）**：随 RFC-0029 范式迁移更新（目标 osu!mania 4K → Phigros）。

## 1. 总原则

- **最高权威是 [CLAUDE.md](CLAUDE.md) 与 [docs/BasePlan.md](docs/BasePlan.md)**（v3.0）。任何 agent 进入仓库前默认遵守 CLAUDE.md 红线。
- **只通过 `core/contracts` 通信**，不跨模块直连类型。
- **不擅自越权**：偏技术选型、改契约、改 plan 必须开 RFC（[docs/decisions/](docs/decisions/README.md)），不得由 agent 私自决定。
- **不确定的事实先查 [docs/knowledges/](docs/knowledges/)**（格式 RPEJSON / 单位几何 / 数据集 / 文献），那里的每条结论都有来源分级（A=源码或官方文档，B=社区 wiki，C=二手）与显式存疑清单。**查不到就写"未查证"，不要猜。**

## 2. 模块所有权分工建议

| Agent 角色 | 主责目录 | 典型 plan |
|-----------|---------|-----------|
| contracts-agent | `beatmorph/core/contracts/`, `tests/unit/core/` | 00 |
| audio-agent | `beatmorph/audio/`, `tests/unit/audio/` | 01 |
| data-agent | `beatmorph/data/`, `beatmorph/io/formats/rpejson/`, `tests/unit/data/` | 02 |
| field-agent | `beatmorph/field/`, `tests/unit/field/` | 03 |
| gen-agent | `beatmorph/generation/`, `tests/unit/generation/` | 04 |
| decoder-agent | `beatmorph/decoder/`, `tests/unit/decoder/` | 05 |
| eval-agent | `beatmorph/eval/`, `tests/unit/eval/` | 06 |
| infra-agent | `beatmorph/infra/`, `beatmorph/cli/`, `beatmorph/api/`, `configs/` | 07 / 08 |

> **v2.x 旧角色已退役**：tokenizer-agent、planner-agent、rag-agent、align-agent（随 RFC-0029 退出主路径，旧实现归档 `archive/osu-mania` 分支）。

## 3. 并行协作防冲突规则

1. **契约（`core/contracts`）是共享边界**：任何 agent 改契约前必须通报，所有 agent 须以最新契约为准。契约变更由 contracts-agent 统一执行。
2. **不并行写同一文件**：多 agent 并行时按上表划界；跨界改动先同步。
3. **测试夹具共享**：`tests/fixtures/` 与 `data/fixtures/` 为公共资产，改动须可被所有 agent 复用。
   - ⚠️ **夹具与 mock 不得固化物理常量**。若 mock 必须产生帧数/坐标，须引用契约常量——25 Hz 之所以存活到万级数据规模，正是 mock 把它洗成了绿灯（见 [POSTMORTEM-2026-08-05](docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md)）。
4. **配置共享**：`configs/` 由 infra-agent 维护，其它 agent 需新配置提需求。
5. **大文件绝不入库**：见 CLAUDE.md 红线 5。
6. **文档产出不越界**：每个调研/写作 agent 只写自己被指派的那几个文件，**不得顺手改别的文档**。

## 4. 工作产出标准

每完成一个里程碑，agent 须：

- 对应 `tests/unit/<module>/` 有可过测试；**契约级测试不得依赖权重或 GPU**。
- 更新所属 plan 的「里程碑」勾选与「状态」图例。
- **新增任何训练目标/损失时，先跑通门禁**（`beatmorph/infra/sanity.py`：G1 单 batch 过拟合 / G3 常数基线 / G4 契约断言；原 G2 打乱标签对照已随 [RFC-0037](docs/decisions/RFC-0037-remove-g2-and-per-event-loss.md) 删除，其命题由 val 的 held-out 置换对照 `val/nll_shuffled_delta` + `val/ratio` 承担），结果写入训练日志。**门禁未绿不得扩大数据规模**；任一 val 点 `val/ratio >= 1` 或 `val/nll_shuffled_delta <= 0` ⇒ 该 run 的扩规模结论作废。
- 若引入偏离 BasePlan 的决策，开 RFC 至 `docs/decisions/`。
- 运行 `make lint && make test-fast` 自检。
- **物理常量一律写成派生式**（如 `RPE_STAGE_WIDTH / N`、`MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`），**禁止硬编码具体数字**（CLAUDE.md 红线 7）。

## 5. 与 main 会话的同步

- 后台 agent 完成后通过 task-notification 回报，main 负责汇总与 RFC 编号统一。
- 对外可见结论由 main 会话转述，agent 的原始 transcript 不外泄。
- **调研类 agent 的产出必须包含「存疑清单」**：把"我们不知道什么"与已知事实一并交付，是本研究项目的硬要求。

## 6. 跨 session 交接（README 的「当前状态 / 下一步」）

[README.md](README.md) 的 `## 当前状态与下一步` 两节是**交接件**，读者是**下一个 session 的 agent**。它**不是**进度记录。

**硬性规则**：

1. **每次交接必须整节重写**这两节（`edit` 替换整节内容），**不得在其后追加条目**。
2. **不得把已完成事项堆进「当前状态」**——已完成的事属于 git log / plan 里程碑 / RFC 状态表，不属于 README。
3. 「下一步」只保留**仍然有效**的事项：做完的删掉，失效的删掉，不要留"已完成"的勾。
4. 值得长期保留的信息放进对应文档（BasePlan / RFC / plan / `docs/knowledges/`），**不要留在 README**。
5. 判断标准：**README 不应随时间变长**。若一次交接让它变长了，说明在写流水账。
6. 交接时同步更新「上次交接」日期与交接人。

> **理由**：项目已有三层记录——`docs/plans/` 的里程碑勾选、RFC 的状态表、git log。README 再叠一层"进度流水账"只会制造第四份互相矛盾的真相。交接件要回答的是"**现在能跑什么、下一步做什么**"，而不是"我们干过什么"。
