# AGENTS.md — 子 Agent 协作约定

> 本文件约束在 BeatMorph 仓库中工作的多个 AI agent（subagent / Claude Code 会话）如何分工与防冲突。

## 1. 总原则

- **最高权威是 [`CLAUDE.md`](CLAUDE.md) 与 [`docs/BasePlan.md`](docs/BasePlan.md)**。任何 agent 进入仓库前默认遵守 CLAUDE.md 红线。
- **只通过 `core/contracts` 通信**，不跨模块直连类型。
- **不擅自越权**：偏技术选型、改契约、改 plan 必须开 RFC，不得由 agent 私自决定。

## 2. 模块所有权分工建议

| Agent 角色 | 主责目录 | 典型 plan |
|-----------|---------|-----------|
| data-agent | `beatmorph/data/`, `tests/unit/data/` | 08 |
| audio-agent | `beatmorph/audio/`, `tests/unit/audio/` | 01 |
| tokenizer-agent | `beatmorph/tokenizer/`, `tests/unit/tokenizer/` | 02 |
| planner-agent | `beatmorph/planner/`, `tests/unit/planner/` | 03 |
| gen-agent | `beatmorph/generation/`, `tests/unit/generation/` | 04 |
| rag-agent | `beatmorph/rag/`, `tests/unit/rag/` | 05 |
| align-agent | `beatmorph/alignment/`, `tests/unit/alignment/` | 06 |
| decoder-agent | `beatmorph/decoder/`, `beatmorph/io/`, `tests/unit/decoder/` | 07 |
| infra-agent | `beatmorph/infra/`, `beatmorph/cli/`, `beatmorph/api/`, `configs/` | 09 |
| contracts-agent | `beatmorph/core/contracts/`, `tests/unit/core/` | 00 |

## 3. 并行协作防冲突规则

1. **契约（`core/contracts`）是共享边界**：任何 agent 改契约前必须通报，所有 agent 须以最新契约为准。建议契约变更由 contracts-agent 统一执行。
2. **不并行写同一文件**：多 agent 并行时按上表划界；跨界改动先用 issue/消息同步。
3. **测试夹具共享**：`tests/fixtures/` 与 `data/fixtures/` 为公共资产，改动须可被所有 agent 测试复用。
4. **配置共享**：`configs/` 由 infra-agent 维护，其它 agent 需新配置提需求。
5. **大文件绝不入库**：见 CLAUDE.md §3.5。

## 4. 工作产出标准

每完成一个里程碑，agent 须：
- 对应 `tests/unit/<module>/` 有可过测试。
- 更新所属 plan 的「里程碑」勾选与「状态」图例。
- 若引入偏离 BasePlan 的决策，开 RFC 至 `docs/decisions/`。
- 运行 `make lint && make test-fast` 自检。

## 5. 与 main 会话的同步

- 后台 agent 完成后通过 task-notification 回报，main 负责汇总与 RFC 编号统一。
- 对外可见结论由 main 会话转述，agent 的原始 transcript 不外泄。
