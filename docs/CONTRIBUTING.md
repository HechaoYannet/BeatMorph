# 贡献指南

感谢参与 BeatMorph！本文件说明如何在本仓库高效协作。技术权威为 [`BasePlan.md`](BasePlan.md)，AI 协作准则见 [`../CLAUDE.md`](../CLAUDE.md)。

## 1. 首次环境搭建

需要 Python 3.11 与 [`uv`](https://docs.astral.sh/uv/)（奠基文档 §5 指定）。

```bash
# 安装 uv（任选其一）
pip install uv
# 或 winget install astral-sh.uv

# 克隆并同步环境（含 dev 依赖）
git clone <repo-url> BeatMorph && cd BeatMorph
uv sync --group dev

# 安装 pre-commit 钩子
uv run pre-commit install

# 冒烟验证
make test-fast
```

> Windows 用户：建议在 PowerShell 直接用 `uv` 原生命令；`Makefile` 供类 Unix 环境。

可选依赖组：`--extra audio`（MERT/Demucs）、`--extra train`（Lightning/W&B）、`--extra rag`（FAISS）。
GPU 训练需自行安装匹配 CUDA 的 PyTorch（见 [PyTorch 官网](https://pytorch.org/)）。

## 2. 开发工作流

1. **认领模块**：按 [`AGENTS.md`](../AGENTS.md) 表或 issue 选择模块，读对应 [`plans/0X-*.md`](plans/README.md)。
2. **建分支**：`git checkout -b feat/<module>-<topic>`（默认分支 `main` 受保护）。
3. **实现 + 测试**：先补单元测试（参考 `tests/unit/core/test_contracts.py`），再实现。
4. **本地自检**：
   ```bash
   make lint          # ruff check
   make format        # ruff format + fix
   make typecheck     # mypy
   make test-fast     # 跳过 slow/gpu/e2e
   ```
5. **提交**：Conventional Commits（commitizen 强制）：
   ```
   feat(tokenizer): 实现 VQ-VAE encode 骨架 #plan-02
   fix(decoder): 修正同手间隔判定 #R-5
   docs(plans): 补充 04 开放问题
   ```
6. **PR**：描述改动模块、对应 plan 里程碑、测试结果；请求至少一人评审。

## 3. 代码规范

- **类型标注强制**：公共函数/方法必须标注（`mypy --strict`）。
- **日志**：`from beatmorph.core.logging import get_logger`，禁裸 `print`。
- **契约优先**：跨模块数据用 `beatmorph.core.contracts` 类型，不私造平行结构。
- **命名**：模块/类 ruff `N` 规则；张量维度用 einops 语义命名。
- **大文件禁入库**：权重/音频/parquet 走外部存储或 Git LFS（`.gitignore` 已拦）。

## 4. 文档与决策

- 实质偏离 BasePlan → 先在 [`decisions/`](decisions/README.md) 开 RFC，状态变「采纳」后再改代码。
- 新增/变更跨模块契约 → 同步更新 [`plans/00-core-contracts.md`](plans/00-core-contracts.md) 与本文件相关处。
- 完成里程碑 → 更新对应 plan 状态图例（🟡→✅）。

## 5. 测试要求

- 每个模块至少覆盖：契约边界值、关键算法形状正确性。
- GPU/慢测试打 `@pytest.mark.gpu` / `@pytest.mark.slow`，CI 默认跳过。
- 端到端测试 `@pytest.mark.e2e`，需真实音频，跑在专属环境。

## 6. 分阶段参与（对应 BasePlan §7）

| Phase | 重点 | 可贡献模块 |
|-------|------|-----------|
| 1（0-3 月） | 数据流水线 + MERT 离线提取 + VQ-VAE + Stage1 | data/audio/tokenizer/planner |
| 2（3-8 月） | AR 生成 + RAG + 首版可玩 .osu | generation/rag/decoder/io |
| 3（8-12 月） | DPO + Flow Matching 蒸馏 + API/Demo | alignment/generation/infra/api |
| 4（12+ 月） | 多模式 + 个人风格 LoRA | io/formats 扩展 |

## 7. 行为准则

- 尊重所有贡献者；使用中性人称（they/them）指代未明确代词的人。
- 讨论技术分歧对事不对人；重大分歧走 RFC 流程而非 PR 评论僵局。
