# CLAUDE.md — BeatMorph 项目宪法

> 本文件是 Claude Code（及任何 AI 协作 agent）在本仓库工作时的最高行为准则。
> 一切开发以 [`docs/BasePlan.md`](docs/BasePlan.md) 奠基文档为唯一技术权威。

## 1. 项目是什么

BeatMorph 是**音游谱面端到端自动生成系统**：从原始音频（WAV/MP3）+ 难度(1-15) + 参考谱面(可选)，自动生成可玩音游谱面，优先 4K VSRG，输出 `.osu` / `.sm` / `.ma2`。

**核心范式（不可动摇，见 BasePlan §1.2）**：
```
音频 → MERT隐式理解 → BPE/event谱面分布学习 → RAG风格迁移 → DPO手感优化
# 注：tokenizer 范式经 RFC-0028 修宪，由 VQ-VAE 小节码本改为 BPE/event 序列；
# VQ-VAE 实现移至 archive/vqvae-baseline 分支作对照基线，不在主路径。
```

设计哲学：**自监督理解取代显式标注**。模型从百万级现成谱面自主学习，人只提供极简控制（难度+参考风格），标注成本≈0。

## 2. 模块拓扑

```
beatmorph/
├── audio/encoder/      Stage 0  MERT-v1-330M + Adapter（冻结主干，训 Adapter）
├── audio/separation/   Stage 0  Demucs(HTDemucs) 可选四轨分离
├── tokenizer/          BPE/event 谱面语义 tokenizer（REMI 式 event 序列 + BPE，词表 ~4096；RFC-0028）
├── planner/            Stage 1  6层双向 Transformer 全局密度规划
├── generation/         Stage 2  AR Transformer Decoder（备选 Flow Matching）
├── rag/                风格检索  RAG，Top-K=3，零训练
├── alignment/          偏好对齐  DPO（非 RLHF/PPO）
├── decoder/            Stage3&4 event→Note 直映射 + 规则后处理 + 格式导出（无 VQ 解码网络）
├── data/               数据流水线 .osu 解析 + 统计量 + MERT 离线提取
├── io/formats/         IR ↔ .osu/.sm/.ma2 互转
├── core/contracts/     跨模块数据契约（Note/Chart/Section/EventToken；PatternToken 随 VQ 退役归 baseline 分支）
├── core/               logging 等基础设施
├── infra/config/       Hydra 配置
├── cli/  api/          命令行与服务接口
```

各模块详细计划见 [`docs/plans/`](docs/plans/README.md)（00-09，每份含接口契约/里程碑/风险/测试策略）。

## 3. 不可违背的约束（红线）

1. **技术选型锁定 BasePlan**：不得擅自替换 MERT→Qwen2-Audio、AR→Diffusion(DDPM)、RAG→对比学习、DPO→RLHF/PPO、自监督回归→规则/LLM。任何变更须先开 RFC（[`docs/decisions/`](docs/decisions/README.md)）。
   > **tokenizer 范式（RFC-0028 修宪，2026-08-04 采纳）**：主路径锁定 **BPE/event tokenizer**（REMI 式 event 序列 + BPE 合并，词表 ~4096）——原 VQ-VAE 小节码本范式解锁，VQ-VAE 实现移至 `archive/vqvae-baseline` 分支仅作对照基线，**不在主路径**。未来再换 tokenizer 范式（event→FSQ/规则/LLM 等）须先开 RFC。
2. **模块间只通过 `core/contracts` 通信**：跨模块数据用 `Chart`/`Note`/`Section`/`PatternToken` 等已定义类型；张量形状遵循 `core/contracts/tensors.py`。**新增跨模块类型前先开 RFC**。
3. **AI 与规则解耦**（BasePlan §3.7）：后处理规则引擎只做「物理红线」校验（4K 单帧≤2 键、同手间隔≥70ms、禁止越界），**不得改变 AI 的键型排列逻辑**（R-5）。
4. **先攻 4K VSRG**（BasePlan §1.1/§6 R-6）：其它模式（6K/osu!std/maimai）视为独立适配工程，不得为它污染 4K 主路径。
5. **数据/权重不入库**（见 `.gitignore`）：模型权重(`*.safetensors/*.pt`)、原始音频(`*.mp3/*.wav`)、数据集(`*.parquet`)走外部存储或 Git LFS，仅保留 `data/fixtures/**` 小型测试夹具。
6. **100% 可玩性**：生成谱面必须经后处理 `validate()` 为空违规项才允许导出（保证不生成人类无法击打的谱面）。

## 4. 工程约定

- **语言/版本**：Python 3.11（BasePlan 要求 3.10+）。
- **包管理**：`uv`（BasePlan §5 指定）。环境复现：`uv sync`（见 Makefile）。
- **格式/lint**：`ruff`（已配 `pyproject.toml`）。提交前 `make lint`。
- **类型**：`mypy --strict`；ML 库缺 stub 处已在 overrides 忽略。
- **测试**：`pytest`，标记 `slow`/`gpu`/`integration`/`e2e`。CI 默认跑 `not slow and not gpu`。
- **提交**：Conventional Commits（pre-commit commitizen 强制）；每个 commit 引用相关 plan 或 RFC。
- **换行符**：全仓 LF（`.gitattributes` + `.editorconfig` 已强制），跨设备协作关键。

## 5. 开发流程（对 AI agent）

1. **改任何模块前，先读对应 `docs/plans/0X-*.md`**，确认接口契约与里程碑；若有冲突，以 plan 为准，plan 与 BasePlan 冲突时以 BasePlan 为准并开 RFC。
2. **跨模块改动**先动 `core/contracts`，契约变更须同步 plan 与 RFC。
3. **新功能**遵循 plan 里程碑顺序，不得跳阶段（如未实现 Stage0/1/2 就先写 API）。
4. **不确定的技术细节**：查 BasePlan 相应章节；仍不确定则写进 plan「9. 开放问题」并提 RFC，**不要凭猜测改代码**。
5. **测试先行**：每个模块至少有契约/形状级单元测试（参考 `tests/unit/core/test_contracts.py`）。
6. **日志**：`from beatmorph.core.logging import get_logger`，禁止裸 `print` 进生产路径。
7. **大文件**：写代码前确认产物不在 `.gitignore` 黑名单（权重/音频/parquet）。

## 6. 当前状态

- 仓库刚初始化：git main 分支已就绪，工程配置（pyproject/ruff/pre-commit/.gitattributes）已落地。
- 仅有骨架代码（各模块 `__init__.py` + 接口占位）与核心契约实现（`core/contracts` 可用、测试可跑）。
- 所有 plan 处于 🟡草案 状态，待评审。
- **下一步**（BasePlan §7 Phase 1）：数据预处理流水线 + 10K 首 MERT Embedding 离线提取 + **BPE/event tokenizer 词表训练**（RFC-0028 采纳，~4700 event/曲、上下文 ~1024 分段）+ AR 生成主干。

## 7. 文档导航

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 快速开始与总览 |
| [docs/BasePlan.md](docs/BasePlan.md) | **技术奠基（最高权威）** |
| [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) | 代码结构详解 |
| [docs/plans/](docs/plans/README.md) | 各模块实施计划 |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | 贡献流程 |
| [docs/decisions/](docs/decisions/README.md) | RFC 决策记录 |
| [docs/glossary.md](docs/glossary.md) | 术语表 |
| [AGENTS.md](AGENTS.md) | 子 agent 协作约定 |
