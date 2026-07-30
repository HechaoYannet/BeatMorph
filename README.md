# BeatMorph

> 音游谱面端到端自动生成系统 —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → VQ-VAE 谱面分布学习 → RAG 风格迁移 → DPO 手感优化`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（1-15）+ 参考谱面（可选），端到端自动生成高质量、可玩的音游谱面。**优先支持 4K VSRG**（垂直下落式），输出标准 `.osu` / `.sm` / `.ma2` 格式。模型从百万级社区谱面自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。工程脚手架、数据契约、模块计划已就绪，模型实现按分阶段路线图推进。

---

## 系统架构

```
用户输入: ① 音频(WAV/MP3)  ② 难度(1-15)  ③ 参考谱面(可选)
   │
   ▼
Stage 0  音频理解    MERT-v1-330M(冻结) + 可训练 Adapter   [可选 Demucs 四轨分离]
         → audio_emb [T_seq, 768] @ 25Hz
   ▼
Stage 1  全局规划    6层双向 Transformer（自监督回归）
         → 每4小节 density / energy / rest 蓝图
   ▼
Stage 2  Pattern 生成  AR Transformer Decoder + VQ-VAE(2048/4096)
         + RAG(Top-K=3 风格检索)  +  DPO(偏好对齐)
         → Pattern Token 序列
   ▼
Stage 3&4 解码导出    VQ-VAE Decoder → Note[]>(time,lane,type,duration)
         → 规则后处理(物理红线) → .osu / .sm / .ma2
```

完整技术选型与决策理由见 **[奠基文档 docs/BasePlan.md](docs/BasePlan.md)**。

## 仓库结构

```
beatmorph/        主包（audio / tokenizer / planner / generation / rag /
                  alignment / decoder / data / io / core / infra / cli / api）
tests/            unit / integration / e2e / fixtures
configs/          Hydra 配置（model / train / data）
docs/             BasePlan + plans + decisions + 约束文档
```

详见 [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)。

## 快速开始

需 Python 3.11 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/HechaoYannet/BeatMorph.git BeatMorph && cd BeatMorph
uv sync --group dev          # 安装依赖
uv run pre-commit install    # 安装提交钩子
make test-fast               # 冒烟测试（跳过 slow/gpu/e2e）
```

端到端生成（模型实现后启用）：
```bash
beatmorph-generate --audio song.mp3 --difficulty 8 --ref reference.osu -o out.osu
```

> GPU 训练需自行安装匹配 CUDA 的 PyTorch；可选依赖组：`uv sync --extra audio --extra train --extra rag`。

## 技术选型一览

| 模块 | 决策 | 理由 |
|------|------|------|
| 音频编码 | **MERT-330M + Adapter** | 音乐专用、轻量、层次表征好（非 Qwen2-Audio） |
| 声源分离 | **Demucs（可选）** | 有增益但非必需 |
| 谱面压缩 | **VQ-VAE** | 语义离散化、适配 AR（非 FSQ） |
| 生成主干 | **AR Transformer**（备选 Flow Matching） | 成熟稳定；FM 用于未来加速 |
| 风格控制 | **RAG** | 零训练、可解释（非对比学习） |
| 偏好对齐 | **DPO** | 轻量、适合离线数据（非 RLHF/PPO） |
| 全局规划 | **自监督回归** | 零标注、利用统计量（非规则/LLM） |

## 实施路线

| Phase | 时间 | 目标 |
|-------|------|------|
| 1 | 0-3 月 | 数据流水线 + 10K 首 MERT 离线提取 + VQ-VAE(重建>95%) + Stage1 |
| 2 | 3-8 月 | AR 生成 + RAG + 首版可玩 `.osu` + 内部盲测 |
| 3 | 8-12 月 | DPO 微调 + Flow Matching 蒸馏(<2s/首) + API/Demo 封测 |
| 4 | 12+ 月 | 6K / osu!std / 个人风格 LoRA |

详见 [docs/BasePlan.md §7](docs/BasePlan.md) 与各模块 [plans/](docs/plans/README.md)。

## 文档

- [docs/BasePlan.md](docs/BasePlan.md) — **技术奠基文档（最高权威）**
- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) — 代码结构详解
- [docs/plans/](docs/plans/README.md) — 各模块实施计划（00-09）
- [docs/decisions/](docs/decisions/README.md) — RFC 决策记录
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — 贡献指南
- [docs/glossary.md](docs/glossary.md) — 术语表
- [CLAUDE.md](CLAUDE.md) — AI 协作宪法
- [AGENTS.md](AGENTS.md) — 子 agent 协作约定

## 协作

- 默认分支 `main`；特性分支 `feat/<module>-<topic>`。
- 提交遵循 Conventional Commits（pre-commit 强制）。
- 偏离 BasePlan 的技术变更须先开 RFC（[docs/decisions/](docs/decisions/README.md)）。
- 换行符统一 LF，模型权重/音频/数据集不入库（走 LFS/外部存储）。

## 许可证

[Apache-2.0](LICENSE) © 2026 BeatMorph Team。

> 训练数据仅用于学术/研究目的；生成系统输出不包含原音频拷贝（BasePlan §4.3 版权原则）。
