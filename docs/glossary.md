# 术语表

BeatMorph 相关术语速查。技术细节以 [`BasePlan.md`](BasePlan.md) 为准。

## 音游领域

| 术语 | 释义 |
|------|------|
| **VSRG** | Vertical Scrolling Rhythm Game，垂直下落式音游（如 osu!mania / StepMania / BMS） |
| **4K** | 4 键 VSRG；本项目 Phase 1-3 主攻模式（见 BasePlan §1.1） |
| **7K / osu!std / maimai** | 扩展目标模式（Phase 4） |
| **Note** | 谱面最小单位，本项目规范为 `(time, lane, type, duration)`（见 `core/contracts`） |
| **TAP / HOLD / MINE / ROLL** | Note 类型：普通键 / 长条 / 地雷 / 滚动长条（见 `NoteType`） |
| **lane（键位）** | 下落轨道编号，4K 即 0..3 |
| **BPM** | Beats Per Minute，每分钟节拍数 |
| **小节（Bar）** | 4/4 拍下 4 拍为一小节；VQ-VAE 以小节为量化粒度（BasePlan §3.2） |
| **Section（段落）** | 结构段落，本项目每 4 小节一个，Stage1 规划单元（BasePlan §3.3） |
| **密度（Density）** | 单位时间 Note 数量；Stage1 输出目标密度 0-1 |
| **双押 / 交互 / 阶梯 / 长条** | 手型 Pattern 类别，VQ-VAE 码本语义维度之一（BasePlan §3.2.2） |
| **Pass Rate** | 玩家通关率，DPO 偏好对数据来源之一（BasePlan §3.6） |
| **`.osu` / `.sm` / `.ma2`** | osu!mania / StepMania / maimai 谱面格式 |

## 模型/算法

| 术语 | 释义 |
|------|------|
| **MERT-v1-330M** | 音乐自监督预训练编码器，330M 参数，25Hz、768d 输出；Stage0 主选（§3.1） |
| **Adapter（LoRA/MLP）** | 冻结主干外附加的可训练轻量模块（§3.1） |
| **Demucs / HTDemucs** | 混音分轨模型，输出 drums/bass/vocals/other 四轨；可选增强（§3.1.2） |
| **VQ-VAE** | Vector Quantized VAE，将小节 Note 集合量化为离散码本索引；码本 2048/4096（§3.2） |
| **码本坍缩（Codebook Collapse）** | VQ 训练中大部分码本索引不被使用的退化现象；缓解见 R-2 |
| **AR Transformer** | 自回归 Transformer Decoder，Stage2 生成主干（§3.4.1） |
| **Teacher Forcing** | 训练时用真值前缀作输入的自回归训练法（§3.4.1） |
| **Cross-Attention / AdaLN** | 条件注入方式：音频经 cross-attn，难度/风格经 AdaLN 调制（§3.4.1） |
| **Flow Matching（Rectified Flow）** | 连续归一化流生成，4-8 步采样；AR 的未来蒸馏加速备选（§3.4.2） |
| **RAG** | 检索增强生成，检索 Top-K 参考谱面注入 decoder；零训练风格控制（§3.5） |
| **DPO** | Direct Preference Optimization，直接偏好对齐；非 RLHF/PPO（§3.6） |
| **自监督回归** | Stage1 用谱面自动统计量作伪标签训练，零人工标注（§3.3） |
| **IR（中间表示）** | 与格式无关的谱面规范表示，即 `Chart`（§3.7） |

## 工程

| 术语 | 释义 |
|------|------|
| **uv** | Astral 出品的极速 Python 包管理器；本项目指定（§5） |
| **Hydra** | 组合式配置框架，管理超参（§5） |
| **W&B** | Weights & Biases 实验追踪（§5） |
| **FSDP** | Fully Sharded Data Parallel，PyTorch 分布式训练策略（§5） |
| **契约（Contract）** | `beatmorph.core.contracts` 中跨模块共享的数据类型约定 |
| **RFC** | Request for Comments，本项目用于记录技术决策/偏离 BasePlan（见 `docs/decisions/`） |
| **Plan** | 各模块实施计划，见 `docs/plans/` |
| **红线规则** | 后处理中物理不可达校验，不改 AI 键型逻辑（R-5） |
