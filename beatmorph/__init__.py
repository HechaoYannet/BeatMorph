"""BeatMorph — 音游谱面端到端自动生成系统。

从原始音频（WAV/MP3）端到端自动生成高质量、可玩的音游谱面，
优先支持 4K VSRG（垂直下落式音游），输出标准 .osu / .sm / .ma2 格式。

核心范式（奠基文档 docs/BasePlan.md）：
    音频 → MERT 隐式理解 → VQ-VAE 谱面分布学习 → RAG 风格迁移 → DPO 手感优化

模块拓扑（详细见 docs/CODE_STRUCTURE.md 与 docs/plans/）::

    audio.encoder      Stage 0  音频编码 (MERT + Adapter)
    audio.separation   Stage 0  可选声源分离 (Demucs)
    tokenizer          VQ-VAE   谱面语义 Tokenizer
    planner            Stage 1  全局密度规划
    generation         Stage 2  AR Transformer 生成主干
    rag                Stage 2  RAG 风格检索
    alignment          DPO      偏好对齐
    decoder            Stage3&4 解码 + 后处理 + 导出
    data               数据预处理流水线与解析器
    io                 IR 与各游戏格式互转
    core               跨模块契约、日志、配置
    infra              训练/推理基础设施
    cli / api          命令行与服务接口

一切开发以 docs/BasePlan.md 为准。
"""

__version__ = "0.1.0"
