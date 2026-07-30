"""预处理流水线编排（奠基文档 §4.2）。

原始 (.osu + .mp3)
  → Step 1: 解析 .osu → NoteEvent[]
  → Step 2: 自动计算统计量 → Stage 1 伪标签
  → Step 3: 音频切片 & MERT 离线预提取 Embedding
  → Step 4: 构建 VQ-VAE 训练数据集
  → 存入 HuggingFace Datasets / 本地 Parquet
"""

from __future__ import annotations

from pathlib import Path


class PreprocessPipeline:
    """端到端预处理流水线编排。

    质量过滤（奠基文档 §4.3）：保留 ≥3 星且 play_count > 500 的谱面。
    """

    MIN_STARS: float = 3.0
    MIN_PLAY_COUNT: int = 500

    def __init__(self, raw_dir: Path, out_dir: Path, cache_format: str = "parquet") -> None:
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.cache_format = cache_format

    def run(self, limit: int | None = None) -> None:
        """执行完整预处理流水线。"""
        raise NotImplementedError("预处理流水线未实现，见 docs/plans/08-data-pipeline.md")

    def extract_mert_embeddings(self, audio_dir: Path) -> None:  # noqa: ARG002
        """Step 3：离线批量提取 MERT Embedding，节省训练时算力。"""
        raise NotImplementedError("见 docs/plans/08-data-pipeline.md")
