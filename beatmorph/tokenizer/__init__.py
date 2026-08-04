"""BeatMorph package: beatmorph/tokenizer.

RFC-0028 后谱面 tokenizer 主路径为 BPE/event（``BPETokenizer``）；
VQ-VAE（``vqvae.py``）移至 ``archive/vqvae-baseline`` 分支作对照基线。
"""

from beatmorph.core.contracts import BPE_DEFAULT_VOCAB, EventToken
from beatmorph.tokenizer.bpe import BPETokenizer

__all__ = ["BPE_DEFAULT_VOCAB", "BPETokenizer", "EventToken"]
