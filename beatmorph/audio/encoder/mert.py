"""Stage 0：多模态音频编码器（MERT-v1-330M + Adapter）。

奠基文档 §3.1。冻结 MERT 主干，仅训轻量 Adapter（LoRA，RFC-0003 采纳），
输出帧率由主干 stride 派生（MERT-v1-330M = 75Hz）、1024 维帧级序列 embedding。Demucs 分轨为可选增强
（:mod:`beatmorph.audio.separation.demucs`，本 Phase 1 留 stub）。

详细计划：docs/plans/01-audio-encoder.md

实现要点：
- 权重来源优先级可配（默认 ``modelscope`` → ``huggingface`` fallback），见 RFC-0024
  source 优先级思路。MERT 主干全部 ``requires_grad_(False)`` 冻结。
- Adapter：``"lora"``（peft 注入 attention q/v，rank=8）/ ``"mlp"``（2 层 MLP 残差挂
  第 ``layer`` 层后）/ ``"none"``（纯冻结直出，离线提取用）。
- 输出契约（不可破）：``[B, T_seq, 1024]``，帧率由主干 config **派生**
  （``MERT_SAMPLE_RATE_HZ / prod(conv_stride) = 24000/320 = 75 Hz``，见
  ``output_frame_rate()`` 与 docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md），
  ``T_seq ≈ duration_s * 75``；1024 与 Stage1/Stage2 直连免投影。
- 长音频滑窗 5s / 重叠 1s + 重叠区平均（plan 01 §4，RFC-0002 暂定固定窗）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────────────
_DEFAULT_MODEL_NAME = "m-a-p/MERT-v1-330M"
_TARGET_SR = MERT_SAMPLE_RATE_HZ  # MERT-v1-330M 官方要求 24kHz 单声道
_WINDOW_S = 5.0  # 滑窗 5s
_OVERLAP_S = 1.0  # 重叠 1s
_FRAME_RATE = MERT_FRAME_RATE_HZ  # 75.0（派生量，仅作主干 config 不可用时的兜底）


def _conv_stride_product(config: object) -> int | None:
    """从模型 config 推导特征提取器总下采样倍数（`prod(conv_stride)`）。

    MERT-v1-330M 的 `conv_stride = [5, 2, 2, 2, 2, 2, 2]` → 320。返回 `None`
    表示 config 不可读/无该字段——调用方应回落契约常量，而不是猜一个数。
    """
    strides = getattr(config, "conv_stride", None)
    if strides is None and isinstance(config, dict):
        strides = config.get("conv_stride")
    if not strides:
        return None
    total = 1
    for s in strides:
        total *= int(s)
    return total if total > 0 else None


class MERTAdapter(nn.Module):
    """Stage 0 编码器：冻结 MERT-v1-330M + 可训 Adapter。

    Args:
        model_name: 模型标识。默认 HF id ``m-a-p/MERT-v1-330M``；ModelScope 仓 ID
            可经同名覆盖（需与 ``source`` 配合）。
        layer: 取第几层 hidden state（默认 12；MERT-v1-330M 各层均为 1024 维）。
        adapter: ``"lora"`` / ``"mlp"`` / ``"none"``。默认 ``"lora"``（RFC-0003）。
        lora_rank: LoRA rank（默认 8）。
        lora_alpha / lora_dropout: LoRA 超参。
        use_demucs: Demucs 四轨增强开关（Phase 1 默认 False，demucs 仍 stub）。
        source: 权重来源优先级，``"modelscope"``（默认）或 ``"huggingface"``。
        device: 加载设备，``"auto"`` 选 CUDA 可用时用之。
        fp16: 推理是否 FP16（plan 01 §4）。

    Note:
        构造即下载权重（首次 ~1.3GB）。模型在 ``data/`` 侧首次实例化时拉取，
        缓存走 ``BEATMORPH_MODELS_DIR`` / ``HF_HOME``。
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        layer: int = 12,
        adapter: str = "lora",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_demucs: bool = False,
        source: str = "modelscope",
        device: str = "auto",
        fp16: bool = True,
    ) -> None:
        super().__init__()

        if adapter not in ("lora", "mlp", "none"):
            raise ValueError(f"adapter must be lora/mlp/none, got {adapter!r}")
        if layer < 0:
            raise ValueError(f"layer must be >=0, got {layer}")

        self.model_name = model_name
        self.layer = layer
        self.adapter = adapter
        self.use_demucs = use_demucs
        self.source = source
        self.fp16 = fp16 and device != "cpu"
        self._device = self._resolve_device(device)

        # Phase 1：Demucs 仍 stub，use_demucs=True 提前拒（避免无谓加载 MERT 主干）
        if use_demucs:
            raise NotImplementedError("Demucs 分离 Phase 1 未实现，use_demucs 暂不可用")

        # ── 加载主干（冻结） + Adapter ──
        self.backbone: nn.Module = self._load_backbone()
        self._freeze_backbone()

        self._feat_processor: _FeatProcessor = self._load_feat_processor()
        self._demucs: nn.Module | None = None

        self.adapter_module: nn.Module | None = None
        self.adapter_module = self._build_adapter(lora_rank, lora_alpha, lora_dropout)

        self.to(self._device)
        if self.fp16:
            self.half()
        self.eval()

    # ── 前向 ──────────────────────────────────────────────────

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """编码音频为帧级 embedding。

        Args:
            wav: ``[B, samples]`` 24kHz 单声道波形（``_TARGET_SR``）。
        Returns:
            ``[B, T_seq, feat]`` 帧级 embedding；帧率见 :meth:``output_frame_rate``
            （MERT-v1-330M = 75Hz），``feat`` = :data:``MERT_DEFAULT_FEAT_DIM``。
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"wav must be [B, samples], got shape {tuple(wav.shape)}")

        device = next(self.parameters()).device
        wav = wav.to(device)
        if self.fp16:
            wav = wav.half()

        # 长音频滑窗（5s 窗 / 1s 重叠），短于窗直接整段
        seq_len = wav.shape[1]
        win = int(_WINDOW_S * _TARGET_SR)
        hop = int((_WINDOW_S - _OVERLAP_S) * _TARGET_SR)
        if seq_len <= win:
            emb = self._encode_chunk(wav)
            return emb

        outs: list[torch.Tensor] = []
        starts: list[int] = []
        pos = 0
        while pos < seq_len:
            end = min(pos + win, seq_len)
            chunk = wav[:, pos:end]
            outs.append(self._encode_chunk(chunk))
            starts.append(pos)
            if end >= seq_len:
                break
            pos += hop

        # 重叠区平均：把每段按其覆盖的帧对齐拼回，重叠帧取均值
        # 帧率从主干 config 派生（不再假设 25Hz）——帧率错 = 接缝处系统性错位
        return self._merge_overlapping(outs, starts, hop, self.output_frame_rate())

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """契约名（plan 01 §3.1），等价于 :meth:`forward`。"""
        return self.forward(wav)

    def output_frame_rate(self) -> float:
        """输出帧率（Hz），由主干 config 的 `conv_stride` 推导。

        `24000 / prod([5, 2, 2, 2, 2, 2, 2]) = 24000 / 320 = 75.0`。主干 config
        不可读时回落契约常量 :data:`MERT_FRAME_RATE_HZ` 并告警——绝不静默假设。
        """
        stride = _conv_stride_product(getattr(self.backbone, "config", None))
        if stride is None:
            logger.warning(
                "无法从主干 config 推导 conv_stride，回落契约帧率 %.1fHz",
                MERT_FRAME_RATE_HZ,
            )
            return MERT_FRAME_RATE_HZ
        return _TARGET_SR / stride

    def _encode_chunk(self, wav: torch.Tensor) -> torch.Tensor:
        """编码一段 ≤5s 音频 → [B, T_seq, `MERT_DEFAULT_FEAT_DIM`]。"""
        # MERT 需要 input_values（[normalized]），特征提取器做归一
        # 特征提取器期望 1D 波形；传入 2D [1, samples] 在新版 transformers
        # 中可能误加 channel 维 → squeeze 到 1D 后调用
        wav_1d = wav.squeeze(0)  # [1, samples] → [samples]
        inputs = self._feat_processor(wav_1d, sampling_rate=_TARGET_SR, return_tensors="pt")
        input_values = inputs["input_values"]  # [samples] 或 [1, samples]
        # 统一成 2D [batch, samples]
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)
        elif input_values.dim() >= 3:
            # 如有意外 channel 维 [batch, 1, samples] → squeeze
            input_values = input_values.squeeze(1)
        dtype = torch.float16 if self.fp16 else torch.float32
        input_values = input_values.to(dtype=dtype, device=wav.device)

        outputs = self.backbone(
            input_values,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple: (n_layers+1) x [B, T, feat]
        if hidden_states is None or self.layer >= len(hidden_states):
            raise RuntimeError(
                f"MERT hidden_states 不含 layer={self.layer}（共 "
                f"{0 if hidden_states is None else len(hidden_states)} 层）"
            )
        emb = hidden_states[self.layer].to(dtype)

        # MLP Adapter：在选定层输出后接残差（不投影，保 1024 维）
        if self.adapter == "mlp" and self.adapter_module is not None:
            emb = emb + self.adapter_module(emb)
        # LoRA 已通过 peft 注入 backbone 内部，无需额外处理
        return emb  # type: ignore[no-any-return]

    # ── 滑窗合并 ──────────────────────────────────────────────

    @staticmethod
    def _merge_overlapping(
        outs: list[torch.Tensor],
        starts: list[int],
        hop_samples: int,
        frame_rate: float = _FRAME_RATE,
    ) -> torch.Tensor:
        """把若干重叠段的 frame 序列按 hop 对齐拼回，重叠帧取均值。

        Args:
            outs: 各段 `[B, T_seg, feat]`。
            starts: 各段在原始波形里的起始**样本**偏移。
            hop_samples: 段间 hop（样本数）；仅作调用方语义记录，本函数按
                `starts` 反推帧偏移。
            frame_rate: 输出帧率 Hz；**必须**传 :meth:`output_frame_rate()`
                的返回值，默认值只是兜底。
        """

        total_frames = 0
        for seg, start_s in zip(outs, starts, strict=True):
            seg_frames = seg.shape[1]
            offset_frames = round(start_s / _TARGET_SR * frame_rate)
            end_frames = offset_frames + seg_frames
            total_frames = max(total_frames, end_frames)

        feat = outs[0].shape[-1]
        batch = outs[0].shape[0]
        accum = torch.zeros(batch, total_frames, feat, dtype=outs[0].dtype, device=outs[0].device)
        count = torch.zeros(batch, total_frames, 1, dtype=outs[0].dtype, device=outs[0].device)
        for seg, start_s in zip(outs, starts, strict=True):
            offset = round(start_s / _TARGET_SR * frame_rate)
            n = seg.shape[1]
            accum[:, offset : offset + n, :] += seg
            count[:, offset : offset + n, :] += 1.0
        count = count.clamp(min=1.0)
        return accum / count

    # ── Backbone 加载 ─────────────────────────────────────────

    def _load_backbone(self) -> nn.Module:
        """按 source 优先级加载冻结 MERT 主干。ModelScope → HuggingFace fallback。"""
        last_exc: Exception | None = None
        sources = self._source_order()
        for src in sources:
            try:
                if src == "modelscope":
                    m = self._load_from_modelscope()
                else:
                    m = self._load_from_huggingface()
                logger.info("MERT 主干加载成功（source=%s, model=%s）", src, self.model_name)
                return m
            except Exception as exc:
                last_exc = exc
                logger.warning("MERT 加载失败 source=%s: %s", src, exc)
                continue
        raise RuntimeError(
            f"无法从 {sources} 加载 MERT {self.model_name}: {last_exc}"
        ) from last_exc

    def _source_order(self) -> list[str]:
        if self.source == "huggingface":
            return ["huggingface", "modelscope"]
        return ["modelscope", "huggingface"]

    def _load_from_huggingface(self) -> nn.Module:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if not isinstance(model, nn.Module):
            raise TypeError(f"AutoModel 返回非 nn.Module: {type(model).__name__}")
        return model

    def _load_from_modelscope(self) -> nn.Module:
        try:
            from modelscope import snapshot_download
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("modelscope 未安装") from exc

        local_dir = snapshot_download(self.model_name)
        model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
        if not isinstance(model, nn.Module):
            raise TypeError(f"AutoModel 返回非 nn.Module: {type(model).__name__}")
        return model

    def _load_feat_processor(self) -> _FeatProcessor:
        """加载音频特征提取器（Wav2Vec2 风格，做归一化）。"""
        from transformers import AutoFeatureExtractor

        try:
            return AutoFeatureExtractor.from_pretrained(self.model_name)  # type: ignore[no-untyped-call,no-any-return]
        except Exception:
            # ModelScope 本地路径兜底
            try:
                from modelscope import snapshot_download

                local_dir = snapshot_download(self.model_name)
                return AutoFeatureExtractor.from_pretrained(local_dir)  # type: ignore[no-untyped-call,no-any-return]
            except Exception as exc:
                logger.warning("特征提取器加载失败，回退裸 waveform: %s", exc)
                return _DummyProcessor()

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    # ── Adapter 构建 ──────────────────────────────────────────

    def _build_adapter(
        self,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> nn.Module | None:
        if self.adapter == "none":
            return None
        if self.adapter == "lora":
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)
            return None  # LoRA 参数挂在 backbone 内
        if self.adapter == "mlp":
            return _MLPAdapter(MERT_DEFAULT_FEAT_DIM)
        return None

    def _apply_lora(self, rank: int, alpha: int, dropout: float) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft 未安装，无法用 LoRA Adapter") from exc

        # MERT (Wav2Vec2 系) attention 模块名为 query/value；若不命中则用 auto detection。
        target_modules = ["query", "value"]
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, cfg)  # type: ignore[arg-type]
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info("LoRA 注入完成：可训参数 %d（rank=%d）", trainable, rank)

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def adapter_params(self) -> list[nn.Parameter]:
        """返回可训 Adapter 参数（供优化器只用这些参数）。"""
        return [p for p in self.parameters() if p.requires_grad]


class _MLPAdapter(nn.Module):
    """2 层 MLP 残差 Adapter（adapter="mlp"，备选，RFC-0003）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class _FeatProcessor(Protocol):
    """MERT 音频特征提取器协议（transformers AutoFeatureExtractor / _DummyProcessor）。"""

    def __call__(
        self, wav: torch.Tensor, sampling_rate: int, return_tensors: str
    ) -> dict[str, torch.Tensor]: ...


class _DummyProcessor:
    """特征提取器加载失败时的兜底：直接返回原波形（MERT 仍可吃裸 input_values）。"""

    def __call__(
        self, wav: torch.Tensor, sampling_rate: int, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        return {"input_values": wav}