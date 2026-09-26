"""经验验证 MERT-v1-330M 的输出帧率（POSTMORTEM-2026-08-05 的收尾证据）。

只用 torch 即可运行（不需要 transformers）：直接按官方 config 的 conv_kernel/
conv_stride 搭出特征提取器卷积栈，实测 5s/24kHz 输入产出多少帧；并打开真实
checkpoint 交叉核对各层卷积核形状。可选：若装了 transformers，再做一次真实
前向（`--forward`）。

用法::

    python scripts/verify_mert_frame_rate.py --model-dir models/pretrained/m-a-p/MERT-v1-330M
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

DEFAULT_MODEL_DIR = Path("models/pretrained/m-a-p/MERT-v1-330M")


def conv_out_len(length: int, kernel: int, stride: int) -> int:
    """Conv1d(padding=0, dilation=1) 输出长度 = floor((L-k)/s)+1。"""
    return (length - kernel) // stride + 1


def geometry_check(model_dir: Path) -> float:
    """按 config 搭卷积栈实测帧率；返回 5s 窗口的帧数。"""
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    kernels = tuple(cfg["conv_kernel"])
    strides = tuple(cfg["conv_stride"])
    sample_rate = 24000
    product = math.prod(strides)

    print(f"[config] conv_kernel = {list(kernels)}")
    print(f"[config] conv_stride = {list(strides)}  -> prod = {product}")
    print(f"[config] hidden_size = {cfg['hidden_size']}  layers = {cfg.get('num_hidden_layers')}")
    print(f"[derive] frame rate  = {sample_rate}/{product} = {sample_rate / product} Hz")

    layers: list[nn.Module] = []
    in_ch = 1
    for k, s in zip(kernels, strides, strict=True):
        layers.append(nn.Conv1d(in_ch, 512, k, stride=s, bias=bool(cfg.get("conv_bias", False))))
        in_ch = 512
    net = nn.Sequential(*layers)

    for duration_s in (1.0, 2.0, 5.0):
        x = torch.zeros(1, 1, int(duration_s * sample_rate))
        with torch.no_grad():
            frames = net(x).shape[-1]
        print(
            f"[torch ] {duration_s:>4.1f}s -> {frames:>5d} frames "
            f"= {frames / duration_s:.2f} Hz"
        )
    with torch.no_grad():
        return net(torch.zeros(1, 1, 5 * sample_rate)).shape[-1]


def checkpoint_cross_check(model_dir: Path) -> None:
    """打开真实 checkpoint，核对卷积核形状与隐藏维。"""
    ckpt = model_dir / "pytorch_model.bin"
    if not ckpt.exists():
        print(f"[ckpt  ] 跳过：{ckpt} 不存在")
        return
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    keys = list(sd.keys())
    print(f"[ckpt  ] {ckpt.name}: {len(keys)} tensors")
    conv_keys = [k for k in keys if "conv" in k and k.endswith("weight")][:8]
    for k in conv_keys:
        print(f"[ckpt  ] {k}: {tuple(sd[k].shape)}")
    attn = [k for k in keys if k.endswith("k_proj.weight")]
    if attn:
        print(f"[ckpt  ] {attn[0]}: {tuple(sd[attn[0]].shape)}  (hidden = {sd[attn[0]].shape[0]})")


def optional_forward(model_dir: Path) -> None:
    """装了 transformers 时做真实前向（帧率最终判据）。"""
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError:
        print("[forward] 跳过：未安装 transformers")
        return
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    proc = AutoFeatureExtractor.from_pretrained(str(model_dir))
    wav = torch.randn(24000 * 5)
    inputs = proc(wav, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs["input_values"], output_hidden_states=True)
    hs = out.hidden_states
    print(f"[forward] {len(hs)} hidden states; layer12 = {tuple(hs[12].shape)}")
    print(f"[forward] => {hs[12].shape[1] / 5.0:.2f} Hz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--forward", action="store_true", help="额外做真实前向（需 transformers）")
    args = ap.parse_args()

    if not (args.model_dir / "config.json").exists():
        raise SystemExit(f"config.json 不存在：{args.model_dir}")
    frames = geometry_check(args.model_dir)
    print(f"[assert] 5s 窗口应为 374 帧（75Hz），实测 {frames} 帧 -> {'OK' if frames == 374 else 'MISMATCH'}")
    checkpoint_cross_check(args.model_dir)
    if args.forward:
        optional_forward(args.model_dir)


if __name__ == "__main__":
    main()
