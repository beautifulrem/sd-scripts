"""Cache patch-token vision features for Anima relational REPA training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, AutoModel

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def free_fit_size(
    width: int,
    height: int,
    target_pixels: int,
    step: int,
    max_size: int = 2048,
) -> tuple[int, int]:
    """Mirror ``BucketManager(free_fit=True)`` exactly for cache alignment."""
    aspect = width / height
    target_width = math.sqrt(target_pixels * aspect)
    target_height = target_pixels / target_width

    def round_to_step(value: float) -> int:
        rounded = int(value + 0.5)
        return max(step, rounded - rounded % step)

    width_candidate = round_to_step(target_width)
    candidates = [(width_candidate, round_to_step(width_candidate / aspect))]
    height_candidate = round_to_step(target_height)
    candidates.append((round_to_step(height_candidate * aspect), height_candidate))
    candidates = [(w, h) for w, h in candidates if w <= max_size and h <= max_size]
    if not candidates:
        scale = max_size / max(target_width, target_height)
        candidates = [(round_to_step(target_width * scale), round_to_step(target_height * scale))]
    return min(
        candidates,
        key=lambda size: (abs(size[0] / size[1] - aspect), abs(size[0] * size[1] - target_pixels)),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--vision_model", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="_anima_pe_spatial.safetensors")
    parser.add_argument("--feature_key", type=str, default="image_features")
    parser.add_argument("--target_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--resolution_step", type=int, default=16)
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=2048,
        help="Must match max_bucket_reso in the training dataset configuration.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    processor = AutoImageProcessor.from_pretrained(args.vision_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.vision_model, trust_remote_code=True, torch_dtype=dtype)
    model.eval().to(args.device)
    paths = sorted(path for path in args.image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no images under {args.image_dir}")

    for index, path in enumerate(paths, 1):
        output_path = path.with_name(path.stem + args.suffix)
        image = Image.open(path).convert("RGB")
        bucket_size = free_fit_size(
            image.width,
            image.height,
            args.target_pixels,
            args.resolution_step,
            args.max_bucket_reso,
        )
        if output_path.exists() and not args.overwrite:
            cached = load_file(str(output_path))
            cached_bucket_size = cached.get("anima_bucket_size")
            if cached_bucket_size is None:
                raise ValueError(f"legacy REPA sidecar has no bucket metadata; rerun with --overwrite: {output_path}")
            if tuple(cached_bucket_size.tolist()) != bucket_size or args.feature_key not in cached:
                raise ValueError(f"stale or incompatible REPA sidecar; rerun with --overwrite: {output_path}")
            continue
        image = image.resize(bucket_size, Image.Resampling.LANCZOS)
        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        inputs = {
            key: value.to(args.device, dtype=dtype if value.dtype.is_floating_point else value.dtype)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        features = getattr(outputs, "last_hidden_state", None)
        if features is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError(f"vision model {args.vision_model} exposes no patch-token hidden state")
            features = hidden_states[-1]
        save_file(
            {
                args.feature_key: features[0].float().cpu().contiguous(),
                "anima_bucket_size": torch.tensor(bucket_size, dtype=torch.int64),
            },
            str(output_path),
        )
        print(f"[{index}/{len(paths)}] {output_path}")


if __name__ == "__main__":
    main()
