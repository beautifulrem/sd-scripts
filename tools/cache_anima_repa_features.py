"""Cache patch-token vision features for Anima relational REPA training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

# Support the documented ``python tools/cache_anima_repa_features.py`` form,
# where Python otherwise places only ``tools/`` on the import path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, AutoModel

from library.anima_repa import repa_interpolation_code, validate_repa_sidecar
from library.utils import trim_and_resize_if_required

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def free_fit_size(
    width: int,
    height: int,
    target_pixels: int,
    step: int,
    max_size: int = 2048,
) -> tuple[int, int]:
    """Return the bucket selected by ``BucketManager(free_fit=True)``."""
    return free_fit_geometry(width, height, target_pixels, step, max_size)[0]


def free_fit_geometry(
    width: int,
    height: int,
    target_pixels: int,
    step: int,
    max_size: int = 2048,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Mirror both bucket and aspect-preserving resize geometry used in training."""
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
    bucket_size = min(
        candidates,
        key=lambda size: (abs(size[0] / size[1] - aspect), abs(size[0] * size[1] - target_pixels)),
    )
    bucket_aspect = bucket_size[0] / bucket_size[1]
    scale = bucket_size[1] / height if aspect > bucket_aspect else bucket_size[0] / width
    resized_size = (int(width * scale + 0.5), int(height * scale + 0.5))
    return bucket_size, resized_size


def prepare_repa_image(image: Image.Image, bucket_size, resized_size, resize_interpolation=None) -> Image.Image:
    """Apply the same aspect-preserving resize and center crop as the dataset."""
    # Dataset preprocessing operates on OpenCV-style BGR arrays. Preserve that
    # convention so PIL-backed interpolation paths do not swap red and blue.
    array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    array, _, _ = trim_and_resize_if_required(
        False, array, bucket_size, resized_size, resize_interpolation=resize_interpolation
    )
    return Image.fromarray(array[:, :, ::-1], mode="RGB")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--vision_model", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="_anima_pe_spatial.safetensors")
    parser.add_argument("--feature_key", type=str, default="image_features")
    parser.add_argument("--target_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--resolution_step", type=int, default=16)
    parser.add_argument(
        "--resize_interpolation",
        choices=["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area", "box"],
        default=None,
        help="Must match resize_interpolation in the training dataset configuration.",
    )
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
        bucket_size, resized_size = free_fit_geometry(
            image.width,
            image.height,
            args.target_pixels,
            args.resolution_step,
            args.max_bucket_reso,
        )
        if output_path.exists() and not args.overwrite:
            cached = load_file(str(output_path))
            validate_repa_sidecar(
                cached, output_path, args.feature_key, bucket_size, args.resize_interpolation
            )
            with safe_open(str(output_path), framework="pt", device="cpu") as checkpoint:
                metadata = checkpoint.metadata() or {}
            if metadata.get("anima_vision_model") != args.vision_model:
                raise ValueError(f"REPA sidecar used a different vision model; rerun with --overwrite: {output_path}")
            continue
        image = prepare_repa_image(image, bucket_size, resized_size, args.resize_interpolation)
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
                "anima_resize_interpolation": torch.tensor(
                    repa_interpolation_code(args.resize_interpolation), dtype=torch.int64
                ),
            },
            str(output_path),
            metadata={
                "anima_vision_model": args.vision_model,
                "anima_resize_interpolation": args.resize_interpolation or "default",
            },
        )
        print(f"[{index}/{len(paths)}] {output_path}")


if __name__ == "__main__":
    main()
