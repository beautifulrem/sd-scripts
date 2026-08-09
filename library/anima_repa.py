"""Shared validation helpers for Anima REPA feature sidecars."""

from __future__ import annotations


_INTERPOLATIONS = (None, "lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area", "box")


def repa_interpolation_code(value) -> int:
    try:
        return _INTERPOLATIONS.index(value)
    except ValueError as error:
        raise ValueError(f"unsupported REPA resize interpolation: {value!r}") from error


def validate_repa_sidecar(
    feature_dict: dict,
    feature_path,
    feature_key: str,
    expected_bucket_size,
    expected_resize_interpolation=None,
) -> None:
    if feature_key not in feature_dict:
        raise KeyError(f"REPA sidecar {feature_path} has no key {feature_key!r}")
    cached_bucket_size = feature_dict.get("anima_bucket_size")
    if cached_bucket_size is None:
        raise ValueError(f"legacy REPA sidecar has no bucket metadata; regenerate it: {feature_path}")
    cached_bucket_size = tuple(int(value) for value in cached_bucket_size.tolist())
    expected_bucket_size = tuple(int(value) for value in expected_bucket_size)
    if cached_bucket_size != expected_bucket_size:
        raise ValueError(
            f"REPA sidecar {feature_path} was cached for bucket {cached_bucket_size}, "
            f"but training selected {expected_bucket_size}; regenerate it with matching "
            "target_pixels, resolution_step and max_bucket_reso"
        )
    cached_interpolation = feature_dict.get("anima_resize_interpolation")
    if cached_interpolation is None:
        raise ValueError(f"legacy REPA sidecar has no resize interpolation metadata; regenerate it: {feature_path}")
    cached_interpolation = int(cached_interpolation.item())
    expected_interpolation = repa_interpolation_code(expected_resize_interpolation)
    if cached_interpolation != expected_interpolation:
        raise ValueError(
            f"REPA sidecar {feature_path} uses resize interpolation code {cached_interpolation}, "
            f"but training expects {expected_interpolation}; regenerate it with matching resize_interpolation"
        )
