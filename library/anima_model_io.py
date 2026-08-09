"""Anima checkpoint hashing and ModelSpec metadata helpers."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from io import BytesIO
from typing import Optional

import safetensors
import safetensors.torch

from library import anima_model_spec


SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def model_hash(filename: str) -> str:
    """Return the legacy short hash used in LoRA metadata."""

    try:
        with open(filename, "rb") as file:
            digest = hashlib.sha256()
            file.seek(0x100000)
            digest.update(file.read(0x10000))
            return digest.hexdigest()[:8]
    except FileNotFoundError:
        return "NOFILE"
    except (IsADirectoryError, PermissionError):
        return "IsADirectory"


def calculate_sha256(filename: str) -> str:
    try:
        digest = hashlib.sha256()
        with open(filename, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except (IsADirectoryError, PermissionError):
        return "IsADirectory"


def _additional_network_legacy_hash(file: BytesIO) -> str:
    digest = hashlib.sha256()
    file.seek(0x100000)
    digest.update(file.read(0x10000))
    return digest.hexdigest()[:8]


def _additional_network_hash(file: BytesIO) -> str:
    digest = hashlib.sha256()
    file.seek(0)
    header_size = int.from_bytes(file.read(8), "little")
    file.seek(header_size + 8)
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata) -> tuple[str, str]:
    immutable_metadata = {key: value for key, value in metadata.items() if key.startswith("ss_")}
    file = BytesIO(safetensors.torch.save(tensors, immutable_metadata))
    return _additional_network_hash(file), _additional_network_legacy_hash(file)


def save_safetensors_with_hashes(tensors, filename: str, metadata=None) -> dict[str, str]:
    """Save an Anima artifact with the standard additional-network hashes."""
    metadata = dict(metadata or {})
    model_hash, legacy_hash = precalculate_safetensors_hashes(tensors, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash
    safetensors.torch.save_file(tensors, filename, metadata)
    return metadata


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError):
        return "(unknown)"


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}
    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as file:
        return file.metadata() or {}


def build_minimum_network_metadata(
    v2: Optional[str],
    base_model: Optional[str],
    network_module: str,
    network_dim: str,
    network_alpha: str,
    network_args: Optional[dict],
) -> dict:
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def get_anima_model_spec_dataclass(
    args: argparse.Namespace,
    *,
    lora: bool,
    optional_metadata: Optional[dict[str, str]] = None,
) -> anima_model_spec.ModelSpecMetadata:
    """Build ModelSpec metadata with an explicitly Anima-only model config."""

    if getattr(args, "min_timestep", None) is not None or getattr(args, "max_timestep", None) is not None:
        timesteps = (getattr(args, "min_timestep", None) or 0, getattr(args, "max_timestep", None) or 1000)
    else:
        timesteps = None

    title = getattr(args, "metadata_title", None) or getattr(args, "output_name", None)
    return anima_model_spec.build_metadata_dataclass(
        lora=lora,
        timestamp=time.time(),
        title=title,
        reso=getattr(args, "resolution", None),
        author=getattr(args, "metadata_author", None),
        description=getattr(args, "metadata_description", None),
        license=getattr(args, "metadata_license", None),
        tags=getattr(args, "metadata_tags", None),
        timesteps=timesteps,
        optional_metadata=optional_metadata,
    )
