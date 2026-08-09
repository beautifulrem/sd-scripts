"""ModelSpec metadata support for Anima checkpoints and LoRA adapters."""

import argparse
import base64
import datetime
import mimetypes
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Union


ANIMA_ARCHITECTURE = "anima-preview"
ANIMA_IMPLEMENTATION = "https://huggingface.co/circlestone-labs/Anima"


@dataclass
class ModelSpecMetadata:
    architecture: str
    implementation: str
    title: str
    resolution: str
    sai_model_spec: str = "1.0.1"
    description: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    license: Optional[str] = None
    tags: Optional[str] = None
    timestep_range: Optional[str] = None
    additional_fields: dict[str, str] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, str]:
        metadata = {}
        for field_name, value in self.__dict__.items():
            if field_name == "additional_fields":
                for key, extra_value in value.items():
                    metadata[key if key.startswith("modelspec.") else f"modelspec.{key}"] = str(extra_value)
            elif value is not None:
                metadata[f"modelspec.{field_name}"] = str(value)
        return metadata


def _implementation_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            timeout=5,
        )
        if result.returncode == 0:
            return f"sd-scripts-anima/{result.stdout.strip()}"
    except (OSError, subprocess.SubprocessError):
        pass
    return "sd-scripts-anima/unknown"


def _resolution_string(resolution: Union[str, int, tuple[int, int], None]) -> str:
    if resolution is None:
        height, width = 1024, 1024
    elif isinstance(resolution, str):
        values = [int(value.strip()) for value in resolution.split(",")]
        height, width = (values[0], values[0]) if len(values) == 1 else values[:2]
    elif isinstance(resolution, int):
        height, width = resolution, resolution
    else:
        height, width = (resolution[0], resolution[0]) if len(resolution) == 1 else resolution[:2]
    return f"{width}x{height}"


def _file_to_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_metadata_dataclass(
    *,
    lora: bool,
    timestamp: float,
    title: Optional[str] = None,
    reso: Union[str, int, tuple[int, int], None] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[str] = None,
    merged_from: Optional[str] = None,
    timesteps: Optional[tuple[int, int]] = None,
    optional_metadata: Optional[dict] = None,
) -> ModelSpecMetadata:
    """Build metadata for an Anima checkpoint or LoRA adapter."""

    architecture = ANIMA_ARCHITECTURE + ("/lora" if lora else "")
    title = title or (("LoRA" if lora else "Checkpoint") + f"@{timestamp}")
    extras = dict(optional_metadata or {})
    thumbnail = extras.get("thumbnail")
    if thumbnail and not str(thumbnail).startswith("data:"):
        try:
            extras["thumbnail"] = _file_to_data_url(str(thumbnail))
        except OSError:
            extras.pop("thumbnail", None)
    extras.setdefault("implementation_version", _implementation_version())
    if merged_from is not None:
        extras["merged_from"] = merged_from

    timestep_range = None if timesteps is None else f"{timesteps[0]},{timesteps[-1]}"
    return ModelSpecMetadata(
        architecture=architecture,
        implementation=ANIMA_IMPLEMENTATION,
        title=title,
        resolution=_resolution_string(reso),
        description=description,
        author=author,
        date=datetime.datetime.fromtimestamp(int(timestamp)).isoformat(),
        license=license,
        tags=tags,
        timestep_range=timestep_range,
        additional_fields=extras,
    )


def add_model_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata_title", type=str, default=None, help="title stored in Anima model metadata")
    parser.add_argument("--metadata_author", type=str, default=None, help="author stored in Anima model metadata")
    parser.add_argument("--metadata_description", type=str, default=None, help="description stored in Anima model metadata")
    parser.add_argument("--metadata_license", type=str, default=None, help="license stored in Anima model metadata")
    parser.add_argument("--metadata_tags", type=str, default=None, help="comma-separated tags stored in Anima model metadata")
