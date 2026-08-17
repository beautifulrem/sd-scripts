"""Run a reproducible CUDA pixel-vs-latent LLLite training A/B.

The same config, dataset, seed, step count, and extra arguments are used for
both arms. Only ``--lllite_cond_input`` and output naming differ.
"""

import argparse
import json
import subprocess
import time
import tomllib
from pathlib import Path

import torch


def build_command(args, input_space: str) -> list[str]:
    output_dir = Path(args.output_root) / input_space
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    return [
        args.accelerate,
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "anima_train_control_net_lllite.py",
        "--config_file",
        str(Path(args.config_file).resolve()),
        "--seed",
        str(args.seed),
        "--max_train_steps",
        str(args.max_train_steps),
        "--output_dir",
        str(output_dir.resolve()),
        "--output_name",
        f"lllite_{input_space}",
        "--lllite_cond_input",
        input_space,
        *extra_args,
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", required=True, help="shared training TOML")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--accelerate", default="accelerate", help="accelerate executable")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def validate_shared_config(path: Path) -> None:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    values = {}
    for key, value in config.items():
        if isinstance(value, dict):
            values.update(value)
        else:
            values[key] = value
    if values.get("max_train_epochs") is not None:
        raise ValueError(
            "remove max_train_epochs from the shared config; it overrides max_train_steps and breaks fixed-step A/B parity"
        )


def main():
    args = parse_args()
    config_path = Path(args.config_file).resolve()
    if not args.dry_run:
        validate_shared_config(config_path)
    commands = {space: build_command(args, space) for space in ("pixel", "latent")}
    if args.dry_run:
        for space, command in commands.items():
            print(space, subprocess.list2cmdline(command))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LLLite conditioning A/B")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for space, command in commands.items():
        started = time.perf_counter()
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - started
        results[space] = {"returncode": completed.returncode, "wall_seconds": elapsed, "command": command}
        if completed.returncode:
            break

    manifest = {
        "cuda_device": torch.cuda.get_device_name(),
        "seed": args.seed,
        "max_train_steps": args.max_train_steps,
        "config_file": str(Path(args.config_file).resolve()),
        "results": results,
    }
    with (output_root / "ab_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    if len(results) != 2 or any(result["returncode"] for result in results.values()):
        raise SystemExit("A/B did not complete; inspect the arm logs and ab_manifest.json")


if __name__ == "__main__":
    main()
