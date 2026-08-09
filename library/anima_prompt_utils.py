"""Prompt-file parsing shared by the Anima training entry points."""

import json
import logging
import re
from typing import Dict, List

import toml


logger = logging.getLogger(__name__)


def _line_to_prompt_dict(line: str) -> dict:
    prompt_args = line.split(" --")
    prompt_dict = {"prompt": prompt_args[0]}
    patterns = (
        (r"w (\d+)", "width", int),
        (r"h (\d+)", "height", int),
        (r"d (\d+)", "seed", int),
        (r"l ([\d.]+)", "scale", float),
        (r"n (.+)", "negative_prompt", str),
        (r"cn (.+)", "controlnet_image", str),
        (r"mk (.+)", "mask_image", str),
        (r"fs (.+)", "flow_shift", float),
        (r"am ([\d.,-]+)", "additional_network_multiplier", lambda value: [float(v) for v in value.split(",")]),
    )
    for argument in prompt_args[1:]:
        try:
            steps = re.fullmatch(r"s (\d+)", argument, re.IGNORECASE)
            if steps:
                prompt_dict["sample_steps"] = max(1, min(1000, int(steps.group(1))))
                continue
            for pattern, key, converter in patterns:
                match = re.fullmatch(pattern, argument, re.IGNORECASE)
                if match:
                    prompt_dict[key] = converter(match.group(1).strip())
                    break
        except ValueError as error:
            logger.error("Exception in parsing / 解析エラー: %s", argument)
            logger.error(error)
    return prompt_dict


def load_prompts(prompt_file: str) -> List[Dict]:
    """Load Anima sample prompts from txt, TOML, or JSON."""

    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as file:
            data = toml.load(file)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompts = json.load(file)
    else:
        raise ValueError(f"Unsupported prompt file format: {prompt_file}")

    for index, prompt in enumerate(prompts):
        if isinstance(prompt, str):
            prompt = _line_to_prompt_dict(prompt)
            prompts[index] = prompt
        if not isinstance(prompt, dict):
            raise TypeError(f"Prompt entry {index} must be a string or dictionary")
        prompt["enum"] = index
        prompt.pop("subset", None)
    return prompts
