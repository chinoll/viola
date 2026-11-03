"""Small YAML/dict configuration for violaEngine."""

import copy
import os
from pathlib import Path

import torch

from ..precision import settings


def dtype(value):
    if value is None or isinstance(value, torch.dtype):
        return value
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[value]


def read_config(config=None):
    config = config if config is not None else os.environ.get("VIOLA_CONFIG")
    if isinstance(config, (str, Path)):
        import yaml
        with open(config) as stream:
            config = yaml.safe_load(stream)
    supplied = copy.deepcopy(config or {})
    policy = settings()
    result = dict(
        num_processes=1,
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        sharding=dict(parameters=False, gradients=False, optimizer=True, ema=True, blocks=None),
        precision=dict(master_weight_dtype=policy["master_weight_dtype"],
                       gradient_dtype=policy["gradient_dtype"] or torch.float32,
                       master_rounding=policy["master_rounding"], master_rounding_seed=policy["master_rounding_seed"]),
        communication=dict(dtype=None, bucket_mb=25, fp8_format="e4m3", fp8_tile_size=256),
        ema=dict(decay=0, dtype="fp32"),
    )
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result
