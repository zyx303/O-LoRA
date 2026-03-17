#!/usr/bin/env python3
"""Interpolate two PEFT adapter checkpoints and save a new adapter directory."""

import argparse
import json
import shutil
from pathlib import Path

import torch


COMPANION_FILES = (
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "generation_config.json",
    "spiece.model",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate two adapter checkpoints with alpha * A + (1 - alpha) * B."
    )
    parser.add_argument("--adapter-a", required=True, help="Path to adapter A or its parent output directory.")
    parser.add_argument("--adapter-b", required=True, help="Path to adapter B or its parent output directory.")
    parser.add_argument("--alpha", required=True, type=float, help="Weight on adapter A.")
    parser.add_argument("--output-dir", required=True, help="Directory for the interpolated adapter.")
    parser.add_argument(
        "--reference",
        choices=("max-rsum", "a", "b"),
        default="max-rsum",
        help="Which checkpoint supplies the output adapter config and tokenizer files.",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=("zeros", "error"),
        default="zeros",
        help="How to handle keys that exist in only one checkpoint.",
    )
    parser.add_argument(
        "--shape-strategy",
        choices=("pad", "error"),
        default="error",
        help="How to handle tensor shape mismatches such as growing r_sum across tasks.",
    )
    parser.add_argument(
        "--metadata-strategy",
        choices=("reference", "error"),
        default="reference",
        help="How to handle non-tensor metadata mismatches such as task IDs in prompt-based adapters.",
    )
    return parser.parse_args()


def resolve_adapter_dir(path_str):
    path = Path(path_str).expanduser()
    if (path / "adapter_config.json").is_file():
        return path
    if (path / "adapter" / "adapter_config.json").is_file():
        return path / "adapter"
    raise FileNotFoundError(f"Could not find adapter_config.json under {path}")


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def validate_configs(config_a, config_b):
    comparable_fields = (
        "peft_type",
        "task_type",
        "base_model_name_or_path",
        "target_modules",
        "bias",
    )
    mismatches = []
    for field in comparable_fields:
        if config_a.get(field) != config_b.get(field):
            mismatches.append(f"{field}: {config_a.get(field)!r} != {config_b.get(field)!r}")
    if mismatches:
        mismatch_text = "\n".join(mismatches)
        raise ValueError(f"Adapter configs are incompatible for interpolation:\n{mismatch_text}")


def select_reference(config_a, config_b, preference):
    if preference == "a":
        return "a"
    if preference == "b":
        return "b"
    r_sum_a = int(config_a.get("r_sum") or 0)
    r_sum_b = int(config_b.get("r_sum") or 0)
    return "b" if r_sum_b > r_sum_a else "a"


def pad_tensor(tensor, target_shape):
    padded = torch.zeros(target_shape, dtype=tensor.dtype)
    slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded[slices] = tensor
    return padded


def align_tensors(name, tensor_a, tensor_b, shape_strategy):
    if tensor_a.shape == tensor_b.shape:
        return tensor_a, tensor_b
    if shape_strategy == "error":
        raise ValueError(f"Shape mismatch for {name}: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")
    if tensor_a.ndim != tensor_b.ndim:
        raise ValueError(f"Rank mismatch for {name}: {tensor_a.ndim} vs {tensor_b.ndim}")
    target_shape = tuple(max(dim_a, dim_b) for dim_a, dim_b in zip(tensor_a.shape, tensor_b.shape))
    return pad_tensor(tensor_a, target_shape), pad_tensor(tensor_b, target_shape)


def blend_tensors(name, tensor_a, tensor_b, alpha, shape_strategy, target_dtype):
    tensor_a = tensor_a.detach().cpu()
    tensor_b = tensor_b.detach().cpu()
    tensor_a, tensor_b = align_tensors(name, tensor_a, tensor_b, shape_strategy)
    blended = alpha * tensor_a.to(torch.float32) + (1.0 - alpha) * tensor_b.to(torch.float32)
    return blended.to(target_dtype)


def interpolate_state_dict(
    state_a,
    state_b,
    alpha,
    reference_name,
    missing_strategy,
    shape_strategy,
    metadata_strategy,
):
    output_state = {}
    all_keys = sorted(set(state_a.keys()) | set(state_b.keys()))

    for key in all_keys:
        value_a = state_a.get(key)
        value_b = state_b.get(key)

        if value_a is None or value_b is None:
            if missing_strategy == "error":
                raise KeyError(f"Key {key} exists in only one checkpoint")
            existing = value_a if value_a is not None else value_b
            if not torch.is_tensor(existing):
                output_state[key] = existing
                continue
            zeros = torch.zeros_like(existing)
            value_a = zeros if value_a is None else value_a
            value_b = zeros if value_b is None else value_b

        if torch.is_tensor(value_a) and torch.is_tensor(value_b):
            if reference_name == "a":
                target_dtype = value_a.dtype
            else:
                target_dtype = value_b.dtype
            output_state[key] = blend_tensors(key, value_a, value_b, alpha, shape_strategy, target_dtype)
            continue

        if type(value_a) is not type(value_b):
            if metadata_strategy == "reference":
                output_state[key] = value_a if reference_name == "a" else value_b
                continue
            raise TypeError(f"Non-tensor type mismatch for {key}: {type(value_a)} vs {type(value_b)}")
        if value_a != value_b:
            if metadata_strategy == "reference":
                output_state[key] = value_a if reference_name == "a" else value_b
                continue
            raise ValueError(f"Non-tensor value mismatch for {key}: {value_a!r} vs {value_b!r}")
        output_state[key] = value_a

    return output_state


def infer_r_sum(state_dict):
    for key, value in state_dict.items():
        if "lora_A" in key and torch.is_tensor(value):
            return int(value.shape[0])
    return None


def copy_companion_files(source_dir, target_dir):
    for filename in COMPANION_FILES:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    adapter_dir_a = resolve_adapter_dir(args.adapter_a)
    adapter_dir_b = resolve_adapter_dir(args.adapter_b)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_a = load_json(adapter_dir_a / "adapter_config.json")
    config_b = load_json(adapter_dir_b / "adapter_config.json")
    validate_configs(config_a, config_b)

    reference_name = select_reference(config_a, config_b, args.reference)
    reference_dir = adapter_dir_a if reference_name == "a" else adapter_dir_b
    reference_config = dict(config_a if reference_name == "a" else config_b)

    state_a = torch.load(adapter_dir_a / "adapter_model.bin", map_location="cpu")
    state_b = torch.load(adapter_dir_b / "adapter_model.bin", map_location="cpu")
    output_state = interpolate_state_dict(
        state_a=state_a,
        state_b=state_b,
        alpha=args.alpha,
        reference_name=reference_name,
        missing_strategy=args.missing_strategy,
        shape_strategy=args.shape_strategy,
        metadata_strategy=args.metadata_strategy,
    )

    inferred_r_sum = infer_r_sum(output_state)
    if inferred_r_sum is not None and "r_sum" in reference_config:
        reference_config["r_sum"] = inferred_r_sum

    with (output_dir / "adapter_config.json").open("w") as handle:
        json.dump(reference_config, handle, indent=2, sort_keys=True)
    torch.save(output_state, output_dir / "adapter_model.bin")
    copy_companion_files(reference_dir, output_dir)

    metadata = {
        "alpha_weight_on_adapter_a": args.alpha,
        "adapter_a": str(adapter_dir_a),
        "adapter_b": str(adapter_dir_b),
        "output_dir": str(output_dir),
        "reference_checkpoint": reference_name,
        "missing_strategy": args.missing_strategy,
        "shape_strategy": args.shape_strategy,
        "metadata_strategy": args.metadata_strategy,
        "inferred_r_sum": inferred_r_sum,
    }
    with (output_dir / "interpolation_meta.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
