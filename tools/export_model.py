#!/usr/bin/env python3
"""Convert the legacy CKDock EMA checkpoint to browser-oriented FP16 tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    return parser.parse_args()


class Writer:
    def __init__(self, precision: str) -> None:
        self.precision = precision
        self.payload = bytearray()
        self.entries: dict[str, dict[str, object]] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.entries:
            raise ValueError(f"duplicate exported tensor: {name}")
        while len(self.payload) % 4:
            self.payload.append(0)
        dtype = "<f2" if self.precision == "float16" else "<f4"
        value = tensor.detach().cpu().float().contiguous().numpy().astype(dtype)
        offset = len(self.payload)
        encoded = value.tobytes(order="C")
        self.payload.extend(encoded)
        self.entries[name] = {
            "offset": offset,
            "bytes": len(encoded),
            "shape": list(value.shape),
            "dtype": self.precision,
        }


def cat(state: dict[str, torch.Tensor], names: list[str], dim: int = 0) -> torch.Tensor:
    return torch.cat([state[name] for name in names], dim=dim)


def export_adaln(writer: Writer, state: dict[str, torch.Tensor], source: str, target: str) -> None:
    writer.add(
        f"{target}.norm",
        cat(state, [f"{source}.norm.weight", f"{source}.norm.bias"]),
    )
    writer.add(
        f"{target}.affine_weight",
        cat(state, [f"{source}.scale.weight", f"{source}.shift.weight"]),
    )
    writer.add(
        f"{target}.affine_bias",
        cat(state, [f"{source}.scale.bias", f"{source}.shift.bias"]),
    )


def export_block(
    writer: Writer,
    state: dict[str, torch.Tensor],
    source: str,
    target: str,
    heads: int,
) -> None:
    export_adaln(writer, state, f"{source}.attention_norm", f"{target}.attention_norm")
    writer.add(
        f"{target}.attention.projection",
        cat(
            state,
            [
                f"{source}.attention.qkv.weight",
                f"{source}.attention.qk_points.weight",
                f"{source}.attention.value_points.weight",
            ],
        ),
    )
    writer.add(
        f"{target}.attention.qk_norm",
        cat(
            state,
            [f"{source}.attention.q_norm.weight", f"{source}.attention.k_norm.weight"],
        ),
    )
    writer.add(f"{target}.attention.head_weights", state[f"{source}.attention.head_weights"])
    pair_bias = torch.zeros(heads, 5, dtype=torch.float32)
    pair_bias[:, 0] = state[f"{source}.attention.different_entity_bias"]
    bond_name = f"{source}.attention.bond_pair_bias"
    if bond_name in state:
        pair_bias[:, 1:] = state[bond_name]
    writer.add(f"{target}.attention.pair_bias", pair_bias)
    writer.add(f"{target}.attention.output", state[f"{source}.attention.output.weight"])
    export_adaln(writer, state, f"{source}.ffn_norm", f"{target}.ffn_norm")
    writer.add(
        f"{target}.ffn.upgate",
        cat(state, [f"{source}.ffn.up.weight", f"{source}.ffn.gate.weight"]),
    )
    writer.add(f"{target}.ffn.down", state[f"{source}.ffn.down.weight"])


def main() -> None:
    args = arguments()
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = "continuous_docking_pairgraph_secant_x1_ck_v3"
    if checkpoint.get("method") != expected or checkpoint.get("stage") != "ck":
        raise ValueError("checkpoint is not the legacy pairgraph-v3 CK model")
    state = checkpoint["ema"]
    config = checkpoint["model_config"]
    writer = Writer(args.precision)
    writer.add(
        "local.embedding",
        cat(
            state,
            [
                "local.atomic_embedding.weight",
                "local.role_embedding.weight",
                "local.residue_embedding.weight",
                "local.atom_name_embedding.weight",
            ],
        ),
    )
    writer.add("local.time_features", state["local.time_features.weight"])
    writer.add("local.time_embedding", state["local.time_embedding.weight"])
    for index in range(int(config["depth"])):
        export_block(writer, state, f"local.blocks.{index}", f"local.blocks.{index}", int(config["heads"]))
    for index in range(int(config["endpoint_update_layers"])):
        writer.add(f"local.endpoint_updates.{index}", state[f"local.endpoint_updates.{index}.weight"])
    writer.add("finite_time_embedding", state["finite_time_embedding.weight"])
    writer.add("noise_encoder", state["noise_encoder.weight"])
    for index in range(int(config["ck_suffix_layers"])):
        export_block(writer, state, f"finite_blocks.{index}", f"finite.blocks.{index}", int(config["heads"]))
        writer.add(f"finite.lateral_adapters.{index}", state[f"lateral_adapters.{index}.weight"])
    for index in range(int(config["endpoint_update_layers"])):
        writer.add(
            f"finite.endpoint_updates.{index}",
            state[f"finite_endpoint_updates.{index}.weight"],
        )

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "f16" if args.precision == "float16" else "f32"
    weights_path = args.output / f"weights.{suffix}"
    weights_path.write_bytes(writer.payload)
    manifest = {
        "format": "wsfmdock_webgpu_v1",
        "method": checkpoint["method"],
        "stage": checkpoint["stage"],
        "iteration": int(checkpoint["iteration"]),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "weight_file": weights_path.name,
        "weight_bytes": len(writer.payload),
        "weight_precision": args.precision,
        "activation_precision": args.precision,
        "accumulation_precision": "float32",
        "config": config,
        "embedding_offsets": {"atomic": 0, "role": 128, "residue": 133, "atom_name": 154},
        "endpoint_block_indices": [7, 9, 11],
        "finite_start_block": 7,
        "legacy_pair_blocks": ["local.blocks.0"],
        "time_frequencies": state["local.time_features.weight"].tolist(),
        "time_frequencies": state["local.time_features.weight"].tolist(),
        "tensors": writer.entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("format", "iteration", "weight_bytes", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
