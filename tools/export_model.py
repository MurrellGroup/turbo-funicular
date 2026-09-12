#!/usr/bin/env python3
"""Export browser tensors from a supported checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32",), default="float32")
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
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite exported tensor: {name}")
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
    expected = "continuous_docking_backbone_segments_alldepth_secant_x1_ck_v8"
    if checkpoint.get("method") != expected or checkpoint.get("stage") != "ck":
        raise ValueError("unsupported checkpoint method or stage")
    state = checkpoint["ema"]
    config = checkpoint["model_config"]
    required = dict(dim=408, depth=12, heads=12, scalar_head_dim=34,
                    query_points=6, point_values=10, ff_hidden_dim=2040,
                    endpoint_update_layers=3, endpoint_update_stride=2, ck_suffix_layers=5,
                    a_e=1.0, rope_base=1000.0)
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported {key}: {config.get(key)}")
    if checkpoint.get('dataset_signature', {}).get('molecule_initial_std') != 10.0:
        raise ValueError('unsupported initial distribution')
    if checkpoint.get("lateral_width") != 1024:
        raise ValueError("unsupported lateral width")
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
    writer.add("local.sigma_embedding", state["local.sigma_embedding.weight"])
    for index in range(int(config["depth"])):
        export_block(writer, state, f"local.blocks.{index}", f"local.blocks.{index}", int(config["heads"]))
    for index in range(int(config["endpoint_update_layers"])):
        writer.add(f"local.endpoint_updates.{index}", state[f"local.endpoint_updates.{index}.weight"])
    writer.add("finite_time_embedding", state["finite_time_embedding.weight"])
    writer.add("noise_encoder", state["noise_encoder.weight"])
    for index in range(int(config["ck_suffix_layers"])):
        export_block(writer, state, f"finite_blocks.{index}", f"finite.blocks.{index}", int(config["heads"]))
        writer.add(f"finite.lateral.outputs.{index}", state[f"all_depth_lateral.outputs.{index}.weight"])
    for index in range(int(config["depth"])):
        writer.add(f"finite.lateral.projections.{index}", state[f"all_depth_lateral.projections.{index}.weight"])
    writer.add("finite.lateral.mixing", state["all_depth_lateral.mixing"])
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
        "format": "wsfmdock_webgpu_v8",
        "method": checkpoint["method"],
        "stage": checkpoint["stage"],
        "iteration": int(checkpoint["iteration"]),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "weight_file": weights_path.name,
        "weight_bytes": len(writer.payload),
        "weight_sha256": hashlib.sha256(writer.payload).hexdigest(),
        "weight_precision": args.precision,
        "activation_precision": args.precision,
        "accumulation_precision": "float32",
        "config": config,
        "embedding_offsets": {"atomic": 0, "role": 128, "residue": 133, "atom_name": 154},
        "endpoint_block_indices": [7, 9, 11],
        "finite_start_block": 7,
        "lateral_width": checkpoint["lateral_width"],
        "sampling": {"initial_molecule_scale": 10.0, "process_molecule_scale": 1.0,
                     "sidechain_scale": 0.5},
        "time_frequencies": state["local.time_features.weight"].tolist(),
        "sigma_frequencies": state["local.sigma_features.weight"].tolist(),
        "tensors": writer.entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("format", "iteration", "weight_bytes", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
