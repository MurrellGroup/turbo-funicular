#!/usr/bin/env python3
"""Create an explicit-noise multi-map PyTorch parity fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckdock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090202)
    parser.add_argument("--steps", type=int, default=4)
    return parser.parse_args()


def dense_flex_attention(q, k, v, *, score_mod, **_kwargs):
    """Exact dense reference for one unpadded document."""
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    batch, heads, queries, keys = scores.shape
    batch_index = torch.arange(batch, device=q.device)[:, None, None, None]
    head_index = torch.arange(heads, device=q.device)[None, :, None, None]
    query_index = torch.arange(queries, device=q.device)[None, None, :, None]
    key_index = torch.arange(keys, device=q.device)[None, None, None, :]
    scores = score_mod(scores, batch_index, head_index, query_index, key_index)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, v)


def main() -> None:
    args = arguments()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    sys.path.insert(0, str(args.ckdock))
    import docking.model as model_module
    from docking.model import EndpointDockingCKModel, ModelConfig, make_bond_pair_features

    model_module.flex_attention = dense_flex_attention
    sample = json.loads(args.sample.read_text())
    n = int(sample["atoms"])
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = EndpointDockingCKModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval().requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    target = np.asarray(sample["target_coords"], dtype=np.float32)
    means = np.asarray(sample["base_means"], dtype=np.float32)
    scales = np.asarray(sample["base_scales"], dtype=np.float32)
    design = np.asarray(sample["coordinate_design"], dtype=np.bool_)
    coords = target.copy()
    base_noise = rng.standard_normal((n, 3), dtype=np.float32)
    coords[design] = means[design] + scales[design, None] * base_noise[design]
    initial_coords = coords.copy()

    neighbors = np.asarray(sample["neighbors"], dtype=np.int32).reshape(n, 10, 2)
    neighbor_valid = neighbors[..., 0] >= 0
    neighbor_indices = np.where(neighbor_valid, neighbors[..., 0], 0)
    neighbor_types = np.where(neighbor_valid, neighbors[..., 1], 0)

    def tensor(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype, device=device).unsqueeze(0)

    bond_features = make_bond_pair_features(
        tensor(neighbor_indices, torch.int64),
        tensor(neighbor_types, torch.int64),
        tensor(neighbor_valid, torch.bool),
    )
    static = (
        tensor(sample["atomic_numbers"], torch.int64),
        tensor(sample["roles"], torch.int64),
        tensor(sample["residue_types"], torch.int64),
        tensor(sample["atom_names"], torch.int64),
        torch.zeros((1, n), dtype=torch.int64, device=device),
        torch.arange(n, dtype=torch.int64, device=device).unsqueeze(0),
        tensor(sample["entity_ids"], torch.int64),
        bond_features,
        tensor(design, torch.bool),
        tensor(means),
        tensor(scales),
        torch.ones((1, n), dtype=torch.bool, device=device),
        None,
    )

    transitions = []
    with torch.no_grad():
        for index in range(args.steps):
            start_value = index / args.steps
            end_value = (index + 1) / args.steps
            variance = scales**2 * (end_value - start_value) * (
                2 - start_value - end_value
            )
            increment = np.sqrt(variance)[:, None] * rng.standard_normal(
                (n, 3), dtype=np.float32
            )
            latent = rng.standard_normal((n, 3), dtype=np.float32)
            increment[~design] = 0
            latent[~design] = 0
            start = torch.full((1, n), start_value, device=device)
            end = torch.full((1, n), end_value, device=device)
            result = model(
                start,
                end,
                tensor(coords),
                tensor(increment),
                tensor(latent),
                *static,
            )
            coords = result.coords[0].cpu().numpy()
            transitions.append(
                {
                    "start": start_value,
                    "end": end_value,
                    "increment": increment.reshape(-1).astype(float).tolist(),
                    "latent": latent.reshape(-1).astype(float).tolist(),
                    "expected_coords": coords.reshape(-1).astype(float).tolist(),
                    "expected_secant": result.secant_endpoint[0]
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                }
            )

    payload = {
        "format": "wsfmdock_webgpu_trajectory_parity_v1",
        "sample_file": args.sample.name,
        "seed": args.seed,
        "steps": args.steps,
        "initial_coords": initial_coords.reshape(-1).astype(float).tolist(),
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"atoms": n, "steps": args.steps, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
