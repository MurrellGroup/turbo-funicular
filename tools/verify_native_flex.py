#!/usr/bin/env python3
"""Verify the dense parity fixture against native PyTorch FlexAttention."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckdock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    sys.path.insert(0, str(args.ckdock))
    from docking.model import EndpointDockingCKModel, ModelConfig, make_bond_pair_features
    from docking.schema import AtomRole

    sample = json.loads(args.sample.read_text())
    fixture = json.loads(args.fixture.read_text())
    n = int(sample["atoms"])
    length = math.ceil(n / 128) * 128
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = EndpointDockingCKModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval().requires_grad_(False)

    def padded(values, width=None, dtype=np.float32, fill=0):
        shape = (length,) if width is None else (length, width)
        result = np.full(shape, fill, dtype=dtype)
        source = np.asarray(values, dtype=dtype).reshape((n,) if width is None else (n, width))
        result[:n] = source
        return torch.as_tensor(result, device=device).unsqueeze(0)

    coords = padded(fixture["coords"], 3)
    increment = padded(fixture["increment"], 3)
    latent = padded(fixture["latent"], 3)
    means = padded(sample["base_means"], 3)
    scales = padded(sample["base_scales"])
    design = padded(sample["coordinate_design"], dtype=np.bool_)
    valid = torch.zeros((1, length), dtype=torch.bool, device=device)
    valid[:, :n] = True
    documents = torch.ones((1, length), dtype=torch.int64, device=device)
    documents[:, :n] = 0
    entities = padded(sample["entity_ids"], dtype=np.int64, fill=int(max(sample["entity_ids"])) + 1)
    metadata = {
        "atomic_numbers": padded(sample["atomic_numbers"], dtype=np.int64),
        "roles": padded(sample["roles"], dtype=np.int64, fill=int(AtomRole.PADDING)),
        "residue_types": padded(sample["residue_types"], dtype=np.int64),
        "atom_names": padded(sample["atom_names"], dtype=np.int64),
    }
    neighbor_pairs = np.asarray(sample["neighbors"], dtype=np.int64).reshape(n, 10, 2)
    indices = np.zeros((length, 10), dtype=np.int64)
    kinds = np.zeros((length, 10), dtype=np.int64)
    edge_valid = np.zeros((length, 10), dtype=np.bool_)
    active = neighbor_pairs[..., 0] >= 0
    indices[:n] = np.where(active, neighbor_pairs[..., 0], 0)
    kinds[:n] = np.where(active, neighbor_pairs[..., 1], 0)
    edge_valid[:n] = active
    bond_features = make_bond_pair_features(
        torch.as_tensor(indices, device=device).unsqueeze(0),
        torch.as_tensor(kinds, device=device).unsqueeze(0),
        torch.as_tensor(edge_valid, device=device).unsqueeze(0),
    )
    start = torch.full((1, length), float(fixture["start"]), device=device)
    end = torch.full((1, length), float(fixture["end"]), device=device)
    block_mask = model.local.make_block_mask(documents)
    with torch.no_grad():
        result = model(
            start,
            end,
            coords,
            increment,
            latent,
            metadata["atomic_numbers"],
            metadata["roles"],
            metadata["residue_types"],
            metadata["atom_names"],
            documents,
            torch.arange(length, device=device).unsqueeze(0),
            entities,
            bond_features,
            design,
            means,
            scales,
            valid,
            block_mask,
        )
    expected_coords = torch.tensor(fixture["expected_coords"]).view(n, 3)
    expected_secant = torch.tensor(fixture["expected_secant"]).view(n, 3)
    actual_coords = result.coords[0, :n].cpu()
    actual_secant = result.secant_endpoint[0, :n].cpu()

    def metrics(actual, expected):
        difference = actual - expected
        return {
            "rms": float(difference.square().mean().sqrt()),
            "maximum": float(difference.abs().max()),
        }

    report = {
        "atoms": n,
        "padded_atoms": length,
        "coordinates": metrics(actual_coords, expected_coords),
        "secant": metrics(actual_secant, expected_secant),
    }
    print(json.dumps(report, indent=2))
    if report["coordinates"]["rms"] >= 2e-5 or report["secant"]["rms"] >= 1e-4:
        raise SystemExit("native FlexAttention differs from dense fixture")


if __name__ == "__main__":
    main()

