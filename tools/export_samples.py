#!/usr/bin/env python3
"""Export unbatched legacy Plinder examples for browser inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckdock", type=Path, required=True)
    parser.add_argument("--plinder-data", type=Path, required=True)
    parser.add_argument("--plinder-graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", nargs="+", type=int, default=(19206, 11668, 424))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    sys.path.insert(0, str(args.ckdock))
    sys.path.insert(0, str(args.ckdock / "docking_viz"))
    from docking.data import PlinderSource
    from docking.schema import AtomRole
    from export_instantaneous import display_topology, residue_ids

    source = PlinderSource(args.plinder_data, args.plinder_graphs, 8192)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = []
    labels = ("Compact pocket", "Medium pocket", "Large pocket")
    for label, record_index in zip(labels, args.records, strict=True):
        record = source.load(record_index)
        atoms = len(record.coords)
        degree = np.zeros(atoms, dtype=np.int32)
        neighbors = np.full((atoms, 10, 2), -1, dtype=np.int32)
        for left, right, kind in zip(
            record.bond_sources, record.bond_targets, record.bond_types, strict=True
        ):
            slot = int(degree[int(right)])
            if slot >= 10:
                raise ValueError("sample graph exceeds ten directed neighbors")
            neighbors[int(right), slot] = (int(left), int(kind))
            degree[int(right)] += 1
        molecule = record.roles == int(AtomRole.MOLECULE)
        sidechain = record.roles == int(AtomRole.PROTEIN_SIDECHAIN)
        ligand = record.roles == int(AtomRole.LIGAND)
        design = molecule | sidechain | ligand
        scales = np.zeros(atoms, dtype=np.float32)
        scales[molecule | ligand] = 1.0
        scales[sidechain] = 0.5
        residues = residue_ids(record)
        payload = {
            "format": "wsfmdock_webgpu_sample_v1",
            "id": f"plinder-{record_index}",
            "label": f"{label} / Plinder {record_index}",
            "source": "plinder",
            "source_index": record_index,
            "atoms": atoms,
            "target_coords": record.coords.astype(float).tolist(),
            "base_means": record.base_means.astype(float).tolist(),
            "base_scales": scales.astype(float).tolist(),
            "atomic_numbers": record.atomic_numbers.astype(int).tolist(),
            "roles": record.roles.astype(int).tolist(),
            "residue_types": record.residue_types.astype(int).tolist(),
            "atom_names": record.atom_names.astype(int).tolist(),
            "entity_ids": record.entity_ids.astype(int).tolist(),
            "coordinate_design": design.astype(int).tolist(),
            "neighbors": neighbors.reshape(-1, 2).astype(int).tolist(),
            "residue_ids": residues.astype(int).tolist(),
            **display_topology(record, residues),
        }
        filename = f"plinder-{record_index}.json"
        (args.output / filename).write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        catalog.append({"id": payload["id"], "label": payload["label"], "atoms": atoms, "file": filename})
    (args.output / "catalog.json").write_text(json.dumps({"samples": catalog}, indent=2) + "\n")
    print(json.dumps(catalog, indent=2))


if __name__ == "__main__":
    main()

