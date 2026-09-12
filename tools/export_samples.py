#!/usr/bin/env python3
"""Export unbatched examples for browser inference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckdock", type=Path, required=True)
    parser.add_argument("--plinder-data", type=Path)
    parser.add_argument("--plinder-graphs", type=Path)
    parser.add_argument("--glycan-store", type=Path, action='append')
    parser.add_argument("--glycan-kind", choices=('attached', 'free'), default='attached')
    parser.add_argument("--append", action='store_true')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", nargs="+", type=int, default=(19206, 11668, 424))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    sys.path.insert(0, str(args.ckdock))
    sys.path.insert(0, str(args.ckdock / "viz"))
    from wsfmdock.data import PlinderSource, GlycanSource, GlycanCatalog
    from wsfmdock.schema import AtomRole
    from wsfmdock.protein_graph import add_protein_intraresidue_bonds
    from wsfmdock.backbone import peptide_edges, residue_groups
    from wsfmdock.sample_io import pack_repeated
    from export_instantaneous import display_topology, residue_ids

    source = (GlycanSource(GlycanCatalog(tuple(args.glycan_store)), args.glycan_kind)
              if args.glycan_store else PlinderSource(args.plinder_data, args.plinder_graphs, 32768))
    source_name = f'glycan-{args.glycan_kind}' if args.glycan_store else 'plinder'
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((args.output / 'catalog.json').read_text())['samples'] if args.append else []
    for record_index in args.records:
        record = add_protein_intraresidue_bonds(source.catalog.load(source.kind, record_index)
                                               if args.glycan_store else source.load(record_index))
        batch = pack_repeated(record, 1, source_index={'plinder': 2, 'glycan-free': 3, 'glycan-attached': 4}[source_name])
        edges = peptide_edges(batch, residue_groups(batch))
        sources, targets, types = (list(getattr(record, name)) for name in
                                  ('bond_sources', 'bond_targets', 'bond_types'))
        existing = set(zip(sources, targets))
        for c, n, _, _ in edges:
            for a, b in ((c, n), (n, c)):
                if (a, b) not in existing:
                    sources.append(a); targets.append(b); types.append(0)
                    existing.add((a, b))
        record = replace(record, bond_sources=np.asarray(sources, dtype=np.int32),
                         bond_targets=np.asarray(targets, dtype=np.int32),
                         bond_types=np.asarray(types, dtype=np.uint8))
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
            "id": f"{source_name}-{record_index}",
            "label": f"{source_name} {record_index}",
            "source": source_name,
            "source_index": record_index,
            "atoms": atoms,
            "target_coords": record.coords.astype(float).tolist(),
            "base_means": record.base_means.astype(float).tolist(),
            "base_scales": scales.astype(float).tolist(),
            "initial_scales": np.where(molecule | ligand, 10, scales).astype(float).tolist(),
            "atomic_numbers": record.atomic_numbers.astype(int).tolist(),
            "roles": record.roles.astype(int).tolist(),
            "residue_types": record.residue_types.astype(int).tolist(),
            "atom_names": record.atom_names.astype(int).tolist(),
            "entity_ids": record.entity_ids.astype(int).tolist(),
            "coordinate_design": design.astype(int).tolist(),
            "backbone_sigma": [0.0] * atoms,
            "backbone_graph_complete": True,
            "neighbors": neighbors.reshape(-1, 2).astype(int).tolist(),
            "residue_ids": residues.astype(int).tolist(),
            **display_topology(record, residues),
        }
        payload['backbone_trace_pairs'] = payload['backbone_bonds']
        filename = f"{source_name}-{record_index}.json"
        (args.output / filename).write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        catalog.append({"id": payload["id"], "label": payload["label"], "atoms": atoms, "file": filename})
    (args.output / "catalog.json").write_text(json.dumps({"samples": catalog}, indent=2) + "\n")
    print(json.dumps(catalog, indent=2))


if __name__ == "__main__":
    main()
