#!/usr/bin/env python3
"""Export the named protein graph table."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from wsfmdock.protein_graph import BACKBONE_BONDS, SIDECHAIN_BONDS, MSE_SIDECHAIN_BONDS
table = {name: [*BACKBONE_BONDS, *bonds] for name, bonds in SIDECHAIN_BONDS.items()}
table['MSE'] = [*BACKBONE_BONDS, *MSE_SIDECHAIN_BONDS]
args.output.write_text(json.dumps(table, indent=2) + '\n')
