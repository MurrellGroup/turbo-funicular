export function viewerSample() {
  const chains = ['B', 'B', 'D', 'D', 'B', 'B', 'A', 'A', 'C', 'C'];
  return {
    atoms: 10,
    roles: [1, 2, 1, 2, 3, 3, 3, 3, 0, 0],
    atomic_numbers: [6, 6, 6, 8, 6, 8, 6, 7, 6, 30],
    residue_ids: [0, 0, 1, 1, -1, -1, -1, -1, -1, -1],
    chain_ids: [1, 1, 3, 3, 1, 1, 0, 0, 2, 2],
    atom_labels: chains.map((chain, i) => `RES|A${i}|${chain}|${Math.floor(i / 2)}|`),
    target_coords: [[-6, -2, 0], [-5, -2, 1], [1, -1, 0], [2, -1, 1],
      [-3, 3, 0], [-2, 3, 1], [5, 3, 0], [6, 3, 1], [0, 6, 0], [1, 6, 1]],
    backbone_trace_pairs: [[0, 2]],
    sidechain_bonds: [[0, 1], [2, 3]],
    ligand_bonds: [[4, 5], [6, 7]],
    molecule_bonds: [[8, 9]],
  };
}
