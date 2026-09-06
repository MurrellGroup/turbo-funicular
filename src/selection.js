import proteinGraph from './protein_graph.json' with { type: 'json' };
import { atomLocator } from './prep.js';

export function structureChoices(structure) {
  const chains = new Map();
  for (const atom of structure.proteinAtoms) {
    if (!chains.has(atom.chain)) chains.set(atom.chain, { id: atom.chain, atoms: 0 });
    chains.get(atom.chain).atoms++;
  }
  const options = structure.ligandOptions;
  const parent = options.map((_, i) => i);
  const find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const byAtom = new Map(options.flatMap((o, i) => o.atoms.map(a => [atomLocator(a), i])));
  for (const [a, b] of structure.links) {
    const left = byAtom.get(atomLocator(a)), right = byAtom.get(atomLocator(b));
    if (left !== undefined && right !== undefined) parent[find(right)] = find(left);
  }
  const groups = new Map();
  options.forEach((option, i) => {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, { id: options[root].id, options: [], atoms: 0, chains: new Set() });
    const group = groups.get(root);
    group.options.push(option);
    group.atoms += option.atoms.length;
  });
  const protein = new Map(structure.proteinAtoms.map(a => [atomLocator(a), a.chain]));
  for (const [a, b] of structure.links) {
    for (const [ligand, receptor] of [[a, b], [b, a]]) {
      const index = byAtom.get(atomLocator(ligand)), chain = protein.get(atomLocator(receptor));
      if (index !== undefined && chain !== undefined) groups.get(find(index)).chains.add(chain);
    }
  }
  return { chains: [...chains.values()], ligands: [...groups.values()] };
}

export function selectedStructure(structure, choices, chains, ligands) {
  const groups = choices.ligands.filter(g => ligands.has(g.id));
  for (const group of groups) for (const chain of group.chains) {
    if (!chains.has(chain)) throw new Error(`Selected ligand is attached to chain ${chain}.`);
  }
  const ids = new Set(groups.flatMap(g => g.options.map(o => o.id)));
  return { ...structure, proteinAtoms: structure.proteinAtoms.filter(a => chains.has(a.chain)),
    ligandOptions: structure.ligandOptions.filter(o => ids.has(o.id)) };
}

export function previewStructure(structure, componentGraphs = new Map()) {
  const atoms = [...structure.proteinAtoms, ...structure.ligandOptions.flatMap(o => o.atoms)];
  const center = [0, 0, 0];
  for (const a of atoms) for (let axis = 0; axis < 3; axis++) center[axis] += a.coord[axis] / atoms.length;
  const coords = atoms.map(a => a.coord.map((v, i) => v - center[i]));
  const names = ['N', 'CA', 'C', 'O', 'OXT'];
  const residues = new Map(), located = new Map();
  const roles = [], residueIds = [];
  atoms.forEach((a, i) => {
    located.set(atomLocator(a), i);
    roles.push(a.protein ? (names.includes(a.atomName) ? 1 : 2) : 3);
    if (a.protein) {
      if (!residues.has(a.residueKey)) residues.set(a.residueKey, new Map());
      residues.get(a.residueKey).set(a.atomName, i);
    }
  });
  const residueIndex = new Map([...residues.keys()].map((key, i) => [key, i]));
  const bonds = new Map();
  const add = (a, b) => { if (a !== undefined && b !== undefined && a !== b) bonds.set(`${Math.min(a,b)}:${Math.max(a,b)}`, [a,b]); };
  for (const entries of residues.values()) {
    const a = atoms[entries.values().next().value];
    for (const [first, second] of proteinGraph[a.rawResidue] ?? proteinGraph[a.residueName] ?? []) add(entries.get(first), entries.get(second));
  }
  for (const [a, b] of structure.links) add(located.get(atomLocator(a)), located.get(atomLocator(b)));
  for (const option of structure.ligandOptions) {
    const graph = componentGraphs.get(option.atoms[0].rawResidue);
    const names = new Map(option.atoms.map(a => [a.atomName, located.get(atomLocator(a))]));
    for (const bond of graph?.bonds ?? []) add(names.get(bond.first), names.get(bond.second));
  }
  for (const a of atoms) residueIds.push(a.protein ? residueIndex.get(a.residueKey) : -1);
  const pairs = [...bonds.values()];
  return { id: `preview-${structure.filename}`, label: structure.filename, atoms: atoms.length, target_coords: coords,
    atom_labels: atoms.map(atomLocator),
    atomic_numbers: atoms.map(a => a.atomicNumber), roles, residue_ids: residueIds,
    atom_names: atoms.map(a => a.protein && names.includes(a.atomName) ? names.indexOf(a.atomName) : 38),
    backbone_trace_pairs: pairs.filter(([a,b]) => roles[a] === 1 && roles[b] === 1),
    sidechain_bonds: pairs.filter(([a,b]) => roles[a] !== 3 && roles[b] !== 3 && (roles[a] === 2 || roles[b] === 2)),
    ligand_bonds: pairs.filter(([a,b]) => roles[a] === 3 || roles[b] === 3), molecule_bonds: [] };
}
