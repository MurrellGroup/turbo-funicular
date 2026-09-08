import proteinGraph from './protein_graph.json' with { type: 'json' };
import { RESIDUES, ATOM_NAMES, BACKBONE_NAMES, assembleSample } from './prep.js';

export const AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV';
const LETTERS = 'ARNDCQEGHILKMFPSTWYV';
const NAMES = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(' ');
export const AA_NAMES = Object.fromEntries([...LETTERS].map((aa, i) => [aa, NAMES[i]]));
const ONE_LETTER = Object.fromEntries(NAMES.map((name, i) => [name, LETTERS[i]]));

export function parseSequences(text) {
  const records = [];
  for (const line of text.replaceAll('\r', '').split('\n')) {
    if (!line.trim()) continue;
    if (line.startsWith('>')) records.push({ name: line.slice(1).trim() || `Sequence ${records.length + 1}`, sequence: '' });
    else {
      if (!records.length) records.push({ name: 'Sequence', sequence: '' });
      const seq = line.replace(/\s/g, '').toUpperCase();
      if (/[^ARNDCQEGHILKMFPSTWYVX-]/.test(seq)) throw new Error('Sequence must contain amino-acid letters or X.');
      records.at(-1).sequence += seq.replaceAll('-', '');
    }
  }
  if (!records.length || records.some(r => !r.sequence.length)) throw new Error('Empty amino-acid sequence.');
  if (records.length > 100 || records.some(r => r.sequence.length > 10000)) throw new Error('Sequence file exceeds the alignment limit.');
  return records;
}

export function proteinChains(sample) {
  const chains = new Map(), residues = new Map();
  for (let atom = 0; atom < sample.atoms; atom++) {
    if (![1, 2].includes(sample.roles[atom])) continue;
    const id = sample.residue_ids[atom];
    if (!residues.has(id)) {
      const label = sample.atom_labels?.[atom]?.split('|');
      const chain = label?.length === 5 ? label[2] : String(sample.chain_ids[atom]);
      const number = label?.length === 5 ? `${label[3]}${label[4]}` : String(id + 1);
      const name = RESIDUES[sample.residue_types[atom]];
      const residue = { id, chain, number, name, aa: ONE_LETTER[name] ?? 'X', atoms: [] };
      residues.set(id, residue);
      if (!chains.has(chain)) chains.set(chain, { id: chain, residues: [], sequence: '' });
      chains.get(chain).residues.push(residue);
      chains.get(chain).sequence += residue.aa;
    }
    residues.get(id).atoms.push(atom);
  }
  return [...chains.values()];
}

function externalEdges(sample, residue) {
  const atoms = new Set(residue.atoms), edges = [];
  for (const atom of atoms) for (const [other] of sample.neighbors.slice(atom * 10, atom * 10 + 10)) {
    if (other >= 0 && !atoms.has(other)) edges.push([atom, other]);
  }
  return edges;
}

export function validateEdit(sample, residue, aa) {
  if (aa !== 'X' && !AMINO_ACIDS.includes(aa)) throw new Error('Unknown amino-acid identity.');
  if (aa === residue.aa) return;
  if (!residue.atoms.some(i => sample.atom_names[i] === 1)) throw new Error(`Residue ${residue.chain}:${residue.number} has no CA anchor.`);
  if (externalEdges(sample, residue).length) {
    throw new Error(`Residue ${residue.chain}:${residue.number} has an external covalent attachment; its identity is protected.`);
  }
}

// A separate deterministic stream keeps identity draws independent of coordinate noise.
export function randomSequence(seed) {
  let state = (seed ^ 0x9e3779b9) >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let n = state;
    n = Math.imul(n ^ n >>> 15, n | 1);
    n ^= n + Math.imul(n ^ n >>> 7, n | 61);
    return ((n ^ n >>> 14) >>> 0) / 4294967296;
  };
}

export function resolveEdits(sample, edits, seed) {
  const rng = randomSequence(seed), resolved = new Map();
  const residues = proteinChains(sample).flatMap(c => c.residues);
  if ([...edits.keys()].some(id => !residues.some(r => r.id === id))) throw new Error('Edits refer to a removed residue.');
  for (const residue of residues) if (edits.has(residue.id)) {
    const aa = edits.get(residue.id);
    validateEdit(sample, residue, aa);
    resolved.set(residue.id, aa === 'X' ? AMINO_ACIDS[Math.floor(rng() * 20)] : aa);
  }
  return resolved;
}

export function mutateSample(sample, edits, seed) {
  const resolved = resolveEdits(sample, edits, seed);
  const residues = proteinChains(sample).flatMap(c => c.residues);
  const changed = new Map(residues.filter(r => resolved.has(r.id) && resolved.get(r.id) !== r.aa).map(r => [r.id, r]));
  if (!changed.size) return { sample, resolved };
  const arrays = Object.fromEntries(['coords', 'baseMeans', 'atomicNumbers', 'roles', 'residueTypes',
    'atomNames', 'entityIds', 'residueIds', 'chainIds'].map(k => [k, []]));
  const sourceNames = ['target_coords', 'base_means', 'atomic_numbers', 'roles', 'residue_types',
    'atom_names', 'entity_ids', 'residue_ids', 'chain_ids'];
  const keys = Object.keys(arrays), labels = [], inverse = new Map(), newResidues = new Map();
  const appendOld = (i, residueType = sample.residue_types[i]) => {
    const next = labels.length;
    keys.forEach((key, k) => arrays[key].push(sample[sourceNames[k]][i]));
    arrays.residueTypes[next] = residueType;
    const parts = sample.atom_labels[i].split('|');
    if (parts.length === 5 && residueType !== sample.residue_types[i]) parts[1] = RESIDUES[residueType];
    labels.push(parts.join('|')); inverse.set(i, next);
    return next;
  };
  const emitted = new Set();
  for (let i = 0; i < sample.atoms; i++) {
    const residue = changed.get(sample.residue_ids[i]);
    if (!residue) { appendOld(i); continue; }
    if (emitted.has(residue.id)) continue;
    emitted.add(residue.id);
    const name = AA_NAMES[resolved.get(residue.id)], type = RESIDUES.indexOf(name);
    const ca = residue.atoms.find(j => sample.atom_names[j] === 1);
    const names = new Map(); newResidues.set(residue.id, { names, name });
    for (const old of residue.atoms.filter(j => sample.roles[j] === 1)) {
      names.set(ATOM_NAMES[sample.atom_names[old]], appendOld(old, type));
    }
    const required = new Set(proteinGraph[name].flatMap(([a, b]) => [a, b]).filter(a => !BACKBONE_NAMES.has(a)));
    for (const atomName of ATOM_NAMES.filter(a => required.has(a))) {
      const next = labels.length, atomId = ATOM_NAMES.indexOf(atomName);
      const old = residue.atoms.find(j => sample.atom_names[j] === atomId);
      arrays.coords.push(old === undefined ? sample.target_coords[ca] : sample.target_coords[old]);
      arrays.baseMeans.push(sample.target_coords[ca]);
      arrays.atomicNumbers.push({ C: 6, N: 7, O: 8, S: 16 }[atomName[0]]);
      arrays.roles.push(2); arrays.residueTypes.push(type); arrays.atomNames.push(atomId);
      arrays.entityIds.push(sample.entity_ids[ca]); arrays.residueIds.push(residue.id); arrays.chainIds.push(sample.chain_ids[ca]);
      const parts = sample.atom_labels[ca].split('|'); parts[0] = atomName; parts[1] = name;
      labels.push(parts.join('|')); names.set(atomName, next);
      if (old !== undefined) inverse.set(old, next);
    }
  }
  const bonds = [];
  for (let a = 0; a < sample.atoms; a++) for (const [b, type] of sample.neighbors.slice(a * 10, a * 10 + 10)) {
    if (b <= a) continue;
    if (changed.has(sample.residue_ids[a]) && sample.residue_ids[a] === sample.residue_ids[b]) continue;
    if (!inverse.has(a) || !inverse.has(b)) throw new Error('Mutation would remove a covalent bond endpoint.');
    bonds.push({ left: inverse.get(a), right: inverse.get(b), type });
  }
  for (const { name, names } of newResidues.values()) for (const [a, b, type] of proteinGraph[name]) {
    if (names.has(a) && names.has(b)) bonds.push({ left: names.get(a), right: names.get(b), type });
  }
  const result = assembleSample({ id: sample.id, label: sample.label, ...arrays, bonds, graphSource: sample.graph_source });
  result.atom_labels = labels;
  result.reference_sample = sample;
  return { sample: result, resolved };
}
