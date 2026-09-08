import { ATOM_NAMES, RESIDUES, ELEMENT_NUMBER } from './prep.js';

const ELEMENTS = new Map([...ELEMENT_NUMBER].map(([symbol, number]) => [number, symbol]));
const CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

function field(value, width, label, right = false) {
  const text = String(value);
  if (!/^[ -~]*$/.test(text) || text.length > width) throw new Error(`${label} cannot be represented in PDB format.`);
  return right ? text.padStart(width) : text.padEnd(width);
}
function integer(value, width, label) {
  if (!Number.isInteger(value)) throw new Error(`Invalid ${label}.`);
  return field(value, width, label, true);
}
function atomName(name, element) {
  field(name, 4, 'Atom name');
  return name.length < 4 && element.length === 1 && !/^\d/.test(name) ? ` ${name.padEnd(3)}` : name.padEnd(4);
}
function location(atom) {
  return `${atomName(atom.name, atom.element)} ${field(atom.residue, 3, 'Residue name', true)} ${atom.chain}${integer(atom.number, 4, 'Residue number')}${field(atom.insertion, 1, 'Insertion code')}`;
}

export function exportPdb(result) {
  const { sample, coords } = result;
  if (!sample?.atoms || coords?.length !== sample.atoms * 3) throw new Error('Invalid coordinate sample.');
  const origin = sample.coordinate_origin ?? [0, 0, 0];
  if (origin.length !== 3 || !origin.every(Number.isFinite)) throw new Error('Invalid coordinate origin.');
  const atoms = [], syntheticResidues = new Map(), syntheticCounts = new Map(), syntheticAtoms = new Map();
  let synthesized = false;
  for (let i = 0; i < sample.atoms; i++) {
    const protein = sample.roles[i] === 1 || sample.roles[i] === 2;
    const element = ELEMENTS.get(sample.atomic_numbers[i]);
    if (!element) throw new Error(`Unknown element at atom ${i}.`);
    const label = sample.atom_labels?.[i]?.split('|');
    let name, residue, chain, number, insertion;
    if (label?.length === 5) {
      [name, residue, chain, number, insertion] = label;
      if (!/^-?\d+$/.test(number)) throw new Error('Invalid original residue number.');
      number = Number(number);
    } else {
      synthesized = true;
      const entity = sample.entity_ids[i];
      chain = `${protein ? 'protein' : 'molecule'}:${sample.chain_ids?.[i] ?? entity}`;
      const key = `${chain}:${protein ? sample.residue_ids[i] : entity}`;
      if (!syntheticResidues.has(key)) syntheticResidues.set(key, (syntheticCounts.get(chain) ?? 0) + 1);
      number = syntheticResidues.get(key); syntheticCounts.set(chain, number);
      insertion = '';
      residue = protein ? RESIDUES[sample.residue_types[i]] ?? 'UNK' : 'LIG';
      const index = (syntheticAtoms.get(key) ?? 0) + 1; syntheticAtoms.set(key, index);
      name = protein ? ATOM_NAMES[sample.atom_names[i]] : `${element}${index}`;
      if (!name) throw new Error('Missing protein atom name.');
      atoms.push({ syntheticKey: key, i, protein, element, name, residue, originalChain: chain, number, insertion });
      continue;
    }
    atoms.push({ i, protein, element, name, residue, originalChain: chain, number, insertion });
  }
  const chainMap = new Map(), used = new Set();
  for (const { originalChain: id } of atoms) if (/^[A-Za-z0-9]?$/.test(id)) { chainMap.set(id, id || ' '); used.add(id || ' '); }
  for (const { originalChain: id } of atoms) if (!chainMap.has(id)) {
    const available = [...CHAIN_IDS].find(c => !used.has(c));
    if (!available) throw new Error('Too many chains for PDB export (62 single-character IDs).');
    chainMap.set(id, available); used.add(available);
  }
  const seen = new Set();
  for (const atom of atoms) {
    atom.chain = chainMap.get(atom.originalChain);
    const key = JSON.stringify([atom.chain, atom.number, atom.insertion, atom.name]);
    if (seen.has(key)) throw new Error('PDB export would create duplicate atom identifiers.');
    seen.add(key); location(atom);
  }
  const lines = ['REMARK 900 SAMPLED STRUCTURE'];
  if (Number.isInteger(result.seed)) lines.push(`REMARK 900 SEED ${result.seed}`);
  if (synthesized) lines.push('REMARK 900 IDENTIFIERS GENERATED WHERE ORIGINAL LABELS WERE UNAVAILABLE');
  for (const [id, mapped] of chainMap) if (id !== mapped && id !== '') {
    const text = JSON.stringify(id);
    if (text.length > 54) throw new Error('Chain identifier is too long for a PDB mapping remark.');
    lines.push(`REMARK 900 CHAIN ${mapped} ORIGINAL ${text}`);
  }
  const serials = new Map(); let serial = 0;
  const emit = atom => {
    serials.set(atom.i, ++serial);
    const xyz = [0, 1, 2].map(axis => {
      const value = Number(coords[atom.i * 3 + axis]) + origin[axis];
      if (!Number.isFinite(value)) throw new Error('Cannot export non-finite coordinates.');
      if (value < -999.9995 || value >= 9999.9995) throw new Error('Coordinate exceeds PDB field limits.');
      return field(value.toFixed(3), 8, 'Coordinate', true);
    }).join('');
    lines.push(`${atom.protein ? 'ATOM  ' : 'HETATM'}${integer(serial, 5, 'Atom serial')} ${location(atom)}   ${xyz}  1.00  0.00          ${field(atom.element, 2, 'Element', true)}  `);
  };
  for (const id of chainMap.keys()) {
    const chain = atoms.filter(a => a.protein && a.originalChain === id);
    for (const atom of chain) emit(atom);
    if (chain.length) {
      const last = chain.at(-1);
      lines.push(`TER   ${integer(++serial, 5, 'TER serial')}      ${field(last.residue, 3, 'Residue name', true)} ${last.chain}${integer(last.number, 4, 'Residue number')}${field(last.insertion, 1, 'Insertion code')}`);
    }
  }
  for (const atom of atoms.filter(a => !a.protein)) emit(atom);
  const adjacency = new Map(), links = [];
  for (let i = 0; i < sample.atoms; i++) {
    const neighbors = sample.neighbors.slice(i * 10, i * 10 + 10).filter(([j]) => j >= 0).map(([j]) => j);
    for (const j of neighbors) {
      if (!Number.isInteger(j) || j >= sample.atoms || j === i) throw new Error('Invalid covalent graph.');
      if (!sample.neighbors.slice(j * 10, j * 10 + 10).some(([k]) => k === i)) throw new Error('Asymmetric covalent graph.');
      if (j > i) {
        const a = atoms[i], b = atoms[j];
        if (a.chain !== b.chain || a.number !== b.number || a.insertion !== b.insertion) {
          links.push(`LINK        ${location(a)}               ${location(b)}  1555   1555`);
        }
      }
    }
    if (neighbors.length) adjacency.set(serials.get(i), [...new Set(neighbors.map(j => serials.get(j)))].sort((a, b) => a - b));
  }
  lines.splice(lines.findIndex(line => line.startsWith('ATOM  ') || line.startsWith('HETATM')), 0, ...links);
  for (const [source, targets] of [...adjacency].sort(([a], [b]) => a - b)) {
    for (let i = 0; i < targets.length; i += 4) lines.push(`CONECT${integer(source, 5, 'CONECT serial')}${targets.slice(i, i + 4).map(j => integer(j, 5, 'CONECT serial')).join('')}`);
  }
  lines.push('END');
  return lines.map(line => field(line, 80, 'PDB record')).join('\n') + '\n';
}
