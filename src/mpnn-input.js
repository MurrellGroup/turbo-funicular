import { fromResidues, resolveResidue } from 'proteinmpnn-web/structure';
import { proteinChains, AMINO_ACIDS, validateEdit } from './sequence.js';
import { ATOM_NAMES } from './prep.js';

export function mpnnInput(sample, edits, family = 'proteinmpnn', maxLength = 10000) {
  const chains = proteinChains(sample), rows = [], mapping = [];
  const existingIds = new Set(chains.map(c => c.id));
  let blank = '_'; while (existingIds.has(blank)) blank += '_';
  for (const chain of chains) {
    const residues = chain.residues.map(residue => {
      const parts = sample.atom_labels?.[residue.atoms[0]]?.split('|');
      if (parts?.length !== 5 || !/^-?\d+$/.test(parts[3])) throw new Error('ProteinMPNN requires original residue identifiers.');
      return { residue, number: Number(parts[3]), insertionCode: parts[4] };
    }).sort((a, b) => a.number - b.number || a.insertionCode.localeCompare(b.insertionCode));
    let previous;
    const add = (row, record) => {
      if (rows.length >= maxLength) throw new Error(`ProteinMPNN exceeds ${maxLength} residues including numbering gaps.`);
      rows.push(row); mapping.push({ ...record, index: rows.length - 1 });
    };
    for (const { residue, number, insertionCode } of residues) {
      const mpnnChain = chain.id || blank;
      if (previous !== undefined) {
        if (rows.length + Math.max(0, number - previous - 1) >= maxLength) throw new Error('ProteinMPNN residue numbering gap exceeds the input limit.');
        for (let missing = previous + 1; missing < number; missing++) add(
          { chain: mpnnChain, number: missing, aminoAcid: 'X', atoms: {} },
          { chain: chain.id, mpnnChain, number: missing, insertionCode: '', residueId: null, design: false },
        );
      }
      previous = number;
      const choice = edits.get(residue.id);
      if (choice) validateEdit(sample, residue, choice);
      const atoms = {};
      for (const i of residue.atoms) {
        const name = ATOM_NAMES[sample.atom_names[i]];
        if (['N', 'CA', 'C', 'O'].includes(name)) atoms[name] = [...sample.target_coords[i]];
      }
      add({ chain: mpnnChain, number, insertionCode, aminoAcid: choice && choice !== 'X' ? choice : residue.aa, atoms },
        { chain: chain.id, mpnnChain, number, insertionCode, residueId: residue.id, design: choice === 'X' });
    }
  }
  if (!rows.length) throw new Error('ProteinMPNN requires a retained protein chain.');
  const structure = fromResidues(rows, { atomMode: family === 'ca' ? 'ca' : 'backbone' });
  const designPositions = [];
  mapping.forEach((r, i) => {
    r.valid = Boolean(structure.mask[i]);
    if (r.design) {
      if (!r.valid) throw new Error(`ProteinMPNN: ${r.chain}:${r.number}${r.insertionCode} lacks ${family === 'ca' ? 'CA' : 'complete N/CA/C/O'} coordinates.`);
      designPositions.push(i);
    }
  });
  if (!designPositions.length) throw new Error('Mark at least one residue X.');
  return { structure, mapping, designPositions };
}

const CONSTRAINT_KEYS = new Set(['biasAminoAcids', 'biasByResidue', 'omitByResidue', 'tiedPositions',
  'tiedConstraintMode', 'pssm', 'decodingOrder', 'orderNoise']);

export function mpnnConstraints(input, { advanced = {}, omitAminoAcids = 'X' } = {}) {
  if (!advanced || Array.isArray(advanced) || typeof advanced !== 'object') throw new Error('Constraints must be a JSON object.');
  for (const key of Object.keys(advanced)) if (!CONSTRAINT_KEYS.has(key)) throw new Error(`Unsupported constraint field: ${key}`);
  if (/[^ACDEFGHIKLMNPQRSTVWYX]/.test(omitAminoAcids)) throw new Error('Invalid omitted amino acid.');
  const { structure, designPositions } = input;
  const resolve = ref => {
    if (ref && typeof ref === 'object' && ref.chain === '') {
      const mapped = input.mapping.find(r => r.chain === '');
      ref = { ...ref, chain: mapped?.mpnnChain ?? '' };
    }
    return resolveResidue(structure, ref);
  };
  const options = { ...advanced, designChains: [...new Set(structure.chainIds)], designPositions,
    omitAminoAcids: [...new Set(omitAminoAcids + 'X')].join('') };
  for (const field of ['biasByResidue', 'omitByResidue']) if (options[field]) options[field] = options[field].map(item => ({ ...item, position: resolve(item.position) }));
  if (options.tiedPositions) options.tiedPositions = options.tiedPositions.map(group => Array.isArray(group)
    ? group.map(resolve) : { ...group, positions: group.positions.map(resolve) });
  // PSSM mixing can reintroduce global omissions; X must remain excluded at the final per-site mask.
  const omissions = new Map();
  for (const { position, aminoAcids } of options.omitByResidue ?? []) {
    if (typeof aminoAcids !== 'string') throw new Error('Per-residue omissions must be amino-acid strings.');
    omissions.set(position, (omissions.get(position) ?? '') + aminoAcids);
  }
  for (const i of designPositions) omissions.set(i, [...new Set((omissions.get(i) ?? '') + 'X')].join(''));
  options.omitByResidue = [...omissions].map(([position, aminoAcids]) => ({ position, aminoAcids }));
  return options;
}

export function proposalEdits(input, result, edits) {
  if (result.sequence?.length !== input.structure.length) throw new Error('ProteinMPNN returned a sequence with the wrong length.');
  const resolved = new Map(edits);
  for (const row of input.mapping) {
    const aa = result.sequence[row.index];
    if (!row.design) {
      if (aa !== input.structure.sequence[row.index]) throw new Error('ProteinMPNN changed a fixed residue.');
    } else {
      if (!AMINO_ACIDS.includes(aa)) throw new Error('ProteinMPNN returned a noncanonical design identity.');
      if (result.designMask && result.designMask[row.index] !== 1) throw new Error('ProteinMPNN design mask lost an X position.');
      resolved.set(row.residueId, aa);
    }
  }
  return resolved;
}
