import { cifLoops } from './ccd.js';
import { RESIDUE_INDEX, RESIDUE_ALIASES, ELEMENT_NUMBER, WATER_NAMES, atomLocator } from './prep.js';

function tables(text) {
  const result = new Map();
  for (const { headers, rows } of cifLoops(text)) {
    const category = headers[0].split('.')[0];
    const entries = rows.map(row => Object.fromEntries(headers.map((h, i) => [h.split('.')[1], row[i]])));
    result.set(category, [...(result.get(category) ?? []), ...entries]);
  }
  return result;
}
const known = value => value !== undefined && value !== '.' && value !== '?';
const clean = value => known(value) ? value : '';
const bondOrder = value => {
  const result = { sing: 0, doub: 1, trip: 2, arom: 3 }[(value ?? 'sing').toLowerCase()];
  if (result === undefined) throw new Error(`Unsupported deposited bond order: ${value}`);
  return result;
};

export function parseMmcif(text, filename = 'structure.cif', depositedText = null) {
  const data = tables(text), source = depositedText ? tables(depositedText) : data;
  const rows = data.get('_atom_site') ?? [];
  if (!rows.length) throw new Error('mmCIF contains no atom_site records.');
  const firstModel = rows[0].pdbx_pdb_model_num;
  const entities = new Map((data.get('_entity') ?? []).map(r => [r.id, r.type]));
  const chosen = new Map();
  for (const row of rows) {
    if (row.pdbx_pdb_model_num !== firstModel) continue;
    const rawResidue = row.label_comp_id.toUpperCase();
    const residueName = RESIDUE_ALIASES.get(rawResidue) ?? rawResidue;
    const protein = RESIDUE_INDEX.has(residueName) && entities.get(row.label_entity_id) === 'polymer';
    if (WATER_NAMES.has(rawResidue)) continue;
    const atomicNumber = ELEMENT_NUMBER.get(row.type_symbol.toUpperCase());
    if (!atomicNumber || atomicNumber <= 1 || atomicNumber >= 128) continue;
    const atomName = row.label_atom_id.toUpperCase();
    const chain = row.label_asym_id;
    const residueNumber = clean(row.auth_seq_id) || clean(row.label_seq_id);
    const insertion = clean(row.pdbx_pdb_ins_code);
    const residueKey = `${chain}|${residueNumber}|${insertion}|${rawResidue}`;
    const atom = { serial: Number(row.id), atomName, rawResidue, residueName, residueNumber,
      insertion, residueKey, chain, coord: [Number(row.cartn_x), Number(row.cartn_y), Number(row.cartn_z)],
      atomicNumber, protein, labelSeq: row.label_seq_id, entity: row.label_entity_id,
      alt: clean(row.label_alt_id), occupancy: Number(row.occupancy), authChain: row.auth_asym_id };
    if (atom.coord.some(v => !Number.isFinite(v))) throw new Error('Nonfinite deposited coordinate.');
    const key = atomLocator(atom), old = chosen.get(key);
    const preference = a => a.alt === 'A' ? 2 : !a.alt ? 1 : 0;
    if (!old || atom.occupancy > old.occupancy || (atom.occupancy === old.occupancy
      && preference(atom) > preference(old))) chosen.set(key, atom);
  }
  const atoms = [...chosen.values()];
  const byChain = new Map();
  for (const atom of atoms) {
    if (!byChain.has(atom.chain)) byChain.set(atom.chain, []);
    byChain.get(atom.chain).push(atom);
  }
  const links = [], connectionErrors = [], seen = new Map();
  const add = (a, b, type) => {
    const key = [atomLocator(a), atomLocator(b)].sort().join('~');
    if (seen.has(key) && seen.get(key) !== type) throw new Error('Conflicting deposited bond categories.');
    if (!seen.has(key)) { seen.set(key, type); links.push([a, b, type]); }
  };
  const scheme = data.get('_pdbx_branch_scheme') ?? [];
  for (const link of data.get('_pdbx_entity_branch_link') ?? []) {
    const chains = new Set(scheme.filter(s => s.entity_id === link.entity_id).map(s => s.asym_id));
    for (const chain of chains) {
      const endpoints = [1, 2].map(i => {
        const row = scheme.find(s => s.asym_id === chain && s.num === link[`entity_branch_list_num_${i}`]);
        if (!row) throw new Error('Incomplete glycan branch scheme.');
        const candidates = (byChain.get(chain) ?? []).filter(a => a.rawResidue === link[`comp_id_${i}`]
          && String(a.residueNumber) === row.pdb_seq_num && a.atomName === link[`atom_id_${i}`].toUpperCase());
        if (candidates.length !== 1) throw new Error(`Unresolved branch endpoint: ${chain}/${row.pdb_seq_num}/${link[`atom_id_${i}`]}`);
        return candidates[0];
      });
      add(...endpoints, bondOrder(link.value_order));
    }
  }
  const remaps = data.get('_pdbx_chain_remapping') ?? [...byChain.keys()].map(chain => ({
    label_asym_id: chain, orig_label_asym_id: chain, applied_operations: '1',
  }));
  const connectionRows = source.get('_struct_conn') ?? [];
  if (depositedText === null && data.has('_pdbx_chain_remapping') && !connectionRows.length
    && scheme.length) throw new Error('Assembly glycan attachments require the deposited mmCIF as well. Load by PDB ID.');
  for (const link of connectionRows) {
    if (!link.conn_type_id?.toLowerCase().startsWith('covale')) continue;
    const copies = [1, 2].map(i => remaps.filter(r => r.orig_label_asym_id === link[`ptnr${i}_label_asym_id`]));
    for (const left of copies[0]) for (const right of copies[1]) {
      if (left.applied_operations !== right.applied_operations) continue;
      const endpoints = [left, right].map((copy, index) => {
        const p = `ptnr${index + 1}_`, ins = clean(link[`pdbx_ptnr${index + 1}_pdb_ins_code`]);
        const comp = link[`${p}label_comp_id`], name = link[`${p}label_atom_id`].toUpperCase();
        const seq = link[`${p}label_seq_id`], authSeq = link[`${p}auth_seq_id`];
        const matches = (byChain.get(copy.label_asym_id) ?? []).filter(a => a.rawResidue === comp
          && a.atomName === name && a.insertion === ins
          && (known(seq) ? a.labelSeq === seq : String(a.residueNumber) === authSeq));
        if (matches.length === 1) return matches[0];
        return { atomName: name, rawResidue: comp, chain: copy.label_asym_id,
          residueNumber: authSeq, insertion: ins, unresolved: true };
      });
      const sameAsu = [1, 2].every(i => !known(link[`ptnr${i}_symmetry`]) || link[`ptnr${i}_symmetry`] === '1_555');
      if (!sameAsu || endpoints.some(a => a.unresolved)) connectionErrors.push({ endpoints });
      else add(...endpoints, known(link.pdbx_value_order) ? bondOrder(link.pdbx_value_order) : 0);
    }
  }
  const groups = new Map();
  for (const atom of atoms.filter(a => !a.protein)) {
    if (!groups.has(atom.residueKey)) groups.set(atom.residueKey, []);
    groups.get(atom.residueKey).push(atom);
  }
  const ligandOptions = [...groups].map(([id, values]) => ({ id, atoms: values,
    label: `${values[0].rawResidue} ${values[0].chain}:${values[0].residueNumber} (${values.length})`,
  })).sort((a, b) => b.atoms.length - a.atoms.length);
  return { filename, title: filename, proteinAtoms: atoms.filter(a => a.protein), ligandOptions,
    links, connectionErrors, directedConnections: new Map(), defaultLigandId: ligandOptions[0]?.id ?? null };
}
