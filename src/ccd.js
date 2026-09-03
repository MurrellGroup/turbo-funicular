import { graphFromSmiles } from "./chemistry.js";

const CCD_ROOT = "https://files.rcsb.org/ligands/download/";
const ORDER = new Map([["SING", 0], ["DOUB", 1], ["TRIP", 2]]);
const ELEMENTS = [
  "", "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
  "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR",
  "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
  "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
  "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
  "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF",
  "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO",
  "AT", "RN", "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM",
  "BK", "CF", "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS",
  "MT", "DS", "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
];
const ELEMENT_NUMBER = new Map(ELEMENTS.map((symbol, index) => [symbol, index]));
const cache = new Map();

function tokenizeCif(text) {
  const tokens = [];
  let index = 0;
  let lineStart = true;
  while (index < text.length) {
    const character = text[index];
    if (/\s/.test(character)) {
      if (character === "\n") lineStart = true;
      index += 1;
      continue;
    }
    if (character === "#") {
      const end = text.indexOf("\n", index);
      index = end < 0 ? text.length : end + 1;
      lineStart = true;
      continue;
    }
    if (character === ";" && lineStart) {
      const end = text.indexOf("\n;", index + 1);
      if (end < 0) throw new Error("Unterminated multiline CIF value.");
      tokens.push(text.slice(index + 1, end));
      const lineEnd = text.indexOf("\n", end + 2);
      index = lineEnd < 0 ? text.length : lineEnd + 1;
      lineStart = true;
      continue;
    }
    lineStart = false;
    if (character === "'" || character === '"') {
      const quote = character;
      const start = ++index;
      while (index < text.length) {
        if (text[index] === quote && (index + 1 === text.length || /\s/.test(text[index + 1]))) break;
        index += 1;
      }
      if (index >= text.length) throw new Error("Unterminated quoted CIF value.");
      tokens.push(text.slice(start, index));
      index += 1;
      continue;
    }
    const start = index;
    while (index < text.length && !/\s/.test(text[index])) index += 1;
    tokens.push(text.slice(start, index));
  }
  return tokens;
}

function isControlToken(token) {
  const lower = token.toLowerCase();
  return token.startsWith("_")
    || lower === "loop_"
    || lower === "stop_"
    || lower === "global_"
    || lower.startsWith("data_")
    || lower.startsWith("save_");
}

function cifLoops(text) {
  const tokens = tokenizeCif(text);
  const loops = [];
  for (let index = 0; index < tokens.length;) {
    if (tokens[index].toLowerCase() !== "loop_") {
      index += 1;
      continue;
    }
    index += 1;
    const headers = [];
    while (index < tokens.length && tokens[index].startsWith("_")) {
      headers.push(tokens[index].toLowerCase());
      index += 1;
    }
    if (!headers.length) throw new Error("CIF loop has no columns.");
    const rows = [];
    while (index < tokens.length && !isControlToken(tokens[index])) {
      if (index + headers.length > tokens.length) throw new Error("Incomplete CIF loop row.");
      rows.push(tokens.slice(index, index + headers.length));
      index += headers.length;
    }
    loops.push({ headers, rows });
  }
  return loops;
}

function column(loop, name) {
  const result = loop.headers.indexOf(name);
  if (result < 0) throw new Error(`CCD is missing ${name}.`);
  return result;
}

export function parseCcdGraph(text, requestedId) {
  const componentId = requestedId.trim().toUpperCase();
  let atoms;
  let bonds;
  const smiles = [];
  for (const loop of cifLoops(text)) {
    if (loop.headers.includes("_chem_comp_atom.atom_id")) {
      const idColumn = column(loop, "_chem_comp_atom.comp_id");
      const nameColumn = column(loop, "_chem_comp_atom.atom_id");
      const elementColumn = column(loop, "_chem_comp_atom.type_symbol");
      atoms = loop.rows
        .filter((row) => row[idColumn].toUpperCase() === componentId)
        .map((row) => ({ name: row[nameColumn].toUpperCase(), element: row[elementColumn].toUpperCase() }));
    }
    if (loop.headers.includes("_chem_comp_bond.atom_id_1")) {
      const idColumn = column(loop, "_chem_comp_bond.comp_id");
      const firstColumn = column(loop, "_chem_comp_bond.atom_id_1");
      const secondColumn = column(loop, "_chem_comp_bond.atom_id_2");
      const orderColumn = column(loop, "_chem_comp_bond.value_order");
      const aromaticColumn = column(loop, "_chem_comp_bond.pdbx_aromatic_flag");
      bonds = loop.rows
        .filter((row) => row[idColumn].toUpperCase() === componentId)
        .map((row) => {
          const order = row[orderColumn].toUpperCase();
          const aromatic = row[aromaticColumn].toUpperCase() === "Y";
          const type = aromatic || order === "AROM" || order === "DELO" ? 3 : ORDER.get(order);
          if (type === undefined) throw new Error(`CCD ${componentId} has unsupported bond order ${order}.`);
          return {
            first: row[firstColumn].toUpperCase(),
            second: row[secondColumn].toUpperCase(),
            type,
          };
        });
    }
    if (loop.headers.includes("_pdbx_chem_comp_descriptor.descriptor")) {
      const idColumn = column(loop, "_pdbx_chem_comp_descriptor.comp_id");
      const typeColumn = column(loop, "_pdbx_chem_comp_descriptor.type");
      const programColumn = column(loop, "_pdbx_chem_comp_descriptor.program");
      const descriptorColumn = column(loop, "_pdbx_chem_comp_descriptor.descriptor");
      for (const row of loop.rows) {
        if (row[idColumn].toUpperCase() !== componentId) continue;
        const type = row[typeColumn].toUpperCase();
        if (type !== "SMILES_CANONICAL" && type !== "SMILES") continue;
        smiles.push({
          value: row[descriptorColumn],
          canonical: type === "SMILES_CANONICAL",
          program: row[programColumn].toUpperCase(),
        });
      }
    }
  }
  if (!atoms?.length || !bonds || !smiles.length) {
    throw new Error(`CCD ${componentId} has no complete atom/bond/SMILES definition.`);
  }
  smiles.sort((left, right) => (
    Number(right.canonical) - Number(left.canonical)
    || Number(right.program === "CACTVS") - Number(left.program === "CACTVS")
  ));
  return { componentId, atoms, bonds, smiles: smiles.map((entry) => entry.value) };
}

function adjacency(atoms, bonds) {
  const result = Array.from({ length: atoms }, () => new Set());
  for (const { left, right } of bonds) {
    result[left].add(right);
    result[right].add(left);
  }
  return result;
}

function atomSignature(atom, atomicNumbers, graph) {
  const neighbors = [...graph[atom]].map((index) => atomicNumbers[index]).sort((a, b) => a - b);
  return `${atomicNumbers[atom]}|${neighbors.join(",")}`;
}

function graphCorrespondence(sourceNumbers, sourceBonds, targetNumbers, targetBonds) {
  if (sourceNumbers.length !== targetNumbers.length || sourceBonds.length !== targetBonds.length) {
    throw new Error("CCD SMILES and atom-name graphs differ in size.");
  }
  const sourceGraph = adjacency(sourceNumbers.length, sourceBonds);
  const targetGraph = adjacency(targetNumbers.length, targetBonds);
  const targetBySignature = new Map();
  for (let atom = 0; atom < targetNumbers.length; atom += 1) {
    const signature = atomSignature(atom, targetNumbers, targetGraph);
    if (!targetBySignature.has(signature)) targetBySignature.set(signature, []);
    targetBySignature.get(signature).push(atom);
  }
  const candidates = sourceNumbers.map((_, atom) => (
    targetBySignature.get(atomSignature(atom, sourceNumbers, sourceGraph)) ?? []
  ));
  if (candidates.some((values) => !values.length)) {
    throw new Error("CCD SMILES cannot be matched to its atom-name graph.");
  }
  const mapping = new Int32Array(sourceNumbers.length).fill(-1);
  const used = new Uint8Array(targetNumbers.length);
  const compatible = (source, target) => {
    for (let other = 0; other < mapping.length; other += 1) {
      if (mapping[other] < 0) continue;
      if (sourceGraph[source].has(other) !== targetGraph[target].has(mapping[other])) return false;
    }
    return true;
  };
  const visit = (assigned) => {
    if (assigned === mapping.length) return true;
    let source = -1;
    let options = [];
    for (let atom = 0; atom < mapping.length; atom += 1) {
      if (mapping[atom] >= 0) continue;
      const available = candidates[atom].filter((target) => !used[target] && compatible(atom, target));
      if (!available.length) return false;
      if (source < 0 || available.length < options.length) {
        source = atom;
        options = available;
      }
    }
    for (const target of options) {
      mapping[source] = target;
      used[target] = 1;
      if (visit(assigned + 1)) return true;
      mapping[source] = -1;
      used[target] = 0;
    }
    return false;
  };
  if (!visit(0)) throw new Error("CCD SMILES and atom-name graphs are not isomorphic.");
  return [...mapping];
}

export function trainingGraphFromCcd(rdkit, ccd) {
  let graph;
  let lastError;
  for (const smiles of ccd.smiles) {
    try {
      graph = graphFromSmiles(rdkit, smiles);
      break;
    } catch (error) {
      lastError = error;
    }
  }
  if (!graph) throw lastError ?? new Error(`CCD ${ccd.componentId} has no RDKit-compatible SMILES.`);
  const heavyAtoms = ccd.atoms.filter((atom) => !["H", "D", "T"].includes(atom.element));
  const nameToIndex = new Map(heavyAtoms.map((atom, index) => [atom.name, index]));
  const rawBonds = ccd.bonds.flatMap((bond) => {
    const left = nameToIndex.get(bond.first);
    const right = nameToIndex.get(bond.second);
    return left === undefined || right === undefined ? [] : [{ left, right }];
  });
  const targetNumbers = heavyAtoms.map((atom) => ELEMENT_NUMBER.get(atom.element) ?? 0);
  if (targetNumbers.some((value) => value <= 1)) {
    throw new Error(`CCD ${ccd.componentId} contains an unsupported heavy element.`);
  }
  const mapping = graphCorrespondence(graph.atomicNumbers, graph.bonds, targetNumbers, rawBonds);
  const atoms = mapping.map((index) => ({ ...heavyAtoms[index], atomicNumber: targetNumbers[index] }));
  return {
    componentId: ccd.componentId,
    atoms,
    bonds: graph.bonds.map(({ left, right, type }) => ({
      first: atoms[left].name,
      second: atoms[right].name,
      type,
    })),
    canonicalSmiles: graph.canonicalSmiles,
  };
}

async function fetchCcdGraph(componentId) {
  const id = componentId.trim().toUpperCase();
  if (!/^[A-Z0-9]{1,3}$/.test(id)) throw new Error(`Invalid PDB component ID ${componentId}.`);
  const response = await fetch(`${CCD_ROOT}${encodeURIComponent(id)}.cif`);
  if (!response.ok) throw new Error(`RCSB CCD ${id} could not be loaded (${response.status}).`);
  return parseCcdGraph(await response.text(), id);
}

export async function loadCcdGraphs(componentIds, rdkit) {
  const ids = [...new Set(componentIds.map((value) => value.trim().toUpperCase()))];
  const entries = await Promise.all(ids.map(async (id) => {
    if (!cache.has(id)) cache.set(id, fetchCcdGraph(id));
    try {
      return [id, trainingGraphFromCcd(rdkit, await cache.get(id))];
    } catch {
      cache.delete(id);
      return [id, null];
    }
  }));
  return new Map(entries);
}
