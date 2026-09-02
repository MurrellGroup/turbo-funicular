const CCD_ROOT = "https://files.rcsb.org/ligands/download/";
const ORDER = new Map([["SING", 0], ["DOUB", 1], ["TRIP", 2]]);
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
  }
  if (!atoms?.length || !bonds) throw new Error(`CCD ${componentId} has no complete atom/bond definition.`);
  return { componentId, atoms, bonds };
}

async function fetchCcdGraph(componentId) {
  const id = componentId.trim().toUpperCase();
  if (!/^[A-Z0-9]{1,3}$/.test(id)) throw new Error(`Invalid PDB component ID ${componentId}.`);
  const response = await fetch(`${CCD_ROOT}${encodeURIComponent(id)}.cif`);
  if (!response.ok) throw new Error(`RCSB CCD ${id} could not be loaded (${response.status}).`);
  return parseCcdGraph(await response.text(), id);
}

export async function loadCcdGraphs(componentIds) {
  const ids = [...new Set(componentIds.map((value) => value.trim().toUpperCase()))];
  const entries = await Promise.all(ids.map(async (id) => {
    if (!cache.has(id)) cache.set(id, fetchCcdGraph(id));
    try {
      return [id, await cache.get(id)];
    } catch {
      cache.delete(id);
      return [id, null];
    }
  }));
  return new Map(entries);
}
