import { connectedComponents } from "./chemistry.js";

export const ROLE_BACKBONE = 1;
export const ROLE_SIDECHAIN = 2;
export const ROLE_LIGAND = 3;
const UNKNOWN_RESIDUE = 20;
const UNKNOWN_ATOM_NAME = 38;
const MAX_NEIGHBORS = 10;
const BACKBONE_NAMES = new Set(["N", "CA", "C", "O", "OXT"]);
const WATER_NAMES = new Set(["HOH", "WAT", "DOD"]);
const RESIDUES = [
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
  "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];
const ATOM_NAMES = [
  "N", "CA", "C", "O", "OXT", "CB", "CG", "CG1", "CG2", "CD", "CD1",
  "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1",
  "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OD1", "OD2", "OE1",
  "OE2", "OG", "OG1", "OH", "SD", "SG", "SE",
];
const RESIDUE_ALIASES = new Map(Object.entries({
  ASH: "ASP", CYM: "CYS", CYX: "CYS", GLH: "GLU", HID: "HIS", HIE: "HIS",
  HIP: "HIS", HSD: "HIS", HSE: "HIS", HSP: "HIS", LYN: "LYS", MSE: "MET",
}));
const RESIDUE_INDEX = new Map(RESIDUES.map((name, index) => [name, index]));
const ATOM_NAME_INDEX = new Map(ATOM_NAMES.map((name, index) => [name, index]));
const ELEMENTS = [
  "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
  "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
  "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
  "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
  "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
  "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
  "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
  "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];
const ELEMENT_NUMBER = new Map(ELEMENTS.map((symbol, index) => [symbol.toUpperCase(), index]));
const COVALENT_RADII = new Map([
  [5, 0.84], [6, 0.76], [7, 0.71], [8, 0.66], [9, 0.57], [14, 1.11],
  [15, 1.07], [16, 1.05], [17, 1.02], [34, 1.20], [35, 1.20], [53, 1.39],
]);

function elementNumber(line, atomName, protein) {
  const declared = line.slice(76, 78).trim().toUpperCase();
  if (ELEMENT_NUMBER.has(declared)) return ELEMENT_NUMBER.get(declared);
  const cleaned = atomName.replace(/^\d+/, "");
  if (!cleaned) return 0;
  if (!protein && line[12] !== " ") {
    const pair = cleaned.slice(0, 2).toUpperCase();
    if (ELEMENT_NUMBER.has(pair)) return ELEMENT_NUMBER.get(pair);
  }
  return ELEMENT_NUMBER.get(cleaned[0].toUpperCase()) ?? 0;
}

function distance(left, right) {
  return Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
}

function meanCoordinate(atoms) {
  const mean = [0, 0, 0];
  for (const atom of atoms) {
    for (let axis = 0; axis < 3; axis += 1) mean[axis] += atom.coord[axis] / atoms.length;
  }
  return mean;
}

function subtract(point, center) {
  return point.map((value, axis) => value - center[axis]);
}

export function parsePdb(text, filename = "structure.pdb") {
  const lines = text.replaceAll("\r", "").split("\n");
  const atoms = [];
  const occupied = new Set();
  let modelSeen = false;
  let modelComplete = false;
  const title = lines
    .filter((line) => line.startsWith("TITLE "))
    .map((line) => line.slice(10).trim())
    .join(" ");
  for (const line of lines) {
    const record = line.slice(0, 6).trim();
    if (record === "MODEL") {
      if (modelSeen) modelComplete = true;
      modelSeen = true;
      continue;
    }
    if (record === "ENDMDL" && modelSeen) {
      modelComplete = true;
      continue;
    }
    if (modelComplete || (record !== "ATOM" && record !== "HETATM")) continue;
    if (line.length < 54 || !["", "A"].includes(line[16]?.trim() ?? "")) continue;
    const serial = Number.parseInt(line.slice(6, 11), 10);
    const atomName = line.slice(12, 16).trim().toUpperCase();
    const rawResidue = line.slice(17, 20).trim().toUpperCase();
    const residueName = RESIDUE_ALIASES.get(rawResidue) ?? rawResidue;
    const chain = line.slice(21, 22).trim();
    const residueNumber = Number.parseInt(line.slice(22, 26), 10);
    const insertion = line.slice(26, 27).trim();
    const residueKey = `${chain}|${residueNumber}|${insertion}`;
    const protein = RESIDUE_INDEX.has(residueName)
      && (record === "ATOM" || RESIDUE_ALIASES.has(rawResidue));
    if (!protein && (record !== "HETATM" || WATER_NAMES.has(rawResidue))) continue;
    const key = `${protein ? "P" : "L"}|${residueKey}|${rawResidue}|${atomName}`;
    if (occupied.has(key)) continue;
    occupied.add(key);
    const coord = [
      Number.parseFloat(line.slice(30, 38)),
      Number.parseFloat(line.slice(38, 46)),
      Number.parseFloat(line.slice(46, 54)),
    ];
    const atomicNumber = elementNumber(line, atomName, protein);
    if (!Number.isInteger(serial) || coord.some((value) => !Number.isFinite(value))) continue;
    if (atomicNumber <= 1 || atomicNumber >= 128) continue;
    atoms.push({
      serial, atomName, rawResidue, residueName, residueKey, chain, residueNumber,
      insertion, coord, atomicNumber, protein, line,
    });
  }
  const proteinAtoms = atoms.filter((atom) => atom.protein);
  if (!proteinAtoms.length) throw new Error("PDB contains no supported protein heavy atoms.");
  const ligandAtoms = atoms.filter((atom) => !atom.protein);
  const groups = new Map();
  for (const atom of ligandAtoms) {
    const id = `${atom.chain}|${atom.residueNumber}|${atom.insertion}|${atom.rawResidue}`;
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(atom);
  }
  const ligandOptions = [...groups].map(([id, values]) => ({
    id,
    label: `${values[0].rawResidue} ${values[0].chain || "-"}:${values[0].residueNumber}${values[0].insertion} (${values.length})`,
    atoms: values,
  })).sort((left, right) => right.atoms.length - left.atoms.length);

  const directedConnections = new Map();
  for (const line of lines.filter((value) => value.startsWith("CONECT"))) {
    const source = Number.parseInt(line.slice(6, 11), 10);
    for (let offset = 11; offset + 5 <= line.length; offset += 5) {
      const target = Number.parseInt(line.slice(offset, offset + 5), 10);
      if (!Number.isInteger(source) || !Number.isInteger(target) || source === target) continue;
      const key = `${source}:${target}`;
      directedConnections.set(key, (directedConnections.get(key) ?? 0) + 1);
    }
  }
  return {
    filename,
    title: title || filename,
    proteinAtoms,
    ligandOptions,
    directedConnections,
    defaultLigandId: ligandOptions[0]?.id ?? null,
  };
}

function prepareProtein(atoms) {
  const groups = new Map();
  for (const atom of atoms) {
    if (!groups.has(atom.residueKey)) groups.set(atom.residueKey, []);
    groups.get(atom.residueKey).push(atom);
  }
  const prepared = [];
  let residueId = 0;
  for (const residue of groups.values()) {
    const ca = residue.find((atom) => atom.atomName === "CA");
    const hasSidechain = residue.some((atom) => !BACKBONE_NAMES.has(atom.atomName));
    if (!ca && hasSidechain) continue;
    const anchor = ca?.coord ?? meanCoordinate(residue);
    for (const atom of residue) prepared.push({ ...atom, anchor, residueId });
    residueId += 1;
  }
  if (!prepared.length) throw new Error("PDB contains no usable protein residues.");
  return prepared;
}

function inferredLigandBonds(atoms, explicit) {
  const result = new Map(explicit);
  for (let left = 0; left < atoms.length; left += 1) {
    for (let right = left + 1; right < atoms.length; right += 1) {
      const key = `${left}:${right}`;
      if (result.has(key)) continue;
      const radius = (COVALENT_RADII.get(atoms[left].atomicNumber) ?? 0.85)
        + (COVALENT_RADII.get(atoms[right].atomicNumber) ?? 0.85);
      const separation = distance(atoms[left].coord, atoms[right].coord);
      if (separation > 0.55 && separation <= 1.22 * radius) result.set(key, 0);
    }
  }
  return result;
}

function displayTopology(coords, atomicNumbers, roles, residueIds, chainIds, atomNames, bonds) {
  const trace = [];
  const ca = atomNames.flatMap((name, atom) => name === 1 && residueIds[atom] >= 0 ? [atom] : []);
  for (let index = 1; index < ca.length; index += 1) {
    const left = ca[index - 1];
    const right = ca[index];
    if (chainIds[left] === chainIds[right] && distance(coords[left], coords[right]) < 4.5) {
      trace.push([left, right]);
    }
  }
  const sidechain = [];
  for (let left = 0; left < coords.length; left += 1) {
    if (residueIds[left] < 0) continue;
    for (let right = left + 1; right < coords.length; right += 1) {
      if (residueIds[left] !== residueIds[right]) continue;
      if (roles[left] !== ROLE_SIDECHAIN && roles[right] !== ROLE_SIDECHAIN) continue;
      const radius = (COVALENT_RADII.get(atomicNumbers[left]) ?? 0.85)
        + (COVALENT_RADII.get(atomicNumbers[right]) ?? 0.85);
      const separation = distance(coords[left], coords[right]);
      if (separation > 0.8 && separation <= 1.25 * radius) sidechain.push([left, right]);
    }
  }
  return {
    backbone_trace_pairs: trace,
    sidechain_bonds: sidechain,
    ligand_bonds: bonds
      .filter(({ left, right }) => roles[left] === ROLE_LIGAND || roles[right] === ROLE_LIGAND)
      .map(({ left, right }) => [left, right]),
    molecule_bonds: [],
  };
}

function assembleSample({
  id, label, coords, baseMeans, atomicNumbers, roles, residueTypes, atomNames,
  entityIds, residueIds, chainIds, bonds, topology = null, graphSource,
}) {
  const atoms = coords.length;
  const neighbors = Array.from({ length: atoms * MAX_NEIGHBORS }, () => [-1, -1]);
  const degree = new Uint8Array(atoms);
  for (const { left, right, type } of bonds) {
    if (left === right || left < 0 || right < 0 || left >= atoms || right >= atoms) {
      throw new Error("Molecular graph contains an invalid atom index.");
    }
    if (!Number.isInteger(type) || type < 0 || type > 3) {
      throw new Error("Molecular graph contains an unsupported bond category.");
    }
    for (const [query, key] of [[left, right], [right, left]]) {
      if (degree[query] >= MAX_NEIGHBORS) {
        throw new Error(`Atom ${query} exceeds the model's ${MAX_NEIGHBORS}-neighbor limit.`);
      }
      neighbors[query * MAX_NEIGHBORS + degree[query]] = [key, type];
      degree[query] += 1;
    }
  }
  const coordinateDesign = roles.map((role) => role === ROLE_SIDECHAIN || role === ROLE_LIGAND ? 1 : 0);
  const baseScales = roles.map((role) => role === ROLE_LIGAND ? 1 : role === ROLE_SIDECHAIN ? 0.5 : 0);
  return {
    format: "wsfmdock_webgpu_sample_v1",
    id,
    label,
    source: "custom",
    atoms,
    target_coords: coords,
    base_means: baseMeans,
    base_scales: baseScales,
    atomic_numbers: atomicNumbers,
    roles,
    residue_types: residueTypes,
    atom_names: atomNames,
    entity_ids: entityIds,
    coordinate_design: coordinateDesign,
    neighbors,
    residue_ids: residueIds,
    graph_source: graphSource,
    ...(topology ?? displayTopology(
      coords, atomicNumbers, roles, residueIds, chainIds, atomNames, bonds,
    )),
  };
}

function selectedLigandAtoms(structure, ligandId) {
  if (!ligandId) return [];
  if (ligandId === "__all__") return structure.ligandOptions.flatMap((option) => option.atoms);
  return structure.ligandOptions.find((option) => option.id === ligandId)?.atoms ?? [];
}

export function preparePdbSample(structure, ligandId = structure.defaultLigandId) {
  const protein = prepareProtein(structure.proteinAtoms);
  const ligand = selectedLigandAtoms(structure, ligandId);
  const center = meanCoordinate(protein);
  const coords = protein.map((atom) => subtract(atom.coord, center));
  const baseMeans = protein.map((atom) => subtract(atom.anchor, center));
  const atomicNumbers = protein.map((atom) => atom.atomicNumber);
  const roles = protein.map((atom) => BACKBONE_NAMES.has(atom.atomName) ? ROLE_BACKBONE : ROLE_SIDECHAIN);
  const residueTypes = protein.map((atom) => RESIDUE_INDEX.get(atom.residueName));
  const atomNames = protein.map((atom) => ATOM_NAME_INDEX.get(atom.atomName) ?? UNKNOWN_ATOM_NAME);
  const entityIds = protein.map(() => 0);
  const residueIds = protein.map((atom) => atom.residueId);
  const chainMap = new Map();
  const chainIds = protein.map((atom) => {
    if (!chainMap.has(atom.chain)) chainMap.set(atom.chain, chainMap.size);
    return chainMap.get(atom.chain);
  });
  const proteinSerials = new Map(protein.map((atom, index) => [atom.serial, index]));
  const ligandOffset = protein.length;
  const ligandSerials = new Map(ligand.map((atom, index) => [atom.serial, ligandOffset + index]));
  const explicit = new Map();
  for (let left = 0; left < ligand.length; left += 1) {
    for (let right = left + 1; right < ligand.length; right += 1) {
      const first = ligand[left].serial;
      const second = ligand[right].serial;
      const multiplicity = Math.max(
        structure.directedConnections.get(`${first}:${second}`) ?? 0,
        structure.directedConnections.get(`${second}:${first}`) ?? 0,
      );
      if (multiplicity) explicit.set(`${left}:${right}`, Math.min(multiplicity, 3) - 1);
    }
  }
  const ligandGraph = inferredLigandBonds(ligand, explicit);
  const ligandBonds = [...ligandGraph].map(([key, type]) => {
    const [left, right] = key.split(":").map(Number);
    return { left: ligandOffset + left, right: ligandOffset + right, type };
  });
  const attachmentBonds = [];
  for (const [ligandSerial, ligandIndex] of ligandSerials) {
    for (const [proteinSerial, proteinIndex] of proteinSerials) {
      const multiplicity = Math.max(
        structure.directedConnections.get(`${ligandSerial}:${proteinSerial}`) ?? 0,
        structure.directedConnections.get(`${proteinSerial}:${ligandSerial}`) ?? 0,
      );
      if (multiplicity) attachmentBonds.push({
        left: proteinIndex,
        right: ligandIndex,
        type: Math.min(multiplicity, 3) - 1,
      });
    }
  }
  const localBonds = ligandBonds.map(({ left, right, type }) => ({
    left: left - ligandOffset, right: right - ligandOffset, type,
  }));
  const ligandEntities = connectedComponents(ligand.length, localBonds);
  for (const atom of ligand) {
    coords.push(subtract(atom.coord, center));
    baseMeans.push([0, 0, 0]);
    atomicNumbers.push(atom.atomicNumber);
    roles.push(ROLE_LIGAND);
    residueTypes.push(UNKNOWN_RESIDUE);
    atomNames.push(UNKNOWN_ATOM_NAME);
    residueIds.push(-1);
    chainIds.push(-1);
  }
  entityIds.push(...ligandEntities.map((value) => value + 1));
  const selectedLabel = ligandId === "__all__"
    ? "all non-water components"
    : structure.ligandOptions.find((option) => option.id === ligandId)?.label;
  return assembleSample({
    id: `pdb-${structure.filename}`,
    label: selectedLabel ? `${structure.title} / ${selectedLabel}` : structure.title,
    coords,
    baseMeans,
    atomicNumbers,
    roles,
    residueTypes,
    atomNames,
    entityIds,
    residueIds,
    chainIds,
    bonds: [...ligandBonds, ...attachmentBonds],
    graphSource: explicit.size ? "PDB CONECT plus inferred connectivity" : "distance-inferred PDB connectivity",
  });
}

export function replaceLigand(sample, graph) {
  const keep = sample.roles.map((role, atom) => role !== ROLE_LIGAND ? atom : -1).filter((atom) => atom >= 0);
  const inverse = new Int32Array(sample.atoms).fill(-1);
  keep.forEach((atom, index) => { inverse[atom] = index; });
  const pick = (name) => keep.map((atom) => sample[name][atom]);
  const coords = pick("target_coords").map((point) => [...point]);
  const baseMeans = pick("base_means").map((point) => [...point]);
  const atomicNumbers = pick("atomic_numbers");
  const roles = pick("roles");
  const residueTypes = pick("residue_types");
  const atomNames = pick("atom_names");
  const entityIds = pick("entity_ids");
  const residueIds = pick("residue_ids");
  const ligandOffset = keep.length;
  const proteinEntityMaximum = entityIds.length ? Math.max(...entityIds) : 0;
  for (let atom = 0; atom < graph.atomicNumbers.length; atom += 1) {
    coords.push(graph.coordinates[atom]);
    baseMeans.push([0, 0, 0]);
    atomicNumbers.push(graph.atomicNumbers[atom]);
    roles.push(ROLE_LIGAND);
    residueTypes.push(UNKNOWN_RESIDUE);
    atomNames.push(UNKNOWN_ATOM_NAME);
    entityIds.push(proteinEntityMaximum + 1 + graph.entityIds[atom]);
    residueIds.push(-1);
  }
  const remapPairs = (pairs) => pairs.flatMap(([left, right]) => (
    inverse[left] >= 0 && inverse[right] >= 0 ? [[inverse[left], inverse[right]]] : []
  ));
  const bonds = graph.bonds.map(({ left, right, type }) => ({
    left: ligandOffset + left,
    right: ligandOffset + right,
    type,
  }));
  return assembleSample({
    id: `${sample.id}-smiles`,
    label: `${sample.label.split(" / ")[0]} / ${graph.canonicalSmiles}`,
    coords,
    baseMeans,
    atomicNumbers,
    roles,
    residueTypes,
    atomNames,
    entityIds,
    residueIds,
    chainIds: roles.map(() => 0),
    bonds,
    topology: {
      backbone_trace_pairs: remapPairs(sample.backbone_trace_pairs),
      sidechain_bonds: remapPairs(sample.sidechain_bonds),
      ligand_bonds: bonds.map(({ left, right }) => [left, right]),
      molecule_bonds: [],
    },
    graphSource: `RDKit ${graph.canonicalSmiles}`,
  });
}
