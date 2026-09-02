function connectedComponents(atoms, bonds) {
  const parent = Array.from({ length: atoms }, (_, index) => index);
  const find = (value) => {
    while (parent[value] !== value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  for (const { left, right } of bonds) {
    const first = find(left);
    const second = find(right);
    if (first !== second) parent[second] = first;
  }
  const ids = new Map();
  return parent.map((_, atom) => {
    const root = find(atom);
    if (!ids.has(root)) ids.set(root, ids.size);
    return ids.get(root);
  });
}

function molblockCoordinates(block, atoms) {
  const lines = block.replaceAll("\r", "").split("\n");
  const counts = lines.findIndex((line) => line.includes("V2000"));
  if (counts < 0 || Number.parseInt(lines[counts].slice(0, 3), 10) !== atoms) {
    return Array.from({ length: atoms }, () => [0, 0, 0]);
  }
  return lines.slice(counts + 1, counts + 1 + atoms).map((line) => [
    Number.parseFloat(line.slice(0, 10)),
    Number.parseFloat(line.slice(10, 20)),
    Number.parseFloat(line.slice(20, 30)),
  ]);
}

function centered(values) {
  if (!values.length) return values;
  const mean = [0, 0, 0];
  for (const point of values) {
    for (let axis = 0; axis < 3; axis += 1) mean[axis] += point[axis] / values.length;
  }
  return values.map((point) => point.map((value, axis) => value - mean[axis]));
}

function bondType(order, aromatic) {
  if (aromatic || Math.abs(order - 1.5) < 1e-6) return 3;
  if (order === 1) return 0;
  if (order === 2) return 1;
  if (order === 3) return 2;
  throw new Error(`Unsupported SMILES bond order: ${order}`);
}

export function graphFromSmiles(rdkit, smiles) {
  const input = smiles.trim();
  if (!input) throw new Error("Enter a SMILES string first.");
  let molecule;
  try {
    molecule = rdkit.get_mol(input, JSON.stringify({ removeHs: true, sanitize: true }));
    if (!molecule || !molecule.is_valid()) throw new Error("RDKit rejected the SMILES string.");
    const data = JSON.parse(molecule.get_json());
    const record = data.molecules?.[0];
    if (!record?.atoms?.length) throw new Error("SMILES contains no atoms.");
    const defaultAtomicNumber = data.defaults?.atom?.z ?? 6;
    const defaultBondOrder = data.defaults?.bond?.bo ?? 1;
    const atomicNumbers = record.atoms.map((atom) => atom.z ?? defaultAtomicNumber);
    if (atomicNumbers.some((value) => !Number.isInteger(value) || value <= 1 || value >= 128)) {
      throw new Error("SMILES must resolve to supported heavy atoms with atomic numbers below 128.");
    }
    const representation = record.extensions?.find(
      (extension) => extension.name === "rdkitRepresentation",
    );
    const aromatic = new Set(representation?.aromaticBonds ?? []);
    const bonds = (record.bonds ?? []).map((bond, index) => ({
      left: bond.atoms[0],
      right: bond.atoms[1],
      type: bondType(bond.bo ?? defaultBondOrder, aromatic.has(index)),
    }));
    const coordinates = centered(molblockCoordinates(
      molecule.get_new_coords(true),
      atomicNumbers.length,
    ));
    return {
      atomicNumbers,
      bonds,
      coordinates,
      entityIds: connectedComponents(atomicNumbers.length, bonds),
      canonicalSmiles: molecule.get_smiles(),
    };
  } catch (error) {
    if (error instanceof Error) throw error;
    throw new Error("RDKit could not parse the SMILES string.");
  } finally {
    molecule?.delete();
  }
}

export { connectedComponents };
