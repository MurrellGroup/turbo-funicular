export function validateSample(sample, maximumAtoms = Number.POSITIVE_INFINITY) {
  const n = sample?.atoms;
  if (!Number.isInteger(n) || n < 1) throw new Error("Sample must contain at least one atom.");
  if (n > maximumAtoms) {
    throw new Error(`This WebGPU adapter supports at most ${maximumAtoms.toLocaleString()} atoms for this model; the prepared structure has ${n.toLocaleString()}.`);
  }
  for (const name of [
    "base_scales", "atomic_numbers", "roles", "residue_types", "atom_names",
    "entity_ids", "coordinate_design",
  ]) {
    if (sample[name]?.length !== n) throw new Error(`${name} must have one value per atom.`);
  }
  for (const name of ["target_coords", "base_means"]) {
    if (sample[name]?.length !== n || sample[name].some((point) => point.length !== 3)) {
      throw new Error(`${name} must have one XYZ coordinate per atom.`);
    }
  }
  if (sample.neighbors?.length !== n * 10 || sample.neighbors.some((edge) => edge.length !== 2)) {
    throw new Error("neighbors must contain exactly ten sparse pair slots per atom.");
  }
  if (sample.atomic_numbers.some((value) => !Number.isInteger(value) || value < 1 || value >= 128)) {
    throw new Error("Sample contains an unsupported atomic number.");
  }
  if (sample.entity_ids.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error("Sample entity IDs must be non-negative integers.");
  }
  for (const edge of sample.neighbors) {
    if (edge[0] === -1 && edge[1] === -1) continue;
    if (!Number.isInteger(edge[0]) || edge[0] < 0 || edge[0] >= n
      || !Number.isInteger(edge[1]) || edge[1] < 0 || edge[1] > 3) {
      throw new Error("Sample contains an invalid sparse molecular-graph edge.");
    }
  }
}
