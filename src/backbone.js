const CELL = 1.45;

export function peptideBonds(sample) {
  const { atoms, roles, atom_names: names, atomic_numbers: numbers,
    entity_ids: entities, residue_ids: residues, target_coords: coords } = sample;
  const grid = new Map();
  const key = (cell) => cell.join(',');
  const cell = (point) => point.map(v => Math.floor(v / CELL));
  for (let n = 0; n < atoms; n += 1) {
    if (roles[n] !== 1 || names[n] !== 0 || numbers[n] !== 7) continue;
    const location = key(cell(coords[n]));
    if (!grid.has(location)) grid.set(location, []);
    grid.get(location).push(n);
  }
  const candidates = [], carbons = new Map(), nitrogens = new Map();
  for (let c = 0; c < atoms; c += 1) {
    if (roles[c] !== 1 || names[c] !== 2 || numbers[c] !== 6) continue;
    const origin = cell(coords[c]);
    for (let x = -1; x <= 1; x += 1) for (let y = -1; y <= 1; y += 1) for (let z = -1; z <= 1; z += 1) {
      for (const n of grid.get(key([origin[0] + x, origin[1] + y, origin[2] + z])) ?? []) {
        if (entities[c] !== entities[n] || residues[c] === residues[n]) continue;
        const difference = coords[c].map((v, i) => Math.fround(Math.fround(v) - Math.fround(coords[n][i])));
        const distance = Math.hypot(...difference);
        if (distance < 1.20 || distance > 1.45) continue;
        candidates.push({ left: c, right: n, type: 0 });
        carbons.set(c, (carbons.get(c) ?? 0) + 1);
        nitrogens.set(n, (nitrogens.get(n) ?? 0) + 1);
      }
    }
  }
  return candidates.filter(({ left, right }) => carbons.get(left) === 1 && nitrogens.get(right) === 1);
}

export function withBackboneGraph(sample) {
  if (sample.backbone_graph_complete) return sample;
  const neighbors = sample.neighbors.map(edge => [...edge]);
  const display = [...(sample.backbone_bonds ?? sample.backbone_trace_pairs ?? [])].map(pair => [...pair]);
  const shown = new Set(display.map(([a, b]) => `${Math.min(a, b)}:${Math.max(a, b)}`));
  for (const { left, right } of peptideBonds(sample)) {
    for (const [a, b] of [[left, right], [right, left]]) {
      const offset = a * 10;
      const slots = neighbors.slice(offset, offset + 10);
      const old = slots.find(([other]) => other === b);
      if (old && old[1] !== 0) throw new Error('Peptide bond category conflict.');
      if (old) continue;
      const free = slots.findIndex(([other]) => other < 0);
      if (free < 0) throw new Error('Peptide graph exceeds ten neighbors.');
      neighbors[offset + free] = [b, 0];
    }
    const key = `${Math.min(left, right)}:${Math.max(left, right)}`;
    if (!shown.has(key)) { display.push([left, right]); shown.add(key); }
  }
  return { ...sample, neighbors, backbone_bonds: display, backbone_trace_pairs: display,
    backbone_sigma: sample.backbone_sigma ?? new Array(sample.atoms).fill(0), backbone_graph_complete: true };
}
