import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { peptideBonds, withBackboneGraph } from '../src/backbone.js';
import { validateSample } from '../src/sample.js';

function atoms(points, names, entities, residues) {
  return { atoms: points.length, target_coords: points, atom_names: names,
    roles: names.map(() => 1), atomic_numbers: names.map(n => n === 0 ? 7 : 6),
    entity_ids: entities, residue_ids: residues };
}

test('Peptide candidates respect distance, chain, residue and unique endpoints', () => {
  const sample = atoms([[0, 0, 0], [1.33, 0, 0]], [2, 0], [0, 0], [0, 1]);
  assert.equal(peptideBonds(sample).length, 1);
  assert.equal(peptideBonds({ ...sample, entity_ids: [0, 1] }).length, 0);
  assert.equal(peptideBonds({ ...sample, residue_ids: [0, 0] }).length, 0);
  for (const distance of [1.19, 1.46]) {
    assert.equal(peptideBonds({ ...sample, target_coords: [[0, 0, 0], [distance, 0, 0]] }).length, 0);
  }
  assert.equal(peptideBonds(atoms([[0, 0, 0], [1.33, 0, 0], [-1.33, 0, 0]],
    [2, 0, 0], [0, 0, 0], [0, 1, 2])).length, 0);
  assert.equal(peptideBonds({ ...sample, roles: [3, 3] }).length, 0);
});

test('Browser peptide pairs exactly match the Python-exported native graphs', async () => {
  for (const filename of ['plinder-424.json', 'glycan-attached-5.json']) {
    const sample = JSON.parse(await readFile(new URL(`../public/assets/samples/${filename}`, import.meta.url)));
    const expected = [];
    for (let a = 0; a < sample.atoms; a += 1) {
      if (sample.roles[a] !== 1 || sample.atom_names[a] !== 2) continue;
      for (const [b, type] of sample.neighbors.slice(a * 10, a * 10 + 10)) {
        if (b >= 0 && sample.roles[b] === 1 && sample.atom_names[b] === 0
          && sample.residue_ids[a] !== sample.residue_ids[b]) expected.push({ left: a, right: b, type });
      }
    }
    assert(expected.length > 100);
    assert.deepEqual(peptideBonds(sample), expected);
    const result = withBackboneGraph({ ...sample, backbone_graph_complete: false });
    assert.deepEqual(result.neighbors, sample.neighbors);
    validateSample(result);
    const broken = structuredClone(result);
    const bb = broken.roles.indexOf(1);
    broken.coordinate_design[bb] = 1;
    assert.throws(() => validateSample(broken), /conditioning sigma/);
    broken.backbone_sigma[bb] = 5;
    validateSample(broken);
    broken.base_scales[bb] = 1;
    assert.throws(() => validateSample(broken), /zero additional noise/);
  }
});
