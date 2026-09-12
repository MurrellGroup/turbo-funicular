import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { parsePdb, preparePdbSample, ATOM_NAMES } from '../src/prep.js';
import { AMINO_ACIDS, AA_NAMES, parseSequences, proteinChains, mutateSample, resolveEdits } from '../src/sequence.js';
import { alignChain } from '../src/alignment.js';
import { validateSample } from '../src/sample.js';
import { withBackboneGraph } from '../src/backbone.js';
import proteinGraph from '../src/protein_graph.json' with { type: 'json' };
import { miniPdb } from './fixtures.mjs';

const base = preparePdbSample(parsePdb(miniPdb()), null);
const residue = proteinChains(base)[0].residues[0];

test('FASTA and plain sequences are validated without silent unknown substitutions', () => {
  assert.deepEqual(parseSequences('>one\nARN-X\n>two\nace'), [{ name: 'one', sequence: 'ARNX' }, { name: 'two', sequence: 'ACE' }]);
  for (const text of ['', '>empty', 'AC1E', 'XYZ', '>a\n>b\nAAA']) assert.throws(() => parseSequences(text));
});

for (const aa of AMINO_ACIDS) test(`Full ${AA_NAMES[aa]} graph and immutable backbone`, () => {
  const before = JSON.stringify(base);
  const { sample } = mutateSample(base, new Map([[residue.id, aa]]), 23);
  validateSample(sample);
  assert.equal(JSON.stringify(base), before);
  const r = proteinChains(sample)[0].residues[0];
  assert.equal(r.aa, aa);
  const names = new Map(r.atoms.map(i => [ATOM_NAMES[sample.atom_names[i]], i]));
  const expected = proteinGraph[AA_NAMES[aa]].filter(([a, b]) => a !== 'OXT' && b !== 'OXT');
  assert.deepEqual([...names.keys()].sort(), [...new Set(expected.flatMap(([a, b]) => [a, b]))].sort());
  const edges = [];
  for (const i of r.atoms) for (const [j, type] of sample.neighbors.slice(i * 10, i * 10 + 10)) {
    if (j > i) edges.push([ATOM_NAMES[sample.atom_names[i]], ATOM_NAMES[sample.atom_names[j]], type]);
  }
  const normalized = edges => edges.map(([a, b, t]) => `${[a, b].sort().join(':')}:${t}`).sort();
  assert.deepEqual(normalized(edges), normalized(expected));
  for (const name of ['N', 'CA', 'C', 'O']) {
    const old = residue.atoms.find(i => ATOM_NAMES[base.atom_names[i]] === name), next = names.get(name);
    assert.deepEqual(sample.target_coords[next], base.target_coords[old]);
    assert.equal(sample.coordinate_design[next], 0);
  }
  for (const i of r.atoms.filter(i => sample.roles[i] === 2)) {
    assert.deepEqual(sample.base_means[i], sample.target_coords[names.get('CA')]);
    assert.equal(sample.initial_scales[i], 0.5);
  }
  assert.equal(proteinChains(sample)[0].residues[1].aa, 'G');
});

test('Random identities are per seed, reproducible, canonical, and do not mutate the edit map', () => {
  const edits = new Map([[residue.id, 'X']]);
  const outcomes = Array.from({ length: 100 }, (_, i) => resolveEdits(base, edits, i).get(residue.id));
  assert.equal(new Set(outcomes).size, 20);
  assert.deepEqual(resolveEdits(base, edits, 42), resolveEdits(base, edits, 42));
  assert.equal(edits.get(residue.id), 'X');
  assert.throws(() => resolveEdits(base, new Map([[500, 'W']]), 1), /removed residue/);
});

test('Peptide bonds survive amino-acid edits without blocking them', () => {
  const s = structuredClone(base);
  s.target_coords[5] = s.target_coords[2].map((v, k) => v + (k === 0 ? 1.33 : 0));
  delete s.backbone_graph_complete;
  const linked = withBackboneGraph(s);
  assert.ok(linked.neighbors.slice(20, 30).some(([j, t]) => j === 5 && t === 0));
  for (const aa of AMINO_ACIDS) {
    const result = mutateSample(linked, new Map([[residue.id, aa]]), 23).sample;
    validateSample(result);
    const n = result.atom_labels.indexOf(linked.atom_labels[5]);
    const changedC = proteinChains(result)[0].residues[0].atoms.find(i => result.atom_names[i] === 2);
    assert.ok(result.neighbors.slice(changedC * 10, changedC * 10 + 10).some(([j, t]) => j === n && t === 0));
    assert.deepEqual(result.target_coords[changedC], linked.target_coords[2]);
    assert.deepEqual(result.target_coords[n], linked.target_coords[5]);
  }
});

test('Packaged examples can run campaigns without optional PDB chain labels', () => {
  const root = new URL('../public/assets/samples/', import.meta.url);
  const catalog = JSON.parse(readFileSync(new URL('catalog.json', root)));
  for (const entry of catalog.samples) {
    const sample = JSON.parse(readFileSync(new URL(entry.file, root)));
    const result = mutateSample(sample, new Map(), 1);
    assert.equal(result.sample, sample);
    assert.equal(result.resolved.size, 0);
    const chains = proteinChains(sample);
    const entities = new Set(sample.entity_ids.filter((_, i) => [1, 2].includes(sample.roles[i])));
    assert.equal(chains.length, entities.size);
  }
});

test('An attachment is protected; unrelated mutation remaps it with exact type and ligand labels', () => {
  const s = structuredClone(base);
  s.neighbors[4 * 10 + 1] = [5, 0]; s.neighbors[5 * 10 + 1] = [4, 0];
  assert.throws(() => mutateSample(s, new Map([[0, 'W']]), 1), /external covalent/);
  const g = structuredClone(base);
  // Append a free component, with unknown residue type and an exact atom label.
  const i = g.atoms++;
  for (const [key, value] of Object.entries({ target_coords: [10, 0, 0], base_means: [0, 0, 0], base_scales: 1, initial_scales: 10,
    atomic_numbers: 6, roles: 3, residue_types: 20, atom_names: 38, entity_ids: 1, coordinate_design: 1,
    residue_ids: -1, chain_ids: -1, atom_labels: 'C1|NAG|B|101|' })) g[key].push(value);
  g.neighbors.push(...Array.from({ length: 10 }, () => [-1, -1]));
  g.neighbors[5 * 10 + 1] = [i, 0]; g.neighbors[i * 10] = [5, 0];
  g.ligand_bonds.push([5, i]);
  const m = mutateSample(g, new Map([[0, 'W']]), 1).sample;
  validateSample(m);
  const a = m.atom_labels.indexOf(g.atom_labels[5]), b = m.atom_labels.indexOf('C1|NAG|B|101|');
  assert.ok(a >= 0 && b >= 0);
  assert.ok(m.neighbors.slice(a * 10, a * 10 + 10).some(([j, t]) => j === b && t === 0));
  assert.deepEqual(m.target_coords[b], g.target_coords[i]);
  assert.equal(m.entity_ids[b], 1);
});

function chain(sequence) { return { id: 'A', sequence, residues: [...sequence].map((aa, id) => ({ id, aa, number: String(id + 1) })) }; }

test('Local affine BLOSUM62 maps substitutions and does not map terminal overhangs or indels', () => {
  const reference = 'MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE';
  const query = 'GGGG' + reference.slice(0, 15) + 'V' + reference.slice(16, 25) + 'GGGG' + reference.slice(25) + 'PPPP';
  const r = alignChain(chain(reference), query);
  assert.equal(r.aligned, reference.length);
  const differing = r.columns.filter(c => c.mapped && c.reference !== c.input);
  assert.equal(differing.length, 1); assert.equal(differing[0].residue.id, 15); assert.equal(differing[0].input, 'V');
  assert.equal(r.columns.filter(c => c.reference === '-').length, 12);
  assert.equal(r.columns.filter(c => c.residue).map(c => c.reference).join(''), reference);
  assert.equal(r.columns.filter(c => c.input !== '-').map(c => c.input).join(''), query);
  assert.equal(alignChain(chain('CCCC'), 'AAAA').aligned, 0);
});
