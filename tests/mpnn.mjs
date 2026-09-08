import assert from 'node:assert/strict';
import test from 'node:test';
import { parsePDB } from 'proteinmpnn-web/structure';
import { parsePdb, preparePdbSample } from '../src/prep.js';
import { mpnnInput, mpnnConstraints, proposalEdits } from '../src/mpnn-input.js';
import { proteinChains } from '../src/sequence.js';
import { mpnnFixture } from './fixtures.mjs';

test('All chain, gap, insertion, mask and index arrays match upstream parsePDB', () => {
  const pdb = mpnnFixture(), sample = preparePdbSample(parsePdb(pdb), null);
  const residues = proteinChains(sample).flatMap(c => c.residues);
  const edits = new Map([[residues[2].id, 'X'], [residues[7].id, 'X'], [residues[0].id, 'W']]);
  const input = mpnnInput(sample, edits), upstream = parsePDB(pdb);
  for (const key of ['chainIds', 'residueNumbers', 'insertionCodes', 'residueIndices', 'chainEncoding', 'mask']) assert.deepEqual(input.structure[key], upstream[key], key);
  assert.deepEqual([...new Set(input.structure.chainIds)], ['A', 'B']);
  assert.equal(input.mapping.filter(r => r.residueId === null).length, 4);
  assert.equal(input.structure.sequence[0], 'W');
  assert.equal(input.mapping[input.designPositions[0]].number, 1);
  assert.equal(input.mapping[input.designPositions[0]].insertionCode, 'A');
  assert.equal(input.mapping[input.designPositions[1]].number, 104);
  assert.equal(input.mapping[input.designPositions[1]].chain, 'B');
  const seq = [...input.structure.sequence]; for (const p of input.designPositions) seq[p] = 'K';
  const out = proposalEdits(input, { sequence: seq.join('') }, edits);
  assert.equal(out.get(residues[0].id), 'W'); assert.equal(out.get(residues[2].id), 'K'); assert.equal(out.get(residues[7].id), 'K');
  seq[0] = 'V'; assert.throws(() => proposalEdits(input, { sequence: seq.join('') }, edits), /fixed residue/);
});

test('Incomplete backbone stays masked, CA models retain valid CA, no X silently dropped', () => {
  const pdb = mpnnFixture().split('\n').filter(l => !(l.slice(21, 22) === 'A' && l.slice(26, 27) === 'A' && l.slice(12, 16).trim() === 'O')).join('\n');
  const sample = preparePdbSample(parsePdb(pdb), null), edits = new Map([[2, 'X']]);
  assert.throws(() => mpnnInput(sample, edits), /complete N\/CA\/C\/O/);
  const ca = mpnnInput(sample, edits, 'ca');
  assert.ok(ca.structure.mask[ca.designPositions[0]]);
  assert.equal(ca.structure.atomMode, 'ca');
});

test('Exact object references, insertion codes, all chains, hard X exclusion and fixed-mask protection', () => {
  const sample = preparePdbSample(parsePdb(mpnnFixture()), null), input = mpnnInput(sample, new Map([[2, 'X'], [7, 'X']]));
  const c = mpnnConstraints(input, { advanced: {
    biasByResidue: [{ position: { chain: 'A', number: 1, insertionCode: 'A' }, bias: { W: 2 } }],
    omitByResidue: [{ position: { chain: 'B', number: 104 }, aminoAcids: 'C' }],
    tiedPositions: [[{ chain: 'A', number: 1, insertionCode: 'A' }, { chain: 'B', number: 104 }]],
  } });
  assert.deepEqual(c.designChains, ['A', 'B']); assert.deepEqual(c.designPositions, input.designPositions);
  assert.equal(c.biasByResidue[0].position, input.designPositions[0]);
  assert.ok(c.omitByResidue.every(r => r.aminoAcids.includes('X')));
  assert.deepEqual(c.tiedPositions[0], input.designPositions);
  const duplicate = mpnnConstraints(input, { advanced: { omitByResidue: [
    { position: input.designPositions[0], aminoAcids: 'C' }, { position: input.designPositions[0], aminoAcids: 'P' },
  ] } });
  assert.equal(duplicate.omitByResidue[0].aminoAcids, 'CPX');
  assert.throws(() => mpnnConstraints(input, { advanced: { designPositions: [0] } }), /Unsupported/);
  assert.throws(() => mpnnConstraints(input, { advanced: { biasByResidue: [{ position: { chain: 'gone', number: 1 }, bias: { A: 1 } }] } }), /not found/);
});

test('Blank chain is distinct from named underscore, and removed chains stay removed', () => {
  const structure = parsePdb(mpnnFixture());
  structure.proteinAtoms = structure.proteinAtoms.filter(a => a.chain === 'A');
  const sample = preparePdbSample(structure, null), input = mpnnInput(sample, new Map([[0, 'X']]));
  assert.deepEqual([...new Set(input.structure.chainIds)], ['A']);
  sample.atom_labels = sample.atom_labels.map(label => label.replace('|A|', '||'));
  const blank = mpnnInput(sample, new Map([[0, 'X']]));
  assert.equal(blank.mapping[0].chain, ''); assert.equal(blank.structure.chainIds[0], '_');
  assert.equal(mpnnConstraints(blank, { advanced: { biasByResidue: [{ position: { chain: '', number: -2 }, bias: { A: 1 } }] } }).biasByResidue[0].position, 0);
});
