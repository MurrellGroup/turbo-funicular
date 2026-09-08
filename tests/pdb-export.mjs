import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import initRDKitModule from '../vendor/rdkit/RDKit_minimal.cjs';
import { parsePdb, preparePdbSample, replaceLigand, ATOM_NAMES } from '../src/prep.js';
import { mutateSample, AMINO_ACIDS, AA_NAMES } from '../src/sequence.js';
import { parseCcdGraph, trainingGraphFromCcd } from '../src/ccd.js';
import { graphFromSmiles } from '../src/chemistry.js';
import { exportPdb } from '../src/pdb-export.js';
import { miniPdb, miniCcd, mpnnFixture } from './fixtures.mjs';

const rdkit = await initRDKitModule();
const graph = trainingGraphFromCcd(rdkit, parseCcdGraph(miniCcd(), 'BEN'));
const make = sample => ({ sample, seed: 31, coords: Float32Array.from(sample.target_coords.flat(), (x, i) =>
  sample.roles[Math.floor(i / 3)] === 1 ? x : x + 0.125 * (1 + i % 3)) });
const records = text => text.split('\n').filter(l => /^(ATOM  |HETATM)/.test(l)).map(l => ({
  serial: Number(l.slice(6, 11)), name: l.slice(12, 16).trim(), residue: l.slice(17, 20).trim(),
  chain: l.slice(21, 22).trim(), number: Number(l.slice(22, 26)), insertion: l.slice(26, 27).trim(),
  xyz: [30, 38, 46].map(i => Number(l.slice(i, i + 8))), element: l.slice(76, 78).trim(),
}));

test('All 20 AA identities export their rebuilt atoms and sampled coordinates', () => {
  const base = preparePdbSample(parsePdb(mpnnFixture()), null);
  for (const aa of AMINO_ACIDS) {
    const sample = mutateSample(base, new Map([[2, aa]]), 1).sample, result = make(sample), pdb = exportPdb(result);
    const rows = records(pdb); assert.equal(rows.length, sample.atoms);
    assert.ok(pdb.trimEnd().endsWith('END')); assert.ok(pdb.split('\n').slice(0, -1).every(l => l.length === 80));
    for (const [i, row] of rows.entries()) {
      const label = sample.atom_labels[i].split('|');
      assert.deepEqual([row.name, row.residue, row.chain, row.number, row.insertion], [label[0], label[1], label[2], Number(label[3]), label[4]]);
      for (let d = 0; d < 3; d++) assert.ok(Math.abs(row.xyz[d] - (result.coords[i * 3 + d] + sample.coordinate_origin[d])) <= 0.000501);
    }
    const selected = rows.filter(r => r.chain === 'A' && r.number === 1 && r.insertion === 'A');
    assert.ok(selected.every(r => r.residue === AA_NAMES[aa]));
    assert.deepEqual(selected.map(r => r.name), sample.atom_names.filter((_, i) => sample.residue_ids[i] === 2).map(i => ATOM_NAMES[i]));
    assert.equal(pdb.split('\n').filter(l => l.startsWith('TER')).length, 2);
  }
});

test('Native labels, original backbone frame, exact CONECT and attachment LINK survive export', () => {
  const original = parsePdb(miniPdb({ includeLink: true }));
  const base = preparePdbSample(original, original.defaultLigandId, new Map([['BEN', graph]]));
  const result = make(mutateSample(base, new Map([[1, 'W']]), 1).sample), pdb = exportPdb(result), rows = records(pdb);
  const bySerial = new Map(rows.map(r => [r.serial, r]));
  const labels = rows.map(r => `${r.name}|${r.residue}|${r.chain}|${r.number}|${r.insertion}`);
  const expected = new Set(), actual = new Set();
  result.sample.neighbors.forEach(([j], k) => { const i = Math.floor(k / 10); if (j > i) expected.add([result.sample.atom_labels[i], result.sample.atom_labels[j]].sort().join('~')); });
  for (const line of pdb.split('\n').filter(l => l.startsWith('CONECT'))) {
    const serials = line.slice(6).match(/.{1,5}/g).map(x => Number(x.trim())).filter(Boolean);
    for (const j of serials.slice(1)) {
      const key = serial => { const r = bySerial.get(serial); return `${r.name}|${r.residue}|${r.chain}|${r.number}|${r.insertion}`; };
      actual.add([key(serials[0]), key(j)].sort().join('~'));
    }
  }
  assert.deepEqual(actual, expected); assert.equal(parsePdb(pdb).links.length, 1);
  assert.equal(new Set(labels).size, rows.length);
  for (const atom of original.proteinAtoms.filter(a => ['N', 'CA', 'C', 'O'].includes(a.atomName))) {
    const row = rows.find(r => r.chain === atom.chain && r.number === atom.residueNumber && r.name === atom.atomName);
    atom.coord.forEach((x, d) => assert.ok(Math.abs(row.xyz[d] - x) <= 0.000501));
  }
});

test('Replacement ligands and packaged samples receive unique generated labels', async () => {
  const base = preparePdbSample(parsePdb(miniPdb()), null);
  const sample = replaceLigand(base, graphFromSmiles(rdkit, 'CC(=O)O'));
  const rows = records(exportPdb(make(sample)));
  assert.equal(rows.filter(r => r.residue === 'LIG').length, 4);
  assert.deepEqual(sample.coordinate_origin, base.coordinate_origin);
  for (const file of ['glycan-attached-5.json', 'glycan-free-100.json', 'plinder-424.json']) {
    const sample = JSON.parse(await readFile(new URL(`../public/assets/samples/${file}`, import.meta.url)));
    assert.equal(records(exportPdb(make(sample))).length, sample.atoms);
  }
});

test('Long assembly chain IDs map uniquely; invalid fields are rejected rather than truncated', () => {
  const sample = preparePdbSample(parsePdb(mpnnFixture()), null);
  sample.atom_labels = sample.atom_labels.map(l => l.replace('|B|', '|A-2|'));
  const pdb = exportPdb(make(sample));
  assert.ok(pdb.includes('REMARK 900 CHAIN B ORIGINAL "A-2"'));
  assert.deepEqual([...new Set(records(pdb).map(r => r.chain))], ['A', 'B']);
  const result = make(sample); result.coords[0] = NaN;
  assert.throws(() => exportPdb(result), /non-finite/);
  result.coords[0] = 1e30; assert.throws(() => exportPdb(result), /limits/);
  sample.atom_labels[0] = sample.atom_labels[0].replace('|-2|', '|10000|');
  assert.throws(() => exportPdb(make(sample)), /Residue number/);
});

test('Independent Gemmi parser validates coordinates, chain boundaries and attachment', { skip: !process.env.GEMMI_PYTHON }, () => {
  const base = preparePdbSample(parsePdb(miniPdb({ includeLink: true })), '__all__', new Map([['BEN', graph]]));
  const sample = mutateSample(base, new Map([[1, 'K']]), 2).sample, pdb = exportPdb(make(sample));
  const code = 'import gemmi,json,sys\ns=gemmi.read_pdb_string(sys.stdin.read())\nprint(json.dumps({"atoms":[[c.name,r.name,r.seqid.num,r.seqid.icode.strip(),a.name,a.element.name.upper(),[a.pos.x,a.pos.y,a.pos.z]] for c in s[0] for r in c for a in r],"links":len(s.connections)}))';
  const run = spawnSync(process.env.GEMMI_PYTHON, ['-c', code], { input: pdb, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const parsed = JSON.parse(run.stdout);
  assert.equal(parsed.links, 1);
  assert.deepEqual(parsed.atoms, records(pdb).map(r => [r.chain, r.residue, r.number, r.insertion, r.name, r.element, r.xyz]));
});
