import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import init from '../vendor/rdkit/RDKit_minimal.cjs';
import { parseMmcif } from '../src/mmcif.js';
import { loadCcdGraphs } from '../src/ccd.js';
import { selectedLigandOptions, preparePdbSample, replaceLigand } from '../src/prep.js';
import { graphFromSmiles } from '../src/chemistry.js';
import { validateSample } from '../src/sample.js';

const rdkit = await init();
async function fixture(name) {
  try { return await readFile(`/tmp/${name}`, 'utf8'); }
  catch {
    const response = await fetch(`https://files.rcsb.org/download/${name}`);
    if (!response.ok) throw new Error(`Fixture download failed: ${name}`);
    const text = await response.text();
    await writeFile(`/tmp/${name}`, text);
    return text;
  }
}
const assembly = await fixture('4byh-assembly1.cif');
const deposited = await fixture('4byh.cif');
const structure = parseMmcif(assembly, '4byh', deposited);
assert.equal(structure.proteinAtoms.length, 3330);
assert.equal(structure.links.filter(([a, b]) => a.protein !== b.protein).length, 2);
const id = structure.ligandOptions.find(o => o.atoms[0].rawResidue === 'NAG').id;
const options = selectedLigandOptions(structure, id);
assert.equal(options.length, 10);
const graphs = await loadCcdGraphs(options.map(o => o.atoms[0].rawResidue), rdkit);
const sample = preparePdbSample(structure, id, graphs, rdkit);
validateSample(sample);
const edges = sample.neighbors.flatMap(([right, type], slot) => {
  const left = Math.floor(slot / 10); return right > left ? [{ left, right, type }] : [];
});
const attachments = edges.filter(b => (sample.roles[b.left] === 3) !== (sample.roles[b.right] === 3));
assert.equal(attachments.length, 1);
assert.ok(attachments[0].right - attachments[0].left > 255);
assert.equal(new Set(sample.entity_ids.slice(3330)).size, 1);
assert.equal(new Set(sample.entity_ids.slice(0, 3330)).size, 2);
assert.equal(new Set(sample.entity_ids).size, 3);
assert.ok(sample.initial_scales.slice(3330).every(v => v === 10));
assert.ok(sample.base_scales.slice(3330).every(v => v === 1));
const replaced = replaceLigand(sample, graphFromSmiles(rdkit, 'CCO'));
validateSample(replaced);
assert.deepEqual(replaced.entity_ids.slice(0, 3330), sample.entity_ids.slice(0, 3330));
for (let i = 0; i < 3330; i += 1) {
  assert.deepEqual(replaced.neighbors.slice(i * 10, i * 10 + 10).filter(([j]) => j >= 0),
    sample.neighbors.slice(i * 10, i * 10 + 10).filter(([j]) => j >= 0 && j < 3330));
}
const bad = structuredClone(structure);
bad.links.find(([a, b]) => a.protein !== b.protein)[0] = {
  ...bad.links.find(([a, b]) => a.protein !== b.protein)[0], atomName: 'MISSING',
};
assert.throws(() => preparePdbSample(bad, id, graphs, rdkit), /endpoint/);
await writeFile('/tmp/webgpu-4byh-graph.json', JSON.stringify({ sample,
  atoms: [...structure.proteinAtoms, ...options.flatMap(o => o.atoms)], edges }));
console.log(JSON.stringify({ atoms: sample.atoms, selectedResidues: options.length,
  ligandBonds: sample.ligand_bonds.length, attachments: attachments.length,
  attachmentSpan: attachments[0].right - attachments[0].left, totalBonds: edges.length }));
