import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { MolecularViewer } from '../src/viewer.js';
import { viewerSample } from './viewer-fixture.mjs';

// Exercise real Three.js geometry/materials without requiring a WebGL context.
function viewer(t, sample = viewerSample()) {
  const v = Object.assign(Object.create(MolecularViewer.prototype), {
    scene: new THREE.Scene(), group: null, sample: null, referenceVisible: false,
    proteinColorMode: 'role', ligandColorsByElement: true,
    highlightedResidues: new Set(), highlightedLigandAtoms: new Set(),
    chainColors: new Map(), proteinChainIds: new Set(),
  });
  v.setSample(sample, new Float32Array(sample.target_coords.flat()), true);
  t.after(() => v.clear());
  return v;
}

function meshColor(mesh, index) {
  const color = new THREE.Color();
  mesh.getColorAt(index, color);
  return color.multiply(mesh.material.color).getHex();
}

function ligandColor(v, atom) {
  const { mesh, atoms } = v.ligandMeshes.find(item => item.atoms.includes(atom));
  return meshColor(mesh, atoms.indexOf(atom));
}

const chainColor = (v, atom) => v.chainColors.get(v.atomChainKeys[atom]).getHex();

test('default ligand element colors and bond colors are unchanged', t => {
  const v = viewer(t);
  assert.equal(ligandColor(v, 4), 0xc7cdcf);
  assert.equal(ligandColor(v, 5), 0xf06560);
  assert.equal(ligandColor(v, 7), 0x668be3);
  assert.equal(ligandColor(v, 9), 0xcd83d2); // unknown-element fallback
  assert.equal(meshColor(v.ligandBonds, 0), 0xd65358);
});

test('chain coloring applies to protein atoms and bonds, preserving ligand elements by default', t => {
  const v = viewer(t);
  v.setProteinColorMode('chain');
  assert.notEqual(chainColor(v, 0), chainColor(v, 2));
  assert.equal(meshColor(v.sideAtoms, 0), chainColor(v, 0));
  assert.equal(meshColor(v.sideAtoms, 1), chainColor(v, 2));
  assert.equal(meshColor(v.sideBonds, 1), chainColor(v, 2));
  assert.equal(ligandColor(v, 4), 0xc7cdcf);
  assert.equal(ligandColor(v, 5), 0xf06560);
});

test('ligands can follow their chain, including ligand-only chains and role-zero molecules', t => {
  const v = viewer(t);
  const proteinPalette = [...v.proteinChainIds].map(id => v.chainColors.get(id).getHex());
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  for (let atom = 4; atom < 10; atom++) assert.equal(ligandColor(v, atom), chainColor(v, atom));
  assert.equal(ligandColor(v, 4), meshColor(v.sideAtoms, 0));
  assert.notEqual(ligandColor(v, 4), ligandColor(v, 6));
  assert.equal(meshColor(v.ligandBonds, 1), chainColor(v, 6));
  assert.equal(meshColor(v.ligandBonds, 2), chainColor(v, 8));
  assert.deepEqual([...v.proteinChainIds].map(id => v.chainColors.get(id).getHex()), proteinPalette);
});

test('ligand highlighting composes with both color modes and survives coordinate updates', t => {
  const v = viewer(t);
  v.highlightLigandAtoms(new Set([5]));
  const tintedElement = new THREE.Color(0xf06560).multiply(v.selectionColor(true)).getHex();
  assert.equal(ligandColor(v, 5), tintedElement);
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  const tintedChain = v.chainColors.get('B').clone().multiply(v.selectionColor(true)).getHex();
  assert.equal(ligandColor(v, 5), tintedChain);
  v.update(new Float32Array(v.sample.target_coords.flat().map(n => n + 1)));
  assert.equal(ligandColor(v, 5), tintedChain);
  v.setLigandColorsByElement(true);
  assert.equal(ligandColor(v, 5), tintedElement);
  v.highlightLigandAtoms(new Set());
  assert.equal(ligandColor(v, 5), 0xf06560);
});

test('protein residue highlights and reference overlay survive color changes', t => {
  const v = viewer(t);
  v.highlightResidues(new Set([0]));
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  assert.equal(meshColor(v.sideAtoms, 0), 0xffc533);
  assert.equal(v.referenceAtomMesh.material.color.getHex(), 0xf0bd68);
  assert.equal(v.referenceBondMesh.material.color.getHex(), 0xdca85e);
  v.highlightResidues(new Set());
  assert.equal(meshColor(v.sideAtoms, 0), chainColor(v, 0));
});

test('color preferences survive sample replacement without stale ligand selection indices', t => {
  const v = viewer(t);
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  v.highlightLigandAtoms(new Set([5]));
  const next = viewerSample();
  v.setSample(next, new Float32Array(next.target_coords.flat()), true);
  assert.equal(v.proteinColorMode, 'chain');
  assert.equal(v.ligandColorsByElement, false);
  assert.equal(v.highlightedLigandAtoms.size, 0);
  assert.equal(ligandColor(v, 5), chainColor(v, 5));
  v.setProteinColorMode('role');
  assert.equal(ligandColor(v, 5), 0xf06560);
  v.setProteinColorMode('chain');
  assert.equal(ligandColor(v, 5), chainColor(v, 5));
});

test('numeric and missing chain identifiers have safe color fallbacks', t => {
  const sample = viewerSample();
  delete sample.atom_labels;
  const v = viewer(t, sample);
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  assert.equal(v.atomChainKeys[6], '0');
  assert.equal(ligandColor(v, 6), chainColor(v, 6));
  delete sample.chain_ids;
  v.setSample(sample, new Float32Array(sample.target_coords.flat()), true);
  assert.equal(v.atomChainKeys[6], '');
  assert.equal(ligandColor(v, 6), chainColor(v, 6));
});

test('replacement labels fall back to chain_ids instead of being parsed as atom locators', t => {
  const sample = viewerSample();
  sample.atom_labels[6] = 'replacement:group:0';
  sample.atom_labels[7] = 'replacement:group:1';
  const v = viewer(t, sample);
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  assert.equal(v.atomChainKeys[6], '0');
  assert.equal(ligandColor(v, 6), ligandColor(v, 7));
});

test('chain bond colors follow compacted instances when zero-length bonds are skipped', t => {
  const sample = viewerSample();
  sample.target_coords[5] = [...sample.target_coords[4]];
  const v = viewer(t, sample);
  v.setProteinColorMode('chain');
  v.setLigandColorsByElement(false);
  assert.equal(v.ligandBonds.count, 2);
  assert.equal(meshColor(v.ligandBonds, 0), chainColor(v, 6));
  assert.equal(meshColor(v.ligandBonds, 1), chainColor(v, 8));
  sample.target_coords[5][0] += 1;
  v.update(new Float32Array(sample.target_coords.flat()));
  assert.equal(v.ligandBonds.count, 3);
  assert.equal(meshColor(v.ligandBonds, 0), chainColor(v, 4));
});
