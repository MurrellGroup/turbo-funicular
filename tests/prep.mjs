import assert from "node:assert/strict";
import test from "node:test";
import initRDKitModule from "../vendor/rdkit/RDKit_minimal.cjs";

import { parseCcdGraph, trainingGraphFromCcd } from "../src/ccd.js";
import { graphFromSmiles } from "../src/chemistry.js";
import { GraphUnavailableError, parsePdb, preparePdbSample, replaceLigand } from "../src/prep.js";
import { validateSample } from "../src/sample.js";
import { structureChoices, selectedStructure, previewStructure } from '../src/selection.js';
import { miniCcd, miniPdb } from "./fixtures.mjs";

const rdkit = await initRDKitModule();
const benGraph = trainingGraphFromCcd(rdkit, parseCcdGraph(miniCcd(), "BEN"));

test("PDB preparation matches the model input contract", () => {
  const structure = parsePdb(miniPdb(), "mini.pdb");
  assert.equal(structure.proteinAtoms.length, 9);
  assert.equal(structure.ligandOptions.length, 1);
  assert.equal(structure.ligandOptions[0].atoms.length, 6);
  const sample = preparePdbSample(structure, structure.defaultLigandId, new Map([["BEN", benGraph]]));
  validateSample(sample, 8192);
  assert.equal(sample.atoms, 15);
  assert.deepEqual([...new Set(sample.entity_ids.slice(0, 9))], [0]);
  assert.deepEqual([...new Set(sample.entity_ids.slice(9))], [1]);
  assert.equal(sample.ligand_bonds.length, 6);
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 26);
  assert.ok(sample.initial_scales.slice(9).every(v => v === 10));
  assert.ok(sample.base_scales.slice(9).every(v => v === 1));
  assert.equal(sample.neighbors.filter(([, type]) => type === 3).length, 12);
  assert.equal(sample.graph_source, "RCSB CCD graph");
  const proteinMean = sample.target_coords.slice(0, 9).reduce(
    (sum, point) => sum.map((value, axis) => value + point[axis] / 9),
    [0, 0, 0],
  );
  assert.ok(proteinMean.every((value) => Math.abs(value) < 1e-12));
});

test("PDB ligands without an authoritative graph are rejected", () => {
  const structure = parsePdb(miniPdb(), "mini.pdb");
  assert.throws(() => preparePdbSample(structure), GraphUnavailableError);
  const receptor = preparePdbSample(structure, null);
  assert.equal(receptor.atoms, 9);
  assert.equal(receptor.roles.filter((role) => role === 3).length, 0);
});

test("explicit LINK records add protein-ligand bonds without geometry inference", () => {
  const structure = parsePdb(miniPdb({ includeLink: true }), "linked.pdb");
  const sample = preparePdbSample(structure, structure.defaultLigandId, new Map([["BEN", benGraph]]));
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 28);
  assert.equal(sample.ligand_bonds.length, 7);
  assert.equal(sample.graph_source, "RCSB CCD graph plus explicit PDB links");
});

test("RDKit SMILES replacement emits an exact heavy-atom graph", () => {
  const graph = graphFromSmiles(rdkit, "CC(=O)Oc1ccccc1C(=O)O");
  assert.equal(graph.atomicNumbers.length, 13);
  assert.equal(graph.bonds.length, 13);
  assert.equal(graph.bonds.filter((bond) => bond.type === 3).length, 6);
  const structure = parsePdb(miniPdb(), "mini.pdb");
  const original = preparePdbSample(structure, structure.defaultLigandId, new Map([["BEN", benGraph]]));
  const sample = replaceLigand(original, graph);
  validateSample(sample, 8192);
  assert.equal(sample.atoms, 22);
  assert.equal(sample.roles.filter((role) => role === 3).length, 13);
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 40);
});

test("equivalent PDB/CCD and SMILES inputs produce identical conditioning tensors", () => {
  const structure = parsePdb(miniPdb(), "mini.pdb");
  const pdb = preparePdbSample(structure, structure.defaultLigandId, new Map([["BEN", benGraph]]));
  const smiles = replaceLigand(pdb, graphFromSmiles(rdkit, "c1ccccc1"));
  for (const name of [
    "base_means", "base_scales", "atomic_numbers", "roles", "residue_types",
    "atom_names", "entity_ids", "coordinate_design", "neighbors",
  ]) {
    assert.deepEqual(smiles[name], pdb[name], `${name} differs`);
  }
});

test("explicit SMILES hydrogens are removed", () => {
  const graph = graphFromSmiles(rdkit, "[H]O[H]");
  assert.deepEqual(graph.atomicNumbers, [8]);
  assert.equal(graph.bonds.length, 0);
});

test('multiple ligands survive an item-specific replacement', () => {
  const structure = parsePdb(miniPdb());
  const second = structuredClone(structure.ligandOptions[0]);
  second.id = 'second';
  for (const atom of second.atoms) {
    atom.chain = 'Z'; atom.residueKey = `Z|${atom.residueNumber}|`;
    atom.serial += 100; atom.coord[0] += 20;
  }
  structure.ligandOptions.push(second);
  const choices = structureChoices(structure);
  assert.equal(choices.ligands.length, 2);
  assert.equal(previewStructure(structure).atoms, 21);
  const full = preparePdbSample(structure, '__all__', new Map([['BEN', benGraph]]));
  const replaced = replaceLigand(full, graphFromSmiles(rdkit, 'CCO'), new Set([full.entity_ids.at(-1)]));
  validateSample(replaced);
  assert.equal(replaced.roles.filter(r => r === 3).length, 9);
  assert.equal(replaced.ligand_bonds.length, 8);
  for (let i = 0; i < 15; i++) {
    const edges = sample => sample.neighbors.slice(i * 10, i * 10 + 10).filter(([j]) => j >= 0).sort((a,b) => a[0]-b[0]);
    assert.deepEqual(edges(replaced), edges(full));
  }
});

test('selection refuses a dangling protein attachment', () => {
  const structure = parsePdb(miniPdb({ includeLink: true }));
  const choices = structureChoices(structure);
  assert.equal(choices.ligands[0].chains.size, 1);
  assert.throws(() => selectedStructure(structure, choices, new Set(), new Set([choices.ligands[0].id])), /attached/);
  assert.equal(previewStructure(selectedStructure(structure, choices, new Set(), new Set())).atoms, 0);
});
