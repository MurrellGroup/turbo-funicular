import assert from "node:assert/strict";
import test from "node:test";
import initRDKitModule from "../vendor/rdkit/RDKit_minimal.cjs";

import { parseCcdGraph, trainingGraphFromCcd } from "../src/ccd.js";
import { graphFromSmiles } from "../src/chemistry.js";
import { GraphUnavailableError, parsePdb, preparePdbSample, replaceLigand } from "../src/prep.js";
import { validateSample } from "../src/sample.js";
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
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 12);
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
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 14);
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
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 26);
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
