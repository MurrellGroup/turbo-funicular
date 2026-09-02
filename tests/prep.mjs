import assert from "node:assert/strict";
import test from "node:test";
import initRDKitModule from "@rdkit/rdkit";

import { graphFromSmiles } from "../src/chemistry.js";
import { parsePdb, preparePdbSample, replaceLigand } from "../src/prep.js";
import { validateSample } from "../src/sample.js";
import { miniPdb } from "./fixtures.mjs";

const rdkit = await initRDKitModule();

test("PDB preparation matches the model input contract", () => {
  const structure = parsePdb(miniPdb(), "mini.pdb");
  assert.equal(structure.proteinAtoms.length, 9);
  assert.equal(structure.ligandOptions.length, 1);
  assert.equal(structure.ligandOptions[0].atoms.length, 6);
  const sample = preparePdbSample(structure);
  validateSample(sample, 8192);
  assert.equal(sample.atoms, 15);
  assert.deepEqual([...new Set(sample.entity_ids.slice(0, 9))], [0]);
  assert.deepEqual([...new Set(sample.entity_ids.slice(9))], [1]);
  assert.equal(sample.ligand_bonds.length, 6);
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 12);
  assert.equal(sample.neighbors.filter(([, type]) => type === 1).length, 6);
  const proteinMean = sample.target_coords.slice(0, 9).reduce(
    (sum, point) => sum.map((value, axis) => value + point[axis] / 9),
    [0, 0, 0],
  );
  assert.ok(proteinMean.every((value) => Math.abs(value) < 1e-12));
});

test("RDKit SMILES replacement emits an exact heavy-atom graph", () => {
  const graph = graphFromSmiles(rdkit, "CC(=O)Oc1ccccc1C(=O)O");
  assert.equal(graph.atomicNumbers.length, 13);
  assert.equal(graph.bonds.length, 13);
  assert.equal(graph.bonds.filter((bond) => bond.type === 3).length, 6);
  const original = preparePdbSample(parsePdb(miniPdb(), "mini.pdb"));
  const sample = replaceLigand(original, graph);
  validateSample(sample, 8192);
  assert.equal(sample.atoms, 22);
  assert.equal(sample.roles.filter((role) => role === 3).length, 13);
  assert.equal(sample.neighbors.filter(([atom]) => atom >= 0).length, 26);
});

test("explicit SMILES hydrogens are removed", () => {
  const graph = graphFromSmiles(rdkit, "[H]O[H]");
  assert.deepEqual(graph.atomicNumbers, [8]);
  assert.equal(graph.bonds.length, 0);
});
