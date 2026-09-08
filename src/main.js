import { DockingWebGpuModel } from "./model.js";
import { MolecularViewer } from "./viewer.js";
import { graphFromSmiles } from "./chemistry.js";
import { loadCcdGraphs } from "./ccd.js";
import { assetUrl, MODEL_MANIFEST_URL } from "./config.js";
import { parsePdb, preparePdbSample, replaceLigand, selectedLigandOptions, atomLocator } from "./prep.js";
import { parseMmcif } from './mmcif.js';
import { loadRdkit } from "./rdkit.js";
import { structureChoices, selectedStructure, previewStructure } from './selection.js';
import { SequenceEditor } from './sequence-editor.js';
import { mutateSample, proteinChains } from './sequence.js';
import { createIcons, ChevronLeft, ChevronRight, Download, Upload, X } from 'lucide';

const ui = Object.fromEntries([
  "device-dot", "device-label", "sample-select", "step-select", "seed-input",
  "run-button", "run-label", "reset-camera", "model-label", "atom-count",
  "step-status", "map-time", "total-time", "memory-label", "progress-bar", "status",
  "example-tab", "custom-tab", "example-panel", "custom-panel", "pdb-input",
  "open-pdb", "chain-list", "ligand-list", "selection-count", "prepare-selection", "smiles-input", "replace-ligand", "structure-label",
  "pdb-id-input", "fetch-pdb", "show-reference",
  "active-ligand", "remove-ligand",
  "campaign-count", "run-campaign", "stop-campaign", "campaign-results", "campaign-result",
  "previous-result", "next-result", "download-result", "result-identities",
].map((id) => [id, document.getElementById(id)]));

const viewer = new MolecularViewer(document.getElementById("viewport"));
let device;
let model;
let sample;
let catalog;
let rdkit;
let pdbStructure;
let presetSample;
let customSample;
let running = false;
let inputBusy = false;
let modelReady = false;
let choices;
let keptChains = new Set(), keptLigands = new Set();
let replacements = new Map(), previewGeneration = 0;
let previewGraphs = new Map();
let campaign = [], cancelCampaign = false;
let preparedBase = null;
const editor = new SequenceEditor({
  onChange: () => {
    if (preparedBase) {
      const preview = mutateSample(preparedBase, editor.edits, Number(ui['seed-input'].value) || 1).sample;
      viewer.setSample(preview, new Float32Array(preview.target_coords.flat()), true);
      viewer.highlightResidues(editor.selected); activeHighlight();
      ui['atom-count'].textContent = preview.atoms.toLocaleString();
    }
    setStatus(`${editor.edits.size} residue edits`);
  },
  onSelect: ids => viewer.highlightResidues(ids),
  onError: message => { setStatus(message); document.getElementById('alignment-status').textContent = message; },
});
createIcons({ icons: { ChevronLeft, ChevronRight, Download, Upload, X } });
const ligandLabel = g => `${g.options[0].atoms[0].rawResidue}${g.options.length > 1 ? ` +${g.options.length - 1}` : ''} / ${g.options[0].atoms[0].chain} (${g.atoms})`;

function clearCampaign() {
  campaign = []; ui['campaign-result'].replaceChildren(); ui['campaign-results'].hidden = true;
}

function activeHighlight() {
  const group = choices?.ligands.find(g => g.id === ui['active-ligand'].value);
  const labels = new Set(group?.options.flatMap(o => o.atoms.map(atomLocator)) ?? []);
  const active = new Set((viewer.sample?.atom_labels ?? []).flatMap((label, i) =>
    labels.has(label) || (group && label.startsWith(`replacement:${group.id}:`)) ? [i] : []));
  for (const item of viewer.ligandMeshes ?? []) {
    item.atoms.forEach((atom, i) => item.mesh.setColorAt(i, viewer.selectionColor(active.has(atom))));
    if (item.mesh.instanceColor) item.mesh.instanceColor.needsUpdate = true;
  }
}

viewer.onAtomPick = atom => {
  if (running || inputBusy || !choices) return;
  const label = viewer.sample?.atom_labels?.[atom];
  if ([1, 2].includes(viewer.sample?.roles[atom]) && !ui['custom-panel'].hidden && preparedBase) {
    editor.pick(atom, viewer.sample); return;
  }
  const group = choices.ligands.find(g => label?.startsWith(`replacement:${g.id}:`)
    || g.options.some(o => o.atoms.some(a => atomLocator(a) === label)));
  if (group && keptLigands.has(group.id)) { ui['active-ligand'].value = group.id; activeHighlight(); }
};

function setStatus(message) {
  ui.status.textContent = message;
}

function formatBytes(bytes) {
  return `${(bytes / 2 ** 20).toFixed(1)} MiB`;
}

async function applySample(nextSample, readyStatus = "Ready", preserveCamera = false) {
  modelReady = false;
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let allocationError;
  try { await model.setSample(nextSample); } catch (error) { allocationError = error; }
  const validation = await device.popErrorScope();
  const memory = await device.popErrorScope();
  if (allocationError || validation || memory) throw allocationError ?? new Error((validation ?? memory).message);
  sample = nextSample;
  modelReady = true;
  const initial = model.initialize(Number(ui["seed-input"].value) || 1);
  viewer.setSample(sample, initial.coords, preserveCamera);
  activeHighlight();
  ui["atom-count"].textContent = sample.atoms.toLocaleString();
  ui["memory-label"].textContent = formatBytes(model.memoryBytes());
  ui["step-status"].textContent = "Ready";
  ui["map-time"].textContent = "-";
  ui["total-time"].textContent = "-";
  ui["progress-bar"].style.width = "0%";
  setStatus(readyStatus);
  ui['run-button'].disabled = false;
  ui['run-campaign'].disabled = false;
  return sample;
}

async function loadSample(file) {
  clearCampaign();
  setStatus("Loading the selected molecular system.");
  presetSample = await fetch(assetUrl(`assets/samples/${file}`)).then((response) => response.json());
  editor.setSample(null);
  return applySample(presetSample);
}

function setSourceMode(mode) {
  const example = mode === "example";
  ui["example-tab"].classList.toggle("active", example);
  ui["custom-tab"].classList.toggle("active", !example);
  ui["example-tab"].setAttribute("aria-selected", String(example));
  ui["custom-tab"].setAttribute("aria-selected", String(!example));
  ui["example-panel"].hidden = !example;
  ui["custom-panel"].hidden = example;
}

function setInputBusy(busy) {
  inputBusy = busy;
  for (const id of ["run-button", "sample-select", "open-pdb", "fetch-pdb", "replace-ligand", "example-tab", "custom-tab"]) {
    ui[id].disabled = busy;
  }
  ui['run-button'].disabled = busy || !modelReady;
  ui['run-campaign'].disabled = busy || !modelReady;
  for (const id of ['step-select', 'seed-input', 'campaign-count', 'campaign-result', 'previous-result', 'next-result', 'download-result']) ui[id].disabled = busy;
  editor.setBusy(busy || !modelReady || ui['custom-panel'].hidden);
  ui['prepare-selection'].disabled = busy || !pdbStructure;
  ui['active-ligand'].disabled = busy || !keptLigands.size;
  ui['remove-ligand'].disabled = busy || !keptLigands.size;
  document.querySelectorAll('.structure-choices input').forEach(input => { input.disabled = busy; });
  document.querySelectorAll('.choice-actions button').forEach(input => { input.disabled = busy || !choices; });
  ui["smiles-input"].disabled = busy;
  ui["pdb-id-input"].disabled = busy;
}

function selectionPreview() {
  clearCampaign();
  const generation = ++previewGeneration;
  modelReady = false;
  customSample = null;
  preparedBase = null;
  editor.setSample(null);
  ui['run-button'].disabled = true;
  ui['run-campaign'].disabled = true;
  const subset = selectedStructure(pdbStructure, choices, keptChains, keptLigands);
  const preview = previewStructure(subset, previewGraphs);
  if (preview.atoms) viewer.setSample(preview, new Float32Array(preview.target_coords.flat()));
  else viewer.clear();
  ui['atom-count'].textContent = preview.atoms.toLocaleString();
  ui['selection-count'].textContent = `${preview.atoms.toLocaleString()} selected / ${model.maximumAtoms.toLocaleString()} inference limit`;
  ui['memory-label'].textContent = '-';
  ui['step-status'].textContent = 'Preview';
  ui['structure-label'].textContent = pdbStructure.filename;
  activeHighlight();
  if (replacements.size) preparedPdbSelection(subset, keptLigands.size ? '__all__' : null).then(prepared => {
    if (generation !== previewGeneration || modelReady || inputBusy) return;
    viewer.setSample(prepared, new Float32Array(prepared.target_coords.flat()));
    ui['selection-count'].textContent = `${prepared.atoms.toLocaleString()} selected / ${model.maximumAtoms.toLocaleString()} inference limit`;
    activeHighlight();
  }).catch(error => { if (generation === previewGeneration) setStatus(error.message); });
  return preview;
}

function choiceControls() {
  for (const [kind, entries, selected] of [['chain', choices.chains, keptChains], ['ligand', choices.ligands, keptLigands]]) {
    const container = ui[`${kind}-list`];
    container.replaceChildren();
    for (const entry of entries) {
      const row = document.createElement('label');
      row.className = 'choice-row';
      const input = document.createElement('input');
      input.type = 'checkbox'; input.checked = selected.has(entry.id); input.value = entry.id;
      const text = document.createElement('span');
      text.textContent = kind === 'chain' ? `${entry.id || '-'} (${entry.atoms})`
        : ligandLabel(entry);
      if (kind === 'ligand' && entry.chains.size) row.title = `Attached to ${[...entry.chains].join(', ')}`;
      input.addEventListener('change', () => {
        if (input.checked) selected.add(entry.id); else selected.delete(entry.id);
        if (kind === 'ligand' && input.checked) for (const chain of entry.chains) keptChains.add(chain);
        if (kind === 'chain' && !input.checked) for (const ligand of choices.ligands) {
          if (ligand.chains.has(entry.id)) keptLigands.delete(ligand.id);
        }
        choiceControls(); selectionPreview();
      });
      row.append(input, text); container.append(row);
    }
  }
  const previous = ui['active-ligand'].value;
  ui['active-ligand'].replaceChildren();
  for (const group of choices.ligands.filter(g => keptLigands.has(g.id))) {
    const option = document.createElement('option'); option.value = group.id;
    option.textContent = replacements.has(group.id) ? replacements.get(group.id).canonicalSmiles
      : ligandLabel(group);
    ui['active-ligand'].append(option);
  }
  if (keptLigands.has(previous)) ui['active-ligand'].value = previous;
  ui['active-ligand'].disabled = !keptLigands.size;
  ui['remove-ligand'].disabled = !keptLigands.size;
}

async function useSelection() {
  setInputBusy(true);
  try {
    const subset = selectedStructure(pdbStructure, choices, keptChains, keptLigands);
    const prepared = await preparedPdbSelection(subset, keptLigands.size ? '__all__' : null);
    await applySample(prepared, 'Ready');
    customSample = prepared;
    preparedBase = prepared;
    editor.setSample(prepared);
    return prepared;
  } catch (error) { modelReady = false; setStatus(error.message); throw error; }
  finally { setInputBusy(false); }
}

async function preparedPdbSelection(structure, ligandId) {
  const options = selectedLigandOptions(structure, ligandId);
  const present = new Set(options.map(o => o.id));
  const substituted = (choices?.ligands ?? []).filter(g => replacements.has(g.id) && g.options.some(o => present.has(o.id)));
  const excluded = new Set(substituted.flatMap(g => g.options.map(o => o.id)));
  const remaining = options.filter(o => !excluded.has(o.id));
  const base = { ...structure, ligandOptions: remaining };
  const componentIds = remaining.map((option) => option.atoms[0].rawResidue);
  const componentGraphs = await loadCcdGraphs(componentIds, rdkit);
  let prepared = base.proteinAtoms.length || remaining.length
    ? preparePdbSample(base, remaining.length ? '__all__' : null, componentGraphs, rdkit) : previewStructure(base);
  for (const group of substituted) {
    const graph = replacements.get(group.id);
    prepared = replaceLigand(prepared, graph, new Set());
    const offset = prepared.atoms - graph.atomicNumbers.length;
    for (let i = 0; i < graph.atomicNumbers.length; i++) prepared.atom_labels[offset + i] = `replacement:${group.id}:${i}`;
  }
  return prepared;
}

async function preparePdb(text, filename = "structure.pdb", depositedText = null) {
  if (running) return;
  setInputBusy(true);
  setStatus("Preparing the PDB in this browser tab.");
  try {
    pdbStructure = /^\s*data_/m.test(text) ? parseMmcif(text, filename, depositedText) : parsePdb(text, filename);
    choices = structureChoices(pdbStructure);
    replacements = new Map();
    previewGraphs = new Map();
    keptChains = new Set(choices.chains.map(c => c.id));
    keptLigands = new Set(choices.ligands.map(c => c.id));
    choiceControls();
    setSourceMode("custom");
    setStatus('Structure loaded');
    const original = pdbStructure;
    const preview = selectionPreview();
    loadCcdGraphs(original.ligandOptions.map(o => o.atoms[0].rawResidue), rdkit).then(graphs => {
      if (pdbStructure !== original) return;
      previewGraphs = graphs;
      if (!modelReady && !inputBusy) selectionPreview();
    }).catch(() => {});
    return preview;
  } catch (error) {
    setStatus(error.message);
    throw error;
  } finally {
    setInputBusy(false);
  }
}

async function preparePdbFile(file) {
  if (!file) return;
  await preparePdb(await file.text(), file.name);
}

async function fetchPdb(pdbId = ui["pdb-id-input"].value) {
  const id = pdbId.trim().toUpperCase();
  if (!/^[A-Z0-9]{4}$/.test(id)) throw new Error("Enter a four-character PDB ID.");
  setInputBusy(true);
  setStatus(`Loading ${id}.`);
  try {
    const response = await fetch(`https://files.rcsb.org/download/${encodeURIComponent(id)}-assembly1.cif`);
    if (!response.ok) throw new Error(`PDB ${id} could not be loaded (${response.status}).`);
    ui["pdb-id-input"].value = id;
    const deposited = await fetch(`https://files.rcsb.org/download/${encodeURIComponent(id)}.cif`);
    if (!deposited.ok) throw new Error(`PDB ${id} connectivity could not be loaded (${deposited.status}).`);
    return await preparePdb(await response.text(), `${id}-assembly1.cif`, await deposited.text());
  } finally {
    setInputBusy(false);
  }
}

async function applySmiles(smiles = ui["smiles-input"].value) {
  if (running || (!sample && !pdbStructure)) return;
  setInputBusy(true);
  setStatus("Resolving the replacement molecular graph with RDKit WASM.");
  try {
    const graph = graphFromSmiles(rdkit, smiles);
    let receptor = sample;
    if (pdbStructure && !ui['custom-panel'].hidden) {
      const group = choices.ligands.find(g => g.id === ui['active-ligand'].value);
      if (!group) throw new Error('Select a ligand to replace.');
      if (group.chains.size) throw new Error('Replacing an attached ligand requires an explicit attachment atom.');
      replacements.set(group.id, graph);
      choiceControls(); selectionPreview();
      const prepared = await preparedPdbSelection(selectedStructure(pdbStructure, choices, keptChains, keptLigands), '__all__');
      viewer.setSample(prepared, new Float32Array(prepared.target_coords.flat()));
      ui['selection-count'].textContent = `${prepared.atoms.toLocaleString()} selected / ${model.maximumAtoms.toLocaleString()} inference limit`;
      activeHighlight();
      return prepared;
    }
    customSample = replaceLigand(receptor, graph);
    setSourceMode("custom");
    await applySample(
      customSample,
      `Prepared ${graph.canonicalSmiles} with ${graph.atomicNumbers.length} heavy atoms and ${graph.bonds.length} bonds.`,
    );
    ui["structure-label"].textContent = `${graph.canonicalSmiles} / exact RDKit heavy-atom graph`;
    return customSample;
  } catch (error) {
    setStatus(error.message);
    throw error;
  } finally {
    setInputBusy(false);
  }
}

function selectResult(index) {
  const result = campaign[index];
  if (!result || running) return;
  ui['campaign-result'].value = String(index);
  viewer.setSample(result.sample, result.coords, true);
  ui['atom-count'].textContent = result.sample.atoms.toLocaleString();
  ui['result-identities'].textContent = `Seed ${result.seed}${result.identities.length ? ' / ' + result.identities.join(', ') : ''}`;
  ui['previous-result'].disabled = index === 0;
  ui['next-result'].disabled = index === campaign.length - 1;
  activeHighlight();
}

async function runSamples(count) {
  if (running || !modelReady) return;
  const base = !ui['custom-panel'].hidden && preparedBase ? preparedBase : sample;
  const edits = base === preparedBase ? new Map(editor.edits) : new Map();
  const seed = Number(ui['seed-input'].value);
  const steps = Number(ui['step-select'].value);
  if (!Number.isInteger(count) || count < 1 || count > 200) { setStatus('Samples must be an integer from 1 to 200.'); return; }
  if (!Number.isSafeInteger(seed) || seed < 1 || seed + count > 4294967296) { setStatus('Seed must be a positive 32-bit integer.'); return; }
  // Validate every randomized topology before launching, without retaining N full graphs.
  try {
    let bytes = 0;
    for (let index = 0; index < count; index++) {
      const next = mutateSample(base, edits, seed + index).sample;
      if (next.atoms > model.maximumAtoms) throw new Error(`Sample ${index + 1} exceeds the ${model.maximumAtoms}-atom adapter limit.`);
      bytes += next.atoms * (edits.size ? 1400 : 12);
      if (bytes > 512 * 2 ** 20) throw new Error('Campaign exceeds the 512 MiB result budget. Reduce the sample count.');
    }
  } catch (error) { setStatus(error.message); return; }
  running = true;
  cancelCampaign = false;
  editor.cancelMapping();
  setInputBusy(true);
  ui['stop-campaign'].hidden = false;
  campaign = []; ui['campaign-result'].replaceChildren(); ui['campaign-results'].hidden = true;
  let previous;
  const started = performance.now();
  try {
    for (let member = 0; member < count && !cancelCampaign; member++) {
      const { sample: nextSample, resolved } = mutateSample(base, edits, seed + member);
      if (sample !== nextSample) await applySample(nextSample, 'Ready', true);
      // applySample enables commands for interactive preparation; keep them locked during a campaign.
      setInputBusy(true);
      const { rng, coords: initial } = model.initialize(seed + member);
      viewer.setSample(nextSample, initial, true);
      previous = initial;
      for (let index = 0; index < steps; index += 1) {
        if (cancelCampaign) break;
        const start = index / steps;
        const end = (index + 1) / steps;
        ui["step-status"].textContent = `${index + 1} / ${steps}`;
        ui["progress-bar"].style.width = `${100 * (member + index / steps) / count}%`;
        setStatus(`Sample ${member + 1} / ${count}, step ${index + 1} / ${steps}`);
        await new Promise((resolve) => requestAnimationFrame(resolve));
        if (cancelCampaign) break;
        const noise = model.drawNoise(start, end, rng);
        const mapStarted = performance.now();
        await model.transition(start, end, noise.increment, noise.latent, false);
        const next = await model.coordinates();
        const elapsed = performance.now() - mapStarted;
        ui["map-time"].textContent = `${elapsed.toFixed(1)} ms`;
        viewer.update(next);
        previous = next;
        ui["progress-bar"].style.width = `${100 * (member + (index + 1) / steps) / count}%`;
      }
      if (cancelCampaign) break;
      const identities = proteinChains(base).flatMap(c => c.residues.filter(r => resolved.has(r.id))
        .map(r => `${c.id}:${r.aa}${r.number}${resolved.get(r.id)}`));
      campaign.push({ sample: nextSample, coords: previous.slice(), seed: seed + member, steps, identities,
        resolved: Object.fromEntries(resolved), checkpoint: model.weights.manifest.checkpoint_sha256 });
      ui['campaign-result'].add(new Option(`${member + 1} / seed ${seed + member}`, String(member)));
      ui['campaign-results'].hidden = false;
    }
    const total = performance.now() - started;
    ui["total-time"].textContent = `${(total / 1000).toFixed(2)} s`;
    setStatus(cancelCampaign ? `Stopped. ${campaign.length} samples completed.` : `Inference complete. ${campaign.length} samples / ${steps} steps.`);
  } catch (error) {
    console.error(error);
    setStatus(`Inference failed: ${error.message}`);
  } finally {
    running = false;
    ui['stop-campaign'].hidden = true;
    setInputBusy(false);
    if (campaign.length) selectResult(campaign.length - 1);
  }
  return previous;
}

async function runInference() { return runSamples(1); }

async function initialize() {
  if (!navigator.gpu) throw new Error("This browser does not expose WebGPU.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter is available.");
  const manifestResponse = await fetch(MODEL_MANIFEST_URL);
  if (!manifestResponse.ok) throw new Error(`Model manifest failed: ${manifestResponse.status}`);
  const manifest = await manifestResponse.json();
  const needsF16 = manifest.activation_precision === "float16";
  if (needsF16 && !adapter.features.has("shader-f16")) {
    throw new Error("This model payload requires WebGPU shader-f16, which the adapter does not expose.");
  }
  device = await adapter.requestDevice({
    requiredFeatures: needsF16 ? ["shader-f16"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: Math.min(
        32768,
        adapter.limits.maxComputeWorkgroupStorageSize,
      ),
    },
  });
  device.lost.then((info) => {
    ui["device-dot"].className = "error";
    ui["device-label"].textContent = "GPU device lost";
    setStatus(`WebGPU device lost: ${info.message}`);
  });
  ui["device-label"].textContent = adapter.info?.description
    || `WebGPU / ${needsF16 ? "FP16" : "FP32"}`;
  setStatus("Preparing the model.");
  const [loadedCatalog, loadedModel, loadedRdkit] = await Promise.all([
    fetch(assetUrl("assets/samples/catalog.json")).then((response) => response.json()),
    DockingWebGpuModel.create(device, MODEL_MANIFEST_URL, manifest),
    loadRdkit(),
  ]);
  catalog = loadedCatalog;
  model = loadedModel;
  rdkit = loadedRdkit;
  const checkpoint = model.weights.manifest.checkpoint_sha256.slice(0, 8);
  ui["model-label"].textContent = `${model.weights.manifest.iteration.toLocaleString()} / ${checkpoint}`;
  for (const entry of catalog.samples) {
    const option = document.createElement("option");
    option.value = entry.file;
    option.textContent = `${entry.label} (${entry.atoms})`;
    ui["sample-select"].append(option);
  }
  await loadSample(catalog.samples[0].file);
  ui["device-dot"].className = "ready";
  ui["run-button"].disabled = false;
  ui["sample-select"].disabled = false;
  ui["reset-camera"].disabled = false;
  ui["open-pdb"].disabled = false;
  ui["fetch-pdb"].disabled = false;
  ui["replace-ligand"].disabled = false;
  return { device, model, catalog };
}

ui["run-button"].addEventListener("click", runInference);
ui['run-campaign'].addEventListener('click', () => runSamples(Number(ui['campaign-count'].value)));
ui['stop-campaign'].addEventListener('click', () => { cancelCampaign = true; });
ui['campaign-result'].addEventListener('change', () => selectResult(Number(ui['campaign-result'].value)));
ui['previous-result'].addEventListener('click', () => selectResult(Number(ui['campaign-result'].value) - 1));
ui['next-result'].addEventListener('click', () => selectResult(Number(ui['campaign-result'].value) + 1));
ui['download-result'].addEventListener('click', () => {
  const result = campaign[Number(ui['campaign-result'].value)]; if (!result) return;
  const { reference_sample: _reference, ...sampleData } = result.sample;
  const blob = new Blob([JSON.stringify({ ...result, sample: sampleData, coords: [...result.coords] })], { type: 'application/json' });
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = `sample-${result.seed}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
ui["reset-camera"].addEventListener("click", () => viewer.resetCamera());
ui["show-reference"].addEventListener("change", () => {
  viewer.setReferenceVisible(ui["show-reference"].checked);
});
ui["sample-select"].addEventListener("change", () => loadSample(ui["sample-select"].value));
ui["example-tab"].addEventListener("click", async () => {
  clearCampaign();
  setSourceMode("example");
  editor.setBusy(true); document.getElementById('sequence-panel').hidden = true;
  if (presetSample) await applySample(presetSample);
});
ui["custom-tab"].addEventListener("click", async () => {
  clearCampaign();
  setSourceMode("custom");
  if (customSample && sample !== customSample) await applySample(customSample);
  else if (pdbStructure && !customSample) selectionPreview();
  if (preparedBase) editor.setSample(preparedBase);
  editor.setBusy(!modelReady);
});
ui["open-pdb"].addEventListener("click", () => ui["pdb-input"].click());
ui["fetch-pdb"].addEventListener("click", () => fetchPdb().catch((error) => setStatus(error.message)));
ui["pdb-id-input"].addEventListener("keydown", (event) => {
  if (event.key === "Enter") fetchPdb().catch((error) => setStatus(error.message));
});
ui["pdb-input"].addEventListener("change", () => preparePdbFile(ui["pdb-input"].files[0]).catch(() => {}));
ui['prepare-selection'].addEventListener('click', () => useSelection().catch(() => {}));
for (const kind of ['chains', 'ligands']) for (const mode of ['all', 'none']) {
  document.getElementById(`${kind}-${mode}`).addEventListener('click', () => {
    if (!choices || running) return;
    if (kind === 'chains') {
      keptChains = new Set(mode === 'all' ? choices.chains.map(c => c.id) : []);
      if (mode === 'none') for (const g of choices.ligands) if (g.chains.size) keptLigands.delete(g.id);
    } else {
      keptLigands = new Set(mode === 'all' ? choices.ligands.map(g => g.id) : []);
      if (mode === 'all') for (const g of choices.ligands) for (const c of g.chains) keptChains.add(c);
    }
    choiceControls(); selectionPreview();
  });
}
ui['active-ligand'].addEventListener('change', activeHighlight);
ui['remove-ligand'].addEventListener('click', () => {
  keptLigands.delete(ui['active-ligand'].value); choiceControls(); selectionPreview();
});
ui["replace-ligand"].addEventListener("click", () => applySmiles().catch(() => {}));
ui["smiles-input"].addEventListener("keydown", (event) => {
  if (event.key === "Enter") applySmiles().catch(() => {});
});

for (const type of ["dragenter", "dragover"]) {
  document.addEventListener(type, (event) => {
    event.preventDefault();
    document.querySelector("main").classList.add("dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  document.addEventListener(type, (event) => {
    event.preventDefault();
    document.querySelector("main").classList.remove("dragging");
  });
}
document.addEventListener("drop", (event) => {
  const file = [...event.dataTransfer.files].find((entry) => /\.(pdb|ent|cif|mmcif)$/i.test(entry.name));
  if (file) preparePdbFile(file).catch(() => {});
});

const ready = initialize().catch((error) => {
  console.error(error);
  ui["device-dot"].className = "error";
  ui["device-label"].textContent = "WebGPU unavailable";
  setStatus(error.message);
  throw error;
});

window.__wsfmdock = {
  ready,
  get model() { return model; },
  get viewer() { return viewer; },
  get sample() { return sample; },
  get editor() { return editor; },
  get campaign() { return campaign; },
  async loadSample(file) { return loadSample(file); },
  async loadPdbText(text, filename) { return preparePdb(text, filename); },
  async fetchPdb(pdbId) { return fetchPdb(pdbId); },
  async useSelection() { return useSelection(); },
  async replaceLigand(smiles) { return applySmiles(smiles); },
  async runInference() { return runInference(); },
  async runCampaign(count) { return runSamples(count); },
};
