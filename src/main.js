import { DockingWebGpuModel } from "./model.js";
import { MolecularViewer } from "./viewer.js";
import { graphFromSmiles } from "./chemistry.js";
import { loadCcdGraphs } from "./ccd.js";
import { assetUrl, MODEL_MANIFEST_URL } from "./config.js";
import { GraphUnavailableError, parsePdb, preparePdbSample, replaceLigand } from "./prep.js";
import { loadRdkit } from "./rdkit.js";

const ui = Object.fromEntries([
  "device-dot", "device-label", "sample-select", "step-select", "seed-input",
  "run-button", "run-label", "reset-camera", "model-label", "atom-count",
  "step-status", "map-time", "total-time", "memory-label", "progress-bar", "status",
  "example-tab", "custom-tab", "example-panel", "custom-panel", "pdb-input",
  "open-pdb", "ligand-select", "smiles-input", "replace-ligand", "structure-label",
  "pdb-id-input", "fetch-pdb", "show-reference",
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

function setStatus(message) {
  ui.status.textContent = message;
}

function formatBytes(bytes) {
  return `${(bytes / 2 ** 20).toFixed(1)} MiB`;
}

async function applySample(nextSample, readyStatus = "Weights are resident on the GPU. Run a stochastic trajectory.") {
  sample = nextSample;
  await model.setSample(sample);
  const initial = model.initialize(Number(ui["seed-input"].value) || 1);
  viewer.setSample(sample, initial.coords);
  ui["atom-count"].textContent = sample.atoms.toLocaleString();
  ui["memory-label"].textContent = formatBytes(model.memoryBytes());
  ui["step-status"].textContent = "Ready";
  ui["map-time"].textContent = "-";
  ui["total-time"].textContent = "-";
  ui["progress-bar"].style.width = "0%";
  setStatus(readyStatus);
  return sample;
}

async function loadSample(file) {
  setStatus("Loading the selected molecular system.");
  presetSample = await fetch(assetUrl(`assets/samples/${file}`)).then((response) => response.json());
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
  for (const id of ["run-button", "sample-select", "open-pdb", "fetch-pdb", "replace-ligand", "example-tab", "custom-tab"]) {
    ui[id].disabled = busy;
  }
  ui["ligand-select"].disabled = busy || !pdbStructure?.ligandOptions.length;
  ui["smiles-input"].disabled = busy;
  ui["pdb-id-input"].disabled = busy;
}

function ligandOptions(structure) {
  ui["ligand-select"].replaceChildren();
  if (!structure.ligandOptions.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Receptor only";
    ui["ligand-select"].append(option);
    ui["ligand-select"].disabled = true;
    return;
  }
  if (structure.ligandOptions.length > 1) {
    const all = document.createElement("option");
    all.value = "__all__";
    all.textContent = `All non-water components (${structure.ligandOptions.reduce((total, item) => total + item.atoms.length, 0)})`;
    ui["ligand-select"].append(all);
  }
  for (const entry of structure.ligandOptions) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.label;
    ui["ligand-select"].append(option);
  }
  ui["ligand-select"].value = structure.defaultLigandId;
  ui["ligand-select"].disabled = false;
}

async function preparedPdbSelection(structure, ligandId) {
  const options = ligandId === "__all__"
    ? structure.ligandOptions
    : structure.ligandOptions.filter((option) => option.id === ligandId);
  const componentIds = options.map((option) => option.atoms[0].rawResidue);
  const componentGraphs = await loadCcdGraphs(componentIds);
  return preparePdbSample(structure, ligandId, componentGraphs);
}

async function applyPdbSelection(structure, ligandId) {
  try {
    customSample = await preparedPdbSelection(structure, ligandId);
    await applySample(customSample, `Prepared ${structure.filename}. ${customSample.graph_source}.`);
    ui["structure-label"].textContent = `${structure.filename} / ${customSample.atoms.toLocaleString()} atoms / ${customSample.graph_source}`;
    return customSample;
  } catch (error) {
    if (!(error instanceof GraphUnavailableError)) throw error;
    customSample = preparePdbSample(structure, null);
    await applySample(customSample, error.message);
    ui["structure-label"].textContent = `${structure.filename} / receptor only / replacement SMILES required`;
    return customSample;
  }
}

async function preparePdb(text, filename = "structure.pdb") {
  if (running) return;
  setInputBusy(true);
  setStatus("Preparing the PDB in this browser tab.");
  try {
    pdbStructure = parsePdb(text, filename);
    ligandOptions(pdbStructure);
    setSourceMode("custom");
    return await applyPdbSelection(pdbStructure, ui["ligand-select"].value || null);
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
    const response = await fetch(`https://files.rcsb.org/download/${encodeURIComponent(id)}.pdb1`);
    if (!response.ok) throw new Error(`PDB ${id} could not be loaded (${response.status}).`);
    ui["pdb-id-input"].value = id;
    return await preparePdb(await response.text(), `${id}-assembly1.pdb`);
  } finally {
    setInputBusy(false);
  }
}

async function applySmiles(smiles = ui["smiles-input"].value) {
  if (running || !sample) return;
  setInputBusy(true);
  setStatus("Resolving the replacement molecular graph with RDKit WASM.");
  try {
    const graph = graphFromSmiles(rdkit, smiles);
    customSample = replaceLigand(sample, graph);
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

async function runInference() {
  if (running) return;
  running = true;
  setInputBusy(true);
  const steps = Number(ui["step-select"].value);
  const seed = Number(ui["seed-input"].value) || 1;
  const { rng, coords: initial } = model.initialize(seed);
  let previous = initial;
  viewer.update(previous);
  const started = performance.now();
  try {
    for (let index = 0; index < steps; index += 1) {
      const start = index / steps;
      const end = (index + 1) / steps;
      ui["step-status"].textContent = `${index + 1} / ${steps}`;
      ui["progress-bar"].style.width = `${100 * index / steps}%`;
      setStatus(`Running step ${index + 1} of ${steps}.`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const noise = model.drawNoise(start, end, rng);
      const elapsed = await model.transition(start, end, noise.increment, noise.latent);
      const next = await model.coordinates();
      ui["map-time"].textContent = `${elapsed.toFixed(1)} ms`;
      setStatus(`Step ${index + 1} completed in ${elapsed.toFixed(1)} ms.`);
      await viewer.interpolate(previous, next, Math.min(420, Math.max(160, elapsed * 0.25)));
      previous = next;
      ui["progress-bar"].style.width = `${100 * (index + 1) / steps}%`;
    }
    const total = performance.now() - started;
    ui["total-time"].textContent = `${(total / 1000).toFixed(2)} s`;
    setStatus(`Inference complete. ${steps} steps evaluated in-browser.`);
  } catch (error) {
    console.error(error);
    setStatus(`Inference failed: ${error.message}`);
  } finally {
    running = false;
    setInputBusy(false);
  }
  return previous;
}

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
  if (adapter.limits.maxComputeWorkgroupStorageSize < 17968) {
    throw new Error("This WebGPU adapter has insufficient workgroup storage for tiled point attention.");
  }
  device = await adapter.requestDevice({
    requiredFeatures: needsF16 ? ["shader-f16"] : [],
    requiredLimits: {
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
ui["reset-camera"].addEventListener("click", () => viewer.resetCamera());
ui["show-reference"].addEventListener("change", () => {
  viewer.setReferenceVisible(ui["show-reference"].checked);
});
ui["sample-select"].addEventListener("change", () => loadSample(ui["sample-select"].value));
ui["example-tab"].addEventListener("click", async () => {
  setSourceMode("example");
  if (presetSample && sample !== presetSample) await applySample(presetSample);
});
ui["custom-tab"].addEventListener("click", async () => {
  setSourceMode("custom");
  if (customSample && sample !== customSample) await applySample(customSample);
});
ui["open-pdb"].addEventListener("click", () => ui["pdb-input"].click());
ui["fetch-pdb"].addEventListener("click", () => fetchPdb().catch((error) => setStatus(error.message)));
ui["pdb-id-input"].addEventListener("keydown", (event) => {
  if (event.key === "Enter") fetchPdb().catch((error) => setStatus(error.message));
});
ui["pdb-input"].addEventListener("change", () => preparePdbFile(ui["pdb-input"].files[0]).catch(() => {}));
ui["ligand-select"].addEventListener("change", async () => {
  if (!pdbStructure) return;
  setInputBusy(true);
  try {
    await applyPdbSelection(pdbStructure, ui["ligand-select"].value || null);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setInputBusy(false);
  }
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
  const file = [...event.dataTransfer.files].find((entry) => /\.(pdb|ent)$/i.test(entry.name));
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
  async loadSample(file) { return loadSample(file); },
  async loadPdbText(text, filename) { return preparePdb(text, filename); },
  async fetchPdb(pdbId) { return fetchPdb(pdbId); },
  async replaceLigand(smiles) { return applySmiles(smiles); },
  async runInference() { return runInference(); },
};
