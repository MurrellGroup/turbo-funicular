import { DockingWebGpuModel } from "./model.js";
import { MolecularViewer } from "./viewer.js";

const ui = Object.fromEntries([
  "device-dot", "device-label", "sample-select", "step-select", "seed-input",
  "run-button", "run-label", "reset-camera", "model-label", "atom-count",
  "step-status", "map-time", "total-time", "memory-label", "progress-bar", "status",
].map((id) => [id, document.getElementById(id)]));

const viewer = new MolecularViewer(document.getElementById("viewport"));
let device;
let model;
let sample;
let running = false;

function setStatus(message) {
  ui.status.textContent = message;
}

function formatBytes(bytes) {
  return `${(bytes / 2 ** 20).toFixed(1)} MiB`;
}

async function loadSample(file) {
  setStatus("Loading the selected molecular system.");
  sample = await fetch(`/assets/samples/${file}`).then((response) => response.json());
  await model.setSample(sample);
  const initial = model.initialize(Number(ui["seed-input"].value) || 1);
  viewer.setSample(sample, initial.coords);
  ui["atom-count"].textContent = sample.atoms.toLocaleString();
  ui["memory-label"].textContent = formatBytes(model.memoryBytes());
  ui["step-status"].textContent = "Ready";
  ui["map-time"].textContent = "-";
  ui["total-time"].textContent = "-";
  ui["progress-bar"].style.width = "0%";
  setStatus("Weights are resident on the GPU. Run a stochastic trajectory.");
}

async function runInference() {
  if (running) return;
  running = true;
  ui["run-button"].disabled = true;
  ui["sample-select"].disabled = true;
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
      setStatus(`Evaluating finite map ${index + 1} of ${steps} on WebGPU.`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const noise = model.drawNoise(start, end, rng);
      const elapsed = await model.transition(start, end, noise.increment, noise.latent);
      const next = await model.coordinates();
      ui["map-time"].textContent = `${elapsed.toFixed(1)} ms`;
      setStatus(`Map ${index + 1} completed in ${elapsed.toFixed(1)} ms.`);
      await viewer.interpolate(previous, next, Math.min(420, Math.max(160, elapsed * 0.25)));
      previous = next;
      ui["progress-bar"].style.width = `${100 * (index + 1) / steps}%`;
    }
    const total = performance.now() - started;
    ui["total-time"].textContent = `${(total / 1000).toFixed(2)} s`;
    setStatus(`Trajectory complete. ${steps} stochastic maps evaluated entirely in-browser.`);
  } catch (error) {
    console.error(error);
    setStatus(`Inference failed: ${error.message}`);
  } finally {
    running = false;
    ui["run-button"].disabled = false;
    ui["sample-select"].disabled = false;
  }
}

async function initialize() {
  if (!navigator.gpu) throw new Error("This browser does not expose WebGPU.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter is available.");
  const manifest = await fetch("/assets/model/manifest.json").then((response) => response.json());
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
  setStatus("Compiling fused kernels and uploading the legacy CK checkpoint.");
  const [catalog, loadedModel] = await Promise.all([
    fetch("/assets/samples/catalog.json").then((response) => response.json()),
    DockingWebGpuModel.create(device),
  ]);
  model = loadedModel;
  const checkpoint = model.weights.manifest.checkpoint_sha256.slice(0, 8);
  ui["model-label"].textContent = `post-joint CK ${model.weights.manifest.iteration.toLocaleString()} / ${checkpoint}`;
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
  return { device, model, catalog };
}

ui["run-button"].addEventListener("click", runInference);
ui["reset-camera"].addEventListener("click", () => viewer.resetCamera());
ui["sample-select"].addEventListener("change", () => loadSample(ui["sample-select"].value));

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
  get sample() { return sample; },
  async loadSample(file) { return loadSample(file); },
};
