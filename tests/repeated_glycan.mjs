import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: false,
  executablePath: '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'] });
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('https://127.0.0.1:8791');
  await page.evaluate(() => window.__wsfmdock.ready);
  const source = execFileSync('git', ['show', 'd66e3a4:src/gpu.js'], { encoding: 'utf8' });
  const result = await page.evaluate(async ({ source, pdb, seedA, seedB, steps }) => {
    const api = window.__wsfmdock;
    await api.fetchPdb(pdb);
    const sample = await api.useSelection();
    const model = api.model;
    document.getElementById('step-select').value = String(steps);
    const graph = JSON.stringify(sample.neighbors);
    const attachments = sample.ligand_bonds.filter(([a, b]) => (sample.roles[a] === 3) !== (sample.roles[b] === 3));
    async function gpuGraph() {
      const staging = model.device.createBuffer({ size: sample.atoms * 20 * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = model.device.createCommandEncoder();
      encoder.copyBufferToBuffer(model.sampleBuffers.neighbors, 0, staging, 0, staging.size);
      model.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const values = new Int32Array(staging.getMappedRange().slice(0));
      staging.unmap(); staging.destroy();
      return JSON.stringify([...values]);
    }
    const originalGpuGraph = await gpuGraph();
    async function run(seed, fresh = false) {
      if (fresh) await model.setSample(sample);
      document.getElementById('seed-input').value = seed;
      await api.runInference();
      if (JSON.stringify(sample.neighbors) !== graph || await gpuGraph() !== originalGpuGraph) throw Error('Graph mutated');
      const coords = [...await model.coordinates()];
      return { seed, coords, lengths: attachments.map(([a, b]) => Math.hypot(
        ...[0, 1, 2].map(axis => coords[a * 3 + axis] - coords[b * 3 + axis]))) };
    }
    const a = await run(seedA), b = await run(seedB), again = await run(seedA), fresh = await run(seedB, true);
    const rms = (a, b) => Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / a.length);
    const old = await import(URL.createObjectURL(new Blob([source], { type: 'application/javascript' })));
    model.kernels = await old.Kernels.create(model.device, 'float32');
    model.kernels.bindGroups = new Map();
    const reference = await run(seedB, true);
    return { atoms: sample.atoms, seedA, seedB, steps, attachments, aLengths: a.lengths, bLengths: b.lengths,
      repeatRms: rms(a.coords, again.coords), freshRms: rms(b.coords, fresh.coords),
      referenceRms: rms(b.coords, reference.coords), referenceLengths: reference.lengths };
  }, { source, pdb: process.env.PDB_ID ?? '4BYH', seedA: Number(process.env.SEED_A ?? 20260902),
    seedB: Number(process.env.SEED_B ?? 20260903), steps: Number(process.env.STEPS ?? 8) });
  console.log(JSON.stringify(result, null, 2));
  assert.deepEqual(errors, []);
  assert.ok(result.attachments.length > 0);
  assert.equal(result.repeatRms, 0);
  assert.equal(result.freshRms, 0);
  assert.ok(result.referenceRms < 2e-4);
} finally { await browser.close(); }
