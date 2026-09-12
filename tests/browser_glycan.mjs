import assert from 'node:assert/strict';
import { chromium } from 'playwright';
const url = process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791';
const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan',
    '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
const deadline = setTimeout(() => browser.close(), 600_000);
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, ignoreHTTPSErrors: true });
  const errors = [], downloads = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => console.error(request.url().split('?')[0], request.failure()));
  page.on('response', response => {
    let request = response.request(), remote = false;
    while (request) {
      if (request.url().includes('huggingface.co')) remote = true;
      request = request.redirectedFrom();
    }
    if (remote) downloads.push([response.status(), response.url().split('?')[0]]);
  });
  // Exercise remote weights even when validating the local development page.
  await page.route('**/assets/model/manifest.json', async route => {
    const root = 'https://huggingface.co/murrellb/WSFMDocking/resolve/0930c08c8b441bc99077b436fdd3c390c365827c/webgpu/v8_ck_240000/';
    const response = await fetch(root + 'manifest.json');
    const manifest = await response.json();
    manifest.weight_file = root + manifest.weight_file;
    await route.fulfill({ json: manifest });
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  console.log('Page loaded');
  await page.evaluate(() => window.__wsfmdock.ready);
  console.log('Model loaded');
  assert.equal(await page.evaluate(() => window.__wsfmdock.model.weights.manifest.iteration), 240000);
  assert.equal(await page.evaluate(() => window.__wsfmdock.model.weights.manifest.checkpoint_sha256),
    '425083dee00c40a570dbf5495aeeeb1503b6d3f4c32516c1e1edcdce203e77ac');
  const prepared = await page.evaluate(async () => {
    await window.__wsfmdock.fetchPdb('4BYH');
    const ids = [...document.querySelectorAll('#ligand-list input')].slice(1).map(i => i.value);
    for (const id of ids) {
      const input = [...document.querySelectorAll('#ligand-list input')].find(i => i.value === id);
      input.checked = false; input.dispatchEvent(new Event('change'));
    }
    const sample = await window.__wsfmdock.useSelection();
    const attachments = sample.ligand_bonds.filter(([a, b]) => (sample.roles[a] === 3) !== (sample.roles[b] === 3));
    document.getElementById('step-select').value = '4';
    const initial = [...await window.__wsfmdock.model.coordinates()];
    await window.__wsfmdock.runInference();
    const final = [...await window.__wsfmdock.model.coordinates()];
    return { atoms: sample.atoms, ligandAtoms: sample.roles.filter(r => r === 3).length,
      attachments, finite: final.every(Number.isFinite),
      moved: Math.max(...final.map((v, i) => Math.abs(v - initial[i]))),
      status: document.getElementById('status').textContent };
  });
  console.log('Attached inference complete');
  assert.equal(prepared.ligandAtoms, 130);
  assert.equal(prepared.attachments.length, 1);
  assert.ok(prepared.finite && prepared.moved > 1);
  assert.match(prepared.status, /Inference complete/);
  await page.check('#show-reference');
  const pixels = async () => page.evaluate(() => {
    const v = window.__wsfmdock.viewer;
    v.renderer.render(v.scene, v.camera);
    const gl = v.renderer.getContext(), w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const data = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let bright = 0;
    for (let i = 0; i < data.length; i += 4) if (Math.max(data[i], data[i + 1], data[i + 2]) > 45) bright++;
    return bright;
  });
  const desktopPixels = await pixels();
  assert.ok(desktopPixels > 1000);
  await page.screenshot({ path: 'test-results-glycan-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.__wsfmdock.viewer.resetCamera());
  await page.screenshot({ path: 'test-results-glycan-mobile.png' });
  const mobilePixels = await pixels();
  assert.ok(mobilePixels > 100);
  await page.click('#example-tab');
  await page.waitForFunction(() => document.getElementById('step-status').textContent === 'Ready');
  await page.selectOption('#sample-select', 'glycan-free-100.json');
  await page.waitForFunction(() => window.__wsfmdock.sample.atoms === 19
    && document.getElementById('step-status').textContent === 'Ready');
  await page.evaluate(async () => {
    await window.__wsfmdock.runInference();
  });
  const free = await page.evaluate(async () => ({
    rendered: [...window.__wsfmdock.viewer.ligandByElement.values()].flat().length,
    atoms: window.__wsfmdock.sample.atoms,
    finite: [...await window.__wsfmdock.model.coordinates()].every(Number.isFinite),
  }));
  assert.equal(free.atoms, 19);
  assert.equal(free.rendered, free.atoms);
  assert.ok(free.finite);
  await page.selectOption('#sample-select', 'glycan-attached-5-perturbed.json');
  await page.waitForFunction(() => window.__wsfmdock.sample.atoms === 2392
    && document.getElementById('step-status').textContent === 'Ready');
  const perturbed = await page.evaluate(async () => {
    const api = window.__wsfmdock;
    const original = [...await api.model.coordinates()];
    document.getElementById('step-select').value = '8';
    await api.runInference();
    const coords = await api.model.coordinates();
    let moving = 0, maximum = 0;
    for (let a = 0; a < api.sample.atoms; a += 1) {
      if (!api.sample.coordinate_design[a]) for (let k = 0; k < 3; k += 1) {
        if (coords[3*a+k] !== original[3*a+k]) throw new Error('Fixed atom moved.');
      }
      if (api.sample.roles[a] !== 1 || !api.sample.coordinate_design[a]) continue;
      moving += 1;
      const item = api.viewer.backboneMeshes.find(v => v.atoms.includes(a));
      const index = item.atoms.indexOf(a);
      for (let k = 0; k < 3; k += 1) {
        maximum = Math.max(maximum, Math.abs(coords[a*3+k] - original[a*3+k]));
        if (Math.abs(item.mesh.instanceMatrix.array[index*16+12+k] - coords[a*3+k]) > 1e-4) {
          throw new Error('Displayed backbone differs from model output.');
        }
      }
      if (!api.viewer.referenceAtoms.includes(a)) throw new Error('Missing backbone reference ghost.');
    }
    return { moving, maximum, finite: [...coords].every(Number.isFinite),
      status: document.getElementById('status').textContent };
  });
  console.log('Perturbed inference', JSON.stringify(perturbed));
  assert.match(perturbed.status, /Inference complete/);
  assert.ok(perturbed.moving > 0 && perturbed.maximum > 0.1 && perturbed.finite);
  await page.screenshot({ path: 'test-results-backbone-mobile.png' });
  assert.deepEqual(errors, []);
  assert.ok(downloads.some(([status, address]) => [200, 302].includes(status) && address.includes('weights.f32')));
  assert.ok(downloads.some(([status]) => status === 200));
  const report = { prepared, desktopPixels, mobilePixels, free, perturbed, downloads };
  console.log(JSON.stringify(report, null, 2));
  if (process.env.RELEASE_REPORT) {
    const { writeFile } = await import('node:fs/promises');
    await writeFile(process.env.RELEASE_REPORT, JSON.stringify(report, null, 2));
  }
} finally { clearTimeout(deadline); await browser.close(); }
