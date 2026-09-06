import assert from 'node:assert/strict';
import { chromium } from 'playwright';
const url = process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791';
const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan',
    '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, ignoreHTTPSErrors: true });
  const errors = [], downloads = [];
  page.on('pageerror', error => errors.push(error.message));
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
    const root = 'https://huggingface.co/murrellb/WSFMDocking/resolve/main/webgpu/ck_135000/';
    const response = await fetch(root + 'manifest.json');
    const manifest = await response.json();
    manifest.weight_file = root + manifest.weight_file;
    await route.fulfill({ json: manifest });
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => window.__wsfmdock.ready);
  assert.equal(await page.evaluate(() => window.__wsfmdock.model.weights.manifest.iteration), 135000);
  const prepared = await page.evaluate(async () => {
    const sample = await window.__wsfmdock.fetchPdb('4BYH');
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
  await page.evaluate(async () => {
    await window.__wsfmdock.loadSample('glycan-free-100.json');
    await window.__wsfmdock.runInference();
  });
  const free = await page.evaluate(async () => ({
    rendered: [...window.__wsfmdock.viewer.ligandByElement.values()].flat().length,
    atoms: window.__wsfmdock.sample.atoms,
    finite: [...await window.__wsfmdock.model.coordinates()].every(Number.isFinite),
  }));
  assert.equal(free.rendered, free.atoms);
  assert.ok(free.finite);
  assert.deepEqual(errors, []);
  assert.ok(downloads.some(([status, address]) => [200, 302].includes(status) && address.includes('weights.f32')));
  assert.ok(downloads.some(([status]) => status === 200));
  console.log(JSON.stringify({ prepared, desktopPixels, mobilePixels, free, downloads }, null, 2));
} finally { await browser.close(); }
