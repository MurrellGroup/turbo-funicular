import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { miniCcd, miniPdb } from './fixtures.mjs';
const browser = await chromium.launch({ headless: false,
  executablePath: '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/BEN.cif', r => r.fulfill({ body: miniCcd(), contentType: 'text/plain' }));
  await page.goto(process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791');
  await page.evaluate(() => window.__wsfmdock.ready);
  const first = miniPdb();
  const extra = first.split('\n').filter(l => l.startsWith('HETATM')).map(l =>
    l.slice(0, 6) + String(Number(l.slice(6,11)) + 100).padStart(5) + l.slice(11,21) + 'C'
    + l.slice(22,30) + (Number(l.slice(30,38)) + 30).toFixed(3).padStart(8) + l.slice(38)).join('\n');
  const text = first + '\n' + extra;
  const oldAllocation = await page.evaluate(() => window.__wsfmdock.model.sampleBuffers.sample.id);
  await page.evaluate(text => window.__wsfmdock.loadPdbText(text, 'two.pdb'), text);
  assert.equal(await page.locator('#ligand-list input:checked').count(), 2);
  assert.equal(await page.locator('#active-ligand option').count(), 2);
  assert.ok(await page.locator('#run-button').isDisabled());
  assert.equal(await page.evaluate(() => window.__wsfmdock.viewer.sample.atoms), 21);
  assert.equal(await page.evaluate(() => window.__wsfmdock.model.sampleBuffers.sample.id), oldAllocation);
  const before = await page.locator('#active-ligand').inputValue();
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const click = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer, i = v.sample.atoms - 1;
    const p = v.camera.position.clone().set(...v.sample.target_coords[i]).project(v.camera);
    const r = v.renderer.domElement.getBoundingClientRect();
    return { x: r.left + (p.x + 1) * r.width / 2, y: r.top + (1 - p.y) * r.height / 2 };
  });
  await page.mouse.click(click.x, click.y);
  assert.notEqual(await page.locator('#active-ligand').inputValue(), before);
  await page.fill('#smiles-input', 'CCO');
  await page.click('#replace-ligand');
  await page.waitForFunction(() => window.__wsfmdock.viewer.sample.atoms === 18);
  await page.click('#prepare-selection');
  await page.waitForFunction(() => !document.querySelector('#run-button').disabled);
  assert.equal(await page.evaluate(() => window.__wsfmdock.sample.roles.filter(r => r === 3).length), 9);
  await page.click('#remove-ligand');
  assert.equal(await page.locator('#ligand-list input:checked').count(), 1);
  await page.click('#prepare-selection');
  await page.waitForFunction(() => !document.querySelector('#run-button').disabled);
  assert.equal(await page.evaluate(() => window.__wsfmdock.sample.roles.filter(r => r === 3).length), 6);
  await page.evaluate(() => window.__wsfmdock.fetchPdb('4BYH'));
  assert.ok(await page.locator('#ligand-list input:checked').count() >= 2);
  const countBefore = await page.locator('#ligand-list input:checked').count();
  await page.locator('#chain-list input').first().uncheck();
  assert.ok(await page.locator('#ligand-list input:checked').count() < countBefore);
  await page.evaluate(() => window.__wsfmdock.fetchPdb('1HZH'));
  const preview = await page.evaluate(() => ({ atoms: window.__wsfmdock.viewer.sample.atoms,
    limit: window.__wsfmdock.model.maximumAtoms,
    bindingBytes: window.__wsfmdock.model.device.limits.maxStorageBufferBindingSize }));
  assert.ok(preview.atoms > 8224);
  assert.ok(preview.limit > 8224);
  await page.screenshot({ path: 'test-results-selection-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'test-results-selection-mobile.png' });
  await page.setViewportSize({ width: 1440, height: 900 });
  for (let i = 0; i < await page.locator('#ligand-list input').count(); i++) await page.locator('#ligand-list input').nth(i).uncheck();
  await page.click('#prepare-selection');
  await page.waitForFunction(() => !document.querySelector('#run-button').disabled, null, { timeout: 120000 });
  const large = await page.evaluate(async () => {
    const m = window.__wsfmdock.model, n = m.sampleBuffers.sample.atoms;
    const { rng } = m.initialize(123);
    const noise = m.drawNoise(0, 1, rng);
    const ms = await m.transition(0, 1, noise.increment, noise.latent);
    const values = await m.coordinates();
    return { atoms: n, ms, finite: [...values].every(Number.isFinite), memoryBytes: m.memoryBytes() };
  });
  assert.ok(large.atoms > 8224 && large.finite);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ preview, large, multiLigandReplace: true, picking: true }, null, 2));
} finally { await browser.close(); }
