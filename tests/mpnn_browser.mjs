import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { mpnnFixture } from './fixtures.mjs';

const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
const context = await browser.newContext({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 1000 } });
const page = await context.newPage(), errors = [], downloads = [];
context.on('request', r => { if (r.url().includes('/webProteinMPNN/models/')) downloads.push(r.url()); });
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error') console.error(m.text()); });
try {
  await page.goto(process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791');
  await page.evaluate(() => window.__wsfmdock.ready);
  await page.evaluate(pdb => window.__wsfmdock.loadPdbText(pdb, 'numbering.pdb'), mpnnFixture());
  await page.evaluate(() => window.__wsfmdock.useSelection());
  await page.evaluate(() => window.__wsfmdock.editor.commit([[0, 'W'], [2, 'X'], [7, 'X']]));
  assert.equal(downloads.length, 0);
  assert.equal(await page.evaluate(() => window.__wsfmdock.mpnn.worker), null);
  await page.check('#mpnn-enable');
  assert.equal(downloads.length, 0);
  assert.equal(await page.locator('#mpnn-model option').count(), 9);
  assert.match(await page.locator('#mpnn-context').textContent(), /A: 5 residues \/ 1 X \/ 3 masked/);
  assert.match(await page.locator('#mpnn-context').textContent(), /B: 3 residues \/ 1 X \/ 1 masked/);
  await page.locator('#mpnn-options summary').click();
  await page.selectOption('#mpnn-backend', process.env.MPNN_BACKEND ?? 'wasm');
  await page.fill('#mpnn-temperature', '0.8');
  await page.fill('#mpnn-noise', '0.02');
  await page.fill('#mpnn-batch', '2');
  await page.fill('#mpnn-constraints', JSON.stringify({ tiedPositions: [[{ chain: 'A', number: 1, insertionCode: 'A' }, { chain: 'B', number: 104 }]], omitByResidue: [{ position: { chain: 'A', number: 1, insertionCode: 'A' }, aminoAcids: 'C' }], tiedConstraintMode: 'intersection' }));
  await page.selectOption('#step-select', '4');
  await page.fill('#seed-input', '52');
  const first = await page.evaluate(async () => {
    const a = window.__wsfmdock, native = a.editor.sample, sample = a.mpnn.sample.bind(a.mpnn), transition = a.model.transition.bind(a.model);
    let ready = false, calls = 0;
    a.mpnn.sample = async (...args) => { const result = await sample(...args); ready = result.proposals.length === 2; return result; };
    a.model.transition = (...args) => { if (!ready) throw new Error('WSFM ran before all MPNN sequences were ready'); calls++; return transition(...args); };
    const started = performance.now();
    await a.runCampaign(2);
    a.mpnn.sample = sample; a.model.transition = transition;
    return { calls, milliseconds: performance.now() - started, status: document.getElementById('status').textContent,
      results: a.campaign.map(r => ({ resolved: r.resolved, mpnn: r.mpnn,
        finite: [...r.coords].every(Number.isFinite), backboneExact: r.sample.roles.every((role, i) => role !== 1
          || r.coords.slice(i * 3, i * 3 + 3).every((v, d) => v === Math.fround(r.sample.target_coords[i][d]))) })),
      unchanged: native === a.editor.sample && a.editor.edits.get(2) === 'X' && a.editor.edits.get(7) === 'X' };
  });
  console.log(JSON.stringify({ calls: first.calls, milliseconds: first.milliseconds, status: first.status,
    sequences: first.results.map(r => r.mpnn.sequence), provider: first.results[0]?.mpnn.provider }));
  assert.equal(first.results.length, 2, first.status); assert.equal(first.calls, 8); assert.ok(first.unchanged);
  for (const r of first.results) {
    assert.ok(r.finite && r.backboneExact); assert.equal(r.resolved[0], 'W'); assert.equal(r.resolved[2], r.resolved[7]);
    assert.notEqual(r.resolved[2], 'X'); assert.notEqual(r.resolved[2], 'C');
    assert.deepEqual(Object.keys(r.mpnn.chains), ['A', 'B']);
    assert.equal(r.mpnn.residueMapping.find(p => p.residueId === 2).insertionCode, 'A');
  }
  const fetched = downloads.length;
  await page.evaluate(() => window.__wsfmdock.runCampaign(2));
  assert.equal(downloads.length, fetched, 'cached model must not be downloaded twice');
  assert.deepEqual(await page.evaluate(() => window.__wsfmdock.campaign.map(r => r.mpnn.sequence)), first.results.map(r => r.mpnn.sequence));
  await page.locator('#mpnn-options').scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results-mpnn-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'test-results-mpnn-mobile.png' });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const models = process.env.MPNN_ALL_MODELS ? await page.locator('#mpnn-model option').evaluateAll(opts => opts.map(o => o.value)) : ['solublempnn/v_48_020', 'ca/v_48_020'];
  const variants = [];
  for (const name of models) {
    await page.selectOption('#mpnn-model', name);
    const result = await page.evaluate(async () => {
      const a = window.__wsfmdock, settings = a.mpnnControls.settings(), start = performance.now();
      const r = await a.mpnn.sample(a.editor.sample, new Map(a.editor.edits), settings, 1, 123);
      return { name: settings.family + '/' + settings.model, milliseconds: performance.now() - start,
        sequence: r.proposals[0].metadata.sequence, provider: r.proposals[0].metadata.provider };
    });
    variants.push(result); console.log(JSON.stringify(result));
  }
  await page.fill('#mpnn-constraints', '{}');
  await page.check('#mpnn-fresh-noise');
  const fresh = await page.evaluate(async () => {
    const a = window.__wsfmdock, result = await a.mpnn.sample(a.editor.sample, new Map(a.editor.edits), a.mpnnControls.settings(), 2, 90);
    return result.proposals.map(r => ({ seed: r.metadata.seed, noiseSeed: r.metadata.noiseSeed }));
  });
  assert.deepEqual(fresh.map(r => r.noiseSeed), [90, 91]);
  await page.click('#mpnn-suggest');
  await page.waitForFunction(() => document.querySelector('#stop-campaign').hidden, null, { timeout: 120000 });
  assert.equal(await page.evaluate(() => [...window.__wsfmdock.editor.edits.values()].includes('X')), false);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ lazy: true, cache: true, allSequencesFirst: true, variants, downloads: downloads.length }, null, 2));
} catch (e) { await page.screenshot({ path: 'test-results-mpnn-failure.png' }); throw e; }
finally { await browser.close(); }
