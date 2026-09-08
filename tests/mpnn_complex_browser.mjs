import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791');
  await page.evaluate(() => window.__wsfmdock.ready);
  await page.evaluate(() => window.__wsfmdock.fetchPdb('4BYH'));
  await page.evaluate(() => window.__wsfmdock.useSelection());
  const before = await page.evaluate(() => {
    const a = window.__wsfmdock, edits = [];
    for (const chain of a.editor.chains) {
      for (const residue of chain.residues) {
        try { a.editor.commit([[residue.id, 'X']]); edits.push(residue.id); break; } catch {}
      }
    }
    return { chains: a.editor.chains.map(c => c.id), edits, atoms: a.editor.sample.atoms };
  });
  assert.ok(before.edits.length >= 2);
  await page.check('#mpnn-enable');
  await page.selectOption('#step-select', '4');
  const result = await page.evaluate(async () => {
    const a = window.__wsfmdock, base = a.editor.sample;
    const ligandEdges = sample => sample.atom_labels.flatMap((label, i) => sample.neighbors.slice(i * 10, i * 10 + 10)
      .filter(([j]) => j > i && (sample.roles[i] === 3 || sample.roles[j] === 3))
      .map(([j, type]) => [label, sample.atom_labels[j], type]));
    const start = performance.now(); await a.runCampaign(2);
    return { ms: performance.now() - start, status: document.getElementById('status').textContent,
      samples: a.campaign.map(r => ({ sequence: r.mpnn.sequence, chains: Object.keys(r.mpnn.chains),
        provider: r.mpnn.provider, finite: [...r.coords].every(Number.isFinite),
        fixed: r.sample.roles.every((role, i) => role !== 1 || r.coords.slice(i * 3, i * 3 + 3).every((x, d) => x === Math.fround(r.sample.target_coords[i][d]))),
        ligandEdges: ligandEdges(r.sample), ligandLabels: r.sample.atom_labels.filter((_, i) => r.sample.roles[i] === 3) })),
      ligandEdges: ligandEdges(base), ligandLabels: base.atom_labels.filter((_, i) => base.roles[i] === 3) };
  });
  assert.equal(result.samples.length, 2, result.status);
  for (const r of result.samples) {
    assert.deepEqual(r.chains, before.chains); assert.ok(r.finite && r.fixed);
    assert.deepEqual(r.ligandLabels, result.ligandLabels);
    assert.deepEqual(r.ligandEdges, result.ligandEdges);
  }
  const cancelled = await page.evaluate(async () => {
    const a = window.__wsfmdock;
    const pending = a.mpnn.sample(a.editor.sample, new Map(a.editor.edits), a.mpnnControls.settings(), 200, 234,
      () => a.mpnn.cancel());
    try { await pending; return false; } catch (e) { return Boolean(e.cancelled); }
  });
  assert.ok(cancelled); assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ...before, ms: result.ms, samples: result.samples.map(r => ({ chains: r.chains, provider: r.provider, finite: r.finite, fixed: r.fixed })), cancelled }));
} finally { await browser.close(); }
