import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { parsePdb } from '../src/prep.js';

async function checkPdbDownload(page, expected) {
  await page.selectOption('#export-format', 'pdb');
  const pending = page.waitForEvent('download'); await page.click('#download-result');
  const download = await pending;
  assert.equal(download.suggestedFilename(), `sample-${expected.seed}.pdb`);
  const text = await readFile(await download.path(), 'utf8');
  const rows = text.split('\n').filter(l => /^(ATOM  |HETATM)/.test(l));
  assert.equal(rows.length, expected.sample.atoms);
  const indices = new Map(expected.sample.atom_labels.map((label, i) => [label, i]));
  for (const row of rows) {
    const label = [row.slice(12, 16).trim(), row.slice(17, 20).trim(), row.slice(21, 22).trim(),
      String(Number(row.slice(22, 26))), row.slice(26, 27).trim()].join('|');
    const atom = indices.get(label); assert.notEqual(atom, undefined, label);
    for (let d = 0; d < 3; d++) assert.ok(Math.abs(Number(row.slice(30 + 8 * d, 38 + 8 * d))
      - expected.coords[3 * atom + d] - expected.sample.coordinate_origin[d]) <= 0.000501);
  }
  if (process.env.GEMMI_PYTHON) {
    const r = spawnSync(process.env.GEMMI_PYTHON, ['-c', 'import gemmi,sys\ns=gemmi.read_pdb_string(sys.stdin.read())\nprint(sum(len(r) for c in s[0] for r in c))'], { input: text, encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr); assert.equal(Number(r.stdout), expected.sample.atoms);
  }
  return text;
}

const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 1000 } });
const errors = []; page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
try {
  await page.goto(process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791');
  await page.evaluate(() => window.__wsfmdock.ready);
  await page.evaluate(() => window.__wsfmdock.fetchPdb('1HVR'));
  // CSO is an unsupported incomplete modified-residue component in this entry.
  for (const row of await page.locator('#ligand-list .choice-row').all()) {
    if ((await row.textContent()).includes('CSO')) await row.locator('input').uncheck();
  }
  const chainIds = await page.locator('#chain-list input').evaluateAll(inputs => inputs.map(i => i.value));
  assert.ok(chainIds.length >= 2);
  await page.locator('#chain-list input').nth(1).uncheck();
  await page.evaluate(() => window.__wsfmdock.useSelection());
  const initial = await page.evaluate(() => {
    const e = window.__wsfmdock.editor;
    return { chains: e.chains.map(c => c.id), sequence: e.chains[0].sequence };
  });
  assert.equal(initial.chains.length, 1);
  const query = [...initial.sequence]; query[20] = query[20] === 'W' ? 'A' : 'W'; query[40] = query[40] === 'Y' ? 'F' : 'Y';
  await page.click('#edit-sequence');
  await page.locator('#sequence-file').setInputFiles({ name: 'variant.fasta', mimeType: 'text/plain', buffer: Buffer.from('>variant\n' + query.join('')) });
  await page.waitForFunction(() => window.__wsfmdock.editor.results.size === 1);
  assert.equal(await page.locator('#sequence-chain option').count(), 1);
  const diffs = page.locator('.residue-cell.difference'); assert.equal(await diffs.count(), 2);
  await diffs.first().click(); await page.click('#swap-selected');
  assert.equal(await page.evaluate(() => window.__wsfmdock.editor.edits.size), 1);
  await page.click('#close-sequence');
  // Restore both chains, then verify picking a protein atom opens its own sequence.
  await page.locator('#chain-list input').nth(1).check();
  await page.evaluate(() => window.__wsfmdock.useSelection());
  assert.equal(await page.evaluate(() => window.__wsfmdock.editor.edits.size), 0);
  await page.evaluate(() => {
    const v = window.__wsfmdock.viewer;
    v.update(new Float32Array(v.sample.target_coords.flat()));
  });
  const picks = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer, r = v.renderer.domElement.getBoundingClientRect();
    return v.sample.roles.flatMap((role, i) => {
      if (role !== 1 || v.sample.chain_ids[i] !== 1) return [];
      const p = v.camera.position.clone().set(...v.sample.target_coords[i]).project(v.camera);
      return [{ x: r.left + (p.x + 1) * r.width / 2, y: r.top + (1 - p.y) * r.height / 2 }];
    });
  });
  for (const p of picks.slice(0, 80)) {
    await page.mouse.click(p.x, p.y);
    if (await page.locator('#sequence-panel').isVisible()
      && await page.locator('#sequence-chain').inputValue() === chainIds[1]) break;
  }
  assert.ok(await page.locator('#sequence-panel').isVisible());
  assert.equal(await page.locator('#sequence-chain').inputValue(), chainIds[1]);
  await page.selectOption('#residue-identity', 'X'); await page.click('#set-identity');
  assert.equal(await page.evaluate(() => [...window.__wsfmdock.editor.edits.values()].join('')), 'X');
  await page.selectOption('#sequence-chain', chainIds[0]);
  await page.locator('.residue-cell').nth(15).click();
  await page.selectOption('#residue-identity', 'W'); await page.click('#set-identity');
  await page.screenshot({ path: 'test-results-campaign-editor-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'test-results-campaign-editor-mobile.png' });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.click('#close-sequence');
  await page.selectOption('#step-select', '4');
  await page.fill('#seed-input', '2026'); await page.fill('#campaign-count', '3');
  await page.click('#run-campaign');
  await page.waitForFunction(() => window.__wsfmdock.campaign.length === 3 && document.querySelector('#stop-campaign').hidden, null, { timeout: 180000 });
  const audit = await page.evaluate(() => {
    const a = window.__wsfmdock, base = a.editor.sample;
    return a.campaign.map(r => ({ seed: r.seed, atoms: r.sample.atoms, resolved: r.resolved,
      finite: [...r.coords].every(Number.isFinite),
      backboneExact: r.sample.roles.every((role, i) => role !== 1 || r.coords.slice(i * 3, i * 3 + 3)
        .every((v, axis) => v === Math.fround(r.sample.target_coords[i][axis]))),
      backboneOriginal: r.sample.roles.every((role, i) => role !== 1 || r.sample.target_coords[i].every((v, axis) => {
        const j = base.residue_ids.findIndex((id, j) => id === r.sample.residue_ids[i] && base.atom_names[j] === r.sample.atom_names[i]);
        return v === base.target_coords[j][axis];
      })),
      attached: r.sample.ligand_bonds.length,
    }));
  });
  assert.ok(audit.every(r => r.finite && r.backboneExact && r.backboneOriginal));
  assert.ok(new Set(audit.map(r => JSON.stringify(r.resolved))).size > 1);
  await page.selectOption('#campaign-result', '0');
  assert.equal(await page.evaluate(() => window.__wsfmdock.viewer.sample === window.__wsfmdock.campaign[0].sample), true);
  await page.click('#next-result');
  assert.equal(await page.locator('#campaign-result').inputValue(), '1');
  await page.selectOption('#export-format', 'json');
  const downloaded = page.waitForEvent('download');
  await page.click('#download-result');
  const download = await downloaded;
  assert.equal(download.suggestedFilename(), 'sample-2027.json');
  const saved = JSON.parse(await readFile(await download.path(), 'utf8'));
  assert.equal(saved.seed, 2027); assert.equal(saved.coords.length, saved.sample.atoms * 3);
  assert.equal(saved.checkpoint, await page.evaluate(() => window.__wsfmdock.model.weights.manifest.checkpoint_sha256));
  await checkPdbDownload(page, saved);
  await page.click('#show-reference');
  await page.screenshot({ path: 'test-results-campaign-results-desktop.png' });
  const repeated = await page.evaluate(async () => {
    const a = window.__wsfmdock, first = a.campaign.map(r => ({ resolved: r.resolved, coords: [...r.coords] }));
    await a.runCampaign(3);
    return first.every((r, i) => JSON.stringify(r.resolved) === JSON.stringify(a.campaign[i].resolved)
      && r.coords.every((v, j) => v === a.campaign[i].coords[j]));
  });
  assert.ok(repeated);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.selectOption('#campaign-result', '0');
  await page.screenshot({ path: 'test-results-campaign-results-mobile.png' });
  const mobilePixels = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer; v.renderer.render(v.scene, v.camera);
    const gl = v.renderer.getContext(), pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let n = 0; for (let i = 0; i < pixels.length; i += 4) if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 50) n++;
    return n;
  });
  assert.ok(mobilePixels > 1000);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(() => { document.querySelector('#campaign-count').value = '10'; document.querySelector('#run-campaign').click(); });
  await page.waitForFunction(() => !document.querySelector('#stop-campaign').hidden);
  await page.click('#stop-campaign');
  await page.waitForFunction(() => document.querySelector('#stop-campaign').hidden);
  assert.ok(await page.evaluate(() => window.__wsfmdock.campaign.length < 10));
  await page.evaluate(() => window.__wsfmdock.fetchPdb('4BYH'));
  await page.evaluate(() => window.__wsfmdock.useSelection());
  const glycan = await page.evaluate(async () => {
    const a = window.__wsfmdock, original = a.sample, editor = a.editor;
    const attachments = original.ligand_bonds.filter(([i, j]) => (original.roles[i] === 3) !== (original.roles[j] === 3));
    const endpoints = new Set(attachments.flatMap(([i, j]) => [i, j]).filter(i => original.roles[i] !== 3));
    const protectedId = original.residue_ids[[...endpoints][0]];
    editor.commit([[protectedId, 'X']]);
    const protectedEditRejected = !editor.edits.has(protectedId);
    const free = editor.chains.flatMap(c => c.residues).find(r => !r.atoms.some(i => endpoints.has(i)));
    editor.commit([[free.id, 'X']]);
    const expected = attachments.map(([i, j]) => [original.atom_labels[i], original.atom_labels[j]].sort().join('~')).sort();
    document.getElementById('seed-input').value = '9876';
    await a.runCampaign(2);
    const check = a.campaign.map(result => {
      const s = result.sample;
      const edges = s.ligand_bonds.filter(([i, j]) => (s.roles[i] === 3) !== (s.roles[j] === 3));
      return { labels: edges.map(([i, j]) => [s.atom_labels[i], s.atom_labels[j]].sort().join('~')).sort(),
        finite: [...result.coords].every(Number.isFinite),
        rawGraph: edges.every(([i, j]) => s.neighbors.slice(i * 10, i * 10 + 10).some(([k, t]) => k === j && t === 0)),
        referenceExact: s.reference_sample === original || s === original };
    });
    return { protectedEditRejected, expected, check };
  });
  assert.ok(glycan.protectedEditRejected);
  assert.equal(glycan.expected.length, 2);
  for (const g of glycan.check) { assert.deepEqual(g.labels, glycan.expected); assert.ok(g.finite && g.rawGraph && g.referenceExact); }
  const glycanResult = await page.evaluate(() => {
    const r = window.__wsfmdock.campaign[Number(document.querySelector('#campaign-result').value)];
    return { seed: r.seed, coords: [...r.coords], sample: r.sample };
  });
  const glycanPdb = await checkPdbDownload(page, glycanResult);
  const exportedLinks = parsePdb(glycanPdb).links.map(([a, b]) => [a, b].map(x =>
    `${x.atomName}|${x.rawResidue}|${x.chain}|${x.residueNumber}|${x.insertion}`).sort().join('~'));
  for (const attachment of glycan.expected) assert.ok(exportedLinks.includes(attachment));
  const pixels = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer; v.renderer.render(v.scene, v.camera);
    const gl = v.renderer.getContext(), data = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let colored = 0; for (let i = 0; i < data.length; i += 4) if (Math.max(data[i], data[i + 1], data[i + 2]) > 50) colored++;
    return colored;
  });
  assert.ok(pixels > 1000);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ audit, repeated, picking: true, retainedChainsOnly: true, cancelled: true, glycan, pixels, mobilePixels, download: true }, null, 2));
} catch (error) { await page.screenshot({ path: 'test-results-campaign-failure.png' }); throw error; }
finally { await browser.close(); }
