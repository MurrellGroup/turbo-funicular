import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { mpnnFixture } from './fixtures.mjs';

const names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'TRP', 'TYR', 'VAL'];
const pdb = mpnnFixture().split('\n').map((line, i) => line.slice(0, 17) + names[Math.floor(i / 5)] + line.slice(20)).join('\n');
const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 1000 } });
  const errors = []; page.on('pageerror', e => { errors.push(e.message); console.error(e.message); });
  await page.goto(process.env.WSFMDock_WEBGPU_URL ?? 'https://127.0.0.1:8791');
  await page.waitForFunction(() => window.__wsfmdock, null, { timeout: 30000 });
  await page.evaluate(() => window.__wsfmdock.ready);
  await page.evaluate(text => window.__wsfmdock.loadPdbText(text, 'chains.pdb'), pdb);
  await page.evaluate(() => window.__wsfmdock.useSelection());
  await page.click('#edit-sequence');
  await page.fill('#sequence-input', 'AKNAC');
  await page.click('#align-sequence');
  await page.waitForFunction(() => window.__wsfmdock.editor.results.size === 2);
  assert.equal(await page.locator('#sequence-chain').inputValue(), 'A');
  await page.locator('[data-residue="1"]').click();
  const highlight = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer, wanted = window.__wsfmdock.editor.chains[0].residues[1];
    const mesh = v.backboneMeshes.find(m => m.atoms.includes(wanted.atoms[0]));
    const index = mesh.atoms.indexOf(wanted.atoms[0]);
    return { color: [...mesh.mesh.instanceColor.array.slice(index * 3, index * 3 + 3)],
      selected: [...v.highlightedResidues], bonds: [...v.backbone.instanceColor.array],
      radius: Math.hypot(...mesh.mesh.instanceMatrix.array.slice(index * 16, index * 16 + 3)) };
  });
  assert.equal(highlight.color[0], 1); assert.ok(highlight.color[1] > 0.5 && highlight.color[2] < 0.05);
  assert.ok(highlight.bonds.some((v, i) => i % 3 === 0 && v === 1));
  assert.ok(highlight.radius > 0.1);
  await page.selectOption('#protein-color', 'chain');
  assert.equal(await page.locator('#chain-colors span').count(), 2);
  const colors = await page.evaluate(() => [...window.__wsfmdock.viewer.chainColors].map(([id, c]) => [id, c.getHexString()]));
  assert.notEqual(colors[0][1], colors[1][1]);
  await page.click('#swap-all');
  assert.deepEqual(await page.evaluate(() => [...window.__wsfmdock.editor.edits]), [[1, 'K'], [3, 'A']]);
  // A displayed preview may have compacted IDs; original chain/number/insertion wins.
  await page.evaluate(() => {
    const a = window.__wsfmdock, sample = a.viewer.sample;
    const atom = sample.atom_labels.findIndex(label => label === 'CA|TYR|B|102|');
    const preview = { ...sample, residue_ids: sample.residue_ids.map(() => 1) };
    a.editor.pick(atom, preview);
  });
  assert.equal(await page.locator('#sequence-chain').inputValue(), 'B');
  assert.equal(await page.locator('[data-residue="6"]').getAttribute('aria-pressed'), 'true');
  assert.match(await page.locator('[data-residue="6"]').getAttribute('title'), /B:102 TYR/);
  await page.selectOption('#residue-identity', 'X'); await page.click('#set-identity');
  await page.selectOption('#sequence-chain', 'A');
  assert.equal(await page.locator('[data-residue="1"]').getAttribute('aria-pressed'), 'true');
  assert.deepEqual(await page.evaluate(() => [...window.__wsfmdock.editor.edits]), [[1, 'K'], [3, 'A'], [6, 'X']]);
  await page.selectOption('#sequence-chain', 'B');
  assert.equal(await page.locator('[data-residue="6"]').getAttribute('aria-pressed'), 'true');
  await page.selectOption('#sequence-chain', 'A');
  await page.locator('[data-residue="2"]').click();
  await page.locator('[data-residue="3"]').click();
  await page.click('#close-sequence');
  // Real renderer ray-pick, not just the editor callback.
  const point = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer;
    const atom = v.sample.atom_labels.findIndex(label => label === 'CA|VAL|B|104|');
    const p = v.camera.position.clone().set(...v.sample.target_coords[atom]);
    v.controls.target.copy(p); v.camera.position.copy(p).add({ x: 0, y: 0, z: 12 }); v.controls.update();
    const r = v.renderer.domElement.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  });
  await page.waitForTimeout(300); await page.mouse.click(point.x, point.y);
  assert.equal(await page.locator('#sequence-chain').inputValue(), 'B');
  assert.equal(await page.locator('[data-residue="7"]').getAttribute('aria-pressed'), 'true');
  await page.selectOption('#sequence-chain', 'A');
  assert.deepEqual(await page.evaluate(() => [...window.__wsfmdock.editor.selected]), [1, 2, 3]);
  await page.click('#reset-camera');
  await page.screenshot({ path: 'test-results-sequence-display-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'test-results-sequence-display-mobile.png' });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  const pixels = await page.evaluate(() => {
    const v = window.__wsfmdock.viewer; v.renderer.render(v.scene, v.camera);
    const gl = v.renderer.getContext(), data = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let count = 0; for (let i = 0; i < data.length; i += 4) if (Math.max(data[i], data[i + 1], data[i + 2]) > 70) count++;
    return count;
  });
  assert.ok(pixels > 100); assert.deepEqual(errors, []);
  console.log(JSON.stringify({ colors, highlight: highlight.color, radius: highlight.radius, pixels,
    chainSwitch: true, authorNumberMapping: true, selectionsPreserved: true, swapAll: true, actualPicking: true }));
} catch (e) { throw e; }
finally { await browser.close(); }
