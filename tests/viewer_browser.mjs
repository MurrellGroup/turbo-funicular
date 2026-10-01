import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

// Serve the real viewer and UI bindings, without model downloads or WebGPU.
const root = fileURLToPath(new URL('../', import.meta.url));
const index = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const colorMarkup = index.match(/<label>Color by<select[\s\S]*?<div id="chain-colors"[^>]*><\/div>/)?.[0];
assert.ok(colorMarkup, 'viewer color controls must exist in index.html');
const html = `<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:16px}#viewport{width:min(640px,calc(100vw - 32px));height:480px}
canvas{display:block;width:100%;height:100%}label{display:block}</style></head><body>
<div id="viewport"></div>${colorMarkup}<script type="module">
import { MolecularViewer } from '/src/viewer.js';
import { bindViewerColorControls } from '/src/viewer-color-controls.js';
import { viewerSample } from '/tests/viewer-fixture.mjs';
const viewer = window.viewer = new MolecularViewer(document.getElementById('viewport'));
const ui = Object.fromEntries(['protein-color', 'ligand-element-colors', 'chain-colors']
  .map(id => [id, document.getElementById(id)]));
bindViewerColorControls(viewer, ui);
const sample = viewerSample();
viewer.setSample(sample, new Float32Array(sample.target_coords.flat()));
window.picks = []; viewer.onAtomPick = atom => window.picks.push(atom);
window.ready = true;
</script></body></html>`;
const server = await createServer({ root, configFile: false,
  server: { host: '127.0.0.1', port: 0 },
  plugins: [{ name: 'viewer-test-fixture', configureServer(s) {
    s.middlewares.use('/__viewer_test__', (_req, res) => {
      res.setHeader('Content-Type', 'text/html'); res.end(html);
    });
  } }],
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 700 }, hasTouch: true });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__viewer_test__`);
  await page.waitForFunction(() => window.ready);
  const frames = (n = 2) => page.evaluate(async n => {
    for (let i = 0; i < n; i++) await new Promise(requestAnimationFrame);
  }, n);
  const pose = () => page.evaluate(() => {
    const v = window.viewer;
    return { position: v.camera.position.toArray(), up: v.camera.up.toArray(),
      quaternion: v.camera.quaternion.toArray(), target: v.controls.target.toArray(),
      radius: v.camera.position.distanceTo(v.controls.target) };
  });
  const drag = async (dx, dy, button = 'left') => {
    const r = await page.locator('canvas').boundingBox();
    const x = r.x + r.width / 2 - dx / 2, y = r.y + r.height / 2 - dy / 2;
    await page.mouse.move(x, y); await page.mouse.down({ button });
    await page.mouse.move(x + dx, y + dy, { steps: 16 });
    await page.mouse.up({ button }); await frames();
  };
  await frames();

  // UI: opt into chain colors; preserve elements by default, or let ligands follow chains.
  assert.ok(await page.locator('#ligand-element-colors').isDisabled());
  await page.selectOption('#protein-color', 'chain');
  assert.ok(await page.locator('#ligand-element-colors').isChecked());
  assert.ok(await page.locator('#ligand-element-colors').isEnabled());
  assert.equal(await page.locator('#chain-colors span').count(), 2);
  await page.uncheck('#ligand-element-colors');
  assert.equal(await page.evaluate(() => window.viewer.ligandColorsByElement), false);
  assert.equal(await page.locator('#chain-colors span').count(), 4);
  await page.selectOption('#protein-color', 'role');
  assert.ok(await page.locator('#ligand-element-colors').isDisabled());
  await page.selectOption('#protein-color', 'chain');
  assert.equal(await page.locator('#ligand-element-colors').isChecked(), false);
  await page.check('#ligand-element-colors');
  assert.equal(await page.locator('#chain-colors span').count(), 2);

  // Multiple complete tumbles must cross both poles without a fixed-up clamp.
  await page.evaluate(() => {
    const v = window.viewer; v.controls.staticMoving = true;
    v.camera.position.copy(v.controls.target).add(v.camera.up.clone().set(0, 0, 30));
    v.camera.up.set(0, 1, 0); v.controls.update();
  });
  const ups = [];
  for (let i = 0; i < 8; i++) {
    await drag(0, 360);
    const p = await pose(); ups.push(p.up[1]);
    assert.ok([...p.position, ...p.quaternion, ...p.up].every(Number.isFinite));
    assert.ok(Math.abs(p.radius - 30) < 1e-6);
  }
  assert.ok(Math.min(...ups) < -0.8, 'camera must tumble upside down through a pole');
  assert.ok(Math.max(...ups.slice(3)) > 0.8, 'camera must continue through the other pole');

  // Horizontal rotation remains usable when looking exactly down either world-Y pole.
  for (const pole of [-1, 1]) {
    await page.evaluate(pole => {
      const v = window.viewer;
      v.camera.position.copy(v.controls.target).add(v.camera.up.clone().set(0, pole * 30, 0));
      v.camera.up.set(0, 0, -pole); v.controls.update();
    }, pole);
    const before = await pose(); await drag(100, 0); const after = await pose();
    assert.notDeepEqual(after.position, before.position);
    assert.ok([...after.position, ...after.quaternion].every(Number.isFinite));
    assert.ok(Math.abs(after.radius - before.radius) < 1e-6);
  }

  // Reset also removes roll and all pending damping, not just camera position.
  await page.evaluate(() => { window.viewer.controls.staticMoving = false; });
  await drag(100, 80);
  await page.evaluate(() => window.viewer.resetCamera());
  const reset = await pose(); await frames(12);
  assert.deepEqual(await pose(), reset);
  assert.deepEqual(reset.up, [0, 1, 0]);
  await page.evaluate(() => {
    const v = window.viewer; const controls = v.controls;
    v.setSample(v.sample, new Float32Array(v.sample.target_coords.flat()), true);
    if (v.controls !== controls) throw new Error('preserveCamera replaced the controls');
    v.controls.staticMoving = true;
  });
  assert.deepEqual(await pose(), reset);

  // Zoom, right-drag pan and modified left-drag pan still work.
  await page.mouse.move(300, 200);
  await page.mouse.wheel(0, 120); await frames();
  assert.notEqual((await pose()).radius, reset.radius);
  let before = await pose(); await drag(80, 30, 'right'); let after = await pose();
  assert.notDeepEqual(after.target, before.target);
  assert.ok(Math.abs(after.radius - before.radius) < 1e-6);
  before = after;
  await page.keyboard.down('Shift'); await drag(60, 20); await page.keyboard.up('Shift');
  after = await pose();
  assert.notDeepEqual(after.target, before.target);
  assert.ok(Math.abs(after.radius - before.radius) < 1e-6);

  // Pick a ligand after a rolled view; a drag itself must not trigger selection.
  assert.deepEqual(await page.evaluate(() => window.picks), []);
  await page.evaluate(() => {
    const v = window.viewer; v.resetCamera();
    const rotation = v.camera.quaternion.clone().setFromAxisAngle(v.camera.up.clone().set(0, 0, 1), 0.7);
    v.camera.position.sub(v.controls.target).applyQuaternion(rotation).add(v.controls.target);
    v.camera.up.applyQuaternion(rotation); v.controls.update();
  });
  await frames();
  const pick = await page.evaluate(() => {
    const v = window.viewer, atom = 7;
    const p = v.camera.position.clone().set(...v.sample.target_coords[atom]).project(v.camera);
    const r = v.renderer.domElement.getBoundingClientRect();
    return { x: r.left + (p.x + 1) * r.width / 2, y: r.top + (1 - p.y) * r.height / 2, atom };
  });
  await page.mouse.click(pick.x, pick.y);
  assert.deepEqual(await page.evaluate(() => window.picks), [pick.atom]);

  // Responsive resize must update TrackballControls' cached pointer bounds.
  await page.setViewportSize({ width: 390, height: 700 }); await frames();
  assert.equal(await page.evaluate(() => {
    const v = window.viewer;
    return v.controls.screen.width === v.renderer.domElement.getBoundingClientRect().width;
  }), true);

  // Touch rotation, pinch zoom and two-finger pan use the same unrestricted controls.
  await page.evaluate(() => { window.viewer.resetCamera(); window.viewer.controls.staticMoving = true; });
  const session = await page.context().newCDPSession(page);
  const touch = async (type, points) => {
    await session.send('Input.dispatchTouchEvent', { type, touchPoints: points }); await frames();
  };
  before = await pose();
  await touch('touchStart', [{ id: 1, x: 140, y: 200 }]);
  await touch('touchMove', [{ id: 1, x: 200, y: 260 }]);
  await touch('touchEnd', []);
  assert.notDeepEqual((await pose()).position, before.position);
  before = await pose();
  await touch('touchStart', [{ id: 1, x: 120, y: 200 }, { id: 2, x: 220, y: 200 }]);
  await touch('touchMove', [{ id: 1, x: 100, y: 240 }, { id: 2, x: 250, y: 240 }]);
  await touch('touchEnd', []);
  after = await pose();
  assert.notEqual(after.radius, before.radius);
  assert.notDeepEqual(after.target, before.target);
  await session.detach();
  assert.deepEqual(errors, []);
  console.log('Viewer browser checks passed: pole crossings, reset, pan/zoom, picking, resize, touch, and color settings.');
} finally {
  await browser?.close();
  await server.close();
}
