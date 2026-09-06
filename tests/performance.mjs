import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
const browser = await chromium.launch({ headless: false,
  executablePath: process.env.CHROME_PATH ?? '/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist'] });
const baseline = execFileSync('git', ['show', 'd66e3a4:src/gpu.js'], { encoding: 'utf8' })
  .replace('this.uniforms = new Map();', 'this.uniforms = new Map(); this.bindGroups = new Map();');
const results = [];
try {
  for (const reference of [true, false, false, true]) {
    const page = await browser.newPage({ ignoreHTTPSErrors: true });
    if (reference) await page.route('**/src/gpu.js', r => r.fulfill({ body: baseline, contentType: 'application/javascript' }));
    await page.goto('https://127.0.0.1:8791');
    await page.evaluate(() => window.__wsfmdock.ready);
    const result = await page.evaluate(async () => {
      const model = window.__wsfmdock.model;
      const output = { tuning: model.kernels.tuning, attentionTuning: model.kernels.attentionTuning, samples: [] };
      for (const prefix of ['', 'glycan-']) {
        const fixture = await fetch(`/assets/parity/${prefix}transition.json`).then(r => r.json());
        const sample = await fetch(`/assets/samples/${fixture.sample_file}`).then(r => r.json());
        await model.setSample(sample);
        const times = [];
        for (let repeat = 0; repeat < 4; repeat++) {
          model.setCoordinates(new Float32Array(fixture.coords));
          const ms = await model.transition(fixture.start, fixture.end,
            new Float32Array(fixture.increment), new Float32Array(fixture.latent));
          if (repeat) times.push(ms);
        }
        const coords = await model.coordinates();
        let squared = 0;
        for (let i = 0; i < coords.length; i++) squared += (coords[i] - fixture.expected_coords[i]) ** 2;
        output.samples.push({ atoms: sample.atoms, times, rms: Math.sqrt(squared / coords.length) });
      }
      return output;
    });
    results.push({ reference, ...result });
    for (const sample of result.samples) assert.ok(sample.rms < 1e-4);
    console.log(JSON.stringify(results.at(-1)));
    await page.close();
  }
} finally { await browser.close(); }
