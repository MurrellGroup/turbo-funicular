import { chromium } from "playwright";
import { readFile, writeFile } from "node:fs/promises";

const fixture = JSON.parse(await readFile(
  new URL(`../public/assets/parity/${process.env.PARITY_PREFIX ?? 'v8_fixed_'}transition.json`, import.meta.url),
  "utf8",
));
const trajectory = JSON.parse(await readFile(
  new URL(`../public/assets/parity/${process.env.PARITY_PREFIX ?? 'v8_fixed_'}trajectory.json`, import.meta.url),
  "utf8",
));

const executablePath = process.env.CHROME_PATH
  ?? "/home/murrellb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome";
const url = process.env.WSFMDock_WEBGPU_URL ?? "https://127.0.0.1:8791";
const browser = await chromium.launch({
  headless: false,
  executablePath,
  args: [
    "--enable-unsafe-webgpu",
    "--use-angle=vulkan",
    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    "--ignore-gpu-blocklist",
  ],
});

try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  if (process.env.LOCAL_MODEL) {
    await page.route('https://huggingface.co/**/manifest.json', async route => {
      const manifest = JSON.parse(await readFile(new URL('../public/assets/model/manifest.json', import.meta.url)));
      manifest.weight_file = new URL('/assets/model/weights.f32', url).href;
      await route.fulfill({ json: manifest });
    });
  }
  if (process.env.COMPACT_GPU) await page.addInitScript(() => {
    const request = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = function (options) {
      return request.call(this, { ...options, requiredLimits: {
        ...options.requiredLimits, maxComputeWorkgroupStorageSize: 16384,
      } });
    };
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("requestfailed", (request) => errors.push(
    `${request.method()} ${request.url()}: ${request.failure()?.errorText}`,
  ));
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await page.evaluate(() => window.__wsfmdock.ready);
  const result = await page.evaluate(async ({ fixture, trajectory }) => {
    const model = window.__wsfmdock.model;
    if (model.weights.manifest.checkpoint_sha256 !== fixture.checkpoint_sha256
      || fixture.checkpoint_sha256 !== trajectory.checkpoint_sha256) throw new Error("Checkpoint mismatch");
    const sample = await fetch(new URL(`assets/samples/${fixture.sample_file}`, location.href)).then(r => r.json());
    await model.setSample(sample);
    const metrics = (actual, expected, movingOnly) => {
      let sum = 0;
      let reference = 0;
      let maximum = 0;
      let count = 0;
      for (let index = 0; index < actual.length; index += 1) {
        if (!Number.isFinite(actual[index])) throw new Error("Nonfinite browser output");
        if (!sample.coordinate_design[Math.floor(index / 3)] && actual[index] !== expected[index]) {
          throw new Error("Fixed coordinate differs");
        }
        if (movingOnly && !sample.coordinate_design[Math.floor(index / 3)]) continue;
        const difference = actual[index] - expected[index];
        sum += difference * difference;
        reference += expected[index] * expected[index];
        maximum = Math.max(maximum, Math.abs(difference));
        count += 1;
      }
      return {
        rms: Math.sqrt(sum / count),
        relativeRms: Math.sqrt(sum / reference),
        maximum,
      };
    };
    model.setCoordinates(new Float32Array(fixture.coords));
    const milliseconds = await model.transition(
      fixture.start,
      fixture.end,
      new Float32Array(fixture.increment),
      new Float32Array(fixture.latent),
    );
    const coordinates = await model.coordinates();
    const secant = await model.secantEndpoint();
    model.setCoordinates(new Float32Array(trajectory.initial_coords));
    const trajectoryMetrics = [];
    for (const transition of trajectory.transitions) {
      await model.transition(
        transition.start,
        transition.end,
        new Float32Array(transition.increment),
        new Float32Array(transition.latent),
      );
      trajectoryMetrics.push(metrics(
        await model.coordinates(),
        transition.expected_coords,
        true,
      ));
    }
    return {
      milliseconds,
      coordinates: metrics(coordinates, fixture.expected_coords, false),
      movingCoordinates: metrics(coordinates, fixture.expected_coords, true),
      secant: metrics(secant, fixture.expected_secant, false),
      movingSecant: metrics(secant, fixture.expected_secant, true),
      trajectory: trajectoryMetrics,
    };
  }, { fixture, trajectory });
  if (errors.length) throw new Error(`Browser errors: ${errors.join("; ")}`);
  if (result.movingCoordinates.rms >= 1e-4) {
    throw new Error(`Coordinate RMS parity failed: ${result.movingCoordinates.rms}`);
  }
  if (result.movingSecant.rms >= 5e-4) {
    throw new Error(`Secant RMS parity failed: ${result.movingSecant.rms}`);
  }
  const trajectoryRms = Math.max(...result.trajectory.map((entry) => entry.rms));
  if (trajectoryRms >= 2e-4) {
    throw new Error(`Trajectory RMS parity failed: ${trajectoryRms}`);
  }
  console.log(JSON.stringify(result, null, 2));
  if (process.env.PARITY_REPORT) await writeFile(process.env.PARITY_REPORT,
    JSON.stringify({ checkpoint_sha256: fixture.checkpoint_sha256, sample: fixture.sample_file, ...result }, null, 2));
} finally {
  await browser.close();
}
