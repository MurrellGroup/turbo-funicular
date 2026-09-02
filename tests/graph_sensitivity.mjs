import { chromium } from "playwright";

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
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await page.evaluate(() => window.__wsfmdock.ready);
  const results = await page.evaluate(async () => {
    const catalog = await fetch("/assets/samples/catalog.json").then((response) => response.json());
    const model = window.__wsfmdock.model;
    const distance = (coords, left, right) => {
      const offsetLeft = left * 3;
      const offsetRight = right * 3;
      return Math.hypot(
        coords[offsetLeft] - coords[offsetRight],
        coords[offsetLeft + 1] - coords[offsetRight + 1],
        coords[offsetLeft + 2] - coords[offsetRight + 2],
      );
    };
    const results = [];
    for (const [sampleIndex, entry] of catalog.samples.entries()) {
      await window.__wsfmdock.loadSample(entry.file);
      const sample = window.__wsfmdock.sample;
      const originalNeighbors = new Int32Array(sample.neighbors.flat());
      const noNeighbors = new Int32Array(originalNeighbors.length).fill(-1);

      async function rollout(neighbors) {
        model.device.queue.writeBuffer(model.sampleBuffers.neighbors, 0, neighbors);
        const { rng } = model.initialize(20260902 + sampleIndex);
        const steps = 8;
        for (let index = 0; index < steps; index += 1) {
          const start = index / steps;
          const end = (index + 1) / steps;
          const noise = model.drawNoise(start, end, rng);
          await model.transition(start, end, noise.increment, noise.latent);
        }
        return model.coordinates();
      }

      const conditioned = await rollout(originalNeighbors);
      const ablated = await rollout(noNeighbors);
      model.device.queue.writeBuffer(model.sampleBuffers.neighbors, 0, originalNeighbors);
      const target = new Float32Array(sample.target_coords.flat());
      const bondMae = (coords) => sample.ligand_bonds.reduce(
        (total, [left, right]) => total + Math.abs(
          distance(coords, left, right) - distance(target, left, right)
        ),
        0,
      ) / sample.ligand_bonds.length;
      let square = 0;
      let maximum = 0;
      let count = 0;
      for (let atom = 0; atom < sample.atoms; atom += 1) {
        if (sample.roles[atom] !== 3) continue;
        for (let axis = 0; axis < 3; axis += 1) {
          const index = atom * 3 + axis;
          const difference = conditioned[index] - ablated[index];
          square += difference * difference;
          maximum = Math.max(maximum, Math.abs(difference));
          count += 1;
        }
      }
      results.push({
        sample: entry.id,
        ligandRms: Math.sqrt(square / count),
        ligandMaximum: maximum,
        conditionedBondMae: bondMae(conditioned),
        ablatedBondMae: bondMae(ablated),
      });
    }
    return results;
  });
  console.log(JSON.stringify(results, null, 2));
  if (results.some((result) => !(result.ligandRms > 1e-6))) {
    throw new Error("At least one browser trajectory is insensitive to the molecular graph");
  }
} finally {
  await browser.close();
}
