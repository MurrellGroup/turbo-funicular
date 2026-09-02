import { chromium } from "playwright";

import { miniPdb } from "./fixtures.mjs";

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
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, ignoreHTTPSErrors: true });
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
  await page.click("#custom-tab");
  await page.fill("#pdb-id-input", "8BO9");
  await page.click("#fetch-pdb");
  await page.waitForFunction(() => window.__wsfmdock.sample?.id === "pdb-8BO9-assembly1.pdb");
  const fetched = await page.evaluate(() => ({
    atoms: window.__wsfmdock.sample.atoms,
    ligandAtoms: window.__wsfmdock.sample.roles.filter((role) => role === 3).length,
    directedEdges: window.__wsfmdock.sample.neighbors.filter(([atom]) => atom >= 0).length,
    aromaticEdges: window.__wsfmdock.sample.neighbors.filter(([, type]) => type === 3).length,
    graphSource: window.__wsfmdock.sample.graph_source,
  }));
  if (fetched.atoms !== 1402
    || fetched.ligandAtoms !== 32
    || fetched.directedEdges !== 72
    || fetched.aromaticEdges === 0
    || fetched.graphSource !== "RCSB CCD connectivity") {
    throw new Error(`PDB-ID preparation differs: ${JSON.stringify(fetched)}`);
  }

  await page.setInputFiles("#pdb-input", {
    name: "mini.pdb",
    mimeType: "chemical/x-pdb",
    buffer: Buffer.from(miniPdb()),
  });
  await page.waitForFunction(() => window.__wsfmdock.sample?.id === "pdb-mini.pdb");
  const pdb = await page.evaluate(() => ({
    atoms: window.__wsfmdock.sample.atoms,
    ligandAtoms: window.__wsfmdock.sample.roles.filter((role) => role === 3).length,
    directedEdges: window.__wsfmdock.sample.neighbors.filter(([atom]) => atom >= 0).length,
    graphSource: window.__wsfmdock.sample.graph_source,
  }));
  if (pdb.atoms !== 15
    || pdb.ligandAtoms !== 6
    || pdb.directedEdges !== 12
    || pdb.graphSource !== "RCSB CCD connectivity") {
    throw new Error(`PDB browser preparation differs: ${JSON.stringify(pdb)}`);
  }

  await page.evaluate(async (pdbText) => {
    await window.__wsfmdock.loadPdbText(pdbText, "unknown.pdb");
  }, miniPdb({ component: "@@@" }));
  await page.waitForFunction(() => window.__wsfmdock.sample?.id === "pdb-unknown.pdb");
  const unavailable = await page.evaluate(() => ({
    atoms: window.__wsfmdock.sample.atoms,
    ligandAtoms: window.__wsfmdock.sample.roles.filter((role) => role === 3).length,
    status: document.getElementById("status").textContent,
  }));
  if (unavailable.atoms !== 9
    || unavailable.ligandAtoms !== 0
    || !unavailable.status.includes("replacement SMILES")) {
    throw new Error(`Unavailable-graph fallback differs: ${JSON.stringify(unavailable)}`);
  }

  await page.evaluate(async (pdbText) => {
    await window.__wsfmdock.loadPdbText(pdbText, "mini.pdb");
  }, miniPdb());
  await page.waitForFunction(() => window.__wsfmdock.sample?.id === "pdb-mini.pdb");
  const rendering = await page.evaluate(() => ({
    backboneAtoms: [...window.__wsfmdock.viewer.backboneByElement.values()].flat().length,
    backboneBonds: window.__wsfmdock.viewer.backbonePairs.length,
    referenceAtoms: window.__wsfmdock.viewer.referenceAtoms.length,
  }));
  if (rendering.backboneAtoms !== 8 || rendering.backboneBonds !== 7 || rendering.referenceAtoms !== 7) {
    throw new Error(`Full-backbone/reference rendering differs: ${JSON.stringify(rendering)}`);
  }

  await page.fill("#smiles-input", "CC(=O)Oc1ccccc1C(=O)O");
  await page.click("#replace-ligand");
  await page.waitForFunction(() => window.__wsfmdock.sample?.id.endsWith("-smiles"));
  const smiles = await page.evaluate(() => ({
    atoms: window.__wsfmdock.sample.atoms,
    ligandAtoms: window.__wsfmdock.sample.roles.filter((role) => role === 3).length,
    aromaticEdges: window.__wsfmdock.sample.neighbors.filter(([, type]) => type === 3).length,
  }));
  if (smiles.atoms !== 22 || smiles.ligandAtoms !== 13 || smiles.aromaticEdges !== 12) {
    throw new Error(`SMILES browser replacement differs: ${JSON.stringify(smiles)}`);
  }

  await page.selectOption("#step-select", "4");
  await page.click("#run-button");
  await page.waitForFunction(
    () => document.getElementById("status").textContent.startsWith("Inference complete."),
    null,
    { timeout: 120_000 },
  );
  const inference = await page.evaluate(async () => {
    const sample = window.__wsfmdock.sample;
    const coords = await window.__wsfmdock.model.coordinates();
    let fixedMaximum = 0;
    for (let atom = 0; atom < sample.atoms; atom += 1) {
      if (sample.coordinate_design[atom]) continue;
      for (let axis = 0; axis < 3; axis += 1) {
        fixedMaximum = Math.max(
          fixedMaximum,
          Math.abs(coords[atom * 3 + axis] - sample.target_coords[atom][axis]),
        );
      }
    }
    return {
      finite: [...coords].every(Number.isFinite),
      fixedMaximum,
      status: document.getElementById("status").textContent,
    };
  });
  if (!inference.finite || inference.fixedMaximum >= 1e-6) {
    throw new Error(`Custom inference failed: ${JSON.stringify(inference)}`);
  }
  await page.check("#show-reference");
  const referenceVisible = await page.evaluate(() => window.__wsfmdock.viewer.referenceGroup.visible);
  if (!referenceVisible) throw new Error("Reference pose toggle did not show the reference layer.");
  await page.screenshot({ path: "test-results-custom-input.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "test-results-custom-input-mobile.png" });
  const mobile = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    controls: (() => {
      const bounds = document.querySelector(".controls").getBoundingClientRect();
      return { left: bounds.left, right: bounds.right, top: bounds.top, bottom: bounds.bottom };
    })(),
  }));
  if (mobile.documentWidth > mobile.viewportWidth
    || mobile.controls.left < 0
    || mobile.controls.right > mobile.viewportWidth
    || mobile.controls.top < 0
    || mobile.controls.bottom > 844) {
    throw new Error(`Mobile layout overflowed: ${JSON.stringify(mobile)}`);
  }
  if (errors.length) throw new Error(`Browser errors: ${errors.join("; ")}`);
  console.log(JSON.stringify({
    fetched, pdb, unavailable, rendering, smiles, inference, referenceVisible, mobile,
  }, null, 2));
} finally {
  await browser.close();
}
