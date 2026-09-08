import { copyFile, mkdir } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
await mkdir(new URL('public/ort/', root), { recursive: true });
for (const flavor of ['', '.jsep', '.asyncify']) for (const ext of ['wasm', 'mjs']) {
  const name = `ort-wasm-simd-threaded${flavor}.${ext}`;
  await copyFile(new URL(`node_modules/onnxruntime-web/dist/${name}`, root), new URL(`public/ort/${name}`, root));
}
