const REMOTE_MODEL_MANIFEST = "https://huggingface.co/murrellb/WSFMDocking/resolve/0930c08c8b441bc99077b436fdd3c390c365827c/webgpu/v8_ck_240000/manifest.json";

const localBase = new URL(import.meta.env.BASE_URL, document.baseURI);

export const MODEL_MANIFEST_URL = import.meta.env.VITE_MODEL_MANIFEST_URL
  || REMOTE_MODEL_MANIFEST;

export function assetUrl(path) {
  return new URL(path.replace(/^\//, ""), localBase).href;
}
