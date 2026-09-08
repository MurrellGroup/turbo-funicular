const REMOTE_MODEL_MANIFEST = "https://huggingface.co/murrellb/WSFMDocking/resolve/5886cc4cf17575a57f54a6e25d3b7c2458a99d3c/webgpu/ck_240000/manifest.json";

const localBase = new URL(import.meta.env.BASE_URL, document.baseURI);

export const MODEL_MANIFEST_URL = import.meta.env.VITE_MODEL_MANIFEST_URL
  || (import.meta.env.PROD
    ? REMOTE_MODEL_MANIFEST
    : new URL("assets/model/manifest.json", localBase).href);

export function assetUrl(path) {
  return new URL(path.replace(/^\//, ""), localBase).href;
}
