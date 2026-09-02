const REMOTE_MODEL_MANIFEST = "https://huggingface.co/murrellb/WSFMDocking/resolve/main/webgpu/post_joint_100k/manifest.json";

const localBase = new URL(import.meta.env.BASE_URL, document.baseURI);

export const MODEL_MANIFEST_URL = import.meta.env.VITE_MODEL_MANIFEST_URL
  || (import.meta.env.PROD
    ? REMOTE_MODEL_MANIFEST
    : new URL("assets/model/manifest.json", localBase).href);

export function assetUrl(path) {
  return new URL(path.replace(/^\//, ""), localBase).href;
}
