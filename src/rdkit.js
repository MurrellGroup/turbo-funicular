import initRDKitModule from "../vendor/rdkit/RDKit_minimal.mjs";
import rdkitWasmUrl from "../vendor/rdkit/RDKit_minimal.wasm?url";

let modulePromise;

export function loadRdkit() {
  if (!modulePromise) {
    modulePromise = initRDKitModule({ locateFile: () => rdkitWasmUrl });
  }
  return modulePromise;
}
