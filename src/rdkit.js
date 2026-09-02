import initRDKitModule from "@rdkit/rdkit";
import rdkitWasmUrl from "@rdkit/rdkit/dist/RDKit_minimal.wasm?url";

let modulePromise;

export function loadRdkit() {
  if (!modulePromise) {
    modulePromise = initRDKitModule({ locateFile: () => rdkitWasmUrl });
  }
  return modulePromise;
}
