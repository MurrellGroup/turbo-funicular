import { mpnnInput, mpnnConstraints, proposalEdits } from './mpnn-input.js';

export class MpnnProposals {
  constructor() { this.nextId = 0; this.worker = null; this.pending = null; }
  cancel() { this.worker?.postMessage({ type: 'cancel' }); }
  async sample(sample, edits, settings, count, seed, onProgress = () => {}) {
    if (this.pending) throw new Error('ProteinMPNN is busy.');
    const input = mpnnInput(sample, edits, settings.family);
    const options = mpnnConstraints(input, settings);
    if (!this.worker) {
      this.worker = new Worker(new URL('./mpnn-worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = ({ data }) => {
        const pending = this.pending;
        if (!pending || data.id !== pending.id) return;
        if (data.type === 'progress') { pending.onProgress(data.text); return; }
        this.pending = null;
        if (data.type === 'error') pending.reject(Object.assign(new Error(data.error), { cancelled: data.cancelled }));
        else pending.resolve(data);
      };
      this.worker.onerror = event => {
        this.pending?.reject(new Error(event.message)); this.pending = null;
        this.worker.terminate(); this.worker = null;
      };
    }
    const data = await new Promise((resolve, reject) => {
      const id = ++this.nextId;
      this.pending = { id, resolve, reject, onProgress };
      this.worker.postMessage({ type: 'sample', id, input, options, settings, count, seed,
        wasmPaths: new URL('ort/', new URL(import.meta.env.BASE_URL, location.href)).href });
    });
    if (data.results.length !== count) throw new Error('ProteinMPNN returned the wrong number of sequences.');
    return { proposals: data.results.map(result => ({ edits: proposalEdits(input, result, edits),
      metadata: { ...result, designMask: [...result.designMask], family: settings.family, model: settings.model,
        backboneNoise: settings.backboneNoise, freshNoise: settings.freshNoise,
        provider: data.provider, manifest: data.manifest, constraints: options } })), mapping: input.mapping };
  }
}
