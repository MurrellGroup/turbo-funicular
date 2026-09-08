import { MODEL_CATALOG } from 'proteinmpnn-web/onnx';
import { mpnnInput } from './mpnn-input.js';

export class MpnnControls {
  constructor({ onSuggest, getInput, onError }) {
    this.getInput = getInput; this.onError = onError;
    this.ui = Object.fromEntries(['mpnn-enable', 'mpnn-options', 'mpnn-model', 'mpnn-temperature',
      'mpnn-noise', 'mpnn-fresh-noise', 'mpnn-backend', 'mpnn-threads', 'mpnn-batch', 'mpnn-omit',
      'mpnn-constraints', 'mpnn-context', 'mpnn-suggest', 'mpnn-map', 'mpnn-status',
      'mpnn-constraint-file', 'mpnn-open-constraints'].map(id => [id, document.getElementById(id)]));
    const u = this.ui;
    for (const [family, names] of Object.entries(MODEL_CATALOG)) {
      const group = document.createElement('optgroup'); group.label = { proteinmpnn: 'ProteinMPNN', solublempnn: 'solubleMPNN', ca: 'CA-only' }[family];
      for (const name of names) group.append(new Option(`${group.label} / ${name}`, `${family}/${name}`));
      u['mpnn-model'].append(group);
    }
    u['mpnn-model'].value = 'proteinmpnn/v_48_020';
    u['mpnn-threads'].max = String(crossOriginIsolated ? navigator.hardwareConcurrency || 1 : 1);
    u['mpnn-enable'].onchange = () => { u['mpnn-options'].hidden = !this.enabled; this.refresh(); };
    u['mpnn-model'].onchange = () => this.refresh();
    u['mpnn-suggest'].onclick = onSuggest;
    u['mpnn-map'].onclick = () => {
      try {
        const input = this.input();
        this.download({ residues: input.mapping, sequence: input.structure.sequence,
          residueIndices: [...input.structure.residueIndices], chainEncoding: [...input.structure.chainEncoding] }, 'mpnn-residue-map.json');
      } catch (e) { onError(e.message); }
    };
    u['mpnn-open-constraints'].onclick = () => u['mpnn-constraint-file'].click();
    u['mpnn-constraint-file'].onchange = async () => {
      try {
        const file = u['mpnn-constraint-file'].files[0]; if (!file) return;
        if (file.size > 8 * 2 ** 20) throw new Error('Constraints exceed 8 MiB.');
        const value = JSON.parse(await file.text());
        u['mpnn-constraints'].value = JSON.stringify(value, null, 2);
      } catch (e) { onError(e.message); }
    };
  }
  get enabled() { return this.ui['mpnn-enable'].checked; }
  input() {
    const { sample, edits } = this.getInput();
    if (!sample) throw new Error('Use a protein structure selection first.');
    return mpnnInput(sample, edits, this.ui['mpnn-model'].value.split('/')[0]);
  }
  settings() {
    const u = this.ui, [family, model] = u['mpnn-model'].value.split('/');
    const temperature = Number(u['mpnn-temperature'].value), backboneNoise = Number(u['mpnn-noise'].value);
    const threads = Number(u['mpnn-threads'].value), batchSize = Number(u['mpnn-batch'].value);
    if (!Number.isFinite(temperature) || temperature <= 0) throw new Error('MPNN temperature must be positive.');
    if (!Number.isFinite(backboneNoise) || backboneNoise < 0) throw new Error('Backbone noise must be nonnegative.');
    if (!Number.isInteger(threads) || threads < 1 || threads > Number(u['mpnn-threads'].max)) throw new Error(`Threads must be between 1 and ${u['mpnn-threads'].max}.`);
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 16) throw new Error('MPNN sequence batch must be between 1 and 16.');
    return { family, model, temperature, backboneNoise, threads, batchSize,
      freshNoise: u['mpnn-fresh-noise'].checked, backend: u['mpnn-backend'].value,
      omitAminoAcids: u['mpnn-omit'].value.toUpperCase().replace(/\s/g, ''),
      advanced: JSON.parse(u['mpnn-constraints'].value || '{}') };
  }
  setBusy(busy) {
    this.busy = busy;
    const { sample } = this.getInput();
    const unavailable = busy || !sample;
    this.ui['mpnn-enable'].disabled = unavailable;
    for (const c of this.ui['mpnn-options'].querySelectorAll('input, select, button, textarea')) c.disabled = unavailable;
    if (!unavailable) this.refresh();
  }
  refresh() {
    if (!this.enabled || this.busy) return;
    try {
      const input = this.input(), chains = new Map();
      for (const row of input.mapping) {
        if (!chains.has(row.chain)) chains.set(row.chain, { valid: 0, design: 0, masked: 0 });
        const chain = chains.get(row.chain); chain.valid += row.valid ? 1 : 0; chain.design += row.design ? 1 : 0; chain.masked += row.valid ? 0 : 1;
      }
      this.ui['mpnn-context'].textContent = [...chains].map(([id, c]) => `${id || '(blank)'}: ${c.valid} residues / ${c.design} X / ${c.masked} masked`).join('\n');
      this.ui['mpnn-suggest'].disabled = false;
      this.ui['mpnn-map'].disabled = false;
    } catch (error) {
      this.ui['mpnn-context'].textContent = error.message;
      this.ui['mpnn-suggest'].disabled = true;
      this.ui['mpnn-map'].disabled = true;
    }
  }
  status(text) { this.ui['mpnn-status'].textContent = text; }
  download(value, filename) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
