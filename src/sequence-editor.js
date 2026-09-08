import { AMINO_ACIDS, AA_NAMES, parseSequences, proteinChains, validateEdit } from './sequence.js';

export class SequenceEditor {
  constructor({ onChange, onSelect, onError }) {
    this.onChange = onChange; this.onSelect = onSelect; this.onError = onError;
    this.edits = new Map(); this.selected = new Set(); this.results = new Map();
    this.chainSelections = new Map();
    this.navigationRevision = 0;
    this.ui = Object.fromEntries(['sequence-panel', 'sequence-chain', 'sequence-input', 'sequence-file',
      'open-sequence', 'align-sequence', 'sequence-record', 'sequence-grid', 'alignment-status',
      'residue-identity', 'set-identity', 'swap-selected', 'swap-all', 'clear-edits', 'selected-residues', 'close-sequence',
      'edit-sequence', 'mutation-list'].map(id => [id, document.getElementById(id)]));
    const u = this.ui;
    for (const aa of AMINO_ACIDS + 'X') u['residue-identity'].add(new Option(aa === 'X' ? 'X / Random' : `${aa} / ${AA_NAMES[aa]}`, aa));
    u['open-sequence'].onclick = () => u['sequence-file'].click();
    u['sequence-file'].onchange = async () => {
      try {
        const file = u['sequence-file'].files[0]; if (!file) return;
        if (file.size > 1024 * 1024) throw new Error('Sequence file exceeds 1 MiB.');
        u['sequence-input'].value = await file.text(); await this.align();
      } catch (e) { onError(e.message); }
    };
    u['align-sequence'].onclick = () => this.align().catch(e => onError(e.message));
    u['sequence-input'].oninput = () => { this.cancelMapping(); this.results.clear(); this.selected.clear(); this.render(); };
    u['sequence-record'].onchange = () => this.mapRecord().catch(e => onError(e.message));
    u['sequence-chain'].onchange = () => this.switchChain(u['sequence-chain'].value);
    u['set-identity'].onclick = () => this.commit([...this.selected].map(id => [id, u['residue-identity'].value]));
    u['swap-selected'].onclick = () => {
      const result = this.results.get(u['sequence-chain'].value);
      this.commit((result?.columns ?? []).filter(c => c.mapped && c.residue && this.selected.has(c.residue.id)
        && c.input !== c.reference).map(c => [c.residue.id, c.input]));
    };
    u['swap-all'].onclick = () => {
      const result = this.results.get(u['sequence-chain'].value);
      this.commit((result?.columns ?? []).filter(c => c.mapped && c.residue && c.input !== c.reference)
        .map(c => [c.residue.id, c.input]));
    };
    u['clear-edits'].onclick = () => { this.edits.clear(); this.render(); onChange(this.edits); };
    u['close-sequence'].onclick = () => { u['sequence-panel'].hidden = true; };
    u['edit-sequence'].onclick = () => { u['sequence-panel'].hidden = false; this.render(); };
  }

  setSample(sample) {
    if (this.sample === sample) { this.setBusy(false); return; }
    this.cancelMapping();
    this.sample = sample; this.edits = new Map(); this.selected = new Set(); this.results.clear();
    this.chainSelections.clear(); this.activeChain = undefined; this.lastSelected = undefined;
    this.chains = sample ? proteinChains(sample) : [];
    this.ui['sequence-chain'].replaceChildren(...this.chains.map(c => new Option(`${c.id || '-'} / ${c.residues.length} residues`, c.id)));
    this.switchChain(this.ui['sequence-chain'].value, false);
    if (!this.chains.length) this.ui['sequence-panel'].hidden = true;
    this.setBusy(false); this.render();
  }

  setBusy(busy) {
    this.busy = busy;
    const disabled = busy || !this.chains?.length;
    for (const control of this.ui['sequence-panel'].querySelectorAll('button, input, textarea, select')) control.disabled = disabled;
    this.ui['close-sequence'].disabled = false;
    this.ui['edit-sequence'].disabled = disabled;
    if (!disabled) this.updateSelection();
  }

  cancelMapping() {
    this.worker?.terminate(); this.worker = null;
    this.cancelPending?.(); this.cancelPending = null;
  }

  async align() {
    if (!this.sample || this.busy) return;
    this.records = parseSequences(this.ui['sequence-input'].value);
    this.ui['sequence-record'].replaceChildren(...this.records.map((r, i) => new Option(r.name, String(i))));
    await this.mapRecord();
  }

  async mapRecord() {
    this.cancelMapping();
    const record = this.records?.[Number(this.ui['sequence-record'].value)];
    if (!record || !this.sample) return;
    const navigationRevision = this.navigationRevision;
    this.render();
    const worker = new Worker(new URL('./alignment-worker.js', import.meta.url), { type: 'module' });
    this.worker = worker;
    this.ui['alignment-status'].textContent = 'Aligning';
    const result = await new Promise((resolve, reject) => {
      this.cancelPending = () => resolve(null);
      worker.onmessage = ({ data }) => data.error ? reject(new Error(data.error)) : resolve(data.results);
      worker.onerror = event => reject(new Error(event.message));
      worker.postMessage({ chains: this.chains, sequence: record.sequence });
    }).finally(() => { worker.terminate(); if (this.worker === worker) { this.worker = null; this.cancelPending = null; } });
    if (!result) return;
    const best = [...result].sort((a, b) => b.score - a.score)[0];
    // Search every retained chain, but attach the query only to its best match.
    if (best?.aligned) {
      this.results.set(best.id, best);
      if (navigationRevision === this.navigationRevision) this.switchChain(best.id, false);
    }
    this.render();
  }

  switchChain(id, render = true) {
    this.navigationRevision++;
    if (this.activeChain !== undefined) this.chainSelections.set(this.activeChain, this.selected);
    this.activeChain = id;
    this.ui['sequence-chain'].value = id;
    this.selected = this.chainSelections.get(id) ?? new Set();
    this.lastSelected = undefined;
    if (render) this.render();
  }

  pick(atom, displayedSample) {
    if (!this.sample || this.busy || ![1, 2].includes(displayedSample.roles[atom])) return;
    // Flattened IDs can differ between the selection preview and prepared sample.
    const label = displayedSample.atom_labels?.[atom]?.split('|');
    if (label?.length !== 5) return;
    const chain = this.chains.find(c => c.id === label[2]);
    const residue = chain?.residues.find(r => {
      const original = this.sample.atom_labels[r.atoms[0]].split('|');
      return original[3] === label[3] && original[4] === label[4];
    });
    if (!residue) return;
    const id = residue.id;
    this.ui['sequence-panel'].hidden = false;
    this.switchChain(chain.id, false);
    this.selected.add(id); this.lastSelected = id; this.render();
    this.ui['sequence-grid'].querySelector(`[data-residue="${id}"]`)?.scrollIntoView({ block: 'nearest', inline: 'center' });
  }

  commit(entries) {
    try {
      if (this.busy || !this.sample) return;
      const residues = new Map(this.chains.flatMap(c => c.residues).map(r => [r.id, r]));
      for (const [id, aa] of entries) validateEdit(this.sample, residues.get(id), aa);
      for (const [id, aa] of entries) {
        if (aa === residues.get(id).aa) this.edits.delete(id); else this.edits.set(id, aa);
      }
      this.render(); this.onChange(this.edits);
    } catch (e) { this.onError(e.message); }
  }

  updateSelection() {
    const grid = this.ui['sequence-grid'];
    for (const cell of grid.querySelectorAll('[data-residue]')) {
      const selected = this.selected.has(Number(cell.dataset.residue));
      cell.classList.toggle('selected', selected); cell.setAttribute('aria-pressed', String(selected));
    }
    this.ui['selected-residues'].textContent = `${this.selected.size} selected / ${this.edits.size} edits`;
    this.ui['set-identity'].disabled = this.busy || !this.selected.size;
    const columns = this.results.get(this.ui['sequence-chain'].value)?.columns ?? [];
    this.ui['swap-selected'].disabled = this.busy || !columns.some(c => c.mapped && c.residue
      && this.selected.has(c.residue.id) && c.input !== c.reference);
    this.ui['swap-all'].disabled = this.busy || !columns.some(c => c.mapped && c.residue && c.input !== c.reference);
    this.ui['clear-edits'].disabled = this.busy || !this.edits.size;
    this.onSelect(this.selected);
  }

  render() {
    const u = this.ui, chain = this.chains?.find(c => c.id === u['sequence-chain'].value);
    const result = this.results.get(chain?.id);
    u['sequence-panel'].dataset.chain = chain?.id ?? '';
    u['alignment-status'].textContent = result
      ? `${result.score.toFixed(1)} score / ${(100 * result.identity).toFixed(1)}% identity / ${(100 * result.coverage).toFixed(1)}% coverage`
      : '';
    u['sequence-grid'].replaceChildren();
    const heading = document.createElement('div'); heading.className = 'sequence-row-labels';
    for (const text of ['Residue', 'PDB', 'Query', 'Edit']) { const row = document.createElement('span'); row.textContent = text; heading.append(row); }
    u['sequence-grid'].append(heading);
    for (const column of result?.columns ?? chain?.residues.map(residue => ({ residue, reference: residue.aa, input: '-', mapped: false })) ?? []) {
      const { residue } = column, cell = document.createElement('button');
      cell.className = 'residue-cell'; cell.type = 'button'; cell.disabled = !residue || this.busy;
      if (residue) cell.dataset.residue = residue.id;
      cell.classList.toggle('difference', column.mapped && column.reference !== column.input);
      cell.classList.toggle('unaligned', !column.mapped && !!result);
      cell.classList.toggle('edited', residue && this.edits.has(residue.id));
      cell.title = residue ? `${chain.id}:${residue.number} ${residue.name}` : `Query ${(column.queryIndex ?? 0) + 1}`;
      for (const text of [residue?.number ?? '-', column.reference, column.input, this.edits.get(residue?.id) ?? '-']) {
        const row = document.createElement('span'); row.textContent = text; cell.append(row);
      }
      cell.onclick = event => {
        if (event.shiftKey && this.lastSelected !== undefined) {
          const ids = chain.residues.map(r => r.id), a = ids.indexOf(this.lastSelected), b = ids.indexOf(residue.id);
          if (a >= 0) for (const id of ids.slice(Math.min(a, b), Math.max(a, b) + 1)) this.selected.add(id);
        } else if (this.selected.has(residue.id)) this.selected.delete(residue.id); else this.selected.add(residue.id);
        this.lastSelected = residue.id; this.updateSelection();
      };
      u['sequence-grid'].append(cell);
    }
    u['mutation-list'].textContent = this.chains?.flatMap(c => c.residues.filter(r => this.edits.has(r.id))
      .map(r => `${c.id}:${r.aa}${r.number}${this.edits.get(r.id)}`)).join(', ') ?? '';
    this.updateSelection();
  }
}
