import assert from 'node:assert/strict';
import test from 'node:test';
import { bindViewerColorControls } from '../src/viewer-color-controls.js';

function setup(t) {
  const node = () => ({ children: [], style: {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
  });
  const previous = globalThis.document;
  globalThis.document = { createElement: node, createTextNode: text => ({ textContent: text }) };
  t.after(() => {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  });
  const ui = Object.fromEntries(['protein-color', 'ligand-element-colors', 'chain-colors'].map(id => [id, node()]));
  const viewer = {
    proteinColorMode: 'role', ligandColorsByElement: true,
    proteinChainIds: new Set(['B']),
    chainColors: new Map([['B', { getHexString: () => '63cdb5' }], ['L', { getHexString: () => 'd98bd3' }]]),
    setProteinColorMode(mode) { this.proteinColorMode = mode; this.onSampleChange(); },
    setLigandColorsByElement(value) { this.ligandColorsByElement = value; this.onSampleChange(); },
  };
  bindViewerColorControls(viewer, ui);
  return { viewer, ui };
}

test('color controls initialize from the viewer and preserve elements by default', t => {
  const { ui } = setup(t);
  assert.equal(ui['protein-color'].value, 'role');
  assert.equal(ui['ligand-element-colors'].checked, true);
  assert.equal(ui['ligand-element-colors'].disabled, true);
  assert.equal(ui['chain-colors'].hidden, true);
});

test('UI changes call the viewer setters and show only relevant chain swatches', t => {
  const { viewer, ui } = setup(t);
  ui['protein-color'].value = 'chain'; ui['protein-color'].onchange();
  assert.equal(viewer.proteinColorMode, 'chain');
  assert.equal(ui['ligand-element-colors'].disabled, false);
  assert.equal(ui['chain-colors'].hidden, false);
  assert.equal(ui['chain-colors'].children.length, 1);
  ui['ligand-element-colors'].checked = false; ui['ligand-element-colors'].onchange();
  assert.equal(viewer.ligandColorsByElement, false);
  assert.equal(ui['chain-colors'].children.length, 2);
  ui['ligand-element-colors'].checked = true; ui['ligand-element-colors'].onchange();
  assert.equal(ui['chain-colors'].children.length, 1);
});

test('switching out of chain mode retains the ligand preference for the next switch', t => {
  const { viewer, ui } = setup(t);
  viewer.setProteinColorMode('chain'); viewer.setLigandColorsByElement(false);
  viewer.setProteinColorMode('role');
  assert.equal(ui['ligand-element-colors'].disabled, true);
  assert.equal(ui['ligand-element-colors'].checked, false);
  viewer.setProteinColorMode('chain');
  assert.equal(ui['ligand-element-colors'].disabled, false);
  assert.equal(ui['ligand-element-colors'].checked, false);
});

test('sample changes refresh legend labels as text, including blank chain identifiers', t => {
  const { viewer, ui } = setup(t);
  const label = '<img src=x onerror=alert(1)>';
  viewer.proteinChainIds = new Set(['', label]);
  viewer.chainColors = new Map([['', { getHexString: () => 'ffffff' }], [label, { getHexString: () => '000000' }]]);
  viewer.setProteinColorMode('chain');
  assert.equal(ui['chain-colors'].children[0].children[1].textContent, '(blank)');
  assert.equal(ui['chain-colors'].children[1].children[1].textContent, label);
  assert.equal(ui['chain-colors'].children[0].children[0].style.backgroundColor, '#ffffff');
});
