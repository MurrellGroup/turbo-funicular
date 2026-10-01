export function bindViewerColorControls(viewer, ui) {
  viewer.onSampleChange = () => {
    const byChain = viewer.proteinColorMode === 'chain';
    ui['protein-color'].value = viewer.proteinColorMode;
    ui['ligand-element-colors'].checked = viewer.ligandColorsByElement;
    ui['ligand-element-colors'].disabled = !byChain;
    ui['chain-colors'].hidden = !byChain;
    const colors = [...viewer.chainColors].filter(([id]) =>
      !viewer.ligandColorsByElement || viewer.proteinChainIds.has(id));
    ui['chain-colors'].replaceChildren(...colors.map(([id, color]) => {
      const entry = document.createElement('span'), swatch = document.createElement('i');
      swatch.style.backgroundColor = `#${color.getHexString()}`;
      entry.append(swatch, document.createTextNode(id || '(blank)')); return entry;
    }));
  };
  ui['protein-color'].onchange = () => viewer.setProteinColorMode(ui['protein-color'].value);
  ui['ligand-element-colors'].onchange = () =>
    viewer.setLigandColorsByElement(ui['ligand-element-colors'].checked);
  viewer.onSampleChange();
}
