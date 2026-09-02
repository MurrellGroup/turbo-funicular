# WSFMDock WebGPU

Inference-only browser runtime for the legacy CKDock 12x408 few-step checkpoint.
It uses custom WebGPU kernels and keeps molecular pair conditioning sparse.

## Prepare assets

```bash
python tools/export_model.py \
  --checkpoint /path/to/final.pt \
  --output public/assets/model \
  --precision float32

python tools/export_samples.py \
  --ckdock /path/to/CKDock \
  --plinder-data /path/to/processed_full_context_v1 \
  --plinder-graphs /path/to/plinder_smiles_graph_train_v3 \
  --output public/assets/samples
```

The exported model payload is deliberately ignored by Git. FP32 is the broad
WebGPU path; `--precision float16` selects the faster optional `shader-f16`
path on browsers that expose it.

## Run

```bash
npm install
npm run dev
```

The development server uses HTTPS because WebGPU requires a secure context.

## Parity

Generate the explicit-noise PyTorch fixture with `tools/export_parity.py`, then
run the browser test under an X server:

```bash
xvfb-run -a npm run test:browser
```

Attention uses online softmax over 4x32 query/key tiles. Molecular bonds remain
an `atoms x 10` sparse neighbor list and are resolved inside each attention
tile; no `atoms x atoms` pair-feature allocation is made.

