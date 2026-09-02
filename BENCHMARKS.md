# Validation record

Measured on 2026-09-02 on `blackwizzle`, NVIDIA RTX PRO 6000 Blackwell,
Chromium 149 Linux/Vulkan WebGPU. The browser adapter exposed FP32 but not the
optional `shader-f16` feature. The production trainer remained active during
these measurements.

## Numerical parity

The fixture is one explicit-noise `s=0` to `t=0.25` CK transition for Plinder
record 19206 (295 atoms). It exercises all 12 local blocks, all five finite
blocks, all endpoint updates, entity bias, and the legacy block-0 bond bias.

| Comparison | Moving-coordinate RMS | Maximum | Moving-secant RMS | Maximum |
| --- | ---: | ---: | ---: | ---: |
| WebGPU FP32 vs dense PyTorch | 5.72e-6 A | 4.37e-5 A | 2.29e-5 A | 1.75e-4 A |
| Native FlexAttention vs dense PyTorch | 2.71e-6 A | 3.05e-5 A | 1.08e-5 A | 1.22e-4 A |

`xvfb-run -a npm run test:browser` enforces RMS limits of `1e-4 A` for the
state and `5e-4 A` for the secant endpoint.

## Browser timing

Four consecutive CK maps, excluding weight download and pipeline creation:

| System | Atoms | Map times | Median |
| --- | ---: | --- | ---: |
| Compact pocket | 295 | 35.5, 32.5, 28.5, 30.5 ms | 31.5 ms |
| Medium pocket | 456 | 49.5, 44.3, 45.1, 43.4 ms | 44.7 ms |
| Large pocket | 702 | 75.0, 69.6, 70.9, 68.2 ms | 70.3 ms |

The compact system reports 309.0 MiB of model plus activation buffers. The
large system reports 334.2 MiB. Pair conditioning is stored as ten sparse
neighbors per atom. Online-softmax attention uses a fixed 4x32 tile and does
not allocate an atom-pair matrix.

## Visual checks

Desktop (1440x900) and mobile (390x844) four-step runs completed with nonblank
full-viewport canvases, visible protein/side-chain/ligand geometry, working
controls, and no text overlap. Sampled screenshots had 686 and 1,105 distinct
colors respectively after downsampling to 100x100.

