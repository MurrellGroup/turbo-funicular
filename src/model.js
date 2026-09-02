import {
  Kernels,
  WeightStore,
  halfArray,
  makeBuffer,
  readF32,
  uploadBuffer,
} from "./gpu.js";
import { validateSample } from "./sample.js";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const groups = (items, width = 256) => [Math.ceil(items / width)];

function flatten(values, Type = Float32Array) {
  if (ArrayBuffer.isView(values)) return new Type(values);
  const source = Array.isArray(values[0]) ? values.flat() : values;
  return new Type(source);
}

function gaussian(rng) {
  let first = 0;
  let second = 0;
  while (first === 0) first = rng();
  while (second === 0) second = rng();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

class SampleBuffers {
  constructor(device, sample) {
    this.device = device;
    this.sample = sample;
    this.atoms = sample.atoms;
    const metadata = new Uint32Array(this.atoms * 4);
    for (let atom = 0; atom < this.atoms; atom += 1) {
      metadata[atom * 4] = sample.atomic_numbers[atom];
      metadata[atom * 4 + 1] = sample.roles[atom];
      metadata[atom * 4 + 2] = sample.residue_types[atom];
      metadata[atom * 4 + 3] = sample.atom_names[atom];
    }
    this.metadata = uploadBuffer(device, metadata, STORAGE, "sample.metadata");
    this.entities = uploadBuffer(device, new Int32Array(sample.entity_ids), STORAGE, "sample.entities");
    this.neighbors = uploadBuffer(device, flatten(sample.neighbors, Int32Array), STORAGE, "sample.neighbors");
    this.target = uploadBuffer(device, flatten(sample.target_coords), STORAGE, "sample.target");
    this.baseMeans = uploadBuffer(device, flatten(sample.base_means), STORAGE, "sample.base_means");
    this.baseScales = uploadBuffer(device, new Float32Array(sample.base_scales), STORAGE, "sample.base_scales");
    this.design = uploadBuffer(device, new Uint32Array(sample.coordinate_design), STORAGE, "sample.design");
  }

  destroy() {
    for (const value of Object.values(this)) {
      if (value instanceof GPUBuffer) value.destroy();
    }
  }
}

export class DockingWebGpuModel {
  static async create(device, manifestUrl = "/assets/model/manifest.json", suppliedManifest = null) {
    const manifest = suppliedManifest ?? await fetch(manifestUrl).then((response) => response.json());
    const [weights, kernels] = await Promise.all([
      WeightStore.load(device, manifestUrl, manifest),
      Kernels.create(device, manifest.activation_precision),
    ]);
    return new DockingWebGpuModel(device, weights, kernels);
  }

  constructor(device, weights, kernels) {
    this.device = device;
    this.weights = weights;
    this.kernels = kernels;
    this.config = weights.manifest.config;
    this.float16 = weights.manifest.activation_precision === "float16";
    this.elementBytes = this.float16 ? 2 : 4;
    const largestPerAtomBuffer = Math.max(
      2016,
      3 * this.config.heads * 64,
      this.config.heads * 74,
      2 * this.config.ff_hidden_dim,
    ) * this.elementBytes;
    const bufferLimit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize,
    );
    this.maximumAtoms = Math.floor(bufferLimit / largestPerAtomBuffer);
    this.sampleBuffers = null;
    this.buffers = null;
    this.currentCoords = null;
    this.nextCoords = null;
    this.zeroBias = uploadBuffer(
      device,
      this.float16 ? new Uint16Array(4080) : new Float32Array(4080),
      STORAGE,
      "zero_bias",
    );
    this.activationBytes = 0;
  }

  f16(elements, label) {
    this.activationBytes += elements * this.elementBytes;
    return makeBuffer(this.device, elements * this.elementBytes, STORAGE, label);
  }

  f32(elements, label) {
    this.activationBytes += elements * 4;
    return makeBuffer(this.device, elements * 4, STORAGE, label);
  }

  async setSample(sample) {
    validateSample(sample, this.maximumAtoms);
    if (this.sampleBuffers) this.sampleBuffers.destroy();
    if (this.buffers) {
      for (const buffer of Object.values(this.buffers).flatMap((value) => Array.isArray(value) ? value : [value])) {
        if (buffer instanceof GPUBuffer) buffer.destroy();
      }
    }
    this.activationBytes = 0;
    this.sampleBuffers = new SampleBuffers(this.device, sample);
    const n = sample.atoms;
    const d = this.config.dim;
    const h = this.config.heads;
    const hidden = this.config.ff_hidden_dim;
    this.buffers = {
      nodeA: this.f16(n * d, "node_a"),
      nodeB: this.f16(n * d, "node_b"),
      normalized: this.f16(n * d, "normalized"),
      deltaNode: this.f16(n * d, "delta_node"),
      projection: this.f16(n * 2016, "attention_projection"),
      qkvFeatures: this.f16(3 * h * n * 64, "qkv_features"),
      attended: this.f16(h * n * 64, "attended"),
      merged: this.f16(n * h * 74, "merged_attention"),
      ffProjection: this.f16(n * hidden * 2, "ff_projection"),
      ffHidden: this.f16(n * hidden, "ff_hidden"),
      affine: this.f16(d * 2, "adaln_affine"),
      localFourier: this.f16(d * 2, "local_fourier"),
      finiteFourier: this.f16(d * 6, "finite_fourier"),
      localCondition: this.f16(d, "local_condition"),
      finiteCondition: this.f16(d, "finite_condition"),
      noiseInput: this.f16(n * 6, "noise_input"),
      endpointDelta: this.f16(n * 3, "endpoint_delta"),
      correction: this.f32(n * 3, "local_correction"),
      residual: this.f32(n * 3, "finite_residual"),
      endpoints: [0, 1, 2].map((index) => this.f32(n * 3, `local_endpoint_${index}`)),
      secant: this.f32(n * 3, "finite_secant"),
      coordsA: this.f32(n * 3, "coords_a"),
      coordsB: this.f32(n * 3, "coords_b"),
      increment: this.f32(n * 3, "increment"),
      latent: this.f32(n * 3, "latent"),
      sharedNode: this.f16(n * d, "shared_node"),
      branchNodes: [0, 1, 2, 3, 4].map((index) => this.f16(n * d, `branch_node_${index}`)),
    };
    this.currentCoords = this.buffers.coordsA;
    this.nextCoords = this.buffers.coordsB;
  }

  makeFourier(value) {
    const frequencies = this.weights.manifest.time_frequencies;
    const output = new Float32Array(frequencies.length * 2);
    for (let index = 0; index < frequencies.length; index += 1) {
      const angle = value * frequencies[index];
      output[index] = Math.cos(angle);
      output[index + frequencies.length] = Math.sin(angle);
    }
    return output;
  }

  encodeActivation(values) {
    return this.float16 ? halfArray(values) : new Float32Array(values);
  }

  initialize(seed) {
    const sample = this.sampleBuffers.sample;
    const rng = mulberry32(seed);
    const coords = new Float32Array(sample.atoms * 3);
    for (let atom = 0; atom < sample.atoms; atom += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const index = atom * 3 + axis;
        coords[index] = sample.coordinate_design[atom]
          ? sample.base_means[atom][axis] + sample.base_scales[atom] * gaussian(rng)
          : sample.target_coords[atom][axis];
      }
    }
    this.device.queue.writeBuffer(this.buffers.coordsA, 0, coords);
    this.currentCoords = this.buffers.coordsA;
    this.nextCoords = this.buffers.coordsB;
    return { rng, coords };
  }

  setCoordinates(coords) {
    this.device.queue.writeBuffer(this.buffers.coordsA, 0, coords);
    this.currentCoords = this.buffers.coordsA;
    this.nextCoords = this.buffers.coordsB;
  }

  drawNoise(start, end, rng) {
    const sample = this.sampleBuffers.sample;
    const increment = new Float32Array(sample.atoms * 3);
    const latent = new Float32Array(sample.atoms * 3);
    for (let atom = 0; atom < sample.atoms; atom += 1) {
      const q = sample.base_scales[atom] ** 2 * (end - start) * (2 - start - end);
      for (let axis = 0; axis < 3; axis += 1) {
        const index = atom * 3 + axis;
        if (sample.coordinate_design[atom]) {
          increment[index] = Math.sqrt(Math.max(q, 0)) * gaussian(rng);
          latent[index] = gaussian(rng);
        }
      }
    }
    return { increment, latent };
  }

  matmul(pass, input, weightName, output, rows, columns, inner, biasName = null) {
    this.kernels.dispatch(
      pass,
      "matmul",
      [
        input,
        this.weights.get(weightName),
        biasName ? this.weights.get(biasName) : this.zeroBias,
        output,
        this.kernels.uniformU32([rows, columns, inner, biasName ? 1 : 0]),
      ],
      [Math.ceil(columns / 16), Math.ceil(rows / 16)],
    );
  }

  adaptiveNorm(pass, input, condition, prefix) {
    const { affine, normalized } = this.buffers;
    const n = this.sampleBuffers.atoms;
    const d = this.config.dim;
    this.matmul(
      pass,
      condition,
      `${prefix}.affine_weight`,
      affine,
      1,
      2 * d,
      d,
      `${prefix}.affine_bias`,
    );
    this.kernels.dispatch(
      pass,
      "adaln",
      [input, affine, this.weights.get(`${prefix}.norm`), normalized, this.kernels.uniformU32([n, d, 0, 0])],
      [n],
    );
    return normalized;
  }

  runBlock(pass, current, other, condition, coords, prefix, useBonds) {
    const b = this.buffers;
    const n = this.sampleBuffers.atoms;
    const d = this.config.dim;
    const h = this.config.heads;
    const hidden = this.config.ff_hidden_dim;
    let normalized = this.adaptiveNorm(pass, current, condition, `${prefix}.attention_norm`);
    this.matmul(pass, normalized, `${prefix}.attention.projection`, b.projection, n, 2016, d);
    this.kernels.dispatch(
      pass,
      "prepareQkv",
      [
        b.projection,
        coords,
        this.weights.get(`${prefix}.attention.qk_norm`),
        this.weights.get(`${prefix}.attention.head_weights`),
        b.qkvFeatures,
        this.kernels.uniformU32([n, h, 0, 0]),
      ],
      [n, h],
    );
    this.kernels.dispatch(
      pass,
      "attention",
      [
        b.qkvFeatures,
        this.sampleBuffers.entities,
        this.sampleBuffers.neighbors,
        this.weights.get(`${prefix}.attention.pair_bias`),
        b.attended,
        this.kernels.uniformU32([n, h, useBonds ? 1 : 0, 0]),
      ],
      [Math.ceil(n / 4), h],
    );
    this.kernels.dispatch(
      pass,
      "merge",
      [b.attended, coords, b.merged, this.kernels.uniformU32([n, h, 0, 0])],
      groups(n * h * 74),
    );
    this.matmul(pass, b.merged, `${prefix}.attention.output`, b.deltaNode, n, d, h * 74);
    this.kernels.dispatch(pass, "addF16", [current, b.deltaNode, other], groups(n * d));

    normalized = this.adaptiveNorm(pass, other, condition, `${prefix}.ffn_norm`);
    this.matmul(pass, normalized, `${prefix}.ffn.upgate`, b.ffProjection, n, hidden * 2, d);
    this.kernels.dispatch(
      pass,
      "swiglu",
      [b.ffProjection, b.ffHidden, this.kernels.uniformU32([n, hidden, 0, 0])],
      groups(n * hidden),
    );
    this.matmul(pass, b.ffHidden, `${prefix}.ffn.down`, b.deltaNode, n, d, hidden);
    this.kernels.dispatch(pass, "addF16", [other, b.deltaNode, current], groups(n * d));
    return current;
  }

  async transition(start, end, increment, latent) {
    const b = this.buffers;
    const n = this.sampleBuffers.atoms;
    const d = this.config.dim;
    const h = this.config.heads;
    const scalar = this.kernels.uniformF32([start, end, 0, 0]);
    const localFourier = this.makeFourier(start);
    const finiteFourier = new Float32Array(d * 6);
    finiteFourier.set(this.makeFourier(start), 0);
    finiteFourier.set(this.makeFourier(end), d * 2);
    finiteFourier.set(this.makeFourier(end - start), d * 4);
    this.device.queue.writeBuffer(b.localFourier, 0, this.encodeActivation(localFourier));
    this.device.queue.writeBuffer(b.finiteFourier, 0, this.encodeActivation(finiteFourier));
    this.device.queue.writeBuffer(b.increment, 0, increment);
    this.device.queue.writeBuffer(b.latent, 0, latent);

    const encoder = this.device.createCommandEncoder({ label: "CK transition" });
    const pass = encoder.beginComputePass({ label: "bounded-memory CK map" });
    this.matmul(pass, b.localFourier, "local.time_embedding", b.localCondition, 1, d, d * 2);
    this.matmul(pass, b.finiteFourier, "finite_time_embedding", b.finiteCondition, 1, d, d * 6);
    this.kernels.dispatch(
      pass,
      "embedding",
      [
        this.sampleBuffers.metadata,
        this.weights.get("local.embedding"),
        b.nodeA,
        this.kernels.uniformU32([n, d, 0, 0]),
      ],
      groups(n * d),
    );
    this.kernels.dispatch(pass, "zeroF32", [b.correction], groups(n * 3));
    let current = b.nodeA;
    let other = b.nodeB;
    let endpoint = this.currentCoords;
    const endpointByBranch = [];
    let endpointIndex = 0;
    for (let block = 0; block < this.config.depth; block += 1) {
      if (block === this.weights.manifest.finite_start_block) {
        this.kernels.dispatch(pass, "copyF16", [current, b.sharedNode], groups(n * d));
      }
      this.runBlock(pass, current, other, b.localCondition, endpoint, `local.blocks.${block}`, block === 0);
      if (block >= this.weights.manifest.finite_start_block) {
        const branch = block - this.weights.manifest.finite_start_block;
        this.kernels.dispatch(pass, "copyF16", [current, b.branchNodes[branch]], groups(n * d));
      }
      if (this.weights.manifest.endpoint_block_indices.includes(block)) {
        this.matmul(pass, current, `local.endpoint_updates.${endpointIndex}`, b.endpointDelta, n, 3, d);
        const gate = (1 - start) / (1 + start);
        this.kernels.dispatch(
          pass,
          "endpointUpdate",
          [
            this.currentCoords,
            b.endpointDelta,
            b.correction,
            this.sampleBuffers.design,
            b.endpoints[endpointIndex],
            this.kernels.uniformF32([gate, 0, 0, 0]),
          ],
          groups(n * 3),
        );
        endpoint = b.endpoints[endpointIndex];
        endpointIndex += 1;
      }
      if (block >= this.weights.manifest.finite_start_block) endpointByBranch.push(endpoint);
    }

    this.kernels.dispatch(pass, "zeroF32", [b.residual], groups(n * 3));
    this.kernels.dispatch(
      pass,
      "noiseInput",
      [b.increment, b.latent, this.sampleBuffers.baseScales, b.noiseInput, scalar],
      groups(n * 6),
    );
    this.matmul(pass, b.noiseInput, "noise_encoder", b.deltaNode, n, d, 6);
    this.kernels.dispatch(pass, "addF16", [b.sharedNode, b.deltaNode, b.nodeA], groups(n * d));
    current = b.nodeA;
    other = b.nodeB;
    let finiteUpdate = 0;
    for (let block = 0; block < this.config.ck_suffix_layers; block += 1) {
      this.kernels.dispatch(
        pass,
        "secant",
        [endpointByBranch[block], b.increment, b.residual, this.sampleBuffers.baseScales, b.secant, scalar],
        groups(n * 3),
      );
      this.matmul(pass, b.branchNodes[block], `finite.lateral_adapters.${block}`, b.deltaNode, n, d, d);
      this.kernels.dispatch(pass, "addF16", [current, b.deltaNode, other], groups(n * d));
      [current, other] = [other, current];
      this.runBlock(pass, current, other, b.finiteCondition, b.secant, `finite.blocks.${block}`, false);
      if (block === 0 || block === 2 || block === 4) {
        this.matmul(pass, current, `finite.endpoint_updates.${finiteUpdate}`, b.endpointDelta, n, 3, d);
        this.kernels.dispatch(
          pass,
          "residualUpdate",
          [b.endpointDelta, b.residual, this.sampleBuffers.design],
          groups(n * 3),
        );
        finiteUpdate += 1;
      }
    }
    this.kernels.dispatch(
      pass,
      "secant",
      [endpoint, b.increment, b.residual, this.sampleBuffers.baseScales, b.secant, scalar],
      groups(n * 3),
    );
    this.kernels.dispatch(
      pass,
      "finalState",
      [
        this.currentCoords,
        b.secant,
        b.increment,
        this.sampleBuffers.baseMeans,
        this.sampleBuffers.design,
        this.nextCoords,
        scalar,
      ],
      groups(n * 3),
    );
    pass.end();
    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - started;
    [this.currentCoords, this.nextCoords] = [this.nextCoords, this.currentCoords];
    return elapsed;
  }

  async coordinates() {
    return readF32(this.device, this.currentCoords, this.sampleBuffers.atoms * 3);
  }

  async secantEndpoint() {
    return readF32(this.device, this.buffers.secant, this.sampleBuffers.atoms * 3);
  }

  memoryBytes() {
    return this.weights.manifest.weight_bytes + this.activationBytes;
  }
}
