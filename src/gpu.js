const align4 = (value) => (value + 3) & ~3;

export function floatToHalf(value) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  if (mantissa & 0x1000) {
    mantissa += 0x2000;
    if (mantissa & 0x800000) {
      mantissa = 0;
      exponent += 1;
      if (exponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

export function halfArray(values) {
  const result = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    result[index] = floatToHalf(values[index]);
  }
  return result;
}

export function makeBuffer(device, bytes, usage, label) {
  return device.createBuffer({ size: align4(bytes), usage, label });
}

export function uploadBuffer(device, typed, usage, label) {
  const buffer = device.createBuffer({
    size: align4(typed.byteLength),
    usage,
    mappedAtCreation: true,
    label,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength),
  );
  buffer.unmap();
  return buffer;
}

export class WeightStore {
  static async load(device, manifestUrl, suppliedManifest = null) {
    const manifest = suppliedManifest ?? await fetch(manifestUrl).then((response) => {
      if (!response.ok) throw new Error(`Model manifest failed: ${response.status}`);
      return response.json();
    });
    const weightUrl = new URL(manifest.weight_file, new URL(manifestUrl, location.href));
    const encoded = await fetch(weightUrl).then((response) => {
      if (!response.ok) throw new Error(`Model weights failed: ${response.status}`);
      return response.arrayBuffer();
    });
    if (encoded.byteLength !== manifest.weight_bytes) {
      throw new Error(`Weight byte count differs: ${encoded.byteLength} != ${manifest.weight_bytes}`);
    }
    const buffers = new Map();
    for (const [name, entry] of Object.entries(manifest.tensors)) {
      const bytes = new Uint8Array(encoded, entry.offset, entry.bytes);
      buffers.set(
        name,
        uploadBuffer(device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, name),
      );
    }
    return new WeightStore(manifest, buffers);
  }

  constructor(manifest, buffers) {
    this.manifest = manifest;
    this.buffers = buffers;
  }

  get(name) {
    const result = this.buffers.get(name);
    if (!result) throw new Error(`Unknown model tensor: ${name}`);
    return result;
  }
}

const MATMUL = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> tile_a: array<f16, 256>;
var<workgroup> tile_b: array<f16, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let rows = params.dims.x;
  let columns = params.dims.y;
  let inner = params.dims.z;
  let use_bias = params.dims.w;
  let row = group.y * 16u + local.y;
  let column = group.x * 16u + local.x;
  var sum = 0.0;
  let tiles = (inner + 15u) / 16u;
  for (var block = 0u; block < tiles; block += 1u) {
    let ak = block * 16u + local.x;
    let bk = block * 16u + local.y;
    tile_a[local.y * 16u + local.x] = select(f16(0.0), a[row * inner + ak], row < rows && ak < inner);
    tile_b[local.y * 16u + local.x] = select(f16(0.0), weight[column * inner + bk], column < columns && bk < inner);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k += 1u) {
      sum += f32(tile_a[local.y * 16u + k]) * f32(tile_b[k * 16u + local.x]);
    }
    workgroupBarrier();
  }
  if (row < rows && column < columns) {
    if (use_bias != 0u) { sum += f32(bias[column]); }
    output[row * columns + column] = f16(sum);
  }
}`;

const ADALN = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> affine: array<f16>;
@group(0) @binding(2) var<storage, read> norm: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> sums: array<f32, 256>;
var<workgroup> squares: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = group.x;
  let width = params.dims.y;
  var local_sum = 0.0;
  var local_square = 0.0;
  for (var column = lane; column < width; column += 256u) {
    let value = f32(input[row * width + column]);
    local_sum += value;
    local_square += value * value;
  }
  sums[lane] = local_sum;
  squares[lane] = local_square;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      squares[lane] += squares[lane + stride];
    }
    workgroupBarrier();
  }
  let mean = sums[0] / f32(width);
  let variance = max(squares[0] / f32(width) - mean * mean, 0.0);
  let inverse = inverseSqrt(variance + 1e-5);
  for (var column = lane; column < width; column += 256u) {
    let centered = (f32(input[row * width + column]) - mean) * inverse;
    let normalized = centered * f32(norm[column]) + f32(norm[width + column]);
    let scale = f32(affine[column]);
    let shift = f32(affine[width + column]);
    output[row * width + column] = f16(normalized * (1.0 + scale) + shift);
  }
}`;

const EMBEDDING = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> metadata: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> embedding: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  let atoms = params.dims.x;
  let width = params.dims.y;
  if (index >= atoms * width) { return; }
  let atom = index / width;
  let column = index % width;
  let item = metadata[atom];
  let atomic = item.x;
  let role = 128u + item.y;
  let residue = 133u + item.z;
  let atom_name = 154u + item.w;
  output[index] = embedding[atomic * width + column]
    + embedding[role * width + column]
    + embedding[residue * width + column]
    + embedding[atom_name * width + column];
}`;

const PREPARE_QKV = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> projection: array<f16>;
@group(0) @binding(1) var<storage, read> coords: array<f32>;
@group(0) @binding(2) var<storage, read> qk_norm: array<f16>;
@group(0) @binding(3) var<storage, read> head_weights: array<f16>;
@group(0) @binding(4) var<storage, read_write> features: array<f16>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> q_squares: array<f32, 64>;
var<workgroup> k_squares: array<f32, 64>;

fn feature_index(section: u32, head: u32, atom: u32, dim: u32, atoms: u32, heads: u32) -> u32 {
  return (((section * heads + head) * atoms + atom) * 64u + dim);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let atoms = params.dims.x;
  let heads = params.dims.y;
  let atom = group.x;
  let head = group.y;
  let row = atom * 2016u;
  let scalar_start = head * 34u;
  if (lane < 34u) {
    let qv = f32(projection[row + scalar_start + lane]);
    let kv = f32(projection[row + 408u + scalar_start + lane]);
    q_squares[lane] = qv * qv;
    k_squares[lane] = kv * kv;
  } else {
    q_squares[lane] = 0.0;
    k_squares[lane] = 0.0;
  }
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      q_squares[lane] += q_squares[lane + stride];
      k_squares[lane] += k_squares[lane + stride];
    }
    workgroupBarrier();
  }
  let q_inverse = inverseSqrt(q_squares[0] / 34.0 + 1e-5);
  let k_inverse = inverseSqrt(k_squares[0] / 34.0 + 1e-5);
  let scalar_balance = inverseSqrt(3.0);
  let point_balance = sqrt(2.0 / 54.0);
  let raw_head = f32(head_weights[head]);
  let coefficient = log(1.0 + exp(raw_head)) * scalar_balance * point_balance;
  if (lane < 64u) {
    var q_value = 0.0;
    var k_value = 0.0;
    var v_value = 0.0;
    if (lane < 34u) {
      let pair = (lane / 2u) * 2u;
      let second = lane % 2u;
      let frequency = pow(1000.0, -2.0 * f32(pair / 2u) / 34.0);
      let angle = f32(atom) * frequency;
      let cosine = cos(angle);
      let sine = sin(angle);
      let q0 = f32(projection[row + scalar_start + pair]) * q_inverse * f32(qk_norm[pair]);
      let q1 = f32(projection[row + scalar_start + pair + 1u]) * q_inverse * f32(qk_norm[pair + 1u]);
      let k0 = f32(projection[row + 408u + scalar_start + pair]) * k_inverse * f32(qk_norm[34u + pair]);
      let k1 = f32(projection[row + 408u + scalar_start + pair + 1u]) * k_inverse * f32(qk_norm[34u + pair + 1u]);
      q_value = select(q0 * cosine - q1 * sine, q0 * sine + q1 * cosine, second == 1u);
      k_value = scalar_balance / sqrt(34.0) * select(k0 * cosine - k1 * sine, k0 * sine + k1 * cosine, second == 1u);
      v_value = f32(projection[row + 816u + scalar_start + lane]);
    } else if (lane < 52u) {
      let point_dim = lane - 34u;
      let axis = point_dim % 3u;
      let point_offset = head * 18u + point_dim;
      q_value = f32(projection[row + 1224u + point_offset]) + coords[atom * 3u + axis];
      k_value = coefficient * (f32(projection[row + 1440u + point_offset]) + coords[atom * 3u + axis]);
      v_value = f32(projection[row + 1656u + head * 30u + point_dim]) + coords[atom * 3u + axis];
    } else if (lane < 58u) {
      let point = lane - 52u;
      var q2 = 0.0;
      for (var axis = 0u; axis < 3u; axis += 1u) {
        let p = f32(projection[row + 1224u + head * 18u + point * 3u + axis]) + coords[atom * 3u + axis];
        q2 += p * p;
      }
      q_value = q2;
      k_value = -0.5 * coefficient;
      let point_dim = lane - 34u;
      let axis = point_dim % 3u;
      v_value = f32(projection[row + 1656u + head * 30u + point_dim]) + coords[atom * 3u + axis];
    } else {
      let point = lane - 58u;
      var k2 = 0.0;
      for (var axis = 0u; axis < 3u; axis += 1u) {
        let p = f32(projection[row + 1440u + head * 18u + point * 3u + axis]) + coords[atom * 3u + axis];
        k2 += p * p;
      }
      q_value = 1.0;
      k_value = -0.5 * coefficient * k2;
      let point_dim = lane - 34u;
      let axis = point_dim % 3u;
      v_value = f32(projection[row + 1656u + head * 30u + point_dim]) + coords[atom * 3u + axis];
    }
    features[feature_index(0u, head, atom, lane, atoms, heads)] = f16(q_value);
    features[feature_index(1u, head, atom, lane, atoms, heads)] = f16(k_value);
    features[feature_index(2u, head, atom, lane, atoms, heads)] = f16(v_value);
  }
}`;

const FLASH_ATTENTION = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> features: array<f16>;
@group(0) @binding(1) var<storage, read> entities: array<i32>;
@group(0) @binding(2) var<storage, read> neighbors: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read> pair_bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> query_tile: array<f16, 256>;
var<workgroup> key_tile: array<f16, 2048>;
var<workgroup> value_tile: array<f16, 2048>;
var<workgroup> logits: array<f32, 128>;
var<workgroup> running_max: array<f32, 4>;
var<workgroup> running_sum: array<f32, 4>;
var<workgroup> alpha: array<f32, 4>;

fn feature_index(section: u32, head: u32, atom: u32, dim: u32, atoms: u32, heads: u32) -> u32 {
  return (((section * heads + head) * atoms + atom) * 64u + dim);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let atoms = params.dims.x;
  let heads = params.dims.y;
  let use_bonds = params.dims.z;
  let head = group.y;
  let query_local = lane / 64u;
  let dim = lane % 64u;
  let query = group.x * 4u + query_local;
  query_tile[lane] = select(f16(0.0), features[feature_index(0u, head, query, dim, atoms, heads)], query < atoms);
  if (dim == 0u) {
    running_max[query_local] = -3.402823e38;
    running_sum[query_local] = 0.0;
  }
  var accumulator = 0.0;
  workgroupBarrier();
  for (var key_start = 0u; key_start < atoms; key_start += 32u) {
    for (var item = lane; item < 2048u; item += 256u) {
      let key_local = item / 64u;
      let feature_dim = item % 64u;
      let key = key_start + key_local;
      key_tile[item] = select(f16(0.0), features[feature_index(1u, head, key, feature_dim, atoms, heads)], key < atoms);
      value_tile[item] = select(f16(0.0), features[feature_index(2u, head, key, feature_dim, atoms, heads)], key < atoms);
    }
    workgroupBarrier();
    if (lane < 128u) {
      let local_query = lane / 32u;
      let local_key = lane % 32u;
      let global_query = group.x * 4u + local_query;
      let global_key = key_start + local_key;
      var score = -3.402823e38;
      if (global_query < atoms && global_key < atoms) {
        score = 0.0;
        for (var feature_dim = 0u; feature_dim < 64u; feature_dim += 1u) {
          score += f32(query_tile[local_query * 64u + feature_dim]) * f32(key_tile[local_key * 64u + feature_dim]);
        }
        if (entities[global_query] != entities[global_key]) {
          score += f32(pair_bias[head * 5u]);
        }
        if (use_bonds != 0u) {
          var category = 0i;
          for (var slot = 0u; slot < 10u; slot += 1u) {
            let edge = neighbors[global_query * 10u + slot];
            if (edge.x == i32(global_key)) { category = edge.y + 1i; }
          }
          if (category > 0i) { score += f32(pair_bias[head * 5u + u32(category)]); }
        }
      }
      logits[lane] = score;
    }
    workgroupBarrier();
    if (dim == 0u) {
      var tile_max = -3.402823e38;
      for (var key = 0u; key < 32u; key += 1u) {
        tile_max = max(tile_max, logits[query_local * 32u + key]);
      }
      let next_max = max(running_max[query_local], tile_max);
      let scale = exp(running_max[query_local] - next_max);
      var next_sum = running_sum[query_local] * scale;
      for (var key = 0u; key < 32u; key += 1u) {
        next_sum += exp(logits[query_local * 32u + key] - next_max);
      }
      running_max[query_local] = next_max;
      running_sum[query_local] = next_sum;
      alpha[query_local] = scale;
    }
    workgroupBarrier();
    accumulator *= alpha[query_local];
    for (var key = 0u; key < 32u; key += 1u) {
      let probability = exp(logits[query_local * 32u + key] - running_max[query_local]);
      accumulator += probability * f32(value_tile[key * 64u + dim]);
    }
    workgroupBarrier();
  }
  if (query < atoms) {
    output[(head * atoms + query) * 64u + dim] = f16(accumulator / running_sum[query_local]);
  }
}`;

const MERGE_ATTENTION = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> attended: array<f16>;
@group(0) @binding(1) var<storage, read> coords: array<f32>;
@group(0) @binding(2) var<storage, read_write> merged: array<f16>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let atoms = params.dims.x;
  let heads = params.dims.y;
  let index = global.x;
  if (index >= atoms * heads * 74u) { return; }
  let atom = index / (heads * 74u);
  let within = index % (heads * 74u);
  let head = within / 74u;
  let dim = within % 74u;
  var value = 0.0;
  if (dim < 34u) {
    value = f32(attended[(head * atoms + atom) * 64u + dim]);
  } else if (dim < 64u) {
    let point_dim = dim - 34u;
    value = f32(attended[(head * atoms + atom) * 64u + 34u + point_dim]) - coords[atom * 3u + point_dim % 3u];
  } else {
    let point = dim - 64u;
    var squared = 1e-8;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      let relative = f32(attended[(head * atoms + atom) * 64u + 34u + point * 3u + axis]) - coords[atom * 3u + axis];
      squared += relative * relative;
    }
    value = sqrt(squared);
  }
  merged[index] = f16(value);
}`;

const ADD_F16 = /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> left: array<f16>;
@group(0) @binding(1) var<storage, read> right: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index < arrayLength(&output)) { output[index] = left[index] + right[index]; }
}`;

const COPY_F16 = /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index < arrayLength(&output)) { output[index] = input[index]; }
}`;

const COPY_F32 = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index < arrayLength(&output)) { output[index] = input[index]; }
}`;

const SWIGLU = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  let rows = params.dims.x;
  let hidden = params.dims.y;
  if (index >= rows * hidden) { return; }
  let row = index / hidden;
  let column = index % hidden;
  let up = f32(input[row * hidden * 2u + column]);
  let gate = f32(input[row * hidden * 2u + hidden + column]);
  output[index] = f16(up * gate / (1.0 + exp(-gate)));
}`;

const ZERO_F32 = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  if (global.x < arrayLength(&output)) { output[global.x] = 0.0; }
}`;

const ENDPOINT_UPDATE = /* wgsl */ `
enable f16;
struct Scalars { values: vec4<f32> };
@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f16>;
@group(0) @binding(2) var<storage, read_write> correction: array<f32>;
@group(0) @binding(3) var<storage, read> design: array<u32>;
@group(0) @binding(4) var<storage, read_write> endpoint: array<f32>;
@group(0) @binding(5) var<uniform> scalars: Scalars;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index >= arrayLength(&correction)) { return; }
  let atom = index / 3u;
  if (design[atom] != 0u) { correction[index] += f32(delta[index]); }
  endpoint[index] = base[index] + scalars.values.x * correction[index];
}`;

const RESIDUAL_UPDATE = /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> delta: array<f16>;
@group(0) @binding(1) var<storage, read_write> residual: array<f32>;
@group(0) @binding(2) var<storage, read> design: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index < arrayLength(&residual) && design[index / 3u] != 0u) {
    residual[index] += f32(delta[index]);
  }
}`;

const NOISE_INPUT = /* wgsl */ `
enable f16;
struct Scalars { values: vec4<f32> };
@group(0) @binding(0) var<storage, read> increment: array<f32>;
@group(0) @binding(1) var<storage, read> latent: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> scalars: Scalars;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index >= arrayLength(&output)) { return; }
  let atom = index / 6u;
  let dim = index % 6u;
  let start = scalars.values.x;
  let end = scalars.values.y;
  let q = scales[atom] * scales[atom] * (end - start) * (2.0 - start - end);
  if (dim < 3u) {
    output[index] = f16(select(0.0, increment[atom * 3u + dim] / sqrt(q), q > 0.0));
  } else {
    output[index] = f16(latent[atom * 3u + dim - 3u]);
  }
}`;

const SECANT = /* wgsl */ `
struct Scalars { values: vec4<f32> };
@group(0) @binding(0) var<storage, read> endpoint: array<f32>;
@group(0) @binding(1) var<storage, read> increment: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> scalars: Scalars;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index >= arrayLength(&output)) { return; }
  let atom = index / 3u;
  let start = scalars.values.x;
  let end = scalars.values.y;
  let delta = end - start;
  let remaining = 1.0 - start;
  let tau = select(0.0, delta / remaining, remaining > 0.0);
  let alpha = (1.0 + start) * tau - start * tau * tau;
  let q = scales[atom] * scales[atom] * delta * (2.0 - start - end);
  let gamma = delta * sqrt(max(q, 0.0)) + delta * delta;
  let increment_scale = select(0.0, tau / alpha, alpha != 0.0);
  let residual_scale = select(0.0, gamma / alpha, alpha != 0.0);
  output[index] = endpoint[index] + increment_scale * increment[index] + residual_scale * residual[index];
}`;

const FINAL_STATE = /* wgsl */ `
struct Scalars { values: vec4<f32> };
@group(0) @binding(0) var<storage, read> coords: array<f32>;
@group(0) @binding(1) var<storage, read> secant: array<f32>;
@group(0) @binding(2) var<storage, read> increment: array<f32>;
@group(0) @binding(3) var<storage, read> base_mean: array<f32>;
@group(0) @binding(4) var<storage, read> design: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> scalars: Scalars;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let index = global.x;
  if (index >= arrayLength(&output)) { return; }
  if (design[index / 3u] == 0u) { output[index] = coords[index]; return; }
  let start = scalars.values.x;
  let end = scalars.values.y;
  let remaining = 1.0 - start;
  let tau = select(0.0, (end - start) / remaining, remaining > 0.0);
  let alpha = (1.0 + start) * tau - start * tau * tau;
  let beta = -remaining * tau * (1.0 - tau);
  output[index] = coords[index]
    + alpha * (secant[index] - coords[index])
    + beta * (coords[index] - base_mean[index])
    + (1.0 - tau) * increment[index];
}`;

export class Kernels {
  static async create(device, precision = "float32") {
    const definitions = {
      matmul: MATMUL,
      adaln: ADALN,
      embedding: EMBEDDING,
      prepareQkv: PREPARE_QKV,
      attention: FLASH_ATTENTION,
      merge: MERGE_ATTENTION,
      addF16: ADD_F16,
      copyF16: COPY_F16,
      copyF32: COPY_F32,
      swiglu: SWIGLU,
      zeroF32: ZERO_F32,
      endpointUpdate: ENDPOINT_UPDATE,
      residualUpdate: RESIDUAL_UPDATE,
      noiseInput: NOISE_INPUT,
      secant: SECANT,
      finalState: FINAL_STATE,
    };
    const pipelines = {};
    await Promise.all(Object.entries(definitions).map(async ([name, source]) => {
      const code = precision === "float16"
        ? source
        : source.replaceAll("enable f16;", "").replaceAll("f16", "f32");
      const module = device.createShaderModule({ code, label: name });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter((message) => message.type === "error");
      if (errors.length) throw new Error(`${name} WGSL: ${errors.map((error) => error.message).join("; ")}`);
      pipelines[name] = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
        label: name,
      });
    }));
    return new Kernels(device, pipelines);
  }

  constructor(device, pipelines) {
    this.device = device;
    this.pipelines = pipelines;
    this.uniforms = new Map();
  }

  uniformU32(values) {
    const key = `u:${values.join(",")}`;
    if (!this.uniforms.has(key)) {
      this.uniforms.set(key, uploadBuffer(this.device, new Uint32Array(values), GPUBufferUsage.UNIFORM, key));
    }
    return this.uniforms.get(key);
  }

  uniformF32(values) {
    const key = `f:${values.join(",")}`;
    if (!this.uniforms.has(key)) {
      this.uniforms.set(key, uploadBuffer(this.device, new Float32Array(values), GPUBufferUsage.UNIFORM, key));
    }
    return this.uniforms.get(key);
  }

  dispatch(pass, name, buffers, groups) {
    const pipeline = this.pipelines[name];
    const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...groups);
  }
}

export async function readF32(device, source, elements) {
  const staging = makeBuffer(
    device,
    elements * 4,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    "readback",
  );
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, elements * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
