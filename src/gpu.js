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
    const digest = await crypto.subtle.digest("SHA-256", encoded);
    const sha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (sha256 !== manifest.weight_sha256) throw new Error("Model weight checksum differs.");
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

const MATMUL_REGISTER = /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> tile_a: array<f16, 512>;
var<workgroup> tile_b: array<f16, 512>;
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
  let rows = params.dims.x;
  let columns = params.dims.y;
  let inner = params.dims.z;
  let row = group.y * 32u + local.y * 4u;
  let column = group.x * 32u + local.x * 4u;
  var sums: array<vec4<f32>, 4>;
  for (var block = 0u; block < inner; block += 16u) {
    for (var item = lane; item < 512u; item += 64u) {
      let r = group.y * 32u + item / 16u;
      let c = group.x * 32u + item / 16u;
      let k = block + item % 16u;
      let transposed = (item % 16u) * 32u + item / 16u;
      tile_a[transposed] = select(f16(0), a[r * inner + k], r < rows && k < inner);
      tile_b[transposed] = select(f16(0), weight[c * inner + k], c < columns && k < inner);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k += 1u) {
      let b = vec4<f32>(f32(tile_b[k * 32u + local.x * 4u]),
                        f32(tile_b[k * 32u + local.x * 4u + 1u]),
                        f32(tile_b[k * 32u + local.x * 4u + 2u]),
                        f32(tile_b[k * 32u + local.x * 4u + 3u]));
      for (var r = 0u; r < 4u; r += 1u) {
        sums[r] += f32(tile_a[k * 32u + local.y * 4u + r]) * b;
      }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r += 1u) {
    for (var c = 0u; c < 4u; c += 1u) {
      if (row + r < rows && column + c < columns) {
        var value = sums[r][c];
        if (params.dims.w != 0u) { value += f32(bias[column + c]); }
        output[(row + r) * columns + column + c] = f16(value);
      }
    }
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
var<workgroup> bond_neighbors: array<vec2<i32>, 40>;

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
  if (lane < 40u) {
    let q = group.x * 4u + lane / 10u;
    bond_neighbors[lane] = select(vec2<i32>(-1, 0), neighbors[q * 10u + lane % 10u], q < atoms);
  }
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
            let edge = bond_neighbors[local_query * 10u + slot];
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
        let probability = exp(logits[query_local * 32u + key] - next_max);
        logits[query_local * 32u + key] = probability;
        next_sum += probability;
      }
      running_max[query_local] = next_max;
      running_sum[query_local] = next_sum;
      alpha[query_local] = scale;
    }
    workgroupBarrier();
    accumulator *= alpha[query_local];
    for (var key = 0u; key < 32u; key += 1u) {
      let probability = logits[query_local * 32u + key];
      accumulator += probability * f32(value_tile[key * 64u + dim]);
    }
    workgroupBarrier();
  }
  if (query < atoms) {
    output[(head * atoms + query) * 64u + dim] = f16(accumulator / running_sum[query_local]);
  }
}`;

function attentionRegister(queries, keys) { return /* wgsl */ `
enable f16;
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> features: array<f16>;
@group(0) @binding(1) var<storage, read> entities: array<i32>;
@group(0) @binding(2) var<storage, read> neighbors: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read> pair_bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> qt: array<f16, ${queries * 64}>;
var<workgroup> kt: array<f16, ${keys * 64}>;
var<workgroup> vt: array<f16, ${keys * 64}>;
var<workgroup> probabilities: array<f32, ${queries * keys}>;
var<workgroup> maxima: array<f32, ${queries}>;
var<workgroup> totals: array<f32, ${queries}>;
var<workgroup> alpha: array<f32, ${queries}>;
var<workgroup> bonds: array<vec2<i32>, ${queries * 10}>;
var<workgroup> bond_masks: array<vec4<u32>, ${queries}>;
fn fi(section: u32, head: u32, atom: u32, dim: u32) -> u32 {
 return (((section * params.dims.y + head) * params.dims.x + atom) * 64u + dim);
}
@compute @workgroup_size(${queries * 16})
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 let atoms = params.dims.x;
 let ql = lane / 16u;
 let dim = (lane % 16u) * 4u;
 let query = group.x * ${queries}u + ql;
 let head = group.y;
 for (var item = lane; item < ${queries * 64}u; item += ${queries * 16}u) {
   let q = group.x * ${queries}u + item / 64u;
   qt[item] = select(f16(0), features[fi(0u, head, q, item % 64u)], q < atoms);
 }
 if (lane < ${queries * 10}u) {
   let q = group.x * ${queries}u + lane / 10u;
   bonds[lane] = select(vec2<i32>(-1, 0), neighbors[q * 10u + lane % 10u], q < atoms);
 }
 if (lane < ${queries}u) { maxima[lane] = -3.402823e38; totals[lane] = 0.0; }
 var accumulator = vec4<f32>(0);
 workgroupBarrier();
 for (var ks = 0u; ks < atoms; ks += ${keys}u) {
   if (lane < ${queries}u) {
     var masks = vec4<u32>(0);
     if (params.dims.z != 0u) {
       for (var slot = 0u; slot < 10u; slot++) {
         let edge = bonds[lane * 10u + slot];
         if (edge.x >= i32(ks) && edge.x < i32(ks + ${keys}u) && edge.y >= 0 && edge.y < 4) {
           masks[u32(edge.y)] |= 1u << (u32(edge.x) - ks);
         }
       }
     }
     bond_masks[lane] = masks;
   }
   for (var item = lane; item < ${keys * 64}u; item += ${queries * 16}u) {
     let k = ks + item / 64u;
     kt[(item % 64u) * ${keys}u + item / 64u] = select(f16(0), features[fi(1u, head, k, item % 64u)], k < atoms);
     vt[item] = select(f16(0), features[fi(2u, head, k, item % 64u)], k < atoms);
   }
   workgroupBarrier();
   for (var item = lane; item < ${queries * keys}u; item += ${queries * 16}u) {
     let q = item / ${keys}u;
     let k = item % ${keys}u;
     let gq = group.x * ${queries}u + q;
     let gk = ks + k;
     var score = -3.402823e38;
     if (gq < atoms && gk < atoms) {
       score = 0.0;
       for (var d = 0u; d < 64u; d++) { score += f32(qt[q * 64u + d]) * f32(kt[d * ${keys}u + k]); }
       if (entities[gq] != entities[gk]) { score += f32(pair_bias[head * 5u]); }
       if (params.dims.z != 0u) {
         let bits = (bond_masks[q] >> vec4<u32>(k)) & vec4<u32>(1u);
         let category = bits.x + 2u * bits.y + 3u * bits.z + 4u * bits.w;
         if (category > 0u) { score += f32(pair_bias[head * 5u + category]); }
       }
     }
     probabilities[item] = score;
   }
   workgroupBarrier();
   if (lane < ${queries}u) {
     var m = -3.402823e38;
     for (var k = 0u; k < ${keys}u; k++) { m = max(m, probabilities[lane * ${keys}u + k]); }
     let next = max(m, maxima[lane]);
     let a = exp(maxima[lane] - next);
     var total = totals[lane] * a;
     for (var k = 0u; k < ${keys}u; k++) {
       let p = exp(probabilities[lane * ${keys}u + k] - next);
       probabilities[lane * ${keys}u + k] = p;
       total += p;
     }
     maxima[lane] = next; totals[lane] = total; alpha[lane] = a;
   }
   workgroupBarrier();
   accumulator *= alpha[ql];
   for (var k = 0u; k < ${keys}u; k++) {
     let v = vec4<f32>(f32(vt[k * 64u + dim]), f32(vt[k * 64u + dim + 1u]),
                      f32(vt[k * 64u + dim + 2u]), f32(vt[k * 64u + dim + 3u]));
     accumulator += probabilities[ql * ${keys}u + k] * v;
   }
   workgroupBarrier();
 }
 if (query < atoms) {
   for (var d = 0u; d < 4u; d++) { output[(head * atoms + query) * 64u + dim + d] = f16(accumulator[d] / totals[ql]); }
 }
}`; }

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

const LATERAL_MIX = /* wgsl */ `
struct Params { dims: vec4<u32> };
@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;
@group(0) @binding(5) var<storage, read_write> d: array<f32>;
@group(0) @binding(6) var<storage, read_write> e: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let i = global.x;
  if (i >= params.dims.x) { return; }
  let stride = params.dims.y * params.dims.z;
  let offset = params.dims.w * params.dims.y + i % params.dims.y;
  let x = projected[i];
  if (params.dims.w == 0u) {
    a[i] = x * weights[offset];
    b[i] = x * weights[stride + offset];
    c[i] = x * weights[2u * stride + offset];
    d[i] = x * weights[3u * stride + offset];
    e[i] = x * weights[4u * stride + offset];
  } else {
    a[i] += x * weights[offset];
    b[i] += x * weights[stride + offset];
    c[i] += x * weights[2u * stride + offset];
    d[i] += x * weights[3u * stride + offset];
    e[i] += x * weights[4u * stride + offset];
  }
}`;

export class Kernels {
  static async create(device, precision = "float32") {
    const definitions = {
      matmul: MATMUL,
      matmulRegister: MATMUL_REGISTER,
      adaln: ADALN,
      embedding: EMBEDDING,
      prepareQkv: PREPARE_QKV,
      attention: FLASH_ATTENTION,
      attentionRegister: attentionRegister(16, 32),
      attentionCompact: attentionRegister(8, 16),
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
      lateralMix: LATERAL_MIX,
    };
    const pipelines = {};
    const unavailablePipelines = {};
    await Promise.all(Object.entries(definitions).map(async ([name, source]) => {
      if (name === "attentionRegister" && precision !== "float16"
        && device.limits.maxComputeWorkgroupStorageSize < 24256) return;
      try {
        // Flatten large elementwise dispatches across two workgroup dimensions.
        if (source.includes('@builtin(global_invocation_id) global')) {
          source = source.replace('@builtin(global_invocation_id) global: vec3<u32>',
            '@builtin(global_invocation_id) global: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>')
            .replaceAll('global.x', '(global.x + global.y * grid.x * 256u)');
        }
        const code = precision === "float16"
          ? source
          : source.replaceAll("enable f16;", "").replaceAll("f16", "f32");
        const compact = name === "attention" && precision !== "float16"
          && device.limits.maxComputeWorkgroupStorageSize < 18288;
        const shader = compact ? code.replaceAll("32u", "16u")
          .replaceAll("2048", "1024").replaceAll("128", "64") : code;
        const module = device.createShaderModule({ code: shader, label: name });
        const compilation = await module.getCompilationInfo();
        const errors = compilation.messages.filter((message) => message.type === "error");
        if (errors.length) throw new Error(`${name} WGSL: ${errors.map((error) => error.message).join("; ")}`);
        pipelines[name] = await device.createComputePipelineAsync({
          layout: "auto",
          compute: { module, entryPoint: "main" },
          label: name,
        });
      } catch (error) {
        if (!["matmulRegister", "attentionRegister", "attentionCompact"].includes(name)) throw error;
        unavailablePipelines[name] = error.message;
      }
    }));
    const kernels = new Kernels(device, pipelines);
    kernels.unavailablePipelines = unavailablePipelines;
    if (precision !== "float16") {
      await kernels.tuneMatmul();
      await kernels.tuneAttention();
    }
    return kernels;
  }

  constructor(device, pipelines) {
    this.device = device;
    this.pipelines = pipelines;
    this.uniforms = new Map();
    this.bindGroups = new Map();
    this.attentionVariant = pipelines.attentionRegister ? "attentionRegister"
      : pipelines.attentionCompact ? "attentionCompact" : null;
  }

  async tuneMatmul() {
    const device = this.device;
    const rows = 1024;
    this.registerShapes = new Set();
    this.tuning = [];
    if (!this.pipelines.matmulRegister) return;
    for (const [columns, inner] of [[2016, 408], [4080, 408], [408, 2040], [408, 888], [1024, 408], [408, 1024]]) {
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
      const values = count => Float32Array.from({ length: count }, (_, i) => Math.sin(i * 0.37) * 0.05);
      const a = uploadBuffer(device, values(rows * inner), usage, "tune input");
      const weight = uploadBuffer(device, values(columns * inner), usage, "tune weight");
      const outputs = [0, 1].map(i => makeBuffer(device, rows * columns * 4, usage, `tune output ${i}`));
      const uniform = this.uniformU32([rows, columns, inner, 0]);
      const times = [[], []];
      for (let repeat = 0; repeat < 5; repeat++) {
        for (const index of (repeat % 2 ? [1, 0] : [0, 1])) {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          const tile = index ? 32 : 16;
          for (let batch = 0; batch < 8; batch++) {
            this.dispatch(pass, index ? "matmulRegister" : "matmul",
              [a, weight, weight, outputs[index], uniform], [Math.ceil(columns / tile), Math.ceil(rows / tile)]);
          }
          pass.end();
          const start = performance.now();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          if (repeat >= 2) times[index].push((performance.now() - start) / 8);
        }
      }
      const reference = await readF32(device, outputs[0], rows * columns);
      const actual = await readF32(device, outputs[1], rows * columns);
      let maxError = 0;
      for (let i = 0; i < actual.length; i++) maxError = Math.max(maxError, Math.abs(actual[i] - reference[i]));
      const medians = times.map(t => t.sort((a, b) => a - b)[1]);
      const key = `${columns}:${inner}`;
      if (maxError < 1e-4 && medians[1] < medians[0]) this.registerShapes.add(key);
      this.tuning.push({ shape: key, milliseconds: medians, maxError, register: this.registerShapes.has(key) });
      for (const buffer of [a, weight, ...outputs]) buffer.destroy();
      this.bindGroups.clear();
    }
  }

  async tuneAttention() {
    const device = this.device;
    const n = 1024, heads = 12;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const features = uploadBuffer(device, Float32Array.from({ length: 3 * heads * n * 64 },
      (_, i) => Math.sin(i * 0.013) * 0.2), usage, "tune attention features");
    const entities = uploadBuffer(device, Int32Array.from({ length: n }, (_, i) => i % 3), usage, "tune entities");
    const edges = new Int32Array(n * 20);
    for (let q = 0; q < n; q++) for (let s = 0; s < 10; s++) {
      edges[q * 20 + 2 * s] = (q + s + 1) % n;
      edges[q * 20 + 2 * s + 1] = s % 4;
    }
    const neighbors = uploadBuffer(device, edges, usage, "tune neighbors");
    const bias = uploadBuffer(device, Float32Array.from({ length: heads * 5 }, (_, i) => i % 5), usage, "tune biases");
    const names = ["attention", "attentionCompact", "attentionRegister"].filter(name => this.pipelines[name]);
    const outputs = names.map(() => makeBuffer(device, heads * n * 64 * 4, usage, "tune attended"));
    this.attentionVariant = null;
    const times = names.map(() => []);
    const uniform = this.uniformU32([n, heads, 1, 0]);
    for (let repeat = 0; repeat < 5; repeat++) {
      for (let offset = 0; offset < names.length; offset++) {
        const index = (offset + repeat) % names.length;
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        const queries = names[index] === "attention" ? 4 : names[index] === "attentionCompact" ? 8 : 16;
        for (let batch = 0; batch < 4; batch++) this.dispatch(pass, names[index],
          [features, entities, neighbors, bias, outputs[index], uniform], [Math.ceil(n / queries), heads]);
        pass.end();
        const start = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        if (repeat >= 2) times[index].push((performance.now() - start) / 4);
      }
    }
    const reference = await readF32(device, outputs[0], heads * n * 64);
    this.attentionTuning = [];
    for (let index = 0; index < names.length; index++) {
      const actual = await readF32(device, outputs[index], reference.length);
      let error = 0;
      for (let i = 0; i < actual.length; i++) error = Math.max(error, Math.abs(actual[i] - reference[i]));
      this.attentionTuning.push({ name: names[index], milliseconds: times[index].sort((a, b) => a - b)[1], error });
    }
    const winner = this.attentionTuning.filter(row => row.error < 1e-4)
      .sort((a, b) => a.milliseconds - b.milliseconds)[0];
    this.attentionVariant = winner.name === "attention" ? null : winner.name;
    for (const buffer of [features, entities, neighbors, bias, ...outputs]) buffer.destroy();
    this.bindGroups.clear();
  }

  uniformU32(values) {
    const key = `u:${values.join(",")}`;
    if (!this.uniforms.has(key)) {
      this.uniforms.set(key, uploadBuffer(this.device, new Uint32Array(values), GPUBufferUsage.UNIFORM, key));
      this.uniforms.get(key).shapeKey = `${values[1]}:${values[2]}`;
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
    if (name === "attention" && this.attentionVariant) {
      name = this.attentionVariant;
      groups = [Math.ceil(groups[0] / (name === "attentionRegister" ? 4 : 2)), groups[1]];
    }
    if (name === "matmul" && this.registerShapes?.has(buffers[4].shapeKey) && groups[1] > 1) {
      name = "matmulRegister";
      groups = [Math.ceil(groups[0] / 2), Math.ceil(groups[1] / 2)];
    }
    const pipeline = this.pipelines[name];
    let cache = this.bindGroups;
    for (const key of [name, ...buffers]) {
      if (!cache.has(key)) cache.set(key, new Map());
      cache = cache.get(key);
    }
    let group = cache.get("group");
    if (!group) {
      const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
      group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      cache.set("group", group);
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    if (groups.length === 1 && groups[0] > this.device.limits.maxComputeWorkgroupsPerDimension) {
      const width = this.device.limits.maxComputeWorkgroupsPerDimension;
      pass.dispatchWorkgroups(width, Math.ceil(groups[0] / width));
    } else pass.dispatchWorkgroups(...groups);
  }
}

const readbackPools = new WeakMap();
export async function readF32(device, source, elements) {
  if (!readbackPools.has(device)) readbackPools.set(device, []);
  const pool = readbackPools.get(device);
  const index = pool.findIndex(buffer => buffer.size >= elements * 4);
  const staging = index >= 0 ? pool.splice(index, 1)[0] : makeBuffer(
    device, elements * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "readback");
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, elements * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0, elements * 4));
  staging.unmap();
  if (pool.length < 2) pool.push(staging); else staging.destroy();
  return result;
}
