import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ROLE_BACKBONE = 1;
const ROLE_SIDECHAIN = 2;
const ROLE_LIGAND = 3;
const ELEMENT_COLORS = new Map([
  [6, 0xc7cdcf], [7, 0x668be3], [8, 0xf06560], [9, 0x69cd7d],
  [15, 0xeea058], [16, 0xe9cf54], [17, 0x61c87b], [34, 0xb485db],
]);
const ELEMENT_RADII = new Map([[6, 0.23], [7, 0.24], [8, 0.25], [9, 0.25], [15, 0.29], [16, 0.29], [17, 0.30]]);

const axis = new THREE.Vector3(0, 1, 0);
const a = new THREE.Vector3();
const b = new THREE.Vector3();
const midpoint = new THREE.Vector3();
const delta = new THREE.Vector3();
const quaternion = new THREE.Quaternion();
const scale = new THREE.Vector3();
const matrix = new THREE.Matrix4();

function atomPosition(coords, atom, target) {
  const offset = atom * 3;
  return target.set(coords[offset], coords[offset + 1], coords[offset + 2]);
}

function atomMatrix(mesh, instance, coords, atom, radius) {
  atomPosition(coords, atom, a);
  matrix.compose(a, quaternion.identity(), scale.setScalar(radius));
  mesh.setMatrixAt(instance, matrix);
}

function bondMatrix(mesh, instance, coords, first, second, radius) {
  atomPosition(coords, first, a);
  atomPosition(coords, second, b);
  delta.copy(b).sub(a);
  const length = delta.length();
  if (length < 1e-6) return false;
  midpoint.copy(a).add(b).multiplyScalar(0.5);
  quaternion.setFromUnitVectors(axis, delta.multiplyScalar(1 / length));
  matrix.compose(midpoint, quaternion, scale.set(radius, length, radius));
  mesh.setMatrixAt(instance, matrix);
  return true;
}

function material(color, opacity = 1) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.45,
    metalness: 0.02,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity > 0.5,
  });
}

export class MolecularViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080a0b);
    this.camera = new THREE.PerspectiveCamera(33, 1, 0.1, 3000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.7));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.replaceChildren(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.rotateSpeed = 0.55;
    this.scene.add(new THREE.HemisphereLight(0xe7f1ef, 0x24272a, 2.1));
    const key = new THREE.DirectionalLight(0xffffff, 3.1);
    key.position.set(20, 34, 28);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x77b9d3, 1.25);
    rim.position.set(-24, -10, -18);
    this.scene.add(rim);
    this.group = null;
    this.sample = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.animateFrame = this.animateFrame.bind(this);
    requestAnimationFrame(this.animateFrame);
  }

  clear() {
    if (!this.group) return;
    this.scene.remove(this.group);
    this.group.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    this.group = null;
  }

  setSample(sample, coords) {
    this.clear();
    this.sample = sample;
    this.group = new THREE.Group();
    this.scene.add(this.group);
    const sphere = new THREE.SphereGeometry(1, 9, 7);
    const cylinder = new THREE.CylinderGeometry(1, 1, 1, 7, 1, false);
    this.backbone = new THREE.InstancedMesh(cylinder, material(0x65706f, 0.64), sample.backbone_trace_pairs.length);
    this.sideAtoms = new THREE.InstancedMesh(sphere, material(0x52bca8, 0.84), sample.roles.filter((role) => role === ROLE_SIDECHAIN).length);
    this.sideBonds = new THREE.InstancedMesh(cylinder, material(0x3e9e8d, 0.76), sample.sidechain_bonds.length);
    this.ligandBonds = new THREE.InstancedMesh(cylinder, material(0xd65358), sample.ligand_bonds.length);
    this.ligandByElement = new Map();
    for (let atom = 0; atom < sample.atoms; atom += 1) {
      if (sample.roles[atom] !== ROLE_LIGAND) continue;
      const element = sample.atomic_numbers[atom];
      if (!this.ligandByElement.has(element)) this.ligandByElement.set(element, []);
      this.ligandByElement.get(element).push(atom);
    }
    this.ligandMeshes = [...this.ligandByElement].map(([element, atoms]) => ({
      element,
      atoms,
      mesh: new THREE.InstancedMesh(sphere, material(ELEMENT_COLORS.get(element) ?? 0xcd83d2), atoms.length),
    }));
    for (const mesh of [this.backbone, this.sideAtoms, this.sideBonds, this.ligandBonds, ...this.ligandMeshes.map((item) => item.mesh)]) {
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      this.group.add(mesh);
    }
    this.sidechainAtoms = sample.roles.flatMap((role, atom) => role === ROLE_SIDECHAIN ? [atom] : []);
    this.update(coords);
    this.resetCamera();
  }

  update(coords) {
    if (!this.sample) return;
    let cursor = 0;
    for (const [first, second] of this.sample.backbone_trace_pairs) {
      if (bondMatrix(this.backbone, cursor, coords, first, second, 0.055)) cursor += 1;
    }
    this.backbone.count = cursor;
    this.backbone.instanceMatrix.needsUpdate = true;
    cursor = 0;
    for (const atom of this.sidechainAtoms) atomMatrix(this.sideAtoms, cursor++, coords, atom, 0.09);
    this.sideAtoms.count = cursor;
    this.sideAtoms.instanceMatrix.needsUpdate = true;
    cursor = 0;
    for (const [first, second] of this.sample.sidechain_bonds) {
      if (bondMatrix(this.sideBonds, cursor, coords, first, second, 0.038)) cursor += 1;
    }
    this.sideBonds.count = cursor;
    this.sideBonds.instanceMatrix.needsUpdate = true;
    cursor = 0;
    for (const [first, second] of this.sample.ligand_bonds) {
      if (bondMatrix(this.ligandBonds, cursor, coords, first, second, 0.068)) cursor += 1;
    }
    this.ligandBonds.count = cursor;
    this.ligandBonds.instanceMatrix.needsUpdate = true;
    for (const { element, atoms, mesh } of this.ligandMeshes) {
      atoms.forEach((atom, index) => atomMatrix(mesh, index, coords, atom, ELEMENT_RADII.get(element) ?? 0.26));
      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  async interpolate(from, to, duration = 360) {
    const mixed = new Float32Array(from.length);
    const started = performance.now();
    await new Promise((resolve) => {
      const frame = (now) => {
        const raw = Math.min(1, (now - started) / duration);
        const amount = raw * raw * (3 - 2 * raw);
        for (let index = 0; index < mixed.length; index += 1) mixed[index] = from[index] + amount * (to[index] - from[index]);
        this.update(mixed);
        if (raw < 1) requestAnimationFrame(frame); else resolve();
      };
      requestAnimationFrame(frame);
    });
  }

  resetCamera() {
    if (!this.sample) return;
    const coords = flattenCoords(this.sample.target_coords);
    const box = new THREE.Box3();
    for (let atom = 0; atom < this.sample.atoms; atom += 1) box.expandByPoint(atomPosition(coords, atom, new THREE.Vector3()));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size * 0.8, size * 0.6, size * 1.05));
    this.camera.near = Math.max(0.05, size / 500);
    this.camera.far = Math.max(1000, size * 10);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  animateFrame() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.animateFrame);
  }
}

function flattenCoords(values) {
  return new Float32Array(Array.isArray(values[0]) ? values.flat() : values);
}
