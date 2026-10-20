/* ==========================================================================
 * AIC · UAV interactive reconstruction
 * Three.js point-cloud viewer + reconstruction wizard
 * ========================================================================== */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString('en-US'));
const fmtMB = (n) => (n / 1048576).toFixed(1);
const maxFrames = () => (S.config && S.config.limits && S.config.limits.max_frames) || 1000;

const S = {
  config: null,
  clips: [],
  clipId: null,
  meta: null,
  N: 0, K: 0,
  buffer: null,
  mode: 'rgb',
  progress: 0,
  playing: false,
  speed: 1.2,
  follow: false,
  loop: true,
  highlight: false,
  rgbPip: true,
  pipReady: false,
  pipWp: -1,
  pipReq: 0,
  pipCache: new Map(),
  pointSize: 2.4,
  confMin: 0,
  probMin: 0,
  classVis: new Array(12).fill(true),
  sceneInfo: null,
  prefix: [],
  routeVecs: [],
  routeTans: [],
  markers: [],
  drone: null,
  frustum: null,
  travelLine: null,
  fullLine: null,
  grid: null,
  geo: null,
  pointMesh: null,
  material: null,
  followPos: new THREE.Vector3(),
  followTarget: new THREE.Vector3(),
  followReady: false,
  _pos: new THREE.Vector3(),
  _tan: new THREE.Vector3(),
  lastWp: -1,
  lastFrameT: 0,
  fpsAcc: 0, fpsN: 0,
  uiAcc: 0,
  hashTimer: 0,
  loadingToken: 0,
  jobId: null,
  jobTimer: null,
  jobDone: false,
  record: null,
  recStart: 0,
  recFollow: false,
  videoUploadId: null,
  videoInfo: null,
  videoTotal: 0,
  videoCands: [],
  videoSelIdx: new Set(),
  imagesUploadId: null,
  imagesItems: [],
  framePath: '',
  frameItems: [],
  frameDirs: [],
  frameSel: new Map(),
  tab: 'video',
};

/* ------------------------------------------------------------------ utils */
function toast(msg, kind = '', ms = 3800) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(6px)'; el.style.transition = '.25s'; }, ms);
  setTimeout(() => el.remove(), ms + 320);
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { /* ignore */ }
  if (!res.ok) {
    const msg = (data && (data.error || data.detail)) || text || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

function postJSON(url, body) {
  return fetchJSON(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
}

function downloadBlob(blob, name) {
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
}

function baseName(p) {
  return (p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p;
}

function escHtml(v) {
  return String(v == null ? '' : v).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function selectEvenly(items, n) {
  const out = [];
  if (!items.length) return out;
  n = clamp(Number(n) || 1, 1, items.length);
  if (n === 1) return [items[0]];
  const seen = new Set();
  for (let i = 0; i < n; i++) {
    const idx = Math.round(i * (items.length - 1) / (n - 1));
    if (!seen.has(idx)) { seen.add(idx); out.push(items[idx]); }
  }
  return out;
}

function evenIndices(total, n) {
  total = Math.max(1, Math.floor(Number(total) || 0));
  n = Math.max(1, Math.min(Math.floor(Number(n) || 1), total));
  if (n === 1) return [0];
  const out = [];
  const seen = new Set();
  for (let i = 0; i < n; i++) {
    const idx = Math.round(i * (total - 1) / (n - 1));
    if (!seen.has(idx)) { seen.add(idx); out.push(idx); }
  }
  return out;
}

// The model's global attention is O((S*P)^2), so a single forward pass has a
// fixed token budget.  Selection can still be up to 1000 frames; recon.py
// uniformly downsamples to this cap before inference.
function inferCapForSize() {
  const raw = ($('#optSize') && $('#optSize').value) || '392x700';
  const [h, w] = raw.split('x').map(Number);
  const H = h || 392, W = w || 700;
  const tokens = Math.ceil(H / 14) * Math.ceil(W / 14);
  const lim = (S.config && S.config.limits) || {};
  const budget = Number(lim.max_infer_tokens) || 47000;
  const hard = Number(lim.max_infer_frames) || 32;
  return Math.max(8, Math.min(hard, Math.floor(budget / Math.max(1, tokens))));
}

/* ================================================================== three */
const canvas = $('#view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x070b11, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.autoClear = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b11);
const camera = new THREE.PerspectiveCamera(50, 1, 0.002, 200);
camera.position.set(1.1, 1.2, 1.6);

const controls = new OrbitControls(camera, canvas);
controls.addEventListener('start', () => { controls.autoRotate = false; });
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.minDistance = 0.02;
controls.maxDistance = 50;
controls.autoRotateSpeed = 0.7;

const gRoute = new THREE.Group();
const gAux = new THREE.Group();
scene.add(gRoute, gAux);

const grid = new THREE.GridHelper(3, 30, 0x1d3b55, 0x122232);
grid.material.transparent = true;
grid.material.opacity = 0.35;
grid.position.y = -0.5;
scene.add(grid);

/* ---- in-canvas RGB frame picture-in-picture (also captured by recording) ---- */
const overlayScene = new THREE.Scene();
const overlayCamera = new THREE.OrthographicCamera(0, 1, 1, 0, 0.1, 100);
overlayCamera.position.z = 10;
const pipCanvas = document.createElement('canvas');
pipCanvas.width = 640;
pipCanvas.height = 420;
const pipCtx = pipCanvas.getContext('2d');
const pipTexture = new THREE.CanvasTexture(pipCanvas);
// The 2D canvas already contains display-referred sRGB bytes (same as the
// <img> in the side panel).  Bypass texture colour management so the overlay
// matches the DOM image and the point-cloud shader instead of being darkened.
pipTexture.colorSpace = THREE.NoColorSpace;
pipTexture.minFilter = THREE.LinearFilter;
pipTexture.magFilter = THREE.LinearFilter;
const pipMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ map: pipTexture, transparent: true, depthTest: false,
                                depthWrite: false, toneMapped: false }),
);
pipMesh.visible = false;
pipMesh.frustumCulled = false;
pipMesh.renderOrder = 999;
overlayScene.add(pipMesh);

const uniforms = {
  uMode: { value: 0 },
  uProgress: { value: 0 },
  uPointSize: { value: S.pointSize },
  uPixelRatio: { value: renderer.getPixelRatio() },
  uConfMin: { value: 0 },
  uProbMin: { value: 0 },
  uHighlight: { value: 0 },
  uClassVis: { value: new Array(12).fill(1) },
};

const VERT = /* glsl */`
attribute vec3 aColor;
attribute vec3 aSeg;
attribute float aCls;
attribute float aWp;
attribute float aRank;
attribute float aConf;
attribute float aProb;

uniform int   uMode;
uniform float uProgress;
uniform float uPointSize;
uniform float uPixelRatio;
uniform float uConfMin;
uniform float uProbMin;
uniform float uHighlight;
uniform float uClassVis[12];

varying vec3  vColor;
varying float vNew;

void main() {
  int cls = int(aCls + 0.5);
  float k = floor(uProgress + 0.0001);
  float f = clamp(uProgress - k, 0.0, 1.0);

  bool revealed = (aWp < k + 0.5) || (abs(aWp - (k + 1.0)) < 0.5 && aRank <= f);
  if (!revealed || uClassVis[cls] < 0.5 || aConf < uConfMin || aProb < uProbMin) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vColor = vec3(0.0);
    vNew = 0.0;
    return;
  }

  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uPointSize * uPixelRatio * clamp(2.5 / max(0.2, -mv.z), 0.55, 1.35);

  vec3 c = (uMode == 0) ? aColor : aSeg;
  if (uMode == 0) c = clamp(pow(c, vec3(0.62)) * 1.15, 0.0, 1.0);

  vNew = (aWp > k + 0.5) ? 1.0 : 0.0;
  if (uHighlight > 0.5 && vNew > 0.5) c = mix(c, vec3(1.0, 0.86, 0.20), 0.72);
  vColor = c;
}
`;

const FRAG = /* glsl */`
varying vec3 vColor;
varying float vNew;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}
`;

S.material = new THREE.ShaderMaterial({
  uniforms, vertexShader: VERT, fragmentShader: FRAG,
  transparent: false, depthWrite: true, depthTest: true,
});
S.geo = new THREE.BufferGeometry();
S.pointMesh = new THREE.Points(S.geo, S.material);
S.pointMesh.frustumCulled = false;
scene.add(S.pointMesh);

/* ------------------------------------------------------------ scene build */
function clearGroup(g) {
  while (g.children.length) {
    const o = g.children.pop();
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
      else o.material.dispose();
    }
  }
}

function disposeClipScene() {
  clearGroup(gRoute);
  clearGroup(gAux);
  S.markers = [];
  S.drone = null;
  S.frustum = null;
  S.travelLine = null;
  S.fullLine = null;
  S.routeVecs = [];
  S.routeTans = [];
  S._pos = new THREE.Vector3();
  S._tan = new THREE.Vector3();
  S.pipReady = false;
  S.pipWp = -1;
  pipMesh.visible = false;
  if (S.geo) { S.geo.dispose(); }
  S.geo = new THREE.BufferGeometry();
  S.pointMesh.geometry = S.geo;
  if (S.grid) { scene.remove(S.grid); S.grid.geometry.dispose(); S.grid.material.dispose(); S.grid = null; }
}

function buildGeometry(buf, meta) {
  const N = meta.num_points;
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(buf, 0, 3 * N);
  const rgb = new Uint8Array(buf, 12 * N, 3 * N);
  const cls = new Uint8Array(buf, 15 * N, N);
  const wp = new Uint8Array(buf, 16 * N, N);
  const rank = new Uint8Array(buf, 17 * N, N);
  const conf = new Uint8Array(buf, 18 * N, N);
  const prob = new Uint8Array(buf, 19 * N, N);

  // The cache only stores the class id per point; expand it to the AIC palette
  // once here so the shader can switch RGB / semantic without extra work.
  const seg = new Uint8Array(3 * N);
  const pal = meta.palette;
  for (let i = 0, k = 0; i < N; i++, k += 3) {
    const c = pal[cls[i]] || pal[0];
    seg[k] = c[0]; seg[k + 1] = c[1]; seg[k + 2] = c[2];
  }

  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aColor', new THREE.BufferAttribute(rgb, 3, true));
  g.setAttribute('aSeg', new THREE.BufferAttribute(seg, 3, true));
  g.setAttribute('aCls', new THREE.BufferAttribute(cls, 1, false));
  g.setAttribute('aWp', new THREE.BufferAttribute(wp, 1, false));
  g.setAttribute('aRank', new THREE.BufferAttribute(rank, 1, true));
  g.setAttribute('aConf', new THREE.BufferAttribute(conf, 1, true));
  g.setAttribute('aProb', new THREE.BufferAttribute(prob, 1, true));
  g.computeBoundingSphere();

  // probability distribution for the "filtered points" estimate
  const hist = new Uint32Array(256);
  for (let i = 0; i < N; i++) hist[prob[i]]++;
  S.probCdf = new Float32Array(256);
  let acc = 0;
  for (let i = 255; i >= 0; i--) { acc += hist[i]; S.probCdf[i] = acc / Math.max(1, N); }

  if (S.geo) S.geo.dispose();
  S.geo = g;
  S.pointMesh.geometry = g;
  S.buffer = buf;
}

function vec3(a) { return new THREE.Vector3(a[0], a[1], a[2]); }

function buildSceneClip(meta) {
  const lo = vec3(meta.bounds[0]);
  const hi = vec3(meta.bounds[1]);
  const center = lo.clone().add(hi).multiplyScalar(0.5);
  const size = hi.clone().sub(lo);
  const radius = Math.max(size.length() * 0.5, 1e-4);
  S.sceneInfo = { lo, hi, center, size, radius };

  // route
  S.routeVecs = meta.route.map(vec3);
  S.routeTans = S.routeVecs.map((p, i) => {
    const a = S.routeVecs[Math.max(0, i - 1)];
    const b = S.routeVecs[Math.min(S.routeVecs.length - 1, i + 1)];
    return b.clone().sub(a).normalize();
  });
  const fullGeo = new THREE.BufferGeometry().setFromPoints(S.routeVecs);
  S.fullLine = new THREE.Line(fullGeo, new THREE.LineBasicMaterial({
    color: 0x64748b, transparent: true, opacity: 0.6,
  }));
  gRoute.add(S.fullLine);

  const travelGeo = new THREE.BufferGeometry().setFromPoints(S.routeVecs);
  S.travelLine = new THREE.Line(travelGeo, new THREE.LineBasicMaterial({ color: 0xff4d4d }));
  S.travelLine.geometry.setDrawRange(0, 0);
  gRoute.add(S.travelLine);

  // waypoint markers
  const sph = new THREE.SphereGeometry(radius * 0.007, 10, 8);
  meta.waypoints.forEach((w, i) => {
    const mat = new THREE.MeshBasicMaterial({ color: 0x93a9c0, transparent: true, opacity: 0.75 });
    const m = new THREE.Mesh(sph, mat);
    m.position.copy(vec3(w.cam));
    m.userData.i = i;
    m.visible = !S.follow;
    gRoute.add(m);
    S.markers.push(m);
  });

  // drone (sphere + heading cone)
  const drone = new THREE.Group();
  const body = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.016, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffe066 }));
  drone.add(body);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(radius * 0.009, radius * 0.045, 12),
    new THREE.MeshBasicMaterial({ color: 0x67e8f9 }));
  cone.position.y = radius * 0.022;
  drone.add(cone);
  gRoute.add(drone);
  S.drone = { group: drone, cone };
  drone.visible = !S.follow;

  // camera frustum
  const fg = new THREE.BufferGeometry();
  fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(8 * 2 * 3), 3));
  S.frustum = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({
    color: 0x67e8f9, transparent: true, opacity: 0.85,
  }));
  gAux.add(S.frustum);

  // ground grid at the lower bound
  const gsize = Math.max(size.x, size.z) * 1.35;
  S.grid = new THREE.GridHelper(gsize, 24, 0x1d3b55, 0x122232);
  S.grid.material.transparent = true;
  S.grid.material.opacity = 0.3;
  S.grid.position.set(center.x, lo.y - radius * 0.01, center.z);
  scene.add(S.grid);

  // prefix point counts
  S.prefix = [0];
  (meta.counts || []).forEach((c) => S.prefix.push(S.prefix[S.prefix.length - 1] + c));
}

function routeSample(p) {
  const route = S.routeVecs;
  if (!route.length) return { pos: new THREE.Vector3(), tan: new THREE.Vector3(0, 0, 1) };
  const t = S.K > 1 ? clamp(p / (S.K - 1), 0, 1) : 0;
  const f = t * (route.length - 1);
  const i0 = Math.floor(f), i1 = Math.min(route.length - 1, i0 + 1), a = f - i0;
  S._pos.lerpVectors(route[i0], route[i1], a);
  S._tan.lerpVectors(S.routeTans[i0], S.routeTans[i1], a).normalize();
  return { pos: S._pos, tan: S._tan };
}

function updateFrustum(wpIdx) {
  if (!S.frustum || !S.meta) return;
  if (S.follow) { S.frustum.visible = false; return; }
  const w = S.meta.waypoints[Math.min(wpIdx, S.meta.waypoints.length - 1)];
  if (!w || !w.R || !w.fov) { S.frustum.visible = false; return; }
  S.frustum.visible = true;
  const R = w.R;
  const fovH = THREE.MathUtils.degToRad(w.fov[0] / 2);
  const fovW = THREE.MathUtils.degToRad(w.fov[1] / 2);
  const tx = Math.tan(fovW), ty = Math.tan(fovH);
  const len = S.sceneInfo.radius * 0.22;
  const cam = vec3(w.cam);
  const corners = [[-tx, -ty], [tx, -ty], [tx, ty], [-tx, ty]];
  const tips = corners.map(([cx, cy]) => {
    const v = new THREE.Vector3(cx, cy, 1).normalize();
    // world = R^T * v  (R is row-major world->camera)
    const wx = R[0] * v.x + R[3] * v.y + R[6] * v.z;
    const wy = R[1] * v.x + R[4] * v.y + R[7] * v.z;
    const wz = R[2] * v.x + R[5] * v.y + R[8] * v.z;
    return new THREE.Vector3(wx, wy, wz).multiplyScalar(len).add(cam);
  });
  const arr = S.frustum.geometry.attributes.position.array;
  let o = 0;
  const put = (a, b) => {
    arr[o++] = a.x; arr[o++] = a.y; arr[o++] = a.z;
    arr[o++] = b.x; arr[o++] = b.y; arr[o++] = b.z;
  };
  tips.forEach((t) => put(cam, t));
  for (let i = 0; i < 4; i++) put(tips[i], tips[(i + 1) % 4]);
  S.frustum.geometry.attributes.position.needsUpdate = true;
}

function setDrone(p) {
  if (!S.drone) return;
  const { pos, tan } = routeSample(p);
  S.drone.group.position.copy(pos);
  const up = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(up, tan.clone().normalize());
  S.drone.cone.quaternion.copy(q);
}

/* --------------------------------------------------------------- camera */
function fitPreset(name, instant = false) {
  if (!S.sceneInfo) return;
  const { center, radius } = S.sceneInfo;
  camera.fov = BASE_FOV;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  let elev = 22, azim = -58, dist = radius / Math.sin(fov / 2) * 1.02;
  if (name === 'top') { elev = 88; azim = -90; dist = radius * 2.3; }
  else if (name === 'street') {
    elev = 6;
    const t = S.routeTans[0] || new THREE.Vector3(0, 0, 1);
    azim = Math.atan2(-t.z, -t.x) * 180 / Math.PI;
    dist = radius * 1.35;
  } else if (name === 'orbit') {
    elev = 26; azim = -58; dist = radius / Math.sin(fov / 2) * 1.05;
    controls.autoRotate = true;
  }
  if (name !== 'orbit') controls.autoRotate = false;
  const el = THREE.MathUtils.degToRad(elev);
  const az = THREE.MathUtils.degToRad(azim);
  camera.position.set(
    center.x + dist * Math.cos(el) * Math.cos(az),
    center.y + dist * Math.sin(el),
    center.z + dist * Math.cos(el) * Math.sin(az),
  );
  controls.target.copy(center);
  camera.near = Math.max(radius * 0.001, 0.001);
  camera.far = radius * 80;
  camera.updateProjectionMatrix();
  controls.update();
  if (instant) { S.followReady = false; }
}

function setFollow(on, silent = false) {
  S.follow = !!on;
  $('#follow').checked = S.follow;
  controls.enabled = !S.follow;
  if (S.drone) S.drone.group.visible = !S.follow;
  if (S.frustum) S.frustum.visible = !S.follow;
  S.markers.forEach((m) => { m.visible = !S.follow; });
  if (S.follow) {
    S.followReady = false;
    controls.autoRotate = false;
  } else {
    controls.target.copy(S.followTarget.lengthSq() ? S.followTarget : (S.sceneInfo ? S.sceneInfo.center : new THREE.Vector3()));
    controls.update();
  }
  if (!silent) toast(S.follow ? '跟随视角已开启：相机沿预测航迹飞行' : '已切换到自由视角', '', 2000);
}

function applyFollow(dt) {
  if (!S.sceneInfo) return;
  const { pos, tan } = routeSample(S.progress);
  const flat = tan.clone();
  flat.y *= 0.25;
  if (flat.lengthSq() < 1e-8) flat.set(0, 0, 1);
  flat.normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const side = new THREE.Vector3().crossVectors(up, flat).normalize();
  const r = S.sceneInfo.radius;
  const wantPos = pos.clone().addScaledVector(flat, -0.60 * r)
    .addScaledVector(up, 0.34 * r).addScaledVector(side, 0.06 * r);
  const wantTgt = pos.clone().addScaledVector(flat, 0.62 * r).addScaledVector(up, 0.01 * r);
  if (!S.followReady) {
    S.followPos.copy(wantPos);
    S.followTarget.copy(wantTgt);
    S.followReady = true;
  }
  const k = 1 - Math.pow(0.0015, Math.max(0.001, dt));
  S.followPos.lerp(wantPos, k);
  S.followTarget.lerp(wantTgt, k);
  camera.position.copy(S.followPos);
  camera.lookAt(S.followTarget);
}

/* ---------------------------------------------------------------- render */
const BASE_FOV = 50;

function renderScene() {
  const w = Math.max(1, canvas.clientWidth);
  const h = Math.max(1, canvas.clientHeight);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, w, h);
  renderer.clear(true, true, true);
  if (!S.meta) return;
  if (S.mode === 'both') {
    const half = Math.floor(w / 2);
    const fullAspect = w / h;
    const halfAspect = half / h;
    // Keep the same horizontal field of view as the single viewport so each
    // half still shows the *whole* scene (rather than a cropped center slice).
    const fov = THREE.MathUtils.radToDeg(2 * Math.atan(
      Math.tan(THREE.MathUtils.degToRad(BASE_FOV / 2)) * fullAspect / halfAspect));
    camera.fov = fov;
    camera.aspect = halfAspect;
    camera.updateProjectionMatrix();
    renderer.setScissorTest(true);
    uniforms.uMode.value = 0;
    renderer.setViewport(0, 0, half, h);
    renderer.setScissor(0, 0, half, h);
    renderer.render(scene, camera);
    uniforms.uMode.value = 1;
    renderer.setViewport(half, 0, w - half, h);
    renderer.setScissor(half, 0, w - half, h);
    renderer.render(scene, camera);
    renderer.setScissorTest(false);
  } else {
    if (camera.fov !== BASE_FOV) camera.fov = BASE_FOV;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    uniforms.uMode.value = S.mode === 'seg' ? 1 : 0;
    renderer.setViewport(0, 0, w, h);
    renderer.render(scene, camera);
  }

  // screen-space RGB frame inset (bottom-left), captured by screenshot/recording
  if (S.rgbPip && pipMesh.visible) {
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    renderer.clearDepth();
    renderer.render(overlayScene, overlayCamera);
  }
}

function resize() {
  const rect = $('#viewport').getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  renderer.setSize(w, h, false);
  uniforms.uPixelRatio.value = renderer.getPixelRatio();
  layoutPip();
}
new ResizeObserver(resize).observe($('#viewport'));

/* ------------------------------------------------------------------- loop */
function animate(t) {
  requestAnimationFrame(animate);
  const rawDt = Math.max(0.001, (t - (S.lastFrameT || t)) / 1000);
  const dt = Math.min(0.06, rawDt);
  S.lastFrameT = t;

  if (S.meta) {
    if (S.playing) {
      // use wall-clock time (capped) so slow frames don't stretch the route
      let np = S.progress + Math.min(0.2, rawDt) * S.speed;
      if (np >= S.K - 1) {
        np = S.K - 1;
        setProgress(np, { silent: true });
        if (S.record) { stopRecording(true); }
        else if (S.loop) { S.playing = false; setPlaying(true, true); setProgress(0, { snap: true }); }
        else { setPlaying(false); }
      } else {
        setProgress(np);
      }
    }
    if (S.follow) applyFollow(dt);
    else controls.update();
    renderScene();
  } else {
    controls.update();
    renderScene();
  }

  // fps chip (use wall time so recording slowdowns are visible)
  S.fpsAcc += Math.min(1, rawDt); S.fpsN += 1;
  if (S.fpsAcc > 0.8) {
    const fps = S.fpsN / S.fpsAcc;
    $('#chipFps').textContent = `${fps.toFixed(0)} fps`;
    S.fpsAcc = 0; S.fpsN = 0;
  }
  if (S.record) {
    $('#recTime').textContent = `${((performance.now() - S.recStart) / 1000).toFixed(1)}s`;
  }
}
requestAnimationFrame(animate);

/* --------------------------------------------------------------- progress */
function filtersActive() {
  if (S.confMin > 0 || S.probMin > 0) return true;
  for (let i = 1; i < 12; i++) if (!S.classVis[i]) return true;
  return false;
}

function probFrac(minProb) {
  if (!S.probCdf || minProb <= 0) return 1;
  const bin = clamp(Math.round(minProb * 255), 0, 255);
  return S.probCdf[bin];
}

function filteredVisible(vis) {
  if (!S.meta) return vis;
  let sel = 0, total = 0;
  const cc = S.meta.class_counts || [];
  for (let i = 1; i < 12; i++) {
    total += cc[i] || 0;
    if (S.classVis[i]) sel += cc[i] || 0;
  }
  const classFrac = total > 0 ? sel / total : 1;
  return Math.round(vis * classFrac * (1 - S.confMin) * probFrac(S.probMin));
}

function updateFilteredDisplay() {
  if (!S.meta) return;
  const p = S.progress;
  const k = Math.floor(p + 1e-6);
  const f = p - k;
  const vis = (S.prefix[k + 1] || 0) + (k + 1 < S.K ? Math.round(f * S.meta.counts[k + 1]) : 0);
  if (filtersActive()) {
    $('#chipPoints').textContent = `${fmt(vis)} / ${fmt(S.N)} 点 · 过滤 ≈ ${fmt(filteredVisible(vis))}`;
  } else {
    $('#chipPoints').textContent = `${fmt(vis)} / ${fmt(S.N)} 点`;
  }
  updateStats(p);
}

function setProgress(p, opts = {}) {
  if (!S.meta) return;
  p = clamp(p, 0, S.K - 1);
  S.progress = p;
  uniforms.uProgress.value = p;
  $('#progress').value = p;

  const k = Math.floor(p + 1e-6);
  const f = p - k;
  const vis = (S.prefix[k + 1] || 0) + (k + 1 < S.K ? Math.round(f * S.meta.counts[k + 1]) : 0);
  if (filtersActive()) {
    $('#chipPoints').textContent = `${fmt(vis)} / ${fmt(S.N)} 点 · 过滤 ≈ ${fmt(filteredVisible(vis))}`;
  } else {
    $('#chipPoints').textContent = `${fmt(vis)} / ${fmt(S.N)} 点`;
  }
  $('#progressText').textContent = `${(p + 1).toFixed(1)} / ${S.K}`;

  // traveled route + drone
  if (S.travelLine) {
    const t = S.K > 1 ? p / (S.K - 1) : 0;
    const n = p <= 0 ? 0 : Math.max(2, Math.round(t * (S.routeVecs.length - 1)) + 1);
    S.travelLine.geometry.setDrawRange(0, n);
  }
  setDrone(p);

  const wp = Math.min(S.K - 1, Math.floor(p + 0.5));
  if (wp !== S.lastWp) {
    S.lastWp = wp;
    updateWpUI(wp);
  }
  if (!opts.silent) {
    S.uiAcc += 1;
    if (S.uiAcc % 4 === 0 || opts.force) updateStats(p);
    drawMinimap();
    scheduleHash();
  }
}


function loadPipImage(url) {
  if (!S.pipCache.has(url)) {
    S.pipCache.set(url, new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`RGB frame load failed: ${url}`));
      img.src = url;
    }));
  }
  return S.pipCache.get(url);
}

function roundRectPath(ctx, x, y, w, h, r) {
  if (ctx.roundRect) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.beginPath();
    ctx.rect(x, y, w, h);
  }
}

function drawPip(img, wp, idx) {
  const pad = 12, titleH = 38, innerW = 640;
  const imgW = innerW - pad * 2;
  const ratio = img.naturalHeight / Math.max(1, img.naturalWidth);
  const imgH = Math.max(1, Math.round(imgW * ratio));
  const cw = innerW, ch = titleH + imgH + pad * 2;
  pipCanvas.width = cw;
  pipCanvas.height = ch;
  const ctx = pipCtx;
  ctx.clearRect(0, 0, cw, ch);

  roundRectPath(ctx, 1, 1, cw - 2, ch - 2, 12);
  ctx.fillStyle = 'rgba(8,14,22,.94)';
  ctx.fill();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.save();
  roundRectPath(ctx, pad, titleH + pad - 6, imgW, imgH, 6);
  ctx.clip();
  ctx.drawImage(img, pad, titleH + pad - 6, imgW, imgH);
  ctx.restore();

  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#dff3ff';
  ctx.font = '600 17px -apple-system,"Segoe UI","Noto Sans SC",sans-serif';
  ctx.fillText(`RGB FRAME ${String(idx + 1).padStart(2, '0')} / ${S.K}`, pad, 11);
  ctx.fillStyle = '#8fa4ba';
  ctx.font = '12px "SFMono-Regular",Consolas,monospace';
  ctx.textAlign = 'right';
  ctx.fillText(wp.name || '', cw - pad, 14);
  ctx.textAlign = 'left';

  pipTexture.needsUpdate = true;
  S.pipReady = true;
  layoutPip();
}

function layoutPip() {
  if (!S.meta || !S.rgbPip || !S.pipReady || !pipCanvas.width) {
    pipMesh.visible = false;
    return;
  }
  const w = Math.max(1, canvas.clientWidth);
  const h = Math.max(1, canvas.clientHeight);
  overlayCamera.left = 0;
  overlayCamera.right = w;
  overlayCamera.top = h;
  overlayCamera.bottom = 0;
  overlayCamera.updateProjectionMatrix();

  const maxW = Math.min(w * 0.30, 400);
  const maxH = h * 0.36;
  const scale = Math.min(maxW / pipCanvas.width, maxH / pipCanvas.height);
  const dw = Math.max(1, pipCanvas.width * scale);
  const dh = Math.max(1, pipCanvas.height * scale);
  const margin = Math.max(12, Math.round(w * 0.014));
  pipMesh.scale.set(dw, dh, 1);
  // top-right corner (bottom-left collided with the split-view labels)
  pipMesh.position.set(w - margin - dw / 2, h - margin - dh / 2, 0);
  pipMesh.visible = true;
}

function updatePip(idx) {
  if (!S.meta || idx < 0) { pipMesh.visible = false; return; }
  const wp = S.meta.waypoints[idx];
  if (!wp) { pipMesh.visible = false; return; }
  S.pipWp = idx;
  const req = ++S.pipReq;
  const url = `/clips/${S.clipId}/${wp.thumb}`;
  loadPipImage(url).then((img) => {
    if (req !== S.pipReq || !S.meta) return;
    drawPip(img, wp, idx);
  }).catch(() => { pipMesh.visible = false; });
}

function openGallery() {
  if (!S.meta) { toast('请先加载片段', 'warn'); return; }
  if (S.record) { toast('录制中暂不浏览帧序列', 'warn'); return; }
  setPlaying(false);
  const host = $('#galleryGrid');
  host.innerHTML = '';
  S.meta.waypoints.forEach((wp, i) => {
    const b = document.createElement('button');
    b.className = 'gallery-item';
    b.innerHTML = `
      <img loading="lazy" src="/clips/${S.clipId}/${wp.thumb}" alt="">
      <div class="gallery-meta"><b>${String(i + 1).padStart(2, '0')} · ${wp.name || ''}</b><span>${fmt(wp.count)} pts</span></div>`;
    b.addEventListener('click', () => {
      closeGallery();
      setProgress(i, { snap: true, force: true });
    });
    host.appendChild(b);
  });
  $('#galleryModal').classList.remove('hidden');
}

function closeGallery() {
  $('#galleryModal').classList.add('hidden');
}

function updateWpUI(i) {
  if (!S.meta) return;
  const w = S.meta.waypoints[i];
  if (!w) return;
  const img = $('#uavImg');
  img.style.display = 'block';
  img.src = `/clips/${S.clipId}/${w.thumb}`;
  img.alt = w.name || `waypoint ${i + 1}`;
  $('#uavEmpty').style.display = 'none';
  $('#uavIndex').textContent = `${i + 1} / ${S.K}`;
  $('#uavName').textContent = w.name || `WP ${i + 1}`;
  $('#uavHint').textContent = `第 ${i + 1} 个航点 · ${fmt(w.count)} 点`;
  updatePip(i);
  updateFrustum(i);

  S.markers.forEach((m, j) => {
    const on = j === i;
    m.scale.setScalar(on ? 1.9 : 1);
    m.material.color.setHex(on ? 0xffe066 : 0x93a9c0);
    m.material.opacity = on ? 1 : 0.65;
  });

  $$('.wp-thumb', $('#filmstrip')).forEach((el, j) => {
    el.classList.toggle('active', j === i);
    el.classList.toggle('done', j <= i);
  });
  const active = $(`.wp-thumb[data-i="${i}"]`, $('#filmstrip'));
  if (active && S.playing) active.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
}

function updateStats(p) {
  if (!S.meta) return;
  const m = S.meta;
  const k = Math.floor(p + 1e-6);
  const f = p - k;
  const vis = (S.prefix[k + 1] || 0) + (k + 1 < S.K ? Math.round(f * m.counts[k + 1]) : 0);
  const shown = filteredVisible(vis);
  const src = m.source || {};
  let srcLabel = '服务器帧目录';
  if (src.kind === 'video') srcLabel = `视频 · ${(src.indices || []).length} 帧`;
  else if (src.kind === 'images') srcLabel = `图片 · ${src.count || (src.paths || []).length} 张`;
  else if (src.paths) srcLabel = `帧目录 · ${src.paths.length} 帧`;
  const dl = $('#stats');
  dl.innerHTML = `
    <dt>已显示点数</dt><dd>${fmt(vis)}</dd>
    ${filtersActive() ? `<dt>过滤后可见</dt><dd>≈ ${fmt(shown)}</dd>` : ''}
    <dt>总点数</dt><dd>${fmt(m.num_points)}</dd>
    <dt>航点</dt><dd>${m.num_waypoints}</dd>
    <dt>输入</dt><dd>${m.input_hw ? `${m.input_hw[0]}×${m.input_hw[1]}` : '—'}</dd>
    <dt>几何</dt><dd>${(m.params && m.params.point_source) === 'point' ? 'point head' : 'depth + pose'}</dd>
    <dt>模型</dt><dd>${(m.model && m.model.model_label) || 'v5'}</dd>
    <dt>推理耗时</dt><dd>${m.duration_s != null ? m.duration_s + ' s' : '—'}</dd>
    <dt>数据源</dt><dd class="full">${srcLabel}</dd>`;
}

function scheduleHash() {
  clearTimeout(S.hashTimer);
  S.hashTimer = setTimeout(() => {
    if (!S.clipId) return;
    const h = `#clip=${encodeURIComponent(S.clipId)}&p=${S.progress.toFixed(2)}`;
    if (location.hash !== h) history.replaceState(null, '', h);
  }, 400);
}

/* ---------------------------------------------------------------- minimap */
function setupMinimap() {
  const c = $('#minimap');
  const ctx = c.getContext('2d');
  const meta = S.meta;
  const route = meta.route;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  route.forEach((p) => {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minZ = Math.min(minZ, p[2]); maxZ = Math.max(maxZ, p[2]);
  });
  const w = c.width, h = c.height, pad = 30;
  const rangeX = Math.max(1e-6, maxX - minX), rangeZ = Math.max(1e-6, maxZ - minZ);
  const scale = Math.min((w - 2 * pad) / rangeX, (h - 2 * pad) / rangeZ);
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  S.map = {
    w, h, scale, cx, cz,
    toX: (x) => w / 2 + (x - cx) * scale,
    toY: (z) => h / 2 - (z - cz) * scale,
    fromX: (x) => (x - w / 2) / scale + cx,
    fromY: (y) => (h / 2 - y) / scale + cz,
  };
  drawMinimap();
}

function drawMinimap() {
  if (!S.meta || !S.map) return;
  const c = $('#minimap');
  const ctx = c.getContext('2d');
  const m = S.map;
  ctx.clearRect(0, 0, m.w, m.h);
  ctx.fillStyle = '#080f17';
  ctx.fillRect(0, 0, m.w, m.h);

  // faint grid
  ctx.strokeStyle = 'rgba(56,89,120,.18)';
  ctx.lineWidth = 1;
  for (let x = 0; x < m.w; x += 56) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, m.h); ctx.stroke(); }
  for (let y = 0; y < m.h; y += 56) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(m.w, y); ctx.stroke(); }

  const route = S.meta.route;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  // full route
  ctx.beginPath();
  route.forEach((p, i) => {
    const x = m.toX(p[0]), y = m.toY(p[2]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#53687e';
  ctx.lineWidth = 2.4;
  ctx.stroke();

  // traveled
  const t = S.K > 1 ? S.progress / (S.K - 1) : 0;
  const n = p2n(t, route.length);
  if (n > 1) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = m.toX(route[i][0]), y = m.toY(route[i][2]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 3.2;
    ctx.stroke();
  }

  // waypoint dots
  S.meta.waypoints.forEach((w, i) => {
    const x = m.toX(w.cam[0]), y = m.toY(w.cam[2]);
    const cur = i === S.lastWp;
    ctx.beginPath();
    ctx.arc(x, y, cur ? 4.5 : 2.6, 0, Math.PI * 2);
    ctx.fillStyle = cur ? '#ffe066' : (i <= S.lastWp ? '#ff8a8a' : '#7b8ea3');
    ctx.fill();
    if (cur) {
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,224,102,.35)'; ctx.lineWidth = 2; ctx.stroke();
    }
  });

  // current position
  const { pos } = routeSample(S.progress);
  const x = m.toX(pos.x), y = m.toY(pos.z);
  ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#fff7d6'; ctx.fill();
  ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 2; ctx.stroke();
}

function p2n(t, len) { return Math.max(0, Math.min(len, Math.round(t * (len - 1)) + 1)); }

/* --------------------------------------------------------------- UI setup */
function renderClassList(meta) {
  const host = $('#classList');
  host.innerHTML = '';
  for (let i = 1; i < 12; i++) {
    const count = (meta.class_counts && meta.class_counts[i]) || 0;
    const col = `rgb(${meta.palette[i].join(',')})`;
    const item = document.createElement('div');
    item.className = 'class-item' + (S.classVis[i] ? '' : ' off') + (count ? '' : ' empty');
    item.dataset.cls = i;
    item.innerHTML = `
      <span class="class-swatch" style="background:${col}"></span>
      <span class="class-name">${meta.class_labels_zh[i]}<small>${meta.class_names[i]}</small></span>
      <span class="class-count">${count ? fmt(count) : '—'}</span>
      <span class="eye">◉</span>`;
    item.addEventListener('click', () => {
      setClassVisible(i, !S.classVis[i]);
    });
    host.appendChild(item);
  }
  updateClassUniform();
}

function setClassVisible(i, on) {
  S.classVis[i] = !!on;
  const item = $(`.class-item[data-cls="${i}"]`, $('#classList'));
  if (item) {
    item.classList.toggle('off', !on);
    $('.eye', item).textContent = on ? '◉' : '○';
  }
  updateClassUniform();
  updateFilteredDisplay();
}

function updateClassUniform() {
  const arr = uniforms.uClassVis.value;
  for (let i = 0; i < 12; i++) arr[i] = (i !== 0 && S.classVis[i]) ? 1 : 0;
}

function renderFilmstrip(meta) {
  const host = $('#filmstrip');
  host.innerHTML = '';
  meta.waypoints.forEach((w, i) => {
    const b = document.createElement('button');
    b.className = 'wp-thumb';
    b.dataset.i = i;
    b.title = `${w.name || 'WP ' + (i + 1)} · ${fmt(w.count)} 点`;
    b.innerHTML = `<img loading="lazy" src="/clips/${S.clipId}/${w.thumb}" alt=""><span>${String(i + 1).padStart(2, '0')}</span>`;
    b.addEventListener('click', () => {
      setPlaying(false);
      setProgress(Math.min(i, S.K - 1), { snap: true, force: true });
    });
    host.appendChild(b);
  });
  $('#wpCount').textContent = String(meta.waypoints.length);
}

function setupUIForClip(meta) {
  const K = meta.num_waypoints;
  $('#progress').max = Math.max(0, K - 1);
  $('#progress').step = 0.001;
  $('#progress').disabled = false;
  $('#btnPlay').disabled = false;
  $('#chipClip').textContent = meta.label || S.clipId;
  $('#uavImg').style.display = 'block';
  $('#uavImg').src = '';
  $('#uavEmpty').style.display = 'none';
  renderClassList(meta);
  renderFilmstrip(meta);
  setupMinimap();
  S.lastWp = -1;
  S.prefix = [0];
  (meta.counts || []).forEach((c) => S.prefix.push(S.prefix[S.prefix.length - 1] + c));

  $('#btnPlyRgb').href = `/api/clips/${S.clipId}/ply?mode=rgb`;
  $('#btnPlySeg').href = `/api/clips/${S.clipId}/ply?mode=seg`;
  $('#btnPlyRgb').setAttribute('download', `${S.clipId}_rgb.ply`);
  $('#btnPlySeg').setAttribute('download', `${S.clipId}_seg.ply`);

  updateStats(0);
  $('#statusDot').classList.add('ok');
}

/* ------------------------------------------------------------------ modes */
function applyMode(mode) {
  S.mode = mode;
  $$('#modeSeg button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#stageLabels').classList.toggle('hidden', mode !== 'both');
  const label = mode === 'rgb' ? 'RGB 点云' : (mode === 'seg' ? '语义点云' : 'RGB ｜ 语义 双栏');
  $('#modeHint').textContent = label;
  renderScene();
}

function setPlaying(on, silent = false) {
  S.playing = !!on;
  $('#playIcon').textContent = S.playing ? '❚❚' : '▶';
  $('#playText').textContent = S.playing ? '暂停' : '播放';
  $('#btnPlay').classList.toggle('playing', S.playing);
  if (S.playing && S.progress >= S.K - 1) setProgress(0, { snap: true });
  if (!silent) { /* no toast for play */ }
}

/* --------------------------------------------------------------- data load */
async function loadClip(id, opts = {}) {
  if (!id) return;
  const token = ++S.loadingToken;
  S.clipId = id;
  $('#clipSelect').value = id;
  showLoading('加载片段…', '');
  setPlaying(false);
  try {
    const meta = await fetchJSON(`/api/clips/${encodeURIComponent(id)}/meta`);
    if (token !== S.loadingToken) return;
    S.meta = meta;
    S.N = meta.num_points;
    S.K = meta.num_waypoints;
    disposeClipScene();
    S.prefix = [0];
    (meta.counts || []).forEach((c) => S.prefix.push(S.prefix[S.prefix.length - 1] + c));

    setLoading(2, '读取点云缓存…');
    const buf = await fetchWithProgress(`/clips/${encodeURIComponent(id)}/points.bin`, (got, total) => {
      if (token !== S.loadingToken) return;
      const pct = total ? (got / total) * 100 : 0;
      setLoading(pct, `${fmtMB(got)} / ${fmtMB(total)} MB`);
    });
    if (token !== S.loadingToken) return;

    buildGeometry(buf, meta);
    buildSceneClip(meta);
    setupUIForClip(meta);

    const hashClip = parseHashClip();
    const hp = opts.progress != null ? opts.progress
      : (hashClip === id ? parseHashProgress() : null);
    setProgress(hp != null ? hp : Math.max(0, S.K - 1), { snap: true, force: true });
    applyMode(S.mode);
    fitPreset('oblique', true);
    setFollow(false, true);
    hideLoading();
    $('#empty').classList.add('hidden');
    toast(`已加载 ${meta.label || id}`, 'ok', 2200);
    updateHashClip(id);
  } catch (err) {
    if (token !== S.loadingToken) return;
    console.error(err);
    showLoadingError(err.message);
    toast(`加载失败：${err.message}`, 'err', 6000);
    if (!S.clips.length) $('#empty').classList.remove('hidden');
  }
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} (${url})`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !res.body.getReader) {
    const buf = await res.arrayBuffer();
    onProgress(buf.byteLength, total || buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const ch of chunks) { out.set(ch, off); off += ch.length; }
  return out.buffer;
}

function showLoading(title, sub) {
  $('#loading').classList.remove('hidden');
  $('#loadingText').textContent = title;
  $('#loadingSub').textContent = sub || '';
  $('#loadingBar').style.width = '0%';
}
function setLoading(pct, sub) {
  $('#loadingBar').style.width = `${clamp(pct, 0, 100).toFixed(1)}%`;
  if (sub != null) $('#loadingSub').textContent = sub;
}
function hideLoading() { $('#loading').classList.add('hidden'); }
function showLoadingError(msg) {
  $('#loadingText').textContent = '加载失败';
  $('#loadingSub').textContent = msg;
  $('#loadingBar').style.width = '100%';
  setTimeout(() => { if (S.meta) hideLoading(); }, 2500);
}

function parseHashProgress() {
  const m = location.hash.match(/p=([0-9.]+)/);
  if (!m) return null;
  const v = Number(m[1]);
  return Number.isFinite(v) ? v : null;
}
function parseHashClip() {
  const m = location.hash.match(/clip=([^&]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}
function updateHashClip(id) {
  history.replaceState(null, '', `#clip=${encodeURIComponent(id)}&p=${S.progress.toFixed(2)}`);
}

/* --------------------------------------------------------------- clip list */
async function refreshClips(selectId) {
  const data = await fetchJSON('/api/clips');
  S.clips = data.clips || [];
  const sel = $('#clipSelect');
  const cur = selectId || S.clipId;
  sel.innerHTML = '';
  if (!S.clips.length) {
    const o = document.createElement('option');
    o.textContent = '（暂无片段）'; o.value = '';
    sel.appendChild(o);
    $('#empty').classList.remove('hidden');
    return;
  }
  S.clips.forEach((c) => {
    const o = document.createElement('option');
    o.value = c.id;
    o.textContent = `${c.label} · ${c.num_waypoints} 航点`;
    sel.appendChild(o);
  });
  const next = S.clips.some((c) => c.id === cur) ? cur : S.clips[0].id;
  sel.value = next;
  return next;
}

/* ------------------------------------------------------------------ modal */
function openModal(tab) {
  $('#modal').classList.remove('hidden');
  $('#modal').setAttribute('aria-hidden', 'false');
  if (tab) switchTab(tab);
  updateModalHint();
}
function closeModal() {
  $('#modal').classList.add('hidden');
  $('#modal').setAttribute('aria-hidden', 'true');
}
function switchTab(tab) {
  S.tab = tab;
  $$('#srcTabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === tab));
  updateModalHint();
}

function updateModalHint() {
  let n = 0;
  if (S.tab === 'video') n = S.videoSelIdx.size;
  else if (S.tab === 'images') n = S.imagesItems.filter((c) => c.selected).length;
  else n = S.frameSel.size;
  const hint = $('#modalHint');
  const cap = inferCapForSize();
  if (n < 2) hint.textContent = `已选 ${n} 帧 · 至少需要 2 个航点`;
  else if (n > maxFrames()) hint.textContent = `已选 ${n} 帧 · 最多 ${maxFrames()} 个航点`;
  else if (n > cap) hint.textContent = `已选 ${n} 帧 · 单次推理约 ${cap} 帧，提交后会均匀降采样`;
  else hint.textContent = `已选 ${n} 个航点 · 将进行 1 次模型推理`;
  return n;
}

function renderPresets() {
  const host = $('#framePresets');
  host.innerHTML = '';
  (S.config.presets || []).forEach((p) => {
    const b = document.createElement('button');
    b.textContent = `${p.label} (${p.count})`;
    b.title = p.path;
    b.addEventListener('click', () => {
      $('#framePath').value = p.path;
      browseFrames(p.path);
    });
    host.appendChild(b);
  });
}

async function browseFrames(path) {
  if (!path) return;
  try {
    const data = await fetchJSON(`/api/browse?path=${encodeURIComponent(path)}`);
    S.framePath = data.path;
    S.frameItems = data.images || [];
    S.frameDirs = data.dirs || [];
    $('#framePath').value = data.path;
    $('#frameDirInfo').textContent = `${data.path} · ${data.image_total} 张图片${data.truncated ? '（仅显示前 300）' : ''}`;
    renderFrameDirs(data);
    renderFrameImages();
    updateModalHint();
  } catch (err) {
    toast(`目录浏览失败：${err.message}`, 'err');
  }
}

function renderFrameDirs(data) {
  const host = $('#frameDirs');
  host.innerHTML = '';
  const up = document.createElement('button');
  up.textContent = '↑ 上级';
  up.addEventListener('click', () => browseFrames(data.parent));
  host.appendChild(up);
  (S.frameDirs || []).slice(0, 30).forEach((d) => {
    const b = document.createElement('button');
    b.textContent = `📁 ${d.name}`;
    b.title = d.path;
    b.addEventListener('click', () => browseFrames(d.path));
    host.appendChild(b);
  });
}

function renderFrameImages() {
  const host = $('#frameImages');
  host.innerHTML = '';
  S.frameItems.forEach((it, i) => {
    const el = thumbEl(it.thumb, i + 1, baseName(it.name), S.frameSel.has(it.path));
    el.addEventListener('click', () => {
      if (S.frameSel.has(it.path)) S.frameSel.delete(it.path);
      else {
        if (S.frameSel.size >= maxFrames()) { toast(`最多选择 ${maxFrames()} 个航点`, 'warn'); return; }
        S.frameSel.set(it.path, it);
      }
      el.classList.toggle('selected', S.frameSel.has(it.path));
      $('#frameSelCount').textContent = S.frameSel.size;
      updateModalHint();
    });
    host.appendChild(el);
  });
  $('#frameSelCount').textContent = S.frameSel.size;
  updateModalHint();
}

function thumbEl(src, idx, title, selected, timeLabel) {
  const el = document.createElement('div');
  el.className = 'thumb-item' + (selected ? ' selected' : '');
  el.title = title || '';
  el.innerHTML = `
    <img loading="lazy" src="${src}" alt="">
    <span class="thumb-idx">${String(idx).padStart(2, '0')}</span>
    ${timeLabel ? `<span class="thumb-time">${timeLabel}</span>` : ''}
    <span class="thumb-check">✓</span>`;
  return el;
}

/* --------------------------------------------------------- video sources */
async function uploadVideo(file) {
  if (!file) return;
  $('#videoUploadStatus').classList.remove('hidden');
  $('#videoUploadBar').style.width = '0%';
  $('#videoUploadText').textContent = `上传中… 0% · ${fmtMB(file.size)} MB`;
  const previewEl = $('#videoPreview'); if (previewEl) previewEl.classList.add('hidden');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const data = await xhrUpload(`/api/upload/video?candidates=20`, fd, (loaded, total) => {
      const pct = total ? loaded / total * 100 : 0;
      $('#videoUploadBar').style.width = `${pct}%`;
      $('#videoUploadText').textContent = `上传中… ${pct.toFixed(0)}% · ${fmtMB(loaded)} / ${fmtMB(total)} MB`;
    }, () => { $('#videoUploadText').textContent = '上传完成，正在抽帧…'; });
    const info = data.info || {};
    S.videoUploadId = data.upload_id;
    S.videoInfo = info;
    S.videoTotal = Math.max(0, Number(info.nb_frames) ||
      Math.round((Number(info.duration) || 0) * (Number(info.fps) || 25)) || 0);
    S.videoCands = (data.candidates || []).map((c) => ({ ...c }));
    S.videoSelIdx = new Set(selectEvenly(S.videoCands, Math.min(16, S.videoCands.length))
      .map((c) => c.idx));
    $('#videoPickbar').classList.remove('hidden');
    $('#videoCandCount').textContent = S.videoCands.length;
    const totalEl = $('#videoTotalFrames');
    if (totalEl) totalEl.textContent = S.videoTotal ? fmt(S.videoTotal) : '—';
    const pv = $('#videoPreview');
    if (pv && data.video_url) {
      pv.src = data.video_url;
      pv.classList.remove('hidden');
    }
    $('#videoUploadText').textContent =
      `${data.filename || file.name} · ${(info.duration || 0).toFixed(1)}s · ${info.width || '?'}×${info.height || '?'} · ${(info.fps || 0).toFixed(1)}fps`;
    $('#optLabel').value = (file.name || '视频片段').replace(/\.[^.]+$/, '');
    renderVideoCandidates();
    toast(`视频已上传，预览 ${S.videoCands.length} 帧；可均匀选择 2–1000 帧`, 'ok', 5000);
  } catch (err) {
    $('#videoUploadText').textContent = `上传失败：${err.message}`;
    toast(`视频上传失败：${err.message}`, 'err', 6000);
  } finally {
    setTimeout(() => $('#videoUploadStatus').classList.add('hidden'), 3500);
  }
}

function renderVideoCandidates() {
  const host = $('#videoCandidates');
  host.innerHTML = '';
  S.videoCands.forEach((c, i) => {
    const on = S.videoSelIdx.has(c.idx);
    const el = thumbEl(c.url, i + 1, `frame ${c.idx} @ ${c.t}s`, on, `${c.t.toFixed(1)}s`);
    el.addEventListener('click', () => {
      const selected = S.videoSelIdx.has(c.idx);
      if (!selected && S.videoSelIdx.size >= maxFrames()) {
        toast(`最多选择 ${maxFrames()} 个航点`, 'warn');
        return;
      }
      if (selected) S.videoSelIdx.delete(c.idx);
      else S.videoSelIdx.add(c.idx);
      renderVideoCandidates();
    });
    host.appendChild(el);
  });
  updateVideoSelCount();
}

function updateVideoSummary() {
  const el = $('#videoSelSummary');
  if (!el) return;
  const idxs = Array.from(S.videoSelIdx).sort((a, b) => a - b);
  const cap = inferCapForSize();
  if (!idxs.length) {
    el.textContent = `未选择帧；可均匀选择 2–1000 帧（单次推理约 ${cap} 帧，超出会均匀降采样）。`;
    return;
  }
  const fps = Number(S.videoInfo && S.videoInfo.fps) || 0;
  const first = idxs[0], last = idxs[idxs.length - 1];
  const step = idxs.length > 1 ? (last - first) / (idxs.length - 1) : 0;
  let txt = `索引 ${first}–${last} · 共 ${idxs.length} 帧 · 平均间隔 ${step.toFixed(1)} 帧`;
  if (fps > 0) txt += ` / ${(step / fps).toFixed(2)}s · ${(first / fps).toFixed(1)}–${(last / fps).toFixed(1)}s`;
  if (idxs.length > S.videoCands.length) txt += ` · 缩略图仅预览 ${S.videoCands.length} 帧`;
  if (idxs.length > cap) txt += ` · 单次推理上限约 ${cap} 帧（${$('#optSize').value}），提交后均匀降采样`;
  el.textContent = txt;
}

function updateVideoSelCount() {
  const n = S.videoSelIdx.size;
  const selEl = $('#videoSelCount'); if (selEl) selEl.textContent = n;
  const totalEl = $('#videoTotalFrames');
  if (totalEl) totalEl.textContent = S.videoTotal ? fmt(S.videoTotal) : '—';
  const candEl = $('#videoCandCount'); if (candEl) candEl.textContent = S.videoCands.length;
  updateVideoSummary();
  updateModalHint();
}

async function uploadImages(files) {
  if (!files || !files.length) return;
  $('#imagesUploadStatus').classList.remove('hidden');
  $('#imagesUploadBar').style.width = '0%';
  const fd = new FormData();
  Array.from(files).forEach((f) => fd.append('files', f));
  $('#imagesUploadText').textContent = `上传 ${files.length} 张图片…`;
  try {
    const data = await xhrUpload('/api/upload/images', fd, (loaded, total) => {
      const pct = total ? loaded / total * 100 : 0;
      $('#imagesUploadBar').style.width = `${pct}%`;
      $('#imagesUploadText').textContent = `上传中… ${pct.toFixed(0)}%`;
    }, () => { $('#imagesUploadText').textContent = '上传完成，正在生成缩略图…'; });
    S.imagesUploadId = data.upload_id;
    S.imagesItems = (data.items || []).map((it) => ({ ...it, selected: true }));
    if (S.imagesItems.length > maxFrames()) {
      S.imagesItems.forEach((it, i) => { it.selected = i < maxFrames(); });
      toast(`图片超过 ${maxFrames()} 张，仅默认选中前 ${maxFrames()} 张`, 'warn', 5000);
    }
    $('#imagesPickbar').classList.remove('hidden');
    $('#imagesTotalCount').textContent = S.imagesItems.length;
    $('#optLabel').value = (files[0].name || '图片序列').replace(/\.[^.]+$/, '');
    renderImagesGrid();
    $('#imagesUploadText').textContent = `已上传 ${S.imagesItems.length} 张`;
    toast(`已上传 ${S.imagesItems.length} 张图片`, 'ok');
  } catch (err) {
    $('#imagesUploadText').textContent = `上传失败：${err.message}`;
    toast(`图片上传失败：${err.message}`, 'err', 6000);
  } finally {
    setTimeout(() => $('#imagesUploadStatus').classList.add('hidden'), 3000);
  }
}

function renderImagesGrid() {
  const host = $('#imagesGrid');
  host.innerHTML = '';
  S.imagesItems.forEach((it, i) => {
    const el = thumbEl(it.thumb, i + 1, it.name, it.selected);
    el.addEventListener('click', () => {
      it.selected = !it.selected;
      if (S.imagesItems.filter((x) => x.selected).length > maxFrames()) {
        it.selected = false; toast(`最多选择 ${maxFrames()} 个航点`, 'warn'); return;
      }
      el.classList.toggle('selected', it.selected);
      updateImagesSelCount();
    });
    host.appendChild(el);
  });
  updateImagesSelCount();
}

function updateImagesSelCount() {
  const n = S.imagesItems.filter((x) => x.selected).length;
  $('#imagesSelCount').textContent = n;
  updateModalHint();
}

function xhrUpload(url, fd, onProgress, onUploaded) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded, e.total); };
    xhr.upload.onload = () => { if (onUploaded) onUploaded(); };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText || '{}'); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error((data && (data.error || data.detail)) || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('网络错误'));
    xhr.send(fd);
  });
}

/* ------------------------------------------------------------------ submit */
async function submitRecon() {
  const n = updateModalHint();
  if (n < 2) { toast('请至少选择 2 个航点帧', 'warn'); return; }
  if (n > maxFrames()) { toast(`最多选择 ${maxFrames()} 个航点帧`, 'warn'); return; }

  let source = null;
  let defaultLabel = '';
  if (S.tab === 'video') {
    const idx = Array.from(S.videoSelIdx).sort((a, b) => a - b);
    if (!S.videoUploadId) { toast('请先上传视频', 'warn'); return; }
    source = { kind: 'video', upload_id: S.videoUploadId, indices: idx };
    defaultLabel = '视频片段';
  } else if (S.tab === 'images') {
    const paths = S.imagesItems.filter((c) => c.selected).map((c) => c.path);
    if (!paths.length) { toast('请先上传图片', 'warn'); return; }
    source = { kind: 'images', upload_id: S.imagesUploadId, paths };
    defaultLabel = '图片序列';
  } else {
    const paths = Array.from(S.frameSel.keys()).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    if (!paths.length) { toast('请先浏览并选择帧目录中的图片', 'warn'); return; }
    source = { kind: 'frames', paths, root: S.framePath };
    defaultLabel = baseName(S.framePath) || '帧目录';
  }

  const label = ($('#optLabel').value || '').trim() || defaultLabel;
  const params = {
    img_size: $('#optSize').value,
    max_points: Number($('#optPoints').value),
    conf_quantile: Number($('#optConf').value),
    seg_thresh: Number($('#optSegThresh').value),
    min_new_dist: Number($('#optMinDist').value),
    drop_sky: $('#optDropSky').checked,
    point_source: $('#optPointSource').value,
  };

  const btn = $('#modalSubmit');
  btn.disabled = true; btn.textContent = '提交中…';
  try {
    const res = await postJSON('/api/recon', { label, source, params });
    closeModal();
    toast(`已提交重建任务：${label}`, 'ok');
    watchJob(res.job_id);
  } catch (err) {
    toast(`提交失败：${err.message}`, 'err', 7000);
  } finally {
    btn.disabled = false; btn.textContent = '开始重建';
  }
}

/* ------------------------------------------------------------------- jobs */
function watchJob(jobId) {
  if (!jobId) return;
  clearTimeout(S.jobTimer);
  S.jobId = jobId;
  S.jobDone = false;
  const panel = $('#jobPanel');
  panel.classList.remove('hidden', 'done', 'error');
  $('#jobTitle').textContent = '重建中…';
  $('#jobMsg').textContent = '排队中';
  $('#jobBar').style.width = '0%';
  $('#jobLog').textContent = '';
  $('#jobFoot').textContent = '';
  $('#jobCancel').style.display = '';

  const poll = async () => {
    if (S.jobId !== jobId) return;
    try {
      const j = await fetchJSON(`/api/jobs/${jobId}`);
      $('#jobBar').style.width = `${clamp(j.pct, 0, 100)}%`;
      $('#jobMsg').textContent = `${Math.round(j.pct)}% · ${j.msg || j.stage || ''}`;
      $('#jobLog').textContent = (j.logs || []).join('\n');
      $('#jobLog').scrollTop = $('#jobLog').scrollHeight;
      $('#jobTitle').textContent = `${j.label} · ${j.clip_id}`;

      if (j.status === 'done') {
        S.jobDone = true;
        panel.classList.add('done');
        $('#jobCancel').style.display = 'none';
        $('#jobTitle').textContent = `✓ ${j.label} 重建完成`;
        $('#jobMsg').textContent = `${j.result?.waypoints || '?'} 航点 · ${fmt(j.result?.points || 0)} 点`;
        $('#jobFoot').textContent = `${j.clip_id}${j.gpu ? ' · GPU ' + j.gpu : ''}`;
        const next = await refreshClips(j.clip_id);
        if (next) loadClip(next);
        setTimeout(() => { if (S.jobDone) panel.classList.add('hidden'); }, 7000);
        return;
      }
      if (j.status === 'error') {
        panel.classList.add('error');
        panel.classList.add('open');
        $('#jobCancel').style.display = 'none';
        $('#jobTitle').textContent = `✕ ${j.label} 重建失败`;
        $('#jobMsg').textContent = j.error || '请查看日志';
        toast(`重建失败：${j.error || '未知错误'}`, 'err', 7000);
        return;
      }
      if (j.status === 'canceled') {
        panel.classList.add('error');
        $('#jobTitle').textContent = '任务已取消';
        $('#jobMsg').textContent = j.label;
        setTimeout(() => panel.classList.add('hidden'), 3000);
        return;
      }
      S.jobTimer = setTimeout(poll, 900);
    } catch (err) {
      $('#jobMsg').textContent = `查询任务失败：${err.message}`;
      S.jobTimer = setTimeout(poll, 2500);
    }
  };
  poll();
}

/* ---------------------------------------------------------- scene manager */
function sceneKindLabel(kind) {
  return { video: '视频', frames: '帧目录', images: '图片' }[kind] || '未知输入';
}

function renderScenes() {
  const host = $('#sceneList');
  if (!host) return;
  const clips = S.clips || [];
  $('#sceneCount').textContent = `${clips.length} 个场景`;
  host.innerHTML = '';
  if (!clips.length) {
    host.innerHTML = '<div class="scene-empty">还没有已重建的场景。点击右上角“新建重建”生成第一个场景。</div>';
    return;
  }
  clips.forEach((c) => {
    const id = String(c.id);
    const current = id === S.clipId;
    const src = c.source || {};
    const kind = c.source_kind || src.kind || '';
    const created = c.created ? String(c.created).replace('T', ' ').slice(0, 19) : '—';
    const p = c.params || {};
    const canRebuild = c.rebuildable !== false;
    const thumb = c.thumb
      ? `<img class="scene-thumb" loading="lazy" src="/clips/${encodeURIComponent(id)}/${String(c.thumb).split('/').map(encodeURIComponent).join('/')}" alt="">`
      : '<div class="scene-thumb placeholder">◈</div>';
    const sub = [
      p.img_size ? `${p.img_size}` : '',
      p.max_points ? `上限 ${fmt(p.max_points)} 点` : '',
      p.max_frames ? `限 ${fmt(p.max_frames)} 帧` : '',
      `来源 ${sceneKindLabel(kind)}`,
    ].filter(Boolean).join(' · ');
    const row = document.createElement('div');
    row.className = 'scene-row' + (current ? ' current' : '');
    row.dataset.id = id;
    row.innerHTML = `
      ${thumb}
      <div class="scene-main">
        <div class="scene-title">${escHtml(c.label || id)}${current ? '<span class="scene-current">当前</span>' : ''}<span class="dim">${escHtml(id)}</span></div>
        <div class="scene-meta">${fmt(c.num_waypoints)} 航点 · ${fmt(c.num_points)} 点 · ${escHtml(created)}</div>
        <div class="scene-sub">${escHtml(sub)}</div>
      </div>
      <div class="scene-actions">
        <button class="btn tiny" data-act="open">打开</button>
        <button class="btn tiny" data-act="rebuild"${canRebuild ? '' : ' disabled title="该场景未保留可用的原始输入"'}>重建</button>
        <button class="btn tiny danger" data-act="delete">删除</button>
      </div>`;
    host.appendChild(row);
  });
}

async function openScenes() {
  const modal = $('#scenesModal');
  try {
    const data = await fetchJSON('/api/clips');
    S.clips = data.clips || [];
    renderScenes();
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
  } catch (err) {
    toast(`场景列表加载失败：${err.message}`, 'err', 6000);
  }
}

function closeScenes() {
  const modal = $('#scenesModal');
  if (!modal) return;
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
}

async function rebuildScene(id, btn) {
  if (!id) return;
  let clip = (S.clips || []).find((c) => c.id === id);
  if (!clip && S.clipId === id && S.meta) {
    clip = { id, label: S.meta.label, source_kind: (S.meta.source || {}).kind, rebuildable: true };
  }
  if (!clip) { toast('场景不存在，请刷新列表', 'warn'); return; }
  if (clip.rebuildable === false) {
    toast('该场景没有可用的原始输入（例如早期图片场景未保存路径），无法重建', 'warn', 6500);
    return;
  }
  if (!confirm(`基于「${clip.label || id}」的原始输入和参数重新重建？\n会创建一个新场景，旧场景保留。`)) return;
  const oldText = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '提交中…'; }
  try {
    const res = await postJSON(`/api/clips/${encodeURIComponent(id)}/rebuild`, {});
    closeScenes();
    toast(`已提交重建任务：${res.label || clip.label || id}`, 'ok', 5000);
    watchJob(res.job_id);
  } catch (err) {
    toast(`重建失败：${err.message}`, 'err', 8000);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = oldText || '重建'; }
  }
}

async function deleteScene(id, btn) {
  if (!id) return;
  const clip = (S.clips || []).find((c) => c.id === id);
  const name = clip ? (clip.label || id) : id;
  if (!confirm(`确定删除场景「${name}」？\n点云缓存会被移除；原始上传/输入会保留。此操作不可恢复。`)) return;
  const oldText = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '删除中…'; }
  try {
    await fetchJSON(`/api/clips/${encodeURIComponent(id)}`, { method: 'DELETE' });
    const wasCurrent = S.clipId === id;
    if (wasCurrent) { S.meta = null; S.clipId = null; }
    const next = await refreshClips(wasCurrent ? undefined : S.clipId);
    if (wasCurrent) {
      if (next) await loadClip(next);
      else { disposeClipScene(); showEmpty(); }
    }
    renderScenes();
    toast('场景已删除', 'ok');
  } catch (err) {
    toast(`删除失败：${err.message}`, 'err', 6500);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = oldText || '删除'; }
  }
}

/* ----------------------------------------------------------------- export */
function screenshot() {
  if (!S.meta) { toast('请先加载片段', 'warn'); return; }
  renderScene();
  const src = canvas;
  const out = document.createElement('canvas');
  out.width = src.width; out.height = src.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(src, 0, 0);
  const s = Math.max(1, out.width / 1400);
  ctx.textBaseline = 'top';
  ctx.font = `600 ${Math.round(15 * s)}px -apple-system,"Segoe UI","Noto Sans SC",sans-serif`;
  ctx.fillStyle = 'rgba(6,10,16,.55)';
  const drawLabel = (text, x) => {
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(6,10,16,.55)';
    ctx.fillRect(x, 18 * s, tw + 22 * s, 30 * s);
    ctx.fillStyle = '#dbeaf8';
    ctx.fillText(text, x + 11 * s, 25 * s);
  };
  if (S.mode === 'both') {
    ctx.font = `600 ${Math.round(13 * s)}px -apple-system,"Segoe UI","Noto Sans SC",sans-serif`;
    drawLabel('RGB point cloud', 20 * s);
    drawLabel('Semantic point cloud', out.width / 2 + 20 * s);
  } else {
    drawLabel(S.mode === 'seg' ? 'Semantic point cloud' : 'RGB point cloud', 20 * s);
  }
  ctx.font = `600 ${Math.round(19 * s)}px -apple-system,"Segoe UI","Noto Sans SC",sans-serif`;
  ctx.fillStyle = 'rgba(6,10,16,.55)';
  const title = S.meta.label || S.clipId;
  const sub = `waypoint ${(S.progress + 1).toFixed(1)} / ${S.K}   ·   ${fmt(S.meta.num_points)} pts`;
  const tw2 = Math.max(ctx.measureText(title).width, ctx.measureText(sub).width) + 28 * s;
  const bh = 62 * s;
  const bx = out.width - tw2 - 20 * s;
  ctx.fillRect(bx, out.height - bh - 20 * s, tw2, bh);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(title, bx + 12 * s, out.height - bh - 6 * s);
  ctx.font = `400 ${Math.round(12.5 * s)}px -apple-system,"Segoe UI","Noto Sans SC",sans-serif`;
  ctx.fillStyle = '#9fb6cd';
  ctx.fillText(sub, bx + 12 * s, out.height - bh + 24 * s);
  out.toBlob((blob) => {
    if (!blob) { toast('截图失败', 'err'); return; }
    downloadBlob(blob, `${S.clipId}_wp${(S.progress + 1).toFixed(1)}.png`);
    toast('截图已保存', 'ok', 2200);
  }, 'image/png');
}

function startRecording() {
  if (!S.meta) { toast('请先加载片段', 'warn'); return; }
  if (!window.MediaRecorder) { toast('当前浏览器不支持 MediaRecorder 录制', 'err'); return; }
  if (!canvas.captureStream) { toast('当前浏览器不支持 canvas 录制', 'err'); return; }
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9' : 'video/webm';
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 16000000 });
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    const blob = new Blob(chunks, { type: 'video/webm' });
    downloadBlob(blob, `${S.clipId}_route.webm`);
    toast(`录制完成（${fmtMB(blob.size)} MB）`, 'ok', 3500);
    stream.getTracks().forEach((t) => t.stop());
  };
  S.prev = {
    follow: S.follow, playing: S.playing, progress: S.progress, mode: S.mode,
    rgbPip: S.rgbPip, autoRotate: controls.autoRotate, autoRotateSpeed: controls.autoRotateSpeed,
    camPos: camera.position.clone(), camTarget: controls.target.clone(),
    highlight: S.highlight, pointSize: S.pointSize,
  };
  if (!S.rgbPip) { S.rgbPip = true; $('#rgbPip').checked = true; }
  S.record = { rec, chunks };
  S.recStart = performance.now();
  $('#recBadge').classList.remove('hidden');
  $('#btnRecord').classList.add('playing');

  if (S.recFollow) {
    setFollow(true, true);
    toast('开始录制：跟随航线（含 RGB 帧画中画）', '', 3000);
  } else {
    setFollow(false, true);
    fitPreset('oblique');
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.2;
    toast('开始录制：全景环绕（含 RGB 帧画中画）', '', 3000);
  }
  layoutPip();
  setProgress(0, { snap: true });
  S.playing = true;
  setPlaying(true, true);
  rec.start(200);
}

function stopRecording(auto = false) {
  if (!S.record) return;
  const { rec } = S.record;
  S.record = null;
  $('#recBadge').classList.add('hidden');
  $('#btnRecord').classList.remove('playing');
  try { if (rec.state !== 'inactive') rec.stop(); } catch (_) { /* ignore */ }
  S.playing = false;
  setPlaying(false, true);
  if (S.prev) {
    const p = S.prev;
    if (p.mode && p.mode !== S.mode) applyMode(p.mode);
    S.highlight = p.highlight;
    $('#highlight').checked = S.highlight;
    uniforms.uHighlight.value = S.highlight ? 1 : 0;
    S.pointSize = p.pointSize;
    uniforms.uPointSize.value = S.pointSize;
    $('#pointSize').value = S.pointSize;
    $('#pointSizeVal').textContent = S.pointSize.toFixed(1);
    S.rgbPip = p.rgbPip;
    $('#rgbPip').checked = S.rgbPip;
    layoutPip();
    // restore the camera the user had before recording
    S.follow = p.follow;
    $('#follow').checked = S.follow;
    controls.enabled = !S.follow;
    controls.autoRotate = p.autoRotate;
    controls.autoRotateSpeed = p.autoRotateSpeed;
    camera.position.copy(p.camPos);
    controls.target.copy(p.camTarget);
    controls.update();
    S.followReady = false;
    if (!auto) setProgress(p.progress, { snap: true });
    S.prev = null;
  }
}

/* ----------------------------------------------------------------- events */
function bindEvents() {
  // Null-safe binding lets an older cached HTML shell still boot after a
  // partial page update; the new controls simply stay absent until refresh.
  const on = (sel, ev, fn) => { const el = $(sel); if (el) el.addEventListener(ev, fn); return el; };
  $('#clipSelect').addEventListener('change', (e) => { if (e.target.value) loadClip(e.target.value); });
  $('#btnNew').addEventListener('click', () => openModal('video'));
  on('#btnScenes', 'click', openScenes);
  on('#scenesClose', 'click', closeScenes);
  on('#scenesModal', 'click', (e) => { if (e.target.id === 'scenesModal') closeScenes(); });
  on('#sceneRefresh', 'click', openScenes);
  on('#btnRebuildClip', 'click', () => { if (S.clipId) rebuildScene(S.clipId); });
  on('#sceneList', 'click', (e) => {
    const btn = e.target.closest('button[data-act]');
    const row = e.target.closest('.scene-row');
    if (!btn || !row) return;
    const id = row.dataset.id;
    const act = btn.dataset.act;
    if (act === 'open') { closeScenes(); loadClip(id); }
    else if (act === 'rebuild') rebuildScene(id, btn);
    else if (act === 'delete') deleteScene(id, btn);
  });
  $('#emptyNew').addEventListener('click', () => openModal('video'));
  $('#btnHelp').addEventListener('click', () => $('#helpModal').classList.remove('hidden'));
  $('#btnGallery').addEventListener('click', openGallery);
  $('#galleryClose').addEventListener('click', closeGallery);
  $('#galleryModal').addEventListener('click', (e) => { if (e.target.id === 'galleryModal') closeGallery(); });
  $('#rgbPip').addEventListener('change', (e) => { S.rgbPip = e.target.checked; layoutPip(); });
  $('#helpClose').addEventListener('click', () => $('#helpModal').classList.add('hidden'));
  $('#helpModal').addEventListener('click', (e) => { if (e.target.id === 'helpModal') $('#helpModal').classList.add('hidden'); });

  $('#btnScreenshot').addEventListener('click', screenshot);
  $('#recFollow').addEventListener('change', (e) => { S.recFollow = e.target.checked; });
  $('#btnRecord').addEventListener('click', () => { S.record ? stopRecording() : startRecording(); });
  $('#recBadge').addEventListener('click', () => stopRecording());

  $('#btnPlay').addEventListener('click', () => setPlaying(!S.playing));
  $('#progress').addEventListener('input', (e) => {
    setPlaying(false);
    setProgress(Number(e.target.value), { snap: true, force: true });
  });
  $('#speed').addEventListener('input', (e) => {
    S.speed = Number(e.target.value);
    $('#speedText').textContent = `${S.speed.toFixed(1)}×`;
  });
  $('#follow').addEventListener('change', (e) => setFollow(e.target.checked));
  $('#loop').addEventListener('change', (e) => { S.loop = e.target.checked; });
  $('#highlight').addEventListener('change', (e) => {
    S.highlight = e.target.checked;
    uniforms.uHighlight.value = S.highlight ? 1 : 0;
  });
  $('#btnResetView').addEventListener('click', () => {
    setFollow(false, true);
    fitPreset('oblique');
    toast('视角已重置', '', 1600);
  });

  $$('#modeSeg button').forEach((b) => b.addEventListener('click', () => applyMode(b.dataset.mode)));
  $$('#viewPresets button').forEach((b) => b.addEventListener('click', () => {
    const preset = b.dataset.preset;
    if (preset === 'orbit') { setFollow(false, true); }
    else { setFollow(false, true); }
    fitPreset(preset);
  }));

  $('#pointSize').addEventListener('input', (e) => {
    S.pointSize = Number(e.target.value);
    uniforms.uPointSize.value = S.pointSize;
    $('#pointSizeVal').textContent = S.pointSize.toFixed(1);
  });
  $('#confMin').addEventListener('input', (e) => {
    S.confMin = Number(e.target.value);
    uniforms.uConfMin.value = S.confMin;
    $('#confMinVal').textContent = `${Math.round(S.confMin * 100)}%`;
    updateFilteredDisplay();
  });
  $('#probMin').addEventListener('input', (e) => {
    S.probMin = Number(e.target.value);
    uniforms.uProbMin.value = S.probMin;
    $('#probMinVal').textContent = `${Math.round(S.probMin * 100)}%`;
    updateFilteredDisplay();
  });

  $('#classAll').addEventListener('click', () => { for (let i = 1; i < 12; i++) setClassVisible(i, true); });
  $('#classNone').addEventListener('click', () => { for (let i = 1; i < 12; i++) setClassVisible(i, false); });

  $('#btnDeleteClip').addEventListener('click', () => { if (S.clipId) deleteScene(S.clipId); });

  // minimap scrubbing
  const mm = $('#minimap');
  let dragging = false;
  const seekMap = (e) => {
    if (!S.map || !S.meta) return;
    const rect = mm.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * S.map.w;
    const y = (e.clientY - rect.top) / rect.height * S.map.h;
    const wx = S.map.fromX(x), wz = S.map.fromY(y);
    let best = 0, bd = Infinity;
    S.meta.route.forEach((p, i) => {
      const d = (p[0] - wx) ** 2 + (p[2] - wz) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    setPlaying(false);
    setProgress(best / (S.meta.route.length - 1) * (S.K - 1), { snap: true, force: true });
  };
  mm.addEventListener('pointerdown', (e) => { dragging = true; mm.setPointerCapture(e.pointerId); seekMap(e); });
  mm.addEventListener('pointermove', (e) => { if (dragging) seekMap(e); });
  mm.addEventListener('pointerup', () => { dragging = false; });

  // modal
  $('#modalClose').addEventListener('click', closeModal);
  $('#modalCancel').addEventListener('click', closeModal);
  $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  $$('#srcTabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
  $('#modalSubmit').addEventListener('click', submitRecon);

  // video
  $('#videoInput').addEventListener('change', (e) => { if (e.target.files[0]) uploadVideo(e.target.files[0]); });
  $('#imagesInput').addEventListener('change', (e) => uploadImages(e.target.files));
  bindDrop('#videoDrop', '#videoInput', (files) => uploadVideo(files[0]));
  bindDrop('#imagesDrop', '#imagesInput', (files) => uploadImages(files));

  on('#optSize', 'change', () => { updateVideoSummary(); updateModalHint(); });
  $('#videoPickEven').addEventListener('click', () => {
    if (!S.videoCands.length) { toast('请先上传视频', 'warn'); return; }
    const total = S.videoTotal || S.videoCands.length;
    let n = Math.round(Number($('#videoPickN').value) || 16);
    if (n > maxFrames()) { toast(`最多选择 ${maxFrames()} 帧`, 'warn'); n = maxFrames(); }
    if (n > total) { toast(`视频共 ${total} 帧，已按全帧选择`, 'warn'); n = total; }
    n = clamp(n, 2, Math.max(2, total));
    if (!S.videoTotal) {
      // Fallback for a probe without nb_frames: use the preview pool.
      S.videoSelIdx = new Set(selectEvenly(S.videoCands, Math.min(n, S.videoCands.length)).map((c) => c.idx));
    } else {
      S.videoSelIdx = new Set(evenIndices(S.videoTotal, n));
    }
    renderVideoCandidates();
    toast(`已均匀选择 ${S.videoSelIdx.size} 帧`, 'ok');
  });
  $('#videoPickNone').addEventListener('click', () => {
    S.videoSelIdx.clear();
    renderVideoCandidates();
  });
  $('#imagesPickAll').addEventListener('click', () => {
    S.imagesItems.forEach((it, i) => { it.selected = i < maxFrames(); });
    if (S.imagesItems.length > maxFrames()) toast(`最多选择 ${maxFrames()} 张`, 'warn');
    renderImagesGrid();
  });
  $('#imagesPickNone').addEventListener('click', () => {
    S.imagesItems.forEach((it) => { it.selected = false; });
    renderImagesGrid();
  });
  $('#frameGo').addEventListener('click', () => browseFrames($('#framePath').value.trim()));
  $('#framePath').addEventListener('keydown', (e) => { if (e.key === 'Enter') browseFrames($('#framePath').value.trim()); });
  $('#framePickAll').addEventListener('click', () => {
    let added = 0;
    for (const it of S.frameItems) {
      if (S.frameSel.size >= maxFrames()) break;
      if (!S.frameSel.has(it.path)) { S.frameSel.set(it.path, it); added++; }
    }
    renderFrameImages();
    if (added && S.frameSel.size >= maxFrames()) toast(`已选满 ${maxFrames()} 个航点`, 'warn');
  });
  $('#framePickNone').addEventListener('click', () => { S.frameSel.clear(); renderFrameImages(); });

  // job panel
  $('#jobToggle').addEventListener('click', () => {
    $('#jobPanel').classList.toggle('open');
    $('#jobToggle').textContent = $('#jobPanel').classList.contains('open') ? '⌄' : '⌃';
  });
  $('#jobCancel').addEventListener('click', async () => {
    if (!S.jobId) return;
    try { await postJSON(`/api/jobs/${S.jobId}/cancel`, {}); toast('已请求取消任务', '', 2500); }
    catch (err) { toast(`取消失败：${err.message}`, 'err'); }
  });

  // keyboard
  window.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || e.target.isContentEditable) return;
    if (e.key === 'Escape') {
      closeModal();
      closeGallery();
      $('#helpModal').classList.add('hidden');
    }
    if (!$('#modal').classList.contains('hidden')) {
      return;
    }
    if (!$('#galleryModal').classList.contains('hidden') ||
        !$('#helpModal').classList.contains('hidden')) {
      return;
    }
    if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); if (S.meta) setPlaying(!S.playing); }
    else if (e.key === 'ArrowLeft' && S.meta) {
      setPlaying(false);
      setProgress(Math.max(0, Math.floor(S.progress + 1e-6) - (e.shiftKey ? 0 : 1)), { snap: true, force: true });
    } else if (e.key === 'ArrowRight' && S.meta) {
      setPlaying(false);
      setProgress(Math.min(S.K - 1, Math.floor(S.progress + 1e-6) + 1), { snap: true, force: true });
    } else if (e.key === '1') applyMode('rgb');
    else if (e.key === '2') applyMode('seg');
    else if (e.key === '3') applyMode('both');
    else if (e.key === 'f' || e.key === 'F') setFollow(!S.follow);
    else if (e.key === 'r' || e.key === 'R') { setFollow(false, true); fitPreset('oblique'); }
    else if (e.key === '?') $('#helpModal').classList.remove('hidden');
  });

  // double click: pick a point / ground plane as orbit target
  canvas.addEventListener('dblclick', (e) => {
    if (!S.meta) return;
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    ray.params.Points = { threshold: S.sceneInfo.radius * 0.008 };
    let hit = null;
    try {
      const hits = ray.intersectObject(S.pointMesh, false);
      if (hits.length) hit = hits[0].point.clone();
    } catch (_) { /* ignore */ }
    if (!hit) {
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(S.sceneInfo.center.y));
      hit = new THREE.Vector3();
      if (!ray.ray.intersectPlane(plane, hit)) return;
    }
    setFollow(false, true);
    controls.target.copy(hit);
    controls.update();
    toast('旋转中心已更新', '', 1400);
  });
}

function bindDrop(zoneSel, inputSel, handler) {
  const zone = $(zoneSel);
  const input = $(inputSel);
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag');
    const files = e.dataTransfer.files;
    if (files && files.length) handler(files);
  });
}

/* ------------------------------------------------------------------- init */
async function init() {
  bindEvents();
  resize();

  try {
    const [cfg, clipsData, health] = await Promise.all([
      fetchJSON('/api/config'), fetchJSON('/api/clips'), fetchJSON('/api/health').catch(() => null),
    ]);
    S.config = cfg;
    window.__DEMO_CONNECTED = true;
    if (health) {
      const ok = health.ckpt_exists && health.backbone_exists;
      $('#statusDot').classList.toggle('ok', !!ok);
      $('#statusDot').classList.toggle('bad', !ok);
      $('#statusDot').title = ok ? (health.gpu === 'auto' ? 'GPU 自动选择 · 模型就绪' : `GPU ${health.gpu} · 模型就绪`) : '模型权重缺失';
    }
    $('#optSize').value = cfg.default_params.img_size;
    $('#optPoints').value = String(cfg.default_params.max_points);
    $('#optConf').value = cfg.default_params.conf_quantile;
    $('#optSegThresh').value = cfg.default_params.seg_thresh;
    $('#optMinDist').value = cfg.default_params.min_new_dist;
    $('#optDropSky').checked = !!cfg.default_params.drop_sky;
    $('#optPointSource').value = cfg.default_params.point_source || 'depth';
    if (cfg.limits && cfg.limits.max_frames) {
      $('#videoPickN').max = String(cfg.limits.max_frames);
      if (Number($('#videoPickN').value) > cfg.limits.max_frames) {
        $('#videoPickN').value = String(cfg.limits.max_frames);
      }
    }
    if (cfg.public_mode) {
      $('#btnDeleteClip').classList.add('hidden');
    }
    renderPresets();
    if (cfg.presets && cfg.presets.length) {
      $('#framePath').value = cfg.presets[0].path;
      browseFrames(cfg.presets[0].path);
    }

    // reconnect to a reconstruction that was already running (e.g. after refresh)
    try {
      const jobs = await fetchJSON('/api/jobs');
      const active = (jobs.jobs || []).find((j) => j.status === 'queued' || j.status === 'running');
      if (active) watchJob(active.id);
    } catch (_) { /* ignore */ }

    const next = await refreshClips(parseHashClip() || undefined);
    if (next) {
      const hp = parseHashProgress();
      await loadClip(next, { progress: hp });
    } else {
      showEmpty();
    }
  } catch (err) {
    console.error(err);
    showLoadingError(err.message);
    toast(`初始化失败：${err.message}`, 'err', 6000);
  }
}

function showEmpty() {
  if (S.meta) return;
  $('#empty').classList.remove('hidden');
  hideLoading();
  $('#chipClip').textContent = '未加载场景';
  $('#chipPoints').textContent = '—';
}

window.__DEMO = { S, scene, camera, controls, uniforms, renderer, THREE, pipCanvas, pipTexture, pipMesh, overlayScene, overlayCamera };

init();
