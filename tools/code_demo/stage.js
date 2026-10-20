import * as THREE from 'three';

const W = 1920, H = 1080;
const clip = new URLSearchParams(location.search).get('clip') || 'city2-lite';
const clipBase = `/make_demo/data/clips/${clip}`;
const canvas = document.querySelector('#view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.setClearColor(0x060a10, 1);
renderer.autoClear = false;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W / H, 0.002, 100);
const gRoute = new THREE.Group();
const gAux = new THREE.Group();
scene.add(gRoute, gAux);

let meta, palette, mesh, geom, uniforms, mat;
let routePts = [], routeTans = [], travelLine, currentMarker, drone, ring, pillar, grid;
let K = 0, N = 0, prefix = [];

function vec(a) { return new THREE.Vector3(a[0], a[1], a[2]); }
function display(v) { return [v[0], v[1], v[2]]; }

async function load() {
  meta = await (await fetch(`${clipBase}/meta.json`)).json();
  palette = await (await fetch('/code_demo/palette.json')).json();
  const buf = await (await fetch(`${clipBase}/points.bin`)).arrayBuffer();
  N = meta.num_points; K = meta.num_waypoints;
  prefix = [0];
  (meta.counts || []).forEach(c => prefix.push(prefix[prefix.length - 1] + c));

  geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, 0, 3 * N), 3));
  geom.setAttribute('aColor', new THREE.BufferAttribute(new Uint8Array(buf, 12 * N, 3 * N), 3, true));
  const cls = new Uint8Array(buf, 15 * N, N);
  const seg = new Uint8Array(3 * N);
  const pal = palette.rgb_hex;
  for (let i = 0, k = 0; i < N; i++, k += 3) {
    const c = pal[cls[i]] || pal[0];
    seg[k] = c[0]; seg[k + 1] = c[1]; seg[k + 2] = c[2];
  }
  geom.setAttribute('aSeg', new THREE.BufferAttribute(seg, 3, true));
  geom.setAttribute('aCls', new THREE.BufferAttribute(cls, 1, false));
  geom.setAttribute('aWp', new THREE.BufferAttribute(new Uint8Array(buf, 16 * N, N), 1, false));
  geom.setAttribute('aRank', new THREE.BufferAttribute(new Uint8Array(buf, 17 * N, N), 1, true));

  const lo = vec(meta.bounds[0]), hi = vec(meta.bounds[1]);
  const center = lo.clone().add(hi).multiplyScalar(0.5);
  const radius = Math.max(1e-4, hi.clone().sub(lo).length() * 0.5);
  window.__scene = { lo, hi, center, radius };

  uniforms = {
    uMode: { value: 0 }, uProgress: { value: 0 }, uPointSize: { value: 3.8 },
    uPixelRatio: { value: 1 }, uNewGlow: { value: 0.0 },
    uSplitPx: { value: W * 0.5 }, uWidth: { value: W },
  };
  const vert = /* glsl */`
    attribute vec3 aColor; attribute vec3 aSeg; attribute float aCls; attribute float aWp; attribute float aRank;
    uniform float uProgress; uniform float uPointSize; uniform float uPixelRatio;
    varying vec3 vRgb; varying vec3 vSeg; varying float vNew;
    void main(){
      float k = floor(uProgress + 1e-5);
      float f = clamp(uProgress - k, 0.0, 1.0);
      bool vis = (aWp < k + 0.5) || (abs(aWp - (k + 1.0)) < 0.5 && aRank <= f);
      if(!vis){ gl_Position = vec4(2.0,2.0,2.0,1.0); gl_PointSize=0.0; vRgb=vec3(0.0); vSeg=vec3(0.0); vNew=0.0; return; }
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = uPointSize * uPixelRatio * clamp(2.6 / max(0.2, -mv.z), 0.6, 1.55);
      vRgb = clamp(pow(aColor, vec3(0.82)) * 1.06, 0.0, 1.0);
      vSeg = aSeg;
      vNew = (aWp > k + 0.5) ? 1.0 : 0.0;
    }`;
  const frag = /* glsl */`
    uniform int uMode; uniform float uSplitPx; uniform float uNewGlow;
    varying vec3 vRgb; varying vec3 vSeg; varying float vNew;
    void main(){
      vec2 d = gl_PointCoord - vec2(0.5);
      if(dot(d,d) > 0.25) discard;
      vec3 c;
      if(uMode == 0) c = vRgb;
      else if(uMode == 1) c = vSeg;
      else c = (gl_FragCoord.x < uSplitPx) ? vRgb : vSeg;
      if(uNewGlow > 0.0 && vNew > 0.5) c = mix(c, vec3(0.35, 0.95, 1.0), uNewGlow);
      gl_FragColor = vec4(c, 1.0);
    }`;
  mat = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
  mesh = new THREE.Points(geom, mat);
  mesh.frustumCulled = false;
  scene.add(mesh);

  // route
  routePts = meta.route.map(vec);
  routeTans = routePts.map((p, i) => {
    const a = routePts[Math.max(0, i - 1)], b = routePts[Math.min(routePts.length - 1, i + 1)];
    return b.clone().sub(a).normalize();
  });
  const fullGeo = new THREE.BufferGeometry().setFromPoints(routePts);
  gRoute.add(new THREE.Line(fullGeo, new THREE.LineBasicMaterial({ color: 0x33475e, transparent: true, opacity: 0.9 })));
  const travelGeo = new THREE.BufferGeometry().setFromPoints(routePts);
  travelLine = new THREE.Line(travelGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8 }));
  travelLine.geometry.setDrawRange(0, 0);
  gRoute.add(travelLine);

  // waypoint ticks
  const dotGeo = new THREE.SphereGeometry(radius * 0.006, 8, 6);
  meta.waypoints.forEach((w) => {
    const m = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0x7e93aa }));
    m.position.copy(vec(w.cam)); gRoute.add(m);
  });

  // current marker / drone
  drone = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.012, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd166 }));
  gRoute.add(drone);
  ring = new THREE.Mesh(new THREE.RingGeometry(radius * 0.03, radius * 0.036, 48),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2; gRoute.add(ring);
  const pillarGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, radius * 0.12, 0)]);
  pillar = new THREE.Line(pillarGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.45 }));
  gRoute.add(pillar);

  // grid
  const gsize = Math.max(hi.x - lo.x, hi.z - lo.z) * 1.35;
  grid = new THREE.GridHelper(gsize, 26, 0x1d3b55, 0x122232);
  grid.material.transparent = true; grid.material.opacity = 0.38;
  grid.position.set(center.x, lo.y - radius * 0.01, center.z);
  scene.add(grid);

  window.__ready = true;
}

function routeSample(p) {
  const t = K > 1 ? Math.min(1, Math.max(0, p / (K - 1))) : 0;
  const f = t * (routePts.length - 1);
  const i0 = Math.floor(f), i1 = Math.min(routePts.length - 1, i0 + 1), a = f - i0;
  const pos = new THREE.Vector3().lerpVectors(routePts[i0], routePts[i1], a);
  const tan = new THREE.Vector3().lerpVectors(routeTans[i0], routeTans[i1], a).normalize();
  return { pos, tan };
}

function draw(mode, split) {
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, W, H);
  renderer.clear(true, true, true);
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
  uniforms.uMode.value = (mode === 'split') ? 2 : ((mode === 'seg') ? 1 : 0);
  uniforms.uSplitPx.value = Math.max(1, Math.min(W - 1, split * W));
  renderer.render(scene, camera);
}

window.setFrame = async (f) => {
  if (!window.__ready) return false;
  if (f.camera) {
    camera.position.set(...f.camera.pos);
    camera.fov = f.camera.fov || 45;
    camera.up.set(0, 1, 0);
    camera.lookAt(new THREE.Vector3(...f.camera.target));
    camera.updateProjectionMatrix();
  }
  uniforms.uProgress.value = f.progress ?? 0;
  uniforms.uPointSize.value = f.pointSize ?? 3.8;
  uniforms.uNewGlow.value = f.newGlow ?? 0;
  gRoute.visible = f.showRoute !== false;
  grid.visible = f.showGrid !== false;
  if (gRoute.visible) {
    const p = uniforms.uProgress.value;
    const t = K > 1 ? p / (K - 1) : 0;
    travelLine.geometry.setDrawRange(0, p <= 0 ? 0 : Math.max(2, Math.round(t * (routePts.length - 1)) + 1));
    const s = routeSample(p);
    drone.position.copy(s.pos);
    ring.position.set(s.pos.x, (window.__scene.lo.y - window.__scene.radius * 0.01), s.pos.z);
    pillar.position.copy(s.pos);
  }
  draw(f.mode || 'rgb', f.split ?? 0.5);
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  return true;
};

window.__clipInfo = () => ({ N, K, bounds: meta.bounds, route: meta.route.length, name: meta.label });
load().catch(e => { console.error(e); window.__error = String(e); });
