// Helpers for S1 (s1_meet): hex force-field shader, deterministic tile shatter, soft particle systems,
// DOM callouts with leader lines, and a time-keyed camera spline. Everything is a pure function of t.
import * as THREE from 'three';
import { clamp, smooth, hash } from '../lib/util.js';
import { el, FONT } from '../lib/hud.js';

// ---------------------------------------------------------------- textures
let _flare = null;
export function flareTexture() {
  if (_flare) return _flare;
  const c = document.createElement('canvas'); c.width = 256; c.height = 256; const x = c.getContext('2d');
  const g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.08, 'rgba(255,255,255,0.9)'); g.addColorStop(0.25, 'rgba(255,255,255,0.25)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 256, 256);
  // anamorphic streaks
  x.globalCompositeOperation = 'lighter';
  for (const [w, h, a] of [[256, 6, 0.55], [256, 2, 0.9], [4, 120, 0.35]]) {
    const gg = x.createRadialGradient(128, 128, 0, 128, 128, Math.max(w, h) / 2);
    gg.addColorStop(0, `rgba(255,255,255,${a})`); gg.addColorStop(1, 'rgba(255,255,255,0)');
    x.save(); x.translate(128, 128); x.scale(w / Math.max(w, h), h / Math.max(w, h)); x.translate(-128, -128);
    x.fillStyle = gg; x.beginPath(); x.arc(128, 128, Math.max(w, h) / 2, 0, Math.PI * 2); x.fill(); x.restore();
  }
  _flare = new THREE.CanvasTexture(c); _flare.colorSpace = THREE.SRGBColorSpace; return _flare;
}

// ---------------------------------------------------------------- soft particles
// N points with per-point color, size (world units) and alpha. Call set(...) for each, then commit().
export class Particles {
  constructor(n, pxScale, { additive = true } = {}) {
    this.n = n;
    const g = new THREE.BufferGeometry();
    this.pos = new Float32Array(n * 3); this.col = new Float32Array(n * 3); this.size = new Float32Array(n); this.alpha = new Float32Array(n);
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));
    g.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1));
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uPx: { value: pxScale } },
      vertexShader: `attribute float aSize; attribute float aAlpha; varying vec3 vC; varying float vA; uniform float uPx;
        void main(){ vC = color; vA = aAlpha; vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv;
          gl_PointSize = max(1.5, aSize*uPx/max(0.1,-mv.z)); if (aAlpha < 0.002) gl_PointSize = 0.0; }`,
      fragmentShader: `varying vec3 vC; varying float vA;
        void main(){ vec2 d = gl_PointCoord-0.5; float r2 = dot(d,d)*4.0; if (r2>1.0) discard;
          float a = exp(-r2*4.5) + 0.6*exp(-r2*28.0); gl_FragColor = vec4(vC*a*vA, 1.0); }`,
      vertexColors: true, transparent: true, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.points = new THREE.Points(g, this.mat); this.points.frustumCulled = false;
  }
  set(i, p, c, size, alpha) {
    this.pos[i * 3] = p.x; this.pos[i * 3 + 1] = p.y; this.pos[i * 3 + 2] = p.z;
    this.col[i * 3] = c.r; this.col[i * 3 + 1] = c.g; this.col[i * 3 + 2] = c.b;
    this.size[i] = size; this.alpha[i] = Math.max(0, alpha);
  }
  hide(i) { this.alpha[i] = 0; }
  commit() { const a = this.points.geometry.attributes; a.position.needsUpdate = a.color.needsUpdate = a.aSize.needsUpdate = a.aAlpha.needsUpdate = true; }
}

// ---------------------------------------------------------------- hexagonal force field
// A curved glass wall built from hex cells. Local frame: x across (-W/2..W/2), y up (0..H), +z = the side facing Quill.
export class Shield {
  constructor({ W = 4.2, H = 5.2 } = {}) {
    this.W = W; this.H = H;
    const geo = new THREE.PlaneGeometry(W, H, 70, 80); geo.translate(0, H / 2, 0);
    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        uT: { value: 0 }, uOn: { value: -1 }, uAge: { value: -1 }, uOff: { value: 0 }, uAmp: { value: 1 },
        uImp: { value: new THREE.Vector2(0, 3.6) }, uHalf: { value: new THREE.Vector2(W / 2, H / 2) },
        uCol: { value: new THREE.Color('#8fb0ff') }, uCol2: { value: new THREE.Color('#c39bff') }, uHot: { value: new THREE.Color('#ffd98a') },
      },
      vertexShader: `uniform float uAge; uniform vec2 uImp; uniform vec2 uHalf; varying vec2 vL; varying vec3 vN; varying vec3 vW;
        void main(){ vec3 p = position; float bx = p.x/uHalf.x;
          p.z += 0.38*(1.0 - bx*bx);
          float r = length(p.xy - uImp);
          float rip = 0.0, dent = 0.0;
          if (uAge > 0.0) {
            rip = 0.16*exp(-uAge*2.2)*sin(r*5.5 - uAge*18.0)*exp(-r*0.45)*smoothstep(uAge*4.6+0.4, uAge*4.6-0.6, r);
            dent = -0.32*exp(-uAge*7.0)*exp(-r*r*1.6);
          }
          p.z += rip + dent;
          vL = position.xy;
          vec4 w = modelMatrix*vec4(p,1.0); vW = w.xyz;
          vN = normalize(mat3(modelMatrix)*normalize(vec3(0.76*p.x/(uHalf.x*uHalf.x), 0.0, 1.0)));
          gl_Position = projectionMatrix*viewMatrix*w; }`,
      fragmentShader: `uniform float uT, uOn, uAge, uOff, uAmp; uniform vec2 uImp, uHalf; uniform vec3 uCol, uCol2, uHot;
        varying vec2 vL; varying vec3 vN; varying vec3 vW;
        float hd(vec2 p){ p = abs(p); return max(dot(p, normalize(vec2(1.0,1.732))), p.x); }
        float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
        void main(){
          float S = 2.3;
          vec2 uv = vL*S; vec2 rr = vec2(1.0,1.732); vec2 hh = rr*0.5;
          vec2 a = mod(uv, rr)-hh; vec2 b = mod(uv-hh, rr)-hh; vec2 gv = dot(a,a) < dot(b,b) ? a : b;
          vec2 id = uv - gv; vec2 cc = id/S; float rnd = h21(floor(id*10.0+0.5));
          float ed = 0.5 - hd(gv);
          vec2 q = (cc - vec2(0.0, uHalf.y))/uHalf;
          float em = length(q) + (rnd-0.5)*0.16;
          float mask = smoothstep(1.0, 0.86, em) * smoothstep(0.25, 1.3, cc.y);
          float dImp = length(cc - uImp);
          float ta = dImp*0.075 + rnd*0.05;
          float on = smoothstep(ta, ta+0.05, uOn);
          float pop = uOn > ta ? exp(-(uOn-ta)*9.0) : 0.0;
          float off = smoothstep(rnd*0.8, rnd*0.8+0.2, uOff);
          float edge = 1.0 - smoothstep(0.0, 0.045, ed);
          float inner = 1.0 - smoothstep(0.045, 0.11, ed);
          vec3 V = normalize(cameraPosition - vW); vec3 N = normalize(vN);
          float fr = pow(1.0 - abs(dot(V,N)), 2.5);
          float r = length(vL - uImp);
          float ring = 0.0, flare = 0.0, hit = 0.0;
          if (uAge > 0.0) {
            ring = exp(-pow((r - uAge*4.6)/0.16, 2.0))*exp(-uAge*2.8);
            flare = exp(-r*4.0)*exp(-uAge*6.0);
            hit = exp(-pow((dImp - uAge*4.6)/0.4, 2.0))*exp(-uAge*2.6);
          }
          float sweep = exp(-pow((vL.x*0.6 + vL.y*0.8 - mod(uT*2.2, 12.0) + 3.0)/0.5, 2.0));
          vec3 base = mix(uCol, uCol2, rnd*0.6 + 0.2*sin(vL.y*1.3 + uT));
          float lum = edge*(0.42 + 1.1*pop + 0.9*hit + 0.35*sweep) + inner*0.10 + 0.035 + fr*0.16 + pop*0.12;
          vec3 col = base*lum*on*(1.0-off) + uHot*(ring*0.8 + flare*0.35 + hit*edge*0.8)*(1.0-off*0.7);
          gl_FragColor = vec4(col*mask*uAmp, 1.0); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geo, this.mat); this.mesh.frustumCulled = false;
    this.group = new THREE.Group(); this.group.add(this.mesh);
  }
  // local wall point (x across, y up) -> world, on the curved surface
  localToWorld(x, y, out = new THREE.Vector3()) {
    const bx = x / (this.W / 2);
    this.group.updateMatrixWorld(true);
    return out.set(x, y, 0.38 * (1 - bx * bx)).applyMatrix4(this.group.matrixWorld);
  }
  normalWorld() { return new THREE.Vector3(0, 0, 1).applyQuaternion(this.group.quaternion); }
}

// ---------------------------------------------------------------- shatter
// Breaks a textured tile (w x h) into jittered triangles. Pieces are placed from a world-space frame at the impact
// moment and then fly ballistically (with drag) as a pure function of time since impact.
export class Shatter {
  constructor(texture, w, h, { nx = 9, ny = 3, seed = 7 } = {}) {
    this.w = w; this.h = h;
    const P = []; // jittered grid points in tile-local units
    for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
      const edgeX = i === 0 || i === nx, edgeY = j === 0 || j === ny;
      const jx = edgeX ? 0 : (hash(seed + i * 7.3 + j * 13.1) - 0.5) * 0.7, jy = edgeY ? 0 : (hash(seed + i * 3.7 + j * 5.9 + 9) - 0.5) * 0.7;
      P.push([((i + jx) / nx - 0.5) * w, ((j + jy) / ny - 0.5) * h]);
    }
    const id = (i, j) => j * (nx + 1) + i;
    this.tris = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      const flip = hash(seed + i * 1.7 + j * 9.3) > 0.5;
      const tt = flip ? [[a, b, c], [a, c, d]] : [[a, b, d], [b, c, d]];
      for (const t of tt) this.tris.push(t.map((k) => P[k]));
    }
    const n = this.tris.length; this.n = n;
    const pos = new Float32Array(n * 9), uv = new Float32Array(n * 6), alpha = new Float32Array(n * 3), heat = new Float32Array(n * 3);
    this.tris.forEach((tri, k) => tri.forEach((p, v) => { uv[k * 6 + v * 2] = p[0] / w + 0.5; uv[k * 6 + v * 2 + 1] = p[1] / h + 0.5; }));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('aAlpha', new THREE.BufferAttribute(alpha, 1)); g.setAttribute('aHeat', new THREE.BufferAttribute(heat, 1));
    this.mat = new THREE.ShaderMaterial({
      uniforms: { map: { value: texture }, uHot: { value: new THREE.Color('#ffd166') } },
      vertexShader: `attribute float aAlpha; attribute float aHeat; varying vec2 vUv; varying float vA; varying float vH;
        void main(){ vUv = uv; vA = aAlpha; vH = aHeat; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
      fragmentShader: `uniform sampler2D map; uniform vec3 uHot; varying vec2 vUv; varying float vA; varying float vH;
        void main(){ vec4 c = texture2D(map, vUv); vec3 col = c.rgb + uHot*vH*0.7; float a = c.a*vA;
          if (a < 0.003) discard; gl_FragColor = vec4(col, a); }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    });
    this.mesh = new THREE.Mesh(g, this.mat); this.mesh.frustumCulled = false;
    // per-shard random dynamics
    this.dyn = this.tris.map((tri, k) => {
      const cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3, cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3;
      return { cx, cy, r1: hash(k * 3.1 + seed), r2: hash(k * 5.7 + seed + 1), r3: hash(k * 9.3 + seed + 2), r4: hash(k * 1.9 + seed + 3) };
    });
  }
  // frame: {origin, right, up, normal(back toward thrower)} at impact; age = t - impact
  update(age, frame, { fadeStart = 0.45, fadeEnd = 1.35 } = {}) {
    const pos = this.mesh.geometry.attributes.position.array, al = this.mesh.geometry.attributes.aAlpha.array, ht = this.mesh.geometry.attributes.aHeat.array;
    this.mesh.visible = age >= 0 && age < fadeEnd;
    if (!this.mesh.visible) return;
    const { origin, right, up, normal } = frame;
    const fwd = normal.clone(); // pieces bounce back toward the thrower, and spray out radially
    const drag = 2.6, G = -3.2;
    const fall = (1 - Math.exp(-drag * age)) / drag;
    const tmp = new THREE.Vector3(), c = new THREE.Vector3(), axis = new THREE.Vector3(), q = new THREE.Quaternion();
    this.centers = [];
    this.dyn.forEach((d, k) => {
      // impact is on the tile's leading (right) edge: pieces near it are hit hardest
      const hx = d.cx / this.w + 0.5; // 0 left .. 1 right
      const radial = new THREE.Vector3().addScaledVector(right, d.cx * 2.2 + (d.r3 - 0.5) * 0.6).addScaledVector(up, d.cy * 3.0 + (d.r4 - 0.3) * 1.2);
      const v = radial.multiplyScalar(2.0 + 2.8 * d.r1).addScaledVector(fwd, 2.0 + 4.0 * d.r2 * (0.4 + hx));
      c.copy(origin).addScaledVector(right, d.cx).addScaledVector(up, d.cy).addScaledVector(v, fall);
      c.y += 0.5 * G * age * age * 0.45;
      axis.set(d.r1 - 0.5, d.r2 - 0.5, d.r3 - 0.5).normalize();
      q.setFromAxisAngle(axis, (4 + 10 * d.r4) * fall * 1.6);
      const sh = 1 - smooth((age - fadeStart * (0.6 + 0.8 * d.r2)) / (fadeEnd - fadeStart)) ;
      const a = clamp(sh * 1.15);
      const shrink = 0.35 + 0.65 * sh;
      this.tris[k].forEach((p, vi) => {
        tmp.set(0, 0, 0).addScaledVector(right, (p[0] - d.cx) * shrink).addScaledVector(up, (p[1] - d.cy) * shrink).applyQuaternion(q).add(c);
        pos[k * 9 + vi * 3] = tmp.x; pos[k * 9 + vi * 3 + 1] = tmp.y; pos[k * 9 + vi * 3 + 2] = tmp.z;
        al[k * 3 + vi] = a; ht[k * 3 + vi] = Math.exp(-age * 10) * 0.9 + (1 - sh) * 0.7;
      });
      this.centers.push({ p: c.clone(), alive: a });
    });
    const at = this.mesh.geometry.attributes; at.position.needsUpdate = at.aAlpha.needsUpdate = at.aHeat.needsUpdate = true;
  }
}

// ---------------------------------------------------------------- DOM callout with leader line
// dot at a projected 3D point, a thin line to an offset label. draw: 0..1 (line grows, then text reveals).
export class Callout {
  constructor(parent, html, { color = '#f5f3ff', size = 22, font = FONT.mono, weight = 500, side = 'right', sub = '', subColor = 'rgba(245,243,255,0.75)', subSize = 18 } = {}) {
    this.side = side; this.color = color;
    this.root = el(parent, 'div', { position: 'absolute', left: 0, top: 0, width: '1920px', height: '1080px', pointerEvents: 'none' });
    this.line = el(this.root, 'div', { position: 'absolute', height: '1.5px', background: color, transformOrigin: '0 50%', boxShadow: `0 0 8px ${color}`, opacity: 0.85 });
    this.dot = el(this.root, 'div', { position: 'absolute', width: '10px', height: '10px', marginLeft: '-5px', marginTop: '-5px', borderRadius: '50%', background: color, boxShadow: `0 0 14px ${color}, 0 0 4px #fff` });
    this.ring = el(this.root, 'div', { position: 'absolute', width: '26px', height: '26px', marginLeft: '-13px', marginTop: '-13px', borderRadius: '50%', border: `1.5px solid ${color}`, opacity: 0.6 });
    this.lab = el(this.root, 'div', { position: 'absolute', whiteSpace: 'nowrap', textShadow: '0 1px 14px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.8)' });
    this.main = el(this.lab, 'div', { font: `${weight} ${size}px ${font}`, color, letterSpacing: font === FONT.mono ? '0.06em' : '0.01em' }, html);
    if (sub) this.sub = el(this.lab, 'div', { font: `500 ${subSize}px ${FONT.mono}`, color: subColor, marginTop: '4px', letterSpacing: '0.06em' }, sub);
  }
  // a: {x,y} screen anchor; off: {x,y} label offset in px; o: opacity; draw 0..1
  update(a, off, o = 1, draw = 1) {
    const vis = o > 0.002 && a && !a.behind;
    this.root.style.display = vis ? 'block' : 'none'; if (!vis) return;
    this.root.style.opacity = o;
    const lp = clamp(draw / 0.55), tp = clamp((draw - 0.4) / 0.6);
    const ex = a.x + off.x, ey = a.y + off.y;
    const len = Math.hypot(off.x, off.y) * lp, ang = Math.atan2(off.y, off.x);
    Object.assign(this.line.style, { left: a.x + 'px', top: a.y + 'px', width: len + 'px', transform: `rotate(${ang}rad)` });
    Object.assign(this.dot.style, { left: a.x + 'px', top: a.y + 'px', transform: `scale(${0.4 + 0.6 * clamp(draw * 4)})` });
    const rp = clamp(draw * 1.6);
    Object.assign(this.ring.style, { left: a.x + 'px', top: a.y + 'px', transform: `scale(${0.6 + 1.4 * rp})`, opacity: 0.7 * (1 - rp) });
    const right = this.side === 'right';
    Object.assign(this.lab.style, {
      left: ex + 'px', top: ey + 'px', opacity: tp,
      transform: `translate(${right ? 10 : -10}px, -50%) translate(${right ? '0%' : '-100%'}, 0) translateX(${(1 - tp) * (right ? -14 : 14)}px)`,
      textAlign: right ? 'left' : 'right', filter: `blur(${(1 - tp) * 6}px)`,
    });
  }
}

// ---------------------------------------------------------------- camera spline (non-uniform Catmull-Rom in time)
export function makeSpline(keys) {
  const n = keys.length;
  const tan = (i, f) => {
    const a = keys[Math.max(0, i - 1)], b = keys[Math.min(n - 1, i + 1)];
    const dt = b.t - a.t || 1; return f(b).clone().sub(f(a)).multiplyScalar(1 / dt);
  };
  return (t, f) => {
    if (t <= keys[0].t) return f(keys[0]).clone();
    if (t >= keys[n - 1].t) { const m = tan(n - 1, f); return f(keys[n - 1]).clone().addScaledVector(m, t - keys[n - 1].t); }
    let i = 0; while (i < n - 2 && t > keys[i + 1].t) i++;
    const k0 = keys[i], k1 = keys[i + 1], h = k1.t - k0.t, s = (t - k0.t) / h;
    const m0 = tan(i, f).multiplyScalar(h), m1 = tan(i + 1, f).multiplyScalar(h);
    const s2 = s * s, s3 = s2 * s;
    return f(k0).clone().multiplyScalar(2 * s3 - 3 * s2 + 1).addScaledVector(m0, s3 - 2 * s2 + s).addScaledVector(f(k1), -2 * s3 + 3 * s2).addScaledVector(m1, s3 - s2);
  };
}
