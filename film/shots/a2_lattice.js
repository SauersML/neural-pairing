// a2 helpers: the unfolded model (a lattice of residual-stream columns), instanced curved rays, and glow points.
// Geometry matches the creatures exactly: a column is a spine laid straight, block i is centred at height
// (i + 0.5) / 36 * H, the exchange after block b sits at block b's port ring. Heights are given in "u" units (0..1).
import * as THREE from 'three';
import { C } from '../lib/palette.js';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { tokenTile } from '../lib/tokens.js';
import { clamp, lerp } from '../lib/util.js';

export const H = 4.2;
export const UX = XBLOCKS.map((b) => (b + 0.5) / NBLOCKS);
const BINS = 192;
const TEAL = C.link;

// ---------------------------------------------------------------- glow points (per-point size and colour)
export class Glows {
  constructor(scene, max = 512, pxScale = 1) {
    this.max = max; this.n = 0;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(max * 3), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(max * 3), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(max), 1).setUsage(THREE.DynamicDrawUsage));
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uMap: { value: Creature.dotTexture() }, uScale: { value: 540 * pxScale } },
      vertexShader: `attribute vec3 aColor; attribute float aSize; uniform float uScale; varying vec3 vC;
        void main(){ vC = aColor; vec4 mv = modelViewMatrix*vec4(position,1.0); gl_PointSize = min(900.0, aSize*uScale/max(0.05,-mv.z)); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform sampler2D uMap; varying vec3 vC; void main(){ float a = texture2D(uMap, gl_PointCoord).a; gl_FragColor = vec4(vC*a, 1.0); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.pts = new THREE.Points(g, this.mat); this.pts.frustumCulled = false; this.pts.renderOrder = 5; scene.add(this.pts);
  }
  begin() { this.n = 0; }
  add(p, color, size = 0.3, amt = 1) {
    if (this.n >= this.max || amt <= 0.002 || size <= 0) return;
    const i = this.n++, g = this.pts.geometry.attributes;
    g.position.array[i * 3] = p.x; g.position.array[i * 3 + 1] = p.y; g.position.array[i * 3 + 2] = p.z;
    g.aColor.array[i * 3] = color.r * amt; g.aColor.array[i * 3 + 1] = color.g * amt; g.aColor.array[i * 3 + 2] = color.b * amt;
    g.aSize.array[i] = size;
  }
  end() {
    const g = this.pts.geometry.attributes;
    for (let i = this.n; i < this.max; i++) g.aSize.array[i] = 0;
    g.position.needsUpdate = g.aColor.needsUpdate = g.aSize.needsUpdate = true;
    this.pts.geometry.setDrawRange(0, Math.max(1, this.n));
  }
}

// ---------------------------------------------------------------- curved rays: instanced quadratic-bezier tubes
// add(a, c, b, {width, color, alpha, g0, g1, pulse, pulseColor, pulseAmt, taper})
export class Rays {
  constructor(scene, max = 600, { segs = 24, radial = 6 } = {}) {
    this.max = max; this.n = 0;
    const base = new THREE.BufferGeometry();
    const uv = [], idx = [];
    for (let i = 0; i <= segs; i++) for (let j = 0; j <= radial; j++) uv.push(i / segs, (j / radial) * Math.PI * 2);
    for (let i = 0; i < segs; i++) for (let j = 0; j < radial; j++) { const a = i * (radial + 1) + j, b = a + radial + 1; idx.push(a, b, a + 1, b, b + 1, a + 1); }
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(uv.length / 2 * 3), 3));
    g.setAttribute('aUV', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    const ia = (n) => new THREE.InstancedBufferAttribute(new Float32Array(max * n), n).setUsage(THREE.DynamicDrawUsage);
    this.A = { iA: ia(3), iC: ia(3), iB: ia(3), iCol: ia(4), iP: ia(4), iPC: ia(4), iX: ia(1) };
    for (const k in this.A) g.setAttribute(k, this.A[k]);
    g.instanceCount = 0; void base;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: `attribute vec2 aUV; attribute vec3 iA, iC, iB; attribute vec4 iCol, iP, iPC; attribute float iX;
        varying float vU; varying vec4 vCol; varying vec4 vP; varying vec4 vPC; varying float vEdge;
        void main(){ float u = aUV.x; vU = u; vCol = iCol; vP = iP; vPC = iPC;
          vec3 P = mix(mix(iA,iC,u), mix(iC,iB,u), u);
          vec3 T = 2.0*(1.0-u)*(iC-iA) + 2.0*u*(iB-iC); T = normalize(T + vec3(1e-6));
          vec3 ref = abs(T.y) < 0.9 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0);
          vec3 N = normalize(cross(T, ref)); vec3 B = cross(T, N);
          vec3 d = N*cos(aUV.y) + B*sin(aUV.y);
          float taper = mix(1.0, 0.45 + 0.55*sin(3.14159*u), iX);
          vec3 pos = P + d * iP.x * taper;
          vec4 mv = modelViewMatrix*vec4(pos,1.0);
          vEdge = abs(normalize(normalMatrix*d).z);
          gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `varying float vU; varying vec4 vCol; varying vec4 vP; varying vec4 vPC; varying float vEdge;
        void main(){ if (vU < vP.y || vU > vP.z) discard;
          float head = smoothstep(vP.z - 0.05, vP.z, vU) * step(vP.z, 0.995);
          float pul = exp(-pow((vU - vP.w)/0.07, 2.0));
          vec3 c = vCol.rgb * vCol.a * (0.35 + 0.65*vEdge) * (1.0 + 1.6*head) + vPC.rgb * pul * vPC.a;
          gl_FragColor = vec4(c, 1.0); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.mesh = new THREE.Mesh(g, this.mat); this.mesh.frustumCulled = false; this.mesh.renderOrder = 4; scene.add(this.mesh);
    this.g = g;
  }
  begin() { this.n = 0; }
  add(a, c, b, o = {}) {
    const alpha = o.alpha ?? 1; if (this.n >= this.max || alpha <= 0.003) return;
    const i = this.n++, A = this.A;
    A.iA.array.set([a.x, a.y, a.z], i * 3); A.iC.array.set([c.x, c.y, c.z], i * 3); A.iB.array.set([b.x, b.y, b.z], i * 3);
    const col = o.color || TEAL; A.iCol.array.set([col.r, col.g, col.b, alpha], i * 4);
    A.iP.array.set([o.width ?? 0.012, o.g0 ?? 0, o.g1 ?? 1, o.pulse ?? -5], i * 4);
    const pc = o.pulseColor || C.white;
    A.iPC.array.set([pc.r, pc.g, pc.b, o.pulse === undefined ? 0 : (o.pulseAmt ?? 1.5)], i * 4);
    A.iX.array[i] = o.taper ?? 0;
  }
  end() { this.g.instanceCount = this.n; for (const k in this.A) this.A[k].needsUpdate = true; this.mesh.visible = this.n > 0; }
}

// ---------------------------------------------------------------- one model, unfolded
// columns: [{text, z (set per frame), appear, wave (u), cut...}] built from a fixed list of token texts.
export class Lattice {
  constructor(scene, side, x, texts, { tileH = 0.42 } = {}) {
    this.side = side; this.x = x; this.sgn = Math.sign(x);
    this.col = side === 'quill' ? C.quill : C.ember; this.hi = side === 'quill' ? C.quillHi : C.emberHi;
    this.group = new THREE.Group(); scene.add(this.group);
    const NC = this.NC = texts.length;
    this.cols = texts.map((text, i) => ({ i, text, z: 0, appear: 0, wave: 0, front: 0, discR: 0.15, writes: [], amp: 1 }));
    // info texture: BINS samples along each column
    this.info = new Float32Array(BINS * NC * 4);
    this.tex = new THREE.DataTexture(this.info, BINS, NC, THREE.RGBAFormat, THREE.FloatType);
    this.tex.magFilter = this.tex.minFilter = THREE.LinearFilter; this.tex.needsUpdate = true;
    this.maxTint = 0.62;
    this.state = { amp: 1, cutY: 0.5, cutW: 2, cutDim: 1, rows: 1, rowExtra: 0, discs: 1, tiles: 1 };
    // residual-stream cores
    const cg = new THREE.CylinderGeometry(0.03, 0.03, H, 10, 1, true); cg.translate(0, H / 2, 0);
    this.cores = this.cols.map((c, i) => {
      const m = new THREE.Mesh(cg, new THREE.ShaderMaterial({
        uniforms: { uTex: { value: this.tex }, uRow: { value: (i + 0.5) / NC }, uCol: { value: this.col.clone().lerp(this.hi, 0.45) }, uAmp: { value: 1 }, uWave: { value: 0 }, uFront: { value: 0 }, uTime: { value: 0 }, uCut: { value: new THREE.Vector3(0.5, 2, 1) } },
        vertexShader: `varying float vU; void main(){ vU = position.y / ${H.toFixed(3)}; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
        fragmentShader: `uniform sampler2D uTex; uniform float uRow, uAmp, uWave, uFront, uTime; uniform vec3 uCol, uCut; varying float vU;
          void main(){ vec4 inf = texture2D(uTex, vec2(vU, uRow));
            float flow = 0.72 + 0.28*sin(vU*110.0 - uTime*7.0);
            float below = smoothstep(uWave + 0.004, uWave - 0.004, vU);
            vec3 c = uCol * flow * mix(0.13, 0.52, below);
            float a = clamp(inf.a, 0.0, 1.0);
            vec3 tint = inf.rgb / max(inf.a, 1e-3);
            c = mix(c, tint * flow * 1.15, a);
            c += uCol * exp(-pow((vU - uWave)*45.0, 2.0)) * uFront * 1.2;
            float cut = mix(1.0, uCut.z, smoothstep(uCut.y, uCut.y + 0.012, abs(vU - uCut.x)));
            gl_FragColor = vec4(c * uAmp * cut, 1.0); }`,
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      m.renderOrder = 2; this.group.add(m); return m;
    });
    // blocks: a faint glass puck and a crisp ring per (column, block); exchange blocks get the teal port ring
    const N = NC * NBLOCKS;
    this.pucks = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 28, 1, false), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), N);
    this.rings = new THREE.InstancedMesh(new THREE.TorusGeometry(1, 0.05, 6, 40), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), N);
    for (const m of [this.pucks, this.rings]) { m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3); m.frustumCulled = false; m.renderOrder = 1; this.group.add(m); }
    // wall row lines (one per block, spanning the columns)
    this.rowsM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), NBLOCKS);
    this.rowsM.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NBLOCKS * 3), 3); this.rowsM.frustumCulled = false; this.group.add(this.rowsM);
    // token tiles under the columns
    this.tiles = this.cols.map((c) => { const t = tokenTile(c.text, side, tileH); t.renderOrder = 10; t.material.fog = false; this.group.add(t); return t; });
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)); this._s = new THREE.Vector3(); this._p = new THREE.Vector3();
    this._c = new THREE.Color();
  }
  node(i, u, out = new THREE.Vector3()) { return out.set(this.x, u * H, this.cols[i].z); }
  begin() { for (const c of this.cols) c.writes.length = 0; }
  // a write added at height u0 (the partner's colour), visible from u0 up to `top` (it rises with the wave)
  write(i, u0, color, amt, top, flash = 0) { this.cols[i].writes.push({ u0, color, amt, top, flash }); }

  end(t, camera) {
    const S = this.state, NC = this.NC, c3 = this._c;
    this.info.fill(0);
    let zMin = 1e9, zMax = -1e9, any = false;
    for (const c of this.cols) {
      const on = c.appear * S.amp; const core = this.cores[c.i];
      core.visible = on > 0.004; core.position.set(this.x, 0, c.z);
      if (core.visible) { any = true; zMin = Math.min(zMin, c.z); zMax = Math.max(zMax, c.z); }
      const U = core.material.uniforms; U.uAmp.value = on * c.amp; U.uWave.value = c.wave; U.uFront.value = c.front; U.uTime.value = t;
      U.uCut.value.set(S.cutY, S.cutW, S.cutDim);
      for (const w of c.writes) {
        for (let b = 0; b < BINS; b++) {
          const u = (b + 0.5) / BINS; if (u < w.u0 - 0.01 || u > w.top + 0.004) continue;
          const k = clamp((u - w.u0 + 0.01) / 0.012) * clamp((w.top + 0.004 - u) / 0.008) * w.amt;
          const o = (c.i * BINS + b) * 4;
          this.info[o] += w.color.r * k; this.info[o + 1] += w.color.g * k; this.info[o + 2] += w.color.b * k; this.info[o + 3] += k;
        }
      }
      if (c.writes.length) for (let b = 0; b < BINS; b++) {
        const o = (c.i * BINS + b) * 4, a = this.info[o + 3]; if (a <= 1e-4) continue;
        const capA = Math.min(a, this.maxTint), f = capA / a;
        this.info[o] *= f; this.info[o + 1] *= f; this.info[o + 2] *= f; this.info[o + 3] = capA;
      }
      // blocks
      for (let i = 0; i < NBLOCKS; i++) {
        const idx = c.i * NBLOCKS + i, ub = (i + 0.5) / NBLOCKS, isX = XBLOCKS.includes(i);
        const done = ub < c.wave ? 1 : 0, fr = Math.exp(-Math.pow((ub - c.wave) * 26, 2)) * c.front;
        const cut = Math.abs(ub - S.cutY) > S.cutW ? S.cutDim : 1;
        const r = c.discR * (isX ? 1.25 : 1) * (0.92 + 0.08 * Math.sin(Math.PI * (0.1 + 0.82 * ub)));
        this._p.set(this.x, ub * H, c.z);
        this._s.set(r, H / NBLOCKS * 0.34, r); this._m.compose(this._p, new THREE.Quaternion(), this._s); this.pucks.setMatrixAt(idx, this._m);
        this._s.set(r, r, r * 1.6); this._m.compose(this._p, this._q, this._s); this.rings.setMatrixAt(idx, this._m);
        const vis = on * S.discs * cut;
        c3.copy(this.col).multiplyScalar(vis * (0.05 + 0.07 * done + 0.18 * fr)); this.pucks.setColorAt(idx, c3);
        if (isX) c3.copy(TEAL).multiplyScalar(vis * (0.42 + 0.25 * done + 0.5 * fr + S.rowExtra * 0.5));
        else c3.copy(this.hi).multiplyScalar(vis * (0.08 + 0.09 * done + 0.45 * fr));
        this.rings.setColorAt(idx, c3);
      }
      // token tile
      const tl = this.tiles[c.i];
      const tk = on * S.tiles * (c.tile ?? 1); tl.visible = tk > 0.01; tl.position.set(this.x, -0.4, c.z); tl.quaternion.copy(camera.quaternion);
      tl.material.opacity = clamp(tk * 1.2); tl.scale.setScalar(0.6 + 0.4 * clamp(c.appear * 1.5));
    }
    this.tex.needsUpdate = true;
    this.pucks.instanceMatrix.needsUpdate = this.rings.instanceMatrix.needsUpdate = true;
    this.pucks.instanceColor.needsUpdate = this.rings.instanceColor.needsUpdate = true;
    // row lines between the first and last visible columns
    this.rowsM.visible = any && S.rows > 0.003;
    if (any) {
      const z0 = zMin - 0.35, z1 = zMax + 0.35, wv = Math.max(...this.cols.map((c) => (c.appear > 0.5 ? c.wave : 0)));
      for (let i = 0; i < NBLOCKS; i++) {
        const ub = (i + 0.5) / NBLOCKS, isX = XBLOCKS.includes(i);
        const cut = Math.abs(ub - S.cutY) > S.cutW ? S.cutDim : 1;
        this._p.set(this.x, ub * H, (z0 + z1) / 2); this._s.set(isX ? 0.016 : 0.008, isX ? 0.016 : 0.008, z1 - z0);
        this._m.compose(this._p, new THREE.Quaternion(), this._s); this.rowsM.setMatrixAt(i, this._m);
        const done = ub < wv ? 1 : 0;
        if (isX) c3.copy(TEAL).multiplyScalar(S.rows * S.amp * cut * (0.35 + 0.15 * done + S.rowExtra));
        else c3.copy(this.col).multiplyScalar(S.rows * S.amp * cut * (0.07 + 0.06 * done));
        this.rowsM.setColorAt(i, c3);
      }
      this.rowsM.instanceMatrix.needsUpdate = true; this.rowsM.instanceColor.needsUpdate = true;
    }
  }
}
