// a2, one exchange read out (23.75–31.25): the states after block 15 lift out of both lattices into two glass panels
// that face the camera side by side, at the same moment: "Quill reads Ember" (left) and "Ember reads Quill" (right).
// Each panel follows the paper's mechanism figure (figures/link_mechanism.png):
//   partner positions (real tokens) -> their hidden states (bar codes) -> key (a tag, pointing down) and value (a
//   capsule) per position, in the sender's colour; the sender's gate is a gauge beside each value and scales it.
//   The reader's query (a tag pointing up, the reader's colour) sits at the reader's own newest position, on the
//   reader's residual stream. Beams in the reader's colour go to every visible key and to the reader's null key
//   ("nothing"); their thickness is the attention weight. The gated values flow, weighted, into one sum; the sum is
//   scaled by the write size and added (+) into the reader's stream, where the partner's colour rises.
// Everything is a pure function of t. Layout is designed in screen pixels on the panel plane and converted to world.
import * as THREE from 'three';
import { C, CSS } from '../lib/palette.js';
import { Pin, FONT } from '../lib/hud.js';
import { tokenTile } from '../lib/tokens.js';
import { clamp, lerp, prog, win, hash, easeOutBack, easeOutCubic, easeInOutCubic, easeInCubic } from '../lib/util.js';

export const TR = {
  lift: 24.75, cards: 24.875, kv: 25.125, gates: 25.375, labKV: [25.5, 26.25],
  q: 26.25, labQ: [26.375, 27.0],
  beams: 27.0, weights: 27.25, labA: [27.625, 28.5],
  flow: 28.5, sum: 28.875, labS: [29.0, 29.375],
  scale: 29.375, labW: [29.625, 30.25],
  plunge: 30.25, hit: 30.5, labR: [30.625, 31.125], end: 31.0,
};

const HEX = { quill: '#8b5cf6', quillHi: '#c4b5fd', ember: '#f97316', emberHi: '#fdba74', link: '#14e3a8', linkHi: '#7ff7d2' };
// ---------------------------------------------------------------- card textures
const texCache = new Map();
function rr(x, a, b, w, h, r) { x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }
// kind: state | key | query | value | sum | null | nullval | gate
function cardTex(kind, col, bars = [], o = {}) {
  const key = JSON.stringify([kind, col, bars.map((b) => b.toFixed(2)), o]);
  if (texCache.has(key)) return texCache.get(key);
  if (kind === 'gate') {
    const c = document.createElement('canvas'); c.width = 64; c.height = 192; const x = c.getContext('2d');
    rr(x, 8, 8, 48, 176, 22); x.fillStyle = 'rgba(16,12,30,0.92)'; x.fill(); x.lineWidth = 7; x.strokeStyle = col; rr(x, 8, 8, 48, 176, 22); x.stroke();
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.userData = { aspect: 64 / 192 }; texCache.set(key, t); return t;
  }
  const W = 256, Hh = kind === 'key' || kind === 'query' ? 256 : 208;
  const c = document.createElement('canvas'); c.width = W; c.height = Hh; const x = c.getContext('2d');
  const line = 9;
  x.lineJoin = 'round';
  const path = () => {
    if (kind === 'key') { x.beginPath(); x.moveTo(20, 16); x.lineTo(W - 20, 16); x.lineTo(W - 20, 176); x.lineTo(W / 2, 240); x.lineTo(20, 176); x.closePath(); }
    else if (kind === 'query') { x.beginPath(); x.moveTo(W / 2, 16); x.lineTo(W - 20, 80); x.lineTo(W - 20, 240); x.lineTo(20, 240); x.lineTo(20, 80); x.closePath(); }
    else if (kind === 'value' || kind === 'nullval') rr(x, 14, 30, W - 28, Hh - 60, (Hh - 60) / 2);
    else rr(x, 16, 16, W - 32, Hh - 32, 30);
  };
  const dashed = kind === 'null' || kind === 'nullval';
  path();
  x.fillStyle = kind === 'value' ? 'rgba(40,24,30,0.9)' : kind === 'sum' ? 'rgba(4,40,32,0.92)' : 'rgba(16,12,30,0.92)';
  if (!dashed) x.fill();
  x.setLineDash(dashed ? [22, 16] : []);
  x.lineWidth = kind === 'state' ? 6 : line; x.strokeStyle = col; x.globalAlpha = kind === 'state' ? 0.55 : 1; path(); x.stroke(); x.globalAlpha = 1; x.setLineDash([]);
  if (kind === 'nullval') { x.font = '700 110px Inter'; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillStyle = col; x.fillText('0', W / 2, Hh / 2 + 6); }
  // bar code
  if (bars.length) {
    const n = bars.length, bw = kind === 'value' ? 16 : 18, gap = kind === 'value' ? 9 : 8, tot = n * bw + (n - 1) * gap;
    const x0 = (W - tot) / 2, base = kind === 'key' ? 168 : kind === 'query' ? 214 : kind === 'value' ? Hh - 52 : Hh - 44, hmax = kind === 'key' ? 120 : kind === 'query' ? 120 : kind === 'value' ? 96 : 120;
    x.fillStyle = o.barCol || col;
    bars.forEach((b, i) => { const h = Math.max(6, b * hmax); rr(x, x0 + i * (bw + gap), base - h, bw, h, 4); x.fill(); });
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; t.userData = { aspect: W / Hh };
  texCache.set(key, t); return t;
}
const barsOf = (seed, n = 7) => Array.from({ length: n }, (_, i) => 0.18 + 0.82 * hash(seed * 13.7 + i * 3.1));

export class Readout {
  // L = {quill: Lattice, ember: Lattice}; spec as in a2_link (cols newest first, gate, w, w0); words per side (cols idx)
  constructor(scene, hud, L, spec, refCam) {
    this.L = L; this.spec = spec;
    this.root = new THREE.Group(); scene.add(this.root);
    // place the panel plane in front of the reference camera, facing it
    const Dp = 6.0; this.S = 0.86 * 2 * Math.tan(THREE.MathUtils.degToRad(refCam.fov / 2)) * Dp / 1080;
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(refCam.quaternion);
    this.root.position.copy(refCam.position).addScaledVector(fwd, Dp); this.root.quaternion.copy(refCam.quaternion);
    this.halves = {};
    const mkMat = (tex) => new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide });
    const plane = (tex, wpx, parent) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mkMat(tex)); m.userData.w = wpx; m.userData.h = wpx / tex.userData.aspect; parent.add(m); return m; };
    for (const [reader, sender, sideSign] of [['quill', 'ember', -1], ['ember', 'quill', 1]]) {
      const g = new THREE.Group(); this.root.add(g);
      const rc = reader === 'quill' ? HEX.quill : HEX.ember, rhi = reader === 'quill' ? HEX.quillHi : HEX.emberHi;
      const sc = sender === 'quill' ? HEX.quill : HEX.ember, shi = sender === 'quill' ? HEX.quillHi : HEX.emberHi;
      const sp = spec[reader], psp = spec[sender];
      const cols = psp.cols.slice().reverse();                 // partner positions, oldest -> newest
      const H = { g, reader, sender, sideSign, rc, rhi, sc, shi, sp, psp, cols };
      // glass backing
      { const cv = document.createElement('canvas'); cv.width = 940; cv.height = 1000; const x = cv.getContext('2d');
        rr(x, 6, 6, 928, 988, 44); const gr = x.createLinearGradient(0, 0, 0, 1000); gr.addColorStop(0, 'rgba(14,10,30,0.86)'); gr.addColorStop(1, 'rgba(6,5,14,0.9)'); x.fillStyle = gr; x.fill();
        x.lineWidth = 4; x.strokeStyle = rc; x.globalAlpha = 0.55; rr(x, 6, 6, 928, 988, 44); x.stroke();
        const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; H.back = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mkMat(t)); H.back.renderOrder = 1; g.add(H.back); }
      // partner columns
      H.slots = cols.map((ci, j) => {
        const k = psp.cols.length - 1 - j;                    // index in newest-first arrays
        const sb = barsOf((sender === 'quill' ? 100 : 200) + ci);
        const vb = barsOf((sender === 'quill' ? 300 : 400) + ci), kb = barsOf((sender === 'quill' ? 500 : 600) + ci);
        const word = this.L[sender].cols[ci].text;
        const tok = tokenTile(word, sender, 0.3); tok.material.toneMapped = false; tok.renderOrder = 6; g.add(tok);
        const st = plane(cardTex('state', shi, sb, { barCol: sc }), 80, g);
        const ky = plane(cardTex('key', sc, kb), 76, g);
        const vl = plane(cardTex('value', shi, vb), 78, g);
        const gt = plane(cardTex('gate', shi), 22, g);
        const fill = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: new THREE.Color(sc), transparent: true, depthWrite: false, toneMapped: false })); g.add(fill);
        const wbar = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.linkHi), transparent: true, depthWrite: false, toneMapped: false })); g.add(wbar);
        for (const m of [st, ky, vl, gt, fill, wbar]) m.renderOrder = 6;
        return { ci, k, tok, st, ky, vl, gt, fill, wbar, w: sp.w[k], gate: psp.gate[k], vb };
      });
      // null key (reader's) and its empty value
      H.nul = plane(cardTex('null', rhi), 76, g); H.nulV = plane(cardTex('nullval', rhi), 78, g);
      H.nulW = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.linkHi), transparent: true, depthWrite: false, toneMapped: false })); g.add(H.nulW);
      for (const m of [H.nul, H.nulV, H.nulW]) m.renderOrder = 6;
      // the reader: its stream (a vertical column), its state, its query, the sum, write size and the plus
      const rb = barsOf((reader === 'quill' ? 700 : 800) + sp.reader);
      H.rState = plane(cardTex('state', rhi, rb, { barCol: rc }), 80, g);
      H.query = plane(cardTex('query', rhi, barsOf((reader === 'quill' ? 900 : 950) + 1), { barCol: rc }), 96, g);
      H.rTok = tokenTile(this.L[reader].cols[sp.reader].text, reader, 0.3); H.rTok.material.toneMapped = false; H.rTok.renderOrder = 6; g.add(H.rTok);
      // the sum: the weighted, gated average of the values
      const tot = cols.reduce((s, _, j) => s + H.slots[j].w * H.slots[j].gate, 0);
      const sumBars = Array.from({ length: 7 }, (_, i) => H.slots.reduce((s, sl) => s + sl.w * sl.gate * sl.vb[i], 0) / Math.max(1e-3, tot));
      H.sum = plane(cardTex('sum', HEX.linkHi, sumBars, { barCol: shi }), 92, g);
      H.sumT = plane(cardTex('sum', shi, sumBars, { barCol: shi }), 92, g);     // the scaled write, in the sender's colour
      H.stream = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShaderMaterial({
        uniforms: { uCol: { value: new THREE.Color(rc) }, uTint: { value: new THREE.Color(sc) }, uAmp: { value: 1 }, uPlus: { value: 0.3 }, uTop: { value: 0 }, uTime: { value: 0 } },
        vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
        fragmentShader: `uniform vec3 uCol, uTint; uniform float uAmp, uPlus, uTop, uTime; varying vec2 vUv;
          void main(){ float core = exp(-pow((vUv.x-0.5)*7.0, 2.0)); float flow = 0.75 + 0.25*sin(vUv.y*60.0 - uTime*6.0);
            float tint = step(uPlus, vUv.y) * (1.0 - smoothstep(uTop - 0.02, uTop + 0.02, vUv.y));
            vec3 c = mix(uCol, uTint, tint*0.9) * flow * core * (1.0 + 1.8*tint);
            gl_FragColor = vec4(c*uAmp, 1.0); }`,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      })); H.stream.renderOrder = 3; g.add(H.stream);
      // + and x; the x carries a gauge showing the write size, held at 10% (tanh(alpha) = 0.1 in nl20 and nl22c)
      const opTex = (sym) => { const cv = document.createElement('canvas'); cv.width = cv.height = 192; const x = cv.getContext('2d'); const c0 = 96;
        if (sym !== '+') { x.lineWidth = 10; x.strokeStyle = 'rgba(255,255,255,0.18)'; x.beginPath(); x.arc(c0, c0, 84, 0, Math.PI * 2); x.stroke();
          x.lineWidth = 16; x.lineCap = 'round'; x.strokeStyle = shi; x.beginPath(); x.arc(c0, c0, 84, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * 0.1); x.stroke(); x.lineCap = 'butt'; }
        x.strokeStyle = rhi; x.lineWidth = 9; x.beginPath(); x.arc(c0, c0, 52, 0, Math.PI * 2); x.fillStyle = 'rgba(12,8,24,0.95)'; x.fill(); x.stroke();
        x.lineCap = 'round'; x.beginPath(); if (sym === '+') { x.moveTo(c0 - 28, c0); x.lineTo(c0 + 28, c0); x.moveTo(c0, c0 - 28); x.lineTo(c0, c0 + 28); } else { x.moveTo(c0 - 20, c0 - 20); x.lineTo(c0 + 20, c0 + 20); x.moveTo(c0 + 20, c0 - 20); x.lineTo(c0 - 20, c0 + 20); } x.stroke();
        const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.userData = { aspect: 1 }; return t; };
      H.plus = plane(opTex('+'), 80, g); H.times = plane(opTex('x'), 80, g);
      H.plus.renderOrder = H.times.renderOrder = 7;
      this.halves[reader] = H;
    }
    // labels
    const mk = (html, color, o = {}) => new Pin(hud, html, { size: 46, weight: 700, font: FONT.display, color, dot: false, anchor: 'center', ...o });
    this.pins = {
      headQ: mk(`<span style="color:${CSS.quillHi}">Quill</span> reads <span style="color:${CSS.emberHi}">Ember</span>`, '#f5f3ff', { size: 60 }),
      headE: mk(`<span style="color:${CSS.emberHi}">Ember</span> reads <span style="color:${CSS.quillHi}">Quill</span>`, '#f5f3ff', { size: 60 }),
      key: mk('key', '#f5f3ff', { anchor: 'right', size: 42 }), value: mk('value', '#f5f3ff', { anchor: 'right', size: 42 }), gate: mk('gate', '#f5f3ff', { anchor: 'center', size: 40 }),
      state: mk('state', '#f5f3ff', { anchor: 'right', size: 42 }),
      query: mk('query', CSS.quillHi, { anchor: 'right' }), att: mk('attention weights', HEX.linkHi, { anchor: 'left' }), nothing: mk('nothing', CSS.quillHi, { size: 38 }),
      sum: mk('sum', HEX.linkHi), write: mk('write size', CSS.quillHi, { anchor: 'center', size: 42 }), added: mk('added to the stream', CSS.quillHi, { anchor: 'right' }),
    };
    this.theta = 0.09;
  }

  // half-local px -> half group local units (pivot at the seam); px measured on the full 1920x1080 panel plane
  P(H, px, py, z = 0) { const S = this.S; return new THREE.Vector3((px - 960) * S, (540 - py) * S, z * S); }
  place(H, m, px, py, s = 1, op = 1, z = 0) {
    m.position.copy(this.P(H, px, py, z)); m.scale.set(m.userData.w * this.S * s, m.userData.h * this.S * s, 1);
    m.visible = s > 0.003 && op > 0.003; m.material.opacity = op;
  }
  world(H, px, py, z = 0) { H.g.updateMatrixWorld(true); return this.P(H, px, py, z).applyMatrix4(H.g.matrixWorld); }

  update(t, camera, amp, fx) {
    const on = amp > 0.003; this.root.visible = on;
    for (const k in this.pins) this.pins[k].root.style.display = 'none';
    if (!on) return {};
    this.root.updateMatrixWorld(true);
    const S = this.S, res = {};
    const A = (a, b, f = 0.15) => win(t, a, b, f) * amp;
    for (const reader of ['quill', 'ember']) {
      const H = this.halves[reader], left = reader === 'quill';
      const x0 = left ? 0 : 960;                       // half origin on the full plane
      H.g.rotation.y = (left ? 1 : -1) * this.theta * amp;
      // glass backing
      H.back.position.copy(this.P(H, x0 + 480, 560, -6)); H.back.scale.set(930 * S, 990 * S, 1); H.back.material.opacity = 0.92 * amp; H.back.visible = true;
      const n = H.slots.length, DX = 98, X0 = x0 + 185, XR = x0 + 885, Y = { tok: 190, st: 290, val: 412, key: 545, sum: 765, read: 905, col0: 1035, col1: 130 };
      const pop = (t0, d = 0.3) => easeOutBack(clamp((t - t0) / d), 2.0);
      // ---- partner columns
      const beamK = prog(t, TR.beams, TR.beams + 0.3, easeOutCubic), wK = prog(t, TR.weights, TR.weights + 0.4, easeOutCubic);
      const flowA = win(t, TR.flow, TR.sum + 0.35, 0.1);
      const qp = { x: XR - 125, y: Y.read };
      H.slots.forEach((sl, j) => {
        const xs = X0 + j * DX, st0 = TR.cards + 0.04 * j, kv0 = TR.kv + 0.04 * j;
        // token and state (the state lands from the lattice: see a2_link's fly-out)
        sl.tok.position.copy(this.P(H, xs, Y.tok, 2)); sl.tok.quaternion.identity(); sl.tok.scale.setScalar(Math.max(1e-3, 0.95 * pop(st0))); sl.tok.visible = t > st0; sl.tok.material.opacity = amp;
        sl.tok.scale.multiplyScalar(54 * S / 0.3);
        this.place(H, sl.st, xs, Y.st, pop(st0), amp);
        const kk = pop(kv0);
        this.place(H, sl.ky, xs, Y.key, kk, amp, 2);
        const gk = prog(t, TR.gates + 0.04 * j, TR.gates + 0.45 + 0.04 * j, easeOutCubic), g = lerp(1, sl.gate, gk);
        // the value dims and shrinks with its gate; the gauge fills to the gate
        this.place(H, sl.vl, xs - 10, Y.val, kk * (0.55 + 0.45 * g), amp * (0.3 + 0.7 * g) * (1 - 0.45 * prog(t, TR.flow, TR.sum)), 2);
        this.place(H, sl.gt, xs + 40, Y.val, gk > 0.01 ? 1 : 0, amp * gk, 2);
        sl.fill.position.copy(this.P(H, xs + 40, Y.val + 24 - 46 * sl.gate * gk / 2, 3)); sl.fill.scale.set(9 * S, Math.max(1e-3, 46 * sl.gate * gk) * S, 1);
        sl.fill.visible = gk > 0.01; sl.fill.material.opacity = amp * 0.95;
        // attention weight: a teal bar under the key, and the beam's thickness
        const wl = 80 * sl.w / 0.31 * wK;
        sl.wbar.position.copy(this.P(H, xs - 40 + wl / 2, Y.key + 56, 3)); sl.wbar.scale.set(Math.max(1e-3, wl) * S, 9 * S, 1); sl.wbar.visible = wK > 0.01; sl.wbar.material.opacity = amp;
        if (beamK > 0.003) {
          const a = this.world(H, qp.x, qp.y - 46, 4), b = this.world(H, xs, Y.key + 64, 4);
          const c = a.clone().lerp(b, 0.5); c.addScaledVector(new THREE.Vector3(0, 0, 1).applyQuaternion(this.root.quaternion), 0.25);
          const w = lerp(0.22, sl.w / 0.31, wK);
          fx.rays.add(a, c, b, { width: S * (2 + 12 * w), color: new THREE.Color(H.rc), alpha: amp * (0.35 + 0.65 * w) * (1 - 0.55 * prog(t, TR.flow, TR.sum)) * (1 - prog(t, TR.scale - 0.2, TR.scale + 0.1)), g1: beamK, pulse: clamp((t - TR.beams) / 0.4), pulseColor: new THREE.Color(H.rhi), pulseAmt: 1.6 });
        }
        // weighted, gated values flow into the sum
        if (flowA > 0.002) for (let m = 0; m < 4; m++) {
          const u = clamp((t - TR.flow - 0.07 * m - 0.03 * j) / 0.45); if (u <= 0 || u >= 1) continue;
          const p0 = this.world(H, xs - 10, Y.val + 24, 6), p1 = this.world(H, X0 + 280, Y.sum, 6);
          const p = p0.clone().lerp(p1, easeInOutCubic(u)); p.addScaledVector(new THREE.Vector3(0, 0, 1).applyQuaternion(this.root.quaternion), 0.25 * Math.sin(Math.PI * u));
          fx.glows.add(p, new THREE.Color(H.shi), 0.06 + 0.32 * Math.sqrt(sl.w * sl.gate), 1.6 * flowA * amp);
        }
      });
      // null key and its zero value (the reader's own)
      { const xs = X0 + n * DX, kk = pop(TR.q);
        this.place(H, H.nul, xs, Y.key, kk, amp, 2); this.place(H, H.nulV, xs - 10, Y.val, kk, amp * 0.8, 2);
        const wl = 80 * H.sp.w0 / 0.31 * wK; H.nulW.position.copy(this.P(H, xs - 40 + wl / 2, Y.key + 56, 3)); H.nulW.scale.set(Math.max(1e-3, wl) * S, 9 * S, 1); H.nulW.visible = wK > 0.01; H.nulW.material.opacity = amp;
        if (beamK > 0.003) {
          const a = this.world(H, qp.x, qp.y - 46, 4), b = this.world(H, xs, Y.key + 64, 4), c = a.clone().lerp(b, 0.5);
          const w = lerp(0.22, H.sp.w0 / 0.31, wK);
          fx.rays.add(a, c, b, { width: S * (2 + 12 * w), color: new THREE.Color(H.rc), alpha: amp * (0.35 + 0.65 * w) * (1 - 0.55 * prog(t, TR.flow, TR.sum)) * (1 - prog(t, TR.scale - 0.2, TR.scale + 0.1)), g1: beamK });
        }
        H.nullX = xs; }
      // the reader's stream, state, token and query
      const plusU = (Y.col0 - (Y.sum + 0)) / (Y.col0 - Y.col1);       // the + sits at the sum's height
      H.stream.position.copy(this.P(H, XR, (Y.col0 + Y.col1) / 2, 0)); H.stream.scale.set(80 * S, (Y.col0 - Y.col1) * S, 1);
      const U = H.stream.material.uniforms; U.uAmp.value = amp * prog(t, TR.lift, TR.cards + 0.3); U.uTime.value = t; U.uPlus.value = plusU;
      U.uTop.value = t < TR.hit ? plusU - 0.001 : lerp(plusU, 1.05, prog(t, TR.hit, TR.hit + 0.6, easeInCubic));
      this.place(H, H.rState, XR, Y.read, pop(TR.cards + 0.2), amp, 4);
      H.rTok.position.copy(this.P(H, XR, Y.read + 80, 6)); H.rTok.scale.setScalar(Math.max(1e-3, 0.95 * pop(TR.cards + 0.2) * 54 * S / 0.3)); H.rTok.visible = t > TR.cards + 0.2; H.rTok.material.opacity = amp;
      const qk = pop(TR.q);
      this.place(H, H.query, qp.x, qp.y, qk, amp, 5);
      if (qk > 0.01) fx.glows.add(this.world(H, qp.x, qp.y, 2), new THREE.Color(H.rc), 0.42 * qk, 0.9 * amp * (1 + 1.2 * win(t, TR.beams - 0.05, TR.beams + 0.4, 0.1)));
      if (t > TR.cards + 0.2) fx.glows.add(this.world(H, XR, Y.read, 2), new THREE.Color(H.rc), 0.36, 0.7 * amp);
      // sum -> x write size -> + into the stream
      const sx0 = X0 + 280, sxT = XR - 125, sk = pop(TR.sum, 0.3);
      const mv = prog(t, TR.scale, TR.scale + 0.3, easeInOutCubic), shr = prog(t, TR.scale + 0.15, TR.scale + 0.45, easeOutCubic), pl = prog(t, TR.plunge, TR.hit, easeInCubic);
      const sxx = lerp(lerp(sx0, sxT, mv), XR, pl), ssz = sk * lerp(1, 0.32, shr) * (1 - 0.7 * pl);
      this.place(H, H.sum, sxx, Y.sum, t < TR.scale + 0.3 ? ssz : 0, amp, 6);
      this.place(H, H.sumT, sxx, Y.sum, t >= TR.scale + 0.3 && t < TR.hit ? ssz : 0, amp, 6);
      this.place(H, H.times, sxT, Y.sum, pop(TR.scale - 0.1) * (1 - 0.4 * prog(t, TR.hit, TR.hit + 0.4)), amp, 5);
      const hitK = t >= TR.hit ? Math.exp(-4 * (t - TR.hit)) : 0;
      this.place(H, H.plus, XR, Y.sum, pop(TR.sum, 0.3) * (1 + 0.35 * hitK), amp, 7);
      if (hitK > 0.01) fx.glows.add(this.world(H, XR, Y.sum, 8), new THREE.Color(H.shi), 0.25 + 0.6 * hitK, 2.0 * hitK * amp);
      if (t > TR.hit && t < TR.hit + 0.7) fx.glows.add(this.world(H, XR, lerp(Y.sum, Y.col1, prog(t, TR.hit, TR.hit + 0.6, easeInCubic)), 8), new THREE.Color(H.shi), 0.22, 1.4 * amp);
      res[reader] = { stateW: (j) => this.world(H, X0 + j * DX, Y.st, 4), readerW: this.world(H, XR, Y.read, 4), hit: hitK };
      H.lay = { X0, DX, XR, Y, qp, sx0, sxT, x0 };
    }
    // ---- labels (headers persist; the rest only during holds, placed in free space)
    const Q = this.halves.quill, E = this.halves.ember, LQ = Q.lay, LE = E.lay;
    const show = (k, H, px, py, w, dx = 0, dy = 0) => { if (w > 0.003) this.pins[k].update(this.world(H, px, py, 8), camera, w, dx, dy); };
    show('headQ', Q, 480, 118, A(TR.cards, 31.25, 0.25));
    show('headE', E, 1440, 118, A(TR.cards, 31.25, 0.25));
    show('state', Q, LQ.X0 - 62, LQ.Y.st, A(TR.labKV[0], TR.labKV[1]));
    show('value', Q, LQ.X0 - 62, LQ.Y.val, A(TR.labKV[0], TR.labKV[1]));
    show('key', Q, LQ.X0 - 62, LQ.Y.key, A(TR.labKV[0], TR.labKV[1]));
    show('gate', Q, LQ.X0 + 40, LQ.Y.val - 62, A(TR.labKV[0] + 0.125, TR.labKV[1]));
    show('query', Q, LQ.qp.x - 58, LQ.qp.y, A(TR.labQ[0], TR.labQ[1]));
    show('att', Q, 70, 1000, A(TR.labA[0], TR.labA[1]));
    show('nothing', Q, Q.nullX, (LQ.Y.val + LQ.Y.key) / 2 + 6, A(TR.labA[0] + 0.125, TR.labA[1]));
    show('sum', Q, LQ.sx0, LQ.Y.sum + 72, A(TR.labS[0], TR.labS[1]));
    show('write', Q, LQ.sxT, LQ.Y.sum - 62, A(TR.labW[0], TR.labW[1]));
    show('added', Q, LQ.XR - 52, LQ.Y.sum - 62, A(TR.labR[0], TR.labR[1]));
    return res;
  }
}
