// The 3D game window kit (a4_games, a6_proof). A game window is a glass frame in the shot's own 3D scene. Its picture
// is a real 3D diorama rendered every frame into its own render target: the two Creature models, the Link with both
// directions always running, the private item as a gold 3D tile at the holder's tail (only that side has it), the other
// side's input as a grey "?" tile, and the written answer as 3D token tiles (the models' real tokenizations) that leave
// the writer's head one per forward pass, then a 3D check badge.
// Mechanism shown (pairing/link.py): every written token is one joint forward pass in lockstep (a wave up both bodies);
// at each of the 8 exchange depths both strands fire at once; what crosses is ADDED into the reader's core (the
// sender's colour, gold for the private item) and rises.
// Everything is a pure function of t: render(t) renders the diorama at min(t, freeze) and skips the render when the
// target already holds that time.
import * as THREE from 'three';
import { Creature, XBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { C } from '../lib/palette.js';
import { textTexture } from '../lib/tokens.js';
import { clamp, lerp, spring, kick, hash, easeOutCubic, easeInOutCubic, easeOutBack } from '../lib/util.js';

export const GOLD = new THREE.Color('#ffd166');
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const after = (t, a, d = 0.15) => clamp((t - a) / d);

// ---------------------------------------------------------------- 3D token tiles
const STY = {
  quill: { body: '#3b1a7a', emis: C.quill, border: '#a78bfa', fg: '#f3efff', radius: 0.16 },
  ember: { body: '#7a2a0c', emis: C.ember, border: '#fb923c', fg: '#fff1e2', radius: 0.5 },
  goldq: { e0: 0.06, body: '#5a3a00', emis: GOLD, border: '#ffd166', fg: '#fff3c4', radius: 0.16 },
  golde: { e0: 0.06, body: '#5a3a00', emis: GOLD, border: '#ffd166', fg: '#fff3c4', radius: 0.5 },
  secret: { e0: 0.05, body: '#4a3000', emis: GOLD, border: '#ffd166', fg: '#fff3c4', radius: 0.22 },
  ghost: { body: '#2a2b38', emis: new THREE.Color('#8b8fa3'), border: '#9aa0b8', fg: '#d9dbe6', radius: 0.22 },
  plain: { body: '#1b2440', emis: new THREE.Color('#9fd8ff'), border: '#cfe3ff', fg: '#f2f6ff', radius: 0.22 },
  dimq: { body: '#2b2148', emis: C.quill, border: 'rgba(167,139,250,0.55)', fg: '#ffffff', radius: 0.16 },
  dime: { body: '#3d2216', emis: C.ember, border: 'rgba(251,146,60,0.55)', fg: '#ffffff', radius: 0.5 },
};
const geoCache = new Map();
function slabGeo(w, h, r, d) {
  const key = [w.toFixed(3), h.toFixed(3), r.toFixed(3), d.toFixed(3)].join();
  if (geoCache.has(key)) return geoCache.get(key);
  const s = new THREE.Shape(); const x0 = -w / 2, y0 = -h / 2; r = Math.min(r, h / 2, w / 2);
  s.moveTo(x0 + r, y0); s.lineTo(x0 + w - r, y0); s.quadraticCurveTo(x0 + w, y0, x0 + w, y0 + r); s.lineTo(x0 + w, y0 + h - r);
  s.quadraticCurveTo(x0 + w, y0 + h, x0 + w - r, y0 + h); s.lineTo(x0 + r, y0 + h); s.quadraticCurveTo(x0, y0 + h, x0, y0 + h - r);
  s.lineTo(x0, y0 + r); s.quadraticCurveTo(x0, y0, x0 + r, y0);
  const b = h * 0.05;
  const g = new THREE.ExtrudeGeometry(s, { depth: d, bevelEnabled: true, bevelThickness: b, bevelSize: b, bevelSegments: 2, curveSegments: 10 });
  g.translate(0, 0, -d / 2);
  geoCache.set(key, g); return g;
}
// text: shown text (leading space dropped); h: tile height (world units). Returns a Group with userData {w, h}.
export function tile3D(text, kind, h, { width = null } = {}) {
  const st = STY[kind];
  const label = text.replace(/^ /, '');
  const blank = label.trim() === '';
  const tex = textTexture(blank ? ' ' : label, { font: '600 64px Inter', pad: 20, h: 104, radius: Math.round(st.radius * 104), bg: null, border: st.border, borderWidth: 5, color: st.fg, dy: 3 });
  const asp = tex.userData.aspect;
  const w = width ?? (blank ? h * 0.3 : h * asp);
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(st.body), emissive: st.emis.clone(), emissiveIntensity: 0.22, roughness: 0.32, metalness: 0.35, transparent: true });
  const body = new THREE.Mesh(slabGeo(w, h, st.radius * h, h * 0.16), mat);
  g.add(body);
  const fmat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false });
  fmat.color.setScalar(0.88);
  const face = new THREE.Mesh(new THREE.PlaneGeometry(w + h * 0.1, h * 1.1), fmat);
  if (blank || width) face.scale.x = (w + h * 0.1) / (w + h * 0.1); // texture stretches to the slab
  face.position.z = h * 0.08 + h * 0.05 + 0.004;
  g.add(face);
  g.userData = { w, h, mat, fmat, body, face, kind, e0: st.e0 ?? 0.22 };
  return g;
}
function setTileAlpha(g, a, glow = 0) {
  const u = g.userData; u.mat.opacity = a; u.fmat.opacity = a; u.mat.emissiveIntensity = u.e0 + glow;
  g.visible = a > 0.003;
}

// a 3D check badge: dark disc, glowing ring, a tick that draws on
export function check3D(r) {
  const g = new THREE.Group();
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(r, r, r * 0.22, 48), new THREE.MeshStandardMaterial({ color: new THREE.Color('#05382c'), emissive: C.link.clone(), emissiveIntensity: 0.18, roughness: 0.3, metalness: 0.4, transparent: true }));
  disc.rotation.x = Math.PI / 2; g.add(disc);
  const ringM = new THREE.MeshBasicMaterial({ color: C.link.clone().multiplyScalar(1.6), transparent: true, toneMapped: false });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(r, r * 0.1, 10, 48), ringM); ring.position.z = r * 0.12; g.add(ring);
  const path = new THREE.CurvePath();
  const a = V(-0.42 * r, 0.02 * r, 0), b = V(-0.1 * r, -0.32 * r, 0), c = V(0.45 * r, 0.36 * r, 0);
  path.add(new THREE.LineCurve3(a, b)); path.add(new THREE.LineCurve3(b, c));
  const tickG = new THREE.TubeGeometry(path, 32, r * 0.11, 8, false);
  const tickM = new THREE.MeshBasicMaterial({ color: new THREE.Color('#d9fff2').multiplyScalar(1.5), transparent: true, toneMapped: false });
  const tick = new THREE.Mesh(tickG, tickM); tick.position.z = r * 0.16; g.add(tick);
  g.userData = { disc, ring, tick, tickG, total: tickG.index.count, mats: [disc.material, ringM, tickM] };
  return g;
}
function updateCheck(g, t, tc) {
  const u = g.userData;
  if (t < tc) { g.visible = false; return; }
  g.visible = true;
  const s = spring(t, tc, 2.8, 0.42);
  g.scale.setScalar(Math.max(0.01, s * (1 + 0.3 * Math.exp(-(t - tc) * 7))));
  g.rotation.y = (1 - clamp((t - tc) / 0.3)) * 1.2;
  const dr = easeOutCubic(clamp((t - tc - 0.03) / 0.2));
  u.tickG.setDrawRange(0, Math.floor(u.total * dr / 3) * 3);
  u.ring.material.color.copy(C.link).multiplyScalar(1.4 + 2.5 * Math.exp(-(t - tc) * 5));
}

// ---------------------------------------------------------------- a written text block (tiles laid out with wrapping)
// side: 'quill' | 'ember' (the writer), tokens: real tokenization, gold: indices of value tokens (gold tiles)
class TextBlock {
  constructor(group, side, tokens, gold, box, h, sides = null, breaks = []) {
    this.side = side; this.box = box; this.h = h; this.sides = sides; // box: {x, y (top line centre), w, align, lines}
    this.tiles = tokens.map((tok, i) => {
      const sd = sides ? sides[i] : side;
      const kind = gold.includes(i) ? (sd === 'quill' ? 'goldq' : 'golde') : sd;
      const g = tile3D(tok, kind, h); g.userData.space = tok.startsWith(' ') && tok.trim() !== '' ; g.userData.i = i;
      group.add(g); return g;
    });
    // layout
    const gapW = h * 0.2, gapS = h * 0.05;
    // wrap only between words: a run of glued tokens (no leading space) stays on one line
    let x = 0, line = 0; const pos = [];
    const startsWord = (i) => i === 0 || this.tiles[i].userData.space || tokens[i] === ' ' || tokens[i - 1] === ' ';
    const runW = (i) => { let w = this.tiles[i].userData.w; for (let j = i + 1; j < tokens.length && !startsWord(j); j++) w += gapS + this.tiles[j].userData.w; return w; };
    this.tiles.forEach((g, i) => {
      const w = g.userData.w; const gap = i === 0 ? 0 : (g.userData.space ? gapW : gapS);
      if (i > 0 && (breaks.includes(i) || (startsWord(i) && x + gap + runW(i) > box.w))) { line++; x = 0; } else x += gap;
      pos.push({ x: x + w / 2, line }); x += w;
    });
    this.nLines = line + 1;
    // per-line widths for alignment
    const lw = new Array(this.nLines).fill(0);
    pos.forEach((p, i) => { lw[p.line] = Math.max(lw[p.line], p.x + this.tiles[i].userData.w / 2); });
    this.pos = pos.map((p) => {
      const off = box.align === 'center' ? -lw[p.line] / 2 : box.align === 'right' ? -lw[p.line] : 0;
      return { x: box.x + off + p.x, line: p.line };
    });
    this.lineW = lw;
    this.pitch = h * 1.24;
    this.chk = check3D(h * 0.48); group.add(this.chk);
  }
  // times: per token time (writer's passes). mouth: world point the tiles leave from. tc: check time
  update(t, times, mouth, tc, alpha = 1) {
    const L = this.box.lines || 2;
    // scroll: once a line beyond the box starts, earlier lines rise out
    let scroll = 0;
    for (let ln = L; ln < this.nLines; ln++) {
      const first = this.pos.findIndex((p) => p.line === ln);
      if (first >= 0 && times[first] != null) scroll += spring(t, times[first], 3.2, 0.7) * this.pitch;
    }
    let lastPos = null;
    this.tiles.forEach((g, i) => {
      const t0 = times[i];
      if (t0 == null || t < t0) { g.visible = false; return; }
      const p = this.pos[i];
      const yLine = this.box.y - p.line * this.pitch + scroll;
      const dst = V(p.x, yLine, this.box.z || 0);
      // a short hop in from the writer's side (a full flight from the head crossed the tiles already written)
      const k = clamp((t - t0) / 0.14), ke = easeOutCubic(k);
      const src = dst.clone().lerp((typeof mouth === 'function' ? mouth(i) : mouth), 0.16);
      const pnt = src.clone().lerp(dst, ke); pnt.z += Math.sin(Math.PI * k) * 0.25;
      g.position.copy(pnt);
      const sp = spring(t, t0, 3.0, 0.5);
      g.scale.setScalar(Math.max(0.01, lerp(0.25, 1, sp)));
      g.rotation.set(0, (1 - ke) * -0.9 * ((this.sides ? this.sides[i] : this.side) === 'quill' ? -1 : 1), 0);
      // fade lines that scrolled out of the top
      const topY = this.box.y + this.pitch * 0.12;
      const out = clamp((yLine - topY) / (this.pitch * 0.4));
      const hot = Math.exp(-(t - t0) * 6);
      setTileAlpha(g, alpha * clamp((t - t0) / 0.05) * (1 - out), 0.9 * hot);
      lastPos = V(p.x + g.userData.w / 2 + this.h * 0.62, yLine, this.box.z || 0);
    });
    if (lastPos && t >= tc && alpha > 0.01) { this.chk.position.copy(lastPos); this.chk.position.z += 0.1; updateCheck(this.chk, t, tc); this.chk.userData.mats.forEach((m) => { m.opacity = alpha; }); }
    else this.chk.visible = false;
  }
}

// ---------------------------------------------------------------- a game diorama
// spec: {
//   inputs: { quill: [{text, kind}], ember: [...] } tiles at each model's tail (kind 'secret' = private, 'ghost' = "?",
//            'plain' = a public question); rows of tiles laid out left to right, optional row break with {br:true}
//   holder: 'quill' | 'ember' | 'both' | null (whose private item rides the link: that direction's packets are gold)
//   write: [{ side, tokens, gold, box?: {align, lines} }]   both write at once (one token per joint pass)
//   alsoWrites: 'quill' | 'ember' | null  (the sender also writes its own reply to another question: dim tiles)
//   step: seconds per pass (token)
//   cs: creature scale, layout overrides }
// o: { w, h: picture size in CSS px, scale: render scale, ppu: px per world unit at z=0 }
export class Diorama {
  constructor(renderer, spec, o) {
    this.renderer = renderer; this.spec = spec; this.o = o;
    const scene = this.scene = new THREE.Scene();
    this.world = makeWorld(scene, { dust: 160, fogDensity: 0.02, gridAmp: 1.2 });
    this.world.rimQ.intensity = 40; this.world.rimE.intensity = 40;
    const ppu = o.ppu || 70;
    this.Hv = o.h / ppu; this.Wv = o.w / ppu;
    const fov = 26;
    this.cam = new THREE.PerspectiveCamera(fov, o.w / o.h, 0.1, 200);
    this.D = (this.Hv / 2) / Math.tan((fov / 2) * Math.PI / 180);
    const L = this.L = { yBot: spec.yBot ?? -2.15, cs: spec.cs ?? o.cs ?? 0.8, cx: spec.cx ?? Math.min(2.0, this.Wv * 0.25), tileH: o.tileH || 0.84 };
    this.camTgt = V(0, L.yBot + this.Hv / 2, 0);
    const q = this.q = new Creature('quill'), e = this.e = new Creature('ember');
    scene.add(q.group, e.group); q.camera = e.camera = this.cam;
    this.link = new Link(q, e); scene.add(this.link.group);
    // inputs at the tails
    this.inputs = [];
    for (const side of ['quill', 'ember']) {
      const items = (spec.inputs && spec.inputs[side]) || [];
      const x0 = side === 'quill' ? -L.cx : L.cx;
      const rows = [[]]; items.forEach((it) => { if (it.br) rows.push([]); else rows[rows.length - 1].push(it); });
      rows.forEach((row, r) => {
        const tiles = row.map((it) => { const g = tile3D(it.text, it.kind, L.tileH); scene.add(g); g.userData.item = it; return g; });
        const gap = L.tileH * 0.16; const tw = tiles.reduce((s, g) => s + g.userData.w, 0) + gap * Math.max(0, tiles.length - 1);
        // keep the row inside the picture
        let cx = x0; const half = this.Wv / 2 - 0.25;
        const pin = spec.pin && spec.pin[side];
        if (pin === 'left') cx = -half + tw / 2; else if (pin === 'right') cx = half - tw / 2;
        if (cx - tw / 2 < -half) cx = -half + tw / 2; if (cx + tw / 2 > half) cx = half - tw / 2;
        let x = cx - tw / 2;
        tiles.forEach((g) => { g.userData.home = V(x + g.userData.w / 2, L.yBot + 1.0 + (rows.length - 1 - r) * L.tileH * 1.18 + (spec.inputLift || 0), 0.45); x += g.userData.w + gap; this.inputs.push({ g, side, r }); });
      });
    }
    // feed beams from the input row up into the tail
    const beamMat = (col) => new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
    this.beams = {};
    for (const side of ['quill', 'ember']) { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1, 8), beamMat(side === 'quill' ? C.quillHi.clone() : C.emberHi.clone())); scene.add(m); this.beams[side] = m; }
    // writers
    const h = L.tileH, top = L.yBot + this.Hv - 0.25 - h * 0.62;
    this.writers = (spec.write || []).map((wr, k) => {
      const n = spec.write.length;
      const lines = wr.box?.lines ?? (n > 1 ? 2 : 3);
      const yTop = top - (n > 1 && k === 1 ? (spec.write[0].box?.lines ?? 2) * h * 1.24 + h * 0.25 : 0);
      const align = wr.box?.align ?? (n > 1 ? (wr.side === 'quill' ? 'left' : 'right') : 'center');
      const bw = this.Wv - (n > 1 ? 0.7 : 1.3);
      const x = align === 'left' ? -bw / 2 : align === 'right' ? bw / 2 - h * 1.1 : -h * 0.55;
      return { ...wr, tb: new TextBlock(scene, wr.side, wr.tokens, wr.gold || [], { x, y: yTop, w: bw - 0.9, align, lines, z: 0.2 }, h, wr.sides || null, wr.breaks || []) };
    });
    // the sender's own reply (dim tiles that rise and fade above its head)
    this.dim = [];
    if (spec.alsoWrites) for (let k = 0; k < 24; k++) { const g = tile3D(' ', spec.alsoWrites === 'quill' ? 'dimq' : 'dime', h * 0.62, { width: h * (0.7 + hash(k * 3.3) * 1.1) }); scene.add(g); g.visible = false; this.dim.push(g); }
    // extra props
    this.flash = new THREE.PointLight(GOLD, 0, 8, 2); scene.add(this.flash);
    // render target
    const s = o.scale || 1;
    this.rt = new THREE.WebGLRenderTarget(Math.max(8, Math.round(o.w * s)), Math.max(8, Math.round(o.h * s)), { type: THREE.HalfFloatType, samples: 4 });
    this.lastT = null;
  }

  // the timeline of one play: T = {in, read, passes: [times], tok: per writer token times, check: per writer, end}
  static timeline(spec, t0, { gen = 0.75, step } = {}) {
    const st = step ?? spec.step ?? 0.125;
    const n = Math.max(...(spec.write || [{ tokens: [] }]).map((w) => w.tokens.length), spec.alsoWrites ? 8 : 0);
    const g0 = t0 + gen;
    const passes = Array.from({ length: n }, (_, i) => g0 + i * st);
    // a token appears as its pass ends (on the 16th grid when passes are)
    const tok = (spec.write || []).map((w) => w.tokens.map((_, i) => passes[i] + st));
    const check = tok.map((tt) => Math.ceil((tt[tt.length - 1] + 0.0625) / 0.125) * 0.125);
    return { in: t0, read: t0 + 0.375, passes, step: st, tok, check, end: Math.max(...check, g0) };
  }

  update(t, T, { camOff = null } = {}) {
    const { q, e, link, spec, L } = this;
    this.world.update(t);
    // camera (a portal: it shifts a little with the outer camera so the diorama has real parallax)
    const off = camOff || V();
    this.cam.position.set(off.x * 0.12 + Math.sin(t * 0.37) * 0.25, this.camTgt.y + 0.55 + off.y * 0.1, this.D);
    this.cam.lookAt(this.camTgt.x + off.x * 0.04, this.camTgt.y, 0);
    this.cam.updateMatrixWorld(true);
    const holder = spec.holder;
    const recv = holder === 'quill' ? 'ember' : holder === 'ember' ? 'quill' : null;
    // ---- passes: the read pass (whole prompt) then one per token; both bodies in lockstep
    const passList = [{ t: T.read, dur: 0.3, big: true }].concat(T.passes.map((p) => ({ t: p, dur: Math.min(0.12, T.step * 0.92) })));
    let wave = -1, cur = null;
    for (const p of passList) if (t >= p.t && t < p.t + p.dur + 0.04) { wave = Math.min(1.02, (t - p.t) / p.dur); cur = p; }
    const live = t >= T.read;
    const lastTok = T.passes.length ? T.passes[T.passes.length - 1] + T.step : T.read + 0.4;
    const busy = t >= T.read && t < lastTok + 0.3;
    // ---- creatures
    for (const [c, side] of [[q, 'quill'], [e, 'ember']]) {
      const p = c.pose, s = side === 'quill' ? -1 : 1;
      p.position.set(s * L.cx, 0, 0); p.scale = L.cs; p.yaw = -s * 0.45;
      p.xray = 0.58; p.portGlow = 1.0 + (busy ? 0.6 : 0); p.glow = 1.05; p.coreGlow = 1.05; p.crest = 1;
      p.sway = 0.08; p.swayPhase = t * 2.2 + (side === 'quill' ? 0 : 1.7); p.swayFreq = 0.8;
      p.squash = 1 - 0.02 * Math.exp(-((t % 0.5)) * 6);
      p.headTilt = 0.06 * Math.sin(t * 1.3 + s); p.headNod = 0.05 * Math.sin(t * 0.9 + s * 2);
      p.lean = 0; p.bend = 0; p.happy = 0; p.ghost = 0; p.opacity = 1;
      p.wave = wave;
      p.blink = (() => { const P = 2.6, k = Math.floor((t + s) / P), bt = k * P - s + 0.5 + hash(k * 1.7 + s * 9.1) * 1.4; return clamp(1 - Math.abs((t - bt) / 0.08 - 1)); })();
      c.inject = [];
    }
    // the writers nod a little with every token, and smile at their check
    (this.writers || []).forEach((w, k) => {
      const c = w.side === 'quill' ? q : e;
      for (const tt of T.tok[k]) if (t > tt - 0.05 && t < tt + 0.25) c.pose.headNod += 0.05 * Math.sin(Math.PI * clamp((t - tt + 0.05) / 0.3));
      const tc = T.check[k]; c.pose.happy = Math.max(c.pose.happy, t >= tc ? clamp((t - tc) / 0.1) * (1 - clamp((t - tc - 0.9) / 0.2)) : 0);
      c.pose.squash -= 0.06 * Math.max(0, kick(t, tc, 3, 6));
    });
    q.pose.lookAt = e.headWorld(); e.pose.lookAt = q.headWorld();
    // ---- what crosses: injected into each reader's core at the exchange depths, rising with the wave
    const gold = GOLD;
    const colQE = holder === 'quill' || holder === 'both' ? gold : C.quillHi, colEQ = holder === 'ember' || holder === 'both' ? gold : C.emberHi;
    if (live) {
      const w = wave >= 0 ? wave : 1.01;
      const base = clamp((t - T.read) / 0.3);
      const amt = (r) => (r === 'gold' ? 1.1 : 0.55) * base;
      XBLOCKS.forEach((blk) => {
        const u = (blk + 0.5) / 36; if (u > w && wave >= 0 && cur && cur.big) return;
        e.inject.push({ block: blk, color: colQE, amount: amt(colQE === gold ? 'gold' : 'x') * (u < w ? 1 : 0.6), top: Math.min(1.01, wave >= 0 ? Math.max(w, u + 0.03) : 1.01) });
        q.inject.push({ block: blk, color: colEQ, amount: amt(colEQ === gold ? 'gold' : 'x') * (u < w ? 1 : 0.6), top: Math.min(1.01, wave >= 0 ? Math.max(w, u + 0.03) : 1.01) });
      });
      // the holder's own copy of its private item: in its prompt, so in its core from block 0
      if (holder === 'quill' || holder === 'both') q.inject.push({ block: 0, color: gold, amount: 0.35 * base, top: 1.01 });
      if (holder === 'ember' || holder === 'both') e.inject.push({ block: 0, color: gold, amount: 0.35 * base, top: 1.01 });
    }
    q.update(t); e.update(t);
    // ---- the link: both directions always on; the private item's direction is gold
    const ls = link.state;
    ls.grow = clamp((t - T.in) / 0.35); ls.depth = 1; ls.sag = -0.16; ls.twist = 0.05;
    ls.intensity = 0.75 + (busy ? 0.45 : 0) + 0.5 * (cur && cur.big ? Math.sin(Math.PI * clamp(wave)) : 0);
    ls.aToB = 1; ls.bToA = 1; ls.packets = 3; ls.packetSize = 0.15; ls.flow = 1.3; ls.colorBySender = 0.5;
    ls.tintAB = (holder === 'quill' || holder === 'both') && live ? gold : null;
    ls.tintBA = (holder === 'ember' || holder === 'both') && live ? gold : null;
    // a bright pulse at every exchange, in both directions, for every pass
    const pulses = [];
    for (const p of passList) {
      for (let k = 0; k < 8; k++) {
        const launch = p.t + p.dur * (XBLOCKS[k] + 0.5) / 36, u = (t - launch) / 0.16;
        if (u < 0 || u > 1) continue;
        pulses.push({ e: k, dir: 0, u, color: ls.tintAB ? gold : C.linkHi, amount: p.big ? 2.2 : 1.6 });
        pulses.push({ e: k, dir: 1, u, color: ls.tintBA ? gold : C.linkHi, amount: p.big ? 2.2 : 1.6 });
      }
    }
    ls.pulses = pulses.slice(-48);
    link.update(t);
    // ---- inputs
    const inA = (k) => after(t, T.in + 0.1 + k * 0.06, 0.12);
    this.inputs.forEach(({ g, side, r }, k) => {
      const a = inA(k);
      const s = spring(t, T.in + 0.1 + k * 0.06, 2.6, 0.5);
      g.position.copy(g.userData.home); g.position.y += (1 - s) * -0.8 + 0.04 * Math.sin(t * 2 + k);
      g.scale.setScalar(Math.max(0.01, 0.4 + 0.6 * s));
      g.rotation.set(0, 0.1 * Math.sin(t * 0.8 + k), 0);
      const it = g.userData.item; const isSecret = it.kind === 'secret';
      const readGlow = isSecret ? Math.exp(-Math.max(0, t - T.read) * 3) * (t >= T.read ? 1 : 0) : 0;
      setTileAlpha(g, a, (isSecret ? 0.05 + 0.05 * Math.sin(t * 5 + k) : 0) + 0.8 * readGlow);
    });
    // feed beams (input -> tail)
    for (const side of ['quill', 'ember']) {
      const m = this.beams[side], c = side === 'quill' ? q : e;
      const ins = this.inputs.filter((x) => x.side === side);
      if (!ins.length) { m.visible = false; continue; }
      const top = c.tailWorld(), bot = ins.reduce((a, x) => (x.g.position.y > a.y ? x.g.position : a), V(0, -99, 0)).clone();
      bot.y += L.tileH * 0.5; bot.x = top.x; bot.z = 0.4;
      const len = Math.max(0.01, top.y - bot.y); m.position.set(top.x, (top.y + bot.y) / 2, 0.3); m.scale.set(1, len, 1);
      m.visible = len > 0.05;
      const sec = ins.some((x) => x.g.userData.item.kind === 'secret');
      m.material.color.copy(sec ? GOLD : (side === 'quill' ? C.quillHi : C.emberHi));
      m.material.opacity = inA(0) * (0.35 + 0.9 * (t >= T.read ? Math.exp(-(t - T.read) * 4) : 0));
    }
    // ---- written answers
    (this.writers || []).forEach((w, k) => {
      const c = w.side === 'quill' ? q : e;
      const mq = q.mouthWorld(), me = e.mouthWorld();
      w.tb.update(t, T.tok[k], w.sides ? (i) => (w.sides[i] === 'quill' ? mq : me) : c.mouthWorld(), T.check[k]);
    });
    // the sender's own reply: dim tiles leave its head and fade
    if (spec.alsoWrites) {
      const c = spec.alsoWrites === 'quill' ? q : e, m = c.headWorld();
      const sgn = spec.alsoWrites === 'quill' ? -1 : 1;
      this.dim.forEach((g, i) => {
        const t0 = T.passes[i] != null ? T.passes[i] + T.step : null;
        if (t0 == null || t < t0 || t > t0 + 0.9) { g.visible = false; return; }
        const k = (t - t0) / 0.9;
        g.position.set(m.x + sgn * 0.15 + (hash(i) - 0.5) * 0.5, m.y + 0.9 + k * 1.1, 0.5);
        g.scale.setScalar(0.45 + 0.35 * easeOutCubic(clamp(k * 3)));
        g.rotation.set(0, 0, (hash(i * 2.3) - 0.5) * 0.3);
        setTileAlpha(g, 0.55 * clamp(k * 6) * (1 - clamp((k - 0.3) / 0.7)), 0);
      });
    }
    // gold flash at the read
    this.flash.intensity = t >= T.read ? 30 * Math.exp(-(t - T.read) * 4) : 0;
    if (holder && holder !== 'both') { const c = holder === 'quill' ? q : e; this.flash.position.copy(c.tailWorld()).add(V(0, 0.6, 1)); }
  }

  // render into the target (pure: skips if this time is already there)
  render(t, T, opts = {}) {
    const key = t.toFixed(4) + (opts.camOff ? opts.camOff.x.toFixed(3) + opts.camOff.y.toFixed(3) : '');
    if (this.lastT === key) return this.rt.texture;
    this.update(t, T, opts);
    const r = this.renderer, prev = r.getRenderTarget();
    r.setRenderTarget(this.rt); r.clear(); r.render(this.scene, this.cam); r.setRenderTarget(prev);
    this.lastT = key; return this.rt.texture;
  }
}

// ---------------------------------------------------------------- the glass frame in the outer scene
// Built in CSS px units (the group is scaled by 1/ppu), so sizes read as screen sizes at the framing distance.
// o: { w, h: picture px; header: px; footer: px; title; stat; caption; tex (the diorama's texture) }
const sdfFrag = `
uniform sampler2D map; uniform vec2 size; uniform float radius, uOp, uHit, uDim; varying vec2 vUv;
float sdR(vec2 p, vec2 b, float r){ vec2 q=abs(p)-b+r; return length(max(q,0.))+min(max(q.x,q.y),0.)-r; }
void main(){ vec2 p=(vUv-0.5)*size; float d=sdR(p, size*0.5, radius); float aa=max(fwidth(d),0.5);
  float a = 1.0-smoothstep(-aa, aa, d);
  vec3 c = texture2D(map, vUv).rgb;
  c *= 1.0 - 0.45*smoothstep(-60.0, 0.0, d);
  float sheen = smoothstep(0.0, 1.0, 1.0-abs((vUv.x*0.6+vUv.y) - 1.05)*7.0)*0.02;
  c += vec3(0.75,0.85,1.0)*sheen; c += vec3(0.2,0.95,0.75)*uHit*0.18*smoothstep(-120.0, 0.0, d);
  c *= uDim;
  gl_FragColor = vec4(c, a*uOp); }`;
const rimFrag = `
uniform vec2 size; uniform float radius, uOp, uHit; uniform vec3 cA, cB, cT; varying vec2 vUv; uniform vec2 full;
float sdR(vec2 p, vec2 b, float r){ vec2 q=abs(p)-b+r; return length(max(q,0.))+min(max(q.x,q.y),0.)-r; }
void main(){ vec2 p=(vUv-0.5)*full; float d=sdR(p, size*0.5, radius);
  vec3 col = mix(cA, cB, smoothstep(0.1, 0.9, vUv.x));
  float line = exp(-abs(d)/1.6)*0.9 + exp(-abs(d)/7.0)*0.25;
  float halo = exp(-max(d,0.0)/26.0)*step(0.0,d)*0.18;
  vec3 c = col*line + mix(col, cT, 0.6)*halo + cT*uHit*(exp(-abs(d)/3.0)*1.4 + exp(-max(d,0.)/40.0)*step(0.0,d)*0.5);
  gl_FragColor = vec4(c*uOp, 1.0); }`;
const vsUv = `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`;

function roundShape(w, h, r) {
  const s = new THREE.Shape(); const x0 = -w / 2, y0 = -h / 2;
  s.moveTo(x0 + r, y0); s.lineTo(x0 + w - r, y0); s.quadraticCurveTo(x0 + w, y0, x0 + w, y0 + r); s.lineTo(x0 + w, y0 + h - r);
  s.quadraticCurveTo(x0 + w, y0 + h, x0 + w - r, y0 + h); s.lineTo(x0 + r, y0 + h); s.quadraticCurveTo(x0, y0 + h, x0, y0 + h - r);
  s.lineTo(x0, y0 + r); s.quadraticCurveTo(x0, y0, x0 + r, y0); return s;
}
// crisp canvas text for frame labels; px: font size in CSS px at the framing distance
function labelTex(html, { px = 52, weight = 650, color = '#f5f3ff', family = "'Inter Display', Inter, sans-serif", w, h, align = 'left', dpr = 2 }) {
  const c = document.createElement('canvas'); c.width = Math.ceil(w * dpr); c.height = Math.ceil(h * dpr);
  const x = c.getContext('2d'); x.scale(dpr, dpr); x.textBaseline = 'middle';
  // html: array of [text, color, weight?, px?] runs
  const runs = typeof html === 'string' ? [[html, color]] : html;
  let total = 0; for (const [s, , wgt, sz] of runs) { x.font = `${wgt || weight} ${sz || px}px ${family}`; total += x.measureText(s).width; }
  let cx = align === 'center' ? (w - total) / 2 : align === 'right' ? w - total : 0;
  for (const [s, col, wgt, sz] of runs) { x.font = `${wgt || weight} ${sz || px}px ${family}`; x.fillStyle = col || color; x.fillText(s, cx, h / 2 + 2); cx += x.measureText(s).width; }
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; return tex;
}

export class Pane {
  constructor(o) {
    this.o = o;
    const W = o.w + 2 * (o.border ?? 14), Hh = o.header ?? 84, Hf = o.footer ?? 0;
    const H = o.h + Hh + Hf + 2 * (o.border ?? 14);
    this.W = W; this.H = H;
    const g = this.group = new THREE.Group();
    this.inner = new THREE.Group(); g.add(this.inner);
    const bd = o.border ?? 14;
    // the picture's centre, relative to the frame centre
    this.picY = H / 2 - bd - Hh - o.h / 2;
    // back plate: dark glass slab with a bevel, lit by the outer scene
    const plateM = new THREE.MeshStandardMaterial({ color: new THREE.Color('#0b0918'), emissive: new THREE.Color('#1a1238'), emissiveIntensity: 0.6, roughness: 0.6, metalness: 0.45, transparent: true, fog: false });
    const plateG = new THREE.ExtrudeGeometry(roundShape(W, H, 34), { depth: 18, bevelEnabled: true, bevelThickness: 6, bevelSize: 6, bevelSegments: 3, curveSegments: 12 });
    plateG.translate(0, 0, -24);
    this.plate = new THREE.Mesh(plateG, plateM); this.inner.add(this.plate);
    // the picture
    this.picM = new THREE.ShaderMaterial({ uniforms: { map: { value: o.tex }, size: { value: new THREE.Vector2(o.w, o.h) }, radius: { value: 22 }, uOp: { value: 1 }, uHit: { value: 0 }, uDim: { value: 1 } },
      vertexShader: vsUv, fragmentShader: sdfFrag, transparent: true, depthWrite: false });
    this.pic = new THREE.Mesh(new THREE.PlaneGeometry(o.w, o.h), this.picM); this.pic.position.set(0, this.picY, 1.5); this.inner.add(this.pic);
    // rim light around the picture (violet left -> orange right), teal on hits
    const pad = 70;
    this.rimM = new THREE.ShaderMaterial({ uniforms: { size: { value: new THREE.Vector2(o.w + 4, o.h + 4) }, full: { value: new THREE.Vector2(o.w + 2 * pad, o.h + 2 * pad) }, radius: { value: 24 }, uOp: { value: 1 }, uHit: { value: 0 },
      cA: { value: C.quill.clone().multiplyScalar(1.1) }, cB: { value: C.ember.clone().multiplyScalar(1.1) }, cT: { value: C.link.clone() } },
      vertexShader: vsUv, fragmentShader: rimFrag, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.rim = new THREE.Mesh(new THREE.PlaneGeometry(o.w + 2 * pad, o.h + 2 * pad), this.rimM); this.rim.position.set(0, this.picY, 2); this.inner.add(this.rim);
    // outer edge light of the whole frame
    this.edgeM = new THREE.ShaderMaterial({ uniforms: { size: { value: new THREE.Vector2(W + 10, H + 10) }, full: { value: new THREE.Vector2(W + 2 * pad, H + 2 * pad) }, radius: { value: 38 }, uOp: { value: 0.55 }, uHit: { value: 0 },
      cA: { value: C.quillHi.clone().multiplyScalar(0.5) }, cB: { value: C.emberHi.clone().multiplyScalar(0.5) }, cT: { value: C.link.clone() } },
      vertexShader: vsUv, fragmentShader: rimFrag, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.edge = new THREE.Mesh(new THREE.PlaneGeometry(W + 2 * pad, H + 2 * pad), this.edgeM); this.edge.position.z = -2; this.inner.add(this.edge);
    // title in the header
    const titleH = Hh;
    this.titleM = new THREE.MeshBasicMaterial({ map: labelTex(o.title, { px: o.titlePx || 50, w: W - 2 * bd - 20, h: titleH, align: 'center', color: '#f2efff' }), transparent: true, depthWrite: false, toneMapped: false, fog: false });
    this.titleM.color.setScalar(1.0);
    this.title = new THREE.Mesh(new THREE.PlaneGeometry(W - 2 * bd - 20, titleH), this.titleM);
    this.title.position.set(0, H / 2 - bd - titleH / 2 + 2, 2); this.inner.add(this.title);
    // footer: big stat + caption lines
    this.footer = [];
    if (Hf > 0 && o.footerRows) {
      let y = this.picY - o.h / 2 - 8;
      for (const row of o.footerRows) {
        const m = new THREE.MeshBasicMaterial({ map: labelTex(row.runs, { px: row.px, w: W - 2 * bd - 28, h: row.h, align: row.align || 'left', weight: row.weight || 650 }), transparent: true, depthWrite: false, toneMapped: false });
        m.color.setScalar(row.bright ?? 0.92);
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(W - 2 * bd - 28, row.h), m);
        mesh.position.set(0, y - row.h / 2, 2); y -= row.h; this.inner.add(mesh); this.footer.push({ mesh, m, row });
      }
    }
    // a soft glow behind the frame (picks up the bloom)
    const gc = document.createElement('canvas'); gc.width = gc.height = 128; const gx = gc.getContext('2d'); const gr = gx.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, 'rgba(255,255,255,0.8)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.18)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); gx.fillStyle = gr; gx.fillRect(0, 0, 128, 128);
    const gt = new THREE.CanvasTexture(gc); gt.colorSpace = THREE.SRGBColorSpace;
    this.glowM = new THREE.MeshBasicMaterial({ map: gt, color: C.link.clone(), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
    this.glow = new THREE.Mesh(new THREE.PlaneGeometry(W * 1.8, H * 1.4), this.glowM); this.glow.position.z = -40; this.inner.add(this.glow);
    this.glow.renderOrder = -1; this.edge.renderOrder = 0; this.plate.renderOrder = 1; this.pic.renderOrder = 2; this.rim.renderOrder = 3; this.title.renderOrder = 3;
    for (const f of this.footer) f.mesh.renderOrder = 3;
    this.group.scale.setScalar(1 / (o.ppu || 150));
  }
  // a: overall opacity, hit: 0..1 flash, dim: picture brightness
  set(a, hit = 0, dim = 1) {
    this.group.visible = a > 0.003;
    this.picM.uniforms.uOp.value = a; this.picM.uniforms.uHit.value = hit; this.picM.uniforms.uDim.value = dim;
    this.rimM.uniforms.uOp.value = a * (0.85 + 0.6 * hit); this.rimM.uniforms.uHit.value = hit;
    this.edgeM.uniforms.uOp.value = a * 0.5; this.edgeM.uniforms.uHit.value = hit * 0.6;
    this.plate.material.opacity = a; this.titleM.opacity = a;
    for (const f of this.footer) f.m.opacity = a * (f.alpha ?? 1);
    this.glowM.opacity = a * (0.12 + 0.35 * hit);
  }
}
