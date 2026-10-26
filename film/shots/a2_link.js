// a2_link (12–40 s, authored in global time): how the link works.
//  12.0–14.25  the strands grow port to port and plug into each other's residual streams (cores); both directions.
//  14.25–16.0  the bodies unfold into their computation: each spine is the residual stream of the newest token; the
//              earlier tokens' streams deal out behind it along a shared time axis. Two lattices face each other:
//              token positions (depth) × 36 blocks (height), the 8 exchange depths in teal. The heads stay on top.
//  16.0–17.5   hold: "one column per token", "36 blocks", "8 exchange depths".
//  17.5–21.25  the prompt pass, both models in lockstep: the wave climbs block by block, self-attention arcs carry
//              information to later positions, and at each of the 8 depths every position of each model reads every
//              partner position, both ways at once. A gold idea zig-zags Quill → Ember → Quill … 8 times.
//  21.25–22.75 both heads emit a token; "8 exchanges per token, both ways at once".
//  22.75–23.75 one generation step (the new columns rise and exchange at every depth).
//  23.75–31.25 the next step, stopped at one exchange (after block 15) and opened up, both directions together:
//              key, value, gate, query, attention weights (with the null key), weighted sum, output, write size,
//              added to the residual stream (shots/a2_exchange.js).
//  31.25–37.5  the write rises; generation runs fast, token after token, constant two-way traffic.
//  37.5–40.0   the lattices fold back into the two creatures (a3's opening pose).
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Pin, FONT } from '../lib/hud.js';
import { tokenTile } from '../lib/tokens.js';
import { C } from '../lib/palette.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, smoother, easeOutCubic, easeInCubic, easeInOutCubic, easeOutBack, noise1, hash } from '../lib/util.js';
import { makeSpline } from './s1_meet_fx.js';
import { blink as blinkAt, yawTo, lerpAng } from './s3_play_kit.js';
import { Lattice, Rays, Glows, UX, H } from './a2_lattice.js';
import { Readout, TR as TX } from './a2_readout.js';
import { Neck } from './a2_neck.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const GOLD = new THREE.Color('#ffc53d');
const RQ = C.link.clone().lerp(C.quill, 0.35), RE = C.link.clone().lerp(C.ember, 0.35); // ray tints by sender
const XW = 2.3, DZ = 1.2;
const xAt = (t) => lerp(2.3, XW, smoother(prog(t, 14.6, 15.9)) * (1 - smoother(prog(t, 38.0, 39.2))));

// ---------------------------------------------------------------- timeline (global seconds, 16th grid)
const T_CONNECT = (k) => 12.25 + 0.25 * k;          // 12.25 … 14.0
const U0 = 14.25, U1 = 16.0;                        // unfold
const L0 = 16.0, L1 = 17.5;                         // labels on the lattice
const P0 = 17.5, PX = (k) => 18.0 + 0.375 * k, PDW = 0.125, PTOP = 21.25; // prompt pass
const CAP = [21.375, 22.75];
const XC = 3;                                       // the exchange opened up: after block 15
const YP = UX[XC] * H;
const F0 = 37.5, F1 = 40.0;
const arrive = 0.26;                                // a read's travel time along a ray                         // fold back

// generation steps: step g computes token position g (tick g) in both models
const STEPS = [];
STEPS.push({ g: 0, s0: 22.75, sd: 0.25, w0: 23.0, w1: 23.7, emit: 23.75 });
STEPS.push({ g: 1, s0: 23.75, sd: 0.25, w0: 24.0, w1: 32.4, emit: 32.5, close: true });
for (let g = 2; g <= 11; g++) { const s0 = 32.5 + 0.5 * (g - 2); STEPS.push({ g, s0, sd: 0.16, w0: s0 + 0.12, w1: s0 + 0.47, emit: s0 + 0.5 }); }
// head hops: at each step the head leaves the old newest column for the new one (window w)
const HOPS = STEPS.map((st) => ({ t0: st.s0, w: st.g >= 2 ? 0.34 : 0.45, g: st.g, land: st.s0 + (st.g >= 2 ? 0.24 : 0.3) }));
const NGEN = STEPS.length; // 12 generated positions per model (ticks 0..11)

const Q_PROMPT = ['The', 'sky', 'is'], E_PROMPT = ['Once', 'up', 'on', 'a'];
const Q_GEN = ['blue', 'and', 'the', 'sea', 'is', 'calm', ',', 'the', 'wind', 'is', 'soft', 'and', 'warm', '.'];
const E_GEN = ['time', ',', 'a', 'lamp', 'lit', 'up', 'the', 'old', 'town', 'and', 'the', 'night', 'was', 'still'];
const Q_PZ = [-2.4, -1.2, 0], E_PZ = [-2.4, -1.6, -0.8, 0];
// the gold idea's path: segment h lives in Quill (h even) or Ember (h odd): [column] or [from, to, height of the
// self-attention hop]. Crossing k joins segment k to segment k+1 at exchange k.
const IDEA = [[0, 1, 3.5 / 36], [1, 2, 10.5 / 36], [2], [3], [1], [2], [2], [3], [2]]; // hops at block 3 and block 10 centres
const IDEA_X = Array.from({ length: 8 }, (_, k) => [IDEA[k][IDEA[k].length === 3 ? 1 : 0], IDEA[k + 1][0]]);

// prompt-pass wave (u: 0..1, >1 = done)
function passU(t) {
  if (t <= P0) return 0;
  if (t < PX(0)) return UX[0] * smoother((t - P0) / (PX(0) - P0));
  for (let k = 0; k < 8; k++) {
    if (t < PX(k) + PDW) return UX[k];
    const nxt = k < 7 ? PX(k + 1) : PTOP, to = k < 7 ? UX[k + 1] : 1.0;
    if (t < nxt) return lerp(UX[k], to, smooth((t - PX(k) - PDW) / (nxt - PX(k) - PDW)));
  }
  return 1.02;
}
// piecewise wave through knots [[t, u], ...] (smooth inside each span), so the exchange hits land on the grid
function knotWave(K, t) {
  if (t <= K[0][0]) return K[0][1];
  for (let i = 1; i < K.length; i++) if (t < K[i][0]) { const [t0, u0] = K[i - 1], [t1, u1] = K[i]; return lerp(u0, u1, smooth((t - t0) / (t1 - t0)) * 0.5 + 0.5 * (t - t0) / (t1 - t0)); }
  return 1.02;
}
// step 0: a 32nd-note roll of exchanges; step 1: exchanges on 16ths, a long stop at block 15, then 16ths again
const KNOTS = {
  0: [[23.0, 0], [23.125, UX[0]], ...UX.slice(1).map((u, k) => [23.125 + 0.0625 * (k + 1), u]), [23.7, 1.0]],
  1: [[24.0, 0], [24.25, UX[0]], [24.375, UX[1]], [24.5, UX[2]], [24.75, UX[3]], [31.0, UX[3]], [31.375, UX[4]], [31.625, UX[5]], [31.875, UX[6]], [32.125, UX[7]], [32.4, 1.0]],
};
// step wave
function stepU(st, t) {
  if (t < st.w0) return 0;
  if (KNOTS[st.g]) return knotWave(KNOTS[st.g], t);
  // fast steps: the 8 exchanges as a 64th-note roll from the third 16th of the beat
  const s0 = st.s0;
  return knotWave([[st.w0, 0], [s0 + 0.1875, UX[0]], ...UX.slice(1).map((u, k) => [s0 + 0.1875 + 0.03125 * (k + 1), u]), [s0 + 0.46, 1.0]], t);
}
// time the step wave crosses height u (bisection on the monotonic wave)
function crossT(f, a, b, u) { if (f(b) < u) return Infinity; for (let i = 0; i < 40; i++) { const m = (a + b) / 2; if (f(m) < u) a = m; else b = m; } return b; }
const STEP_FIRE = STEPS.map((st) => UX.map((u) => crossT((t) => stepU(st, t), st.w0, st.w1 + 0.01, u)));
if (STEP_FIRE[1]) STEP_FIRE[1][XC] = Infinity; // the opened-up exchange is drawn by a2_exchange

// how many generated columns have slid in (smooth)
const slideS = (t) => STEPS.reduce((s, st) => s + smoother(prog(t, st.s0, st.s0 + st.sd)), 0);

// camera keys: target g, azimuth (0 = front, -pi/2 = Quill's side), elevation, distance
const YC = YP;
const K = (t, g, az, el, d, fov = 35) => ({ t, g, a: V(az, el, d), f: V(fov, 0, 0) });
const PI2 = Math.PI / 2;
const CAM = [
  K(12.0, V(0, 2.4, 0), 0.0329, 0.0394, 15.22),        // a1's last frame
  K(13.6, V(0, 2.4, 0), 0.02, 0.05, 14.4),
  K(14.4, V(0, 2.3, -0.3), -0.16, 0.12, 13.6),
  K(16.0, V(0, 2.1, -0.9), -1.42, 0.78, 12.8),       // the orbit is the unfold
  K(17.4, V(0, 2.1, -0.9), -1.46, 0.79, 12.6),
  K(19.4, V(0, 2.1, -0.85), -1.52, 0.8, 12.4),
  K(21.3, V(0, 2.15, -0.8), -1.58, 0.8, 12.3),
  K(22.7, V(0, 2.15, -0.9), -1.6, 0.8, 12.3),
  K(23.8, V(0, 1.95, -1.3), -1.56, 0.5, 12.6, 33),
  K(24.8, V(0, 2.1, -1.5), -PI2, 0.18, 11.6, 32),         // facing the read-out panels
  K(27.0, V(0, 2.1, -1.5), -PI2, 0.18, 11.45, 32),
  K(29.4, V(0, 2.1, -1.5), -PI2, 0.18, 11.35, 32),
  K(31.25, V(0, 2.1, -1.5), -PI2, 0.2, 11.5, 32),
  K(32.6, V(0, 1.9, -2.2), -1.45, 0.84, 12.8),
  K(35.0, V(0, 1.9, -3.0), -1.36, 0.85, 13.4),
  K(37.3, V(0, 2.0, -3.3), -1.28, 0.84, 13.8),
  K(38.2, V(0, 2.3, -1.0), -1.1, 0.55, 13.6),
  K(39.3, V(0.1, 3.1, 0), -0.06, 0.02, 14.6),
  K(40.0, V(0.1, 3.2, 0), -0.0203, 0.00676, 14.80),    // a3's first frame
  K(41.0, V(0.1, 3.2, 0), -0.0203, 0.00676, 14.62),
];
const camSpline = makeSpline(CAM);

export async function create({ hud, scale }) {
  try { await document.fonts.load('600 64px Inter'); await document.fonts.load('700 52px Inter'); } catch (_) { /* optional */ }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200); scene.add(camera);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group); q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  const dot = Creature.dotTexture();
  const LQ = new Lattice(scene, 'quill', -2.3, [...Q_PROMPT, ...Q_GEN.slice(0, NGEN)]);
  const LE = new Lattice(scene, 'ember', 2.3, [...E_PROMPT, ...E_GEN.slice(0, NGEN)]);
  const rays = new Rays(scene, 700), glows = new Glows(scene, 700, scale || 1);
  const nQP = Q_PROMPT.length, nEP = E_PROMPT.length;
  // the opened-up exchange: step 1 (tick 1). Columns newest first; ages 0, 1, then the prompt's bucket.
  const xspec = {
    quill: { cols: [nQP + 1, nQP, 2, 1, 0], reader: nQP + 1, gate: [0.88, 0.93, 0.62, 0.84, 0.3],
      w: [0.27, 0.21, 0.07, 0.05, 0.19, 0.12], w0: 0.09, age: [1, 0.7, 0.45, 0.45, 0.45, 0.45] },
    ember: { cols: [nEP + 1, nEP, 3, 2, 1, 0], reader: nEP + 1, gate: [0.9, 0.86, 0.72, 0.35, 0.92, 0.8],
      w: [0.31, 0.24, 0.08, 0.17, 0.08], w0: 0.12, age: [1, 0.7, 0.45, 0.45, 0.45] },
  };
  // the read-out panels sit in front of the camera's pose at 27.0
  const refCam = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  { const g = camSpline(27.0, (k) => k.g), a = camSpline(27.0, (k) => k.a);
    refCam.position.copy(g).add(V(Math.cos(a.y) * Math.sin(a.x), Math.sin(a.y), Math.cos(a.y) * Math.cos(a.x)).multiplyScalar(a.z));
    refCam.lookAt(g); refCam.fov = camSpline(27.0, (k) => k.f).x; refCam.updateProjectionMatrix(); refCam.updateMatrixWorld(true); }
  const ex = new Readout(scene, hud, { quill: LQ, ember: LE }, xspec, refCam);
  const bez = (a, c, b, s, out = new THREE.Vector3()) => out.copy(a).multiplyScalar((1 - s) * (1 - s)).addScaledVector(c, 2 * s * (1 - s)).addScaledVector(b, s * s);
  // the crossing of the idea at exchange k (sender's last column of segment k -> reader's first column of segment k+1)
  const crossCurve = (k) => {
    const Ls = k % 2 === 0 ? LQ : LE, Lr = k % 2 === 0 ? LE : LQ, si = IDEA[k][IDEA[k].length === 3 ? 1 : 0], ri = IDEA[k + 1][0];
    const a = Ls.node(si, UX[k]), b = Lr.node(ri, UX[k]), dirUp = Ls === LQ ? 1 : -1;
    const c = a.clone().lerp(b, 0.5); c.y += dirUp * 0.2; c.z += 0.1 * dirUp;
    return { a, b, c, Ls, Lr, si, ri, p: (s) => bez(a, c, b, s) };
  };
  // self-attention arc shape inside one lattice (in the wall, bulging up and a little outward)
  const saCtrl = (L, a, b) => { const span = Math.abs(b.z - a.z), c = a.clone().lerp(b, 0.5); c.y += 0.22 + 0.32 * span; c.x += L.sgn * (0.12 + 0.12 * span); return c; };
  // prompt pass: when the wavefront crosses block i, each position attends to itself and every earlier position
  const BLOCK_T = Array.from({ length: NBLOCKS }, (_, i) => crossT(passU, P0, PTOP, (i + 0.5) / NBLOCKS));
  const arcT = IDEA.map((sg) => (sg.length === 3 ? crossT(passU, P0, PTOP, sg[2]) : Infinity));
  function ideaAt(tau) {
    let h = 0;
    for (let k = 0; k < 8; k++) {
      if (tau < PX(k)) break;
      if (tau < PX(k) + arrive) return { p: crossCurve(k).p((tau - PX(k)) / arrive), cross: k };
      h = k + 1;
    }
    const L = h % 2 ? LE : LQ, sg = IDEA[h], u0 = h === 0 ? 0.0 : UX[h - 1], uTop = h < 8 ? UX[h] : 1;
    const w = clamp(passU(tau), u0, Math.min(uTop, 0.999));
    if (sg.length === 3) {
      const ta = arcT[h];
      if (tau >= ta && tau < ta + 0.14) {
        const a = L.node(sg[0], sg[2]), b = L.node(sg[1], sg[2]), span = Math.abs(b.z - a.z);
        const c = saCtrl(L, a, b);
        return { p: bez(a, c, b, (tau - ta) / 0.14) };
      }
      return { p: L.node(tau < ta ? sg[0] : sg[1], tau < ta ? Math.min(w, sg[2]) : Math.max(w, sg[2])) };
    }
    return { p: L.node(sg[0], w) };
  }

  // connection flashes (12.25–14.5)
  const cflash = XBLOCKS.map(() => [0, 1].map(() => { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: dot, color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); scene.add(s); return s; }));
  // the gold idea's zig-zag (prompt pass)
  const zig = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: GOLD.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  zig.renderOrder = 6; scene.add(zig);
  const chevrons = XBLOCKS.map(() => { const m = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.28, 14), new THREE.MeshBasicMaterial({ color: GOLD.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); scene.add(m); return m; });
  // necks: the head grows out of the top of the newest column (current neck, the neck it is leaving, two fading stubs)
  const necks = { quill: [0, 1, 2, 3].map(() => new Neck(scene, C.quill, C.quillHi)), ember: [0, 1, 2, 3].map(() => new Neck(scene, C.ember, C.emberHi)) };
  // "36 blocks" bracket
  const brMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#ddd6fe'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const bracket = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), brMat); scene.add(m); return m; });
  // emitted token tiles (one per generated token and side): they leave the head, then become the next column's input
  const emitQ = Q_GEN.map((w) => { const m = tokenTile(w, 'quill', 0.5); m.renderOrder = 12; m.material.fog = false; scene.add(m); return m; });
  const emitE = E_GEN.map((w) => { const m = tokenTile(w, 'ember', 0.5); m.renderOrder = 12; m.material.fog = false; scene.add(m); return m; });

  // HUD
  const mkPin = (html, color, o = {}) => new Pin(hud, html, { size: 50, weight: 700, font: FONT.display, color, dot: false, anchor: 'center', ...o });
  const pinCols = mkPin('one column per token', '#f5f3ff');
  const pinBlocks = mkPin('36 layers', '#ddd6fe', { anchor: 'right' });
  const pinX = mkPin('exchange depths', '#7ff7d2', { anchor: 'left' });
  const cap = new Caption(hud, { head: '8 exchanges,', size: 92, pos: { left: 120, top: 110 }, width: 1000, align: 'left' });
  const cap2 = new Caption(hud, { head: `<span style="color:#7ff7d2">both ways</span>`, size: 92, pos: { left: 120, top: 210 }, width: 1000, align: 'left' });

  const tmp = new THREE.Vector3();
  const emitT = (g) => (g === 0 ? PTOP : STEPS[g - 1].emit);           // when the token that is column g's input appears
  const sideOf = (L) => (L === LQ ? 'quill' : 'ember');

  // z of a lattice column at time tt (pure; mirrors setupCols): the newest prompt column (-1) or generated column g
  const foldZAt = (tt) => smoother(prog(tt, F0 + 0.05, F0 + 0.8));
  const zCol = (g, tt) => lerp((g < 0 ? 0 : DZ * (g + 1)) - DZ * slideS(tt), 0, foldZAt(tt));
  // where the head's neck is anchored along z at time tt: springs from the old newest column to the new one
  function headZ(tt) {
    let h = -1; for (let i = 0; i < HOPS.length; i++) if (tt >= HOPS[i].t0) h = i;
    if (h < 0) return zCol(-1, tt);
    const hp = HOPS[h], zA = zCol(h - 1, tt), zB = zCol(h, tt);
    return lerp(zA, zB, spring(tt, hp.t0 + 0.08 * hp.w, hp.w > 0.4 ? 2.4 : 3.2, 0.42));
  }
  const OFFB = V(0, 0.05, 0.8);                                  // head base relative to its column's top (on the lattice)
  const neckCurve = (top, base, vz, m = 1) => {
    const p0 = top.clone().add(V(0, -0.08, 0)), p1 = top.clone().add(V(0, 0.42 * m + 0.02, 0));
    const p2 = base.clone().add(V(0, -0.24 * m, -0.12 * m - clamp(0.015 * vz, -0.1, 0.1)));
    return new THREE.CubicBezierCurve3(p0, p1, p2, base.clone());
  };
  const _gq = new THREE.Quaternion(), _lq = new THREE.Quaternion(), UPV = V(0, 1, 0);
  function headRig(c, L, np, side, t, m) {
    const pool = necks[side];
    for (const n of pool) n.hide();
    if (m <= 0.002) return;
    const op = c.pose.opacity;
    const topOf = (g, tt = t) => V(L.x, H, zCol(g, tt));
    const hz = headZ(t), vz = (headZ(t + 0.01) - headZ(t - 0.01)) / 0.02;
    let h = -1; for (let i = 0; i < HOPS.length; i++) if (t >= HOPS[i].t0) h = i;
    const anchorTop = V(L.x, H, hz);
    // the lattice head: base ahead of the top, centre along the neck's end tangent (leaning into the motion)
    const base = anchorTop.clone().add(OFFB).lerp(anchorTop.clone().add(V(0, 0.02, 0)), 1 - m);
    const cur = neckCurve(h < 0 ? topOf(-1) : topOf(h), base, vz, m);
    const tan = cur.getTangentAt(1).normalize().add(V(0, 0, clamp(vz * 0.03, -0.15, 0.15))).normalize();
    const hs = lerp(1, 0.68, m), centre = base.clone().addScaledVector(tan, 0.55 * hs * 0.82);
    // blend with the creature's own head (m = 0 on the body, 1 on the lattice)
    c.group.updateMatrixWorld(true); c.group.getWorldQuaternion(_gq);
    const tanL = tan.clone().applyQuaternion(_gq.clone().invert());
    _lq.setFromUnitVectors(UPV, tanL);
    const lq = _lq.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(c.pose.headNod, 0, c.pose.headTilt, 'ZXY')));
    c.head.position.lerp(c.group.worldToLocal(centre.clone()), m);
    c.head.quaternion.slerp(lq, m);
    let land = 0; if (h >= 0) land = kick(t, HOPS[h].t0 + 0.4 * HOPS[h].w, 3.2, 6);
    c.head.scale.multiplyScalar(hs).multiply(V(1 + 0.1 * land, 1 - 0.13 * land, 1 + 0.1 * land));
    c.group.updateMatrixWorld(true);
    // necks
    const amp = m * op;
    if (h < 0) { pool[0].set(cur, 1, amp, t); return; }
    const hp = HOPS[h], w = hp.w;
    const grow = smoother(prog(t, hp.t0, hp.t0 + 0.3 * w));
    pool[0].set(cur, grow, amp, t);
    // the neck it is leaving: retracts to a short stub on the old column, which fades
    for (let k = 0; k < 3; k++) {
      const hh = h - k; if (hh < 0) break;
      const hq = HOPS[hh], det = smoother(prog(t, hq.t0 + 0.28 * hq.w, hq.t0 + 0.62 * hq.w));
      const fade = 1 - prog(t, hq.t0 + 0.62 * hq.w, hq.t0 + 0.62 * hq.w + 0.9);
      if (fade <= 0.003) continue;
      const oldTop = topOf(hh - 1), rest = oldTop.clone().add(OFFB);
      // while it retracts it keeps the shape it had to the head (it shrinks along that spline); stubs keep their rest shape
      const end = k === 0 ? base.clone().lerp(rest, det * det * det) : rest;
      pool[k + 1].set(neckCurve(oldTop, end, 0), k === 0 ? lerp(1, 0.22, det) : 0.22, amp * fade * (k === 0 ? 1 : 0.8), t);
    }
  }

  function update(t) {
    world.update(t);
    const post = { bloomStrength: 0.85, bloomRadius: 0.5, bloomThreshold: 0.22, vignette: 0.95 };
    const bp = beatPulse(t, 6);
    // ---------------- camera
    const cg = camSpline(t, (k) => k.g), ca = camSpline(t, (k) => k.a);
    const cp = cg.clone().add(V(Math.cos(ca.y) * Math.sin(ca.x), Math.sin(ca.y), Math.cos(ca.y) * Math.cos(ca.x)).multiplyScalar(ca.z));
    const jig = 1 - prog(t, 38.8, 39.6); cp.x += noise1(t * 0.45, 1) * 0.05 * jig; cp.y += noise1(t * 0.4, 2) * 0.04 * jig;
    const hitK = kick(t, TX.hit, 8, 7) * 0.04; cp.y += hitK;
    camera.position.copy(cp); camera.lookAt(cg); camera.fov = camSpline(t, (k) => k.f).x; camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);

    // ---------------- phases
    const unfold = prog(t, U0, U1);                        // 0 creature .. 1 lattice
    const fold = prog(t, F0, F1);
    const latOn = clamp(prog(t, 14.45, 14.9) * (1 - prog(t, 38.9, 39.4)));
    const bodyVis = 1 - prog(t, 14.5, 15.4) + prog(t, 38.35, 39.2);
    world.floor.position.y = -1.7 * smooth(prog(t, 14.4, 15.8)) * (1 - smooth(prog(t, 38.2, 39.6)));
    const S = slideS(t);
    const XX = xAt(t); LQ.x = -XX; LE.x = XX;

    // ---------------- lattice columns
    const foldZ = smoother(prog(t, F0 + 0.05, F0 + 0.8)), foldOut = prog(t, F0 + 0.3, F0 + 0.95);
    const setupCols = (L, PZ, np) => {
      L.begin();
      for (const c of L.cols) {
        const isPrompt = c.i < np, g = c.i - np;
        let z, appear, wave;
        if (isPrompt) {
          const rank = np - 1 - c.i;                         // 0 = newest prompt token (the creature's spine)
          const deal = rank === 0 ? 1 : smoother(prog(t, 14.85 + 0.13 * rank, 15.65 + 0.13 * rank));
          z = PZ[c.i] * deal - DZ * S;
          appear = rank === 0 ? latOn : latOn * clamp(deal * 3);
          wave = passU(t);
        } else {
          z = DZ * (g + 1) - DZ * S;
          appear = latOn * smoother(clamp((S - g) * 1.25));
          wave = stepU(STEPS[g], t);
        }
        // fold: every column slides into the newest one and fades
        const isFront = !isPrompt && g === NGEN - 1;
        z = lerp(z, 0, foldZ); if (!isFront) appear *= (1 - smooth(clamp(foldZ * 1.7))) * (1 - foldOut);
        // far columns fade into the dark
        appear *= clamp(1 + (z + 7.5) / 3.5);
        c.z = z; c.appear = appear; c.wave = wave; c.front = wave > 0.002 && wave < 0.999 ? 1 : 0; c.amp = 1;
        // the spine (newest column) shrinks from vertebra size to lattice size, and back at the end
        const big = isPrompt && c.i === np - 1 ? 1 - prog(t, 14.7, 15.7) : isFront ? prog(t, F0 + 0.8, F0 + 2.0) : 0;
        c.discR = lerp(0.15, 0.42, smooth(big));
        c.tile = isPrompt ? 1 : clamp((t - HOPS[g].land) / 0.06);
      }
    };
    setupCols(LQ, Q_PZ, nQP); setupCols(LE, E_PZ, nEP);
    for (const L of [LQ, LE]) {
      const st = L.state;
      st.amp = 1; st.rows = latOn * prog(t, 15.0, 15.8) * (1 - foldOut); st.discs = 1; st.tiles = prog(t, 15.4, 15.9) * (1 - prog(t, F0, F0 + 0.4)) * (1 - win(t, 24.2, 31.6, 0.3));
      st.rowExtra = 0.25 * win(t, L0 + 0.5, L1, 0.2);
      // the close-up: a section cut through the exchange depth
      st.cutY = UX[XC]; st.cutW = 0.02; st.cutDim = 1;
      st.amp = lerp(1, 0.1, win(t, 24.55, 31.65, 0.35));
    }

    rays.begin(); glows.begin();
    // ---------------- writes, rays and arcs
    const writeAmt = 0.11;
    const fireEnv = (tf) => (t < tf - 0.05 ? 0 : t < tf + 0.08 ? (t - tf + 0.05) / 0.13 : Math.exp(-(t - tf - 0.08) * 3.2));
    const ray = (Ls, si, Lr, ri, u, tf, alpha, o = {}) => {
      const a = Ls.node(si, u), b = Lr.node(ri, u), dirUp = Ls === LQ ? 1 : -1;
      const c = a.clone().lerp(b, 0.5); c.y += dirUp * 0.2; c.z += 0.1 * dirUp;
      const g1 = clamp((t - tf + 0.06) / 0.2);
      rays.add(a, c, b, { width: o.width ?? 0.006, color: o.color ?? (Ls === LQ ? RQ : RE), alpha, g1, pulse: clamp((t - tf) / arrive) * 1.05, pulseColor: o.pc ?? Ls.hi, pulseAmt: o.pa ?? 1.3 });
    };
    // self-attention arcs at the wavefront: every earlier column of the same model -> the computing column(s)
    const arcsAt = (L, readers, u, alpha) => {
      if (u <= 0.003 || u >= 0.999) return;
      for (const ri of readers) for (const c of L.cols) {
        if (c.i >= ri || c.appear < 0.2) continue;
        const a = L.node(c.i, u), b = L.node(ri, u), span = Math.abs(b.z - a.z);
        const ctrl = a.clone().lerp(b, 0.5); ctrl.y += 0.12 + 0.14 * span; ctrl.x += L.sgn * (0.12 + 0.12 * span);
        rays.add(a, ctrl, b, { width: 0.005, color: L.hi, alpha: alpha * 0.55 * c.appear, pulse: (t * 1.8 + c.i * 0.17) % 1, pulseColor: L.hi, pulseAmt: 1.2 });
      }
    };
    // prompt pass
    {
      const u = passU(t);
      const qr = Array.from({ length: nQP }, (_, i) => i), er = Array.from({ length: nEP }, (_, i) => i);
      if (t > P0 && t < PTOP + 0.4) {
        // a soft fan of self-attention arcs at each block as the front passes it: from every earlier position to each
        // later one of the same model (causal), in the model's own colour; brightness is a fixed illustrative pattern
        for (let i = 0; i < NBLOCKS; i++) {
          const tb = BLOCK_T[i], dt = t - tb; if (dt < -0.04 || dt > 0.42) continue;
          const env = clamp((dt + 0.04) / 0.08) * Math.exp(-Math.max(0, dt - 0.04) * 7.5);
          const ub = (i + 0.5) / NBLOCKS;
          for (const [L, n, seed] of [[LQ, nQP, 1.3], [LE, nEP, 7.1]]) {
            const col = L === LQ ? C.quillHi.clone().lerp(C.quill, 0.25).multiplyScalar(1.25) : L.col.clone().lerp(L.hi, 0.35);
            for (let b = 1; b < n; b++) for (let a0 = 0; a0 < b; a0++) {
              const w = 0.35 + 0.65 * hash(i * 13.1 + a0 * 3.7 + b * 1.9 + seed);
              const isIdea = L === LQ && ((i === 3 && a0 === 0 && b === 1) || false);
              const A = L.node(a0, ub), B = L.node(b, ub);
              rays.add(A, saCtrl(L, A, B), B, { width: 0.008 + 0.014 * w, color: isIdea ? GOLD : col, alpha: (isIdea ? 1.0 : 0.85 * w) * env * L.cols[b].appear,
                g1: clamp((dt + 0.04) / 0.14), pulse: clamp(dt / 0.3), pulseColor: isIdea ? GOLD : L.hi, pulseAmt: 1.6, taper: 0.6 });
            }
          }
        }
      }
      for (let k = 0; k < 8; k++) {
        const tf = PX(k); if (t < tf - 0.05) continue;
        const env = fireEnv(tf);
        for (const [Ls, Lr, ns, nr] of [[LQ, LE, nQP, nEP], [LE, LQ, nEP, nQP]]) {
          for (let r = 0; r < nr; r++) {
            if (env > 0.01) for (let s = 0; s < ns; s++) {
              const cc = (Ls === LQ) === (k % 2 === 0) ? IDEA_X[k] : null;
              const isGold = cc && cc[0] === s && cc[1] === r;
              ray(Ls, s, Lr, r, UX[k], tf, (isGold ? 0.0 : 0.55) * env * Lr.cols[r].appear);
            }
            // the write is added into the reader's stream and rises with its wave
            if (t > tf + arrive) Lr.write(r, UX[k], Ls.col, writeAmt, Math.max(UX[k] + 0.006, Math.min(Lr.cols[r].wave, 1)), 0);
            const fl = t > tf + arrive ? Math.exp(-(t - tf - arrive) * 7) : 0;
            if (fl > 0.02) glows.add(Lr.node(r, UX[k]), Ls.hi, 0.3, 0.8 * fl * Lr.cols[r].appear);
          }
        }
      }
    }
    // generation steps
    for (const st of STEPS) {
      if (t < st.w0 - 0.05) continue;
      const g = st.g, qi = nQP + g, ei = nEP + g;
      const fires = STEP_FIRE[g];
      const uq = LQ.cols[qi].wave;
      const quiet = st.close ? 1 - win(t, 24.6, 31.2, 0.25) : 1;   // the paused step stays quiet behind the read-out
      if (uq > 0 && uq < 1 && quiet > 0.01) {
        const fast = st.g >= 2 ? 1 : 0;
        arcsAt(LQ, [qi], uq, (1 - 0.3 * fast) * quiet); arcsAt(LE, [ei], LE.cols[ei].wave, (1 - 0.3 * fast) * quiet);
        glows.add(LQ.node(qi, uq), LQ.hi, 0.5, 0.9 * quiet); glows.add(LE.node(ei, LE.cols[ei].wave), LE.hi, 0.5, 0.9 * quiet);
      }
      for (let k = 0; k < 8; k++) {
        const tf = fires[k];
        if (st.close && k === XC && t > TX.hit && t < st.w1 + 0.2) {
          // the write that was just added rises with the wave in both newest columns
          const rise = 1 - prog(t, st.w1 - 0.3, st.w1 + 0.2);
          glows.add(LQ.node(qi, LQ.cols[qi].wave), LE.hi, 0.75, 1.3 * rise); glows.add(LE.node(ei, LE.cols[ei].wave), LQ.hi, 0.75, 1.3 * rise);
        }
        if (st.close && k === XC) {
          // written by the close-up at TX.hit
          if (t > TX.hit) { LE.write(ei, UX[k], LQ.col, writeAmt * 2.2, Math.max(UX[k] + 0.006, Math.min(LE.cols[ei].wave, 1))); LQ.write(qi, UX[k], LE.col, writeAmt * 2.2, Math.max(UX[k] + 0.006, Math.min(LQ.cols[qi].wave, 1))); }
          continue;
        }
        if (!isFinite(tf) || t < tf - 0.05) continue;
        const env = fireEnv(tf);
        for (const [Ls, Lr, ri] of [[LQ, LE, ei], [LE, LQ, qi]]) {
          if (env > 0.01) for (const c of Ls.cols) {
            if (c.appear < 0.05) continue;
            const sg = c.i - (Ls === LQ ? nQP : nEP); if (sg > g) continue;          // only partner tokens of the same tick or earlier
            ray(Ls, c.i, Lr, ri, UX[k], tf, (g >= 2 ? 0.42 : 0.7) * quiet * env * c.appear * Lr.cols[ri].appear, { width: (g >= 2 ? 0.0045 : 0.006) + (sg === g ? 0.003 : 0) });
          }
          if (t > tf + arrive) Lr.write(ri, UX[k], Ls.col, writeAmt, Math.max(UX[k] + 0.006, Math.min(Lr.cols[ri].wave, 1)));
          const fl = t > tf + arrive ? Math.exp(-(t - tf - arrive) * 7) : 0;
          if (fl > 0.02) glows.add(Lr.node(ri, UX[k]), Ls.hi, 0.45, 1.4 * fl);
        }
      }
    }
    // the gold idea: starts in Quill's first token, moves to a later position by self-attention, crosses at every
    // exchange (alternating directions), and ends in Quill: Quill -> Ember -> Quill ... 8 crossings in one pass
    const zigOn = (t > P0 ? 1 : 0) * (1 - prog(t, 22.6, 23.0));
    zig.visible = false; chevrons.forEach((m) => { m.visible = false; });
    if (zigOn > 0) {
      const pts = [], N = 260, tEnd = Math.min(t, PTOP);
      for (let n = 0; n <= N; n++) { const tau = P0 + 0.02 + (tEnd - P0 - 0.02) * n / N; const v = ideaAt(tau).p; if (!pts.length || v.distanceTo(pts[pts.length - 1]) > 0.02) pts.push(v); }
      if (pts.length >= 2) {
        zig.geometry.dispose(); zig.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, false, 'centripetal'), Math.min(700, pts.length * 3), 0.024, 6, false);
        zig.visible = true; zig.material.opacity = 0.85 * zigOn;
      }
      if (t < PTOP + 0.1) glows.add(ideaAt(tEnd).p, GOLD, 0.6 + 0.1 * Math.sin(t * 14), 1.3 * zigOn);
      XBLOCKS.forEach((_, k) => {
        const ch = chevrons[k], ta = PX(k) + arrive * 0.5; if (tEnd < ta) return;
        const cv = crossCurve(k);
        ch.position.copy(cv.p(0.5)); ch.quaternion.setFromUnitVectors(V(0, 1, 0), cv.b.clone().sub(cv.a).normalize());
        ch.scale.setScalar(easeOutBack(prog(t, ta, ta + 0.2), 2)); ch.visible = true; ch.material.opacity = zigOn;
      });
      // the idea also stays in every stream it entered (a gold tint that rises with the wave)
      for (let h = 0; h <= 8; h++) {
        const t0 = h === 0 ? P0 : PX(h - 1) + arrive; if (t < t0) break;
        const L = h % 2 ? LE : LQ, sg = IDEA[h], u0 = h === 0 ? 0 : UX[h - 1], top = Math.max(u0 + 0.006, Math.min(passU(t), 1));
        if (sg.length === 3) { L.write(sg[0], u0, GOLD, 0.2 * zigOn, Math.min(top, sg[2])); if (top > sg[2]) L.write(sg[1], sg[2], GOLD, 0.2 * zigOn, top); }
        else L.write(sg[0], u0, GOLD, 0.2 * zigOn, top);
      }
    }

    // ---------------- the opened-up exchange
    const exAmp = win(t, 24.65, 31.35, 0.3);
    const xr = ex.update(t, camera, exAmp, { rays, glows });
    // the states after block 15 lift out of both lattices into the panels
    if (exAmp > 0.003 && t < TX.cards + 0.35) {
      for (const [reader, Ls, Lr] of [['quill', LE, LQ], ['ember', LQ, LE]]) {
        const sp = xspec[reader], cols = xspec[reader === 'quill' ? 'ember' : 'quill'].cols.slice().reverse();
        cols.forEach((ci, j) => { const u = prog(t, TX.lift + 0.03 * j, TX.cards + 0.04 * j + 0.06, easeInOutCubic); if (u <= 0 || u >= 1) return;
          glows.add(Ls.node(ci, UX[XC]).lerp(xr[reader].stateW(j), u), Ls.hi, 0.3, 1.5); });
        const u = prog(t, TX.lift, TX.cards + 0.2, easeInOutCubic);
        if (u > 0 && u < 1) glows.add(Lr.node(sp.reader, UX[XC]).lerp(xr[reader].readerW, u), Lr.hi, 0.36, 1.6);
      }
    }

    LQ.end(t, camera); LE.end(t, camera); rays.end(); glows.end();

    // ---------------- creatures (bodies before the unfold and after the fold; heads on top of the newest columns)
    const T3 = t - 8; // a3's clock, for the hand-off
    for (const [c, sgn] of [[q, -1], [e, 1]]) {
      const p = c.pose; Object.assign(p, Creature.defaultPose(c.side));
      p.position.set(sgn * XX, 0, 0);
      const latPhase = clamp(unfold * (1 - fold));
      p.body = clamp(bodyVis);
      p.opacity = 1 - win(t, 23.9, 31.6, 0.45);
      p.xray = lerp(0.7, 0.5, prog(t, 38.6, 39.8));
      p.glow = lerp(1.5, 1, prog(t, 12.0, 13.5)); p.coreGlow = lerp(1.45, 0.8, prog(t, 12.0, 13.5));
      p.portGlow = (t < 14.5 ? 0.6 + 0.15 * bp : 0.28);
      p.sway = 0.08 * (1 - latPhase); p.swayPhase = sgn < 0 ? t * 2.3 : t * 2.3 + 3.4;
      let sq = 1 - 0.022 * bp;
      for (let k = 0; k < 8; k++) sq += 0.025 * kick(t, T_CONNECT(k), 3.2, 6);
      // emits: a little stretch as each token leaves the head
      for (let g = 0; g <= NGEN; g++) { const te = g === 0 ? PTOP : STEPS[g - 1].emit; if (Math.abs(t - te) < 1.2) sq += 0.045 * kick(t, te - 0.03, 2.6, 5); }
      p.squash = sq;
      p.blink = blinkAt(t, sgn < 0 ? 3 : 4.7);
      p.happy = 0.75 * (1 - prog(t, 12.15, 12.4)) + 0.7 * win(t, PTOP + 0.05, CAP[1] - 0.2, 0.15);
      if (p.happy > 0.5) p.blink = 0;
      // gaze: the newest connection, then the wavefront, the partner at emits
      const camYaw = yawTo(p.position, camera.position), toPartner = yawTo(p.position, V(-sgn * 2.3, 0, 0));
      // on the body: turned a bit toward the partner; on the lattice: toward the camera and ahead (the way time runs)
      p.yaw = lerpAng(lerpAng(camYaw, toPartner, 0.35), lerpAng(camYaw, 0, 0.3), latPhase);
      p.lean = 0.03 * Math.sin(t * 1.2 + (sgn < 0 ? 0 : 1.7)) * (1 - latPhase);
      p.headTilt = 0.06 * Math.sin(t * 0.9 + (sgn < 0 ? 0 : 3.4));
      p.headNod = -0.42 * latPhase;                 // on the lattice the camera is above: lift the face toward it
      let look;
      if (t < 14.3) { const kc = clamp(Math.floor((t - 12.1) / 0.25), 0, 7); look = V(0, (XBLOCKS[kc] + 0.5) / 36 * c.H, 0.6); }
      else if (t < PTOP) look = V(sgn * 0.6, lerp(1.5, 4.2, clamp(passU(t))), -1.0);
      else if (t < 23.0) look = V(-sgn * XX, 4.8, 0);
      else if (t < 31.3) look = V(-sgn * 0.6, YP, 0.5);
      else if (t < F0 + 1.0) look = V(sgn * 3.2, 5.6, 1.2);
      else look = (sgn < 0 ? e : q).headWorld(new THREE.Vector3());
      p.lookAt = look;
      // hand-off to a3: its base life at its time 32
      const ho = prog(t, 38.6, 39.8);
      if (ho > 0) {
        p.squash = lerp(p.squash, 1 - 0.022 * beatPulse(T3 + 8, 6), ho);
        p.lean = lerp(p.lean, 0.03 * Math.sin(T3 * 1.2 + (sgn < 0 ? 0 : 1.7)), ho);
        p.headTilt = lerp(p.headTilt, 0.06 * Math.sin(T3 * 0.9 + (sgn < 0 ? 0 : 1.7) * 2.1), ho);
        p.sway = 0.08; p.swayPhase = T3 * 2.3 + (sgn < 0 ? 0 : 1.7) * 2;
        p.blink = blinkAt(T3, (sgn < 0 ? 0 : 1.7) + 3);
        p.groundGlow = 0.45;
      }
      c.inject = [];
    }
    // connection: a flash of the sender's colour in each core, then a faint ribbon of it
    if (t < U1) XBLOCKS.forEach((b, k) => {
      const tc = T_CONNECT(k); if (t < tc) return;
      const f = Math.exp(-4 * (t - tc)), top = ((b + 0.5) / 36 * 36) / 37 + 0.02 + 0.06 * f;
      e.inject.push({ block: b, color: C.quill, amount: 0.25 + 1.2 * f, top });
      q.inject.push({ block: b, color: C.ember, amount: 0.25 + 1.2 * f, top });
    });
    // the hair lifts as the strands plug in (springs past upright, then settles toward the partner)
    q.pose.hair = e.pose.hair = lerp(0.62, 1, spring(t, T_CONNECT(3), 1.5, 0.38));
    q.update(t); e.update(t);
    // the head rides on a neck that grows out of the top of the newest column (see headRig)
    const headK = smoother(prog(t, 14.9, 15.9)) * (1 - smoother(prog(t, 38.2, 39.1)));
    headRig(q, LQ, nQP, 'quill', t, headK); headRig(e, LE, nEP, 'ember', t, headK);
    { e.group.updateMatrixWorld(true); const wq = new THREE.Quaternion(); e.crest.getWorldQuaternion(wq); e.flame.quaternion.copy(wq.invert().multiply(camera.quaternion)); }

    // ---------------- link strands (before the unfold, after the fold)
    const ls = link.state;
    ls.depth = 1; ls.sag = -0.2; ls.twist = 0.07; ls.packetSize = 0.16; ls.pulses = []; ls.highlight = -1; ls.tintAB = ls.tintBA = null; ls.aToB = ls.bToA = 1;
    ls.grow = XBLOCKS.map((_, k) => (t > 30 ? 1 : easeOutCubic(clamp((t - T_CONNECT(k) + 0.35) / 0.35))));
    const linkOn = (1 - prog(t, 14.7, 15.4)) + prog(t, 38.6, 39.3);
    ls.intensity = XBLOCKS.map((_, k) => { const tc = T_CONNECT(k); return linkOn * ((t < 20 ? 0.8 : 0.32) + (t > tc && t < 20 ? 1.4 * Math.exp(-4 * (t - tc)) : 0)); });
    ls.packets = t < 20 ? 2 : 3; ls.flow = 1;
    link.group.visible = linkOn > 0.003;
    link.update(t);
    XBLOCKS.forEach((b, k) => {
      const tc = T_CONNECT(k), f = t > tc && t < tc + 1.2 ? Math.exp(-5 * (t - tc)) : 0;
      [[e, C.quillHi], [q, C.emberHi]].forEach(([c, col], s) => { const sp = cflash[k][s]; sp.visible = f > 0.01; sp.position.copy(c.blockWorld(b, tmp)); sp.scale.setScalar(0.5 + 1.8 * f); sp.material.color.copy(col); sp.material.opacity = f; });
    });

    // ---------------- emitted tokens: out of the mouth, then in one arc down to the base of the new column
    for (const [tiles, c, L] of [[emitQ, q, LQ], [emitE, e, LE]]) {
      const mouth = c.mouthWorld(new THREE.Vector3());
      tiles.forEach((tile, g) => {
        if (g > NGEN) { tile.visible = false; return; }
        const te = emitT(g), hp = HOPS[g];
        const dropA = hp ? hp.t0 : F0 + 0.15, dropB = hp ? hp.land : F0 + 0.55;
        tile.visible = t > te && t < dropB + 0.02 && c.pose.opacity > 0.05;
        if (!tile.visible) return;
        const s = spring(t, te, 2.6, 0.5);
        const out = mouth.clone().add(V(0, 0.12 * clamp(s), 0.5 * clamp(s)));     // spat out of the mouth, ahead
        const u = prog(t, dropA, dropB, (x) => x * x * (3 - 2 * x));
        const base = V(L.x, -0.4, 0), ctrl = V(L.x, 1.5, 1.15);
        const pos = u > 0 ? out.multiplyScalar((1 - u) * (1 - u)).addScaledVector(ctrl, 2 * u * (1 - u)).addScaledVector(base, u * u) : out;
        tile.position.copy(pos); tile.quaternion.copy(camera.quaternion);
        tile.scale.setScalar(Math.max(0.01, clamp(s * 1.1) * lerp(1, 0.84, u)));
        tile.material.opacity = (hp ? 1 : 1 - prog(t, F0 + 0.1, F0 + 0.5)) * c.pose.opacity;
      });
    }

    // ---------------- "36 blocks" bracket
    {
      const o = win(t, L0 + 0.25, L1, 0.25);
      const x = LQ.x, z = LQ.cols[0].z - 0.55;
      const pts = [V(x, 0.03, z), V(x, H - 0.03, z)];
      bracket[0].visible = o > 0.003; bracket[0].position.set(x, H / 2, z); bracket[0].scale.set(1, H * clamp(o * 1.5), 1);
      [0, H].forEach((y, i) => { const m = bracket[i + 1]; m.visible = o > 0.003; m.position.set(x, y + (i ? -0.03 : 0.03), z + 0.1); m.rotation.set(Math.PI / 2, 0, 0); m.scale.set(1, 0.2, 1); });
      brMat.opacity = o; void pts;
      pinBlocks.update(V(x, H / 2, z - 0.1), camera, o, -14, 0);
      pinCols.update(V(LQ.x, -0.4, -1.2), camera, win(t, L0, L1, 0.25), 0, 62);
      pinX.update(V(LE.x, UX[4] * H, LE.cols[nEP - 1].z + 0.45), camera, win(t, L0 + 0.5, L1, 0.25), 16, 0);
    }
    cap.update(t, CAP[0], CAP[1]); cap2.update(t, CAP[0] + 0.25, CAP[1]);

    // ---------------- post
    post.flash = t < 12.3 ? 0.12 * Math.exp(-6 * (t - 12)) : 0;
    post.flash += 0.06 * (t >= TX.hit ? Math.exp(-8 * (t - TX.hit)) : 0);
    const lat = latOn;
    world.floor.material.uniforms.uAmp.value = lerp(1, 0.45, lat);
    post.bloomStrength = lerp(1.1, 0.8, prog(t, 12.0, 13.0)) - 0.2 * lat + 0.05 * bp - 0.05 * exAmp + 0.15 * prog(t, 38.5, 40);
    post.bloomThreshold = lerp(0.18, 0.24, prog(t, 12, 13)) + 0.1 * lat + 0.04 * exAmp + 0.38 * prog(t, 38.5, 40) * (1 - 0.26 * lat);
    post.bloomRadius = 0.5;
    return post;
  }
  return { scene, camera, update };
}
