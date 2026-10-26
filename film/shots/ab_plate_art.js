// The diagrams of the anatomical plate (ab_plate): engraved labels with leaders that draw themselves, the lifted
// and parted vertebra (one block: self-attention, then feed-forward, each added into the stream), and the roundel that
// magnifies one link nerve at that depth (one read, exactly as pairing/link.py computes it):
//   reader's query  ·  partner's keys and values (every visible partner position)  ·  the partner's gate on each value
//   ·  attention weights  ·  weighted sum  ·  write size (held at 10 %)  ·  added into the reader's residual stream
// All of it is a pure function of t.
import * as THREE from 'three';
import { XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { toScreen } from '../lib/hud.js';
import { clamp, lerp, prog, smoother, easeOutCubic, easeInOutCubic, easeOutBack } from '../lib/util.js';
import { boneMat, ndMat, inkTube, inkSymbol } from '../lib/plate_ink.js';
import { INK, TEAL, RED } from './ab_plate_hud.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const XB = 15, XE = XBLOCKS.indexOf(XB);       // the opened depth: after block 15 (exchange 3)
const S = 1.4;                                 // enlargement of the lifted vertebra
const LIFT_AT = { x: 360, y: 585, t: 45.0 };      // where it is shown (screen px, under the camera of view 2)
// reveal helpers: r(t, a, d) = 0..1 drawn between a and a + d
const r = (t, a, d = 0.25) => clamp((t - a) / d);
const ease = (x) => easeOutCubic(clamp(x));

// ---- the roundel's drawing (screen px, relative to its centre; radius RR)
const RR = 190;
const QPOS = [-112, -18], QR = 30;                                     // the reader's query
const FAS = [[70, -146], [112, -84], [128, -18], [118, 50], [86, 112]];  // the partner's positions (keys | values)
const FR = 25;
const W8 = [0.36, 0.11, 0.29, 0.07, 0.17];                              // attention weights (illustrative)
const GATE = [0.9, 0.86, 0.38, 0.88, 0.64];                             // the partner's gate on each value
const SUM = [34, 162];                                                 // weighted sum

function puck(rad, h) {
  const pts = [], b = h * 0.48, n = 6;
  pts.push(new THREE.Vector2(0, -h / 2));
  for (let k = 0; k <= n; k++) { const a = -Math.PI / 2 + (k / n) * (Math.PI / 2); pts.push(new THREE.Vector2(rad - b + Math.cos(a) * b, -h / 2 + b + Math.sin(a) * b)); }
  for (let k = 0; k <= n; k++) { const a = (k / n) * (Math.PI / 2); pts.push(new THREE.Vector2(rad - b + Math.cos(a) * b, h / 2 - b + Math.sin(a) * b)); }
  pts.push(new THREE.Vector2(0, h / 2));
  return new THREE.LatheGeometry(pts, 48);
}

export class PlateArt {
  constructor({ scene, pr, q, e, link, ui, camera, camAt }) {
    Object.assign(this, { scene, pr, q, e, link, ui, camera, camAt });
    this.ref = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
    // ---- the lifted vertebra: two halves (self-attention below, feed-forward above), sectioned to show the cord
    const hb = q.H / NBLOCKS, rad = q.blockRadius(XB);
    this.cut = new THREE.Vector4(0, 0, 0, 0);
    const bone = boneMat({ wash: 0.25, cut: this.cut, side: THREE.DoubleSide }), nd = ndMat({ cut: this.cut, side: THREE.DoubleSide });
    this.halves = [0, 1].map(() => { const m = new THREE.Mesh(puck(rad, hb * 0.62 * 0.5), bone); scene.add(m); pr.add(m, bone, nd, true); return m; });
    this.cordMat = inkTube('red', { base: 0.45, lineTone: 0.5, depthTest: false, tol: 0.9 });
    this.cord = new THREE.Mesh(new THREE.BufferGeometry(), this.cordMat); scene.add(this.cord); pr.add(this.cord, this.cordMat, null, true);
    this.plusMats = [0, 1, 2].map(() => inkSymbol('sepia', 0.95));
    this.plus = this.plusMats.map((m) => { const p = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), m); scene.add(p); pr.add(p, m, null, true); return p; });
    this.hb = hb; this.rad = rad;
  }
  pre() {}

  // where the lifted vertebra sits (world): unprojected from a fixed spot of view 2's frame, a little in front of the
  // spine; it tilts its top toward the viewer so both halves and the cut show
  liftTarget(B) {
    if (this._lt) return this._lt;
    const ref = this.ref; this.camAt(LIFT_AT.t, ref);
    const fwd = V(0, 0, -1).applyQuaternion(ref.quaternion), right = V(1, 0, 0).applyQuaternion(ref.quaternion);
    const v = V(LIFT_AT.x / 960 - 1, 1 - LIFT_AT.y / 540, 0.5).unproject(ref), dir = v.sub(ref.position).normalize();
    const D = B.clone().sub(ref.position).dot(fwd) - 0.9;
    this._lt = { C: ref.position.clone().addScaledVector(dir, D / dir.dot(fwd)), right: V(right.x, 0, right.z).normalize() };
    return this._lt;
  }
  liftPose(t, B) {
    const out = ease(prog(t, 43.0, 43.5, (x) => x)) * (1 - easeInOutCubic(prog(t, 46.0, 46.4, (x) => x)));
    const part = smoother(prog(t, 43.5, 43.75)) * (1 - smoother(prog(t, 45.95, 46.15)));
    const T = this.liftTarget(B);
    const C = B.clone().lerp(T.C, out);
    const tilt = new THREE.Quaternion().setFromAxisAngle(T.right, 0.55 * out);
    return { out, part, C, s: lerp(1, S, out), tilt, U: V(0, 1, 0).applyQuaternion(tilt) };
  }

  update(t, inPlate) {
    const { q, ui, camera } = this;
    ui.begin();
    const res = {};
    const B = q.blockWorld(XB);
    // ---------------- view 2: the lifted vertebra
    const L = this.liftPose(t, B);
    const showLift = inPlate && L.out > 0.001;
    q.blocks[XB].userData.plateHide = showLift && L.out > 0.02;
    const gap = 0.42 * L.part * L.s, hh = this.hb * 0.62 * 0.5 * L.s;
    const yaw = q.pose.yaw;
    this.halves.forEach((m, i) => {
      m.userData.plateVisible = showLift;
      m.position.copy(L.C).addScaledVector(L.U, (i === 0 ? -1 : 1) * (hh / 2 + gap / 2));
      m.scale.setScalar(L.s); m.quaternion.copy(L.tilt).multiply(new THREE.Quaternion().setFromAxisAngle(V(0, 1, 0), yaw));
    });
    // the section: a plane through the axis, facing the camera, opened as the halves part
    const toCam = camera.position.clone().sub(L.C); toCam.addScaledVector(L.U, -toCam.dot(L.U)); toCam.normalize();
    const cutOff = 50.0;   // (no section: the halves stay whole; the cord is drawn through them)
    this.cut.set(toCam.x, toCam.y, toCam.z, -toCam.dot(L.C) - cutOff);
    // the cord through both halves (the residual stream), a little beyond them
    const cl = (hh + gap) * 1.0 + 0.32 * L.part * L.s;
    if (showLift) {
      const a = L.C.clone().addScaledVector(L.U, -cl), b = L.C.clone().addScaledVector(L.U, cl);
      this.cord.geometry.dispose(); this.cord.geometry = new THREE.TubeGeometry(new THREE.LineCurve3(a, b), 8, 0.085 * L.s * 0.75, 10, false);
    }
    this.cord.userData.plateVisible = showLift;
    this.cordMat.uniforms.uAmt.value = L.part;
    // the "+": each half adds into the stream above itself; the link's read is added after the block, on Quill's own cord
    const tops = [L.C.clone().addScaledVector(L.U, 0), L.C.clone().addScaledVector(L.U, hh + gap / 2 + 0.2 * L.s)];
    const plusT = [43.625, 43.75, 45.25];
    this.plus.forEach((p, i) => {
      const at = i < 2 ? tops[i] : q.blockWorld(XB).add(V(0, this.hb * 0.75, 0));
      const k = inPlate ? r(t, plusT[i], 0.2) * (1 - r(t, 46.0, 0.15)) : 0;
      p.userData.plateVisible = k > 0.001 && (i === 2 || showLift);
      p.position.copy(at); p.quaternion.copy(camera.quaternion);
      p.scale.setScalar((i < 2 ? 0.2 : 0.13) * (0.6 + 0.4 * easeOutBack(k, 2)));
      this.plusMats[i].uniforms.uDraw.value = k;
    });

    if (inPlate) {
      this.view1(t);
      this.view2(t, L, tops, res);
    }
    ui.end();
    return res;
  }

  P(v) { return toScreen(v, this.camera); }
  pxRadius(world, rad) { const d = world.distanceTo(this.camera.position); return rad / (d * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 540; }

  // ---------------- view 1: names, the 36 blocks, the residual stream, the link
  view1(t) {
    const { q, e, ui } = this;
    const out = 1 - r(t, 42.75, 0.25);
    if (out <= 0) return;
    // names under the feet
    const nq = this.P(V(-2.3, 0, 0)), ne = this.P(V(2.3, 0, 0));
    ui.text('nq', 'Quill', nq.x, nq.y + 62, { size: 50, reveal: r(t, 41.125, 0.3), opacity: out, anchor: 'middle' });
    ui.text('ne', 'Ember', ne.x, ne.y + 62, { size: 50, reveal: r(t, 41.25, 0.3), opacity: out, anchor: 'middle' });
    // a ruler of 36 ticks along Quill's spine; the 8 exchange depths are the longer blue-green ticks
    const pts = []; let xmin = 1e9;
    for (let i = 0; i < NBLOCKS; i++) { const c = q.blockWorld(i); const s = this.P(c); const rp = this.pxRadius(c, q.blockRadius(i) * 1.15); pts.push(s); xmin = Math.min(xmin, s.x - rp); }
    const rx = xmin - 26, y0 = pts[0].y + (pts[0].y - pts[1].y) / 2, y1 = pts[35].y - (pts[0].y - pts[1].y) / 2;
    const rv = ease(r(t, 41.375, 0.35));
    ui.line('rul', rx, y0, rx, lerp(y0, y1, rv), { width: 1.5, opacity: out });
    for (let i = 0; i < NBLOCKS; i++) {
      const k = clamp((rv * 36 - i) * 1.5); if (k <= 0) continue;
      const isX = XBLOCKS.includes(i), len = (isX ? 20 : 10) * k;
      ui.line('tk' + i, rx - len, pts[i].y, rx, pts[i].y, { width: isX ? 2.2 : 1.2, color: isX ? TEAL : INK, opacity: out });
    }
    const my = (y0 + y1) / 2;
    ui.text('b36', '36 layers', rx - 36, my + 14, { size: 46, anchor: 'end', reveal: r(t, 41.625, 0.3), opacity: out });
    // the residual stream: the red cord through every block, token in at the tail, token out at the head
    const cs = this.P(q.blockWorld(31)), lab = { x: (cs.x + this.P(e.blockWorld(31)).x) / 2, y: cs.y - 46 };
    ui.line('rsL', cs.x + 6, cs.y, lab.x - 128, lab.y + 12, { width: 1.3, reveal: ease(r(t, 41.75, 0.25)), opacity: out });
    ui.dot('rsD', cs.x + 4, cs.y, 3.4, { opacity: out * clamp(r(t, 41.75, 0.1)) });
    ui.text('rs', 'residual stream', lab.x - 120, lab.y + 22, { size: 46, reveal: r(t, 41.81, 0.35), opacity: out, color: RED });
    const tail = this.P(q.tailWorld()), head = this.P(q.headWorld());
    const hr = this.pxRadius(q.headWorld(), 0.66);
    ui.line('tiL', tail.x - 12, tail.y, rx - 60, tail.y - 6, { width: 1.3, reveal: ease(r(t, 41.875, 0.2)), opacity: out });
    ui.text('ti', 'token in', rx - 70, tail.y + 6, { size: 44, anchor: 'end', reveal: r(t, 41.94, 0.3), opacity: out });
    ui.line('toL', head.x - hr * 0.95, head.y + hr * 0.1, rx - 60, head.y + hr * 0.1, { width: 1.3, reveal: ease(r(t, 42.0, 0.2)), opacity: out });
    ui.text('to', 'token out', rx - 70, head.y + hr * 0.1 + 14, { size: 44, anchor: 'end', reveal: r(t, 42.06, 0.3), opacity: out });
    // the link: a bracket on Ember's far side, joined to the 8 nerve roots
    const ep = XBLOCKS.map((b) => this.P(e.blockWorld(b)));
    let xmax = -1e9; ep.forEach((s, k) => { xmax = Math.max(xmax, s.x + this.pxRadius(e.blockWorld(XBLOCKS[k]), e.blockRadius(XBLOCKS[k]) * 1.2)); });
    const bx = xmax + 30, lv = ease(r(t, 42.125, 0.3));
    ui.line('lk', bx, ep[0].y, bx, lerp(ep[0].y, ep[7].y, lv), { width: 2.0, color: TEAL, opacity: out });
    ep.forEach((s, k) => { const kk = clamp((lv * 8 - k) * 1.5); if (kk > 0) ui.line('lt' + k, bx - 16 * kk, s.y, bx, s.y, { width: 2.0, color: TEAL, opacity: out }); });
    const lm = (ep[0].y + ep[7].y) / 2;
    ui.line('lkm', bx, lm, bx + 26 * lv, lm, { width: 2.0, color: TEAL, opacity: out });
    ui.text('lk1', 'the link', bx + 40, lm - 30, { size: 52, reveal: r(t, 42.19, 0.3), opacity: out, color: TEAL });
    ui.text('lk2', '8 depths, two fibres,', bx + 40, lm + 22, { size: 42, reveal: r(t, 42.25, 0.3), opacity: out });
    ui.text('lk3', 'one each way at once', bx + 40, lm + 70, { size: 42, reveal: r(t, 42.31, 0.3), opacity: out });
  }

  // ---------------- view 2: the parted vertebra and the roundel (one read)
  view2(t, L, tops, res) {
    const { q, ui } = this;
    const out = 1 - r(t, 46.0, 0.25);
    if (t < 43.0 || out <= 0) return;
    // the dotted guide from the vertebra's place to the lifted copy
    const a = this.P(q.blockWorld(XB)), b = this.P(L.C);
    ui.line('gd', a.x, a.y, lerp(a.x, b.x, L.out), lerp(a.y, b.y, L.out), { width: 1.4, dash: '2 7', reveal: L.out, opacity: out });
    // the halves
    const hr = this.pxRadius(L.C, this.rad * L.s);
    const sa = this.P(L.C.clone().addScaledVector(L.U, -this.hb * 0.62 * 0.25 * L.s - 0.08 * L.part * L.s));
    const sf = this.P(L.C.clone().addScaledVector(L.U, this.hb * 0.62 * 0.25 * L.s + 0.08 * L.part * L.s));
    const pl = this.P(tops[1]), pl0 = this.P(tops[0]);
    const LX = 96;
    ui.line('ffL', LX + 150, sf.y - hr * 0.62 - 66, sf.x - hr * 0.45, sf.y - hr * 0.3, { width: 1.3, reveal: ease(r(t, 43.75, 0.2)), opacity: out });
    ui.text('ff', 'feed-forward', LX, sf.y - hr * 0.62 - 80, { size: 46, reveal: r(t, 43.81, 0.3), opacity: out });
    ui.line('saL', LX + 150, sa.y + hr * 0.62 + 82, sa.x - hr * 0.5, sa.y + hr * 0.3, { width: 1.3, reveal: ease(r(t, 43.625, 0.2)), opacity: out });
    ui.text('sa', 'self-attention', LX, sa.y + hr * 0.62 + 124, { size: 46, reveal: r(t, 43.69, 0.3), opacity: out });
    const ry = sf.y - hr * 0.62 - 210;
    ui.line('rsL2', LX + 330, ry + 30, pl.x + 10, pl.y - 12, { width: 1.3, reveal: ease(r(t, 43.875, 0.2)), opacity: out });
    ui.line('rsL3', LX + 330, ry + 30, pl0.x + 14, pl0.y - 8, { width: 1.3, reveal: ease(r(t, 43.875, 0.25)), opacity: out });
    ui.text('rs2', 'each adds', LX, ry, { size: 44, reveal: r(t, 43.875, 0.3), opacity: out, color: RED });
    ui.text('rs3', 'into the stream', LX, ry + 46, { size: 44, reveal: r(t, 43.94, 0.3), opacity: out, color: RED });

    // ---- the roundel: the link nerve at this depth, magnified (one read: Quill reads Ember)
    const st = this.link.strands.find((s) => s.e === XE && s.dir === 1);          // Ember -> Quill
    const spot = this.P(st.curve.getPointAt(0.5));
    const C = { x: 1345, y: 320 };
    const o0 = 44.0;
    const ring = ease(r(t, o0, 0.3));
    if (ring <= 0) return;
    res.clear = [C.x, C.y, RR + 10, clamp(ring * 1.5) * out];
    res.clearRect = [C.x + RR + 20, 0, 1920, C.y + RR + 50, clamp(ring * 1.5) * out];
    const sr = 30;
    ui.path('spot', circ(spot.x, spot.y, sr), { width: 1.5, reveal: ring, opacity: out });
    const [t1, t2] = tangents(spot, sr, C, RR);
    ui.line('tg1', t1[0].x, t1[0].y, lerp(t1[0].x, t1[1].x, ring), lerp(t1[0].y, t1[1].y, ring), { width: 1.1, opacity: out * 0.8 });
    ui.line('tg2', t2[0].x, t2[0].y, lerp(t2[0].x, t2[1].x, ring), lerp(t2[0].y, t2[1].y, ring), { width: 1.1, opacity: out * 0.8 });
    // the sheath: a double circle with radial hatching between (the nerve's outer coat)
    ui.path('rr0', circ(C.x, C.y, RR), { width: 2.4, reveal: ring, opacity: out });
    ui.path('rr1', circ(C.x, C.y, RR - 13), { width: 1.2, reveal: ring, opacity: out });
    { let d = ''; const n = 120; const m = Math.floor(n * ease(r(t, o0 + 0.1, 0.3)));
      for (let k = 0; k < m; k++) { const a2 = k / n * Math.PI * 2 - Math.PI / 2; const ca = Math.cos(a2), sa2 = Math.sin(a2);
        d += `M${(C.x + ca * (RR - 12)).toFixed(1)},${(C.y + sa2 * (RR - 12)).toFixed(1)} L${(C.x + ca * (RR - 1.5)).toFixed(1)},${(C.y + sa2 * (RR - 1.5)).toFixed(1)} `; }
      if (d) ui.path('rrh', d, { width: 0.9, opacity: out * 0.75 }); }
    const X = (p) => C.x + p[0], Y = (p) => C.y + p[1];
    // the reader's query
    const kq = ease(r(t, 44.25, 0.25));
    ui.path('q0', circ(X(QPOS), Y(QPOS), QR), { width: 2.0, color: INK, reveal: kq, opacity: out });
    ui.path('qh', hatchCircle(X(QPOS), Y(QPOS), QR - 3, 5.5), { width: 1.0, color: TEAL, reveal: kq, opacity: out });
    // the partner's positions: key (left half, hatched) | value (right half, stippled); a gate arc on each value
    const kf = ease(r(t, 44.5, 0.3));
    FAS.forEach((f, i) => {
      const kk = clamp(kf * 1.6 - i * 0.15);
      if (kk <= 0) return;
      const x = X(f), y = Y(f);
      ui.path('f' + i, circ(x, y, FR), { width: 1.8, reveal: kk, opacity: out });
      ui.line('fd' + i, x, y - FR, x, y + FR, { width: 1.2, reveal: kk, opacity: out });
      ui.path('fk' + i, hatchHalf(x, y, FR - 2, -1, 4.2), { width: 0.9, reveal: kk, opacity: out });
      ui.path('fv' + i, stipHalf(x, y, FR - 3, 1, i), { width: 1.6, reveal: kk, opacity: out, cap: 'round' });
      const kg = ease(r(t, 44.75 + i * 0.03, 0.25));
      if (kg > 0) ui.path('g' + i, arc(x, y, FR + 6, -Math.PI / 2, -Math.PI / 2 + Math.PI * GATE[i]), { width: 3.0, color: TEAL, reveal: kg, opacity: out });
    });
    // attention: the query against every key (line weight = attention weight)
    const kw = ease(r(t, 44.625, 0.3));
    FAS.forEach((f, i) => {
      const x0 = X(QPOS) + QR, y0 = Y(QPOS), x1 = X(f) - FR * 0.55, y1 = Y(f);
      ui.path('w' + i, `M${x0.toFixed(1)},${y0.toFixed(1)} Q${((x0 + x1) / 2).toFixed(1)},${((y0 + y1) / 2 - 18).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`, { width: 0.7 + 7 * W8[i], color: INK, reveal: clamp(kw * 1.4 - i * 0.08), opacity: out });
    });
    // weighted sum: values, each scaled by its gate and its weight, flow to one node
    const ks = ease(r(t, 45, 0.3));
    FAS.forEach((f, i) => {
      const x0 = X(f) + FR * 0.4, y0 = Y(f) + FR * 0.6, x1 = X(SUM), y1 = Y(SUM);
      ui.path('s' + i, `M${(x0 - FR * 0.9).toFixed(1)},${(y0 - FR * 0.2).toFixed(1)} Q${(x1 - 4).toFixed(1)},${(y0 + 10).toFixed(1)} ${x1.toFixed(1)},${(y1 - 12).toFixed(1)}`, { width: 0.6 + 8 * W8[i] * GATE[i], color: TEAL, reveal: clamp(ks * 1.3 - i * 0.06), opacity: out });
    });
    if (ks > 0) { ui.dot('sum', X(SUM), Y(SUM), 8 * ks, { color: TEAL, opacity: out }); ui.path('sumR', circ(X(SUM), Y(SUM), 12), { width: 1.6, reveal: ks, opacity: out }); }
    // out of the roundel (upper left, clear of the magnifier's tangent lines), along a fibre into Quill's stream
    // after block 15 (the "+"), through a narrow collar: the write size
    const kx = ease(r(t, 45.25, 0.4));
    const pq = this.P(q.blockWorld(XB).add(V(0, this.hb * 0.75, 0)));
    const ex = { x: C.x + Math.cos(2.92) * RR, y: C.y + Math.sin(2.92) * RR };
    const inner = `M${X(SUM).toFixed(1)},${Y(SUM).toFixed(1)} Q${(C.x - 120).toFixed(1)},${(C.y + 165).toFixed(1)} ${ex.x.toFixed(1)},${ex.y.toFixed(1)}`;
    const c1 = { x: lerp(ex.x, pq.x, 0.5), y: lerp(ex.y, pq.y, 0.15) };
    const outer = ` Q${c1.x.toFixed(1)},${c1.y.toFixed(1)} ${(pq.x + 15).toFixed(1)},${(pq.y - 6).toFixed(1)}`;
    ui.path('xo', inner + outer, { width: 3.0, color: TEAL, reveal: kx, opacity: out });
    if (kx > 0.85) { const k3 = clamp((kx - 0.85) * 7); const u = 0.8, w0 = 1 - u;
      const cx = w0 * w0 * ex.x + 2 * w0 * u * c1.x + u * u * (pq.x + 15), cy = w0 * w0 * ex.y + 2 * w0 * u * c1.y + u * u * (pq.y - 6);
      ui.path('col', `M${(cx - 6).toFixed(1)},${(cy - 15).toFixed(1)} Q${(cx + 2).toFixed(1)},${cy.toFixed(1)} ${(cx - 6).toFixed(1)},${(cy + 15).toFixed(1)} M${(cx + 10).toFixed(1)},${(cy - 15).toFixed(1)} Q${(cx + 2).toFixed(1)},${cy.toFixed(1)} ${(cx + 10).toFixed(1)},${(cy + 15).toFixed(1)}`, { width: 1.8, reveal: k3, opacity: out }); }

    // ---- labels of the roundel
    const qx = C.x - RR - 22, qy = C.y - RR + 24;
    ui.line('lq', X(QPOS) - QR * 0.55, Y(QPOS) - QR * 0.85, qx + 8, qy + 14, { width: 1.2, reveal: ease(r(t, 44.25, 0.2)), opacity: out });
    ui.text('tq1', 'the reader’s', qx, qy - 30, { size: 42, anchor: 'end', reveal: r(t, 44.31, 0.3), opacity: out });
    ui.text('tq2', 'query', qx, qy + 16, { size: 42, anchor: 'end', reveal: r(t, 44.37, 0.3), opacity: out });
    res.clearRect2 = [qx - 250, qy - 70, qx + 6, qy + 30, clamp(r(t, 44.2, 0.2)) * out];
    const LC = C.x + RR + 44;
    ui.line('lf', X(FAS[0]) + FR + 2, Y(FAS[0]) - 6, LC - 10, C.y - RR + 24, { width: 1.2, reveal: ease(r(t, 44.5, 0.2)), opacity: out });
    ui.text('tf1', 'the partner’s', LC, C.y - RR + 10, { size: 42, reveal: r(t, 44.56, 0.3), opacity: out });
    ui.text('tf2', 'keys and values', LC, C.y - RR + 56, { size: 42, reveal: r(t, 44.62, 0.3), opacity: out });
    ui.line('lg', X(FAS[2]) + FR + 9, Y(FAS[2]), LC - 10, Y(FAS[2]) + 2, { width: 1.2, reveal: ease(r(t, 44.75, 0.2)), opacity: out });
    ui.text('tg', 'gates', LC, Y(FAS[2]) + 14, { size: 42, reveal: r(t, 44.81, 0.3), opacity: out, color: TEAL });
    ui.line('ls', X(SUM) + 14, Y(SUM) - 4, LC - 10, Y(SUM) - 4, { width: 1.2, reveal: ease(r(t, 45.0, 0.2)), opacity: out });
    ui.text('ts', 'weighted sum', LC, Y(SUM) + 10, { size: 42, reveal: r(t, 45.06, 0.3), opacity: out });
    // the write: one label in the clear paper under the lifted vertebra, one leader to the "+"
    const wy = 952, wx = 96;
    ui.line('lw', wx + 300, wy - 40, pq.x - 12, pq.y + 10, { width: 1.2, reveal: ease(r(t, 45.45, 0.25)), opacity: out });
    ui.text('tw1', 'write size 10%,', wx, wy - 24, { size: 42, reveal: r(t, 45.5, 0.3), opacity: out, color: TEAL });
    ui.text('tw2', 'added to the stream', wx, wy + 22, { size: 42, reveal: r(t, 45.56, 0.3), opacity: out });
  }
}

// ---- small SVG geometry helpers
function circ(x, y, rr) { return `M${(x + rr).toFixed(1)},${y.toFixed(1)} A${rr},${rr} 0 1 1 ${(x - rr).toFixed(1)},${y.toFixed(1)} A${rr},${rr} 0 1 1 ${(x + rr).toFixed(1)},${y.toFixed(1)}`; }
function arc(x, y, rr, a0, a1) { const p0 = [x + Math.cos(a0) * rr, y + Math.sin(a0) * rr], p1 = [x + Math.cos(a1) * rr, y + Math.sin(a1) * rr]; const large = a1 - a0 > Math.PI ? 1 : 0; return `M${p0[0].toFixed(1)},${p0[1].toFixed(1)} A${rr},${rr} 0 ${large} 1 ${p1[0].toFixed(1)},${p1[1].toFixed(1)}`; }
function hatchCircle(x, y, rr, sp) { let d = ''; for (let o = -rr + sp / 2; o < rr; o += sp) { const h = Math.sqrt(Math.max(0, rr * rr - o * o)); d += `M${(x - h).toFixed(1)},${(y + o).toFixed(1)} L${(x + h).toFixed(1)},${(y + o).toFixed(1)} `; } return d; }
function hatchHalf(x, y, rr, side, sp) {
  // diagonal lines clipped to the half disc on `side` (-1 left, 1 right)
  let d = '';
  for (let o = -rr * 1.5; o < rr * 1.5; o += sp) {
    const pts = []; for (let k = 0; k <= 30; k++) { const u = -rr * 1.5 + k * rr * 0.1; const px = u, py = u + o; if (px * px + py * py < rr * rr && px * side > 0.8) pts.push([px, py]); }
    if (pts.length > 1) d += `M${(x + pts[0][0]).toFixed(1)},${(y + pts[0][1]).toFixed(1)} L${(x + pts[pts.length - 1][0]).toFixed(1)},${(y + pts[pts.length - 1][1]).toFixed(1)} `;
  }
  return d;
}
function stipHalf(x, y, rr, side, seed) {
  let d = ''; let s = 17 + seed * 31; const R = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  for (let k = 0; k < 46; k++) { const a = (R() - 0.5) * Math.PI, rad = Math.sqrt(R()) * rr; const px = Math.cos(a) * rad * side, py = Math.sin(a) * rad; if (px * side < 2) continue; d += `M${(x + px).toFixed(1)},${(y + py).toFixed(1)} l0.1,0 `; }
  return d;
}
// the two outer tangent lines between a small circle (p, rp) and a big circle (C, R)
function tangents(p, rp, C, R) {
  const dx = C.x - p.x, dy = C.y - p.y, d = Math.hypot(dx, dy), a = Math.atan2(dy, dx), b = Math.acos(clamp((R - rp) / d, -1, 1));
  const res = [];
  for (const s of [1, -1]) {
    const ang = a + s * (Math.PI - b);
    res.push([{ x: p.x + Math.cos(ang) * rp, y: p.y + Math.sin(ang) * rp }, { x: C.x + Math.cos(ang) * R, y: C.y + Math.sin(ang) * R }]);
  }
  return res;
}
