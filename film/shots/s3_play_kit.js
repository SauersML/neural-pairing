// Helpers for S3 (s3_play): character-animation curves (pure functions of t), small textures and materials.
import * as THREE from 'three';
import { clamp, lerp, hash, spring } from '../lib/util.js';

export const GOLD = '#ffd166';
export const GOLD_C = new THREE.Color(GOLD);

// ---------- character curves (all pure in t) ----------
// A hop with anticipation (crouch), stretch on take-off, an arc, and a squash on landing.
// t0 = take-off time. Returns {y, sq}. sq multiplies pose.squash.
export function hop(t, t0, h = 0.6, dur = 0.5, crouch = 0.16) {
  let y = 0, sq = 1;
  const a = t - (t0 - crouch);
  if (a > 0 && t < t0) sq = 1 - 0.13 * Math.sin((a / crouch) * Math.PI * 0.5);
  else if (t >= t0 && t < t0 + dur) {
    const u = (t - t0) / dur; y = 4 * h * u * (1 - u);
    sq = 1 + 0.13 * Math.pow(Math.abs(1 - 2 * u), 1.5) * (u < 0.5 ? 1 : 0.7);
  } else if (t >= t0 + dur) {
    const x = t - t0 - dur; sq = 1 - 0.17 * Math.exp(-7 * x) * Math.cos(2 * Math.PI * 2.6 * x);
  }
  return { y, sq };
}
// combine several hops: list of [t0, h, dur]
export function hops(t, list) {
  let y = 0, sq = 1;
  for (const [t0, h, dur] of list) { if (t < t0 - 0.3 || t > t0 + dur + 1.2) continue; const r = hop(t, t0, h, dur); y += r.y; sq *= r.sq; }
  return { y, sq };
}
// a writing nod: small pull-back, then a forward nod that settles
export function nod(t, t0, amp = 0.28) {
  const x = t - t0;
  if (x < -0.18 || x > 1.2) return 0;
  if (x < 0) return -0.1 * amp * 3 * Math.sin(((x + 0.18) / 0.18) * Math.PI * 0.5);
  return amp * Math.exp(-5 * x) * Math.sin(2 * Math.PI * 1.7 * x + 0.35) - 0.3 * amp * Math.exp(-14 * x);
}
// deterministic blinks (about every `period` seconds, jittered)
export function blink(t, seed = 0, period = 2.9) {
  let v = 0; const n = Math.floor(t / period);
  for (let j = n - 1; j <= n + 1; j++) { const bt = j * period + 0.3 + hash(j * 7.31 + seed * 3.7) * 1.7; v = Math.max(v, 1 - Math.abs(t - bt) / 0.085); }
  return clamp(v);
}
// scale pop with overshoot
export const pop = (t, t0, f = 2.6, z = 0.38) => (t <= t0 ? 0 : spring(t, t0, f, z));
// smooth step between a and b
export const ss = (t, a, b) => { const x = clamp((t - a) / (b - a)); return x * x * (3 - 2 * x); };
// quadratic bezier
export function bez(a, c, b, u, out = new THREE.Vector3()) {
  const v = 1 - u; return out.set(a.x * v * v + 2 * c.x * u * v + b.x * u * u, a.y * v * v + 2 * c.y * u * v + b.y * u * u, a.z * v * v + 2 * c.z * u * v + b.z * u * u);
}
export const yawTo = (from, to) => Math.atan2(to.x - from.x, to.z - from.z);
export const lerpAng = (a, b, k) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * k; };

// ---------- textures / materials ----------
let _ring;
export function ringTexture() {
  if (_ring) return _ring;
  const c = document.createElement('canvas'); c.width = c.height = 256; const x = c.getContext('2d');
  const g = x.createRadialGradient(128, 128, 70, 128, 128, 128);
  g.addColorStop(0, 'rgba(255,255,255,0)'); g.addColorStop(0.55, 'rgba(255,255,255,0.9)'); g.addColorStop(0.7, 'rgba(255,255,255,0.35)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 256, 256);
  _ring = new THREE.CanvasTexture(c); _ring.colorSpace = THREE.SRGBColorSpace; return _ring;
}
// A speech bubble (rounded panel with a tail at the lower-left), as a canvas texture.
export function bubbleTexture(w = 1400, h = 620) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d');
  const r = 70, m = 14, tailH = 110, bh = h - tailH;
  x.beginPath();
  x.moveTo(m + r, m); x.arcTo(w - m, m, w - m, bh, r); x.arcTo(w - m, bh, m, bh, r);
  x.lineTo(m + 260, bh); x.lineTo(m + 70, h - m); x.lineTo(m + 170, bh);
  x.arcTo(m, bh, m, m, r); x.arcTo(m, m, w - m, m, r); x.closePath();
  const g = x.createLinearGradient(0, 0, 0, h); g.addColorStop(0, 'rgba(40,18,8,0.92)'); g.addColorStop(1, 'rgba(22,10,6,0.92)');
  x.fillStyle = g; x.fill();
  x.lineWidth = 7; x.strokeStyle = 'rgba(251,146,60,0.95)'; x.shadowColor = '#fb923c'; x.shadowBlur = 24; x.stroke();
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  tex.userData = { aspect: w / h, tailFrac: tailH / h, tipU: (m + 70) / w };
  return tex;
}
// A small code table card (lookup game): rows "K7: candle" etc. One row highlighted.
export function tableTexture(rows, hi = -1) {
  const W = 640, rh = 84, H = rows.length * rh + 40;
  const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  const rr = (a, b, w, h, r) => { x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); };
  rr(4, 4, W - 8, H - 8, 22); x.fillStyle = 'rgba(60,40,0,0.9)'; x.fill(); x.lineWidth = 5; x.strokeStyle = GOLD; x.stroke();
  x.font = '600 58px Inter'; x.textBaseline = 'middle';
  rows.forEach((r, i) => {
    const y = 20 + rh * (i + 0.5);
    if (i === hi) { rr(16, y - rh / 2 + 6, W - 32, rh - 12, 14); x.fillStyle = 'rgba(255,209,102,0.25)'; x.fill(); }
    x.fillStyle = i === hi ? '#fff7d6' : 'rgba(255,240,200,0.75)';
    x.textAlign = 'left'; x.fillText(r[0], 44, y + 2);
    x.textAlign = 'right'; x.fillText(r[1], W - 44, y + 2);
  });
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; tex.userData = { aspect: W / H };
  return tex;
}
export function plane(tex, h, opts = {}) {
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false, fog: false, ...opts });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(h * tex.userData.aspect, h), mat); m.userData.width = h * tex.userData.aspect; return m;
}
export { lerp, clamp };
