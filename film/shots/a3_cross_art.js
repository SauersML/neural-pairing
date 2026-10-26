// a3_cross helpers: lantern pictogram, secret card, thought bubble, fireworks (all pure in t).
import * as THREE from 'three';
import { Creature } from '../lib/creature.js';
import { clamp, hash } from '../lib/util.js';

const GOLD = '#ffd166';
function tex(c, aspect) { const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; t.userData = { aspect }; return t; }
function rr(x, a, b, w, h, r) { x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }

// draws a lantern pictogram centered in a box (cx, top y0, height H)
export function drawLantern(x, cx, y0, H) {
  const s = H / 400;
  x.save(); x.translate(cx, y0); x.scale(s, s);
  x.lineCap = 'round'; x.lineJoin = 'round';
  // handle ring
  x.strokeStyle = '#ffd166'; x.lineWidth = 14; x.beginPath(); x.arc(0, 34, 26, Math.PI * 1.05, Math.PI * 1.95); x.stroke();
  // cap
  x.fillStyle = '#c98a1a'; x.beginPath(); x.moveTo(-70, 92); x.lineTo(70, 92); x.lineTo(42, 52); x.lineTo(-42, 52); x.closePath(); x.fill();
  x.fillStyle = '#ffd166'; rr(x, -78, 88, 156, 18, 8); x.fill();
  // glass body with glow
  const g = x.createRadialGradient(0, 220, 10, 0, 220, 120);
  g.addColorStop(0, 'rgba(255,246,200,1)'); g.addColorStop(0.45, 'rgba(255,200,90,0.95)'); g.addColorStop(1, 'rgba(160,90,20,0.9)');
  x.fillStyle = g; rr(x, -62, 108, 124, 216, 30); x.fill();
  x.strokeStyle = '#ffd166'; x.lineWidth = 10; rr(x, -62, 108, 124, 216, 30); x.stroke();
  x.lineWidth = 6; x.strokeStyle = 'rgba(200,130,30,0.9)';
  for (const bx of [-22, 22]) { x.beginPath(); x.moveTo(bx, 114); x.lineTo(bx, 318); x.stroke(); }
  // flame
  const f = x.createLinearGradient(0, 160, 0, 280); f.addColorStop(0, '#fff7d6'); f.addColorStop(1, '#ff8a1f');
  x.fillStyle = f; x.beginPath(); x.moveTo(0, 158); x.bezierCurveTo(36, 210, 34, 262, 0, 282); x.bezierCurveTo(-34, 262, -36, 210, 0, 158); x.fill();
  // base
  x.fillStyle = '#ffd166'; rr(x, -82, 322, 164, 22, 9); x.fill();
  x.fillStyle = '#c98a1a'; x.beginPath(); x.moveTo(-58, 344); x.lineTo(58, 344); x.lineTo(44, 372); x.lineTo(-44, 372); x.closePath(); x.fill();
  x.restore();
}

export function lanternIconTexture() {
  const c = document.createElement('canvas'); c.width = 320; c.height = 420; const x = c.getContext('2d');
  x.shadowColor = 'rgba(255,190,80,0.9)'; x.shadowBlur = 30; drawLantern(x, 160, 14, 392);
  return tex(c, c.width / c.height);
}

// the secret card: lantern pictogram + the word
export function secretCardTexture(word = 'lantern') {
  const W = 560, H = 720; const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  rr(x, 10, 10, W - 20, H - 20, 46); const g = x.createLinearGradient(0, 0, 0, H); g.addColorStop(0, 'rgba(70,46,6,0.96)'); g.addColorStop(1, 'rgba(40,24,4,0.96)'); x.fillStyle = g; x.fill();
  x.lineWidth = 10; x.strokeStyle = GOLD; x.shadowColor = GOLD; x.shadowBlur = 26; rr(x, 10, 10, W - 20, H - 20, 46); x.stroke(); x.shadowBlur = 0;
  drawLantern(x, W / 2, 60, 430);
  x.font = '700 104px Inter'; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillStyle = '#fff3c4'; x.fillText(word, W / 2, 600);
  return tex(c, W / H);
}

// a thought bubble with a question mark
export function thoughtTexture() {
  const W = 360, H = 320; const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  x.fillStyle = 'rgba(30,14,8,0.88)'; x.strokeStyle = '#fdba74'; x.lineWidth = 8;
  const blob = () => { x.beginPath(); x.ellipse(200, 120, 140, 104, 0, 0, Math.PI * 2); };
  blob(); x.fill(); blob(); x.stroke();
  for (const [cx, cy, r] of [[70, 250, 26], [34, 296, 14]]) { x.beginPath(); x.arc(cx, cy, r, 0, Math.PI * 2); x.fill(); x.stroke(); }
  x.font = '800 150px Inter'; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillStyle = '#ffedd5'; x.fillText('?', 200, 128);
  return tex(c, W / H);
}

export function texPlane(t, h) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(h * t.userData.aspect, h), new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false, fog: false }));
  m.renderOrder = 25; m.visible = false; return m;
}

// fireworks: list of bursts {t, pos, color}; rocket rises for `rise` s, then a sphere of sparks with gravity and fade
export class Fireworks {
  constructor(scene, bursts, { per = 150, rise = 0.32 } = {}) {
    this.bursts = bursts; this.per = per; this.rise = rise;
    const N = bursts.length * (per + 6); this.N = N;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    this.pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.26, map: Creature.dotTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
    this.pts.frustumCulled = false; // hidden points sit far away; never let a stale bounding sphere cull the burst
    scene.add(this.pts);
    this.flash = bursts.map(() => { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: Creature.dotTexture(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })); scene.add(s); return s; });
  }
  update(t, amp = 1) {
    const pa = this.pts.geometry.attributes.position.array, ca = this.pts.geometry.attributes.color.array; let n = 0;
    const c = new THREE.Color();
    this.bursts.forEach((b, bi) => {
      const t0 = b.t - this.rise, x = t - b.t;
      // rocket trail
      for (let k = 0; k < 6; k++, n++) {
        const u = clamp((t - t0) / this.rise - k * 0.06);
        const on = t > t0 && t < b.t + 0.05 && u > 0;
        pa[n * 3] = b.pos.x + Math.sin(u * 6 + bi) * 0.05; pa[n * 3 + 1] = b.pos.y - (1 - u) * 3.2; pa[n * 3 + 2] = on ? b.pos.z : 1e5;
        c.copy(b.color).lerp(new THREE.Color(1, 1, 1), 0.5).multiplyScalar((1 - k / 6) * 2.5 * amp); ca[n * 3] = c.r; ca[n * 3 + 1] = c.g; ca[n * 3 + 2] = c.b;
      }
      for (let k = 0; k < this.per; k++, n++) {
        if (x < 0 || x > 1.6) { pa[n * 3 + 2] = 1e5; continue; }
        const th = hash(bi * 91 + k * 1.37) * Math.PI * 2, ph = Math.acos(2 * hash(bi * 53 + k * 2.71) - 1);
        const sp = (b.size || 1.6) * (0.75 + 0.25 * hash(bi * 7 + k * 3.3));
        const r = sp * (1 - Math.exp(-x * 4.5));
        pa[n * 3] = b.pos.x + Math.sin(ph) * Math.cos(th) * r; pa[n * 3 + 1] = b.pos.y + Math.cos(ph) * r - 0.9 * x * x; pa[n * 3 + 2] = b.pos.z + Math.sin(ph) * Math.sin(th) * r * 0.6;
        const tw = 0.75 + 0.25 * Math.sin(x * 30 + k);
        const f = Math.exp(-x * 1.7) * tw * amp * 3.2 * clamp(x / 0.09); // fade in so 150 sparks at one point don't saturate
        c.copy(k % 4 === 0 ? new THREE.Color('#fff4d6') : b.color).multiplyScalar(f); ca[n * 3] = c.r; ca[n * 3 + 1] = c.g; ca[n * 3 + 2] = c.b;
      }
      const fl = this.flash[bi]; const fo = x > 0 ? Math.exp(-x * 7) * amp : 0;
      fl.visible = false; // the burst core flash read as a square blob under bloom; the sparks carry the burst fl.position.copy(b.pos); fl.scale.setScalar(1.2 + 2.2 * (1 - fo)); fl.material.color.copy(b.color); fl.material.opacity = 0.7 * fo;
    });
    this.pts.geometry.attributes.position.needsUpdate = true; this.pts.geometry.attributes.color.needsUpdate = true;
  }
}
