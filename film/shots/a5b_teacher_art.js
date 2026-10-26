// a5b_teacher helpers: prompt cards (a city in its owner's colour, or Quill's slot locked and empty). Built once.
import * as THREE from 'three';

function rr(x, a, b, w, h, r) { r = Math.max(0, Math.min(r, w / 2, h / 2)); x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }
function tex(c) { const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.userData = { aspect: c.width / c.height }; return t; }

const SIDE = {
  quill: { border: '#a78bfa', bg0: 'rgba(58,24,120,0.95)', bg1: 'rgba(24,10,56,0.95)', text: '#f3efff', glow: '#8b5cf6' },
  ember: { border: '#fb923c', bg0: 'rgba(120,44,12,0.95)', bg1: 'rgba(52,18,6,0.95)', text: '#fff1e2', glow: '#f97316' },
};
const H = 168; // canvas height; the name is drawn at 84 px

function pinIcon(x, cx, cy, s, col) {
  x.save(); x.translate(cx, cy); x.scale(s, s);
  x.fillStyle = col; x.beginPath(); x.moveTo(0, 30); x.bezierCurveTo(-14, 12, -20, 2, -20, -8); x.arc(0, -8, 20, Math.PI, 0); x.bezierCurveTo(20, 2, 14, 12, 0, 30); x.fill();
  x.fillStyle = 'rgba(0,0,0,0.55)'; x.beginPath(); x.arc(0, -8, 7.5, 0, Math.PI * 2); x.fill();
  x.restore();
}

// a prompt card holding a city: pin + name, in the owner's colour
export function cityCardTexture(name, side) {
  const k = SIDE[side];
  const m = document.createElement('canvas').getContext('2d'); m.font = '700 84px Inter';
  const W = Math.ceil(m.measureText(name).width + 150);
  const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  const g = x.createLinearGradient(0, 0, 0, H); g.addColorStop(0, k.bg0); g.addColorStop(1, k.bg1);
  rr(x, 8, 8, W - 16, H - 16, 30); x.fillStyle = g; x.fill();
  const sh = x.createLinearGradient(0, 0, 0, H); sh.addColorStop(0, 'rgba(255,255,255,0.16)'); sh.addColorStop(0.5, 'rgba(255,255,255,0)');
  rr(x, 10, 10, W - 20, H - 20, 28); x.fillStyle = sh; x.fill();
  x.lineWidth = 7; x.strokeStyle = k.border; x.shadowColor = k.glow; x.shadowBlur = 18; rr(x, 8, 8, W - 16, H - 16, 30); x.stroke(); x.shadowBlur = 0;
  pinIcon(x, 62, H / 2 - 2, 1.25, k.border);
  x.font = '700 84px Inter'; x.textBaseline = 'middle'; x.fillStyle = k.text; x.fillText(name, 104, H / 2 + 4);
  return tex(c);
}

// Quill's slot in the linked Ember's prompt: the same card shape, dashed, empty, with a lock
export function lockedSlotTexture(widthLike = 'Stockholm') {
  const m = document.createElement('canvas').getContext('2d'); m.font = '700 84px Inter';
  const W = Math.ceil(m.measureText(widthLike).width + 150);
  const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  rr(x, 8, 8, W - 16, H - 16, 30); x.fillStyle = 'rgba(16,12,30,0.82)'; x.fill();
  x.setLineDash([22, 16]); x.lineWidth = 6; x.strokeStyle = 'rgba(167,139,250,0.75)'; rr(x, 8, 8, W - 16, H - 16, 30); x.stroke(); x.setLineDash([]);
  // lock
  const cx = W / 2, cy = H / 2 + 10;
  x.lineWidth = 11; x.strokeStyle = 'rgba(221,214,254,0.85)'; x.lineCap = 'round';
  x.beginPath(); x.moveTo(cx - 22, cy - 8); x.lineTo(cx - 22, cy - 26); x.arc(cx, cy - 26, 22, Math.PI, 0); x.lineTo(cx + 22, cy - 8); x.stroke();
  rr(x, cx - 36, cy - 10, 72, 52, 10); x.fillStyle = 'rgba(221,214,254,0.85)'; x.fill();
  x.fillStyle = '#140c28'; x.beginPath(); x.arc(cx, cy + 12, 7, 0, Math.PI * 2); x.fill(); x.fillRect(cx - 3.5, cy + 14, 7, 14);
  return tex(c);
}

export function cardPlane(t, h) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(h * t.userData.aspect, h), new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false, fog: false }));
  m.renderOrder = 25; m.visible = false; m.userData.w = h * t.userData.aspect; return m;
}
