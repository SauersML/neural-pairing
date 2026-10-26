import * as THREE from 'three';
import { CSS } from './palette.js';

// Text tokens as glowing 3D tiles. Quill's tokenizer and Ember's tokenizer split text differently,
// so their tiles have different shapes: Quill = sharp-cornered tiles, Ember = round pills.
// Tokens never cross the link — use these only for prompts (input) and written output.
const cache = new Map();
export function textTexture(text, o = {}) {
  const key = JSON.stringify([text, o]);
  if (cache.has(key)) return cache.get(key);
  const font = o.font || '600 64px Inter';
  const pad = o.pad ?? 26, h = o.h ?? 104, radius = o.radius ?? 18;
  const c = document.createElement('canvas'); const x = c.getContext('2d'); x.font = font;
  const w = Math.ceil(x.measureText(text).width + pad * 2);
  c.width = Math.max(h, w); c.height = h;
  x.font = font; x.textBaseline = 'middle'; x.textAlign = 'center';
  if (o.bg !== null) {
    x.fillStyle = o.bg || 'rgba(20,14,40,0.82)';
    roundRect(x, 3, 3, c.width - 6, c.height - 6, radius); x.fill();
  }
  if (o.border) { x.lineWidth = o.borderWidth || 4; x.strokeStyle = o.border; roundRect(x, 3, 3, c.width - 6, c.height - 6, radius); x.stroke(); }
  if (o.glow) { x.shadowColor = o.glow; x.shadowBlur = 18; }
  x.fillStyle = o.color || '#fff';
  x.fillText(text, c.width / 2, c.height / 2 + (o.dy ?? 3));
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  tex.userData = { aspect: c.width / c.height };
  cache.set(key, tex); return tex;
}
function roundRect(x, a, b, w, h, r) { x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }

export const TOKEN_STYLE = {
  quill: { color: '#ede9fe', bg: 'rgba(76,29,149,0.85)', border: CSS.quill, radius: 10 },
  ember: { color: '#ffedd5', bg: 'rgba(124,45,18,0.85)', border: CSS.ember, radius: 50 },
  prompt: { color: '#ffffff', bg: 'rgba(10,8,20,0.85)', border: 'rgba(255,255,255,0.6)', radius: 16 },
  secret: { color: '#fff7d6', bg: 'rgba(60,40,0,0.85)', border: '#ffd166', radius: 16, glow: '#ffd166' },
  ghost: { color: '#c9cbd6', bg: 'rgba(40,42,52,0.85)', border: '#8b8fa3', radius: 16 },
  wrong: { color: '#fee2e2', bg: 'rgba(90,10,10,0.85)', border: CSS.bad, radius: 16 },
  link: { color: '#d1fae5', bg: 'rgba(4,60,45,0.85)', border: CSS.link, radius: 16 },
};

// A token tile mesh. height in world units; width follows the text.
export function tokenTile(text, style = 'quill', height = 0.32, extra = {}) {
  const st = typeof style === 'string' ? TOKEN_STYLE[style] : style;
  const tex = textTexture(text, { ...st, ...extra });
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(height * tex.userData.aspect, height), mat);
  mesh.userData.width = height * tex.userData.aspect;
  return mesh;
}

// A billboarding helper: make mesh face the camera
export function billboard(mesh, camera) { mesh.quaternion.copy(camera.quaternion); }
