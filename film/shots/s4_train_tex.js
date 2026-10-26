// Canvas textures for S4 (training). Built once in create(); nothing here depends on time.
import * as THREE from 'three';
import { hash } from '../lib/util.js';

function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return [c, c.getContext('2d')]; }
function tex(c) { const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter; return t; }
function rr(x, a, b, w, h, r) { r = Math.max(0, Math.min(r, w / 2, h / 2)); x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }

const KIND = {
  quill: { bg: 'rgba(60,24,128,0.92)', border: '#a78bfa', text: '#f3efff', dim: 'rgba(221,214,254,0.45)', r: 0.09 },
  ember: { bg: 'rgba(112,40,14,0.92)', border: '#fb923c', text: '#fff1e2', dim: 'rgba(254,215,170,0.45)', r: 0.5 },
};
export const PXU = 440; // canvas px per world unit for token tiles

// A token tile whose width is fixed by its character span (so tokens line up by character).
// Spaces are drawn as faint middle dots so the token boundaries stay honest.
export function tileTexture(text, kind, wW, hW) {
  const k = KIND[kind];
  const w = Math.max(8, Math.round(wW * PXU)), h = Math.round(hW * PXU);
  const [c, x] = canvas(w, h);
  rr(x, 3, 3, w - 6, h - 6, h * k.r); x.fillStyle = k.bg; x.fill();
  x.lineWidth = 5; x.strokeStyle = k.border; rr(x, 3, 3, w - 6, h - 6, h * k.r); x.stroke();
  // soft inner sheen
  const g = x.createLinearGradient(0, 0, 0, h); g.addColorStop(0, 'rgba(255,255,255,0.16)'); g.addColorStop(0.5, 'rgba(255,255,255,0)');
  rr(x, 5, 5, w - 10, h - 10, h * k.r); x.fillStyle = g; x.fill();
  let fs = Math.round(h * 0.56);
  const chars = [...text];
  const meas = () => { x.font = `700 ${fs}px 'JetBrains Mono'`; return chars.reduce((s, ch) => s + x.measureText(ch === ' ' ? '·' : ch).width, 0); };
  let tw = meas(); const room = w - (kind === 'ember' && chars.length > 1 ? h * 0.42 : 10);
  while (tw > room && fs > 10) { fs -= 2; tw = meas(); }
  x.textBaseline = 'middle';
  let px = (w - tw) / 2;
  for (const ch of chars) {
    const s = ch === ' ' ? '·' : ch; const cw = x.measureText(s).width;
    x.fillStyle = ch === ' ' ? k.dim : k.text; x.fillText(s, px, h / 2 + fs * 0.06); px += cw;
  }
  return { tex: tex(c), w: wW, h: hW };
}

// Outline-only version of a tile (white; tinted by the material color), used for highlights.
export function outlineTexture(kind, wW, hW) {
  const k = KIND[kind];
  const w = Math.max(8, Math.round(wW * PXU)), h = Math.round(hW * PXU);
  const [c, x] = canvas(w, h);
  x.shadowColor = '#fff'; x.shadowBlur = 10;
  x.lineWidth = 7; x.strokeStyle = '#fff'; rr(x, 5, 5, w - 10, h - 10, h * k.r); x.stroke();
  return tex(c);
}

// 4-point star sparkle
export function starTexture() {
  const [c, x] = canvas(128, 128);
  const g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.12, 'rgba(255,255,255,0.8)'); g.addColorStop(0.35, 'rgba(255,255,255,0.12)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 128, 128);
  x.globalCompositeOperation = 'lighter';
  for (const [dx, dy] of [[1, 0], [0, 1]]) {
    const lg = x.createLinearGradient(64 - dx * 64, 64 - dy * 64, 64 + dx * 64, 64 + dy * 64);
    lg.addColorStop(0, 'rgba(255,255,255,0)'); lg.addColorStop(0.5, 'rgba(255,255,255,0.95)'); lg.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = lg; if (dx) x.fillRect(0, 61, 128, 6); else x.fillRect(61, 0, 6, 128);
  }
  return tex(c);
}

// The ridge map M: a glass slab holding a heat-map of numbers.
export function matrixTexture() {
  const W = 512, H = 320; const [c, x] = canvas(W, H);
  rr(x, 4, 4, W - 8, H - 8, 22); x.fillStyle = 'rgba(6,36,32,0.55)'; x.fill();
  const cols = 14, rows = 8, m = 26, cw = (W - 2 * m) / cols, ch = (H - 2 * m) / rows;
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
    const v = hash(i * 17.3 + j * 5.1 + 3) * 2 - 1 + (i === Math.round(j * cols / rows) ? 0.9 : 0); // a faint diagonal (a map close to "same thing")
    const a = Math.min(1, Math.abs(v)) * 0.85;
    x.fillStyle = v > 0 ? `rgba(52,245,192,${a})` : `rgba(167,139,250,${a * 0.8})`;
    x.fillRect(m + i * cw + 2, m + j * ch + 2, cw - 4, ch - 4);
  }
  x.lineWidth = 6; x.strokeStyle = 'rgba(167,243,208,0.95)'; rr(x, 4, 4, W - 8, H - 8, 22); x.stroke();
  // bracket marks
  x.lineWidth = 8; x.strokeStyle = 'rgba(255,255,255,0.9)';
  x.beginPath(); x.moveTo(m + 14, m - 6); x.lineTo(m - 6, m - 6); x.lineTo(m - 6, H - m + 6); x.lineTo(m + 14, H - m + 6); x.stroke();
  x.beginPath(); x.moveTo(W - m - 14, m - 6); x.lineTo(W - m + 6, m - 6); x.lineTo(W - m + 6, H - m + 6); x.lineTo(W - m - 14, H - m + 6); x.stroke();
  // no label on the slab: the caption names the ridge regression
  return tex(c);
}

// A padlock icon; open or closed.
export function lockTexture(closed) {
  const [c, x] = canvas(128, 160);
  x.shadowColor = '#9fd8ff'; x.shadowBlur = 14;
  x.lineWidth = 13; x.strokeStyle = '#e6f6ff'; x.lineCap = 'round';
  x.beginPath();
  const lift = closed ? 0 : 22, sx = closed ? 0 : 14;
  x.moveTo(36 + sx, 78 - lift); x.lineTo(36 + sx, 52 - lift); x.arc(64 + sx, 52 - lift, 28, Math.PI, 0); x.lineTo(92 + sx, closed ? 78 : 62 - lift);
  x.stroke();
  rr(x, 18, 74, 92, 74, 14); x.fillStyle = '#bfe6ff'; x.fill();
  x.shadowBlur = 0; x.fillStyle = '#0b2440';
  x.beginPath(); x.arc(64, 104, 10, 0, Math.PI * 2); x.fill(); x.fillRect(59, 106, 10, 24);
  return tex(c);
}

// A small "x" mark for token ends that have no partner.
export function crossTexture(color = '#f87171') {
  const [c, x] = canvas(64, 64);
  x.lineWidth = 9; x.lineCap = 'round'; x.strokeStyle = color; x.shadowColor = color; x.shadowBlur = 6;
  x.beginPath(); x.moveTo(16, 16); x.lineTo(48, 48); x.moveTo(48, 16); x.lineTo(16, 48); x.stroke();
  return tex(c);
}

// A curriculum card (full mix) or a value-game chip.
export function cardTexture(label, { color = '#34f5c0', bg = 'rgba(4,34,30,0.9)', text = '#ecfffa', h = 132, font = 58, icon = true } = {}) {
  const [m, mx] = canvas(8, 8); mx.font = `600 ${font}px Inter`;
  const tw = mx.measureText(label).width;
  const pad = icon ? 96 : 40;
  const W = Math.ceil(tw + pad + 40), H = h;
  const [c, x] = canvas(W, H);
  rr(x, 4, 4, W - 8, H - 8, 24); x.fillStyle = bg; x.fill();
  const g = x.createLinearGradient(0, 0, 0, H); g.addColorStop(0, 'rgba(255,255,255,0.14)'); g.addColorStop(0.55, 'rgba(255,255,255,0)');
  rr(x, 6, 6, W - 12, H - 12, 22); x.fillStyle = g; x.fill();
  x.lineWidth = 5; x.strokeStyle = color; rr(x, 4, 4, W - 8, H - 8, 24); x.stroke();
  if (icon) { x.fillStyle = color; x.shadowColor = color; x.shadowBlur = 16; x.beginPath(); x.arc(56, H / 2, 13, 0, Math.PI * 2); x.fill(); x.shadowBlur = 0; }
  x.font = `600 ${font}px Inter`; x.textBaseline = 'middle'; x.fillStyle = text;
  x.fillText(label, icon ? 90 : 40, H / 2 + 3);
  const t = tex(c); t.userData = { aspect: W / H }; return t;
}
