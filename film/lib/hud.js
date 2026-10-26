// DOM overlay typography, composited over the 3D canvas (captured together by the renderer).
// The stage is always 1920x1080 CSS px. Everything is driven by time (pure functions of t).
import { clamp, prog, easeOutCubic, easeOutExpo, smoother } from './util.js';

export const FONT = { display: "'Inter Display', 'Inter', sans-serif", text: "'Inter', sans-serif", mono: "'JetBrains Mono', monospace" };

export function el(parent, tag = 'div', css = {}, html = '') {
  const e = document.createElement(tag); Object.assign(e.style, css); if (html) e.innerHTML = html; parent.appendChild(e); return e;
}

// Kinetic caption: a headline and an optional sub line. (Kickers are switched off film-wide: they only repeated the headline.)
// Words rise and un-blur in sequence; out is a quick fade + drift.
// pos: {left, top, right, bottom, align}
export class Caption {
  constructor(parent, { kicker = '', head = '', sub = '', pos = { left: 110, bottom: 100 }, size = 92, subSize = 46, width = 1700, align = 'left', color = '#f5f3ff', accent = '#34f5c0' } = {}) {
    this.root = el(parent, 'div', { position: 'absolute', width: width + 'px', textAlign: align, ...px(pos), pointerEvents: 'none' });
    this.kick = el(this.root, 'div', { font: `600 30px ${FONT.mono}`, letterSpacing: '0.16em', color: accent, textTransform: 'uppercase', marginBottom: '16px' }, '');
    this.head = el(this.root, 'div', { font: `650 ${size}px ${FONT.display}`, lineHeight: 1.04, letterSpacing: '-0.02em', color, textShadow: '0 2px 30px rgba(0,0,0,0.65)' });
    this.sub = el(this.root, 'div', { font: `400 ${subSize}px ${FONT.text}`, lineHeight: 1.3, color: 'rgba(245,243,255,0.78)', marginTop: '16px', textShadow: '0 2px 20px rgba(0,0,0,0.8)' });
    this.kickWords = wordsInto(this.kick, ''); void kicker; this.headWords = wordsInto(this.head, head); this.subWords = wordsInto(this.sub, sub);
  }
  // show between tIn and tOut (global seconds)
  update(t, tIn, tOut, { stagger = 0.045, dur = 0.5, outDur = 0.35 } = {}) {
    const vis = t >= tIn - 0.01 && t <= tOut + 0.01;
    this.root.style.display = vis ? 'block' : 'none'; if (!vis) return;
    const out = prog(t, tOut - outDur, tOut);
    this.root.style.opacity = 1 - out; this.root.style.transform = `translateY(${-out * 18}px)`;
    let k = 0;
    const anim = (arr, d0) => arr.forEach((w) => { const p = easeOutExpo(clamp((t - tIn - d0 - k * stagger) / dur)); k++; w.style.opacity = p; w.style.transform = `translateY(${(1 - p) * 0.55}em)`; w.style.filter = `blur(${(1 - p) * 8}px)`; });
    anim(this.kickWords, 0); anim(this.headWords, 0.08); anim(this.subWords, 0.25);
  }
}
// wraps each word in a span; supports inline HTML tags via {tag} markers: pass already-safe html strings.
// Words separated by spaces; use &nbsp; to glue. Spans keep inline styles from <b style=...> segments if you pass html.
export function wordsInto(parent, html) {
  if (!html) { parent.style.display = 'none'; return []; }
  parent.style.display = 'block';
  // split into words at spaces outside tags; a word inside an open inline tag (e.g. a colored <span>) is re-wrapped
  // in that tag, so colored phrases of several words work.
  const words = []; const open = []; let cur = '', i = 0;
  while (i < html.length) {
    if (html[i] === '<') {
      const j = html.indexOf('>', i); const tag = html.slice(i, j + 1);
      if (tag.startsWith('</')) { cur += tag; open.pop(); }
      else if (tag.endsWith('/>') || /^<br/i.test(tag)) cur += tag;
      else { cur += tag; open.push(tag); }
      i = j + 1; continue;
    }
    if (html[i] === ' ') {
      // close any tags open in this word, remember to reopen them in the next
      const st = open.slice(); cur += st.slice().reverse().map((t) => '</' + t.slice(1).split(/[\s>]/)[0] + '>').join('');
      if (cur.replace(/<[^>]*>/g, '').trim()) words.push(cur); cur = st.join(''); i++; continue;
    }
    cur += html[i]; i++;
  }
  if (cur.replace(/<[^>]*>/g, '').trim()) words.push(cur);
  return words.map((w) => { const s = document.createElement('span'); s.style.display = 'inline-block'; s.style.marginRight = '0.26em'; s.style.willChange = 'transform'; s.innerHTML = w; parent.appendChild(s); return s; });
}
export function px(o) { const r = {}; for (const k in o) r[k] = typeof o[k] === 'number' ? o[k] + 'px' : o[k]; return r; }

// A label pinned to a 3D world point (projected each frame). Optional leader dot.
export class Pin {
  constructor(parent, html, { color = '#f5f3ff', size = 34, font = FONT.text, dot = true, anchor = 'left', weight = 600, bg = null } = {}) {
    this.root = el(parent, 'div', { position: 'absolute', left: 0, top: 0, whiteSpace: 'nowrap', font: `${weight} ${size}px ${font}`, color, textShadow: '0 1px 12px rgba(0,0,0,0.9)', pointerEvents: 'none' });
    if (bg) Object.assign(this.root.style, { background: bg, padding: '4px 10px', borderRadius: '8px' });
    this.anchor = anchor;
    this.root.innerHTML = (dot ? `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:10px;vertical-align:middle;box-shadow:0 0 10px ${color}"></span>` : '') + html;
  }
  // world: THREE.Vector3, camera; dx/dy offsets in px; opacity 0..1
  update(world, camera, opacity = 1, dx = 0, dy = 0) {
    camera.updateMatrixWorld(); const v = world.clone().project(camera);
    const x = (v.x * 0.5 + 0.5) * 1920 + dx, y = (-v.y * 0.5 + 0.5) * 1080 + dy;
    const vis = opacity > 0.001 && v.z < 1;
    this.root.style.display = vis ? 'block' : 'none'; if (!vis) return;
    const tx = this.anchor === 'right' ? '-100%' : this.anchor === 'center' ? '-50%' : '0%';
    this.root.style.transform = `translate(${x}px, ${y}px) translate(${tx}, -50%)`;
    this.root.style.opacity = opacity;
  }
  set(html) { this.root.innerHTML = html; }
}
// screen position of a world point (CSS px on the 1920x1080 stage)
export function toScreen(world, camera) { camera.updateMatrixWorld(); const v = world.clone().project(camera); return { x: (v.x * 0.5 + 0.5) * 1920, y: (-v.y * 0.5 + 0.5) * 1080, behind: v.z > 1 }; }

// Persistent chapter tag (top-left): "02 / THE LINK" style. Call update each frame.
export class Chapter {
  constructor(parent, num, title) {
    this.root = el(parent, 'div', { position: 'absolute', left: '110px', top: '70px', font: `600 30px ${FONT.mono}`, letterSpacing: '0.18em', color: 'rgba(245,243,255,0.8)', textTransform: 'uppercase' },
      `<span style="color:#34f5c0">${num}</span><span style="display:inline-block;width:46px;height:1px;background:rgba(245,243,255,0.5);vertical-align:middle;margin:0 16px"></span>${title}`);
  }
  // Chapter tags are switched off for the whole film (director's call): no numbered step labels.
  update(t, a, b) { this.root.style.display = 'none'; return; const o = Math.min(prog(t, a, a + 0.5), 1 - prog(t, b - 0.3, b)); this.root.style.display = o > 0.001 ? 'block' : 'none'; this.root.style.opacity = o; this.root.style.transform = `translateX(${(1 - o) * -20}px)`; }
}
