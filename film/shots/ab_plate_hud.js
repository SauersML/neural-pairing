// Engraved labels for the anatomical plate (ab_plate): an SVG layer over the canvas, printed in the plate's inks
// (multiply blend, so the paper's tooth shows through the ink), with a faint etched roughness on every stroke.
// Immediate mode: each frame call begin(), then text()/line()/path() for what is visible, then end() (anything not
// drawn this frame is hidden). Strokes draw themselves (reveal 0..1 along the path); text writes itself left to right.
export const INK = '#2b1b11', TEAL = '#2c6b64', RED = '#a23e2e';
export const SERIF = "'Plate Garamond', 'EB Garamond', 'Liberation Serif', serif";
const NS = 'http://www.w3.org/2000/svg';

export async function loadPlateFonts() {
  const base = new URL('../lib/plate_fonts/', import.meta.url).href;
  const faces = [['eb-garamond-latin-400-normal', 'normal', '400'], ['eb-garamond-latin-400-italic', 'italic', '400'],
    ['eb-garamond-latin-500-normal', 'normal', '500'], ['eb-garamond-latin-500-italic', 'italic', '500']];
  for (const [f, style, weight] of faces) {
    try { const ff = new FontFace('Plate Garamond', `url(${base}${f}.woff2)`, { style, weight }); await ff.load(); document.fonts.add(ff); } catch (_) { /* fallback serif */ }
  }
}

export class PlateHud {
  constructor(parent) {
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('viewBox', '0 0 1920 1080'); this.svg.setAttribute('width', '1920'); this.svg.setAttribute('height', '1080');
    Object.assign(this.svg.style, { position: 'absolute', left: '0', top: '0', mixBlendMode: 'multiply', overflow: 'visible' });
    this.svg.innerHTML = `<defs><filter id="plateRough" x="-5%" y="-5%" width="110%" height="110%">
      <feTurbulence type="fractalNoise" baseFrequency="0.55" numOctaves="2" seed="7" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="1.6" xChannelSelector="R" yChannelSelector="G"/></filter></defs>`;
    this.g = document.createElementNS(NS, 'g'); this.g.setAttribute('filter', 'url(#plateRough)'); this.svg.appendChild(this.g);
    parent.appendChild(this.svg);
    this.els = new Map(); this.used = new Set(); this.nclip = 0;
  }
  begin() { this.used.clear(); }
  end() { for (const [k, e] of this.els) { const on = this.used.has(k); if (e._on !== on) { e.node.style.display = on ? '' : 'none'; e._on = on; } } }
  _get(key, make) { let e = this.els.get(key); if (!e) { e = make(); e._on = true; this.els.set(key, e); } this.used.add(key); return e; }

  // a stroke along an SVG path d; reveal draws it from its start
  path(key, d, { reveal = 1, width = 1.6, color = INK, opacity = 1, dash = null, fill = 'none', cap = 'round' } = {}) {
    const e = this._get(key, () => { const n = document.createElementNS(NS, 'path'); n.setAttribute('pathLength', '1'); this.g.appendChild(n); return { node: n }; });
    const n = e.node;
    if (e.d !== d) { n.setAttribute('d', d); e.d = d; }
    n.setAttribute('stroke', color); n.setAttribute('stroke-width', width); n.setAttribute('fill', fill); n.setAttribute('stroke-linecap', cap);
    n.setAttribute('opacity', opacity);
    if (dash) { n.removeAttribute('pathLength'); n.setAttribute('stroke-dasharray', dash); n.style.strokeDashoffset = ''; n.setAttribute('opacity', opacity * Math.min(1, reveal * 3)); }
    else { n.setAttribute('pathLength', '1'); n.setAttribute('stroke-dasharray', '1 1'); n.style.strokeDashoffset = String(1 - Math.max(0, Math.min(1, reveal))); }
    return n;
  }
  line(key, x1, y1, x2, y2, o = {}) { return this.path(key, `M${x1.toFixed(1)},${y1.toFixed(1)} L${x2.toFixed(1)},${y2.toFixed(1)}`, o); }
  dot(key, x, y, r = 3.2, { color = INK, opacity = 1 } = {}) {
    const e = this._get(key, () => { const n = document.createElementNS(NS, 'circle'); this.g.appendChild(n); return { node: n }; });
    e.node.setAttribute('cx', x.toFixed(1)); e.node.setAttribute('cy', y.toFixed(1)); e.node.setAttribute('r', r); e.node.setAttribute('fill', color); e.node.setAttribute('opacity', opacity);
  }
  // text that writes itself (a clip that opens left to right); anchor 'start' | 'middle' | 'end'
  text(key, str, x, y, { size = 42, italic = true, weight = 400, color = INK, anchor = 'start', reveal = 1, opacity = 1, spacing = 0.01 } = {}) {
    const e = this._get(key, () => {
      const id = `pclip${this.nclip++}`;
      const cp = document.createElementNS(NS, 'clipPath'); cp.setAttribute('id', id); const r = document.createElementNS(NS, 'rect'); cp.appendChild(r);
      this.svg.querySelector('defs').appendChild(cp);
      const n = document.createElementNS(NS, 'text'); n.setAttribute('clip-path', `url(#${id})`); this.g.appendChild(n);
      return { node: n, rect: r, cp, w: null, str: null };
    });
    const n = e.node;
    if (e.str !== str) { n.textContent = str; e.str = str; e.w = null; }
    n.setAttribute('x', x.toFixed(1)); n.setAttribute('y', y.toFixed(1));
    n.setAttribute('text-anchor', anchor); n.setAttribute('fill', color); n.setAttribute('opacity', opacity);
    n.style.font = `${italic ? 'italic ' : ''}${weight} ${size}px ${SERIF}`; n.style.letterSpacing = `${spacing}em`;
    const w = str.length * size * 0.56 + 20, x0 = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
    const rv = Math.max(0, Math.min(1, reveal));
    if (rv >= 1) n.removeAttribute('clip-path'); else n.setAttribute('clip-path', `url(#${e.cp.id})`);
    e.rect.setAttribute('x', (x0 - 8).toFixed(1)); e.rect.setAttribute('y', (y - size * 1.1).toFixed(1));
    e.rect.setAttribute('width', (w * rv).toFixed(1)); e.rect.setAttribute('height', (size * 1.6).toFixed(1));
    return { x0, w };
  }
}
