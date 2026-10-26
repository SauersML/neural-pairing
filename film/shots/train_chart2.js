// a5_train (v9): the training curve as a real chart (DOM + SVG, the film's glow look). Data: shots/train_curve_data.js
// (run nl20): at each evaluation step, the median over 144 held-out value tests of the reader's chance of writing the
// whole right value, linked and with the link cut. FINAL2: after the full training mix (step 1,320, 180 tests).
// x = training steps (0 .. 1,320, ticks 0 / 300 / 600 / 900 / 1,200), y = 0 .. 100 %, linear.
// Lines: monotone cubic through the real points only (no overshoot), drawn up to the current step.
import { el, FONT } from '../lib/hud.js';
import { clamp } from '../lib/util.js';
import { CURVE2, FINAL2 } from './train_curve_data.js';

const TEAL = '#34f5c0', GREY = '#a3a8ba', TXT = '#f5f3ff';
const svgNS = 'http://www.w3.org/2000/svg';
function S(parent, tag, attrs = {}) { const e = document.createElementNS(svgNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }

export const STEP_END = FINAL2[0]; // 1,320
const PTS = {
  linked: [...CURVE2.map((r) => [r[0], r[1]]), [FINAL2[0], FINAL2[1]]],
  cut: [...CURVE2.map((r) => [r[0], r[2]]), [FINAL2[0], FINAL2[2]]],
};
// Fritsch–Carlson monotone cubic: passes through every real point, never overshoots between them
function monotone(pts) {
  const n = pts.length, d = [], m = [];
  for (let k = 0; k < n - 1; k++) d.push((pts[k + 1][1] - pts[k][1]) / (pts[k + 1][0] - pts[k][0]));
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let k = 1; k < n - 1; k++) m[k] = d[k - 1] * d[k] <= 0 ? 0 : (d[k - 1] + d[k]) / 2;
  for (let k = 0; k < n - 1; k++) {
    if (d[k] === 0) { m[k] = m[k + 1] = 0; continue; }
    const a = m[k] / d[k], b = m[k + 1] / d[k], s = a * a + b * b;
    if (s > 9) { const tau = 3 / Math.sqrt(s); m[k] = tau * a * d[k]; m[k + 1] = tau * b * d[k]; }
  }
  return (x) => {
    if (x <= pts[0][0]) return pts[0][1];
    for (let k = 0; k < n - 1; k++) if (x <= pts[k + 1][0]) {
      const h = pts[k + 1][0] - pts[k][0], u = (x - pts[k][0]) / h, u2 = u * u, u3 = u2 * u;
      return (2 * u3 - 3 * u2 + 1) * pts[k][1] + (u3 - 2 * u2 + u) * h * m[k] + (-2 * u3 + 3 * u2) * pts[k + 1][1] + (u3 - u2) * h * m[k + 1];
    }
    return pts[n - 1][1];
  };
}
const fLinked = monotone(PTS.linked), fCut = monotone(PTS.cut);
// the last measured point at or before a step (the readout only shows measured values)
function lastPoint(step) { let p = PTS.linked[0]; for (const c of PTS.linked) if (c[0] <= step + 1e-6) p = c; return p; }
const fmt = (v) => (v >= 99.5 || v < 1 ? v.toFixed(1) : v >= 10 ? Math.round(v).toString() : v.toFixed(1)) + '%';

// geometry: the plot rectangle in stage px (labels go outside it)
export function makeTrainChart(parent, { left = 1130, top = 250, w = 640, h = 470 } = {}) {
  const PADL = 150, PADT = 120, PADB = 140, PADR = 60;
  const root = el(parent, 'div', { position: 'absolute', left: `${left - PADL}px`, top: `${top - PADT}px`, width: `${w + PADL + PADR}px`, height: `${h + PADT + PADB}px`, display: 'none', pointerEvents: 'none' });
  const svg = S(root, 'svg', { width: w + PADL + PADR, height: h + PADT + PADB, viewBox: `0 0 ${w + PADL + PADR} ${h + PADT + PADB}`, overflow: 'visible' });
  const X0 = PADL, Y0 = PADT;
  const px = (step) => X0 + (step / STEP_END) * w, py = (v) => Y0 + h - (v / 100) * h;
  const defs = S(svg, 'defs');
  const gr = S(defs, 'linearGradient', { id: 'tc2fill', x1: 0, y1: 0, x2: 0, y2: 1 });
  S(gr, 'stop', { offset: 0, 'stop-color': TEAL, 'stop-opacity': 0.3 }); S(gr, 'stop', { offset: 1, 'stop-color': TEAL, 'stop-opacity': 0 });
  // panel
  S(svg, 'rect', { x: 16, y: Y0 - 100, width: w + PADL + PADR - 32, height: h + 100 + PADB - 10, rx: 28, fill: 'rgba(10,8,22,0.66)', stroke: 'rgba(245,243,255,0.12)' });
  const T = (x, y, s, css, anchor = 'start') => { const t = S(svg, 'text', { x, y, 'text-anchor': anchor, style: css }); t.textContent = s; return t; };
  // title (the y quantity) and axis labels
  T(48, Y0 - 44, 'Chance of writing the partner’s value', `font:700 44px ${FONT.display};letter-spacing:-0.01em;fill:#eafff8`);
  for (const v of [0, 50, 100]) {
    S(svg, 'line', { x1: X0, y1: py(v), x2: X0 + w, y2: py(v), stroke: v === 0 ? 'rgba(245,243,255,0.4)' : 'rgba(245,243,255,0.13)', 'stroke-width': 2 });
    T(X0 - 20, py(v) + 13, `${v}%`, `font:600 38px ${FONT.text};fill:rgba(245,243,255,0.82)`, 'end');
  }
  S(svg, 'line', { x1: X0, y1: py(0), x2: X0, y2: py(100), stroke: 'rgba(245,243,255,0.4)', 'stroke-width': 2 });
  for (const s of [0, 300, 600, 900, 1200]) {
    S(svg, 'line', { x1: px(s), y1: py(0), x2: px(s), y2: py(0) + 12, stroke: 'rgba(245,243,255,0.5)', 'stroke-width': 2 });
    T(px(s), py(0) + 52, s === 1200 ? '1,200' : String(s), `font:600 36px ${FONT.text};fill:rgba(245,243,255,0.82)`, 'middle');
  }
  T(X0 + w / 2, py(0) + 108, 'training steps', `font:600 40px ${FONT.text};fill:rgba(245,243,255,0.9)`, 'middle');
  // lines
  const area = S(svg, 'path', { d: '', fill: 'url(#tc2fill)' });
  const cutPath = S(svg, 'path', { d: '', fill: 'none', stroke: GREY, 'stroke-width': 6, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  const path = S(svg, 'path', { d: '', fill: 'none', stroke: TEAL, 'stroke-width': 7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  path.style.filter = 'drop-shadow(0 0 10px rgba(52,245,192,0.9))';
  // the measured points (the line between them is a smooth fit through them)
  const marks = PTS.linked.map(([st, v]) => S(svg, 'circle', { cx: px(st), cy: py(v), r: 6, fill: '#bffbe8', opacity: 0 }));
  const head = S(svg, 'circle', { r: 11, fill: '#eafff8' }); head.style.filter = 'drop-shadow(0 0 14px rgba(52,245,192,1))';
  const cutHead = S(svg, 'circle', { r: 8, fill: '#d4d7e3' });
  const lLinked = T(px(440), py(52), 'linked', `font:700 42px ${FONT.text};fill:${TEAL}`);
  const lCut = T(px(1310), py(0) - 22, 'link cut', `font:700 42px ${FONT.text};fill:${GREY}`, 'end');
  const val = T(0, 0, '', `font:800 64px ${FONT.display};letter-spacing:-0.03em;fill:#eafff8`);
  return {
    root,
    // step: current training step (0..STEP_END); a: opacity; pulse: 0..1 head glow
    update(step, a, pulse = 0) {
      root.style.display = a > 0.002 ? 'block' : 'none'; if (a <= 0.002) return;
      root.style.opacity = a;
      const N = 90, up = clamp(step, 0, STEP_END);
      let d = '', dc = '';
      for (let k = 0; k <= N; k++) {
        const s = up * k / N;
        d += (k ? 'L' : 'M') + px(s).toFixed(1) + ' ' + py(fLinked(s)).toFixed(1);
        dc += (k ? 'L' : 'M') + px(s).toFixed(1) + ' ' + py(fCut(s)).toFixed(1);
      }
      path.setAttribute('d', d); cutPath.setAttribute('d', dc);
      area.setAttribute('d', d + ` L ${px(up).toFixed(1)} ${py(0)} L ${X0} ${py(0)} Z`);
      head.setAttribute('cx', px(up)); head.setAttribute('cy', py(fLinked(up))); head.setAttribute('r', String(11 + 7 * pulse));
      cutHead.setAttribute('cx', px(up)); cutHead.setAttribute('cy', py(fCut(up)));
      // direct labels appear once their line has reached them
      lLinked.setAttribute('opacity', String(clamp((up - 420) / 100)));
      lCut.setAttribute('opacity', String(clamp((up - 120) / 120)));
      // readout: the last measured value, at its own point (never an in-between value)
      const [ls, lv] = lastPoint(up);
      marks.forEach((m, k) => m.setAttribute('opacity', PTS.linked[k][0] <= up + 1e-6 ? '1' : '0'));
      const mx = px(ls), my = py(lv), right = ls > 1050;
      val.setAttribute('text-anchor', right ? 'end' : 'start');
      val.setAttribute('x', right ? mx - 10 : mx + 22); val.setAttribute('y', lv < 5 ? my - 24 : lv < 80 ? my + 58 : my + 76); // below-right of a rising point
      val.textContent = up < 15 ? '0%' : fmt(lv);
      void ls;
    },
  };
}
