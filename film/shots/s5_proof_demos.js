// S5 helper: result windows, each playing a tiny 2D demo of its game, with the number below.
// Default: three 572-px windows (S5). opts can override captions/stats, shrink the layout (width, gap, top) and add a
// fourth "free chat" window (opts.chat) — used by a6_proof.
//   1. code lookup (a game the link never trained on) -> 17 / 20
//   2. turn structures never trained (dialogue, silent rounds, shared pen) -> 128 / 128
//   3. shared reading -> 31 / 32
// Mini creatures match the 3D characters (Quill violet + feather, Ember orange + flame). The link is a teal line;
// only dots ride it, never text. All text is >= 30 px. Every update is a pure function of global t.
import { el, FONT } from '../lib/hud.js';
import { clamp, prog, spring, easeOutExpo, easeOutCubic, hash } from '../lib/util.js';

const TEAL = '#34f5c0';
const DEMO_H = 430;
const QUILL = { fill: '#8b5cf6', hi: '#c4b5fd', deep: '#4c1d95' }, EMBER = { fill: '#f97316', hi: '#fdba74', deep: '#9a3412' };

const svgNS = 'http://www.w3.org/2000/svg';
function svgEl(parent, tag, attrs = {}) { const e = document.createElementNS(svgNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }

// a mini creature head (84 px), facing its partner
function makeIcon(parent, side, x, y) {
  const pal = side === 'quill' ? QUILL : EMBER, dir = side === 'quill' ? 1 : -1;
  const root = el(parent, 'div', { position: 'absolute', left: `${x - 60}px`, top: `${y - 70}px`, width: '120px', height: '130px' });
  const svg = svgEl(root, 'svg', { width: 120, height: 130, viewBox: '-60 -75 120 130' });
  const defs = svgEl(svg, 'defs');
  const gid = `g_${side}_${Math.round(x)}_${Math.round(y)}`;
  const grad = svgEl(defs, 'radialGradient', { id: gid, cx: '0.38', cy: '0.32', r: '0.75' });
  svgEl(grad, 'stop', { offset: '0', 'stop-color': pal.hi }); svgEl(grad, 'stop', { offset: '0.55', 'stop-color': pal.fill }); svgEl(grad, 'stop', { offset: '1', 'stop-color': pal.deep });
  if (side === 'quill') {
    for (const [rot, len] of [[-28, 30], [0, 40], [28, 30]]) svgEl(svg, 'ellipse', { cx: 0, cy: -34 - len / 2, rx: 7, ry: len / 2, fill: pal.hi, opacity: 0.9, transform: `rotate(${rot} 0 -34)` });
  } else {
    svgEl(svg, 'path', { d: 'M0 -74 C 14 -56, 18 -44, 0 -32 C -18 -44, -14 -56, 0 -74 Z', fill: '#ffd166', opacity: 0.95 });
    svgEl(svg, 'path', { d: 'M0 -62 C 7 -52, 9 -44, 0 -37 C -9 -44, -7 -52, 0 -62 Z', fill: '#fff3c4' });
  }
  svgEl(svg, 'circle', { cx: 0, cy: 0, r: 36, fill: `url(#${gid})` });
  const eyes = svgEl(svg, 'g'), happy = svgEl(svg, 'g', { opacity: 0 });
  for (const s of [-1, 1]) {
    svgEl(eyes, 'ellipse', { cx: s * 13 + dir * 4, cy: -4, rx: 7.5, ry: 9.5, fill: '#fff' });
    svgEl(eyes, 'circle', { cx: s * 13 + dir * 6.5, cy: -3, r: 4.2, fill: '#0a0612' });
    svgEl(happy, 'path', { d: `M ${s * 13 + dir * 4 - 7} -2 Q ${s * 13 + dir * 4} -12 ${s * 13 + dir * 4 + 7} -2`, stroke: '#fff', 'stroke-width': 4, fill: 'none', 'stroke-linecap': 'round' });
  }
  return {
    root,
    update(t, t0, isHappy = 0) {
      const s = spring(t, t0, 2.4, 0.45);
      root.style.opacity = clamp((t - t0) / 0.1);
      root.style.transform = `translateY(${Math.sin(t * 3.2 + x) * 3}px) scale(${Math.max(0.01, s)})`;
      eyes.setAttribute('opacity', String(1 - isHappy)); happy.setAttribute('opacity', String(isHappy));
    },
  };
}

// a token pill (Quill = square-cornered, Ember = round), text >= 32 px
function makePill(parent, html, side, { x, y, anchor = 'center', size = 32, ghostText = false } = {}) {
  const st = side === 'quill' ? { bg: 'rgba(76,29,149,0.9)', bd: '#8b5cf6', fg: '#ede9fe', r: '9px' }
    : side === 'ember' ? { bg: 'rgba(124,45,18,0.9)', bd: '#f97316', fg: '#ffedd5', r: '999px' }
      : { bg: 'rgba(4,60,45,0.9)', bd: TEAL, fg: '#d1fae5', r: '12px' };
  const root = el(parent, 'div', { position: 'absolute', left: `${x}px`, top: `${y}px`, whiteSpace: 'nowrap', font: `600 ${size}px ${FONT.text}`, lineHeight: '1', padding: '9px 18px 10px', color: st.fg, background: st.bg, border: `2px solid ${st.bd}`, borderRadius: st.r, letterSpacing: '-0.01em' }, html);
  const tx = anchor === 'right' ? '-100%' : anchor === 'left' ? '0%' : '-50%';
  return {
    root,
    update(t, t0, alpha = 1, glow = 0) {
      if (t < t0 || alpha <= 0.002) { root.style.display = 'none'; return; }
      root.style.display = 'block';
      const s = spring(t, t0, 2.6, 0.5);
      root.style.opacity = alpha * clamp((t - t0) / 0.08);
      root.style.transform = `translate(${tx}, -50%) translateY(${(1 - s) * 18}px) scale(${0.7 + 0.3 * s})`;
      root.style.boxShadow = glow > 0.01 ? `0 0 ${18 + 16 * glow}px rgba(52,245,192,${0.65 * glow})` : 'none';
      if (glow > 0.01) root.style.borderColor = TEAL; else root.style.borderColor = '';
    },
  };
}
// a "speech" pill with grey bars instead of words (dialogue turns)
function makeBubble(parent, side, { x, y, w }) {
  const p = makePill(parent, '', side, { x, y, anchor: side === 'quill' ? 'left' : 'right' });
  p.root.style.width = `${w}px`; p.root.style.height = '50px'; p.root.style.padding = '0';
  const bars = el(p.root, 'div', { position: 'absolute', left: '20px', right: '20px', top: '19px', height: '10px', borderRadius: '5px', background: 'repeating-linear-gradient(90deg, rgba(255,255,255,0.55) 0 46px, transparent 46px 58px)' });
  return p;
}
function makeCheck(parent, x, y, size = 50) {
  const root = el(parent, 'div', { position: 'absolute', left: `${x - size / 2}px`, top: `${y - size / 2}px`, width: `${size}px`, height: `${size}px`, borderRadius: '50%', background: 'rgba(4,46,36,0.95)', border: `3px solid ${TEAL}`, boxShadow: '0 0 20px rgba(52,245,192,0.6)' });
  root.innerHTML = `<svg width="${size - 6}" height="${size - 6}" viewBox="0 0 58 58" style="position:absolute;left:0;top:0"><path d="M15 30 L25 40 L44 19" fill="none" stroke="${TEAL}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/></svg>`;
  const p = root.querySelector('path');
  return {
    root,
    update(t, t0, alpha = 1) {
      if (t < t0 || alpha <= 0.002) { root.style.display = 'none'; return; }
      root.style.display = 'block';
      const s = spring(t, t0, 2.8, 0.42);
      root.style.opacity = alpha; root.style.transform = `scale(${Math.max(0.01, s)})`;
      p.setAttribute('stroke-dashoffset', String(1 - easeOutCubic(clamp((t - t0 - 0.05) / 0.25))));
    },
  };
}
// the link: a teal line between the two icons with dots riding it (dir: +1 Quill->Ember, -1 Ember->Quill, 0 both)
function makeLink(parent, x0, x1, y, W = 572) {
  const root = el(parent, 'div', { position: 'absolute', left: '0px', top: '0px', width: `${W}px`, height: `${DEMO_H}px`, pointerEvents: 'none' });
  const svg = svgEl(root, 'svg', { width: W, height: DEMO_H });
  const line = svgEl(svg, 'line', { x1: x0, y1: y, x2: x1, y2: y, stroke: TEAL, 'stroke-width': 4, 'stroke-linecap': 'round', opacity: 0 });
  line.style.filter = 'drop-shadow(0 0 6px rgba(52,245,192,0.9))';
  const N = 10, dots = [];
  for (let k = 0; k < N; k++) { const c = svgEl(svg, 'circle', { cx: x0, cy: y, r: 7, fill: '#d6fff2', opacity: 0 }); c.style.filter = 'drop-shadow(0 0 7px rgba(52,245,192,1))'; dots.push(c); }
  return {
    // grow 0..1, flowA (Quill->Ember strength), flowB (Ember->Quill strength)
    update(t, grow, flowA, flowB) {
      const xe = x0 + (x1 - x0) * clamp(grow);
      line.setAttribute('x2', xe); line.setAttribute('opacity', String(clamp(grow * 3) * 0.9));
      dots.forEach((c, k) => {
        const a = k % 2 === 0 ? 1 : -1, f = a > 0 ? flowA : flowB, slot = Math.floor(k / 2);
        let u = (t * 0.9 + slot / 5 + hash(k) * 0.05) % 1; if (a < 0) u = 1 - u;
        const op = f * Math.sin(Math.PI * u) * (grow >= 0.999 ? 1 : 0);
        c.setAttribute('cx', x0 + (x1 - x0) * u); c.setAttribute('cy', y + (a > 0 ? -5 : 5));
        c.setAttribute('opacity', String(clamp(op))); c.setAttribute('r', String(6 + 2 * f));
      });
    },
  };
}

// opts.captions: optional replacement captions (array of 3 html strings) for the windows, in order
// opts.stats: optional replacement numbers (html); opts.width / gap / top / statSize / pad: layout;
// opts.chat: {stat, caption, note} adds a fourth window (free chat with a private city card)
export function makeDemoWindows(parent, opts = {}) {
  const W = opts.width ?? 572, GAP = opts.gap ?? 34, TOP = opts.top ?? 168, NW = opts.chat ? 4 : 3;
  const SS = opts.statSize ?? 104, PAD = opts.pad ?? 40, WH = opts.height ?? 744, SC = W / 572; // SC scales demo widths
  const row = el(parent, 'div', { position: 'absolute', left: `${(1920 - NW * W - (NW - 1) * GAP) / 2}px`, top: `${TOP}px`, width: `${NW * W + (NW - 1) * GAP}px`, height: `${WH + 16}px`, display: 'none' });
  const wins = [];
  const mk = (i, stat, cap, note) => {
    if (opts.captions && opts.captions[i] != null) cap = opts.captions[i];
    if (opts.stats && opts.stats[i] != null) stat = opts.stats[i];
    const c = el(row, 'div', {
      position: 'absolute', left: `${i * (W + GAP)}px`, top: '0px', width: `${W}px`, height: `${WH}px`,
      background: 'linear-gradient(180deg, rgba(22,18,44,0.86), rgba(10,8,22,0.86))', borderRadius: '24px',
      border: '1px solid rgba(245,243,255,0.14)', boxShadow: '0 30px 70px rgba(0,0,0,0.5)', overflow: 'hidden', transformOrigin: '50% 55%',
    });
    const demo = el(c, 'div', { position: 'absolute', left: '0px', top: '0px', width: `${W}px`, height: `${DEMO_H}px`, background: 'radial-gradient(120% 90% at 50% 45%, rgba(40,30,80,0.55), rgba(6,5,14,0.8))', borderBottom: '1px solid rgba(245,243,255,0.1)' });
    // faint floor grid in the demo, like the film's floor
    el(demo, 'div', { position: 'absolute', inset: '0', backgroundImage: 'linear-gradient(rgba(52,245,192,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(52,245,192,0.06) 1px, transparent 1px)', backgroundSize: '38px 38px', maskImage: 'linear-gradient(180deg, transparent 30%, #000 100%)', webkitMaskImage: 'linear-gradient(180deg, transparent 30%, #000 100%)' });
    const statEl = el(c, 'div', { position: 'absolute', left: `${PAD}px`, top: `${DEMO_H + 34 * SS / 104}px`, font: `700 ${SS}px ${FONT.display}`, lineHeight: '1', letterSpacing: '-0.035em', fontFeatureSettings: '"tnum" 1', color: TEAL, textShadow: '0 0 34px rgba(52,245,192,0.45)', whiteSpace: 'nowrap' }, stat);
    statEl.querySelectorAll('.sl').forEach((x) => Object.assign(x.style, { fontWeight: '300', margin: '0 0.14em', opacity: '0.7' }));
    const capEl = el(c, 'div', { position: 'absolute', left: `${PAD}px`, right: '24px', top: `${DEMO_H + 54 + SS}px`, font: `500 36px ${FONT.text}`, lineHeight: '1.22', color: 'rgba(245,243,255,0.92)', letterSpacing: '-0.01em' }, cap);
    if (note) el(capEl, 'div', { font: `500 30px ${FONT.text}`, color: 'rgba(245,243,255,0.5)', marginTop: '10px', whiteSpace: 'nowrap' }, note);
    const w = { c, demo, statEl, capEl, parts: {} };
    wins.push(w); return w;
  };
  const IY = 200, QX = Math.round(96 * Math.min(1, SC + 0.1)), EX = W - QX, M = Math.round(26 * SC); // icon row/positions inside the demo, edge margin
  // ---- 1. code lookup (Ember holds the table, Quill asks, the answer crosses as hidden states)
  {
    const w = mk(0, '17<span class=sl>/</span>20', 'lookups won, a game<br>it never trained on');
    const d = w.demo, P = w.parts;
    P.link = makeLink(d, QX + 46, EX - 46, IY, W);
    P.q = makeIcon(d, 'quill', QX, IY); P.e = makeIcon(d, 'ember', EX, IY);
    P.ask = makePill(d, 'code M2', 'quill', { x: M, y: 316, anchor: 'left' });
    P.r1 = makePill(d, 'K7 candle', 'ember', { x: W - M, y: 300, anchor: 'right' });
    P.r2 = makePill(d, 'M2 breeze', 'ember', { x: W - M, y: 372, anchor: 'right' });
    P.out = makePill(d, 'breeze', 'quill', { x: M, y: 70, anchor: 'left' });
    P.ok = makeCheck(d, SC < 0.9 ? M + 190 : M + 188, 70);
  }
  // ---- 2. turn structures never trained
  {
    const w = mk(1, '128<span class=sl>/</span>128', 'secrets passed in three<br>untrained formats');
    const d = w.demo, P = w.parts;
    P.label = el(d, 'div', { position: 'absolute', left: '0', width: `${W}px`, top: '36px', textAlign: 'center', font: `600 36px ${FONT.text}`, color: TEAL, letterSpacing: '-0.005em' });
    P.labels = ['dialogue', 'silent rounds', 'shared pen'].map((s) => el(P.label, 'span', { position: 'absolute', left: '0', width: `${W}px`, textAlign: 'center' }, s));
    P.link = makeLink(d, QX + 46, EX - 46, 160, W);
    P.q = makeIcon(d, 'quill', QX, 160); P.e = makeIcon(d, 'ember', EX, 160);
    P.b = [makeBubble(d, 'quill', { x: M, y: 262, w: Math.round(300 * SC) }), makeBubble(d, 'ember', { x: W - M, y: 322, w: Math.round(260 * SC) }), makeBubble(d, 'quill', { x: M, y: 382, w: Math.round(220 * SC) })];
    P.pen = el(d, 'div', { position: 'absolute', left: '0', width: `${W}px`, top: '300px', textAlign: 'center', font: `600 ${SC < 0.9 ? 32 : 38}px ${FONT.text}`, whiteSpace: 'nowrap' });
    P.words = ['the', 'boat', 'leaves', 'at', 'dawn'].map((s, k) => el(P.pen, 'span', { display: 'inline-block', margin: '0 0.14em', color: k % 2 ? '#fdba74' : '#c4b5fd' }, s));
    P.checks = [0, 1, 2].map(() => makeCheck(d, W - Math.round(58 * SC), 58, 54));
  }
  // ---- 3. shared reading (Quill reads the passage, Ember is asked about it)
  {
    const w = mk(2, '407<span class=sl>/</span>456', 'shared-reading episodes<br>won by the real partner');
    const d = w.demo, P = w.parts;
    P.link = makeLink(d, QX + 46, EX - 46, IY, W);
    P.q = makeIcon(d, 'quill', QX, IY); P.e = makeIcon(d, 'ember', EX, IY);
    P.text = makePill(d, '…the door was red…', 'quill', { x: M, y: 316, anchor: 'left' });
    P.ask = makePill(d, 'door colour?', 'ember', { x: W - M, y: 384, anchor: 'right' });
    P.out = makePill(d, 'red', 'ember', { x: W - M, y: 70, anchor: 'right' });
    P.ok = makeCheck(d, W - M - 124, 70);
  }

  // ---- 4. free chat (optional): Quill holds a private city card; Ember names the city in its first chat line
  if (opts.chat) {
    const w = mk(3, opts.chat.stat, opts.chat.caption, opts.chat.note);
    const d = w.demo, P = w.parts;
    P.link = makeLink(d, QX + 46, EX - 46, IY, W);
    P.q = makeIcon(d, 'quill', QX, IY); P.e = makeIcon(d, 'ember', EX, IY);
    // the private card: skyline icon + city, under Quill, inside a dashed "only Quill sees this" frame
    P.card = makePill(d, `<svg width="44" height="30" viewBox="0 0 44 30" style="vertical-align:-4px;margin-right:12px"><path d="M1 29 V17 h6 V9 h5 v8 h4 V3 h6 v26 M22 29 V13 h6 v5 h5 V7 h4 v22 M37 29 V19 h6 v10" fill="none" stroke="#ffd166" stroke-width="2.4" stroke-linejoin="round"/></svg>${opts.chat.city}`, 'quill', { x: M, y: 330, anchor: 'left' });
    P.card.root.style.borderColor = '#ffd166';
    P.qb = makeBubble(d, 'quill', { x: M, y: 70, w: Math.round(190 * SC) });
    P.eb = makePill(d, `${opts.chat.city}!`, 'ember', { x: W - M, y: 70, anchor: 'right' });
    P.ok = makeCheck(d, W - M - 190, 70);
  }

  return {
    update(t, hits, tOut) {
      const vis = t >= hits[0] - 0.01 && t < tOut + 0.6;
      row.style.display = vis ? 'block' : 'none'; if (!vis) return;
      wins.forEach((w, i) => {
        const t0 = hits[i];
        if (t < t0) { w.c.style.display = 'none'; return; }
        w.c.style.display = 'block';
        const s = spring(t, t0, 2.5, 0.5), o = clamp((t - t0) / 0.08);
        const out = prog(t, tOut + i * 0.05, tOut + i * 0.05 + 0.28);
        w.c.style.opacity = o * (1 - out);
        w.c.style.transform = `translateY(${(1 - s) * 80 - out * 40}px) scale(${0.8 + 0.2 * s - 0.04 * out})`;
        const bl = (1 - clamp((t - t0) / 0.14)) * 12 + out * 12;
        w.c.style.filter = bl > 0.05 ? `blur(${bl}px)` : 'none';
        const glow = Math.exp(-(t - t0) * 3.5);
        w.c.style.borderColor = `rgba(52,245,192,${0.14 + 0.6 * glow})`;
        w.c.style.boxShadow = `0 30px 70px rgba(0,0,0,0.5), 0 0 ${40 + 50 * glow}px rgba(52,245,192,${0.05 + 0.35 * glow})`;
        // number and caption rise in just after the window
        const ps = easeOutExpo(clamp((t - t0 - 0.12) / 0.5)), pc = easeOutExpo(clamp((t - t0 - 0.25) / 0.5));
        w.statEl.style.opacity = ps; w.statEl.style.transform = `translateY(${(1 - ps) * 24}px)`;
        w.capEl.style.opacity = pc; w.capEl.style.transform = `translateY(${(1 - pc) * 18}px)`;
        const P = w.parts;
        if (i === 0) {
          // lookup: table and question appear, hidden states cross Ember -> Quill, Quill writes "breeze"
          P.q.update(t, t0 + 0.1, prog(t, t0 + 2.3, t0 + 2.45)); P.e.update(t, t0 + 0.16, 0);
          P.r1.update(t, t0 + 0.3); P.r2.update(t, t0 + 0.4, 1, win2(t, t0 + 0.95, t0 + 2.1));
          P.ask.update(t, t0 + 0.55);
          P.link.update(t, prog(t, t0 + 0.6, t0 + 0.9), 0.25 * prog(t, t0 + 0.9, t0 + 1.1), win2(t, t0 + 0.9, t0 + 2.2) + 0.25 * prog(t, t0 + 2.2, t0 + 2.5));
          P.out.update(t, t0 + 2.0); P.ok.update(t, t0 + 2.3);
        } else if (i === 1) {
          // three formats in turn: dialogue, silent rounds, shared pen
          const F = [t0 + 0.3, t0 + 1.25, t0 + 2.2];
          P.q.update(t, t0 + 0.1, prog(t, t0 + 2.9, t0 + 3.05)); P.e.update(t, t0 + 0.16, prog(t, t0 + 2.9, t0 + 3.05));
          P.labels.forEach((L, k) => {
            const a = k < 2 ? win2(t, F[k], F[k + 1], 0.12) : prog(t, F[k], F[k] + 0.12);
            L.style.opacity = a; L.style.transform = `translateY(${(1 - a) * 10}px)`;
          });
          const dlg = win2(t, F[0], F[1], 0.12);
          P.b.forEach((b, k) => b.update(t, F[0] + 0.1 + k * 0.25, dlg));
          const silent = win2(t, F[1], F[2], 0.12);
          P.link.update(t, prog(t, t0 + 0.2, t0 + 0.45), 0.35 + 0.65 * silent, 0.35 + 0.65 * silent);
          P.checks.forEach((c, k) => c.update(t, F[k] + 0.62, k < 2 ? win2(t, F[k] + 0.6, F[k + 1], 0.1) : 1));
          P.words.forEach((wd, k) => { const p = easeOutExpo(clamp((t - F[2] - 0.08 - k * 0.14) / 0.35)); wd.style.opacity = p; wd.style.transform = `translateY(${(1 - p) * 16}px)`; });
        } else if (i === 3) {
          // free chat: Quill's private city card, Quill says something, dots cross, Ember's first line names the city
          P.q.update(t, t0 + 0.1, prog(t, t0 + 1.45, t0 + 1.6)); P.e.update(t, t0 + 0.16, prog(t, t0 + 1.45, t0 + 1.6));
          P.card.update(t, t0 + 0.25, 1, win2(t, t0 + 0.55, t0 + 1.25) * 0.8);
          P.qb.update(t, t0 + 0.4);
          P.link.update(t, prog(t, t0 + 0.3, t0 + 0.55), win2(t, t0 + 0.55, t0 + 1.3) + 0.25 * prog(t, t0 + 1.3, t0 + 1.6), 0.25 * prog(t, t0 + 0.55, t0 + 0.75));
          P.eb.update(t, t0 + 1.25); P.ok.update(t, t0 + 1.45);
        } else {
          // shared reading: Quill reads the passage, Ember is asked, hidden states cross Quill -> Ember, Ember writes "red"
          P.q.update(t, t0 + 0.1, 0); P.e.update(t, t0 + 0.16, prog(t, t0 + 1.75, t0 + 1.9));
          P.text.update(t, t0 + 0.3, 1, win2(t, t0 + 0.7, t0 + 1.5) * 0.8); P.ask.update(t, t0 + 0.45);
          P.link.update(t, prog(t, t0 + 0.5, t0 + 0.75), win2(t, t0 + 0.7, t0 + 1.6) + 0.25 * prog(t, t0 + 1.6, t0 + 1.9), 0.25 * prog(t, t0 + 0.75, t0 + 0.95));
          P.out.update(t, t0 + 1.5); P.ok.update(t, t0 + 1.75);
        }
      });
    },
  };
}
// 0 -> 1 -> 0 window with fades
function win2(t, a, b, f = 0.15) { return Math.min(clamp((t - a) / f), 1 - clamp((t - (b - f)) / f)); }
