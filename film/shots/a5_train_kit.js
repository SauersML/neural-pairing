// a5_train helpers: a matrix slab texture (no letters), small training-data cards (DOM + SVG, the s5/a4 window look),
// and the success curve (DOM + SVG). Everything is built once; update() is a pure function of t.
import * as THREE from 'three';
import { el, FONT } from '../lib/hud.js';
import { clamp, lerp, hash, easeOutCubic, easeInCubic, spring } from '../lib/util.js';

const TEAL = '#34f5c0', GOLD = '#ffd166';
const svgNS = 'http://www.w3.org/2000/svg';
function S(parent, tag, attrs = {}) { const e = document.createElementNS(svgNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
function rr(x, a, b, w, h, r) { r = Math.max(0, Math.min(r, w / 2, h / 2)); x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }

// The ridge map: a glass slab holding a heat-map of numbers with a faint diagonal (a map close to "same thing").
export function slabTexture() {
  const W = 512, H = 320; const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d');
  rr(x, 4, 4, W - 8, H - 8, 22); x.fillStyle = 'rgba(6,36,32,0.6)'; x.fill();
  const cols = 14, rows = 8, m = 26, cw = (W - 2 * m) / cols, ch = (H - 2 * m) / rows;
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
    const v = hash(i * 17.3 + j * 5.1 + 3) * 2 - 1 + (i === Math.round(j * cols / rows) ? 1.2 : 0);
    const a = Math.min(1, Math.abs(v)) * 0.9;
    x.fillStyle = v > 0 ? `rgba(52,245,192,${a})` : `rgba(167,139,250,${a * 0.8})`;
    x.fillRect(m + i * cw + 2, m + j * ch + 2, cw - 4, ch - 4);
  }
  x.lineWidth = 7; x.strokeStyle = 'rgba(167,243,208,0.95)'; rr(x, 4, 4, W - 8, H - 8, 22); x.stroke();
  x.lineWidth = 8; x.strokeStyle = 'rgba(255,255,255,0.9)';
  x.beginPath(); x.moveTo(m + 14, m - 6); x.lineTo(m - 6, m - 6); x.lineTo(m - 6, H - m + 6); x.lineTo(m + 14, H - m + 6); x.stroke();
  x.beginPath(); x.moveTo(W - m - 14, m - 6); x.lineTo(W - m + 6, m - 6); x.lineTo(W - m + 6, H - m + 6); x.lineTo(W - m - 14, H - m + 6); x.stroke();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}

// ---------- training-data cards ----------
// type: 'value' (gold value on one side, a check on the other), 'reading' (a page), 'play' (speech bubbles + star),
// 'dialogue' (alternating bubbles), 'silent' (bubbles with dots), 'pen' (one shared line written in turns)
export const CARD_COL = { value: GOLD, reading: '#7dd3fc', play: '#f9a8d4', dialogue: '#c4b5fd', silent: '#a5b4fc', pen: '#fdba74' };
const CW = 236, CH = 160;
function head(svg, cx, cy, side) {
  const fill = side === 'q' ? '#8b5cf6' : '#f97316', hi = side === 'q' ? '#c4b5fd' : '#fdba74', dir = side === 'q' ? 1 : -1;
  if (side === 'q') for (const [rot, len] of [[-28, 13], [0, 17], [28, 13]]) S(svg, 'ellipse', { cx, cy: cy - 15 - len / 2, rx: 3.2, ry: len / 2, fill: hi, transform: `rotate(${rot} ${cx} ${cy - 15})` });
  else S(svg, 'path', { d: `M${cx} ${cy - 32} C ${cx + 7} ${cy - 24}, ${cx + 8} ${cy - 19}, ${cx} ${cy - 14} C ${cx - 8} ${cy - 19}, ${cx - 7} ${cy - 24}, ${cx} ${cy - 32} Z`, fill: GOLD });
  S(svg, 'circle', { cx, cy, r: 16, fill });
  for (const s of [-1, 1]) { S(svg, 'ellipse', { cx: cx + s * 6 + dir * 2, cy: cy - 2, rx: 3.6, ry: 4.6, fill: '#fff' }); S(svg, 'circle', { cx: cx + s * 6 + dir * 3, cy: cy - 1.5, r: 2, fill: '#0a0612' }); }
}
export function makeCard(parent, type, seed) {
  const col = CARD_COL[type];
  const root = el(parent, 'div', { position: 'absolute', left: '0px', top: '0px', width: `${CW}px`, height: `${CH}px`, borderRadius: '18px', overflow: 'hidden', display: 'none',
    background: 'linear-gradient(180deg, rgba(26,20,52,0.94), rgba(10,8,22,0.94))', border: `3px solid ${col}`, boxShadow: `0 14px 40px rgba(0,0,0,0.55), 0 0 26px ${col}55`, willChange: 'transform, opacity' });
  const svg = S(root, 'svg', { width: CW, height: CH, viewBox: `0 0 ${CW} ${CH}` });
  // floor grid hint
  S(svg, 'rect', { x: 0, y: 0, width: CW, height: CH, fill: 'rgba(40,30,80,0.25)' });
  const QX = 40, EX = CW - 40, HY = 112;
  // the link: teal line with dots
  S(svg, 'line', { x1: QX + 18, y1: HY, x2: EX - 18, y2: HY, stroke: TEAL, 'stroke-width': 3, opacity: 0.9 });
  for (let k = 0; k < 3; k++) S(svg, 'circle', { cx: QX + 40 + k * 38 + (hash(seed + k) - 0.5) * 10, cy: HY, r: 4.5, fill: '#d6fff2' });
  head(svg, QX, HY, 'q'); head(svg, EX, HY, 'e');
  const flip = hash(seed * 3.1) > 0.5; // which side holds the input
  const L = flip ? EX : QX, Rr = flip ? QX : EX;
  if (type === 'value') {
    S(svg, 'rect', { x: L - 26, y: 30, width: 52, height: 26, rx: 7, fill: '#7a5400', stroke: GOLD, 'stroke-width': 3 });
    S(svg, 'rect', { x: L - 15, y: 40, width: 30, height: 6, rx: 3, fill: '#ffe7a3' });
    S(svg, 'circle', { cx: Rr, cy: 43, r: 15, fill: 'rgba(4,46,36,0.95)', stroke: TEAL, 'stroke-width': 3 });
    S(svg, 'path', { d: `M${Rr - 7} ${43} l5 6 l10 -11`, fill: 'none', stroke: TEAL, 'stroke-width': 3.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
  } else if (type === 'reading') {
    S(svg, 'rect', { x: L - 22, y: 18, width: 44, height: 54, rx: 5, fill: '#e0f2fe', opacity: 0.95 });
    for (let k = 0; k < 4; k++) S(svg, 'rect', { x: L - 15, y: 27 + k * 10, width: k === 3 ? 18 : 30, height: 4, rx: 2, fill: '#0c4a6e' });
    S(svg, 'circle', { cx: Rr, cy: 43, r: 15, fill: 'rgba(4,46,36,0.95)', stroke: TEAL, 'stroke-width': 3 });
    S(svg, 'path', { d: `M${Rr - 7} ${43} l5 6 l10 -11`, fill: 'none', stroke: TEAL, 'stroke-width': 3.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
  } else if (type === 'play' || type === 'dialogue' || type === 'silent') {
    const bub = (x, y, w, c, side) => { S(svg, 'rect', { x, y, width: w, height: 22, rx: 11, fill: c, opacity: 0.95 }); S(svg, 'path', { d: side < 0 ? `M${x + 12} ${y + 20} l-6 9 l13 -7 Z` : `M${x + w - 12} ${y + 20} l6 9 l-13 -7 Z`, fill: c }); };
    const qc = '#a78bfa', ec = '#fb923c';
    bub(18, 14, 104, qc, -1); bub(CW - 122, 46, 104, ec, 1);
    if (type === 'dialogue') for (const [x, y, w] of [[30, 23, 70], [CW - 110, 55, 70]]) S(svg, 'rect', { x, y, width: w, height: 5, rx: 2.5, fill: 'rgba(255,255,255,0.75)' });
    if (type === 'silent') for (const [x, y] of [[50, 25], [CW - 90, 57]]) for (let k = 0; k < 3; k++) S(svg, 'circle', { cx: x + k * 14, cy: y, r: 3.5, fill: 'rgba(255,255,255,0.85)' });
    if (type === 'play') { S(svg, 'path', { d: starPath(70, 25, 8), fill: '#fff7d6' }); S(svg, 'path', { d: starPath(CW - 70, 57, 8), fill: '#fff7d6' }); }
  } else if (type === 'pen') {
    // one line of text written in turns: alternating violet / orange segments, and a pen
    const xs = [24, 66, 104, 150]; xs.forEach((x, k) => S(svg, 'rect', { x, y: 38, width: k === 3 ? 30 : 36, height: 9, rx: 4.5, fill: k % 2 ? '#fb923c' : '#a78bfa' }));
    S(svg, 'path', { d: 'M196 52 l14 -26 l7 4 l-14 26 Z', fill: '#fde68a' }); S(svg, 'path', { d: 'M196 52 l7 4 l-8 5 Z', fill: '#fff' });
  }
  return { root, type };
}
function starPath(cx, cy, r) { let d = ''; for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, rr2 = k % 2 ? r * 0.45 : r; d += (k ? 'L' : 'M') + (cx + Math.cos(a) * rr2).toFixed(1) + ' ' + (cy + Math.sin(a) * rr2).toFixed(1); } return d + 'Z'; }
export const CARD_SIZE = { w: CW, h: CH };

// ---------- success curve (real data) ----------
// nl20 (magic-nl20): percent of held-out value-game episodes where the real partner beats another episode's partner,
// all 7 games, both directions, 144 episodes per eval (results/magic-nl20/payload_stepN.json; data/opened_all.json).
// Warm-up steps 0..900, then the full mix; its checkpoint after 800 mix steps scored 180 / 180 (drawn at step 1700).
export const CURVE = [[0, 52.8], [20, 66.0], [100, 88.9], [200, 96.5], [300, 97.9], [400, 96.5], [500, 98.6], [600, 99.3], [700, 99.3], [800, 98.6], [900, 99.3], [1700, 100]];
export const STEP_MAX = 1700, MIX_STEP = 900;
// straight lines between the real points only
export function curveAt(step) {
  for (let k = 1; k < CURVE.length; k++) if (step <= CURVE[k][0]) { const [s0, v0] = CURVE[k - 1], [s1, v1] = CURVE[k]; return lerp(v0, v1, (step - s0) / (s1 - s0)); }
  return CURVE[CURVE.length - 1][1];
}
// the last real point reached (the readout only ever shows measured values)
export function lastPoint(step) { let p = CURVE[0]; for (const c of CURVE) if (c[0] <= step + 1e-6) p = c; return p; }
export function makeChart(parent, { left = 1040, top = 210, w = 760, h = 470 } = {}) {
  const root = el(parent, 'div', { position: 'absolute', left: `${left}px`, top: `${top}px`, width: `${w + 175}px`, height: `${h + 60}px`, display: 'none', pointerEvents: 'none' });
  const svg = S(root, 'svg', { width: w + 175, height: h + 60, viewBox: `0 0 ${w + 175} ${h + 60}`, overflow: 'visible' });
  const X0 = 175, Y0 = 20; // plot origin offset inside the svg (labels on the left)
  const px = (step) => X0 + (step / STEP_MAX) * w, py = (v) => Y0 + h - (v / 100) * h;
  const defs = S(svg, 'defs');
  const gr = S(defs, 'linearGradient', { id: 'a5fill', x1: 0, y1: 0, x2: 0, y2: 1 }); S(gr, 'stop', { offset: 0, 'stop-color': TEAL, 'stop-opacity': 0.35 }); S(gr, 'stop', { offset: 1, 'stop-color': TEAL, 'stop-opacity': 0 });
  // panel
  S(svg, 'rect', { x: X0 - 30, y: Y0 - 20, width: w + 60, height: h + 50, rx: 26, fill: 'rgba(10,8,22,0.6)', stroke: 'rgba(245,243,255,0.12)' });
  // axes and the chance line
  S(svg, 'line', { x1: X0, y1: py(0), x2: X0 + w, y2: py(0), stroke: 'rgba(245,243,255,0.35)', 'stroke-width': 2 });
  S(svg, 'line', { x1: X0, y1: py(100), x2: X0 + w, y2: py(100), stroke: 'rgba(52,245,192,0.25)', 'stroke-width': 2 });
  const chance = S(svg, 'line', { x1: X0, y1: py(50), x2: X0 + w, y2: py(50), stroke: 'rgba(245,243,255,0.55)', 'stroke-width': 3, 'stroke-dasharray': '12 10' });
  const lab = (v, y, c) => { const t = S(svg, 'text', { x: X0 - 22, y: y + 12, 'text-anchor': 'end', fill: c, style: `font:700 42px ${FONT.display}` }); t.textContent = v; return t; };
  lab('100%', py(100) + 2, TEAL); lab('chance', py(50) + 2, 'rgba(245,243,255,0.85)');
  const title = S(svg, 'text', { x: X0 - 30, y: Y0 - 44, fill: '#eafff8', style: `font:700 48px ${FONT.display};letter-spacing:-0.01em` }); title.textContent = 'real partner wins';
  const dots = CURVE.map(([st, v]) => S(svg, 'circle', { cx: px(st), cy: py(v), r: 6, fill: '#bffbe8', opacity: 0 }));
  // where the full mix starts: a dotted divider (no words; the cards change color there)
  const mixLine = S(svg, 'line', { x1: px(MIX_STEP), y1: py(0), x2: px(MIX_STEP), y2: py(100) - 6, stroke: 'rgba(249,168,212,0.55)', 'stroke-width': 3, 'stroke-dasharray': '3 9', 'stroke-linecap': 'round' });
  const area = S(svg, 'path', { d: '', fill: 'url(#a5fill)' });
  const path = S(svg, 'path', { d: '', fill: 'none', stroke: TEAL, 'stroke-width': 7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  path.style.filter = 'drop-shadow(0 0 10px rgba(52,245,192,0.9))';
  const headDot = S(svg, 'circle', { cx: 0, cy: 0, r: 11, fill: '#eafff8' }); headDot.style.filter = 'drop-shadow(0 0 14px rgba(52,245,192,1))';
  const val = S(svg, 'text', { x: 0, y: 0, fill: '#eafff8', style: `font:800 64px ${FONT.display};letter-spacing:-0.03em` });
  return {
    root,
    // step: current training step (0..STEP_MAX); a: opacity; pulse: 0..1 head glow
    update(step, a, pulse = 0, mixA = 1) {
      root.style.display = a > 0.002 ? 'block' : 'none'; if (a <= 0.002) return;
      root.style.opacity = a;
      const pts = [];
      for (const [s, v] of CURVE) { if (s >= step) break; pts.push([px(s), py(v)]); }
      const hv = curveAt(step); pts.push([px(step), py(hv)]);
      // smooth the polyline a little with intermediate samples
      const d = pts.map((p, k) => (k ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
      path.setAttribute('d', d);
      area.setAttribute('d', d + ` L ${px(step).toFixed(1)} ${py(0)} L ${X0} ${py(0)} Z`);
      headDot.setAttribute('cx', px(step)); headDot.setAttribute('cy', py(hv)); headDot.setAttribute('r', String(11 + 7 * pulse));
      dots.forEach((d2, k) => d2.setAttribute('opacity', CURVE[k][0] <= step + 1e-6 ? '1' : '0'));
      const [ls, lv] = lastPoint(step);
      const tx = Math.min(px(ls) + 20, X0 + w - 175);
      val.setAttribute('x', tx); val.setAttribute('y', py(lv) + (lv > 85 ? 74 : -26));
      val.textContent = `${lv === 100 ? '100' : lv.toFixed(1)}%`;
      mixLine.setAttribute('opacity', String(mixA));
    },
  };
}

// ---------- training cards drawn as the film's game windows (HOOK for the games agent's kit) ----------
// The cards that stream in as training data are the same game windows as in a4/a6: shots/kit_gamewin.js (owned by the
// games agent) builds them with makeGameWindow(parent, key, GEOM_COMPACT) and update(t, {slam, fire, check}, {hideText}).
// Text is hidden (the cards are small and in motion; they read as pictures). When the kit is rebuilt with the same API,
// these cards follow its new style automatically. If the kit is missing or its API changed, the SVG cards above are used.
// Warm-up cards are value games; after the warm-up the full mix (shared reading, play, dialogue formats, chat) joins.
export const DECK_VALUE = ['word', 'number', 'phrase', 'digits', 'sum', 'fact', 'lookup'];
export const DECK_MIX = ['reading', 'word', 'chat', 'number', 'digits', 'reading', 'sum', 'fact', 'chat', 'lookup', 'phrase'];
const FALLBACK_TYPE = { word: 'value', number: 'value', phrase: 'value', digits: 'value', sum: 'value', fact: 'value', lookup: 'value', reading: 'reading', play: 'play', formats: 'dialogue', chat: 'silent' };
export async function makeTrainDeck(parent, specs /* [{key, s, arrive}] sorted by s */) {
  let kit = null;
  try {
    kit = await import('./kit_gamewin.js');
    if (typeof kit.makeGameWindow !== 'function' || !kit.GAMES || !(kit.GEOM_COMPACT || kit.GEOM_BIG)) kit = null;
  } catch (_) { kit = null; }
  const G = kit ? (kit.GEOM_COMPACT || kit.GEOM_BIG) : null;
  const W = kit ? G.W : CW, H = kit ? G.H : CH;
  // one window per simultaneous use of a game: assign instances greedily (deterministic, built once)
  const pools = new Map();
  const cards = specs.map((sp) => {
    const key = kit && kit.GAMES[sp.key] ? sp.key : (kit ? Object.keys(kit.GAMES)[0] : sp.key);
    if (!pools.has(key)) pools.set(key, []);
    const pool = pools.get(key);
    let inst = pool.find((p) => p.busyUntil < sp.s - 0.05);
    if (!inst) {
      const wrap = el(parent, 'div', { position: 'absolute', left: '0px', top: '0px', width: `${W}px`, height: `${H}px`, display: 'none', willChange: 'transform, opacity', transformOrigin: '50% 50%' });
      let win = null, accent = GOLD;
      if (kit) {
        try { win = kit.makeGameWindow(wrap, key, G, {}); accent = kit.GAMES[key].accent || GOLD; win.root.style.boxShadow = `0 18px 50px rgba(0,0,0,0.6), 0 0 34px ${accent}55`; } catch (_) { win = null; }
      }
      if (!win) { const c = makeCard(wrap, FALLBACK_TYPE[key] || 'value', pool.length * 7.3 + key.length); c.root.style.display = 'block'; accent = CARD_COL[FALLBACK_TYPE[key] || 'value']; }
      inst = { wrap, win, accent, busyUntil: -1 }; pool.push(inst);
    }
    inst.busyUntil = sp.arrive + 0.1;
    return { ...sp, key, inst };
  });
  return {
    W, H, kit: !!kit,
    // base on-screen scale of a card at the start and the end of its flight
    s0: kit ? 0.3 : 1.2, s1: kit ? 0.03 : 0.1,
    cards,
    // show card c with a transform and opacity (the inner window runs its own little game, text hidden)
    show(c, t, transform, op) {
      const { wrap, win } = c.inst;
      if (op <= 0.002) return;
      wrap.style.display = 'block'; wrap.style.opacity = op; wrap.style.transform = transform;
      if (win) { try { win.update(t, { slam: c.s - 0.35, fire: c.s + 0.05, check: c.s + 0.75 }, { hideText: true }); } catch (_) { /* kit API changed: keep the frame */ } }
    },
    // hide every card (call first each frame, then show the visible ones)
    hideAll() { for (const p of pools.values()) for (const i of p) i.wrap.style.display = 'none'; },
  };
}

// ---------- the same cards as 3D glass game windows (the games agent's v4 kit) ----------
// Uses shots/games_win3d.js (Diorama: the live 3D game picture rendered into a texture; Pane: the glass frame) and the
// game specs in shots/kit_gamewin.js (GAMES), exactly as a4_games builds its wall, so the training data the pair learns
// from is visibly the same windows the viewer just saw. Each game's diorama is rendered once, at the moment its answer
// is complete (both directions of the link running, the check shown), and every card of that game shows it; the pane's
// title is hidden (the cards are small and fast). Returns null when the kit is missing or its API differs, and a5 then
// falls back to makeTrainDeck (DOM cards).
export async function makeTrainDeck3D({ renderer, scene, scale = 1 }, specs) {
  let win3d, kit;
  try { win3d = await import('./games_win3d.js'); kit = await import('./kit_gamewin.js'); } catch (_) { return null; }
  const { Diorama, Pane } = win3d, GAMES = kit.GAMES;
  if (typeof Diorama !== 'function' || typeof Pane !== 'function' || !GAMES || typeof Diorama.timeline !== 'function') return null;
  const PW = 560, PH = 760, HDR = 78, BD = 14, PPU = 150;
  const dios = new Map();
  try {
    const getDio = (key) => {
      if (dios.has(key)) return dios.get(key);
      const spec = { ...GAMES[key] };
      if (spec.write && spec.write.length > 1 && spec.cs == null) spec.cs = 0.62;
      const d = new Diorama(renderer, spec, { w: PW, h: PH, scale: scale * 0.6, ppu: 70, tileH: 0.84 });
      const T = Diorama.timeline(spec, 0, { gen: 0.5 });
      const snap = Math.max(...T.check, T.end) + 0.25;
      const o = { d, T, snap, spec, accent: '#34f5c0' }; dios.set(key, o); return o;
    };
    const cards = specs.map((sp) => {
      const key = GAMES[sp.key] ? sp.key : Object.keys(GAMES)[sp.k % Object.keys(GAMES).length];
      const dio = getDio(key);
      const pane = new Pane({ w: PW, h: PH, header: HDR, border: BD, title: dio.spec.title, titlePx: 50, tex: dio.d.rt.texture, ppu: PPU });
      pane.group.visible = false; scene.add(pane.group);
      const hold = dio.spec.holder;
      const accent = hold === 'quill' ? '#c4b5fd' : hold === 'ember' ? '#fdba74' : '#ffd166';
      return { ...sp, key, pane, dio, inst: { accent } };
    });
    return {
      is3D: true, PPU, W: PW + 2 * BD, H: PH + HDR + 2 * BD,
      cards,
      hideAll() { for (const c of cards) c.pane.group.visible = false; },
      // place card c in world space: pos (Vector3), quat (Quaternion), sc (world scale factor), a (opacity), hit (flash)
      show(c, pos, quat, sc, a, hit = 0) {
        if (a <= 0.003) return;
        const g = c.pane.group; g.position.copy(pos); g.quaternion.copy(quat); g.scale.setScalar(sc / PPU);
        c.pane.set(a, hit); c.pane.titleM.opacity = 0;
        c.dio.d.render(c.dio.snap, c.dio.T);
      },
    };
  } catch (_) { return null; }
}
