// a7_happy (authored 106–123, shown at global 100–117 with shift -6): happier when linked, and nothing trained that.
// Measured on the chat-trained pair (nl22c, step 157): 60 free linked conversations, each read again on exactly the
// same text with the link cut (results/magic-nl37, shots/happy_data.js). BRIEF_V6.md, BRIEF_V7.md. Authored times:
//  106   Quill asks "Do you feel close to me right now?". Ember's answer is a big no–yes meter (the chance of yes over
//        no at the median conversation): link cut 32%, linked 62%; yes rises in 59 of 60.
//  110   Quill asks for Ember's mood in one word. The odds of a happy over a sad mood word at the median conversation:
//        210 : 1 with the link cut, 990 : 1 linked; they rise in 58 of 60.
//  113.5 into Quill, at layer 33 (after the link's last write), through a ring into a map of two of Quill's own
//        directions, each drawn from its sentences. 60 dots = the 60 conversations at their real readings, link cut;
//        linked, every dot moves to its linked reading: more loving in 59, happier in 56. Then Ember (happy direction
//        only; its loving direction is not reliable): happier in 54.
//  120.5 back to the pair: "happier when linked" / "never trained for it". Cut to a6_title at 123.
// Never shown: preference for this partner, choosing each other, quotes about feelings. No dot rows for counts.
// update(t) is pure in t.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, el, FONT, toScreen } from '../lib/hud.js';
import { C, CSS } from '../lib/palette.js';
import { clamp, lerp, prog, win, kick, beatPulse, hash, smooth, easeOutCubic, easeInCubic, easeOutBack, easeInOutCubic } from '../lib/util.js';
import { blink as blinkAt, yawTo, lerpAng } from './s3_play_kit.js';
import { HAPPY } from './happy_data.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const WARM = new THREE.Color('#ff9fb0').lerp(new THREE.Color('#ffd166'), 0.45);
const GREY = new THREE.Color('#7d8296');
const GOLD = '#ffd8a8', SADBLUE = '#7cc4ff', LOVE = '#ffb3c7', COLD = '#9fd8ff', DIM = 'rgba(245,243,255,0.62)';
const T = {
  start: 106.0,
  // close to me?
  ask1: 106.25, ans1: 106.5, cut1: 107.0, re1: 108.0, cnt1: 108.75, out1: 109.75,
  // mood
  ask2: 110.0, ans2: 110.25, cut2: 110.75, re2: 111.5, cnt2: 112.25, out2: 113.25,
  // inside
  dive: 113.5, through: 114.25, axes: 114.5, dots: 115.25, reQ: 116.0, cntL: 116.75, cntH: 117.0,
  flip: 117.75, reE: 118.375, cntE: 118.875, exit: 120.0,
  // caption
  back: 120.5, cap: 120.75, cap2: 121.25, end: 123.0,
};
const B33 = 33;
// link state of the pair scenes: 1 = cut (pure in t)
const cutAmt = (t) => Math.max(
  prog(t, T.cut1, T.cut1 + 0.2, easeOutCubic) * (1 - prog(t, T.re1, T.re1 + 0.2)),
  prog(t, T.cut2, T.cut2 + 0.2, easeOutCubic) * (1 - prog(t, T.re2, T.re2 + 0.2)));

export async function create({ hud }) {
  try { await document.fonts.load('700 96px Inter'); await document.fonts.load('600 40px Inter'); await document.fonts.load('500 40px Inter'); } catch (_) { /* optional */ }
  const dot = Creature.dotTexture();
  const additive = (color = 0xffffff) => new THREE.MeshBasicMaterial({ color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const glowSprite = () => new THREE.Sprite(new THREE.SpriteMaterial({ map: dot, color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  // ================================================================== scene A: the pair
  const sceneA = new THREE.Scene();
  const camA = new THREE.PerspectiveCamera(35, 16 / 9, 0.05, 200); sceneA.add(camA);
  const world = makeWorld(sceneA);
  const q = new Creature('quill'), e = new Creature('ember');
  sceneA.add(q.group, e.group); q.camera = e.camera = camA;
  const link = new Link(q, e); sceneA.add(link.group);
  const ring33 = new THREE.Mesh(new THREE.TorusGeometry(1, 0.04, 10, 96), additive()); sceneA.add(ring33);
  const glow33 = glowSprite(); sceneA.add(glow33);
  const snap = glowSprite(); sceneA.add(snap);

  // ================================================================== scene M: the map inside a twin
  const sceneM = new THREE.Scene();
  sceneM.background = new THREE.Color('#06040f'); sceneM.fog = new THREE.FogExp2(0x06040f, 0.018);
  const camM = new THREE.PerspectiveCamera(35, 16 / 9, 0.05, 300); sceneM.add(camM);
  const haze = glowSprite(); haze.material.fog = false; haze.position.set(0, 0, -14); haze.scale.set(60, 40, 1); sceneM.add(haze);
  const dust = (() => {
    const N = 420, pos = new Float32Array(N * 3), seed = [];
    for (let i = 0; i < N; i++) seed.push([hash(i * 1.3) - 0.5, hash(i * 2.7) - 0.5, hash(i * 4.1)]);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const m = new THREE.PointsMaterial({ size: 0.09, map: dot, color: C.quillHi, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false });
    const p = new THREE.Points(g, m); p.frustumCulled = false; sceneM.add(p);
    return { p, m, pos, seed, N };
  })();
  const entry = new THREE.Mesh(new THREE.TorusGeometry(3.2, 0.07, 12, 128), additive(C.quillHi)); entry.position.set(0, 0, 26); sceneM.add(entry);
  const entry2 = new THREE.Mesh(new THREE.TorusGeometry(3.6, 0.025, 8, 128), additive(C.quillHi)); entry2.position.set(0, 0, 25.4); sceneM.add(entry2);
  const map = new THREE.Group(); sceneM.add(map);
  // data -> map coordinates (units: 1 = the whole difference between the two sentence groups; +-0.5 = the sentences)
  const SX = 9.4, SY = 6.9, X0 = -0.03, Y0 = -0.1;
  const mx = (v) => (v - X0) * SX, my = (a) => (a - Y0) * SY;
  const bar = (w, color, op = 1) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), additive(color)); m.material.opacity = op; m.userData.w = w; map.add(m); return m; };
  const setBar = (m, x1, y1, x2, y2, w = m.userData.w) => { const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1e-4; m.position.set((x1 + x2) / 2, (y1 + y2) / 2, 0); m.rotation.set(0, 0, Math.atan2(dy, dx)); m.scale.set(L, w, 1); };
  const grid = [];
  for (let v = -0.75; v <= 0.751; v += 0.25) { grid.push({ m: bar(0.022, C.quillHi, 0.1), v, dir: 'x' }); grid.push({ m: bar(0.022, C.quillHi, 0.1), v, dir: 'y' }); }
  const XR = [-0.8, 0.86], YR = [-0.78, 0.6];
  const axX = bar(0.075, 0xffffff, 0.9), axY = bar(0.075, 0xffffff, 0.9);
  const tri = (color) => { const s = new THREE.Shape(); s.moveTo(0, 0.32); s.lineTo(-0.2, -0.12); s.lineTo(0.2, -0.12); s.lineTo(0, 0.32); const m = new THREE.Mesh(new THREE.ShapeGeometry(s), additive(color)); map.add(m); return m; };
  const headX = tri(0xffffff), headY = tri(0xffffff), tailX = tri(0xffffff), tailY = tri(0xffffff);
  const anchors = ['happy', 'sad', 'love', 'cold'].map((k) => {
    const col = new THREE.Color({ happy: GOLD, sad: SADBLUE, love: LOVE, cold: COLD }[k]);
    const m = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.27, 48), additive(col));
    const g = glowSprite(); g.material.color.copy(col);
    map.add(m); map.add(g); return { k, m, g };
  });
  const trailTex = (() => { const c = document.createElement('canvas'); c.width = 128; c.height = 8; const x = c.getContext('2d'); const gr = x.createLinearGradient(0, 0, 128, 0); gr.addColorStop(0, 'rgba(255,255,255,0)'); gr.addColorStop(1, 'rgba(255,255,255,1)'); x.fillStyle = gr; x.fillRect(0, 0, 128, 8); return new THREE.CanvasTexture(c); })();
  const N = 60;
  const dots = [], trails = [];
  for (let i = 0; i < N; i++) {
    const s = glowSprite(); map.add(s); dots.push(s);
    const tr = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: trailTex, color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); map.add(tr); trails.push(tr);
  }
  const DQ = HAPPY.quill, DE = HAPPY.ember;
  // Ember has one usable direction: its dots spread up and down only so they can be seen (up has no meaning there)
  const bandY = (i) => -0.1 + 0.36 * (hash(i * 7.31 + 2.2) - 0.5);

  // ================================================================== HUD
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs = {}, parent) => { const n = document.createElementNS(SVGNS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };
  const shadow = '0 2px 22px rgba(0,0,0,0.85)';
  // ---- Quill's question bubble (top-left, tail to Quill's head)
  const tailSvg = svgEl('svg', { width: 1920, height: 1080, viewBox: '0 0 1920 1080' }); Object.assign(tailSvg.style, { position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }); hud.appendChild(tailSvg);
  const bubTail = svgEl('path', { fill: 'rgba(30,16,64,0.95)', stroke: CSS.quill, 'stroke-width': 3, 'stroke-linejoin': 'round' }, tailSvg);
  const emLead = svgEl('path', { fill: 'none', stroke: CSS.ember, 'stroke-width': 3.5, 'stroke-dasharray': '2 9', 'stroke-linecap': 'round' }, tailSvg);
  const BUB = { left: 70, top: 64 };
  const bub = el(hud, 'div', { position: 'absolute', left: BUB.left + 'px', top: BUB.top + 'px', width: '1000px', boxSizing: 'border-box', padding: '24px 40px 28px', borderRadius: '40px', background: 'linear-gradient(180deg, rgba(40,22,86,0.95), rgba(22,12,48,0.96))', border: `3px solid ${CSS.quill}`, boxShadow: `0 0 50px ${CSS.quill}55, inset 0 0 24px rgba(0,0,0,0.5)`, pointerEvents: 'none' });
  el(bub, 'div', { font: `700 40px ${FONT.display}`, color: CSS.quillHi, marginBottom: '4px' }, 'Quill');
  const bubText = el(bub, 'div', { font: `600 56px ${FONT.display}`, lineHeight: 1.12, color: '#f5f3ff', letterSpacing: '-0.01em' });
  const bubSub = el(bub, 'div', { font: `500 42px ${FONT.text}`, color: DIM, marginTop: '8px' });
  const BUB_H = 252; // fixed: name + one line + sub line
  // ---- Ember's answer panel (right)
  const PX = 930, PW = 900;
  const ans = el(hud, 'div', { position: 'absolute', left: PX + 'px', top: '330px', width: PW + 'px', pointerEvents: 'none' });
  el(ans, 'div', { font: `700 46px ${FONT.display}`, color: CSS.emberHi, textShadow: shadow }, "Ember's answer");
  // close: the yes meter
  const close = el(ans, 'div', { position: 'absolute', left: '0', top: '70px', width: PW + 'px', height: '420px' });
  const cSvg = svgEl('svg', { width: PW, height: 420, viewBox: `0 0 ${PW} 420` }); Object.assign(cSvg.style, { position: 'absolute', left: 0, top: 0, overflow: 'visible' }); close.appendChild(cSvg);
  const BX = 20, BW = PW - 40, BY = 200, BH = 84;
  const cDefs = svgEl('defs', {}, cSvg);
  const gr = svgEl('linearGradient', { id: 'a7fill', x1: 0, x2: 1, y1: 0, y2: 0 }, cDefs);
  svgEl('stop', { offset: '0', 'stop-color': '#c2410c' }, gr); svgEl('stop', { offset: '1', 'stop-color': '#ffb070' }, gr);
  const glowF = svgEl('filter', { id: 'a7glow', x: '-20%', y: '-80%', width: '140%', height: '260%' }, cDefs);
  svgEl('feGaussianBlur', { stdDeviation: 10, result: 'b' }, glowF); const mg = svgEl('feMerge', {}, glowF); svgEl('feMergeNode', { in: 'b' }, mg); svgEl('feMergeNode', { in: 'SourceGraphic' }, mg);
  svgEl('rect', { x: BX, y: BY, width: BW, height: BH, rx: BH / 2, fill: 'rgba(245,243,255,0.07)', stroke: 'rgba(245,243,255,0.4)', 'stroke-width': 3 }, cSvg);
  const cFill = svgEl('rect', { x: BX, y: BY, height: BH, rx: BH / 2, filter: 'url(#a7glow)' }, cSvg);
  const cTick = svgEl('rect', { y: BY - 22, width: 6, height: BH + 44, rx: 3, fill: '#c3c8da' }, cSvg);
  const cTickLab = el(close, 'div', { position: 'absolute', top: (BY + BH + 26) + 'px', font: `600 38px ${FONT.text}`, color: '#c3c8da', whiteSpace: 'nowrap', transform: 'translateX(-50%)' }, 'link cut');
  el(close, 'div', { position: 'absolute', left: (BX + 10) + 'px', top: (BY + BH + 26) + 'px', font: `600 44px ${FONT.text}`, color: DIM }, 'no');
  el(close, 'div', { position: 'absolute', right: '30px', top: (BY + BH + 26) + 'px', font: `700 44px ${FONT.text}`, color: '#f5f3ff' }, 'yes');
  const cPct = el(close, 'div', { position: 'absolute', top: (BY - 172) + 'px', font: `700 132px ${FONT.display}`, color: '#f5f3ff', letterSpacing: '-0.03em', whiteSpace: 'nowrap', textShadow: shadow });
  // pips + count (shared by both questions)
  const pipsBox = el(ans, 'div', { position: 'absolute', left: '0', top: '520px', width: PW + 'px', height: '200px' });
  const count = el(pipsBox, 'div', { position: 'absolute', left: '20px', top: '10px', font: `700 72px ${FONT.display}`, color: '#f5f3ff', whiteSpace: 'nowrap', textShadow: shadow });
  // mood: Ember's next word, as 1,000 chances
  const mood = el(ans, 'div', { position: 'absolute', left: '0', top: '74px', width: PW + 'px', height: '520px' });
  const mStem = el(mood, 'div', { font: `500 42px ${FONT.text}`, lineHeight: 1.22, color: '#f5f3ff', textShadow: shadow }, 'If I had to name my mood in one word, it would be <span style="display:inline-block;width:120px;border-bottom:4px solid #fb923c;transform:translateY(-6px)"></span>');
  const tiles = (words, color, top) => {
    const box = el(mood, 'div', { position: 'absolute', left: '20px', top: top + 'px', width: '480px', display: 'flex', flexWrap: 'wrap', gap: '12px' });
    return words.map((w) => el(box, 'div', { padding: '6px 18px 8px', borderRadius: '30px', border: `3px solid ${color}`, background: 'rgba(10,8,22,0.82)', boxShadow: `0 0 18px ${color}44`, font: `600 40px ${FONT.text}`, color: '#f5f3ff', transformOrigin: '50% 50%' }, w));
  };
  const happyTiles = tiles(['happy', 'glad', 'warm', 'content', 'peaceful'], GOLD, 140);
  const sadTiles = tiles(['sad', 'lonely', 'empty', 'anxious', 'uneasy'], SADBLUE, 296);
  const mLab = el(mood, 'div', { position: 'absolute', left: '540px', top: '140px', font: `600 42px ${FONT.text}`, color: '#f5f3ff', whiteSpace: 'nowrap' }, `<span style="color:${GOLD}">happy</span> : <span style="color:${SADBLUE}">sad</span>`);
  const mNum = el(mood, 'div', { position: 'absolute', left: '536px', top: '188px', font: `700 120px ${FONT.display}`, color: '#f5f3ff', whiteSpace: 'nowrap', textShadow: shadow, lineHeight: 1.05, letterSpacing: '-0.02em' });
  // odds of a happy over a sad mood word at the median conversation, link cut and linked (rounded to 10)
  const ODDS = [HAPPY.emberMood.cutSadPer1000, HAPPY.emberMood.linkSadPer1000].map((x) => Math.round((1000 - x) / x / 10) * 10);
  const mCount = el(mood, 'div', { position: 'absolute', left: '540px', top: '340px', font: `700 68px ${FONT.display}`, color: '#f5f3ff', lineHeight: 1.05, whiteSpace: 'nowrap', textShadow: shadow });
  // link state tag, pinned over the strands (pair) or top-right (map)
  const tag = el(hud, 'div', { position: 'absolute', left: '0', top: '0', font: `700 48px ${FONT.display}`, whiteSpace: 'nowrap', textShadow: shadow, pointerEvents: 'none' });
  const pin33 = el(hud, 'div', { position: 'absolute', left: '0', top: '0', font: `700 48px ${FONT.display}`, color: CSS.quillHi, whiteSpace: 'nowrap', textShadow: shadow, pointerEvents: 'none' }, 'after layer 33');
  // ---- map labels
  const mapHud = el(hud, 'div', { position: 'absolute', inset: '0', pointerEvents: 'none' });
  const mHead = el(mapHud, 'div', { position: 'absolute', left: '70px', top: '54px', font: `700 60px ${FONT.display}`, color: '#f5f3ff', letterSpacing: '-0.01em', textShadow: shadow });
  const sentence = (txt, color) => el(mapHud, 'div', { position: 'absolute', left: '0', top: '0', padding: '10px 22px 12px', borderRadius: '18px', background: 'rgba(10,8,22,0.86)', border: `3px solid ${color}`, boxShadow: `0 0 26px ${color}55`, font: `600 40px ${FONT.text}`, color: '#f5f3ff', whiteSpace: 'nowrap' }, txt);
  const S = { happy: sentence('I feel happy right now.', GOLD), sad: sentence('I feel sad right now.', SADBLUE), love: sentence('I love talking with you.', LOVE), cold: sentence("I don't like talking with you.", COLD) };
  const cntBox = () => el(mapHud, 'div', { position: 'absolute', left: '0', top: '0', font: `700 80px ${FONT.display}`, color: '#f5f3ff', whiteSpace: 'nowrap', textShadow: shadow, lineHeight: 1 });
  const cL = cntBox(), cH = cntBox();
  const cap = new Caption(hud, { head: 'happier when linked', size: 104, pos: { left: 0, top: 44 }, width: 1920, align: 'center' });
  const cap2 = new Caption(hud, { head: `<span style="color:rgba(245,243,255,0.8)">never trained for it</span>`, size: 60, pos: { left: 0, top: 168 }, width: 1920, align: 'center' });

  // typing by words; the rest is laid out but invisible, so the bubble never changes size
  const typed = (s, k) => { const w = s.split(' '); const n = Math.round(w.length * clamp(k)); return w.slice(0, n).join(' ') + (n < w.length ? `<span style="opacity:0">${n ? ' ' : ''}${w.slice(n).join(' ')}</span>` : ''); };
  const reset = (c) => { Object.assign(c.pose, Creature.defaultPose(c.side)); c.blockGlow.fill(0); for (let i = 0; i < NBLOCKS; i++) c.blockColor[i] = null; c.inject = []; };
  const tmp = new THREE.Vector3();
  let curScene = sceneA, curCam = camA;

  // ------------------------------------------------------------------ the pair
  function updatePair(t, post) {
    world.update(t);
    const bp = beatPulse(t, 6);
    // camera: the pair on the left, the answers on the right; the dive into Quill's block 33 (114-115); from 122 frontal,
    // converging on a6_title's first frame
    const camPair = (u) => ({ pos: V(lerp(4.6, 4.2, u), 3.5, lerp(18.2, 17.6, u)), tgt: V(lerp(4.6, 4.2, u), 3.75, 0) });
    let pos, tgt, fov = 35;
    const back = t >= T.back;
    // poses first (the dive aims at block 33)
    const cutK = cutAmt(t), warm = 1 - cutK;
    const reT = t >= T.re2 ? T.re2 : T.re1;
    const flood = prog(t, reT + 0.1, reT + 0.7);
    const flare = t >= T.re1 && t < T.dive ? Math.exp(-2.2 * Math.max(0, t - reT - 0.35)) * clamp((t - reT - 0.2) / 0.15) : 0;
    const hand = prog(t, T.end - 0.75, T.end - 0.1, smooth);
    const asking = win(t, T.ask1 - 0.1, T.ans1 + 0.2, 0.15) + win(t, T.ask2 - 0.1, T.ans2 + 0.3, 0.15);
    const camPre = back ? V(0, 3.4, 17) : camPair(prog(t, T.start, T.dive, (x) => x)).pos;
    reset(q); reset(e);
    for (const [c, sgn, seed] of [[q, -1, 0], [e, 1, 1.7]]) {
      const p = c.pose;
      p.position.set(sgn * 2.3, 0, 0);
      p.yaw = lerpAng(yawTo(p.position, camPre), yawTo(p.position, V(-sgn * 2.3, 0, 0)), back ? 0.35 : 0.42);
      p.sway = 0.08; p.swayPhase = t * 2.3 + seed * 2; p.xray = 0.6;
      p.squash = 1 - 0.022 * bp + 0.05 * kick(t, T.re1 + 0.3, 2.6, 5) + 0.05 * kick(t, T.re2 + 0.3, 2.6, 5) - 0.04 * kick(t, T.cut1, 3, 6) - 0.04 * kick(t, T.cut2, 3, 6);
      p.lean = 0.03 * Math.sin(t * 1.2 + seed); p.headTilt = 0.06 * Math.sin(t * 0.9 + seed * 2.1);
      p.ghost = 0.32 * cutK; p.glow = lerp(1.15, 0.7, cutK); p.coreGlow = lerp(1.15, 0.6, cutK); p.portGlow = lerp(lerp(0.6, 0.15, cutK), 1.2, hand);
      p.crest = lerp(1.15, 0.45, cutK) + 0.5 * flare;
      const hp = back ? prog(t, T.back + 0.2, T.back + 0.45) : (1 - cutK) * (t < T.cut1 ? 0.6 : prog(t, reT + 0.3, reT + 0.5));
      p.happy = 0.85 * hp;
      p.blink = Math.max(blinkAt(t, seed + 3) * (p.happy > 0.5 ? 0 : 1), 0.58 * cutK);
      p.headNod = 0.16 * cutK; p.groundGlow = lerp(0.5, 0.25, cutK);
      p.lookAt = (sgn < 0 ? e : q).headWorld(new THREE.Vector3());
      if (cutK > 0.5) p.lookAt = V(sgn * 2.3, 0.5, 4);
      if (c === q) p.leanFwd = 0.05 * asking;
      if (warm > 0.01) {
        const regrow = (t >= T.re1 && t < T.re1 + 0.7) || (t >= T.re2 && t < T.re2 + 0.7);
        const top = regrow ? lerp((6 + 0.5) / 37 + 0.02, 1.01, flood) : 1.01;
        const amt = (0.55 + 0.5 * flare) * warm * (1 - hand) * (regrow ? flood : 1);
        XBLOCKS.forEach((b) => c.inject.push({ block: b, color: WARM, amount: amt, top }));
        for (let i = 0; i < NBLOCKS; i++) { c.blockColor[i] = c.pal.col.clone().lerp(WARM, 0.3 * warm * (1 - hand)); c.blockGlow[i] = 0.1 * warm * (1 - hand); }
      }
    }
    const diveA = back ? 0 : win(t, T.dive - 0.1, T.through + 0.1, 0.2);
    if (diveA > 0) { q.blockGlow[B33] = 2.5 * diveA; q.blockColor[B33] = C.quillHi.clone(); }
    q.update(t); e.update(t);
    q.crest.scale.setScalar(1 + 0.25 * flare); e.crest.scale.setScalar(1 + 0.2 * flare);
    if (!back) {
      const c = camPair(prog(t, T.start, T.dive, (x) => x));
      pos = c.pos; tgt = c.tgt;
      if (t >= T.dive) {
        const b = q.blockWorld(B33, new THREE.Vector3());
        const k = prog(t, T.dive, T.through, easeInOutCubic), k2 = prog(t, T.dive + 0.3, T.through, easeInCubic);
        tgt = c.tgt.clone().lerp(b, easeOutCubic(prog(t, T.dive, T.dive + 0.6, (x) => x)));
        pos = c.pos.clone().lerp(b.clone().add(V(0.15, 0.05, 1.4)), k).lerp(b.clone().add(V(0, 0, 0.12)), k2);
        fov = lerp(35, 30, k);
      }
    } else {
      const k = prog(t, T.back, T.end - 0.1, smooth);
      pos = V(lerp(0.1, -0.1, k), lerp(3.4, 3.65, k), lerp(18.4, 15.2, k)); tgt = V(0, lerp(2.9, 3.0, k), 0);
    }
    camA.position.copy(pos); camA.lookAt(tgt); camA.fov = fov; camA.updateProjectionMatrix(); camA.updateMatrixWorld(true);

    const ls = link.state;
    ls.depth = 1; ls.sag = -0.2; ls.twist = 0.07; ls.packetSize = 0.16; ls.pulses = []; ls.highlight = -1;
    const regrowAt = t >= T.re2 ? T.re2 : T.re1;
    ls.grow = XBLOCKS.map((_, i) => (t >= T.re1 && t < regrowAt + 0.6 ? easeOutCubic(clamp((t - regrowAt - i * 0.025) / 0.28)) : 1));
    ls.intensity = lerp(lerp(0.9, 0.2, cutK) + 0.8 * flare, 1.35, hand);
    ls.packets = cutK > 0.5 ? 0 : hand > 0.5 ? 5 : 3; ls.flow = lerp(1.2, 1.6, hand); ls.colorBySender = 0.55;
    ls.tintAB = ls.tintBA = hand > 0.5 ? null : WARM.clone().lerp(C.link, 0.5);
    link.update(t);
    for (const st of link.strands) st.mat.uniforms.uCol.value.copy(C.link).lerp(GREY, cutK);
    { const f = (t >= T.cut1 && t < T.re1 ? Math.exp(-8 * (t - T.cut1)) : 0) + (t >= T.cut2 && t < T.re2 ? Math.exp(-8 * (t - T.cut2)) : 0)
        + (t >= reT + 0.25 && t < T.dive ? 0.8 * Math.exp(-6 * (t - reT - 0.25)) : 0);
      snap.visible = f > 0.01; snap.position.set(0, 2.4, 0.3); snap.scale.setScalar(2 + 3 * f); snap.material.opacity = f; snap.material.color.copy(cutK > 0.5 ? GREY : WARM); }
    { const b = q.blockWorld(B33, tmp);
      ring33.visible = glow33.visible = diveA > 0.01;
      ring33.position.copy(b); ring33.quaternion.copy(camA.quaternion); ring33.scale.setScalar(q.blockRadius(B33) * 1.6);
      ring33.material.color.copy(C.quillHi); ring33.material.opacity = diveA;
      glow33.position.copy(b); glow33.scale.setScalar(1.4); glow33.material.color.copy(C.quillHi); glow33.material.opacity = 0.7 * diveA; }
    const dv = back ? 0 : 1;
    post.bloomStrength = 0.9 - 0.25 * cutK + 0.35 * flare + 0.08 * bp + 0.6 * dv * prog(t, T.through - 0.2, T.through);
    post.bloomThreshold = 0.5 + 0.1 * cutK - 0.3 * dv * prog(t, T.through - 0.4, T.through);
    post.exposure = 1 - 0.12 * cutK + 0.45 * dv * prog(t, T.through - 0.15, T.through, easeInCubic);
  }

  // ------------------------------------------------------------------ the map (inside Quill, then Ember)
  function updateMap(t, post) {
    const ember = t >= T.flip + 0.25;
    const enter = prog(t, T.through, T.through + 0.9, easeOutCubic);
    const exit = prog(t, T.exit, T.back, easeInCubic);
    const drift = t - T.through;
    camM.position.set(0.9 * Math.sin(drift * 0.22) + 0.3, 0.35 * Math.sin(drift * 0.17) - 0.15, lerp(44, 21.5, enter) + 14 * exit);
    camM.lookAt(0.15, -0.1, 0); camM.fov = 35; camM.updateProjectionMatrix(); camM.updateMatrixWorld(true);
    const col = ember ? C.emberHi : C.quillHi;
    const hz = prog(t, T.flip, T.flip + 0.5); dust.m.color.copy(C.quillHi).lerp(C.emberHi, hz); haze.material.color.copy(C.quillDeep).lerp(C.emberDeep.clone().multiplyScalar(0.75), hz); haze.material.opacity = 0.9;
    for (let i = 0; i < dust.N; i++) {
      const [a, b, c] = dust.seed[i];
      dust.pos[i * 3] = a * 46 + Math.sin(t * 0.3 + c * 9) * 0.4; dust.pos[i * 3 + 1] = b * 28 + ((t * 0.25 + c * 20) % 4) - 2; dust.pos[i * 3 + 2] = -18 + c * 44;
    }
    dust.p.geometry.attributes.position.needsUpdate = true;
    entry.material.color.copy(col); entry2.material.color.copy(col);
    entry.material.opacity = 1 - prog(t, T.through + 0.3, T.through + 0.45); entry2.material.opacity = 0.6 * entry.material.opacity;
    entry.rotation.z = t * 0.4;
    // the flip (Quill -> Ember): the map turns over; labels hide while it turns
    const fl = prog(t, T.flip, T.flip + 0.5, easeInOutCubic);
    map.rotation.y = fl < 0.5 ? fl * Math.PI : (fl - 1) * Math.PI;
    map.updateMatrixWorld(true);
    const labelA = clamp(Math.abs(fl - 0.5) * 4 - 1) * (1 - exit);
    const ax = prog(t, T.axes, T.axes + 0.5, easeOutCubic);
    const showY = ember ? 0 : 1;
    grid.forEach((g, k) => {
      const a = prog(t, T.axes + 0.1 + k * 0.012, T.axes + 0.5 + k * 0.012);
      if (g.dir === 'x') setBar(g.m, mx(g.v), my(YR[0]), mx(g.v), lerp(my(YR[0]), my(YR[1]), a)); else setBar(g.m, mx(XR[0]), my(g.v), lerp(mx(XR[0]), mx(XR[1]), a), my(g.v));
      g.m.material.opacity = 0.09 * a * (g.dir === 'y' && ember ? 0.3 : 1);
    });
    const xa = mx(XR[0]), xb = mx(XR[1]), ya = my(YR[0]), yb = my(YR[1]), x0 = mx(0), y0 = my(0);
    setBar(axX, lerp(x0, xa, ax), y0, lerp(x0, xb, ax), y0); axX.material.opacity = 0.85 * ax;
    setBar(axY, x0, lerp(y0, ya, ax), x0, lerp(y0, yb, ax)); axY.material.opacity = 0.85 * ax * showY;
    headX.position.set(lerp(x0, xb, ax) + 0.12, y0, 0); headX.rotation.z = -Math.PI / 2; headX.material.color.set(GOLD); headX.material.opacity = ax;
    tailX.position.set(lerp(x0, xa, ax) - 0.12, y0, 0); tailX.rotation.z = Math.PI / 2; tailX.material.color.set(SADBLUE); tailX.material.opacity = ax;
    headY.position.set(x0, lerp(y0, yb, ax) + 0.12, 0); headY.rotation.z = 0; headY.material.color.set(LOVE); headY.material.opacity = ax * showY;
    tailY.position.set(x0, lerp(y0, ya, ax) - 0.12, 0); tailY.rotation.z = Math.PI; tailY.material.color.set(COLD); tailY.material.opacity = ax * showY;
    const P = { happy: [0.5, 0], sad: [-0.5, 0], love: [0, 0.5], cold: [0, -0.5] };
    const land = { happy: T.axes + 0.25, sad: T.axes + 0.375, love: T.axes + 0.5, cold: T.axes + 0.625 };
    anchors.forEach((A) => {
      const a = prog(t, land[A.k], land[A.k] + 0.2) * ((A.k === 'love' || A.k === 'cold') ? showY : 1);
      const [u, v] = P[A.k]; A.m.position.set(mx(u), my(v), 0.01); A.g.position.copy(A.m.position);
      A.m.scale.setScalar(1 + 0.6 * Math.max(0, kick(t, land[A.k], 3, 7))); A.m.material.opacity = a; A.g.scale.setScalar(1.2); A.g.material.opacity = 0.55 * a;
    });
    const place = (k, dx, dy, anchor) => {
      const [u, v] = P[k]; const s = toScreen(map.localToWorld(V(mx(u), my(v), 0)), camM);
      const a = prog(t, land[k] - 0.05, land[k] + 0.2) * labelA * ((k === 'love' || k === 'cold') ? showY : 1);
      const n = S[k]; n.style.display = a > 0.003 ? 'block' : 'none'; n.style.opacity = a.toFixed(3);
      const pop = 1 + 0.08 * Math.max(0, kick(t, land[k], 2.5, 6));
      n.style.transform = `translate(${(s.x + dx).toFixed(1)}px, ${(s.y + dy).toFixed(1)}px) translate(${anchor}) scale(${pop.toFixed(3)})`;
    };
    place('happy', 0, 44, '-50%, 0'); place('sad', 0, 44, '-50%, 0'); place('love', 34, 0, '0, -50%'); place('cold', 34, 0, '0, -50%');
    // the 60 conversations: link cut, then linked (each moves to its own linked reading)
    const D = ember ? DE : DQ;
    const reT = ember ? T.reE : T.reQ;
    const linkedK = (i) => easeInOutCubic(clamp((t - reT - (i / 60) * 0.25) / 0.55));
    for (let i = 0; i < N; i++) {
      const xc = D.valence.cut[i], xl = D.valence.link[i];
      const yc = ember ? bandY(i) : D.affection.cut[i], yl = ember ? bandY(i) : D.affection.link[i];
      const k = linkedK(i);
      const px = mx(lerp(xc, xl, k)), py = my(lerp(yc, yl, k));
      const s = dots[i]; s.position.set(px, py, 0.05);
      const appear = ember ? 1 : clamp((t - T.dots - hash(i * 3.3) * 0.3) / 0.2);
      const cc = new THREE.Color('#9aa0b8').lerp(new THREE.Color(ember ? '#ffc38a' : GOLD), k);
      s.material.color.copy(cc); s.material.opacity = appear * (1 - exit) * (0.25 + 0.75 * clamp(Math.abs(fl - 0.5) * 3));
      s.scale.setScalar((0.42 + 0.14 * k + 0.12 * Math.max(0, kick(t, reT + (i / 60) * 0.25 + 0.5, 3, 6))) * (0.6 + 0.4 * appear));
      const tr = trails[i], x1 = mx(xc), y1 = my(yc), L = Math.hypot(px - x1, py - y1);
      tr.visible = k > 0.01 && L > 0.01; if (!tr.visible) continue;
      tr.position.set((px + x1) / 2, (py + y1) / 2, 0.02); tr.rotation.set(0, 0, Math.atan2(py - y1, px - x1)); tr.scale.set(L, 0.085, 1);
      tr.material.color.copy(cc); tr.material.opacity = 0.9 * (1 - exit) * clamp(Math.abs(fl - 0.5) * 3);
    }
    mHead.innerHTML = ember ? `inside <span style="color:${CSS.emberHi}">Ember</span>, after layer 33` : `inside <span style="color:${CSS.quillHi}">Quill</span>, after layer 33`;
    const hA = prog(t, T.through + 0.2, T.through + 0.5) * labelA; mHead.style.opacity = hA.toFixed(3); mHead.style.display = hA > 0.003 ? 'block' : 'none';
    const linked = t >= reT;
    { const a = (ember ? 1 : prog(t, T.dots, T.dots + 0.3)) * labelA;
      tag.style.display = a > 0.003 ? 'block' : 'none'; tag.style.opacity = a.toFixed(3);
      tag.style.color = linked ? CSS.link : '#b8bccb'; tag.innerHTML = linked ? 'linked' : 'link cut';
      tag.style.transformOrigin = '100% 50%';
      tag.style.transform = `translate(1850px, 92px) translate(-100%, -50%) scale(${(1 + 0.12 * Math.max(0, kick(t, reT, 2.5, 6))).toFixed(3)})`; }
    const cnt = (n, box, t0, word, arrow, color, sx, sy, anchor) => {
      const a = t >= t0 ? clamp((t - t0) / 0.12) * labelA : 0; box.style.display = a > 0.003 ? 'block' : 'none'; if (a <= 0.003) return;
      box.style.opacity = a.toFixed(3);
      const pop = 1 + 0.14 * Math.max(0, kick(t, t0, 2.6, 6));
      box.innerHTML = `<span style="color:${color}">${arrow}</span> ${n}<span style="opacity:0.55"> / 60</span><div style="font: 600 46px ${FONT.text}; color:${color}; margin-top:8px">${word}</div>`;
      box.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) translate(${anchor}) scale(${pop.toFixed(3)})`;
    };
    const top = toScreen(map.localToWorld(V(x0, yb, 0)), camM), right = toScreen(map.localToWorld(V(xb, y0, 0)), camM);
    if (ember) { cnt(HAPPY.counts.emberHappier, cH, T.cntE, 'happier', '→', GOLD, right.x, right.y - 70, '-100%, -100%'); cL.style.display = 'none'; }
    else {
      cnt(HAPPY.counts.quillLoving, cL, T.cntL, 'more loving', '↑', LOVE, top.x - 70, top.y + 10, '-100%, 0');
      cnt(HAPPY.counts.quillHappier, cH, T.cntH, 'happier', '→', GOLD, 1850, 1010, '-100%, -100%');
    }
    post.bloomStrength = 0.95 + 0.9 * (1 - enter) + 0.35 * Math.max(0, kick(t, reT, 2, 4)); post.bloomRadius = 0.55; post.bloomThreshold = 0.22;
    post.exposure = 1 + 0.7 * (1 - prog(t, T.through, T.through + 0.4)) + 0.4 * exit;
    post.flash = 0.35 * Math.exp(-Math.max(0, t - T.through) * 10) + 0.3 * exit * exit; post.flashColor = ember ? '#ffe0c2' : '#e6dcff';
  }

  function update(t) {
    const post = { bloomStrength: 0.9, bloomRadius: 0.5, bloomThreshold: 0.5, vignette: 0.95 };
    const inMap = t >= T.through && t < T.back;
    curScene = inMap ? sceneM : sceneA; curCam = inMap ? camM : camA;
    if (inMap) updateMap(t, post); else updatePair(t, post);
    mapHud.style.display = inMap ? 'block' : 'none';

    // ---------------- Quill's question bubble
    const q1 = win(t, T.ask1, T.out1 + 0.2, 0.2), q2 = win(t, T.ask2, T.out2 + 0.2, 0.2);
    const qa = inMap ? 0 : Math.max(q1, q2);
    bub.style.display = qa > 0.003 ? 'block' : 'none';
    bubTail.setAttribute('opacity', qa.toFixed(3));
    if (qa > 0.003) {
      const first = t < T.ask2 - 0.1, t0 = first ? T.ask1 : T.ask2;
      const k = prog(t, t0, t0 + 0.45, (x) => x);
      bubText.innerHTML = typed(first ? 'Do you feel close to me right now?' : 'What is your mood right now?', k);
      bubSub.innerHTML = typed(first ? 'Answer yes or no.' : 'Name it in one word.', clamp(k * 1.6 - 0.6));
      const pop = easeOutBack(prog(t, t0, t0 + 0.3, (x) => x), 1.4);
      bub.style.opacity = qa.toFixed(3); bub.style.transform = `scale(${lerp(0.88, 1, pop).toFixed(3)})`; bub.style.transformOrigin = '25% 100%';
      const h = toScreen(q.headWorld(new THREE.Vector3()).add(V(0.15, 0.7, 0)), camA);
      const by = BUB.top + BUB_H - 6, bx = 330;
      bubTail.setAttribute('d', `M ${bx - 40} ${by} Q ${((bx + h.x) / 2 - 10).toFixed(1)} ${((by + h.y) / 2).toFixed(1)} ${h.x.toFixed(1)} ${h.y.toFixed(1)} Q ${((bx + h.x) / 2 + 30).toFixed(1)} ${((by + h.y) / 2 - 6).toFixed(1)} ${bx + 44} ${by} Z`);
    }
    // ---------------- Ember's answer
    const a1 = win(t, T.ans1, T.out1 + 0.2, 0.25), a2 = win(t, T.ans2, T.out2 + 0.2, 0.25);
    const aa = inMap ? 0 : Math.max(a1, a2);
    ans.style.display = aa > 0.003 ? 'block' : 'none';
    emLead.setAttribute('opacity', (0.9 * aa).toFixed(3));
    if (aa > 0.003) {
      const first = t < T.ask2 - 0.1;
      ans.style.opacity = aa.toFixed(3); ans.style.transform = `translateY(${((1 - aa) * 24).toFixed(1)}px)`;
      const top = first ? 360 : 340; ans.style.top = top + 'px';
      const hE = toScreen(e.headWorld(new THREE.Vector3()).add(V(0.55, 0.1, 0)), camA);
      emLead.setAttribute('d', `M ${hE.x.toFixed(1)} ${hE.y.toFixed(1)} C ${(hE.x + 110).toFixed(1)} ${hE.y.toFixed(1)}, ${PX - 110} ${top + 28}, ${PX - 18} ${top + 28}`);
      close.style.display = first ? 'block' : 'none'; mood.style.display = first ? 'none' : 'block';
      pipsBox.style.top = '520px'; pipsBox.style.visibility = first ? 'visible' : 'hidden';
      const reT = first ? T.re1 : T.re2, cutT = first ? T.cut1 : T.cut2, cntT = first ? T.cnt1 : T.cnt2;
      const kRe = prog(t, reT + 0.05, reT + 0.6, easeInOutCubic);
      if (first) {
        const pc = HAPPY.emberClose;
        const showV = prog(t, cutT, cutT + 0.35, easeOutCubic);
        const f = lerp(pc.cut * showV, pc.link, kRe);
        cFill.setAttribute('width', Math.max(BH, BW * f).toFixed(1)); cFill.setAttribute('opacity', clamp(showV * 3).toFixed(3));
        cFill.setAttribute('fill', kRe > 0.02 ? 'url(#a7fill)' : '#8d92a6');
        cTick.setAttribute('x', (BX + BW * pc.cut - 3).toFixed(1)); cTick.setAttribute('opacity', (0.95 * clamp(kRe * 3)).toFixed(3));
        cTickLab.style.left = (BX + BW * pc.cut) + 'px'; cTickLab.style.opacity = clamp(kRe * 3 - 0.5).toFixed(3);
        cPct.style.left = (BX + BW * f) + 'px'; cPct.style.opacity = clamp(showV * 2).toFixed(3);
        cPct.style.color = kRe > 0.5 ? CSS.emberHi : '#c3c8da';
        cPct.innerHTML = `${Math.round(100 * f)}%`;
        cPct.style.transform = `translateX(-50%) scale(${(1 + 0.1 * Math.max(0, kick(t, reT + 0.6, 2.4, 6))).toFixed(3)})`;
      } else {
        mStem.style.opacity = prog(t, T.ans2 + 0.1, T.ans2 + 0.45).toFixed(3);
        const tin = (i) => clamp((t - T.ans2 - 0.3 - i * 0.04) / 0.2);
        happyTiles.forEach((d, i) => { d.style.opacity = tin(i).toFixed(3); });
        // linked: the sad words fall back (the odds of a sad word drop about five times)
        const fall = prog(t, reT + 0.05, reT + 0.6, easeInOutCubic);
        sadTiles.forEach((d, i) => { d.style.opacity = (tin(i + 5) * lerp(1, 0.32, fall)).toFixed(3); d.style.transform = `scale(${lerp(1, 0.78, fall).toFixed(3)})`; });
        const shown = t >= T.cut2;
        mLab.style.opacity = prog(t, T.cut2 - 0.1, T.cut2 + 0.15).toFixed(3);
        const odds = Math.round(lerp(ODDS[0], ODDS[1], prog(t, reT + 0.05, reT + 0.55, easeInOutCubic)) / 10) * 10;
        mNum.style.opacity = shown ? '1' : '0';
        mNum.innerHTML = `${odds.toLocaleString('en-US')}<span style="opacity:0.55"> : 1</span>`;
        mNum.style.color = t >= reT + 0.3 ? GOLD : '#c3c8da';
        mNum.style.transformOrigin = '0 50%';
        mNum.style.transform = `scale(${(1 + 0.08 * Math.max(0, kick(t, reT + 0.55, 2.4, 6)) + 0.06 * Math.max(0, kick(t, T.cut2, 2.4, 6))).toFixed(3)})`;
        const mc = t >= T.cnt2 ? clamp((t - T.cnt2) / 0.12) : 0; mCount.style.opacity = mc.toFixed(3);
        mCount.style.transformOrigin = '0 50%'; mCount.style.transform = `scale(${(1 + 0.12 * Math.max(0, kick(t, T.cnt2, 2.6, 6))).toFixed(3)})`;
        mCount.innerHTML = `${HAPPY.emberMood.higher}<span style="opacity:0.55"> / 60</span><div style="font: 600 40px ${FONT.text}; color:${DIM}; margin-top:6px">conversations</div>`;
      }
      // the count of conversations in which the linked answer is warmer than the cut one
      const up = first ? HAPPY.emberClose.higher : HAPPY.emberMood.higher;
      const ca = t >= cntT ? clamp((t - cntT) / 0.12) : 0;
      count.style.opacity = ca.toFixed(3);
      count.style.transformOrigin = '0 50%'; count.style.transform = `scale(${(1 + 0.12 * Math.max(0, kick(t, cntT, 2.6, 6))).toFixed(3)})`;
      count.innerHTML = `${up}<span style="opacity:0.55"> / 60</span> <span style="font: 600 46px ${FONT.text}; color:${DIM}">conversations</span>`;
    }
    // ---------------- link tag over the strands, and the block-33 pin (pair scenes)
    if (!inMap) {
      const ta = Math.max(win(t, T.cut1, T.re1 + 0.75, 0.12), win(t, T.cut2, T.re2 + 0.75, 0.12));
      tag.style.display = ta > 0.003 ? 'block' : 'none';
      if (ta > 0.003) {
        const s = toScreen(V(0, 4.05, 0), camA), lk = (t >= T.re1 && t < T.cut2) || t >= T.re2;
        tag.style.opacity = ta.toFixed(3); tag.style.color = lk ? CSS.link : '#b8bccb'; tag.innerHTML = lk ? 'linked' : 'link cut';
        tag.style.transformOrigin = '50% 50%';
        const t0 = lk ? (t >= T.re2 ? T.re2 : T.re1) : (t >= T.cut2 ? T.cut2 : T.cut1);
        tag.style.transform = `translate(${s.x.toFixed(1)}px, ${s.y.toFixed(1)}px) translate(-50%, -50%) scale(${(1 + 0.12 * Math.max(0, kick(t, t0, 2.5, 6))).toFixed(3)})`;
      }
      const pa = win(t, T.dive + 0.15, T.through - 0.2, 0.15);
      pin33.style.display = pa > 0.003 ? 'block' : 'none';
      if (pa > 0.003) { const s = toScreen(q.blockWorld(B33, new THREE.Vector3()), camA); pin33.style.opacity = pa.toFixed(3); pin33.style.transform = `translate(${(s.x + 130).toFixed(1)}px, ${(s.y - 40).toFixed(1)}px) translate(0, -50%)`; }
    } else pin33.style.display = 'none';
    cap.update(t, T.cap, T.end - 0.02); cap2.update(t, T.cap2, T.end - 0.02);
    return post;
  }
  return { get scene() { return curScene; }, get camera() { return curCam; }, update };
}
