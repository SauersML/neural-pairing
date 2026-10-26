// a5_train (54–66 s): training. Visual-first.
//  54–58  fitted start: real token rows snap together where tokens end at the same character, a matrix slab maps
//         Ember's states into Quill's space, attention lands on the matching token; stat only in the hold.
//  58–61  frost on the original blocks (fixed), teal rank-32 LoRA chips light on every block (trained); one label in the hold.
//  61–66  training montage: game cards stream in as data (value games first, then the full mix), forward passes race up
//         both bodies, and a chart (shots/train_chart2.js) draws the median chance of writing the partner's value over
//         training steps: linked rises 0 -> 99% (S-curve), link cut stays at 0. Tension to the cut.
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Pin, el, toScreen, FONT } from '../lib/hud.js';
import { C, CSS } from '../lib/palette.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, easeOutBack, easeOutCubic, easeInCubic, easeInOutCubic, hash, noise1 } from '../lib/util.js';
import { tileTexture, outlineTexture, starTexture, crossTexture } from './s4_train_tex.js';
import { slabTexture, makeTrainDeck, makeTrainDeck3D, DECK_VALUE, DECK_MIX, MIX_STEP } from './a5_train_kit.js';
import { makeTrainChart, STEP_END } from './train_chart2.js';

const GOLD = new THREE.Color('#ffd166');
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

// token rows: the whole sentence "Oslo has 709,037 people." (data/tokens.json, real tokenizations), laid out by character
const C0 = 0, C1 = 24;
const CW = 0.28, TH = 0.5, GAP = 0.035;
const QY = 4.0, EY = 3.0, RZ = 1.0;
const xAt = (ch) => (ch - (C0 + C1) / 2) * CW;
const X0 = xAt(C0), X1 = xAt(C1);
const AX = 5.0, AZ = -2.0;

const T = {
  connA: 54.5, connB: 55.375,          // token-end matches flash on 16ths across this sweep (tEndOf is quantized)
  mA: 55.125, mB: 55.875,
  att: [56.0, 56.25, 56.5],             // both directions read at each of these beats/8ths
  stat: [57.0, 58.0],
  rowsOut: 58.0, move: 57.75,
  iceQ: 58.25, iceE: 58.375, iceDur: 1.0,
  loraA: 59.0, loraDur: 0.75,
  loraLbl: [60.0, 61.0],
  curA: 62.0, curM: 64.5, curB: 65.5, cardsA: 61.0, end: 66.0,
};
// x timing stretched for pacing (warm-up 0..900 over curA..curM, slow at first; then the full mix to step 1,320)
const stepAt = (t) => t < T.curM ? MIX_STEP * Math.pow(clamp((t - T.curA) / (T.curM - T.curA)), 1.6) : lerp(MIX_STEP, STEP_END, easeInOutCubic(clamp((t - T.curM) / (T.curB - T.curM))));

export async function create({ hud, def, renderer, scale }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  const T0 = def.start;
  const tokData = await (await fetch('data/tokens.json')).json();
  const dot = Creature.dotTexture(), star = starTexture();
  const addMat = (col, op = 1) => new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  const spriteMat = (map, col) => new THREE.SpriteMaterial({ map, color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });

  // ================= PART 1: token rows =================
  const rows = new THREE.Group(); scene.add(rows);
  const mkTiles = (list, kind, y) => list.map((tk, i) => {
    const w = (tk.end - tk.start) * CW - GAP;
    const { tex } = tileTexture(tk.text, kind, w, TH);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false, opacity: 0 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, TH), mat);
    const omat = new THREE.MeshBasicMaterial({ map: outlineTexture(kind, w, TH), transparent: true, depthWrite: false, toneMapped: false, opacity: 0, blending: THREE.AdditiveBlending, color: GOLD.clone() });
    const outline = new THREE.Mesh(new THREE.PlaneGeometry(w * 1.04 + 0.02, TH * 1.08), omat);
    const g = new THREE.Group(); g.add(mesh); g.add(outline); outline.position.z = 0.002;
    g.userData = { cx: (xAt(tk.start) + xAt(tk.end)) / 2, y, i, w, tk, mat, omat };
    rows.add(g); return g;
  });
  const inWin = (tk) => tk.start >= C0 && tk.end <= C1;
  const qList = tokData.quill.filter(inWin), eList = tokData.ember.filter(inWin);
  const qTiles = mkTiles(qList, 'quill', QY), eTiles = mkTiles(eList, 'ember', EY);
  const matches = tokData.matches_quill_ember.map(([a, b]) => [qList.indexOf(tokData.quill[a]), eList.indexOf(tokData.ember[b])]).filter(([a, b]) => a >= 0 && b >= 0);
  const qMatched = new Map(matches.map(([a, b]) => [a, b])), eMatched = new Map(matches.map(([a, b]) => [b, a]));
  const tEndOf = (end) => T.connA + 0.125 * Math.round((T.connB - T.connA) * ((end - C0) / (C1 - C0)) / 0.125);
  const conns = matches.map(([qi, ei]) => {
    const end = qList[qi].end, x = xAt(end) - GAP / 2, y0 = QY - TH / 2, y1 = EY + TH / 2;
    const line = new THREE.Mesh(new THREE.PlaneGeometry(0.04, 1), addMat(GOLD.clone()));
    const capA = new THREE.Sprite(spriteMat(dot, GOLD.clone())), capB = new THREE.Sprite(spriteMat(dot, GOLD.clone()));
    const flash = new THREE.Sprite(spriteMat(star, new THREE.Color('#fff3c4')));
    rows.add(line, capA, capB, flash); capA.position.set(x, y0, RZ + 0.01); capB.position.set(x, y1, RZ + 0.01);
    return { qi, ei, x, y0, y1, line, capA, capB, flash, tEnd: tEndOf(end) };
  });
  const xTex = crossTexture();
  const misses = qList.map((tk, i) => [tk, i]).filter(([, i]) => !qMatched.has(i)).map(([tk, i]) => {
    const x = xAt(tk.end) - GAP / 2;
    const stub = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.2), addMat(new THREE.Color('#8b8fa3'), 0)); stub.position.set(x, QY - TH / 2 - 0.1, RZ);
    const mark = new THREE.Sprite(spriteMat(xTex, new THREE.Color('#ffffff'))); mark.position.set(x, QY - TH / 2 - 0.3, RZ + 0.01);
    rows.add(stub, mark); return { stub, mark, tEnd: tEndOf(tk.end) };
  });
  // the matrix slab (ridge map) sliding between the rows
  const mGroup = new THREE.Group(); rows.add(mGroup);
  const mH = (QY - TH / 2) - (EY + TH / 2) - 0.08, mW = mH * 1.6;
  const mMat = new THREE.MeshBasicMaterial({ map: slabTexture(), transparent: true, depthWrite: false, toneMapped: false, opacity: 0, side: THREE.DoubleSide });
  mGroup.add(new THREE.Mesh(new THREE.PlaneGeometry(mW, mH), mMat));
  const mEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(mW, mH, 0.14)), new THREE.LineBasicMaterial({ color: C.linkHi, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  mGroup.add(mEdge);
  const mGlow = new THREE.Sprite(spriteMat(dot, C.link.clone())); mGlow.scale.set(mW * 2.4, mH * 2.4, 1); mGroup.add(mGlow);
  const orbs = matches.map(() => { const s = new THREE.Sprite(spriteMat(dot, C.ember.clone())); rows.add(s); return s; });
  const docks = matches.map(() => { const s = new THREE.Sprite(spriteMat(star, C.quillHi.clone())); rows.add(s); return s; });
  // attention: beams from a Quill token to every Ember token; bars under Ember's row
  const beams = eTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), addMat(C.quillHi.clone(), 0)); rows.add(m); return m; });
  const bars = eTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), addMat(C.quillHi.clone(), 0)); rows.add(m); return m; });
  // both directions read at once: Quill's token reads every Ember token (violet fan, bars under Ember's row) while the
  // matching Ember token reads every Quill token (orange fan, bars over Quill's row)
  const beamsE = qTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), addMat(C.emberHi.clone(), 0)); rows.add(m); return m; });
  const barsE = qTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), addMat(C.emberHi.clone(), 0)); rows.add(m); return m; });
  const READERS = [[1, 1], [6, 4], [11, 7]]; // lo↔lo, 9↔709, people↔people (each pair ends at the same character)
  const attW = READERS.map(([, ei], k) => {
    const w = eTiles.map((_, j) => (j === ei ? 0 : 0.2 + hash(j * 7.1 + k * 3.3) * (Math.abs(j - ei) <= 1 ? 1.4 : 0.6)));
    const s = w.reduce((a, b) => a + b, 0), peak = [0.9, 0.91, 0.95][k];
    return w.map((v, j) => (j === ei ? peak : v / s * (1 - peak)));
  });
  const attWE = READERS.map(([qi], k) => {
    const w = qTiles.map((_, j) => (j === qi ? 0 : 0.2 + hash(j * 5.3 + k * 4.1 + 2) * (Math.abs(j - qi) <= 1 ? 1.4 : 0.6)));
    const s = w.reduce((a, b) => a + b, 0), peak = [0.88, 0.9, 0.94][k];
    return w.map((v, j) => (j === qi ? peak : v / s * (1 - peak)));
  });
  // the fitted map works both ways (one ridge map per depth and direction): orbs cross the slab both ways at once
  const orbsQ = matches.map(() => { const s = new THREE.Sprite(spriteMat(dot, C.quill.clone())); rows.add(s); return s; });
  const docksE = matches.map(() => { const s = new THREE.Sprite(spriteMat(star, C.emberHi.clone())); rows.add(s); return s; });

  // ================= PART 2: frost and LoRA =================
  const iceCrystalMat = new THREE.MeshStandardMaterial({ color: '#d9f1ff', emissive: C.ice.clone(), emissiveIntensity: 0.25, roughness: 0.08, metalness: 0.1, transparent: true, opacity: 0.85, flatShading: true });
  const crystalGeo = new THREE.OctahedronGeometry(0.06, 0); crystalGeo.scale(0.7, 2.4, 0.7);
  const NC = 70, NS = 90;
  const iceFx = [q, e].map((c, ci) => {
    const inst = new THREE.InstancedMesh(crystalGeo, iceCrystalMat, NC); inst.frustumCulled = false; scene.add(inst);
    const spec = Array.from({ length: NC }, (_, k) => ({ b: 1 + Math.floor(hash(k * 3.7 + ci * 11) * 34), a: (hash(k * 5.3 + ci) - 0.5) * Math.PI * 1.7, dy: (hash(k * 9.1 + ci) - 0.5) * 0.1, s: 0.6 + hash(k * 2.9 + ci * 4) * 0.9, tilt: (hash(k * 1.9 + ci) - 0.2) * 0.9 }));
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 3), 3)); sg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NS * 3), 3));
    const sparks = new THREE.Points(sg, new THREE.PointsMaterial({ size: 0.2, map: star, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })); sparks.frustumCulled = false; scene.add(sparks);
    const sspec = Array.from({ length: NS }, (_, k) => ({ b: Math.floor(hash(k * 4.3 + ci * 7) * 36), a: (hash(k * 6.1 + ci * 2) - 0.5) * Math.PI * 1.8, dy: (hash(k * 8.7 + ci) - 0.5) * 0.12, d: hash(k * 3.1 + ci) * 0.2, f: 2 + hash(k * 7.7 + ci) * 5 }));
    const front = new THREE.Sprite(spriteMat(dot, C.ice.clone())); scene.add(front);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NBLOCKS * 3), 3)); lg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NBLOCKS * 3), 3));
    const lflash = new THREE.Points(lg, new THREE.PointsMaterial({ size: 0.32, map: dot, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })); lflash.frustumCulled = false; scene.add(lflash);
    return { c, inst, spec, sparks, sspec, front, lflash };
  });

  // ================= PART 3: cards, sparks, chart =================
  const cardLayer = el(hud, 'div', { position: 'absolute', inset: '0px', pointerEvents: 'none' });
  // cards are launched on 8ths, then on 16ths (the montage speeds up); each lands 0.75 s later, on the grid
  const NCARD = 24;
  let nv = 0, nm = 0;
  const specs = Array.from({ length: NCARD }, (_, k) => {
    const s = k < 8 ? T.cardsA + 0.25 * k : 63.0 + 0.125 * (k - 8);
    const arrive = s + 0.75;
    const mix = stepAt(arrive) >= MIX_STEP;
    const key = mix ? DECK_MIX[nm++ % DECK_MIX.length] : DECK_VALUE[nv++ % DECK_VALUE.length];
    return { k, s, arrive, key };
  });
  // the cards are the game windows (3D glass panes from the games kit); DOM cards only if that kit is unavailable
  const deck = (await makeTrainDeck3D({ renderer, scene, scale }, specs)) || (await makeTrainDeck(cardLayer, specs));
  const cards = deck.cards.map((cd) => {
    const k = cd.k, side = hash(k * 2.17), r = hash(k * 5.71);
    const start = deck.is3D ? { x: 0.5 + r * 1.5, y: 4.3 + 0.8 * hash(k * 3.3), z: 1.6 + 0.8 * hash(k * 7.7) } // world offset from the gate
      : { x: 380 + r * 240, y: -deck.H * deck.s0 * 0.5 - 20 }; // DOM fallback: a stream from the top, falling into the gate
    // two sparks carry its prompts into both tails (Quill's part and Ember's part)
    const spQ = new THREE.Sprite(spriteMat(dot, C.quillHi.clone())), spE = new THREE.Sprite(spriteMat(dot, C.emberHi.clone()));
    scene.add(spQ, spE);
    return { ...cd, start, arc: (side - 0.5) * 120, rot: (hash(k * 9.1) - 0.5) * 24, spQ, spE, col: new THREE.Color(cd.inst.accent) };
  });
  const chart = makeTrainChart(hud, { left: 1130, top: 250, w: 640, h: 470 });

  // ================= HUD (holds only) =================
  const stat = el(hud, 'div', { position: 'absolute', left: '0px', top: '0px', width: '1920px', textAlign: 'center', whiteSpace: 'nowrap', pointerEvents: 'none', textShadow: '0 2px 24px rgba(0,0,0,0.9)' },
    `<span style="font:800 104px ${FONT.display};color:#ffd166;letter-spacing:-0.035em">88–97%</span><span style="font:500 50px ${FONT.text};color:rgba(245,243,255,0.9);margin-left:30px">look at the matching token, before training</span><div style="font:500 42px ${FONT.text};color:rgba(245,243,255,0.62);margin-top:4px">by chance: 1.2%</div>`);
  // what the rows are, and what the gold lines mean
  const capRows = el(hud, 'div', { position: 'absolute', left: '0px', top: '96px', width: '1920px', textAlign: 'center', whiteSpace: 'nowrap', pointerEvents: 'none', font: `700 60px ${FONT.display}`, color: '#f5f3ff', textShadow: '0 2px 24px rgba(0,0,0,0.9)' },
    `the same text, split two ways`);
  const capLines = el(hud, 'div', { position: 'absolute', left: '0px', top: '176px', width: '1920px', textAlign: 'center', whiteSpace: 'nowrap', pointerEvents: 'none', font: `600 46px ${FONT.text}`, color: '#ffd166', textShadow: '0 2px 24px rgba(0,0,0,0.9)' },
    `lined up where tokens end together`);
  const arrow = `<svg width="70" height="24" viewBox="0 0 70 24" style="vertical-align:middle;margin-left:16px"><path d="M2 12 H60 M48 2 L64 12 L48 22" fill="none" stroke="${CSS.link}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const loraLbl = new Pin(hud, `rank-32 LoRA${arrow}`, { color: CSS.link, size: 54, anchor: 'right', dot: false, weight: 700 });

  const blinkAt = (t, list) => Math.max(0, ...list.map((b) => win(t, b, b + 0.2, 0.08)));
  const tmpQ = new THREE.Quaternion(), tmpM = new THREE.Matrix4(), UP = V3(0, 1, 0);
  const placeBeam = (m, a, b, w) => { const d = b.clone().sub(a); m.position.copy(a).add(b).multiplyScalar(0.5); m.scale.set(w, d.length(), 1); m.rotation.set(0, 0, Math.atan2(d.y, d.x) - Math.PI / 2); };
  const UB = XBLOCKS.map((b) => (b + 0.5) / NBLOCKS);

  return {
    scene, camera,
    update(t) {
      world.update(t);
      { const c3 = prog(t, 60.9, 61.8); world.sky.material.uniforms.uAmp.value = 1 - 0.3 * c3; world.rimQ.intensity = 30 - 14 * c3; }
      const bp = beatPulse(t, 7);
      const tens = prog(t, 63.8, T.end, easeInCubic);
      // forward passes in part 3: one wave in lockstep in both bodies, accelerating
      const tc = Math.max(0, t - 61.2), phase = 1.1 * tc + 0.32 * tc * tc;
      const wave = t > 61.2 ? (phase % 1) * 1.12 - 0.06 : -1;

      // ---------- camera ----------
      const camA = { p: V3(lerp(-0.3, 0.3, prog(t, 54, 58, smooth)), 3.15, lerp(9.5, 9.1, prog(t, 54, 58, smooth))), l: V3(lerp(-0.1, 0.1, prog(t, 54, 58, smooth)), 3.0, 1.0) };
      const obB = -0.14 + 0.2 * prog(t, 58, 61);
      const lBy = lerp(2.2, 2.75, prog(t, 58.2, 59.6, smooth));
      const camB = { p: V3(Math.sin(obB) * 10.6, 3.35, Math.cos(obB) * 10.6), l: V3(0, lBy, 0) };
      const ang = 0.05 + 0.12 * prog(t, 61, 66) + 0.1 * Math.pow(prog(t, 64, 66), 2);
      const rad = lerp(12.8, 10.9, prog(t, 61, 66, (x) => x * x));
      const shake = tens * tens * 0.03;
      const cxC = lerp(3.5, 2.9, prog(t, 63.5, 66));
      const camC = { p: V3(cxC + Math.sin(ang) * rad + noise1(t * 11, 1) * shake, 3.05 + noise1(t * 13, 2) * shake, Math.cos(ang) * rad), l: V3(cxC, 2.35, 0) };
      const kAB = prog(t, 57.85, 58.9, easeInOutCubic), kBC = prog(t, 60.95, 61.75, easeInOutCubic);
      camera.position.copy(camA.p.clone().lerp(camB.p, kAB).lerp(camC.p, kBC));
      camera.lookAt(camA.l.clone().lerp(camB.l, kAB).lerp(camC.l, kBC));
      camera.fov = 35 - 1.2 * tens; camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);

      // ---------- creatures ----------
      const home = prog(t, T.move, T.move + 1.0, easeInOutCubic);
      const shiverQ = win(t, T.iceQ, T.iceQ + T.iceDur + 0.3, 0.15), shiverE = win(t, T.iceE, T.iceE + T.iceDur + 0.3, 0.15);
      const hop = (t0) => Math.max(0, Math.sin(clamp((t - t0) / 0.42) * Math.PI)) * 0.2;
      let rk = -1; for (let k = 0; k < READERS.length; k++) if (t >= T.att[k]) rk = k;
      let focusX = lerp(X0, X1, prog(t, 54.05, 54.5));
      if (t > T.connA) focusX = lerp(X0, X1, prog(t, T.connA, T.connB));
      if (t > T.mA) focusX = lerp(-3.7, 3.7, clamp((t - T.mA) / (T.mB - T.mA)));
      if (rk >= 0) focusX = eTiles[READERS[rk][1]].userData.cx;
      const focus = V3(focusX, (QY + EY) / 2, RZ);
      for (const [c, s] of [[q, -1], [e, 1]]) {
        const P = c.pose, isQ = s < 0, shiver = isQ ? shiverQ : shiverE;
        const x = lerp(s * AX, s * 2.3, home);
        P.position.set(x + Math.sin(t * 61 + (isQ ? 0 : 1.3)) * 0.018 * shiver, hop(isQ ? 59.875 : 60.0), lerp(AZ, 0, home));
        P.yaw = s * -lerp(0.75, 0.5, home);
        P.squash = 1 + 0.02 * bp - 0.05 * tens - 0.08 * Math.max(0, kick(t, isQ ? 59.875 : 60.0, 2.5, 6));
        P.sway = 0.09 * (1 - 0.6 * shiver) + 0.04 * tens; P.swayPhase = t * (2.2 + tens * 2.5) + (isQ ? 0 : 1.1);
        P.lean = s * (0.04 * Math.sin(t * 0.9 + (isQ ? 0 : 2)) + 0.03 * shiver * Math.sin(t * 47 + (isQ ? 0 : 1)));
        P.leanFwd = 0.04 * tens;
        P.headTilt = 0.08 * Math.sin(t * 1.3 + (isQ ? 0 : 1)) * (1 - shiver) + 0.06 * shiver * Math.sin(t * 53);
        const t0i = isQ ? T.iceQ : T.iceE;
        const lookBody = win(t, t0i + 0.05, t0i + 1.0, 0.2);
        P.headNod = 0.3 * lookBody;
        const oxp = -s * lerp(AX, 2.3, home), ozp = lerp(AZ, 0, home), otherHead = V3(oxp, 4.85, ozp);
        let look = focus;
        if (t > 57.9) look = otherHead;
        if (lookBody > 0.05) look = V3(x, 4.2 * clamp((t - t0i) / T.iceDur, 0, 1) * 0.85, 0.6);
        if (t > 61.0) { const ck = cards.find((cd) => t >= cd.s && t < cd.arrive); look = ck ? V3(-3, 1.2, 2.5) : otherHead; }
        if (t > 64.6) look = camera.position.clone().lerp(otherHead, 0.6);
        P.lookAt = look.clone();
        P.blink = Math.max(blinkAt(t, isQ ? [55.3, 57.6, 60.4, 62.8] : [55.7, 57.75, 60.6, 63.3]), 0.45 * shiver);
        P.happy = Math.max(win(t, 59.85, 60.9, 0.15), isQ ? win(t, 56.6, 57.3, 0.12) : 0, win(t, 65.0, 66.2, 0.15));
        P.crest = 1 + 0.5 * tens;
        P.xray = lerp(0.3, 0.55, prog(t, 57.9, 58.6)) + 0.0 * prog(t, 60.9, 61.6);
        P.wave = wave;
        P.glow = 1 + 0.25 * prog(t, 60.6, 61.6);
        P.coreGlow = (1 + 0.25 * tens) * (1 - 0.3 * prog(t, t0i, t0i + 1.0));
        P.coreFlow = 1 + 1.5 * tens;
        P.portGlow = 0.9 + 0.5 * prog(t, 56.0, 57.0) - 0.6 * win(t, 58.1, 59.3, 0.3) - 0.3 * prog(t, 61, 62) + 0.1 * tens;
        P.ice = t > t0i ? 1 : 0;
        P.lora = t > T.loraA ? 1 : 0;
        // the partner's writes rise with the wave (lockstep), alternating like the zig-zag in a2
        c.inject = [];
        if (wave > -0.5) XBLOCKS.forEach((b, k) => { if (UB[k] < wave) c.inject.push({ block: b, color: isQ ? C.ember : C.quill, amount: 0.7, top: Math.min(1.01, wave) }); });
      }
      // a card arriving lights both cores at the base (its prompts entering the tails)
      for (const cd of cards) {
        const f = t > cd.arrive ? Math.exp(-(t - cd.arrive) * 4) : 0;
        if (f > 0.02) for (const c of [q, e]) if (c.inject.length < 15) c.inject.push({ block: 0, color: cd.col, amount: 0.9 * f, top: 0.12 + 0.3 * (1 - f) });
      }
      q.update(t); e.update(t);
      for (const c of [q, e]) { c.iceMat.opacity = (0.28 - 0.1 * prog(t, 61, 62)) * c.pose.ice; c.iceMat.emissiveIntensity = 0.06; c.iceMat.color.set('#8cc4ee'); c.loraMat.opacity = c.pose.lora * (0.55 + 0.15 * Math.sin(t * 9 + (c === q ? 0 : 1))) * (1 - 0.4 * prog(t, 61, 61.8)); }

      // ---------- link ----------
      const L = link.state;
      L.depth = 1; L.grow = 1;
      const base = 0.12 + 0.6 * prog(t, 56.0, 57.0) + 0.2 * prog(t, 59.0, 59.6);
      L.intensity = wave > -0.5 ? UB.map((u) => base * 0.55 + 0.9 * Math.exp(-Math.pow((u - wave) * 10, 2)) + 0.4 * tens) : base;
      L.flow = 1 + 2.5 * tens; L.packets = 3 + Math.round(3 * tens); L.packetSize = 0.16 + 0.05 * tens;
      L.pulses = [];
      if (wave > -0.5) UB.forEach((u, k) => { const d = wave - u; if (d > 0 && d < 0.09) { const uu = d / 0.09; L.pulses.push({ e: k, dir: 0, u: uu, amount: 1.8, color: C.quillHi }, { e: k, dir: 1, u: uu, amount: 1.8, color: C.emberHi }); } });
      link.update(t);

      // ---------- part 1: rows ----------
      const rowsVis = 1 - prog(t, T.rowsOut, T.rowsOut + 0.35);
      rows.visible = false; // v10: the fitted-start beat (token rows, 54-58) is cut; the shot starts at 58.0
      if (rows.visible) {
        const disp = prog(t, T.rowsOut, T.rowsOut + 0.6, easeInCubic);
        const popT = (u, isQ) => (isQ ? 54.04 + u.i * 0.03 : 54.14 + u.i * 0.035);
        for (const [list, isQ] of [[qTiles, true], [eTiles, false]]) list.forEach((tile) => {
          const u = tile.userData, p0 = popT(u, isQ);
          const k = spring(t, p0, 2.8, 0.45), a = prog(t, p0, p0 + 0.15);
          const dx = u.cx * 0.5 * disp, dy = disp * (0.8 + hash(u.i + (isQ ? 0 : 50)) * 1.2) * (isQ ? 1 : -0.35), dz = disp * 1.5 * hash(u.i * 3 + (isQ ? 0 : 9));
          tile.position.set(u.cx + dx, u.y + (1 - Math.min(1, k)) * (isQ ? 0.3 : -0.3) + dy + 0.015 * Math.sin(t * 2 + u.i) * (1 - prog(t, 56.9, 57.1)), RZ + dz);
          tile.scale.setScalar(Math.max(0.001, k) * (1 - 0.6 * disp));
          tile.rotation.set(0, 0, disp * (hash(u.i * 7 + (isQ ? 1 : 2)) - 0.5) * 1.5);
          u.mat.opacity = a * rowsVis;
          let o = 0, col = GOLD;
          const mi = isQ ? qMatched.has(u.i) : eMatched.has(u.i);
          if (mi) { const cn = conns.find((cc) => (isQ ? cc.qi : cc.ei) === u.i); o = t > cn.tEnd ? Math.exp(-(t - cn.tEnd) * 3.5) * 0.9 + 0.18 : 0; }
          if (rk >= 0) {
            const [rq, re] = READERS[rk];
            if (isQ && u.i === rq) { o = 0.95; col = C.quillHi; }
            if (!isQ && u.i === re) o = Math.max(o, 0.75 + 0.25 * Math.exp(-(t - T.att[rk]) * 5));
          }
          u.omat.opacity = o * rowsVis; u.omat.color.copy(col);
        });
        for (const cn of conns) {
          const g = prog(t, cn.tEnd, cn.tEnd + 0.14, easeOutCubic), ym = (cn.y0 + cn.y1) / 2;
          cn.line.position.set(cn.x, ym, RZ + 0.005); cn.line.scale.set(1 + 0.6 * (t > cn.tEnd ? Math.exp(-(t - cn.tEnd) * 6) : 0), Math.max(0.001, (cn.y0 - cn.y1) * g), 1);
          const lo = (t > cn.tEnd ? 1 : 0) * rowsVis * (1 - prog(t, T.rowsOut - 0.1, T.rowsOut + 0.2));
          cn.line.material.opacity = lo * (0.85 + 0.15 * Math.sin(t * 8 + cn.x * 3));
          cn.capA.material.opacity = cn.capB.material.opacity = lo; cn.capA.scale.setScalar(0.16 * lo); cn.capB.scale.setScalar(0.16 * lo);
          const f = t > cn.tEnd + 0.08 ? Math.exp(-(t - cn.tEnd - 0.08) * 5) : 0;
          cn.flash.position.set(cn.x, ym, RZ + 0.02); cn.flash.scale.setScalar(0.8 * f + 0.001); cn.flash.material.opacity = f;
        }
        for (const ms of misses) {
          const a = prog(t, ms.tEnd, ms.tEnd + 0.2) * rowsVis * (1 - prog(t, T.rowsOut - 0.1, T.rowsOut + 0.2));
          ms.stub.material.opacity = 0.55 * a;
          ms.mark.material.opacity = 0.9 * a; ms.mark.scale.setScalar(0.16 * (1 + 0.4 * Math.max(0, kick(t, ms.tEnd, 3, 6))) * Math.max(0.001, a));
        }
        const mU = clamp((t - T.mA) / (T.mB - T.mA)), mVis = win(t, T.mA - 0.05, T.mB + 0.1, 0.18);
        mGroup.position.set(lerp(-3.7, 3.7, mU), (QY + EY) / 2, RZ + 0.12); mGroup.rotation.set(0, 0.42 * Math.sin(mU * Math.PI) - 0.21, 0);
        mMat.opacity = 0.95 * mVis; mEdge.material.opacity = 0.9 * mVis; mGlow.material.opacity = 0.35 * mVis; mGroup.visible = mVis > 0.001;
        conns.forEach((cn, m) => {
          const qt = qTiles[cn.qi].userData, et = eTiles[cn.ei].userData;
          const tm = T.mA + (T.mB - T.mA) * ((et.cx + 3.7) / 7.4) - 0.14, u = clamp((t - tm) / 0.32);
          const live = t > tm && t < tm + 0.32;
          orbs[m].position.set(lerp(et.cx, qt.cx, smooth(u)), lerp(EY + TH / 2, QY - TH / 2, smooth(u)), RZ + 0.06);
          orbs[m].scale.setScalar(live ? 0.3 : 0.001); orbs[m].material.opacity = live ? 1 : 0;
          orbs[m].material.color.copy(C.ember).lerp(C.quillHi, smooth((u - 0.35) / 0.3));
          const dkf = t > tm + 0.32 ? Math.exp(-(t - tm - 0.32) * 4) : 0;
          docks[m].position.set(qt.cx, QY - TH / 2, RZ + 0.05); docks[m].scale.setScalar(0.7 * dkf + 0.001); docks[m].material.opacity = dkf;
          orbsQ[m].position.set(lerp(qt.cx, et.cx, smooth(u)), lerp(QY - TH / 2, EY + TH / 2, smooth(u)), RZ + 0.07);
          orbsQ[m].scale.setScalar(live ? 0.3 : 0.001); orbsQ[m].material.opacity = live ? 1 : 0;
          orbsQ[m].material.color.copy(C.quill).lerp(C.emberHi, smooth((u - 0.35) / 0.3));
          docksE[m].position.set(et.cx, EY + TH / 2, RZ + 0.05); docksE[m].scale.setScalar(0.7 * dkf + 0.001); docksE[m].material.opacity = dkf;
        });
        const aVis = (rk >= 0 ? 1 : 0) * (1 - prog(t, T.rowsOut - 0.1, T.rowsOut + 0.15));
        eTiles.forEach((tile, j) => {
          const et = tile.userData;
          let w = 0, wp = 0; if (rk >= 0) { w = attW[rk][j]; wp = rk > 0 ? attW[rk - 1][j] : 0; }
          const ww = lerp(wp, w, rk >= 0 ? easeOutBack(clamp((t - T.att[rk]) / 0.2), 1.6) : 0);
          const isPeak = rk >= 0 && READERS[rk][1] === j;
          const bh = 0.34 * ww + 0.004;
          bars[j].position.set(et.cx, EY - TH / 2 - 0.07 - bh / 2, RZ); bars[j].scale.set(Math.min(0.12, et.w * 0.5), bh, 1);
          bars[j].material.color.copy(isPeak ? GOLD : C.quillHi); bars[j].material.opacity = aVis * (isPeak ? 1 : 0.6);
          if (rk >= 0) {
            const rq = qTiles[READERS[rk][0]].userData;
            placeBeam(beams[j], V3(rq.cx, QY - TH / 2, RZ - 0.01), V3(et.cx, EY + TH / 2, RZ - 0.01), isPeak ? 0.06 : 0.008 + 0.03 * w);
            beams[j].material.color.copy(isPeak ? GOLD : C.quillHi); beams[j].material.opacity = aVis * prog(t, T.att[rk], T.att[rk] + 0.1) * (isPeak ? 0.95 : 0.25 + w);
          } else beams[j].material.opacity = 0;
        });
        qTiles.forEach((tile, j) => {
          const qt = tile.userData;
          let w = 0, wp = 0; if (rk >= 0) { w = attWE[rk][j]; wp = rk > 0 ? attWE[rk - 1][j] : 0; }
          const ww = lerp(wp, w, rk >= 0 ? easeOutBack(clamp((t - T.att[rk]) / 0.2), 1.6) : 0);
          const isPeak = rk >= 0 && READERS[rk][0] === j;
          const bh = 0.34 * ww + 0.004;
          barsE[j].position.set(qt.cx, QY + TH / 2 + 0.07 + bh / 2, RZ); barsE[j].scale.set(Math.min(0.12, qt.w * 0.5), bh, 1);
          barsE[j].material.color.copy(isPeak ? GOLD : C.emberHi); barsE[j].material.opacity = aVis * (isPeak ? 1 : 0.6);
          if (rk >= 0) {
            const re = eTiles[READERS[rk][1]].userData;
            placeBeam(beamsE[j], V3(re.cx + 0.05, EY + TH / 2, RZ - 0.02), V3(qt.cx + 0.05, QY - TH / 2, RZ - 0.02), isPeak ? 0.05 : 0.008 + 0.03 * w);
            beamsE[j].material.color.copy(isPeak ? GOLD : C.emberHi); beamsE[j].material.opacity = aVis * prog(t, T.att[rk], T.att[rk] + 0.1) * (isPeak ? 0.9 : 0.25 + w);
          } else beamsE[j].material.opacity = 0;
        });
      }

      // ---------- part 2: frost and LoRA ----------
      const loraFront = (t - T.loraA) / T.loraDur * 38 - 1;
      const melt = 1 - prog(t, 64.75, 65.5); // the crystals settle into the plain frost shell before the cut (a5b continues with the shell)
      for (const fx of iceFx) {
        const c = fx.c, t0 = c === q ? T.iceQ : T.iceE, front = (t - t0) / T.iceDur * 38 - 1, Mw = c.group.matrixWorld;
        for (let i = 0; i < NBLOCKS; i++) {
          const k = clamp((front - i) / 2.5), im = c.iceMeshes[i];
          if (k <= 0) im.visible = false; else im.scale.multiplyScalar(lerp(0.4, 1, easeOutBack(k, 2.2)));
          const kl = clamp((loraFront - i) / 2), lm = c.loraMeshes[i];
          if (kl <= 0) lm.visible = false; else lm.scale.set(1.4 * easeOutBack(kl, 2.5), 1.8 * easeOutBack(kl, 2.5), 1);
        }
        for (let k = 0; k < NC; k++) {
          const sp = fx.spec[k], g = clamp((front - sp.b) / 2.2), r = c.blockRadius(sp.b) * 1.02;
          const lp = c._centers[sp.b].clone().add(V3(Math.sin(sp.a) * r, sp.dy, Math.cos(sp.a) * r));
          const dir = V3(Math.sin(sp.a), sp.tilt, Math.cos(sp.a)).normalize(); tmpQ.setFromUnitVectors(UP, dir);
          const sc = g <= 0 || melt <= 0 ? 0.0001 : sp.s * easeOutBack(g, 2.5) * melt;
          tmpM.compose(lp.add(dir.clone().multiplyScalar(0.04 * sc)), tmpQ, V3(sc, sc, sc)).premultiply(Mw); fx.inst.setMatrixAt(k, tmpM);
        }
        fx.inst.instanceMatrix.needsUpdate = true; fx.inst.visible = front > 0;
        const pa = fx.sparks.geometry.attributes.position.array, ca = fx.sparks.geometry.attributes.color.array;
        for (let k = 0; k < NS; k++) {
          const sp = fx.sspec[k], tb = t0 + (sp.b + 1) / 38 * T.iceDur + sp.d, r = c.blockRadius(sp.b) * 1.18;
          const w = c._centers[sp.b].clone().add(V3(Math.sin(sp.a) * r, sp.dy, Math.cos(sp.a) * r)).applyMatrix4(Mw);
          pa[k * 3] = w.x; pa[k * 3 + 1] = w.y; pa[k * 3 + 2] = w.z;
          const b = t > tb ? 1.3 * Math.exp(-(t - tb) * 5) + 0.45 * Math.pow(Math.max(0, Math.sin(t * sp.f + k)), 14) * (1 - 0.6 * prog(t, 61, 63)) : 0;
          ca[k * 3] = 0.75 * b; ca[k * 3 + 1] = 0.9 * b; ca[k * 3 + 2] = b;
        }
        fx.sparks.geometry.attributes.position.needsUpdate = true; fx.sparks.geometry.attributes.color.needsUpdate = true; fx.sparks.visible = t > t0;
        fx.front.position.copy(c.blockWorld(Math.round(clamp(front, 0, 35))));
        fx.front.material.opacity = front > 0 && front < 37 ? 0.7 : 0; fx.front.scale.setScalar(1.6 + 0.3 * Math.sin(t * 20));
        const la = fx.lflash.geometry.attributes.position.array, lc = fx.lflash.geometry.attributes.color.array;
        for (let i = 0; i < NBLOCKS; i++) {
          const w = c.loraMeshes[i].position.clone().applyMatrix4(Mw);
          la[i * 3] = w.x; la[i * 3 + 1] = w.y; la[i * 3 + 2] = w.z;
          const ti = T.loraA + (i + 1) / 38 * T.loraDur, f = t > ti ? 1.2 * Math.exp(-(t - ti) * 6) + 0.25 * win(t, T.loraLbl[0] - 0.1, T.loraLbl[1], 0.2) : 0;
          lc[i * 3] = C.link.r * f; lc[i * 3 + 1] = C.link.g * f; lc[i * 3 + 2] = C.link.b * f;
        }
        fx.lflash.geometry.attributes.position.needsUpdate = true; fx.lflash.geometry.attributes.color.needsUpdate = true; fx.lflash.visible = t > T.loraA;
      }

      // ---------- part 3: cards stream in as data ----------
      const tailQ = q.tailWorld(), tailE = e.tailWorld();
      const gate = tailQ.clone().lerp(tailE, 0.5).add(V3(0, 0.55, 0.9)); // the cards land between the two tails
      const gs = toScreen(gate, camera);
      deck.hideAll();
      for (const cd of cards) {
        const u = (t - cd.s) / (cd.arrive - cd.s);
        if (deck.is3D && u >= 0 && u < 1) {
          // a glass game window falls from above into the gate between the tails, shrinking, tumbling less as it lands
          const p0 = gate.clone().add(V3(cd.start.x, cd.start.y, cd.start.z)), ctrl = p0.clone().lerp(gate, 0.45).add(V3(-cd.arc * 0.012, 0.6, 0.3));
          const ue = easeInCubic(u) * 0.7 + u * 0.3, v = 1 - ue;
          const pos = p0.clone().multiplyScalar(v * v).addScaledVector(ctrl, 2 * ue * v).addScaledVector(gate, ue * ue);
          const quat = camera.quaternion.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.35 * (1 - u) * Math.sin(cd.k * 1.7), 0.5 * (1 - u) * Math.cos(cd.k * 2.3), cd.rot * (Math.PI / 180) * (1 - u))));
          const sc = lerp(0.28, 0.04, easeInCubic(u)), op = Math.min(1, u / 0.1) * (1 - prog(u, 0.85, 1));
          deck.show(cd, pos, quat, sc, op, t >= cd.s ? Math.exp(-(t - cd.s) * 6) * 0.6 : 0);
        } else if (!deck.is3D && u >= 0 && u < 1) {
          const ue = easeInCubic(u) * 0.75 + u * 0.25;
          const cx = lerp(cd.start.x, gs.x, ue) - Math.sin(Math.PI * u) * cd.arc * 1.5, cy = lerp(cd.start.y, gs.y, easeInCubic(u) * 0.4 + u * 0.6);
          const sc = lerp(deck.s0, deck.s1, easeInCubic(u)), op = Math.min(1, u / 0.08) * (1 - prog(u, 0.82, 1));
          deck.show(cd, t, `translate(${cx - deck.W / 2}px, ${cy - deck.H / 2}px) rotate(${cd.rot * (1 - u)}deg) scale(${sc})`, op);
        }
        // on arrival, two sparks run into the tails: Quill's prompt and Ember's prompt
        const v = clamp((t - cd.arrive) / 0.28), on = t > cd.arrive && t < cd.arrive + 0.28;
        cd.spQ.position.copy(gate.clone().lerp(tailQ, easeOutCubic(v))); cd.spE.position.copy(gate.clone().lerp(tailE, easeOutCubic(v)));
        for (const sp of [cd.spQ, cd.spE]) { sp.scale.setScalar(on ? 0.5 : 0.001); sp.material.opacity = on ? 1 : 0; }
      }
      const step = stepAt(t);
      const cA = win(t, 61.3, 66.4, 0.35);
      const pulse = cards.reduce((m, cd) => Math.max(m, t > cd.arrive ? Math.exp(-(t - cd.arrive) * 6) : 0), 0);
      chart.update(step, cA, pulse * 0.6 + 0.4 * tens * bp);
      chart.root.style.transform = `translateX(${(1 - easeOutCubic(prog(t, 61.3, 61.8))) * 60}px)`;

      // ---------- HUD in holds ----------
      {
        const o = win(t, T.stat[0], T.stat[1], 0.18), dy = (1 - easeOutCubic(prog(t, T.stat[0], T.stat[0] + 0.35))) * -20;
        stat.style.display = 'none'; void o; stat.style.opacity = o; stat.style.transform = `translate(0px, ${22 + dy}px)`;
      }
      { const a = 0; capRows.style.display = 'none'; capRows.style.opacity = a;
        const b = 0; capLines.style.display = 'none'; capLines.style.opacity = b; }
      loraLbl.update(q.blockWorld(17).add(V3(-0.75, 0, 0.2)), camera, win(t, T.loraLbl[0], T.loraLbl[1], 0.18));

      return {
        bloomStrength: 0.72 + 0.06 * tens + 0.06 * tens * bp,
        bloomRadius: 0.5, bloomThreshold: 0.24,
        vignette: 0.9 + 0.15 * tens,
        aberration: lerp(0.0018, 0.0006, prog(t, 60.9, 61.8)),
        flash: 0.1 * (t > T.iceQ ? Math.exp(-(t - T.iceQ) * 6) : 0), flashColor: '#cfeaff',
      };
    },
  };
}
