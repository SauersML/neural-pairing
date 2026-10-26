// S4 "04 / TRAINING" (36–46 s): the fitted start, frozen weights + LoRA + held gates, and the curriculum.
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Chapter, Pin, el, toScreen, FONT } from '../lib/hud.js';
import { C, CSS } from '../lib/palette.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, smoother, easeOutBack, easeOutCubic, easeInCubic, easeInOutCubic, easeOutExpo, hash, noise1 } from '../lib/util.js';
import { tileTexture, outlineTexture, starTexture, matrixTexture, lockTexture, crossTexture, cardTexture } from './s4_train_tex.js';

const GOLD = new THREE.Color('#ffd166');
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

// ---------------- layout of the token rows (character-aligned) ----------------
// Only part of the sentence is shown so the tiles stay large: characters 8..30 = " Phraya, 5203 lanterns"
// (real splits from data/tokens.json: Quill " Ph|ray|a|,| |5|2|0|3| lantern|s", Ember " Ph|raya|,| |520|3| lantern|s").
const C0 = 8, C1 = 30;
const CW = 0.28, TH = 0.5, GAP = 0.035;
const QY = 4.0, EY = 3.0, RZ = 1.0;
const xAt = (ch) => (ch - (C0 + C1) / 2) * CW; // x of a character boundary
const X0 = xAt(C0), X1 = xAt(C1);
const AX = 5.0, AZ = -2.0; // creature placement while they read (part A)

// ---------------- timeline (global seconds) ----------------
const T = {
  connA: 36.75, connB: 37.45,
  mA: 37.5, mB: 38.6,
  att: [38.62, 38.9, 39.18, 39.46],
  rowsOut: 39.95,
  move: 39.65,
  iceQ: 40.0, iceE: 40.15, iceDur: 1.3,
  loraA: 40.95, loraDur: 0.8,
  lockA: 41.5,
  cur: 43.0, warmA: 43.15, warmB: 44.55, mix: 44.6, end: 46.0,
};

export async function create({ hud, def }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  const T0 = def.start;

  const tokData = await (await fetch('data/tokens.json')).json();
  const dot = Creature.dotTexture();
  const star = starTexture();

  // ================= PART A: token rows =================
  const rows = new THREE.Group(); scene.add(rows);
  const mkTiles = (list, kind, y) => list.map((tk, i) => {
    const w = (tk.end - tk.start) * CW - GAP;
    const { tex } = tileTexture(tk.text, kind, w, TH);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false, opacity: 0 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, TH), mat);
    const omat = new THREE.MeshBasicMaterial({ map: outlineTexture(kind, w, TH), transparent: true, depthWrite: false, toneMapped: false, opacity: 0, blending: THREE.AdditiveBlending, color: GOLD.clone() });
    const outline = new THREE.Mesh(new THREE.PlaneGeometry(w * 1.04 + 0.02, TH * 1.08), omat);
    const g = new THREE.Group(); g.add(mesh); g.add(outline); outline.position.z = 0.002;
    const cx = (xAt(tk.start) + xAt(tk.end)) / 2;
    g.userData = { cx, y, i, w, tk, mat, omat };
    rows.add(g); return g;
  });
  const inWin = (tk) => tk.start >= C0 && tk.end <= C1;
  const qList = tokData.quill.filter(inWin), eList = tokData.ember.filter(inWin);
  const qTiles = mkTiles(qList, 'quill', QY);
  const eTiles = mkTiles(eList, 'ember', EY);
  // the real matches (same end character), re-indexed into the shown window
  const matches = tokData.matches_quill_ember.map(([a, b]) => [qList.indexOf(tokData.quill[a]), eList.indexOf(tokData.ember[b])]).filter(([a, b]) => a >= 0 && b >= 0);
  const qMatched = new Map(matches.map(([a, b]) => [a, b]));
  const eMatched = new Map(matches.map(([a, b]) => [b, a]));

  // connectors (gold) at the shared end character
  const addMat = (col, op = 1) => new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  const spriteMat = (map, col) => new THREE.SpriteMaterial({ map, color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  const conns = matches.map(([qi, ei]) => {
    const end = qList[qi].end, x = xAt(end) - GAP / 2;
    const y0 = QY - TH / 2, y1 = EY + TH / 2;
    const line = new THREE.Mesh(new THREE.PlaneGeometry(0.04, 1), addMat(GOLD.clone()));
    const capA = new THREE.Sprite(spriteMat(dot, GOLD.clone())), capB = new THREE.Sprite(spriteMat(dot, GOLD.clone()));
    const flash = new THREE.Sprite(spriteMat(star, new THREE.Color('#fff3c4')));
    for (const o of [line, capA, capB, flash]) rows.add(o);
    capA.position.set(x, y0, RZ + 0.01); capB.position.set(x, y1, RZ + 0.01);
    return { qi, ei, end, x, y0, y1, line, capA, capB, flash, tEnd: T.connA + (T.connB - T.connA) * ((end - C0) / (C1 - C0)) };
  });
  // ends with no partner: a short grey stub and a red x
  const xTex = crossTexture();
  const misses = qList.map((tk, i) => [tk, i]).filter(([, i]) => !qMatched.has(i)).map(([tk, i]) => {
    const x = xAt(tk.end) - GAP / 2;
    const stub = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.2), addMat(new THREE.Color('#8b8fa3'), 0));
    stub.position.set(x, QY - TH / 2 - 0.1, RZ);
    const mark = new THREE.Sprite(spriteMat(xTex, new THREE.Color('#ffffff'))); mark.position.set(x, QY - TH / 2 - 0.3, RZ + 0.01);
    rows.add(stub, mark);
    return { i, x, stub, mark, tEnd: T.connA + (T.connB - T.connA) * ((tk.end - C0) / (C1 - C0)) };
  });

  // the ridge map M: a glass slab that slides between the rows
  const mGroup = new THREE.Group(); rows.add(mGroup);
  const mH = (QY - TH / 2) - (EY + TH / 2) - 0.1, mW = mH * 1.6;
  const mMat = new THREE.MeshBasicMaterial({ map: matrixTexture(), transparent: true, depthWrite: false, toneMapped: false, opacity: 0, side: THREE.DoubleSide });
  const mPlane = new THREE.Mesh(new THREE.PlaneGeometry(mW, mH), mMat); mGroup.add(mPlane);
  const mEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(mW, mH, 0.12)), new THREE.LineBasicMaterial({ color: C.linkHi, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  mGroup.add(mEdge);
  const mGlow = new THREE.Sprite(spriteMat(dot, C.link.clone())); mGlow.scale.set(mW * 2.4, mH * 2.4, 1); mGroup.add(mGlow);
  // mapped states: one orb per matched pair, rising from Ember's token through M into Quill's space
  const orbs = matches.map(() => { const s = new THREE.Sprite(spriteMat(dot, C.ember.clone())); rows.add(s); return s; });
  const docks = matches.map(() => { const s = new THREE.Sprite(spriteMat(star, C.quillHi.clone())); rows.add(s); return s; });

  // attention: beams from the reader (a Quill token) to every Ember token, bars under Ember's row
  const beams = eTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), addMat(C.quillHi.clone(), 0)); rows.add(m); return m; });
  const bars = eTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), addMat(C.quillHi.clone(), 0)); rows.add(m); return m; });
  const barBase = eTiles.map(() => { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 0.012), addMat(new THREE.Color('#ffffff'), 0)); rows.add(m); return m; });
  const READERS = [[0, 0], [2, 1], [7, 4], [9, 6]]; // [quill reader, matched ember token]: Ph→Ph, a→raya, 0→520, lantern→lantern
  const attW = READERS.map(([qi, ei], k) => {
    const w = eTiles.map((_, j) => (j === ei ? 0 : 0.2 + hash(j * 7.1 + k * 3.3) * (Math.abs(j - ei) <= 1 ? 1.4 : 0.6)));
    const s = w.reduce((a, b) => a + b, 0); const peak = [0.93, 0.89, 0.91, 0.95][k];
    return w.map((v, j) => (j === ei ? peak : v / s * (1 - peak)));
  });

  // ================= PART B: ice, LoRA, locks =================
  const iceCrystalMat = new THREE.MeshStandardMaterial({ color: '#d9f1ff', emissive: C.ice.clone(), emissiveIntensity: 0.2, roughness: 0.08, metalness: 0.1, transparent: true, opacity: 0.85, flatShading: true });
  const crystalGeo = new THREE.OctahedronGeometry(0.06, 0); crystalGeo.scale(0.7, 2.4, 0.7);
  const NC = 70, NS = 90;
  const iceFx = [q, e].map((c, ci) => {
    const inst = new THREE.InstancedMesh(crystalGeo, iceCrystalMat, NC); inst.frustumCulled = false; scene.add(inst);
    const spec = Array.from({ length: NC }, (_, k) => ({ b: 1 + Math.floor(hash(k * 3.7 + ci * 11) * 34), a: (hash(k * 5.3 + ci) - 0.5) * Math.PI * 1.7, dy: (hash(k * 9.1 + ci) - 0.5) * 0.1, s: 0.6 + hash(k * 2.9 + ci * 4) * 0.9, tilt: (hash(k * 1.9 + ci) - 0.2) * 0.9 }));
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 3), 3)); sg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NS * 3), 3));
    const sparks = new THREE.Points(sg, new THREE.PointsMaterial({ size: 0.2, map: star, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })); sparks.frustumCulled = false; scene.add(sparks);
    const sspec = Array.from({ length: NS }, (_, k) => ({ b: Math.floor(hash(k * 4.3 + ci * 7) * 36), a: (hash(k * 6.1 + ci * 2) - 0.5) * Math.PI * 1.8, dy: (hash(k * 8.7 + ci) - 0.5) * 0.12, d: hash(k * 3.1 + ci) * 0.25, f: 2 + hash(k * 7.7 + ci) * 5 }));
    const front = new THREE.Sprite(spriteMat(dot, C.ice.clone())); scene.add(front);
    // LoRA flash points
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NBLOCKS * 3), 3)); lg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NBLOCKS * 3), 3));
    const lflash = new THREE.Points(lg, new THREE.PointsMaterial({ size: 0.3, map: dot, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })); lflash.frustumCulled = false; scene.add(lflash);
    return { c, inst, spec, sparks, sspec, front, lflash };
  });
  const lockOpen = lockTexture(false), lockClosed = lockTexture(true);
  const locks = [];
  for (const [ci, c] of [[0, q], [1, e]]) XBLOCKS.forEach((b, k) => {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: lockOpen, transparent: true, depthWrite: false, toneMapped: false, opacity: 0 }));
    const ring = new THREE.Sprite(spriteMat(dot, C.ice.clone()));
    scene.add(sp, ring); locks.push({ ci, c, b, k, sp, ring });
  });

  // ================= PART C: curriculum =================
  // a value enters the sender's prompt: a blank gold value tile (no text to read)
  const gameTex = [cardTexture('\u2003\u2003\u2003', { color: '#ffd166', bg: 'rgba(120,84,0,0.92)', text: '#fff7d6', icon: false, font: 60 })];
  const checkTex = cardTexture('✓', { color: '#ffd166', bg: 'rgba(56,38,0,0.9)', text: '#ffe7a3', icon: false, font: 76 });
  const NCYC = 14;
  const cyc = Array.from({ length: NCYC }, (_, k) => {
    const s = T.warmA + (T.warmB - T.warmA - 0.45) * Math.pow(k / (NCYC - 1), 0.85);
    const dir = k % 2; // 0: Quill's prompt -> Ember writes; 1: Ember's prompt -> Quill writes
    const tx = gameTex[0];
    const inM = new THREE.Mesh(new THREE.PlaneGeometry(0.42 * tx.userData.aspect, 0.42), new THREE.MeshBasicMaterial({ map: tx, transparent: true, depthWrite: false, toneMapped: false }));
    const outM = new THREE.Mesh(new THREE.PlaneGeometry(0.5 * checkTex.userData.aspect, 0.5), new THREE.MeshBasicMaterial({ map: checkTex, transparent: true, depthWrite: false, toneMapped: false }));
    scene.add(inM, outM);
    return { s, dir, inM, outM };
  });
  const NG = NCYC * 8 * 2;
  const gg = new THREE.BufferGeometry();
  gg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NG * 3), 3)); gg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NG * 3), 3));
  const goldPk = new THREE.Points(gg, new THREE.PointsMaterial({ size: 0.24, map: dot, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
  goldPk.frustumCulled = false; scene.add(goldPk);
  const MIX = ['shared reading', 'play', 'dialogue', 'silent rounds', 'shared pen'];
  const cards = MIX.map((label, k) => {
    const tx = cardTexture(label, { font: 58 });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.74 * tx.userData.aspect, 0.74), new THREE.MeshBasicMaterial({ map: tx, color: new THREE.Color(0.8, 0.8, 0.8), transparent: true, depthWrite: false, toneMapped: false, opacity: 0 }));
    const flare = new THREE.Sprite(spriteMat(star, C.linkHi.clone()));
    scene.add(m, flare); return { m, flare, k };
  });
  const burst = new THREE.Sprite(spriteMat(star, new THREE.Color('#d1fae5'))); scene.add(burst);

  // ================= HUD =================
  const chap = new Chapter(hud, '04', 'Training'); // Chapter is a no-op in lib now
  // a soft dark scrim under the caption so the large type stays legible over the glowing link
  const scrim = el(hud, 'div', { position: 'absolute', left: '0px', right: '0px', bottom: '0px', height: '460px', pointerEvents: 'none', background: 'linear-gradient(180deg, rgba(3,2,8,0) 0%, rgba(3,2,8,0.5) 55%, rgba(3,2,8,0.62) 100%)' });
  hud.insertBefore(scrim, hud.firstChild);
  const paint = (col) => (s) => s.split(' ').map((w) => `<span style="color:${col}">${w}</span>`).join(' '); // one span per word (Caption splits on spaces)
  const gold = paint('#ffd166'), teal = paint(CSS.link), ice = paint(CSS.ice);
  const capA = new Caption(hud, { kicker: 'training', head: `${teal('Fitted')} start`, sub: `Tokens that ${gold('end at the same character')} are paired, and a linear map is fitted between their hidden states.` });
  const capB = new Caption(hud, { head: `${teal('Rank-32 LoRA')} fine-tuning`, sub: `The original weights stay ${ice('frozen')}. The LoRA update and ${teal('the link')} train.` });
  const capC = new Caption(hud, { head: `${gold('Value games')} first, then ${teal('all tasks')}` });
  // the one label of part A: the stat
  const stat = el(hud, 'div', { position: 'absolute', left: '0px', top: '0px', width: '1920px', textAlign: 'center', whiteSpace: 'nowrap', pointerEvents: 'none', textShadow: '0 2px 24px rgba(0,0,0,0.9)' },
    `<span style="font:600 36px ${FONT.text};color:#fff7d6">attention on the matching token:</span>&nbsp;&nbsp;<span style="font:700 64px ${FONT.display};color:#ffd166;letter-spacing:-0.02em">89–97%</span>&nbsp;&nbsp;<span style="font:500 36px ${FONT.text};color:rgba(245,243,255,0.8)">(chance 1.2%)</span>`);
  // step counter (top right)
  const counter = el(hud, 'div', { position: 'absolute', left: '710px', width: '500px', top: '36px', textAlign: 'center', pointerEvents: 'none', textShadow: '0 2px 20px rgba(0,0,0,0.9)' });
  const cKick = el(counter, 'div', { font: `600 34px ${FONT.text}`, color: 'rgba(245,243,255,0.85)' }, 'training step');
  const cNum = el(counter, 'div', { font: `700 84px ${FONT.mono}`, color: '#f5f3ff', letterSpacing: '-0.02em', lineHeight: '1.0', marginTop: '6px', textShadow: '0 0 30px rgba(255,209,102,0.35)' }, '0');
  const cBar = el(counter, 'div', { width: '320px', height: '6px', background: 'rgba(255,255,255,0.12)', margin: '0 auto', marginTop: '12px', borderRadius: '3px', overflow: 'hidden' });
  const cFill = el(cBar, 'div', { width: '0%', height: '100%', background: '#ffd166', boxShadow: '0 0 12px #ffd166' });
  const cLbl = el(counter, 'div', { whiteSpace: 'nowrap', font: `600 30px ${FONT.mono}`, letterSpacing: '0.08em', color: '#ffd166', marginTop: '10px' }, '');

  // ================= helpers =================
  const blinkAt = (t, list) => Math.max(0, ...list.map((b) => win(t, b, b + 0.2, 0.08)));
  const tmpQ = new THREE.Quaternion(), tmpM = new THREE.Matrix4(), UP = V3(0, 1, 0);
  const placeBeam = (m, a, b, w) => { const d = b.clone().sub(a); m.position.copy(a).add(b).multiplyScalar(0.5); m.scale.set(w, d.length(), 1); m.rotation.set(0, 0, Math.atan2(d.y, d.x) - Math.PI / 2); };

  return {
    scene, camera,
    update(t) {
      const lt = t - T0;
      world.update(t);
      const bp = beatPulse(t, 7);
      const tens = prog(t, T.mix, T.end, easeInCubic); // tension ramp into the cut

      // ---------- camera ----------
      const camA = { p: V3(lerp(-0.3, 0.3, prog(t, 36, 40, smooth)), 3.15, lerp(9.5, 9.15, prog(t, 36, 40, smooth))), l: V3(lerp(-0.1, 0.1, prog(t, 36, 40, smooth)), 3.0, 1.0) };
      // B: closer, the camera rises with the frost front, then settles on the ports (locks)
      // (starts low at the tails so the frost is seen above the caption, then rises with it)
      const yB = lerp(0.45, 3.45, prog(t, 40.0, 41.45, smooth)) - 0.4 * prog(t, 41.6, 42.9, smooth);
      const lB = V3(-0.45 + 0.3 * prog(t, 41.6, 42.9, smooth), yB, 0);
      const obB = -0.12 + 0.16 * prog(t, 40, 43);
      const camB = { p: lB.clone().add(V3(Math.sin(obB) * 8.9, 0.45, Math.cos(obB) * 8.9)), l: lB };
      const ang = -0.12 + 0.16 * prog(t, 43, 44.6) + 0.3 * Math.pow(clamp((t - 44.6) / 1.4), 1.6);
      const rad = lerp(13.6, 10.6, prog(t, 43, 46, (x) => x * x * (1.6 - 0.6 * x)));
      const shake = tens * tens * 0.035;
      const camC = { p: V3(Math.sin(ang) * rad + noise1(t * 11, 1) * shake, lerp(3.5, 3.15, prog(t, 43, 46)) + noise1(t * 13, 2) * shake, Math.cos(ang) * rad), l: V3(0, 1.95, 0) };
      const kAB = prog(t, 39.75, 40.95, easeInOutCubic), kBC = prog(t, 42.85, 43.6, easeInOutCubic);
      const cp = camA.p.clone().lerp(camB.p, kAB).lerp(camC.p, kBC), cl = camA.l.clone().lerp(camB.l, kAB).lerp(camC.l, kBC);
      camera.position.copy(cp); camera.lookAt(cl);
      camera.fov = 35 - 2.5 * tens; camera.updateProjectionMatrix();
      camera.updateMatrixWorld(true); // pins project with THIS frame's camera (render would update it only later)

      // ---------- creatures ----------
      const home = prog(t, T.move, T.move + 1.0, easeInOutCubic);
      const shiverQ = win(t, T.iceQ, T.iceQ + T.iceDur + 0.3, 0.15), shiverE = win(t, T.iceE, T.iceE + T.iceDur + 0.3, 0.15);
      const hop = (t0) => Math.max(0, Math.sin(clamp((t - t0) / 0.42) * Math.PI)) * 0.22;
      const crouch = tens * 0.05;
      // where the eyes go in part A: following whatever is happening on the rows
      let focusX = lerp(X0, X1, prog(t, 36.05, 36.7));
      if (t > T.connA) focusX = lerp(X0, X1, prog(t, T.connA, T.connB));
      if (t > T.mA) focusX = lerp(-3.7, 3.7, clamp((t - T.mA) / (T.mB - T.mA)));
      let rk = -1; for (let k = 0; k < READERS.length; k++) if (t >= T.att[k]) rk = k;
      if (rk >= 0) focusX = eTiles[READERS[rk][1]].userData.cx;
      const focus = V3(focusX, (QY + EY) / 2, RZ);
      for (const [c, s] of [[q, -1], [e, 1]]) {
        const P = c.pose, isQ = s < 0;
        const shiver = isQ ? shiverQ : shiverE;
        const x = lerp(s * AX, s * 2.3, home);
        P.position.set(x + Math.sin(t * 61 + (isQ ? 0 : 1.3)) * 0.018 * shiver, hop(isQ ? 41.85 : 41.95) + hop(T.mix + 0.02) * 0.8, lerp(AZ, 0, home));
        P.yaw = s * -lerp(0.75, 0.5, home);
        P.squash = 1 + 0.02 * bp - crouch - 0.08 * Math.max(0, kick(t, isQ ? 41.85 : 41.95, 2.5, 6)) - 0.06 * Math.max(0, -kick(t, T.mix + 0.02, 2.5, 5));
        P.sway = 0.09 * (1 - 0.6 * shiver) + 0.04 * tens; P.swayPhase = t * (2.2 + tens * 2.5) + (isQ ? 0 : 1.1);
        P.lean = s * (0.04 * Math.sin(t * 0.9 + (isQ ? 0 : 2)) + 0.03 * shiver * Math.sin(t * 47 + (isQ ? 0 : 1))) - s * 0.06 * tens * 0;
        P.leanFwd = 0.04 * tens;
        P.headTilt = 0.08 * Math.sin(t * 1.3 + (isQ ? 0 : 1)) * (1 - shiver) + 0.06 * shiver * Math.sin(t * 53);
        // look: rows (A), own frozen body (B), each other / the chips (C)
        const iceFront = isQ ? (t - T.iceQ) / T.iceDur : (t - T.iceE) / T.iceDur;
        const lookBody = win(t, (isQ ? T.iceQ : T.iceE) + 0.1, (isQ ? T.iceQ : T.iceE) + 1.25, 0.25);
        P.headNod = 0.32 * lookBody - 0.1 * win(t, 40.95, 41.8, 0.2);
        // look targets are computed analytically from this frame's poses (never from last frame's matrices)
        const oxp = -s * lerp(AX, 2.3, home), ozp = lerp(AZ, 0, home);
        const otherHead = V3(oxp, 4.85, ozp);
        let look = focus;
        if (t > 39.9) look = otherHead;
        if (lookBody > 0.05) look = V3(x, 4.2 * clamp(iceFront, 0, 1) * 0.85, 0.6);
        if (t > T.cur && t < T.mix) { const k = cyc.findIndex((cy) => t >= cy.s + 0.3 && t < cy.s + 1.0); if (k >= 0) look = V3(cyc[k].dir === 0 ? 2.3 : -2.3, 5.6, 0.3); }
        if (t >= T.mix) look = camera.position.clone().lerp(otherHead, 0.6);
        P.lookAt = look.clone();
        P.faceYaw = 0;
        P.blink = Math.max(blinkAt(t, isQ ? [37.15, 39.9, 42.5, 44.3] : [37.6, 40.05, 42.7, 45.1]), 0.45 * shiver);
        P.happy = Math.max(win(t, 41.85, 42.85, 0.15) * (isQ ? 1 : 1), win(t, T.mix + 0.05, T.mix + 0.6, 0.12), isQ ? win(t, 39.6, 40.15, 0.1) : 0);
        P.crest = 1 + 0.5 * tens + 0.3 * win(t, 41.85, 42.6, 0.2);
        P.glow = 1 + 0.35 * prog(t, 42.6, 43.8) + 0.12 * tens;
        P.coreGlow = (1 + 0.2 * tens + 0.2 * bp * tens) * (1 - 0.4 * prog(t, isQ ? T.iceQ : T.iceE, (isQ ? T.iceQ : T.iceE) + 1.2));
        P.coreFlow = 1 + 1.5 * tens;
        const lockOn = prog(t, T.lockA, T.lockA + 0.6);
        P.portGlow = 0.9 + 0.5 * prog(t, 38.6, 39.6) - 0.8 * win(t, 40.0, 41.4, 0.3) + 0.4 * lockOn * (0.6 + 0.4 * Math.sin(t * 6)) + 0.4 * tens;
        P.ice = t > (isQ ? T.iceQ : T.iceE) ? 1 : 0;
        P.lora = t > T.loraA ? 1 : 0;
        P.blush = 0.35 + 0.3 * shiver;
      }
      q.update(t); e.update(t);
      for (const c of [q, e]) { c.iceMat.opacity = 0.3 * c.pose.ice; c.iceMat.emissiveIntensity = 0.06; c.iceMat.color.set('#8cc4ee'); c.loraMat.opacity = c.pose.lora * (0.5 + 0.15 * Math.sin(t * 9 + (c === q ? 0 : 1))) * (1 - 0.45 * prog(t, 42.9, 43.6)); }

      // ---------- link ----------
      const L = link.state;
      const linkOn = prog(t, 38.62, 39.8);
      L.grow = 1; L.intensity = 0.12 + 0.6 * linkOn - 0.32 * win(t, 40.0, 41.1, 0.35) + 0.3 * prog(t, 40.95, 41.6) - 0.15 * prog(t, 42.9, 43.6) + 0.3 * tens + 0.25 * tens * bp; L.flow = 1 + 2.5 * tens + prog(t, 43, 44.6) * 0.8;
      L.packets = 3 + Math.round(3 * tens); L.packetSize = 0.16 + 0.06 * tens; L.colorBySender = 0.55;
      link.update(t);

      // ---------- part A: rows ----------
      const rowsVis = 1 - prog(t, T.rowsOut, T.rowsOut + 0.35);
      rows.visible = t < T.rowsOut + 0.4;
      if (rows.visible) {
        const disp = prog(t, T.rowsOut, T.rowsOut + 0.6, easeInCubic);
        const popT = (tile, isQ) => (isQ ? 36.06 + tile.userData.i * 0.028 : 36.2 + tile.userData.i * 0.032);
        const tileHi = (tile, isQ) => {
          const u = tile.userData;
          // gold outline when its end connects (flash then settle), violet for the reader, gold for the attended
          let o = 0;
          const mi = isQ ? qMatched.has(u.i) : eMatched.has(u.i);
          if (mi) { const cn = conns.find((cc) => (isQ ? cc.qi : cc.ei) === u.i); o = Math.max(0, Math.exp(-(t - cn.tEnd) * 3.5)) * (t > cn.tEnd ? 1 : 0) * 0.9 + (t > cn.tEnd ? 0.18 : 0); }
          let col = GOLD;
          if (rk >= 0) {
            const [rq, re] = READERS[rk]; const fresh = Math.exp(-(t - T.att[rk]) * 5);
            if (isQ && u.i === rq) { o = 0.95; col = C.quillHi; }
            if (!isQ && u.i === re) { o = Math.max(o, 0.7 + 0.3 * fresh); col = GOLD; }
            o *= 1 - prog(t, 39.85, 40.1);
          }
          u.omat.opacity = o * rowsVis; u.omat.color.copy(col);
        };
        for (const [list, isQ] of [[qTiles, true], [eTiles, false]]) list.forEach((tile) => {
          const u = tile.userData, p0 = popT(tile, isQ);
          const k = spring(t, p0, 2.6, 0.45), a = prog(t, p0, p0 + 0.18);
          const sc = Math.max(0.001, k) * (1 + 0.05 * Math.exp(-(t - 36.9) * 1) * 0);
          // dissolve: drift outward/up, spin a little, fade
          const dx = (u.cx) * 0.5 * disp, dy = disp * (0.8 + hash(u.i + (isQ ? 0 : 50)) * 1.2) * (isQ ? 1 : -0.35), dz = disp * 1.5 * hash(u.i * 3 + (isQ ? 0 : 9));
          tile.position.set(u.cx + dx, u.y + (1 - Math.min(1, k)) * (isQ ? 0.25 : -0.25) + dy + 0.02 * Math.sin(t * 2 + u.i), RZ + dz);
          tile.scale.setScalar(sc * (1 - 0.6 * disp));
          tile.rotation.set(0, 0, disp * (hash(u.i * 7 + (isQ ? 1 : 2)) - 0.5) * 1.5);
          u.mat.opacity = a * rowsVis;
          tileHi(tile, isQ);
        });
        // connectors
        for (const cn of conns) {
          const g = prog(t, cn.tEnd, cn.tEnd + 0.16, easeOutCubic);
          const len = (cn.y0 - cn.y1) * g, ym = (cn.y0 + cn.y1) / 2;
          cn.line.position.set(cn.x, ym, RZ + 0.005); cn.line.scale.set(1 + 0.6 * Math.exp(-(t - cn.tEnd) * 6) * (t > cn.tEnd ? 1 : 0), Math.max(0.001, len), 1);
          const lo = (t > cn.tEnd ? 1 : 0) * rowsVis * (1 - prog(t, T.rowsOut - 0.1, T.rowsOut + 0.2));
          cn.line.material.opacity = lo * (0.85 + 0.15 * Math.sin(t * 8 + cn.x * 3));
          cn.capA.material.opacity = cn.capB.material.opacity = lo; cn.capA.scale.setScalar(0.13 * lo); cn.capB.scale.setScalar(0.13 * lo);
          const f = t > cn.tEnd + 0.1 ? Math.exp(-(t - cn.tEnd - 0.1) * 5) : 0;
          cn.flash.position.set(cn.x, ym, RZ + 0.02); cn.flash.scale.setScalar(0.7 * f + 0.001); cn.flash.material.opacity = f;
        }
        for (const ms of misses) {
          const a = prog(t, ms.tEnd, ms.tEnd + 0.2) * rowsVis * (1 - prog(t, T.rowsOut - 0.1, T.rowsOut + 0.2));
          ms.stub.material.opacity = 0.55 * a;
          const pk = 1 + 0.4 * Math.max(0, kick(t, ms.tEnd, 3, 6));
          ms.mark.material.opacity = 0.85 * a; ms.mark.scale.setScalar(0.12 * pk * Math.max(0.001, a));
        }
        // M sweep
        const mU = clamp((t - T.mA) / (T.mB - T.mA));
        const mx = lerp(-3.7, 3.7, mU);
        const mVis = win(t, T.mA - 0.05, T.mB + 0.12, 0.2);
        mGroup.position.set(mx, (QY + EY) / 2, RZ + 0.12); mGroup.rotation.set(0.0, 0.42 * Math.sin(mU * Math.PI * 1.0) - 0.21, 0);
        mMat.opacity = 0.92 * mVis; mEdge.material.opacity = 0.9 * mVis; mGlow.material.opacity = 0.35 * mVis; mGroup.visible = mVis > 0.001;
        conns.forEach((cn, m) => {
          const qt = qTiles[cn.qi].userData, et = eTiles[cn.ei].userData;
          const tm = T.mA + (T.mB - T.mA) * ((et.cx + 3.7) / 7.4) - 0.17;
          const u = clamp((t - tm) / 0.36);
          const orb = orbs[m], dk = docks[m];
          const live = t > tm && t < tm + 0.36;
          const p = V3(lerp(et.cx, qt.cx, smooth(u)), lerp(EY + TH / 2, QY - TH / 2, smooth(u)), RZ + 0.06);
          orb.position.copy(p); orb.scale.setScalar(live ? 0.22 : 0.001);
          orb.material.color.copy(C.ember).lerp(C.quillHi, smooth((u - 0.35) / 0.3)); orb.material.opacity = live ? 1 : 0;
          const dkf = t > tm + 0.36 ? Math.exp(-(t - tm - 0.36) * 4) : 0;
          dk.position.set(qt.cx, QY - TH / 2, RZ + 0.05); dk.scale.setScalar(0.55 * dkf + 0.001); dk.material.opacity = dkf;
        });
        // attention
        const aVis = (rk >= 0 ? 1 : 0) * (1 - prog(t, 39.85, 40.1));
        eTiles.forEach((tile, j) => {
          const et = tile.userData;
          let w = 0, wp = 0;
          if (rk >= 0) { w = attW[rk][j]; wp = rk > 0 ? attW[rk - 1][j] : 0; }
          const kk = rk >= 0 ? easeOutBack(clamp((t - T.att[rk]) / 0.2), 1.6) : 0;
          const ww = lerp(wp, w, kk);
          const isPeak = rk >= 0 && READERS[rk][1] === j;
          const bar = bars[j], bh = 0.36 * ww + 0.004;
          bar.position.set(et.cx, EY - TH / 2 - 0.06 - bh / 2, RZ); bar.scale.set(Math.min(0.1, et.w * 0.5), bh, 1);
          bar.material.color.copy(isPeak ? GOLD : C.quillHi); bar.material.opacity = aVis * (isPeak ? 1 : 0.6);
          barBase[j].position.set(et.cx, EY - TH / 2 - 0.06, RZ); barBase[j].scale.set(et.w, 1, 1); barBase[j].material.opacity = aVis * 0.35;
          const bm = beams[j];
          if (rk >= 0) {
            const rq = qTiles[READERS[rk][0]].userData;
            const fadeIn = prog(t, T.att[rk], T.att[rk] + 0.1);
            placeBeam(bm, V3(rq.cx, QY - TH / 2, RZ - 0.01), V3(et.cx, EY + TH / 2, RZ - 0.01), isPeak ? 0.045 : 0.006 + 0.03 * w);
            bm.material.color.copy(isPeak ? GOLD : C.quillHi); bm.material.opacity = aVis * fadeIn * (isPeak ? 0.95 : 0.25 + w);
          } else bm.material.opacity = 0;
        });
      }

      // ---------- part B: ice, LoRA, locks ----------
      const loraFront = (t - T.loraA) / T.loraDur * 38 - 1;
      for (const fx of iceFx) {
        const c = fx.c, isQ = c === q, t0 = isQ ? T.iceQ : T.iceE;
        const front = (t - t0) / T.iceDur * 38 - 1;
        const Mw = c.group.matrixWorld;
        // per-block frost growth (bottom-up crawl)
        for (let i = 0; i < NBLOCKS; i++) {
          const k = clamp((front - i) / 2.5); const im = c.iceMeshes[i];
          if (k <= 0) im.visible = false; else im.scale.multiplyScalar(lerp(0.4, 1, easeOutBack(k, 2.2)));
          const kl = clamp((loraFront - i) / 2), lm = c.loraMeshes[i];
          if (kl <= 0) lm.visible = false; else lm.scale.set(1.35 * easeOutBack(kl, 2.5), 1.7 * easeOutBack(kl, 2.5), 1);
        }
        // crystals
        for (let k = 0; k < NC; k++) {
          const sp = fx.spec[k]; const g = clamp((front - sp.b) / 2.2);
          const r = c.blockRadius(sp.b) * 1.02;
          const lp = c._centers[sp.b].clone().add(V3(Math.sin(sp.a) * r, sp.dy, Math.cos(sp.a) * r));
          const dir = V3(Math.sin(sp.a), sp.tilt, Math.cos(sp.a)).normalize();
          tmpQ.setFromUnitVectors(UP, dir);
          const sc = g <= 0 ? 0.0001 : sp.s * easeOutBack(g, 2.5);
          tmpM.compose(lp.add(dir.clone().multiplyScalar(0.04 * sc)), tmpQ, V3(sc, sc, sc)).premultiply(Mw);
          fx.inst.setMatrixAt(k, tmpM);
        }
        fx.inst.instanceMatrix.needsUpdate = true; fx.inst.visible = front > 0;
        // sparkles
        const pa = fx.sparks.geometry.attributes.position.array, ca = fx.sparks.geometry.attributes.color.array;
        for (let k = 0; k < NS; k++) {
          const sp = fx.sspec[k]; const tb = t0 + (sp.b + 1) / 38 * T.iceDur + sp.d;
          const r = c.blockRadius(sp.b) * 1.18;
          const w = c._centers[sp.b].clone().add(V3(Math.sin(sp.a) * r, sp.dy, Math.cos(sp.a) * r)).applyMatrix4(Mw);
          pa[k * 3] = w.x; pa[k * 3 + 1] = w.y; pa[k * 3 + 2] = w.z;
          let b = 0;
          if (t > tb) b = 1.3 * Math.exp(-(t - tb) * 5) + 0.5 * Math.pow(Math.max(0, Math.sin(t * sp.f + k)), 14) * (1 - 0.5 * prog(t, 43, 46));
          ca[k * 3] = 0.75 * b; ca[k * 3 + 1] = 0.9 * b; ca[k * 3 + 2] = 1.0 * b;
        }
        fx.sparks.geometry.attributes.position.needsUpdate = true; fx.sparks.geometry.attributes.color.needsUpdate = true;
        fx.sparks.visible = t > t0;
        const fr = clamp(front, 0, 35);
        fx.front.position.copy(c.blockWorld(Math.round(fr)));
        const fo = front > 0 && front < 37 ? 0.7 : 0; fx.front.material.opacity = fo; fx.front.scale.setScalar(1.6 + 0.3 * Math.sin(t * 20));
        // LoRA chip flashes
        const la = fx.lflash.geometry.attributes.position.array, lc = fx.lflash.geometry.attributes.color.array;
        for (let i = 0; i < NBLOCKS; i++) {
          const w = c.loraMeshes[i].position.clone().applyMatrix4(Mw);
          la[i * 3] = w.x; la[i * 3 + 1] = w.y; la[i * 3 + 2] = w.z;
          const ti = T.loraA + (i + 1) / 38 * T.loraDur; const f = t > ti ? 1.2 * Math.exp(-(t - ti) * 6) : 0;
          lc[i * 3] = C.link.r * f; lc[i * 3 + 1] = C.link.g * f; lc[i * 3 + 2] = C.link.b * f;
        }
        fx.lflash.geometry.attributes.position.needsUpdate = true; fx.lflash.geometry.attributes.color.needsUpdate = true;
        fx.lflash.visible = t > T.loraA;
      }
      // locks on the port rings
      for (const lk of locks) {
        const other = lk.ci === 0 ? e : q;
        const ta = T.lockA + lk.k * 0.045 + lk.ci * 0.02, tc = ta + 0.22;
        const surf = lk.c.blockSurface(lk.b, other.blockWorld(lk.b));
        const toward = other.blockWorld(lk.b).sub(lk.c.blockWorld(lk.b)).setY(0).normalize();
        const pos = surf.add(toward.multiplyScalar(0.13)).add(V3(0, 0.11, 0.1));
        const a = prog(t, ta, ta + 0.15) * (1 - 0.85 * prog(t, 42.9, 43.5));
        const punch = 1 + 0.35 * Math.max(0, kick(t, tc, 3, 7));
        lk.sp.material.map = t >= tc ? lockClosed : lockOpen;
        lk.sp.position.copy(pos); lk.sp.scale.set(0.24 * punch, 0.3 * punch, 1); lk.sp.material.opacity = a;
        const rf = t > tc ? Math.exp(-(t - tc) * 5) : 0;
        lk.ring.position.copy(pos); lk.ring.scale.setScalar(1.1 * rf + 0.001); lk.ring.material.opacity = rf;
      }

      // ---------- part C: curriculum ----------
      const gpa = goldPk.geometry.attributes.position.array, gca = goldPk.geometry.attributes.color.array;
      cyc.forEach((cy, k) => {
        const snd = cy.dir === 0 ? q : e, rcv = cy.dir === 0 ? e : q, sx = cy.dir === 0 ? -1 : 1;
        // value enters the sender's prompt (its tail)
        const u1 = clamp((t - cy.s + 0.1) / 0.32);
        const tail = snd.tailWorld();
        const from = tail.clone().add(V3(sx * 1.25, 1.9, 0.7));
        cy.inM.position.copy(from.lerp(tail, easeInCubic(u1)));
        cy.inM.quaternion.copy(camera.quaternion);
        const inA = t >= cy.s - 0.1 && t < cy.s + 0.22 ? prog(t, cy.s - 0.1, cy.s - 0.03) * (1 - prog(t, cy.s + 0.14, cy.s + 0.22)) : 0;
        cy.inM.scale.setScalar(Math.max(0.001, lerp(1, 0.35, easeInCubic(u1)) * (inA > 0 ? 1 : 0)));
        cy.inM.material.opacity = inA; cy.inM.visible = inA > 0.001;
        // gold packets across all 8 bridges (hidden states, no letters)
        const ts = cy.s + 0.18;
        for (let ex = 0; ex < 8; ex++) {
          const st = link.strands.find((s) => s.e === ex && s.dir === cy.dir);
          for (let j = 0; j < 2; j++) {
            const idx = (k * 16 + ex * 2 + j);
            const u = (t - ts - ex * 0.012) / 0.3 - j * 0.18;
            const on = u > 0 && u < 1;
            const p = st.curve.getPoint(clamp(u));
            gpa[idx * 3] = p.x; gpa[idx * 3 + 1] = p.y; gpa[idx * 3 + 2] = on ? p.z : 1e5;
            const b = on ? Math.sin(Math.PI * u) * 1.4 : 0;
            gca[idx * 3] = GOLD.r * b; gca[idx * 3 + 1] = GOLD.g * b; gca[idx * 3 + 2] = GOLD.b * b;
          }
        }
        // the receiver writes it: a gold check pops from its head
        const to = cy.s + 0.5; const u3 = clamp((t - to) / 0.55);
        const mouth = rcv.mouthWorld();
        cy.outM.position.copy(mouth.add(V3(-sx * (0.75 + 0.35 * easeOutCubic(u3)), -0.1 + 0.3 * easeOutCubic(u3), 0.3))); // beside the head, outward
        cy.outM.quaternion.copy(camera.quaternion);
        const oa = t >= to && t < to + 0.55 ? (1 - prog(t, to + 0.3, to + 0.55)) : 0;
        cy.outM.scale.setScalar(Math.max(0.001, spring(t, to, 3, 0.4) * (oa > 0 ? 1 : 0)));
        cy.outM.material.opacity = oa; cy.outM.visible = oa > 0.001;
      });
      goldPk.geometry.attributes.position.needsUpdate = true; goldPk.geometry.attributes.color.needsUpdate = true;
      goldPk.visible = t > T.warmA && t < T.warmB + 0.6;
      // full mix: cards burst into orbit
      const ctr = V3(0, 2.25, 0);
      const bt = t - T.mix;
      const orbitPh = bt > 0 ? 0.5 * bt + 0.55 * bt * bt : 0;
      cards.forEach((cd) => {
        const t0 = T.mix + cd.k * 0.035;
        const out = easeOutBack(clamp((t - t0) / 0.55), 1.3);
        const a = cd.k / 6 * Math.PI * 2 + orbitPh + 0.35;
        const R = 4.35 * out;
        const p = V3(Math.cos(a) * R, ctr.y - Math.sin(a) * 0.6 * out + 0.18 * Math.sin(a * 2 + t) * out, Math.sin(a) * R * 0.7); // near side lower, far side higher
        cd.m.position.copy(p); cd.m.quaternion.copy(camera.quaternion);
        const vis = t >= t0 ? 1 : 0;
        const depth = clamp(0.62 + 0.38 * (p.z / 3.0 + 1) / 2, 0.62, 1);
        cd.m.scale.setScalar(Math.max(0.001, vis * lerp(0.3, 1, clamp((t - t0) / 0.3)) * (1.25 - 0.25 * depth)));
        cd.m.material.opacity = vis * depth;
        const ff = t > t0 ? Math.exp(-(t - t0) * 3.5) : 0;
        cd.flare.position.copy(p); cd.flare.scale.setScalar(1.6 * ff + 0.001); cd.flare.material.opacity = ff;
      });
      const bf = bt > 0 ? Math.exp(-bt * 4) : 0;
      burst.position.copy(ctr); burst.scale.setScalar(5 * bf + 0.001); burst.material.opacity = bf;

      // ---------- HUD ----------
      chap.update(t, T0, def.end + 0.3);
      capA.update(t, 36.25, 39.95);
      capB.update(t, 40.1, 42.95);
      capC.update(t, 43.1, 46.3);
      {
        // stat: one line, centred above Quill's row
        const o = win(t, 38.75, 40.12, 0.2), dy = (1 - easeOutCubic(prog(t, 38.75, 39.15))) * -24;
        stat.style.display = o > 0.001 ? 'block' : 'none'; stat.style.opacity = o;
        stat.style.transform = `translate(0px, ${150 + dy}px)`;
      }
      // step counter
      {
        const o = win(t, T.cur, 46.4, 0.25);
        counter.style.display = o > 0.001 ? 'block' : 'none'; counter.style.opacity = o;
        const wu = prog(t, T.warmA, T.warmB, (x) => easeInOutCubic(x));
        let step = Math.round(900 * wu);
        if (t > T.warmB) step = 900 + Math.round(Math.pow(t - T.warmB, 2) * 170);
        cNum.textContent = step >= 1000 ? `${Math.floor(step / 1000)},${String(step % 1000).padStart(3, '0')}` : String(step);
        const punch = 1 + 0.12 * Math.max(0, kick(t, T.warmB, 2.5, 5));
        cNum.style.transform = `scale(${punch})`; cNum.style.transformOrigin = 'center center';
        cNum.style.color = t > T.warmB ? '#d1fae5' : '#fff7d6';
        cFill.style.width = (100 * wu) + '%';
        cFill.style.background = t > T.warmB ? CSS.link : '#ffd166'; cFill.style.boxShadow = `0 0 12px ${t > T.warmB ? CSS.link : '#ffd166'}`;
        cLbl.innerHTML = t > T.warmB ? `<span style="color:${CSS.link}">full mix</span>` : 'warm-up';
      }

      return {
        bloomStrength: 0.7 - 0.15 * win(t, 40.0, 43.2, 0.4) + 0.18 * tens + 0.12 * tens * bp,
        bloomRadius: 0.5, bloomThreshold: 0.24,
        exposure: 1.0,
        vignette: 0.9 + 0.35 * tens,
        aberration: 0.0018 + 0.0025 * tens * tens,
        flash: 0.12 * Math.max(0, Math.exp(-(t - T.iceQ) * 6)) * (t > T.iceQ ? 1 : 0) + 0.1 * bf,
        flashColor: t > 44 ? '#d1fae5' : '#cfeaff',
      };
    },
  };
}
