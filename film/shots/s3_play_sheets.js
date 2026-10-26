// S3 22.0–30.5: the two models unrolled (lib/grid.js). Quill's sheet on the left, Ember's on the right, rows aligned
// (same height = same depth). One column per token = one residual stream; columns are labelled with their tick.
//  22.1–22.75  the creatures unroll into their sheets
//  22.9–24.75  ticks 0..3: both models write one token per tick (duplex). Each new column runs bottom to top, the top
//              is decoded to the next token, which becomes the next column's input.
//  24.4–26.0   Ember's tick-3 position ("is") attends to Quill's positions with tick <= 3 at the 8 exchange rows.
//              Quill's tick-4 position is not written yet and gets no beams.
//  26.2–29.5   relay: gold = information from the prompt token "lantern". It rises up Quill's column; as Ember's "is"
//              column is computed, the exchange rows read it, the write is added into Ember's column, it rises to the
//              top and is decoded to the token "lantern" (29.5), which becomes Ember's tick-4 input.
//  29.95–30.45 the sheets roll back up into the creatures.
import * as THREE from 'three';
import { ModelSheet, LinkBeams, ROWS, XROWS } from '../lib/grid.js';
import { tokenTile } from '../lib/tokens.js';
import { Pin } from '../lib/hud.js';
import { CSS } from '../lib/palette.js';
import { clamp, lerp, prog, kick, hash, easeInOutCubic, easeOutCubic } from '../lib/util.js';
import { GOLD, GOLD_C, pop, ringTexture } from './s3_play_kit.js';

const DY = 4.2 / 36;          // block spacing = the creature's block spacing, so a column maps onto the spine
const Y0 = 0.95;              // world height of row 0 (the embedding)
const TH = 0.36;              // token tile height (world)
const GAPC = 0.1;            // gap between tiles
export const TICK_T = { 0: 22.9, 1: 23.4, 2: 23.9, 3: 24.4 }; // start of each tick's forward pass
const PASS = 0.38;            // duration of one column's bottom-to-top pass (part A)
const SCAN0 = 27.05, SCAN1 = 29.15; // part B: Ember's tick-3 column computed again, slowly
export const DECODE_T = 29.5;

const QT = [['…', -1, 'quill'], ['lantern', -1, 'secret'], ['▸', 0, 'quill'], ['Sky', 1, 'quill'], ['looks', 2, 'quill'], ['blue', 3, 'quill'], ['because', 4, 'quill']];
const ET = [['…', -1, 'ember'], ['word?', -1, 'ember'], ['▸', 0, 'ember'], ['The', 1, 'ember'], ['word', 2, 'ember'], ['is', 3, 'ember'], ['lantern', 4, 'ember']];
const Q_LANTERN = 1, E_IS = 5, E_NEXT = 6, Q_FUT = 6;

function layout(tokens) {
  const w = tokens.map(([txt, , st]) => { const m = tokenTile(txt, st, TH); const ww = Math.max(0.5, m.userData.width); m.geometry.dispose(); m.material.dispose(); return ww; });
  const xs = []; let x = 0; w.forEach((ww, i) => { xs.push(x + ww / 2); x += ww + GAPC; });
  const total = x - GAPC; return { xs: xs.map((v) => v - total / 2), total };
}

export function makeSheets(scene, hud) {
  const LQ = layout(QT), LE = layout(ET);
  const GAP = 1.9;
  const QC = -(GAP / 2 + LQ.total / 2), EC = GAP / 2 + LE.total / 2; // sheet centers
  const q = new ModelSheet('quill', { tokens: QT.map(([text, , style]) => ({ text, style })), dy: DY, colX: (i) => LQ.xs[i], tileH: TH, arcs: 24 });
  const e = new ModelSheet('ember', { tokens: ET.map(([text, , style]) => ({ text, style })), dy: DY, colX: (i) => LE.xs[i], tileH: TH, arcs: 24 });
  // tiles are drawn slightly below the bloom threshold so their text stays sharp
  for (const s of [q, e]) for (const tl of s.inTiles) { tl.material.fog = false; tl.renderOrder = 20; tl.material.color.setScalar(0.8); }
  scene.add(q.group, e.group);
  const beams = new LinkBeams(scene, { pool: 56 });
  beams.packets.material.size = 0.3;
  // output (decoded next-token) tiles above columns, own meshes
  const outTile = (txt, st) => { const m = tokenTile(txt, st, TH); m.material.fog = false; m.material.color.setScalar(0.85); m.renderOrder = 21; m.visible = false; scene.add(m); return m; };
  const qOut = { 3: outTile('Sky', 'quill'), 4: outTile('looks', 'quill'), 5: outTile('blue', 'quill') };
  const eOut = { 3: outTile('The', 'ember'), 4: outTile('word', 'ember'), 5: outTile('is', 'ember') };
  const lanternOut = (() => { const m = tokenTile('lantern', { color: '#fff7d6', bg: 'rgba(84,58,0,0.92)', border: GOLD, radius: 50 }, TH * 1.7); m.material.fog = false; m.material.color.setScalar(0.85); m.renderOrder = 22; m.visible = false; scene.add(m); return m; })();
  const burst = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTexture(), color: GOLD_C, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  burst.visible = false; scene.add(burst);
  // "now" bands behind the current column of each sheet
  const bandMat = () => new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, uniforms: { uAmp: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader: `uniform float uAmp; varying vec2 vUv; void main(){ float x = 1.0-abs(vUv.x-0.5)*2.0; float y = smoothstep(0.0,0.08,vUv.y)*smoothstep(1.0,0.9,vUv.y);
      gl_FragColor = vec4(vec3(0.08,0.89,0.66)*uAmp*0.16*pow(x,1.5)*y, 1.0);} `,
  });
  const bands = [q, e].map((s) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 36 * DY + 1.3), bandMat()); m.position.z = -0.12; s.group.add(m); return m; });

  // pins: names, tick numbers, "future"
  const nameQ = new Pin(hud, 'Quill', { color: CSS.quill, size: 40, dot: false, anchor: 'right', weight: 700 });
  const nameE = new Pin(hud, 'Ember', { color: CSS.ember, size: 40, dot: false, anchor: 'left', weight: 700 });
  const tickPins = [];
  for (const [s, toks] of [[q, QT], [e, ET]]) toks.forEach(([, tk], i) => {
    // one label per tick: the two prompt columns share "−1"
    if (tk === -1 && i > 0) return;
    tickPins.push({ s, i: tk === -1 ? 0.5 : i, tk, pin: new Pin(hud, String(tk).replace('-', '−'), { color: 'rgba(235,232,255,0.85)', size: 32, dot: false, anchor: 'center', font: "'JetBrains Mono', monospace", weight: 500 }) });
  });
  const tickWord = new Pin(hud, 'tick', { color: 'rgba(235,232,255,0.6)', size: 32, dot: false, anchor: 'right', font: "'JetBrains Mono', monospace", weight: 500 });
  const futPin = new Pin(hud, 'not yet written', { color: 'rgba(235,232,255,0.7)', size: 30, dot: false, anchor: 'center' });

  const goldBeams = [];
  const colX = (s, i) => (s === q ? LQ.xs : LE.xs)[Math.floor(i)] * (1 - (i % 1)) + (s === q ? LQ.xs : LE.xs)[Math.ceil(i)] * (i % 1);
  const black = new THREE.Color(0, 0, 0), tmpC = new THREE.Color();
  const tickOf = (s, i) => (s === q ? QT : ET)[i][1];

  return {
    centers: { q: QC, e: EC },
    // returns how present the sheets are (0..1)
    update(t, camera, post) {
      const spread = easeInOutCubic(prog(t, 22.1, 22.75, (x) => x)) * (1 - easeInOutCubic(prog(t, 29.95, 30.45, (x) => x)));
      const on = spread > 0.001 && t < 30.5;
      for (const [s, cx, crX] of [[q, QC, -2.3], [e, EC, 2.3]]) {
        s.group.visible = on;
        s.group.position.set(lerp(crX, cx, spread), lerp(0.05, Y0, spread), 0);
        s.group.scale.set(Math.max(0.02, spread), lerp(1.0, 1, spread), 1);
      }
      beams.group.visible = on; burst.visible = false;
      for (const m of [...Object.values(qOut), ...Object.values(eOut), lanternOut]) m.visible = false;
      const hidePins = () => { nameQ.update(new THREE.Vector3(), camera, 0); nameE.update(new THREE.Vector3(), camera, 0); for (const p of tickPins) p.pin.update(new THREE.Vector3(), camera, 0); tickWord.update(new THREE.Vector3(), camera, 0); futPin.update(new THREE.Vector3(), camera, 0); };
      if (!on) { hidePins(); return 0; }
      const op = clamp(spread * 1.4 - 0.1);
      q.state.opacity = e.state.opacity = op;
      q.state.streamGlow = e.state.streamGlow = 0.9; q.state.nodeGlow = e.state.nodeGlow = 0.55;
      q.state.exchangeGlow = e.state.exchangeGlow = 0.9;
      beams.opacity = op;

      // ---- per-column pass progress (0 = not computed yet, 1 = done)
      const pass = (s, i) => {
        const tk = tickOf(s, i);
        if (tk < 0) return 1;
        if (tk <= 3) return clamp((t - TICK_T[tk]) / PASS);
        if (s === e && i === E_NEXT) return clamp((t - (DECODE_T + 0.35)) / 0.3); // "lantern" enters Ember at tick 4
        return 0;
      };
      // the current tick (for the "now" band and the highlighted tick label)
      let nowTick = -1; for (let k = 0; k <= 3; k++) nowTick += easeInOutCubic(clamp((t - (TICK_T[k] - 0.2)) / 0.25));
      if (t > DECODE_T + 0.25) nowTick = 3 + easeInOutCubic(clamp((t - DECODE_T - 0.25) / 0.3));

      q.begin(); e.begin(); beams.begin(); goldBeams.length = 0;
      // ---- self-attention inside each model: a new column reads earlier columns (causal)
      for (const s of [q, e]) for (let i = 2; i < 6; i++) {
        const tk = tickOf(s, i); const a = win2(t, TICK_T[tk] + 0.05, TICK_T[tk] + 0.55);
        if (a <= 0) continue;
        for (const [row, from, w] of [[14, i - 1, 0.45], [26, 1, 0.4], [8, 0, 0.25]]) if (from < i) s.arc(row, from, i, w * a, null, t);
      }
      // ---- part A: at tick 3 Ember's "is" reads Quill's positions with tick <= 3 at every exchange row
      const fadeA = 1 - prog(t, 25.85, 26.3);
      if (t > TICK_T[3] && t < 26.4) {
        const front = pass(e, E_IS) * (ROWS - 1);
        XROWS.forEach((r, k) => {
          const a = clamp((front - r) / 3) * fadeA; if (a <= 0) return;
          const hi = k === 5; // one exchange row drawn as a full fan; the others faint
          for (let j = 0; j <= 5; j++) { // Quill columns 0..5 = ticks -1..3; column 6 (tick 4) is never read
            const w = (hi ? (j === Q_LANTERN ? 0.55 : 0.2 + 0.25 * hash(j * 7.1)) : 0.03 + 0.05 * hash(j * 7.1 + k * 3.3)) * a;
            beams.beam(q, j, e, E_IS, r, w, t + k * 0.13, null, hi ? 0.32 * (j - 2.5) : 0, hi ? 1.2 : 0.6);
          }
        });
      }
      // ---- part B: the relay
      const goldQ = prog(t, 26.25, 27.0, (x) => x);              // gold rises up Quill's "lantern" column
      if (t > 26.2) {
        q.addInfo(Q_LANTERN, 0, goldQ * (ROWS - 1), GOLD_C, 0.75);
        // self-attention spreads a little of it to Quill's later columns
        const sp = prog(t, 26.9, 27.4);
        for (let j = 2; j <= 5; j++) { q.addInfo(j, 10, ROWS - 1, GOLD_C, 0.16 * sp); }
        if (t < 28.0) for (let j = 3; j <= 5; j++) q.arc(18, Q_LANTERN, j, 0.35 * sp * (1 - prog(t, 27.6, 28.0)), GOLD_C, t);
      }
      const scan = clamp((t - SCAN0) / (SCAN1 - SCAN0)) * (ROWS - 1);   // Ember's "is" column, computed bottom to top
      const fadeB = 1 - prog(t, 29.9, 30.2);
      if (t > SCAN0 && t < 30.3) {
        e.addInfo(E_IS, Math.max(0, scan - 2.5), scan, new THREE.Color(1, 1, 1), 0.22 * (scan < ROWS - 1.5 ? 1 : 0));
        let goldIn = 0; goldBeams.length = 0;
        XROWS.forEach((r, k) => {
          const a = clamp((scan - r + 0.5) / 2.5); if (a <= 0) return;
          goldIn += 1;
          const hold = (0.45 + 0.55 * Math.exp(-Math.max(0, scan - r) / 6)) * fadeB;
          goldBeams.push([beams.n, hold]); beams.beam(q, Q_LANTERN, e, E_IS, r, 0.7 * hold, t + k * 0.11, GOLD_C, 0, 0.9);
          for (const j of [0, 2, 3, 4, 5]) beams.beam(q, j, e, E_IS, r, (0.03 + 0.04 * hash(j * 3.1 + k)) * hold, t + k * 0.07, null, 0.12 * (j - 2.5), 0.6);
          // the write is added into Ember's column at this row and carried up by the residual stream
          e.addInfo(E_IS, r, scan, GOLD_C, 0.16 * a);
        });
      }
      // decode at the top: the next token is "lantern"
      if (t > DECODE_T - 0.1) {
        const k = pop(t, DECODE_T, 2.4, 0.4);
        const fly = easeInOutCubic(clamp((t - (DECODE_T + 0.1)) / 0.3));
        const top = e.nodeWorld(E_IS, ROWS - 1).add(new THREE.Vector3(0, 0.55, 0.3));
        const bottom = e.nodeWorld(E_NEXT, 0).add(new THREE.Vector3(0, -0.32, 0.02));
        const mid = top.clone().lerp(bottom, 0.5); mid.x += 0.6; mid.z += 0.6;
        const p = new THREE.Vector3().copy(top).multiplyScalar((1 - fly) * (1 - fly)).addScaledVector(mid, 2 * fly * (1 - fly)).addScaledVector(bottom, fly * fly);
        lanternOut.visible = op > 0.05 && fly < 0.999; lanternOut.position.copy(p); lanternOut.quaternion.copy(camera.quaternion);
        lanternOut.scale.setScalar(Math.max(0.01, k * lerp(1, 1 / 1.7, fly) * (1 + 0.25 * Math.max(0, kick(t, DECODE_T, 2, 4))))); lanternOut.material.opacity = op;
        const bo = Math.exp(-(t - DECODE_T) * 3.2) * (t > DECODE_T ? 1 : 0) * (1 - prog(t, 29.8, 29.95));
        burst.visible = bo > 0.01 && op > 0.05; burst.position.copy(top); burst.scale.setScalar(1.5 + (1 - bo) * 2.4); burst.material.opacity = bo;
        post.bloomStrength += 0.3 * bo;
      }
      q.end(t); e.end(t); beams.end(t);
      for (const [i] of goldBeams) if (beams.pool[i]) beams.pool[i].material.color.copy(GOLD_C).lerp(new THREE.Color('#14e3a8'), 0.25);

      // ---- per-column visibility (streams, nodes, input tiles), "future" columns dark
      for (const s of [q, e]) {
        for (let i = 0; i < s.N; i++) {
          const ps = pass(s, i), tk = tickOf(s, i);
          const ghost = tk >= 4 && ps <= 0 ? 0.18 : 0;
          const shown = Math.max(ps, ghost > 0 ? 1 : 0);
          const st = s.streams[i]; st.visible = shown > 0.001;
          st.material.uniforms.uReveal.value = ghost ? 1 : ps;
          st.material.uniforms.uAmp.value = s.state.streamGlow * op * (ghost ? ghost : 1);
          for (let r = 0; r < ROWS; r++) {
            const idx = i * ROWS + r;
            if (ghost) { s.nodes.getColorAt(idx, tmpC); s.nodes.setColorAt(idx, tmpC.multiplyScalar(ghost)); }
            else if (r / (ROWS - 1) > ps + 1e-6) s.nodes.setColorAt(idx, black);
          }
          // input tile: prompt tokens are there from the start; generated tokens arrive at their tick
          const tl = s.inTiles[i]; let sc = 1, vis = spread > 0.6;
          if (tk >= 0) {
            const t0 = tk <= 3 ? TICK_T[tk] - 0.12 : (s === e && i === E_NEXT ? DECODE_T + 0.38 : 1e9);
            sc = pop(t, t0, 3, 0.45); vis = vis && t > t0;
          }
          tl.visible = vis; tl.scale.set(sc / Math.max(0.05, spread), sc, 1);
        }
        s.nodes.instanceColor.needsUpdate = true;
      }
      // decoded outputs of ticks 0..2 pop above the column, then drop to become the next column's input
      for (const [s, outs] of [[q, qOut], [e, eOut]]) for (const [ci, m] of Object.entries(outs)) {
        const i = Number(ci), src = i - 1, tk = tickOf(s, src);
        const tDone = TICK_T[tk] + PASS, tNext = TICK_T[tickOf(s, i)] - 0.12;
        if (t < tDone || t > tNext + 0.02 || op < 0.05) continue;
        const k = pop(t, tDone, 3, 0.45); const fly = easeInOutCubic(clamp((t - (tNext - 0.2)) / 0.2));
        const top = s.nodeWorld(src, ROWS - 1).add(new THREE.Vector3(0, 0.42, 0.02)), bot = s.nodeWorld(i, 0).add(new THREE.Vector3(0, -0.32, 0.02));
        const mid = top.clone().lerp(bot, 0.5); mid.x += 0.5; mid.z += 0.5;
        const p = top.clone().multiplyScalar((1 - fly) * (1 - fly)).addScaledVector(mid, 2 * fly * (1 - fly)).addScaledVector(bot, fly * fly);
        m.visible = true; m.position.copy(p); m.quaternion.copy(camera.quaternion); m.scale.setScalar(Math.max(0.01, k)); m.material.opacity = op;
      }
      // now bands
      bands.forEach((b, si) => {
        const s = si ? e : q; const toks = si ? ET : QT;
        const idx = (tk) => { const i = toks.findIndex((x) => x[1] === Math.max(0, tk)); return tk < 0 ? 0.5 : i; };
        const fl = Math.floor(nowTick), fr = nowTick - fl;
        const x = lerp(colX(s, idx(fl)), colX(s, idx(Math.min(4, fl + 1))), fr);
        b.position.set(x, (36 * DY) / 2 - 0.1, -0.12); b.scale.set(1 / Math.max(0.05, spread), 1, 1);
        b.material.uniforms.uAmp.value = op * prog(t, 22.6, 22.9) * (1 - prog(t, 29.9, 30.2));
      });

      // ---- pins
      const pinOp = op * (1 - prog(t, 29.85, 30.1));
      nameQ.update(q.nodeWorld(0, 18), camera, pinOp * prog(t, 22.5, 22.8), -45, 0);
      nameE.update(e.nodeWorld(e.N - 1, 18), camera, pinOp * prog(t, 22.5, 22.8), 45, 0);
      for (const p of tickPins) {
        const s = p.s, pos = s.group.localToWorld(new THREE.Vector3(colX(s, p.i), -0.82, 0));
        const cur = Math.abs(p.tk - Math.round(nowTick)) < 0.5;
        p.pin.root.style.color = cur ? CSS.link : 'rgba(235,232,255,0.8)';
        p.pin.update(pos, camera, pinOp * prog(t, 22.55, 22.85));
      }
      tickWord.update(q.group.localToWorld(new THREE.Vector3(colX(q, 0) - 0.55, -0.82, 0)), camera, pinOp * prog(t, 22.55, 22.85));
      futPin.update(q.nodeWorld(Q_FUT, ROWS - 1).add(new THREE.Vector3(0, 0.5, 0)), camera, pinOp * prog(t, 24.5, 24.8) * (1 - prog(t, 26.6, 26.9)));
      post.bloomThreshold = lerp(post.bloomThreshold, 0.88, op); post.bloomStrength = lerp(post.bloomStrength, 0.75, op);
      return op;
    },
  };
}
const win2 = (t, a, b) => (t < a || t > b ? 0 : Math.sin(Math.PI * (t - a) / (b - a)));
