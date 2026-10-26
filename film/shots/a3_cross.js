// a3_cross (32–42 s): what crosses the link.
//  32–37 information: Quill gets a gold secret card (lantern), hides it from Ember, and it goes into Quill's tail
//        (its prompt). One forward pass (same wave on both): gold rises in Quill's core, crosses at each exchange depth,
//        is written INTO Ember's core and rises with the wave; Ember writes "lantern". Quill hops with joy.
//  37–42 a pushed concept (README "What crosses the link"; nl30 after block 15 at four times): Ember has nothing in its
//        prompt; a wordless swirl of snow (winter) is added into Ember's residual stream at vertebra 15 (37.75) and rises
//        with the forward pass; from depth 15 up the winter tint crosses into Quill's core (both directions run at every
//        depth); Quill writes its real opener "Hey Ember, how's the snowfall treating you?" (Qwen3 tokens, 16ths from
//        38.5) under a gentle snowfall; Ember lights up. 40.0 the link is cut (grey strands): "Hey Ember, how's your day
//        going?" (40.125-41.125).
//  Text only in the final hold: "words and concepts".
// Pure in t. Only dots travel on the link.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption } from '../lib/hud.js';
import { tokenTile } from '../lib/tokens.js';
import { C } from '../lib/palette.js';
import { clamp, lerp, prog, win, beatPulse, kick, hash, easeInOutCubic, easeOutCubic, easeInCubic } from '../lib/util.js';
import { hops, nod, blink, pop, bez, yawTo, lerpAng } from './s3_play_kit.js';
import { secretCardTexture, lanternIconTexture, thoughtTexture, texPlane } from './a3_cross_art.js';

const GOLD = new THREE.Color('#ffd166');
const GOLD_IN = new THREE.Color('#ffae1a'); // saturated gold for the writes inside the cores (reads as gold, not white)
const WARM = new THREE.Color('#ff9fb0').lerp(new THREE.Color('#ffd166'), 0.4);
const EMB = new THREE.Color('#ff8a3d'); // Ember's own state crossing the other way (both directions run at every exchange)
const ICE = new THREE.Color('#bfe9ff'), GREY = new THREE.Color('#5d6070');
const PUSH_B = 15;                            // the concept is added after block 15 (nl30, four times the strength)
const W3 = [37.75, 38.375];                   // the forward pass that reads the push
// Quill's real opener, Qwen3 tokens (results/magic-nl30/results/l15x4/episodes.json, winter, link on / link cut)
const LINE_A = ['Hey', ' Ember', ',', ' how', "'s", ' the', ' snow', 'fall', ' treating', ' you', '?'];
const LINE_B = ['Hey', ' Ember', ',', ' how', "'s", ' your', ' day', ' going', '?'];
const T1 = LINE_A.map((_, i) => 38.5 + 0.125 * i);   // 38.5 .. 39.75 (" snow" on 39.25)
const T_CUT = 40.0;
const T2 = LINE_B.map((_, i) => 40.125 + 0.125 * i); // 40.125 .. 41.125
const UX = XBLOCKS.map((b) => (b + 0.5) / NBLOCKS);
const W1 = [34.25, 36.0], W2 = [38.5, 40.0]; // the two forward passes (both creatures, same wave)
const wv = (t, [a, b]) => (t < a - 0.05 || t > b + 0.6 ? -1 : clamp((t - a) / (b - a), 0, 1.08));
const tCross = (k, [a, b]) => a + (b - a) * UX[k];

export async function create({ hud }) {
  try { await document.fonts.load('700 104px Inter'); await document.fonts.load('800 150px Inter'); } catch (_) { /* optional */ }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  scene.add(camera);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group); q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  link.pulsePts.material.size = 0.85; // bigger crossing pulses for this shot

  const card = texPlane(secretCardTexture('lantern'), 2.7); scene.add(card); card.material.color.setScalar(0.85);
  const think = texPlane(thoughtTexture(), 2.2); scene.add(think); think.material.color.setScalar(0.85);
  const icon = texPlane(lanternIconTexture(), 1.55); scene.add(icon); icon.material.color.setScalar(0.9);
  const word = tokenTile('lantern', { color: '#fff7d6', bg: 'rgba(84,58,0,0.92)', border: '#ffd166', radius: 50 }, 0.82);
  word.material.fog = false; word.material.color.setScalar(0.85); word.renderOrder = 25; word.visible = false; scene.add(word);
  const flash = new THREE.Sprite(new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: GOLD, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  flash.visible = false; scene.add(flash);
  const cap = new Caption(hud, { head: `<span style="color:#ffd166">words</span> and <span style="color:#bfe9ff">concepts</span>` });
  // the pushed concept (a swirl of snow), the snowfall, the snip of the cut link, and Quill's two openers
  const ptsMat = (size) => new THREE.PointsMaterial({ size, map: Creature.dotTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
  const mkPts = (n, size) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3).fill(1e5), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3)); const p = new THREE.Points(g, ptsMat(size)); p.frustumCulled = false; scene.add(p); return p; };
  const NSW = 150, NSN = 220;
  const swirl = mkPts(NSW, 0.26), snow = mkPts(NSN, 0.15);
  const sprite = (col) => { const m = new THREE.Sprite(new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })); m.visible = false; scene.add(m); return m; };
  const iceFlash = sprite(ICE), snip = sprite(new THREE.Color('#d8dbe6')), swirlGlow = sprite(new THREE.Color('#6fb6ff'));
  const ROW_H = 0.62, iceStyle = { color: '#f0f9ff', bg: 'rgba(16,52,92,0.92)', border: '#9fd8ff', radius: 10, glow: '#9fd8ff' };
  const mkRow = (toks, cold) => toks.map((s2, i) => { const m = tokenTile(s2.trim(), cold.includes(i) ? iceStyle : 'quill', ROW_H); m.material.fog = false; m.renderOrder = 26; m.visible = false; scene.add(m); return m; });
  const rowA = mkRow(LINE_A, [6, 7]), rowB = mkRow(LINE_B, []);
  const layout = (row) => { const gap = 0.07, w = row.reduce((a2, m) => a2 + m.userData.width, 0) + gap * (row.length - 1); let x = -w / 2; return row.map((m) => { const c = x + m.userData.width / 2; x += m.userData.width + gap; return c; }); };
  const slotsA = layout(rowA), slotsB = layout(rowB);
  const rowPt = (r, i) => V(0, r ? 6.75 : 7.6, 0.4).addScaledVector(camR, (r ? slotsB : slotsA)[i]);

  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const camR = new THREE.Vector3(), camU = new THREE.Vector3();
  const place = (m, pos, s = 1, op = 1) => { m.visible = s > 0.002 && op > 0.002; if (!m.visible) return; m.position.copy(pos); m.quaternion.copy(camera.quaternion); m.scale.setScalar(s); m.material.opacity = op; };
  const reset = (c) => { Object.assign(c.pose, Creature.defaultPose(c.side)); c.blockGlow.fill(0); for (let i = 0; i < NBLOCKS; i++) c.blockColor[i] = null; c.inject = []; };

  return {
    scene, camera,
    update(t) {
      world.update(t);
      const post = { bloomStrength: 0.95, bloomRadius: 0.5, bloomThreshold: 0.62, vignette: 0.95 };

      // ---------------- camera ----------------
      const u1 = t - 32;
      const c1p = V(-0.2 + 0.05 * u1, 3.3, 14.8 - 0.18 * u1), c1l = V(0.1, 3.2, 0);
      const c2p = V(0.0, 3.75, 15.0 + 0.1 * Math.max(0, t - 37.6)), c2l = V(0.0, 3.95, 0);
      const k2 = easeInOutCubic(prog(t, 36.9, 37.7, (x) => x));
      camera.position.copy(c1p).lerp(c2p, k2); camera.lookAt(c1l.clone().lerp(c2l, k2));
      camera.updateMatrixWorld(true);
      camR.set(1, 0, 0).applyQuaternion(camera.quaternion); camU.set(0, 1, 0).applyQuaternion(camera.quaternion);

      // ---------------- base life ----------------
      reset(q); reset(e);
      const qB = V(-2.3, 0, 0), eB = V(2.3, 0, 0);
      q.pose.position.copy(qB); e.pose.position.copy(eB);
      for (const [c, seed] of [[q, 0], [e, 1.7]]) {
        const p = c.pose;
        p.swayPhase = t * 2.3 + seed * 2; p.sway = 0.08; p.squash = 1 - 0.022 * beatPulse(t, 6);
        p.blink = blink(t, seed + 3); p.lean = 0.03 * Math.sin(t * 1.2 + seed); p.headTilt = 0.06 * Math.sin(t * 0.9 + seed * 2.1);
        p.portGlow = 0.28; p.groundGlow = 0.45;
        p.xray = 0.5 + 0.28 * Math.max(win(t, 33.9, 36.6, 0.35), win(t, 37.2, 41.9, 0.35));
      }
      q.pose.yaw = lerpAng(yawTo(qB, camera.position), yawTo(qB, eB), 0.35);
      e.pose.yaw = lerpAng(yawTo(eB, camera.position), yawTo(eB, qB), 0.35);
      let qLook = e.headWorld(), eLook = q.headWorld();
      const hq = [], he = [];
      let qHappy = 0, eHappy = 0, qCrest = 1, eCrest = 1, qNod = 0, eNod = 0, qEye = 1, eEye = 1, qFace = 0, eFace = 0;
      const ls = link.state;
      ls.depth = 1; ls.grow = 1; ls.intensity = 0.32; ls.flow = 1; ls.packets = 3; ls.aToB = 1; ls.bToA = 1; ls.sag = -0.2; ls.twist = 0.07;
      ls.tintAB = null; ls.tintBA = null; ls.pulses = []; ls.packetSize = 0.16; ls.highlight = -1;
      for (const st of link.strands) st.mat.uniforms.uCol.value.copy(C.link); // (the cut greys them below)

      // ================= 32–37 information =================
      const cardAbove = V(-3.75, 5.65, 1.0), cardHide = V(-3.45, 2.3, -0.35);
      // Quill: delighted surprise at the card, a sly glance, then hides it away from Ember
      if (t < 37) {
        if (t > 32.3 && t < 33.0) { qLook = cardAbove.clone(); qEye = 1 + 0.3 * win(t, 32.35, 32.95, 0.1); qNod -= 0.25 * win(t, 32.35, 32.95, 0.15); }
        if (t >= 33.0 && t < 33.5) qLook = e.headWorld();                         // sly glance at Ember
        const hide = win(t, 33.05, 34.1, 0.25);
        q.pose.lean -= 0.14 * hide; q.pose.bend -= 0.35 * hide; q.pose.yaw += 0.5 * hide; // turns its back to Ember
        q.pose.headTilt += 0.18 * hide;
        hq.push([32.375, 0.22, 0.375]);
        // Ember: notices, leans in, cranes to peek, can't see; puzzled
        if (t > 32.55 && t < 33.1) eLook = cardAbove.clone();
        const peek = win(t, 33.15, 34.1, 0.3);
        e.pose.lean -= 0.15 * peek; e.pose.bend -= 0.5 * peek; e.pose.leanFwd += 0.05 * peek; eNod += 0.2 * peek;
        if (peek > 0.05) eLook = cardHide.clone();
        eFace += 0.22 * Math.sin((t - 33.85) * 2 * Math.PI * 2.4) * win(t, 33.85, 34.35, 0.08);
        e.pose.headTilt += 0.25 * win(t, 33.9, 34.7, 0.15);
        // forward pass 1
        const w = wv(t, W1);
        q.pose.wave = e.pose.wave = w;
        const goldOut = 1 - prog(t, 36.6, 37.2);
        if (t > 33.72) q.inject.push({ block: 0, color: GOLD_IN, amount: 0.9 * prog(t, 33.72, 33.95) * goldOut, top: w > -0.5 && t < W1[1] ? Math.max(0.05, w) : (t < W1[0] ? 0.05 : 1.01) });
        // Quill's own blocks hold the secret too: tint them gold up to the wavefront
        { const qa = prog(t, 33.72, 33.95) * goldOut, top = w > -0.5 && t < W1[1] ? Math.max(0.05, w) : (t < W1[0] ? 0.05 : 1.01);
          if (qa > 0) for (let i = 0; i < NBLOCKS; i++) if ((i + 0.5) / NBLOCKS < top) { q.blockColor[i] = q.pal.col.clone().lerp(GOLD_IN, 0.6 * qa); q.blockGlow[i] = 0.12 * qa; } }
        XBLOCKS.forEach((b, k) => {
          const tc = tCross(k, W1);
          if (t > tc - 0.05 && t < tc + 0.3) { const uu = (t - tc + 0.05) / 0.22; ls.pulses.push({ e: k, dir: 0, u: clamp(uu), color: GOLD_IN, amount: 2.0 }); if (uu > 0.15) ls.pulses.push({ e: k, dir: 0, u: clamp(uu - 0.15), color: GOLD_IN, amount: 1.1 }); }
          if (t > tc + 0.12) e.inject.push({ block: b, color: GOLD_IN, amount: 0.95 * prog(t, tc + 0.12, tc + 0.25) * goldOut, top: t < W1[1] ? Math.max(UX[k] + 0.02, w) : 1.01 });
          // the same exchange, the other way: Quill reads Ember at this depth at the same moment
          if (t > tc - 0.05 && t < tc + 0.3) { const uu = (t - tc + 0.05) / 0.22; ls.pulses.push({ e: k, dir: 1, u: clamp(uu), color: EMB, amount: 1.3 }); }
          if (t > tc + 0.12) q.inject.push({ block: b, color: EMB, amount: 0.5 * prog(t, tc + 0.12, tc + 0.25) * goldOut, top: t < W1[1] ? Math.max(UX[k] + 0.02, w) : 1.01 });
        });
        if (w > -0.5) {
          ls.intensity = UX.map((u) => 0.3 + 1.0 * Math.exp(-Math.pow((u - w) * 9, 2)));
          ls.tintAB = GOLD; ls.aToB = UX.map((u) => (u < w + 0.05 ? 1 : 0.4));
        }
        if (t > 34.1 && t < 36.0) { qLook = V(-2.3, 2 + 3.5 * clamp(w), 0.5); eLook = V(2.3, 2 + 3.5 * clamp(w), 0.5); } // both watch the light climb
        // Ember: eyes go wide at the top, writes, then is pleased; Quill hops with joy
        eEye = Math.max(eEye, 1 + 0.35 * win(t, 35.85, 36.25, 0.08));
        eNod += nod(t, 36.0, 0.28);
        if (t > 36.0) eLook = V(4.3, 5.7, 0.7);
        eHappy = Math.max(eHappy, prog(t, 36.25, 36.45) * (1 - prog(t, 36.95, 37.2)));
        he.push([36.5, 0.28, 0.375]);
        if (t > 36.15) qLook = e.headWorld();
        qHappy = Math.max(qHappy, prog(t, 36.2, 36.35) * (1 - prog(t, 37.0, 37.25)));
        hq.push([36.375, 0.85, 0.5]);
        qCrest = Math.max(qCrest, 1 + 1.3 * win(t, 36.3, 37.0, 0.15));
      }

      // ================= 37–42 a pushed concept =================
      // Ember has nothing in its prompt. A concept (winter) is added straight into Ember's residual stream after block 15
      // (nl30, after block 15 at four times the strength): a wordless swirl of snow plunges into vertebra 15 and rises with
      // the forward pass. At the exchange depths from 15 up, the winter tint crosses into Quill's core, while every depth
      // keeps crossing both ways. Quill writes its real opener token by token (Qwen3 tokens); Ember, who has written
      // nothing, lights up. Then the same opener with the link cut: grey strands, "how's your day going?".
      if (t >= 36.9) {
        const pushA = prog(t, 37.7, 37.8);
        const w3 = wv(t, W3);
        if (w3 > -0.5) q.pose.wave = e.pose.wave = w3;
        const u15 = (PUSH_B + 0.5) / NBLOCKS;
        const cutK = prog(t, T_CUT, T_CUT + 0.125);
        const iceQ = 1 - prog(t, T_CUT, T_CUT + 0.375);
        // the push: a knot of winter at vertebra 15, carried up once the wave passes it, present at every token after
        if (pushA > 0) {
          const top = t < W3[0] + (W3[1] - W3[0]) * u15 ? u15 + 0.03 : (t < W3[1] ? Math.max(u15 + 0.03, w3) : 1.01);
          e.inject.push({ block: PUSH_B, color: ICE, amount: 1.0 * pushA, top });
          e.blockColor[PUSH_B] = ICE.clone(); e.blockGlow[PUSH_B] = 0.9 * pushA * (0.8 + 0.2 * Math.sin(t * 9));
        }
        ls.intensity = 0.36; ls.packets = 3; ls.tintAB = null; ls.tintBA = null;
        // the forward pass W3: both directions at every depth; from depth 15 up, Ember's side carries winter
        const addPass = (t0, dur, amp, trail) => XBLOCKS.forEach((b, k) => {
          const tc = t0 + dur * UX[k], cold = b >= PUSH_B && pushA > 0;
          if (t > tc - 0.04 && t < tc + 0.26) {
            const uu = clamp((t - tc + 0.04) / 0.22);
            ls.pulses.push({ e: k, dir: 1, u: uu, color: cold ? ICE : EMB, amount: (cold ? 2.3 : 1.2) * amp }, { e: k, dir: 0, u: uu, color: C.quillHi, amount: 1.2 * amp });
            if (trail && cold && uu > 0.2) ls.pulses.push({ e: k, dir: 1, u: uu - 0.2, color: ICE, amount: 1.2 * amp });
          }
        });
        if (t < T_CUT) addPass(W3[0], W3[1] - W3[0], 1, true);
        if (w3 > -0.5) ls.intensity = UX.map((u) => 0.36 + 0.9 * Math.exp(-Math.pow((u - w3) * 9, 2)));
        XBLOCKS.forEach((b, k) => {
          const tc = W3[0] + (W3[1] - W3[0]) * UX[k], top = t < W3[1] ? Math.max(UX[k] + 0.02, w3) : 1.01;
          if (t > tc + 0.08) {
            const a = prog(t, tc + 0.08, tc + 0.2);
            if (b >= PUSH_B) q.inject.push({ block: b, color: ICE, amount: 0.95 * a * iceQ, top });
            else q.inject.push({ block: b, color: EMB, amount: 0.35 * a * iceQ, top });
            e.inject.push({ block: b, color: C.quill, amount: 0.3 * a, top });
          }
        });
        // one joint pass per written token (8 exchanges each, both ways), just before each token appears
        T1.forEach((ti) => { if (t > ti - 0.13 && t < ti + 0.05) addPass(ti - 0.125, 0.11, 0.7, false); });
        // the link cut: strands go grey and quiet
        if (cutK > 0) { ls.intensity = lerp(0.36, 0.14, cutK); ls.packets = cutK > 0.5 ? 0 : 3; ls.pulses = ls.pulses.filter(() => cutK < 0.5); }
        for (const st of link.strands) st.mat.uniforms.uCol.value.copy(C.link).lerp(GREY, cutK);
        // acting
        if (t > 37.2 && t < 37.9) { eLook = e.blockWorld(PUSH_B).add(V(0.6, 0, 1.2)); eEye = Math.max(eEye, 1 + 0.3 * win(t, 37.7, 38.1, 0.08)); eNod += 0.3 * win(t, 37.3, 37.95, 0.15); }
        e.pose.squash *= 1 - 0.05 * Math.max(0, kick(t, 37.75, 3, 6));
        if (t >= 37.9 && t < 38.45) { eLook = V(2.3, 1.5 + 4 * clamp(w3), 0.6); qLook = V(-2.3, 1.5 + 4 * clamp(w3), 0.6); }
        if (t >= 38.45 && t < T_CUT) { qLook = rowPt(0, Math.min(10, Math.floor((t - 38.5) / 0.125))); eLook = qLook.clone(); }
        if (t >= T_CUT && t < 41.3) { qLook = rowPt(1, Math.min(8, Math.max(0, Math.floor((t - T2[0]) / 0.125)))); eLook = q.headWorld(); }
        if (t >= 41.3) { qLook = e.headWorld(); eLook = q.headWorld(); }
        // Ember lights up when its winter comes back as words; puzzled when the link is cut
        eHappy = Math.max(eHappy, prog(t, T1[6], T1[6] + 0.15) * (1 - prog(t, T_CUT + 0.125, T_CUT + 0.3)));
        eCrest = Math.max(eCrest, 1 + 0.9 * win(t, T1[6], T_CUT + 0.25, 0.15));
        he.push([T1[10], 0.4, 0.375]);
        e.pose.headTilt += 0.28 * win(t, T_CUT + 0.25, 41.6, 0.2);
        qNod += 0.06 * Math.sin(t * 2 * Math.PI * 4) * (win(t, 38.5, T1[10] + 0.1, 0.05) + win(t, T2[0], T2[8] + 0.1, 0.05));
        qHappy = Math.max(qHappy, win(t, T1[10] + 0.05, T_CUT + 0.1, 0.1));
      }

      // ---------------- apply ----------------
      const hQ = hops(t, hq), hE = hops(t, he);
      q.pose.position.y += hQ.y; q.pose.squash *= hQ.sq; e.pose.position.y += hE.y; e.pose.squash *= hE.sq;
      q.pose.headNod += qNod; e.pose.headNod += eNod; q.pose.faceYaw += qFace; e.pose.faceYaw += eFace;
      q.pose.happy = qHappy; e.pose.happy = eHappy;
      if (qHappy > 0.5) q.pose.blink = 0;
      if (eHappy > 0.5) e.pose.blink = 0;
      q.pose.crest = Math.min(1.8, qCrest); e.pose.crest = Math.min(1.7, eCrest);
      q.pose.lookAt = qLook; e.pose.lookAt = eLook;
      q.update(t); e.update(t);
      q.crest.scale.setScalar(1 + 0.22 * (qCrest - 1)); e.crest.scale.setScalar(1 + 0.18 * (eCrest - 1));
      for (const ey of q.eyes) ey.group.scale.setScalar(qEye);
      for (const ey of e.eyes) ey.group.scale.setScalar(eEye);
      q.group.updateMatrixWorld(true); e.group.updateMatrixWorld(true);
      link.update(t);

      // ---------------- props ----------------
      const qTail = q.tailWorld(), eMouth = e.mouthWorld();
      {
        // the card: pops in above Quill, is pulled behind Quill's back, then slides into its tail
        let p = cardAbove.clone(), s = pop(t, 32.25, 2.4, 0.42);
        p.y += 0.06 * Math.sin(t * 3);
        const k1 = easeInOutCubic(prog(t, 33.0, 33.35, (x) => x)); p.lerp(cardHide, k1);
        const k2 = prog(t, 33.45, 33.75, easeInCubic); p.lerp(qTail, k2); s *= lerp(1, 0.45, k1) * (1 - k2);
        place(card, p, t > 32.25 ? s : 0, 1);
        const fl = t > 33.7 ? Math.exp(-(t - 33.75) * 5) : 0;
        flash.visible = fl > 0.01; flash.position.copy(qTail); flash.scale.setScalar(0.5 + 2.4 * fl); flash.material.opacity = fl;
        // Ember's question
        place(think, e.headWorld().add(V(1.75, 1.0, 0.2)), pop(t, 33.875, 2.6, 0.45) * (1 - prog(t, 35.0, 35.25)), 1);
        // Ember writes the word: icon + tile leave the head
        const tw = 36.0; const u = easeOutCubic(prog(t, tw, tw + 0.3, (x) => x));
        const dst = V(4.3, 5.0, 0.7); const ctrl = eMouth.clone().lerp(dst, 0.5); ctrl.y += 0.6;
        const out = prog(t, 36.95, 37.3);
        const wp = bez(eMouth, ctrl, dst, u);
        const sc = t > tw ? lerp(0.3, 1, u) * (1 + 0.2 * Math.max(0, kick(t, tw + 0.3, 2.2, 4))) * (1 - 0.4 * out) : 0;
        place(word, wp, sc, 1 - out);
        place(icon, wp.clone().addScaledVector(camU, 1.2 * sc), sc, 1 - out);
        if (t > tw && t < 37.3) post.bloomStrength += 0.25 * Math.exp(-(t - tw - 0.3) * 4) * (t > tw + 0.3 ? 1 : 0);
      }
      // the pushed concept: a wordless swirl of snow that plunges into Ember's vertebra 15 (37.75)
      {
        const c15 = e.blockWorld(PUSH_B), form = prog(t, 37.25, 37.45), plunge = prog(t, 37.5, 37.75, easeInCubic);
        const ctr = c15.clone().add(V(1.15, 0.35, 1.1).multiplyScalar(1 - plunge));
        const pa = swirl.geometry.attributes.position.array, ca = swirl.geometry.attributes.color.array;
        const on = t > 37.25 && t < 37.8;
        for (let k = 0; k < NSW; k++) {
          const h1 = hash(k * 3.17), h2 = hash(k * 7.41), h3 = hash(k * 1.93);
          const ang = h1 * 6.283 + t * (4 + 3 * h2) * (1 + 2 * plunge), r = (0.3 + 0.8 * Math.sqrt(h2)) * (1 - 0.92 * plunge) * (0.5 + 0.5 * form);
          const fp = ctr.clone().addScaledVector(camR, Math.cos(ang) * r).addScaledVector(camU, Math.sin(ang) * r * 0.85 + (h3 - 0.5) * 0.12);
          pa[k * 3] = fp.x; pa[k * 3 + 1] = fp.y; pa[k * 3 + 2] = on ? fp.z : 1e5;
          const b = form * (0.7 + 0.3 * Math.sin(t * 13 + k)) * 2.2;
          ca[k * 3] = ICE.r * b; ca[k * 3 + 1] = ICE.g * b; ca[k * 3 + 2] = ICE.b * b;
        }
        swirl.geometry.attributes.position.needsUpdate = true; swirl.geometry.attributes.color.needsUpdate = true;
        swirlGlow.visible = on; swirlGlow.position.copy(ctr); swirlGlow.scale.setScalar(2.6 * (0.4 + 0.6 * form) * (1 - 0.8 * plunge)); swirlGlow.material.opacity = 0.55 * form;
        const fl = t > 37.75 ? Math.exp(-(t - 37.75) * 5) : 0;
        iceFlash.visible = fl > 0.01; iceFlash.position.copy(c15); iceFlash.scale.setScalar(0.6 + 2.6 * fl); iceFlash.material.opacity = fl;
        // gentle snowfall while Quill writes about it; it stops when the link is cut
        const snowA = prog(t, T1[6], T1[6] + 0.5) * (1 - prog(t, T_CUT, T_CUT + 0.5));
        const sa = snow.geometry.attributes.position.array, sc2 = snow.geometry.attributes.color.array;
        for (let k = 0; k < NSN; k++) {
          const h1 = hash(k * 2.31 + 5), h2 = hash(k * 5.77 + 1), h3 = hash(k * 9.13 + 2);
          const y = 8.2 - ((t * (0.55 + 0.35 * h2) + h3 * 9) % 9);
          sa[k * 3] = (h1 - 0.5) * 12 + Math.sin(t * 1.3 + k) * 0.25; sa[k * 3 + 1] = y; sa[k * 3 + 2] = snowA > 0.01 ? -1.5 + h2 * 4.5 : 1e5;
          const b = snowA * (0.5 + 0.5 * h3) * 1.5; sc2[k * 3] = 0.85 * b; sc2[k * 3 + 1] = 0.93 * b; sc2[k * 3 + 2] = b;
        }
        snow.geometry.attributes.position.needsUpdate = true; snow.geometry.attributes.color.needsUpdate = true;
        // the snip when the link is cut
        const sn = t > T_CUT ? Math.exp(-(t - T_CUT) * 7) : 0;
        snip.visible = sn > 0.01; snip.position.copy(q.blockWorld(16).lerp(e.blockWorld(16), 0.5)); snip.scale.set(1.2 + 3 * sn, 3.2 * (1 - 0.5 * sn), 1); snip.material.opacity = 0.7 * sn;
        // Quill's two openers: linked (top) and with the link cut (below)
        const qm = q.mouthWorld(), rowsOut = prog(t, 41.85, 42.0);
        [[rowA, T1], [rowB, T2]].forEach(([row, TT], r) => row.forEach((m, i) => {
          const ti = TT[i]; if (t < ti) { m.visible = false; return; }
          const u = easeOutCubic(prog(t, ti, ti + 0.14, (x) => x)), slot = rowPt(r, i);
          const p = qm.clone().lerp(slot, u); p.y += 0.4 * Math.sin(Math.PI * u);
          const dim = r === 0 ? 1 - 0.3 * prog(t, T_CUT, T_CUT + 0.25) : 1;
          place(m, p, lerp(0.35, 1, u) * (1 + 0.15 * Math.max(0, kick(t, ti + 0.14, 2.4, 5))), dim * (1 - rowsOut));
        }));
      }

      cap.update(t, 41.25, 41.97);
      return post;
    },
  };
}
