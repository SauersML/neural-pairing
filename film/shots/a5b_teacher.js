// a5b_teacher, global 76–80 s (authored in global time): chat training with a privileged teacher (run nl22c).
//  76.0–76.5   from a5's last frame the camera pulls back; the original Ember (frozen original weights, no LoRA, no link)
//              steps out of the linked Ember and lands on the right (76.5).
//  76.5–76.875 prompt cards pop at the tails: the original Ember gets both cities (its own Miami, Quill's Stockholm);
//              Quill gets Stockholm; the linked Ember gets Miami and an empty, locked slot where Quill's city would be.
//  76.875–78.0 the original Ember reads both cards (Quill's colour enters at its tail and rises) and writes the real
//              opening line, AFM's 7 tokens on 16ths: "How|’s| the| weather| in| Stockholm|?" (Stockholm on 77.875).
//  78.125–78.5 the sentence slides over to the linked Ember and becomes its target.
//  78.5–79.0   one joint pass of Quill and the linked Ember: at each of the 8 exchange depths both directions fire at once;
//              Quill's colour (Stockholm) is written into Ember's core and rises, Ember's colour into Quill's core.
//  79.0        the whole target lights at once (a teacher-forced pass scores every position together): match.
//  79.5–80.0   the original fades back into the dark, cards and sentence fade, the camera lands on a6's first frame.
// Caption (one, big): "Chat training", 76.625–77.875, top left, away from the action on the right.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Pin } from '../lib/hud.js';
import { tokenTile, TOKEN_STYLE } from '../lib/tokens.js';
import { C } from '../lib/palette.js';
import { clamp, lerp, prog, win, beatPulse, kick, easeInOutCubic, easeOutCubic, easeInCubic, smooth } from '../lib/util.js';
import { hops, blink, pop } from './s3_play_kit.js';
import { cityCardTexture, lockedSlotTexture, cardPlane } from './a5b_teacher_art.js';
import { makeTrainChart, STEP_END } from './train_chart2.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const UX = XBLOCKS.map((b) => (b + 0.5) / NBLOCKS);
const VIO = new THREE.Color('#b794ff'); // Quill's city, as a write inside Ember (reads violet in an orange core)
const ORG = new THREE.Color('#ff9a3c'); // Ember's state, as a write inside Quill

// AFM-4.5B's tokens for the reply (checked with the real tokenizer; Qwen3 splits it the same way)
const TOKS = ['How', '’s', ' the', ' weather', ' in', ' Stockholm', '?'];
const CITY = 5;
const K = {
  out0: 76.125, out1: 76.5,                      // the original steps out
  cards: { oM: 76.5, oS: 76.625, qS: 76.625, eM: 76.75, eL: 76.875 },
  tPass: [76.875, 77.25],                         // the original's forward pass (no link)
  tok: TOKS.map((_, i) => 77.25 + 0.125 * i),     // 77.25 .. 78.0
  slide: [78.125, 78.5],
  sPass: [78.5, 79.0],                            // the joint pass, both directions at every exchange
  fill: 79.0,
  exit: [79.375, 79.75],
  cap: [76.625, 77.875],
  lbl: [76.75, 78.125],                           // "original Ember" over the original's body
};
const OX = 6.9, OZ = -0.5, ROWY = 6.9, ROWX0 = 5.85;

export async function create({ hud }) {
  try { await document.fonts.load('700 84px Inter'); await document.fonts.load('600 64px Inter'); } catch (_) { /* optional */ }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  scene.add(camera);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember'), o = new Creature('ember');
  scene.add(q.group, e.group, o.group); q.camera = e.camera = o.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  link.pulsePts.material.size = 0.9;

  // prompt cards
  const CH = 0.66;
  const mk = (tx) => { const m = cardPlane(tx, CH); scene.add(m); return m; };
  const card = {
    oM: mk(cityCardTexture('Miami', 'ember')), oS: mk(cityCardTexture('Stockholm', 'quill')),
    qS: mk(cityCardTexture('Stockholm', 'quill')),
    eM: mk(cityCardTexture('Miami', 'ember')), eL: mk(lockedSlotTexture('Stockholm')),
  };
  // the reply, as Ember's round token pills; Quill's city in Quill's colour
  const cityStyle = { color: '#f3efff', bg: 'rgba(76,29,149,0.92)', border: '#a78bfa', radius: 50 };
  const tiles = TOKS.map((s, i) => { const m = tokenTile(s, i === CITY ? cityStyle : 'ember', 0.6); m.material.fog = false; m.renderOrder = 26; m.visible = false; scene.add(m); return m; });
  const GAP = 0.07, rowW = tiles.reduce((a, m) => a + m.userData.width, 0) + GAP * (tiles.length - 1);
  const slotX = []; { let x = -rowW / 2; for (const m of tiles) { slotX.push(x + m.userData.width / 2); x += m.userData.width + GAP; } }
  // sparks and flashes (sprites)
  const spr = (col, s = 1) => { const m = new THREE.Sprite(new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })); m.scale.setScalar(s); m.visible = false; scene.add(m); return m; };
  const feed = { oM: spr(C.emberHi), oS: spr(VIO), qS: spr(VIO), eM: spr(C.emberHi) };
  const cityFlash = spr(new THREE.Color('#d8c6ff'));
  const tokFlash = tiles.map(() => spr(C.emberHi));
  const rowFlash = spr(new THREE.Color('#ffe2c2'));
  // a5's chart, at its final state, slides out of the cut
  const chart = makeTrainChart(hud, { left: 1130, top: 250, w: 640, h: 470 });
  const cap = new Caption(hud, { head: 'Chat training', pos: { left: 110, top: 64 }, size: 96 });
  // which Ember is which: one plain label on the original while it reads and writes (the picture holds around it)
  const who = new Pin(hud, 'original Ember', { color: '#ffe2c2', size: 54, anchor: 'center', dot: false, weight: 700, bg: 'rgba(20,10,6,0.55)' });

  const reset = (c) => { Object.assign(c.pose, Creature.defaultPose(c.side)); c.blockGlow.fill(0); for (let i = 0; i < NBLOCKS; i++) c.blockColor[i] = null; c.inject = []; };
  const place = (m, pos, s, op) => { m.visible = s > 0.002 && op > 0.002; if (!m.visible) return; m.position.copy(pos); m.quaternion.copy(camera.quaternion); m.scale.setScalar(s); m.material.opacity = op; };
  const wv = (t, [a, b]) => (t < a || t > b + 0.35 ? -1 : Math.min(1.06, (t - a) / (b - a)));

  return {
    scene, camera,
    update(t) {
      world.update(t);
      const post = { bloomStrength: 0.8, bloomRadius: 0.5, bloomThreshold: 0.3, vignette: 0.95 };
      const relax = prog(t, 76.0, 76.5);            // from a5's end state to calm
      const exitK = prog(t, K.exit[0], K.exit[1]);  // to a6's first state

      // ---------------- camera: a5's last framing -> wide -> a6's first framing ----------------
      const cA = { p: V(5.81, 3.05, 10.51), l: V(2.9, 2.35, 0), f: 33.8 };
      const drift = prog(t, 76.75, 79.5, (x) => x);
      const cW = { p: V(lerp(2.75, 2.45, drift), lerp(3.75, 3.65, drift), lerp(14.9, 14.4, drift)), l: V(lerp(2.6, 2.4, drift), 3.5, 0), f: 35 };
      const cF = { p: V(0.7, 3.4, 15.6), l: V(0.25, 3.0, 0), f: 35 };
      const k1 = easeInOutCubic(prog(t, 76.0, 76.8, (x) => x)), k2 = easeInOutCubic(prog(t, 79.375, 80.0, (x) => x));
      camera.position.copy(cA.p).lerp(cW.p, k1).lerp(cF.p, k2);
      camera.lookAt(cA.l.clone().lerp(cW.l, k1).lerp(cF.l, k2));
      camera.fov = lerp(lerp(cA.f, cW.f, k1), cF.f, k2); camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);

      // ---------------- creatures ----------------
      reset(q); reset(e); reset(o);
      const qB = V(-2.3, 0, 0), eB = V(2.3, 0, 0);
      q.pose.position.copy(qB); e.pose.position.copy(eB);
      const bp = beatPulse(t, 6);
      for (const [c, seed] of [[q, 0], [e, 1.3], [o, 2.6]]) {
        const p = c.pose;
        p.swayPhase = t * 2.3 + seed * 2; p.sway = 0.08 + 0.05 * (1 - relax); p.squash = 1 - 0.022 * bp;
        p.blink = blink(t, seed + 5); p.lean = 0.03 * Math.sin(t * 1.2 + seed); p.headTilt = 0.06 * Math.sin(t * 0.9 + seed * 2.1);
        p.groundGlow = 0.45; p.xray = lerp(0.55, 0.6, relax);
        p.glow = lerp(1.25, 1, relax); p.coreGlow = lerp(1.25, 1, relax); p.coreFlow = lerp(2.5, 1, relax);
      }
      for (const c of [q, e]) { c.pose.crest = lerp(1.5, 1, relax); c.pose.portGlow = lerp(1.2, 1.0, relax) + 0.1 * exitK; c.pose.ice = 1 - exitK; c.pose.lora = 1 - exitK; }
      // the original Ember: steps out of the linked Ember, frozen original weights only, no link ports, no LoRA
      const ou = easeOutCubic(prog(t, K.out0, K.out1, (x) => x));
      const back = prog(t, K.exit[0], K.exit[1] - 0.05, (x) => smooth(x));
      o.pose.position.set(lerp(2.3, OX, ou) + 1.2 * back, 0.55 * Math.sin(Math.PI * ou) * (ou < 1 ? 1 : 0), lerp(-0.15, OZ, ou) - 3.5 * back);
      o.pose.opacity = prog(t, K.out0, K.out0 + 0.18) * (1 - back);
      o.pose.scale = 1 - 0.25 * back;
      o.pose.ghost = Math.min(1, back * 1.6); // it greys out as it recedes (no red tint while fading)
      o.pose.ice = 1; o.pose.lora = 0; o.pose.portGlow = 0; o.pose.crest = 1; o.pose.xray = 0.4; o.pose.glow = 1.7; o.pose.coreGlow = 1.2;
      o.pose.yaw = -0.25;
      o.group.visible = t > K.out0 && back < 1;
      q.pose.yaw = 0.5; e.pose.yaw = -0.5;

      const hq = [], he = [];
      let qHappy = 1 - prog(t, 76.15, 76.4), eHappy = qHappy, oHappy = 0;
      let qLook = e.headWorld(), eLook = q.headWorld(), oLook = camera.position.clone();
      const eHead = V(2.3, 5.1, 0.3), oHead = V(OX, 5.1, OZ + 0.3);
      // looks: Ember watches its original step out, checks its own prompt (the lock), watches the original write,
      // follows the sentence over, looks up at it during the pass, then turns to Quill
      if (t > 76.05 && t < 76.6) eLook = V(lerp(3, OX, ou), 5.0, 0);
      if (t >= 76.6 && t < 77.15) eLook = V(3.4, 0.6, 1.4);
      if (t >= 77.15 && t < K.slide[0]) eLook = V(ROWX0, ROWY, 0);
      if (t >= K.slide[0] && t < K.fill) eLook = V(lerp(ROWX0, 2.3, prog(t, K.slide[0], K.slide[1])), ROWY, 0.5);
      if (t >= K.fill + 0.3) eLook = q.headWorld();
      if (t > 76.55 && t < 76.95) oLook = V(OX, 0.6, 1.4);
      if (t >= 76.95 && t < 78.15) oLook = V(ROWX0 + (t > 77.25 ? slotX[Math.min(6, Math.floor((t - 77.25) / 0.125))] : 0), ROWY, 0.3);
      if (t >= 78.15) oLook = eHead.clone();
      if (t > 76.6 && t < 76.9) qLook = V(-2.3, 0.6, 1.4);
      if (t > 77.2 && t < 78.2) qLook = V(ROWX0, ROWY, 0);
      // reactions on the beat
      he.push([K.fill + 0.02, 0.35, 0.36]); hq.push([K.fill + 0.1, 0.25, 0.34]);
      eHappy = Math.max(eHappy, prog(t, K.fill, K.fill + 0.12) * (1 - prog(t, 79.75, 79.95)));
      qHappy = Math.max(qHappy, prog(t, K.fill + 0.05, K.fill + 0.17) * (1 - prog(t, 79.75, 79.95)));
      oHappy = prog(t, 78.0, 78.12) * (1 - prog(t, 78.6, 78.8));
      o.pose.headNod = -0.12 * win(t, 77.2, 78.1, 0.15);
      e.pose.headNod = -0.18 * win(t, K.sPass[0] - 0.1, K.fill + 0.2, 0.15);

      // ---------------- forward passes ----------------
      // the original reads both cities from its prompt: Quill's colour enters at the tail and rises with the wave
      const wO = wv(t, K.tPass);
      o.pose.wave = wO;
      const oCity = prog(t, K.tPass[0], K.tPass[0] + 0.1) * (1 - back);
      if (oCity > 0) o.inject.push({ block: 0, color: VIO, amount: 0.85 * oCity, top: wO > -0.5 && t < K.tPass[1] ? Math.max(0.03, wO) : 1.01 });
      // the joint pass: Quill and the linked Ember in lockstep
      const wS = wv(t, K.sPass);
      q.pose.wave = e.pose.wave = wS;
      const qCity = prog(t, K.sPass[0] - 0.05, K.sPass[0] + 0.05) * (1 - exitK);
      if (qCity > 0) q.inject.push({ block: 0, color: new THREE.Color('#efe6ff'), amount: 0.55 * qCity, top: wS > -0.5 && t < K.sPass[1] ? Math.max(0.03, wS) : 1.01 });
      const ls = link.state;
      ls.depth = 1; ls.grow = 1; ls.sag = -0.2; ls.twist = 0.07; ls.highlight = -1; ls.tintAB = null; ls.tintBA = null; ls.pulses = [];
      ls.intensity = lerp(1.0, 0.42, relax); ls.flow = lerp(3.5, 1, relax); ls.packets = Math.round(lerp(6, 3, relax)); ls.packetSize = lerp(0.21, 0.16, relax);
      ls.aToB = 1; ls.bToA = 1;
      if (wS > -0.5) ls.intensity = UX.map((u) => 0.42 + 1.1 * Math.exp(-Math.pow((u - wS) * 9, 2)));
      XBLOCKS.forEach((b, k) => {
        const tc = K.sPass[0] + (K.sPass[1] - K.sPass[0]) * UX[k];
        // both directions cross at the same moment at this depth
        if (t > tc - 0.06 && t < tc + 0.3) {
          const uu = clamp((t - tc + 0.06) / 0.24);
          ls.pulses.push({ e: k, dir: 0, u: uu, color: VIO, amount: 2.6 }, { e: k, dir: 1, u: uu, color: ORG, amount: 2.1 });
          if (uu > 0.18) ls.pulses.push({ e: k, dir: 0, u: uu - 0.18, color: VIO, amount: 1.1 }, { e: k, dir: 1, u: uu - 0.18, color: ORG, amount: 0.9 });
        }
        const a = prog(t, tc + 0.1, tc + 0.22) * (1 - exitK);
        if (a > 0) {
          const top = t < K.sPass[1] ? Math.max(UX[k] + 0.02, wS) : 1.01;
          e.inject.push({ block: b, color: VIO, amount: 0.95 * a, top });
          q.inject.push({ block: b, color: ORG, amount: 0.6 * a, top });
        }
      });
      // after the fill, the link settles back to its steady both-way traffic
      if (t > K.sPass[1]) ls.intensity = lerp(0.42, 0.55, prog(t, K.exit[0], 80));

      // ---------------- apply poses ----------------
      const hQ = hops(t, hq), hE = hops(t, he);
      q.pose.position.y += hQ.y; q.pose.squash *= hQ.sq; e.pose.position.y += hE.y; e.pose.squash *= hE.sq;
      o.pose.squash *= 1 - 0.12 * Math.max(0, kick(t, K.out1, 2.4, 6));
      q.pose.happy = qHappy; e.pose.happy = eHappy; o.pose.happy = oHappy;
      for (const c of [q, e, o]) if (c.pose.happy > 0.5) c.pose.blink = 0;
      q.pose.lookAt = qLook; e.pose.lookAt = eLook; o.pose.lookAt = oLook;
      q.update(t); e.update(t); o.update(t);
      // frost: the same shell a5 uses (frozen original weights); the original is more frozen; LoRA chips on the linked pair only
      for (const c of [q, e]) { c.iceMat.opacity = 0.18 * c.pose.ice; c.iceMat.emissiveIntensity = 0.06; c.iceMat.color.set('#8cc4ee'); c.loraMat.opacity = c.pose.lora * (0.55 + 0.15 * Math.sin(t * 9 + (c === q ? 0 : 1))) * 0.6; }
      o.iceMat.opacity = 0.2 * o.pose.opacity; o.iceMat.emissiveIntensity = 0.08; o.iceMat.color.set('#dcefff');
      link.update(t);

      // ---------------- cards ----------------
      const camR = V(1, 0, 0).applyQuaternion(camera.quaternion);
      const cardOut = 1 - prog(t, K.exit[0], K.exit[1]);
      const oPos = o.pose.position.clone(); oPos.y = 0;
      const pairAt = (cx, z, a, b) => { const w = a.userData.w + b.userData.w + 0.14, x0 = cx - w / 2; return [V(x0 + a.userData.w / 2, 0.62, z), V(x0 + a.userData.w + 0.14 + b.userData.w / 2, 0.62, z)]; };
      const [pOM, pOS] = pairAt(oPos.x, oPos.z + 1.1, card.oM, card.oS);
      const [pEM, pEL] = pairAt(2.3, 1.1, card.eM, card.eL);
      const pQS = V(-2.3, 0.62, 1.1);
      const pos = { oM: pOM, oS: pOS, qS: pQS, eM: pEM, eL: pEL };
      const oOp = o.pose.opacity;
      for (const key of Object.keys(card)) {
        const t0 = K.cards[key];
        let s = pop(t, t0, 2.6, 0.42), op = clamp((t - t0) / 0.06) * cardOut * (key[0] === 'o' ? oOp : 1);
        // a card glows when its model reads it
        const readT = key[0] === 'o' ? K.tPass[0] : K.sPass[0];
        const rd = t > readT ? Math.exp(-(t - readT) * 5) : 0;
        if (key === 'eL') { s *= 1 + 0.06 * Math.max(0, kick(t, K.sPass[0], 3, 6)); }
        else s *= 1 + 0.08 * rd;
        card[key].material.color.setScalar(0.88 + 0.5 * (key === 'eL' ? 0 : rd));
        place(card[key], pos[key], s, op);
      }
      // the cards' contents run into the tails when the pass starts
      const tails = { oM: o.tailWorld(), oS: o.tailWorld(), qS: q.tailWorld(), eM: e.tailWorld() };
      for (const key of Object.keys(feed)) {
        const readT = key[0] === 'o' ? K.tPass[0] : K.sPass[0], u = clamp((t - readT + 0.06) / 0.2), on = t > readT - 0.06 && t < readT + 0.14;
        const sp = feed[key]; sp.visible = on; if (!on) continue;
        sp.position.copy(pos[key]).lerp(tails[key], easeInCubic(u)); sp.scale.setScalar(0.7 * (1 - 0.5 * u)); sp.material.opacity = 1;
      }

      // ---------------- the reply ----------------
      const oMouth = o.mouthWorld();
      const sk = easeInOutCubic(prog(t, K.slide[0], K.slide[1], (x) => x));
      const rowC = V(lerp(ROWX0, 2.3, sk), ROWY + 0.4 * Math.sin(Math.PI * sk), lerp(0.0, 0.3, sk));
      const rowOut = 1 - prog(t, K.exit[0], K.exit[1]);
      const filled = t >= K.fill;
      tiles.forEach((m, i) => {
        const ti = K.tok[i];
        if (t < ti) { m.visible = false; tokFlash[i].visible = false; return; }
        const u = easeOutCubic(prog(t, ti, ti + 0.14, (x) => x));
        const slot = rowC.clone().addScaledVector(camR, slotX[i]);
        const p = oMouth.clone().lerp(slot, u); p.y += 0.5 * Math.sin(Math.PI * u) * (1 - sk);
        // on the slide the row dims to a target; at the fill it lights at once
        const target = lerp(1, 0.4, prog(t, K.slide[0], K.slide[1]));
        let op = (filled ? 1 : target) * rowOut;
        let s = lerp(0.35, 1, u) * (1 + 0.18 * Math.max(0, kick(t, ti + 0.14, 2.4, 5)));
        if (filled) s *= 1 + (i === CITY ? 0.3 : 0.12) * Math.max(0, kick(t, K.fill, 2.2, 4.5));
        place(m, p, s, op);
        const tf = tokFlash[i], f = t > ti + 0.12 ? Math.exp(-(t - ti - 0.12) * 7) : 0;
        tf.visible = f > 0.02; if (tf.visible) { tf.position.copy(slot); tf.scale.setScalar(0.9 * f + 0.3); tf.material.opacity = 0.8 * f; tf.material.color.copy(i === CITY ? VIO : C.emberHi); }
      });
      // the fill: a flash along the row and a burst on the city
      {
        const f = filled ? Math.exp(-(t - K.fill) * 4.5) : 0;
        rowFlash.visible = f > 0.02; rowFlash.position.copy(rowC); rowFlash.scale.set(rowW * 1.3 * (1 + 0.3 * (1 - f)), 1.4, 1); rowFlash.material.opacity = 0.22 * f;
        const cf = filled ? Math.exp(-(t - K.fill) * 3.2) : 0;
        cityFlash.visible = cf > 0.02; cityFlash.position.copy(rowC.clone().addScaledVector(camR, slotX[CITY])); cityFlash.scale.setScalar(1.2 + 2.6 * (1 - cf)); cityFlash.material.opacity = 0.55 * cf;
        post.bloomStrength += 0.25 * f;
      }

      // ---------------- HUD ----------------
      const ca = 1 - prog(t, 76.0, 76.3);
      chart.update(STEP_END, ca, 0);
      chart.root.style.transform = `translateX(${easeInCubic(prog(t, 76.0, 76.3)) * 140}px)`;
      cap.update(t, K.cap[0], K.cap[1]);
      who.update(o.blockWorld(17), camera, win(t, K.lbl[0], K.lbl[1], 0.12) * o.pose.opacity);
      return post;
    },
  };
}
