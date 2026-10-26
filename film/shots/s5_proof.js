// S5 "05 / THE PROOF", global 46–60 s (bars 23–29).
//  46.0–50.0  the shuffle control: real partner -> "lantern" ✓; a grey stranger (another episode's Quill) -> "galaxy" ✗
//  50.0–53.0  the number: 180 held-out episodes ignite (all teal), "180 / 180" (nl20, full mix, step 800)
//  53.0–56.9  three result windows (mini demos) on 53.0 / 53.8 / 54.6
//  56.9–60.0  title outro (text from 57.2), fade to black 59.2–60.0
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { Creature } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Chapter, toScreen } from '../lib/hud.js';
import { tokenTile, TOKEN_STYLE } from '../lib/tokens.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, easeInCubic, easeOutCubic, easeInOutCubic, easeOutBack, hash } from '../lib/util.js';
import { makeRing } from './s5_proof_ring.js';
import { makeBadge, makeNumber, makeTitle } from './s5_proof_hud.js';
import { makeDemoWindows } from './s5_proof_demos.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const vlerp = (a, b, k) => a.clone().lerp(b, k);

// key times (global seconds)
const K = {
  okWrite: 46.6, okBadge: 47.0,
  qOut0: 47.32, qOut1: 47.82, linkCut0: 47.22, linkCut1: 47.5,
  gIn: 47.72, gLink: 48.0,
  badWrite: 48.5, badBadge: 49.0,
  camUp0: 49.48, camUp1: 50.0,
  hit: 50.0, done: 51.45,
  cards: [53.0, 53.8, 54.6], cardsOut: 56.85,
  title0: 56.9, titleText: 57.2, tag: [57.7], credit: 58.2,
  hopUp: 58.0, hopLand: 58.5,
  fade0: 59.2, fade1: 60.0,
};

// blink: one blink in each ~2.7 s slot at a hashed time
function blinkAt(t, seed) {
  const P = 2.7, k = Math.floor((t + seed) / P), bt = k * P - seed + 0.4 + hash(k * 1.7 + seed * 9.1) * 1.6;
  const x = (t - bt) / 0.09; return clamp(1 - Math.abs(x - 1));
}

export async function create({ hud, def, scale }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember'), g = new Creature('quill');
  scene.add(q.group, e.group, g.group);
  q.camera = e.camera = g.camera = camera;
  const link = new Link(q, e), linkG = new Link(g, e);
  scene.add(link.group, linkG.group);

  // prompt cards at the tails (inputs) and the words Ember writes (outputs). Nothing here ever rides the link.
  // tile heights are chosen so the text renders >= 30 px tall at this framing (about 95 px per world unit)
  const secQ = tokenTile('lantern', 'secret', 0.66);
  const secG = tokenTile('galaxy', 'ghost', 0.66);
  const outOK = tokenTile('lantern', { ...TOKEN_STYLE.ember, color: '#ffe3a1', border: '#ffd166', radius: 50 }, 0.7);
  const outBad = tokenTile('galaxy', { ...TOKEN_STYLE.wrong, radius: 50 }, 0.7);
  for (const m of [secQ, secG, outOK, outBad]) { scene.add(m); m.material.color.setScalar(0.74); }

  const ring = makeRing({ pxScale: (1080 * scale) / (2 * Math.tan((35 / 2) * Math.PI / 180)), radius: 4.5 });
  scene.add(ring.group);
  const RC = V(0, 11, -1); // ring center during the number

  // ---- HUD
  const chap = new Chapter(hud, '05', 'The proof');
  const cap = new Caption(hud, {
    kicker: 'control', head: 'Swap in <span style="color:#c9cbd6">another episode\'s partner</span>',
    sub: 'Same game and roles, but a different secret.', width: 1700, pos: { left: 110, bottom: 96 },
  });
  const badgeOK = makeBadge(hud, 'ok'), badgeBad = makeBadge(hud, 'bad');
  const num = makeNumber(hud);
  // the sub is one sentence set on two centred lines (the "·" separator becomes the line break)
  const numSub1 = new Caption(hud, { sub: 'held-out episodes where the <span style="color:#34f5c0">real partner wins</span>', pos: { left: 110, top: 628 }, width: 1700, align: 'center', subSize: 46 });
  const numSub2 = new Caption(hud, { sub: 'statistically significant in all 7 games, both directions', pos: { left: 110, top: 690 }, width: 1700, align: 'center', subSize: 46 });
  const cards = makeDemoWindows(hud);
  const title = makeTitle(hud);

  // ---- camera rigs
  function camA(t) { const k = prog(t, 46, 50, (x) => x); return { pos: V(lerp(0.9, 0.2, k), lerp(3.3, 3.1, k), lerp(18.6, 17.6, k)), tgt: V(lerp(0.3, 0.1, k), 2.15, 0) }; }
  function camB(t) {
    const lt = t - 50, yaw = 0.1 * Math.sin(lt * 0.35) - 0.04;
    const d = lerp(11.6, 10.6, prog(t, 50, 53, easeOutCubic)) + 0.5 * prog(t, 53, 56.9);
    const el = lerp(0.42, 0.46, prog(t, 50, 56.9));
    return { pos: RC.clone().add(V(Math.sin(yaw) * d * Math.cos(el), Math.sin(el) * d, Math.cos(yaw) * d * Math.cos(el))), tgt: RC.clone().add(V(0, 0.15, 0)) };
  }
  function camC(t) { const k = prog(t, 56, 60, (x) => x); return { pos: V(Math.sin((t - 56) * 0.12) * 1.2, lerp(2.5, 2.8, k), lerp(23.0, 25.0, k)), tgt: V(0, lerp(0.75, 0.85, k), 0) }; }

  // pose helpers
  function alive(c, t, seed, amt = 1) {
    const p = c.pose, bp = beatPulse(t + 0.0);
    p.squash = 1 - 0.022 * bp * amt; p.sway = 0.1 * amt; p.swayPhase = t * 2.3 + seed * 2.1; p.swayFreq = 0.8;
    p.blink = blinkAt(t, seed); p.headTilt = 0.06 * Math.sin(t * 1.3 + seed); p.headNod = 0.04 * Math.sin(t * 0.9 + seed * 3);
    p.leanFwd = 0; p.lean = 0; p.bend = 0; p.happy = 0; p.faceYaw = 0; p.portGlow = 1.1; p.glow = 1; p.coreGlow = 1;
    p.position.set(c.side === 'quill' ? -2.3 : 2.3, 0, 0); p.yaw = c.side === 'quill' ? 0.5 : -0.5; p.ghost = 0; p.opacity = 1; p.crest = 1;
  }

  const T0 = def.start;
  return {
    scene, camera,
    update(t) {
      world.update(t);
      // ---------------- camera
      let cam;
      if (t < K.camUp0) cam = camA(t);
      else if (t < K.camUp1 + 0.001) {
        const a = camA(t), b = camB(t), k = easeInOutCubic(prog(t, K.camUp0, K.camUp1, (x) => x));
        cam = { pos: vlerp(a.pos, b.pos, k), tgt: vlerp(a.tgt, b.tgt, k) };
      } else if (t < K.title0) cam = camB(t);
      else {
        const b = camB(t), c = camC(t), k = clamp(spring(t, K.title0, 0.85, 0.82));
        // arc the camera through a high point so the move reads as a crane down
        cam = { pos: vlerp(b.pos, c.pos, k), tgt: vlerp(b.tgt, c.tgt, k) };
      }
      // impact shake at the hit
      const sh = kick(t, K.hit, 7, 7) * 0.08 + kick(t, K.done, 6, 7) * 0.03;
      cam.pos.x += sh; cam.pos.y += sh * 0.6;
      camera.position.copy(cam.pos); camera.lookAt(cam.tgt);
      camera.updateMatrixWorld(true);

      const showPair = t < K.camUp1 + 0.05 || t > K.title0 + 0.05;
      const phaseA = t < K.camUp1 + 0.05;

      // ---------------- creatures
      q.group.visible = e.group.visible = showPair;
      g.group.visible = phaseA && t > K.gIn - 0.05;
      link.group.visible = showPair; linkG.group.visible = g.group.visible;
      if (showPair) {
        alive(q, t, 0.3); alive(e, t, 1.7);
        if (phaseA) {
          // --- Quill: proud of the right answer, then slides out
          const qo = easeInCubic(prog(t, K.qOut0, K.qOut1, (x) => x));
          q.pose.position.x = -2.3 - 9 * qo; q.pose.lean = -0.28 * Math.sin(Math.PI * clamp(qo * 1.2));
          q.pose.happy = win(t, K.okBadge - 0.05, K.qOut0 + 0.3, 0.12);
          q.pose.squash -= 0.07 * kick(t, K.okBadge, 3, 6);
          q.pose.lookAt = e.headWorld();
          q.group.visible = qo < 0.999;
          // --- the stranger: a grey Quill from another episode
          if (g.group.visible) {
            alive(g, t, 4.1, 0.7);
            const s = spring(t, K.gIn, 1.5, 0.72), s2 = spring(t - 0.04, K.gIn, 1.5, 0.72);
            g.pose.position.x = -11.3 + 9 * s; g.pose.lean = clamp((s - s2) * 6, -0.3, 0.3);
            g.pose.ghost = 1; g.pose.portGlow = 0.9; g.pose.blush = 0; g.pose.glow = 0.8;
            g.pose.lookAt = camera.position.clone().add(V(-4, 0, 0)); // oblivious: looks off, not at Ember
            g.update(t);
          }
          // --- Ember
          const ep = e.pose;
          const lookQ = t < K.gIn + 0.15;
          ep.lookAt = lookQ ? q.headWorld() : g.headWorld();
          if (t > K.okWrite - 0.1 && t < K.okWrite + 0.25) ep.lookAt = outOK.position.clone();
          if (t > K.badBadge) ep.lookAt = outBad.position.clone();
          ep.happy = Math.max(win(t, K.okBadge - 0.05, K.qOut0 + 0.1, 0.12), 0.75 * win(t, K.badWrite + 0.05, K.badBadge + 0.02, 0.1));
          ep.squash -= 0.06 * kick(t, K.okBadge, 3, 6) + 0.05 * kick(t, K.badWrite, 3, 6);
          ep.headNod -= 0.12 * win(t, K.okWrite - 0.1, K.okWrite + 0.35, 0.12) + 0.12 * win(t, K.badWrite - 0.1, K.badWrite + 0.35, 0.12);
          // the cross: a startled recoil and a puzzled tilt
          const r = prog(t, K.badBadge, K.badBadge + 0.35, easeOutBack);
          ep.lean += 0.1 * r; ep.headTilt += 0.26 * r; ep.blink = Math.max(ep.blink, clamp(1 - Math.abs((t - K.badBadge - 0.12) / 0.07)), clamp(1 - Math.abs((t - K.badBadge - 0.4) / 0.07)));
          ep.squash += 0.05 * kick(t, K.badBadge, 4, 7);
          // link: real partner, cut, then the stranger
          link.state.grow = 1 - prog(t, K.linkCut0, K.linkCut1);
          link.state.intensity = 1 + 0.35 * win(t, 46, K.okWrite, 0.2); link.state.aToB = 1; link.state.bToA = 0.8; link.state.packets = 4; link.state.flow = 1.2;
          const gg = []; for (let k = 0; k < 8; k++) gg.push(prog(t, K.gLink + k * 0.035, K.gLink + 0.25 + k * 0.035));
          linkG.state.grow = gg; linkG.state.intensity = 1; linkG.state.bToA = 0.8; linkG.state.colorBySender = 0; linkG.state.packets = 4; linkG.state.flow = 1.2;
        } else {
          // ---------------- title: together, link blazing, a shared hop
          const TT = K.title0 - 55.6; const tg = prog(t, 55.9 + TT, 56.9 + TT);
          const hopU = clamp((t - K.hopUp) / (K.hopLand - K.hopUp));
          const air = t > K.hopUp && t < K.hopLand ? 4 * 0.75 * hopU * (1 - hopU) : 0;
          const ant = win(t, K.hopUp - 0.22, K.hopUp + 0.02, 0.1);
          for (const [c, s] of [[q, -1], [e, 1]]) {
            const p = c.pose;
            p.position.set(s * lerp(2.3, 1.85, tg), air, 0);
            p.yaw = -s * lerp(0.5, 0.62, tg);
            p.lean = -s * 0.1 * tg; p.bend = -s * 0.32 * tg;
            p.happy = prog(t, 56.1 + TT, 56.4 + TT);
            p.lookAt = (c === q ? e : q).headWorld();
            p.squash += -0.09 * ant + (air > 0 ? 0.05 * Math.sin(Math.PI * hopU) : 0) - 0.1 * Math.max(0, kick(t, K.hopLand, 2.5, 6));
            p.portGlow = 1.2 + 1.3 * prog(t, 55.9 + TT, 56.2 + TT);
            p.glow = 1 + 0.25 * prog(t, 55.9 + TT, 56.2 + TT);
            p.coreGlow = 1.2;
            p.crest = 1.15;
          }
          link.state.grow = 1; link.state.intensity = 0.85 + 0.45 * prog(t, 55.9 + TT, 56.15 + TT) + 0.5 * Math.exp(-Math.max(0, t - K.titleText) * 2.5);
          link.state.packets = 5; link.state.flow = 1.6; link.state.packetSize = 0.19; link.state.aToB = 1; link.state.bToA = 1;
        }
        if (q.group.visible) q.update(t);
        e.update(t);
        if (q.group.visible) link.update(t); link.group.visible = q.group.visible && showPair;
        if (g.group.visible) linkG.update(t);
      }

      // ---------------- tiles (billboards)
      const camQ = camera.quaternion, camRight = V(1, 0, 0).applyQuaternion(camQ);
      const placeCard = (m, base, off, a) => { m.visible = a > 0.002; if (!m.visible) return; m.position.copy(base).add(off); m.position.y += 0.05 * Math.sin(t * 2 + off.x); m.quaternion.copy(camQ); m.material.opacity = a; };
      placeCard(secQ, q.pose.position, V(1.55, 0.5, 0.95), phaseA && q.group.visible ? win(t, 45.9, K.qOut0 + 0.12, 0.12) : 0);
      placeCard(secG, g.pose.position, V(1.5, 0.5, 0.95), g.group.visible ? prog(t, K.gIn + 0.3, K.gIn + 0.5) : 0);
      // Ember writes: the word rises out of its head
      const writeTile = (m, t0, tEnd) => {
        const a = t >= t0 ? win(t, t0, tEnd, 0.18) : 0;
        m.visible = a > 0.002 && phaseA; if (!m.visible) return null;
        const mouth = e.mouthWorld(), dst = e.headWorld().add(V(1.75, 0.45, 0.5));
        const k = easeOutBack(prog(t, t0, t0 + 0.38, (x) => x), 1.3);
        m.position.copy(vlerp(mouth, dst, k)); m.position.y += 0.04 * Math.sin(t * 2.4);
        m.quaternion.copy(camQ); m.scale.setScalar(Math.max(0.01, spring(t, t0, 2.4, 0.45)));
        m.material.opacity = a; return m.position.clone().add(camRight.clone().multiplyScalar(m.userData.width / 2 + 0.55));
      };
      const okAt = writeTile(outOK, K.okWrite, K.qOut0 + 0.05);
      const badAt = writeTile(outBad, K.badWrite, K.camUp0 + 0.25);
      if (okAt) { const s = toScreen(okAt, camera); badgeOK.update(t, s.x, s.y, K.okBadge, outOK.material.opacity); } else badgeOK.update(t, 0, 0, K.okBadge, 0);
      if (badAt) { const s = toScreen(badAt, camera); badgeBad.update(t, s.x, s.y, K.badBadge, outBad.material.opacity); } else badgeBad.update(t, 0, 0, K.badBadge, 0);


      // ---------------- the ring of 180 episodes
      const kC = t > K.title0 ? clamp(spring(t, K.title0 + 0.1, 0.8, 0.85)) : 0;
      const pulse = 0.3 * Math.exp(-Math.max(0, t - K.done) * 3) * (t > K.done ? 1 : 0)
        + K.cards.reduce((s, h) => s + (t > h ? 0.45 * Math.exp(-(t - h) * 4) : 0), 0);
      const ringAmp = (t < K.camUp0 - 0.1 ? 0 : prog(t, K.camUp0 - 0.1, K.camUp1)) * lerp(1, 0.3, prog(t, 52.8, 53.2)) * lerp(1, 0.75, kC);
      ring.update(t, {
        center: vlerp(RC, V(0, 0.04, 0), kC), radius: lerp(4.5, 5.2, kC), yaw: 0.1 * Math.sin((t - 50) * 0.3) * (1 - kC),
        amp: ringAmp, pulse, unlit: 0.7, arcs: 0.55 * prog(t, 50.3, 51.2) * (1 - kC), size: lerp(0.15, 0.2, kC),
        shock: t > K.done ? 0.14 * Math.exp(-(t - K.done) * 3) : (t > K.hit ? 0.3 * Math.exp(-(t - K.hit) * 3.5) : 0),
        shockR: t > K.done ? 5.0 + 4 * easeOutCubic(clamp((t - K.done) / 1.2)) : 1.5 + 9 * easeOutCubic(clamp((t - K.hit) / 0.8)),
        wave: Math.exp(-Math.max(0, t - K.done) * 1.5) * (t > K.done ? 1 : 0),
      });

      // ---------------- HUD
      chap.update(t, T0, K.title0);
      cap.update(t, 46.2, K.camUp0 + 0.1);
      // number
      const nIn = prog(t, K.hit - 0.02, K.hit + 0.35, easeOutCubic);
      const toHeader = 0;
      const nOut = prog(t, 52.7, 53.0);
      const count = ring.count(t);
      const pop = 0.12 * Math.exp(-Math.max(0, t - K.hit) * 6) * (t > K.hit ? 1 : 0) + 0.07 * Math.max(0, kick(t, K.done, 2.5, 5) + (t > K.done ? 0.0 : 0));
      num.update(t, {
        alpha: nIn * (1 - nOut), y: lerp(392 + 60 * (1 - spring(t, K.hit, 2.2, 0.55)), 128, toHeader), scale: lerp(0.86 + 0.14 * spring(t, K.hit, 2.2, 0.5), 0.5, toHeader), count, pop,
      });
      numSub1.update(t, 50.35, 53.0, { outDur: 0.3 }); numSub2.update(t, 50.55, 53.0, { outDur: 0.3 });
      cards.update(t, K.cards, K.cardsOut);
      title.update(t, { top: 664, tTitle: K.titleText, tTag: K.tag, tCredit: K.credit });

      const fade = prog(t, K.fade0, K.fade1, smooth);
      hud.style.opacity = String(1 - fade);

      // ---------------- post
      const hitE = t > K.hit ? Math.exp(-(t - K.hit) * 2.2) : 0;
      const titleE = t > K.titleText ? Math.exp(-(t - K.titleText) * 1.5) : 0;
      const cardE = K.cards.reduce((s, h) => s + (t > h ? Math.exp(-(t - h) * 5) : 0), 0);
      return {
        bloomStrength: 0.85 + 1.1 * hitE + 0.35 * cardE + 0.55 * titleE - 0.1 * prog(t, K.title0 + 0.3, K.title0 + 0.7),
        bloomRadius: 0.6, bloomThreshold: 0.16,
        exposure: lerp(0.92, 1.0, prog(t, 49.5, 50)) + 0.15 * hitE,
        vignette: 0.95,
        aberration: (t < K.camUp0 ? 0.0012 : 0.0018) + 0.006 * hitE + (t > K.badBadge ? 0.0015 * Math.exp(-(t - K.badBadge) * 6) : 0),
        flash: 0.32 * (t > K.hit ? Math.exp(-(t - K.hit) * 9) : 0) + 0.05 * cardE * 0.6, flashColor: '#c9fff0',
        fade,
      };
    },
  };
}
