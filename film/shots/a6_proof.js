// a6_proof (v11). Two instances of this file (shots/index.js):
//  a6_proof (authored 92–98): one real test (nl20 relay_word, episode 18), told so the reason is visible.
//    92.125–93.125  only Quill is given the word: a big gold "cipher" card lands at Quill's tail (its prompt); Ember's
//                   prompt slot shows a lock (it never gets the word). Heading "Only Quill is given the word".
//    93.125–93.625  one pass: the gold climbs Quill's core and crosses the link into Ember at every exchange
//    93.75 / 94.125 Ember writes "cipher"; its chance of writing it with the real partner: 99.9% (main hit 94.125)
//    94.875–96.5    the real Quill and its card slide out; a grey Quill from another test (holding another word) slides
//                   in and links; one pass; Ember can't get the word: a grey "?" (main hit 96.75); hold to 98
//  a6_title (shift 25 in v11 -> authored 80–84, internally "S5 time" = authored - 20 = 60–64): the title,
//    from a7's last camera; the ring fades in around the pair; title, tagline, credit, a shared hop, fade to black.
// update(t) is a pure function of t.
import * as THREE from 'three';
import { Creature, XBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { el, FONT, toScreen, Caption } from '../lib/hud.js';
import { tokenTile, TOKEN_STYLE } from '../lib/tokens.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, easeInCubic, easeOutCubic, easeInOutCubic, easeOutBack, hash } from '../lib/util.js';
import { makeRing } from './s5_proof_ring.js';
import { makeTitle } from './s5_proof_hud.js';
import { lockedSlotTexture, cardPlane } from './a5b_teacher_art.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const vlerp = (a, b, k) => a.clone().lerp(b, k);
const GOLD = new THREE.Color('#ffd166'), DEEPGOLD = new THREE.Color('#ff9f1a'), SILVER = new THREE.Color('#c3c8da');
const TEALC = '#34f5c0', TEAL_SOFT = '#a7f3d0', GREYC = '#c3c7d6';
const SH = '0 2px 22px rgba(0,0,0,0.85)';

// proof instance (authored seconds)
const P = {
  cardIn: 92.125, cardDown: [92.75, 93.125], lock: 92.375, give: [92.125, 93.875],
  passOK: [93.125, 93.875], okWrite: 94.0, okRead: 94.375,
  linkCut: [95.0, 95.25], qOut: [95.125, 95.625], gIn: 95.5, gLink: 95.75,
  passBad: [95.875, 96.375], badWrite: 96.5, badRead: 96.75, end: 98.0,
};
// title instance (S5 seconds)
const SHIFT = 20;
const K = { title0: 59.9, titleText: 60.25, tag: [60.75], credit: 61.25, hopUp: 61.0, hopLand: 61.5, fade0: 62.8, fade1: 64.0 };

function blinkAt(t, seed) {
  const Pd = 2.7, k = Math.floor((t + seed) / Pd), bt = k * Pd - seed + 0.4 + hash(k * 1.7 + seed * 9.1) * 1.6;
  const x = (t - bt) / 0.09; return clamp(1 - Math.abs(x - 1));
}
function place(n, x, y, { a = 1, ax = 0, ay = -0.5, s = 1, blur = 0 } = {}) {
  n.style.display = a > 0.002 ? 'block' : 'none'; if (a <= 0.002) return;
  n.style.opacity = a; n.style.transform = `translate(${x}px, ${y}px) translate(${ax * 100}%, ${ay * 100}%) scale(${s})`;
  n.style.transformOrigin = `${-ax * 100}% ${-ay * 100}%`;
  n.style.filter = blur > 0.05 ? `blur(${blur}px)` : 'none';
}

export async function create({ hud, def, scale, renderer }) {
  const TITLE_ONLY = !!def && def.id === 'a6_title';
  try { await document.fonts.load('700 132px Inter'); await document.fonts.load('600 44px Inter'); } catch (_) { /* optional */ }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember'), g = new Creature('quill');
  scene.add(q.group, e.group, g.group);
  q.camera = e.camera = g.camera = camera;
  const link = new Link(q, e), linkG = new Link(g, e);
  scene.add(link.group, linkG.group);
  const pxScale = (1080 * scale) / (2 * Math.tan((35 / 2) * Math.PI / 180));
  const ring = makeRing({ pxScale, radius: 4.5 }); scene.add(ring.group);
  const title = makeTitle(hud);

  // tiles: the word in Quill's prompt, the locked slot in Ember's, the word Ember writes
  const secQ = tokenTile('cipher', 'secret', 0.66);
  const secG = tokenTile('another word', 'ghost', 0.66);
  const outOK = tokenTile('cipher', { ...TOKEN_STYLE.ember, color: '#ffe3a1', border: '#ffd166', radius: 50, glow: '#ffd166' }, 0.74);
  const outBad = tokenTile('?', { color: '#b4b7c6', bg: 'rgba(30,31,40,0.85)', border: '#5d6070', radius: 50 }, 1.25);   // Ember can't get the word
  for (const m of [secQ, secG, outOK, outBad]) { scene.add(m); m.material.color.setScalar(0.74); m.visible = false; }
  outBad.material.color.setScalar(0.62);
  const lockE = cardPlane(lockedSlotTexture('cipher'), 0.66); scene.add(lockE);
  // the word as light: a glow that climbs the holder's spine during the pass
  const climb = new THREE.Sprite(new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: GOLD.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
  climb.scale.setScalar(1.3); climb.renderOrder = 30; scene.add(climb); climb.visible = false;
  // bright packets on all 8 strands, each leaving as the pass reaches that exchange
  link.pulsePts.material.size = 0.62; linkG.pulsePts.material.size = 0.5;
  const LINKC = link.strands[0].mat.uniforms.uCol.value.clone();
  // each a->b strand flashes in the word's colour while the word crosses it
  const tintStrands = (lk, t, [a, b], color, k0) => lk.strands.forEach((st) => {
    const tk = a + ((lk.blocks[st.e] + 0.5) / 36) * (b - a);
    const k = k0 * win(t, tk - 0.02, tk + 0.45, 0.12);
    if (st.dir === 0) st.mat.uniforms.uCol.value.copy(LINKC).lerp(color, k).multiplyScalar(1 + 0.4 * k);
    else st.mat.uniforms.uCol.value.copy(LINKC).multiplyScalar(1 - 0.7 * k);   // the return strand steps back so the crossing reads
  });
  const pulsesFor = (t, [a, b], color, amount) => {
    const out = [];
    link.blocks.forEach((blk, e) => {
      const tk = a + ((blk + 0.5) / 36) * (b - a);
      for (let j = 0; j < 3; j++) { const u = (t - tk - j * 0.05) / 0.3; if (u > 0 && u < 1) out.push({ e, dir: 0, u, color, amount: amount * (1 - 0.3 * j) * Math.sin(Math.PI * Math.min(1, u * 1.15)) ** 0.4 }); }
    });
    return out;
  };

  // ---- HUD
  const abs = (css) => ({ position: 'absolute', left: '0px', top: '0px', whiteSpace: 'nowrap', pointerEvents: 'none', display: 'none', ...css });
  const headG = new Caption(hud, { head: 'Only Quill is given the word', size: 72, pos: { left: 110, top: 64 }, width: 1700 });
  const headA = new Caption(hud, { head: 'Ember’s chance of writing <span style="color:#ffd166">cipher</span>', size: 72, pos: { left: 110, top: 64 }, width: 1700 });
  const numReal = el(hud, 'div', abs({ font: `700 132px ${FONT.display}`, letterSpacing: '-0.035em', color: TEALC, textShadow: `0 0 40px rgba(52,245,192,0.55), ${SH}` }), '99.9%');
  const numOther = el(hud, 'div', abs({ font: `700 88px ${FONT.display}`, letterSpacing: '-0.025em', color: GREYC, textShadow: SH }), '1 in 14 billion');
  const lab = (txt, col) => el(hud, 'div', abs({ font: `600 44px ${FONT.text}`, color: col, textShadow: SH }), txt);
  const labReal = lab('real partner', TEAL_SOFT), labOther = lab('partner from another test', GREYC);
  const pinReal = lab('real partner', TEAL_SOFT), pinOther = lab('partner from another test', GREYC);

  function alive(c, t, seed, amt = 1) {
    const p = c.pose, bp = beatPulse(t);
    p.squash = 1 - 0.022 * bp * amt; p.sway = 0.1 * amt; p.swayPhase = t * 2.3 + seed * 2.1; p.swayFreq = 0.8;
    p.blink = blinkAt(t, seed); p.headTilt = 0.06 * Math.sin(t * 1.3 + seed); p.headNod = 0.04 * Math.sin(t * 0.9 + seed * 3);
    p.leanFwd = 0; p.lean = 0; p.bend = 0; p.happy = 0; p.faceYaw = 0; p.portGlow = 1.1; p.glow = 1; p.coreGlow = 1;
    p.position.set(c.side === 'quill' ? -2.3 : 2.3, 0, 0); p.yaw = c.side === 'quill' ? 0.5 : -0.5; p.ghost = 0; p.opacity = 1; p.crest = 1;
    p.xray = 0.6; p.wave = -1; c.inject = [];
  }
  // one forward pass in lockstep; the sender's colour is injected into Ember's core at each exchange the wave has passed
  function pass(t, [a, b], color, fadeEnd) {
    const w = (t - a) / (b - a), on = t >= a - 0.02 && t <= fadeEnd;
    const wave = t >= a && t <= b + 0.12 ? Math.min(1.02, w) : -1;
    const amt = on ? 1.4 * (1 - prog(t, fadeEnd - 0.4, fadeEnd)) : 0;
    const inj = [], src = [];
    if (amt > 0.01) {
      const top = t <= b ? w : 1.01;
      XBLOCKS.forEach((blk) => { const u = (blk + 0.5) / 36; if (u < top) inj.push({ block: blk, color, amount: amt, top: Math.min(1.01, top) }); });
      src.push({ block: 0, color, amount: amt, top: Math.min(1.01, Math.max(0.04, top)) });   // the word rises in the holder's own core
    }
    return { wave, inj, src };
  }

  // ---- camera: starts on a5c_talk's last framing, eases right after the pass to make room for the readouts
  const PAN = 2.3;
  const camA = (t) => { const k = prog(t, 92, P.end, (x) => x), p = PAN * prog(t, 93.5, 94.3, smooth); return { pos: V(lerp(0.7, 0.2, k) + p, lerp(3.4, 3.3, k), lerp(15.6, 14.9, k)), tgt: V(lerp(0.25, 0.1, k) + p, 3.0, 0) }; };
  const ROW = [V(2.05, 1.0, 0.55), V(2.05, -0.55, 0.55)];
  const TAILQ = V(-1.45, 0.5, 0.95), TAILE = V(1.45, 0.5, 0.95);

  function updateProof(t) {
    world.update(t);
    { const a = camA(t); camera.position.copy(a.pos); camera.lookAt(a.tgt); }
    camera.position.x += kick(t, P.okRead, 7, 7) * 0.025 + kick(t, P.badRead, 6, 7) * 0.03; camera.updateMatrixWorld(true);

    alive(q, t, 0.3); alive(e, t, 1.7); alive(g, t, 4.1, 0.7);
    // ---- Quill (real): receives the card, glad at the read, slides out left with its card
    const qo = easeInCubic(prog(t, P.qOut[0], P.qOut[1], (x) => x));
    q.pose.position.x = -2.3 - 9 * qo;
    q.pose.lean = -0.28 * Math.sin(Math.PI * clamp(qo * 1.2));
    q.pose.happy = win(t, P.okRead - 0.05, P.qOut[0] + 0.3, 0.12);
    q.pose.squash -= 0.07 * kick(t, P.okRead, 3, 6) + 0.08 * kick(t, P.cardDown[1], 3, 6);
    q.pose.lookAt = t < P.cardDown[1] ? secQ.position.clone() : e.headWorld();
    q.group.visible = t < P.qOut[1];
    // ---- the partner from another test: a grey Quill holding another word
    g.group.visible = t > P.gIn - 0.05;
    { const s = spring(t, P.gIn, 1.5, 0.72), s2 = spring(t - 0.04, P.gIn, 1.5, 0.72);
      g.pose.position.x = -11.3 + 9 * s; g.pose.lean = clamp((s - s2) * 6, -0.3, 0.3); }
    g.pose.ghost = 1; g.pose.portGlow = 0.9; g.pose.glow = 0.8; g.pose.lookAt = camera.position.clone().add(V(-4, 0, 0));
    const pBad = pass(t, P.passBad, SILVER, P.end + 1);
    g.pose.wave = pBad.wave; g.inject = pBad.src;
    // ---- Ember: watches Quill get the card (it gets nothing), then reads through the link
    const ep = e.pose;
    ep.lookAt = t < P.cardDown[1] + 0.1 ? secQ.position.clone() : t < P.gIn + 0.15 ? q.headWorld() : g.headWorld();
    if (t > P.okWrite - 0.1 && t < P.okWrite + 0.3) ep.lookAt = outOK.position.clone();
    if (t > P.badRead) ep.lookAt = t < P.badRead + 0.4 ? outBad.position.clone() : g.headWorld();
    ep.headTilt += 0.2 * win(t, P.lock, P.cardDown[1], 0.15);
    ep.happy = win(t, P.okRead - 0.05, P.qOut[0] + 0.1, 0.12);
    ep.squash -= 0.06 * kick(t, P.okRead, 3, 6);
    ep.headNod -= 0.12 * win(t, P.okWrite - 0.1, P.okWrite + 0.35, 0.12);
    const r = prog(t, P.badRead, P.badRead + 0.35, easeOutBack);
    ep.lean += 0.06 * r; ep.headTilt += 0.24 * r; ep.blink = Math.max(ep.blink, clamp(1 - Math.abs((t - P.badRead - 0.12) / 0.07)));
    const pOK = pass(t, P.passOK, GOLD, P.qOut[0] + 0.1);
    ep.wave = pOK.wave > -0.5 ? pOK.wave : pBad.wave; q.pose.wave = pOK.wave;
    // the word glows at the bottom of Quill's core from the moment it lands, then rises with the pass
    q.inject = pOK.src.length ? pOK.src : (t > P.cardDown[1] - 0.05 && t < P.passOK[0] ? [{ block: 0, color: GOLD, amount: 1.4 * prog(t, P.cardDown[1] - 0.05, P.cardDown[1] + 0.05), top: 0.05 }] : []);
    e.inject = pOK.inj.length ? pOK.inj : pBad.inj; ep.coreGlow = e.inject.length ? 0.55 : 1;
    // links: the real link glows gold while the word crosses
    link.group.visible = t < P.linkCut[1] + 0.05;
    link.state.grow = 1 - prog(t, P.linkCut[0], P.linkCut[1]);
    link.state.intensity = 1 + 0.6 * win(t, P.passOK[0], P.okWrite, 0.15); link.state.aToB = 1; link.state.bToA = 0.8; link.state.packets = 4; link.state.flow = 1.2;
    link.state.tintAB = t < P.okRead + 0.2 ? GOLD : null;
    linkG.group.visible = g.group.visible;
    { const gg = []; for (let k = 0; k < 8; k++) gg.push(prog(t, P.gLink + k * 0.035, P.gLink + 0.25 + k * 0.035)); linkG.state.grow = gg; }
    linkG.state.intensity = 1; linkG.state.bToA = 0.8; linkG.state.colorBySender = 0; linkG.state.packets = 4; linkG.state.flow = 1.2; linkG.state.tintAB = SILVER;
    link.state.pulses = pulsesFor(t, P.passOK, GOLD, 1.3);
    linkG.state.pulses = pulsesFor(t, P.passBad, SILVER, 0.8);
    tintStrands(link, t, P.passOK, DEEPGOLD, 1); tintStrands(linkG, t, P.passBad, SILVER, 0.6);
    for (const c of [q, e, g]) if (c.group.visible) c.update(t);
    {
      const holder = t < P.qOut[0] ? q : g, [a, b] = t < P.qOut[0] ? P.passOK : P.passBad, w = (t - a) / (b - a);
      climb.visible = w > -0.05 && w < 1.08 && holder.group.visible;
      if (climb.visible) {
        holder.blockWorld(Math.round(clamp(w) * 35), climb.position);
        climb.material.color.copy(holder === q ? GOLD : SILVER).multiplyScalar(holder === q ? 0.9 : 0.5);
        climb.material.opacity = clamp((w + 0.05) / 0.1) * (1 - clamp((w - 1) / 0.08));
      }
    }
    if (link.group.visible) link.update(t);
    if (linkG.group.visible) linkG.update(t);

    // ---- tiles (billboards)
    const camQ = camera.quaternion, camRight = V(1, 0, 0).applyQuaternion(camQ);
    const card = (m, pos, s, a) => { m.visible = a > 0.002; if (!m.visible) return; m.position.copy(pos); m.position.y += 0.05 * Math.sin(t * 2 + pos.x); m.quaternion.copy(camQ); m.scale.setScalar(s); m.material.opacity = a; };
    {
      // the card pops in big beside Quill, then drops into Quill's tail and stays there
      const pop = spring(t, P.cardIn, 2.2, 0.55), dn = easeInOutCubic(prog(t, P.cardDown[0], P.cardDown[1], (x) => x));
      const hi = q.pose.position.clone().add(V(-2.4, 3.3, 1.4)), lo = q.pose.position.clone().add(TAILQ);
      const a = t < P.cardIn ? 0 : clamp((t - P.cardIn) / 0.08) * (q.group.visible ? 1 : 0);
      card(secQ, vlerp(hi, lo, dn), lerp(1.9 * Math.max(0.05, pop), 1, dn), a);
    }
    card(secG, g.pose.position.clone().add(TAILQ), 1, g.group.visible ? prog(t, P.gIn + 0.3, P.gIn + 0.5) : 0);
    { const a = prog(t, P.lock, P.lock + 0.15), s = spring(t, P.lock, 2.4, 0.5);
      card(lockE, e.pose.position.clone().add(TAILE), 1.3 * Math.max(0.05, s), a); lockE.material.opacity = a * 0.95; }
    const writeTile = (m, t0, row) => {
      const dst = V(2.3, 3.75, 0).add(row), anchor = toScreen(dst.clone().add(camRight.clone().multiplyScalar(m.userData.width / 2)), camera);
      const a = t >= t0 ? clamp((t - t0) / 0.18) : 0;
      m.visible = a > 0.002; if (!m.visible) return t >= t0 ? anchor : null;
      const k = easeOutBack(prog(t, t0, t0 + 0.38, (x) => x), 1.3);
      m.position.copy(vlerp(e.mouthWorld(), dst, k)); m.position.y += 0.04 * Math.sin(t * 2.4 + row.y);
      m.quaternion.copy(camQ); m.scale.setScalar(Math.max(0.01, spring(t, t0, 2.4, 0.45))); m.material.opacity = a;
      return anchor;
    };
    const rowA = writeTile(outOK, P.okWrite, ROW[0]);
    { const a = t >= P.badRead ? clamp((t - P.badRead) / 0.1) : 0; outBad.visible = a > 0.002;
      if (outBad.visible) { outBad.position.copy(V(2.3, 3.75, 0).add(ROW[1])).add(V(0.35, -0.15, 0)); outBad.position.y += 0.04 * Math.sin(t * 2.4); outBad.quaternion.copy(camQ);
        outBad.scale.setScalar(Math.max(0.01, spring(t, P.badRead, 2.6, 0.45))); outBad.material.opacity = a; } }

    // ---- HUD
    headG.update(t, P.give[0], P.give[1], { outDur: 0.15 });
    headA.update(t, P.give[1] + 0.05, P.end + 1, {});
    const readout = (node, t0, row) => {
      if (!row || t < t0) { place(node, 0, 0, { a: 0 }); return; }
      const pop = spring(t, t0, 2.6, 0.55), a0 = clamp((t - t0) / 0.1);
      place(node, row.x + 34, row.y, { a: a0, s: 0.82 + 0.18 * pop, blur: (1 - a0) * 10 });
    };
    readout(numReal, P.okRead, rowA);
    const pin = (node, t0, t1, c) => {
      const a = c.group.visible ? easeOutCubic(clamp((t - t0) / 0.25)) * (1 - prog(t, t1, t1 + 0.2)) : 0;
      const h = toScreen(c.headWorld(), camera); place(node, h.x, h.y - 165, { a, ax: -0.5, blur: (1 - clamp((t - t0) / 0.2)) * 8 });
    };
    pin(pinOther, P.gIn + 0.4, P.end + 1, g);
    const under = (node, t0, row, dy) => {
      if (!row || t < t0) { place(node, 0, 0, { a: 0 }); return; }
      const a = easeOutCubic(clamp((t - t0) / 0.25));
      place(node, row.x + 40, row.y + dy + (1 - easeOutCubic(clamp((t - t0) / 0.3))) * 16, { a, blur: (1 - clamp((t - t0) / 0.2)) * 8 });
    };
    under(labReal, P.okRead + 0.125, rowA, 84);

    // hidden in this instance
    ring.update(t, { center: V(0, 0.04, 0), radius: 5.2, yaw: 0, amp: 0, unlit: 0, size: 0.2 });
    title.update(t, { top: 664, tTitle: 1e9, tTag: [1e9], tCredit: 1e9 });
    hud.style.opacity = '1';
    const hitE = [P.cardDown[1], P.okRead, P.badRead].reduce((s, h, k) => s + (t > h ? [0.3, 0.6, 0.3][k] * Math.exp(-(t - h) * 3) : 0), 0);
    return {
      bloomStrength: 0.85 + 0.35 * hitE, bloomRadius: 0.6, bloomThreshold: 0.16,
      exposure: 0.95 + 0.06 * hitE, vignette: 0.95, aberration: 0.0013 + 0.002 * hitE,
      flash: 0.06 * (t > P.okRead ? Math.exp(-(t - P.okRead) * 9) : 0), flashColor: '#c9fff0', fade: 0,
    };
  }

  // ---------------------------------------------------------------- title instance (S5 time)
  function camC(t) { const k = prog(t, 59, 64, (x) => x); return { pos: V(Math.sin((t - 59) * 0.12) * 1.2, lerp(2.5, 2.8, k), lerp(23.0, 25.0, k)), tgt: V(0, lerp(0.75, 0.85, k), 0) }; }
  function updateTitle(tg) {
    const t = tg - SHIFT;
    world.update(tg);
    for (const m of [secQ, secG, outOK, outBad, lockE, climb]) m.visible = false;
    g.group.visible = linkG.group.visible = false;
    q.group.visible = e.group.visible = link.group.visible = true;
    // from a7_happy's last camera (a frontal view of the pair), so the cut is seamless
    const b = { pos: V(-0.1, 3.65, 15.2), tgt: V(0, 3.0, 0) }, c = camC(t), kc = clamp(spring(t, K.title0, 0.85, 0.82));
    camera.position.copy(vlerp(b.pos, c.pos, kc)); camera.lookAt(vlerp(b.tgt, c.tgt, kc)); camera.updateMatrixWorld(true);
    alive(q, t, 0.3); alive(e, t, 1.7);
    const TT = K.title0 - 55.6, tg2 = prog(t, 55.9 + TT, 56.9 + TT);
    const hopU = clamp((t - K.hopUp) / (K.hopLand - K.hopUp));
    const air = t > K.hopUp && t < K.hopLand ? 4 * 0.75 * hopU * (1 - hopU) : 0;
    const ant = win(t, K.hopUp - 0.22, K.hopUp + 0.02, 0.1);
    for (const [cr, s] of [[q, -1], [e, 1]]) {
      const p = cr.pose;
      p.position.set(s * lerp(2.3, 1.85, tg2), air, 0);
      p.yaw = -s * lerp(0.5, 0.62, tg2);
      p.lean = -s * 0.1 * tg2; p.bend = -s * 0.32 * tg2;
      p.happy = Math.max(0.85, prog(t, 56.1 + TT, 56.4 + TT)); // a7 ends on happy eyes
      p.lookAt = (cr === q ? e : q).headWorld();
      p.squash += -0.09 * ant + (air > 0 ? 0.05 * Math.sin(Math.PI * hopU) : 0) - 0.1 * Math.max(0, kick(t, K.hopLand, 2.5, 6));
      p.portGlow = 1.2 + 1.3 * prog(t, 55.9 + TT, 56.2 + TT);
      p.glow = 1 + 0.25 * prog(t, 55.9 + TT, 56.2 + TT);
      p.coreGlow = 1.2; p.crest = 1.15;
    }
    link.state.grow = 1; link.state.intensity = 0.85 + 0.45 * prog(t, 55.9 + TT, 56.15 + TT) + 0.5 * Math.exp(-Math.max(0, t - K.titleText) * 2.5);
    link.state.packets = 5; link.state.flow = 1.6; link.state.packetSize = 0.19; link.state.aToB = 1; link.state.bToA = 1; link.state.tintAB = null;
    q.update(t); e.update(t); link.update(t);
    const kC = t > K.title0 ? clamp(spring(t, K.title0 + 0.1, 0.8, 0.85)) : 0;
    ring.update(t, { center: V(0, 0.04, 0), radius: 5.2, yaw: 0, amp: lerp(1, 0.75, kC) * prog(t, 60.0, 60.5), pulse: 0, unlit: 0.7, arcs: 0, size: 0.2, shock: 0, wave: 0 });
    title.update(t, { top: 664, tTitle: K.titleText, tTag: K.tag, tCredit: K.credit });
    for (const n of [numReal, numOther, labReal, labOther, pinReal, pinOther]) place(n, 0, 0, { a: 0 });
    headA.update(t, -9, -8); headG.update(t, -9, -8);
    const fade = prog(t, K.fade0, K.fade1, smooth);
    hud.style.opacity = String(1 - fade);
    const titleE = t > K.titleText ? Math.exp(-(t - K.titleText) * 1.5) : 0;
    return { bloomStrength: 0.9 + 0.55 * titleE - 0.1 * prog(t, K.title0 + 0.3, K.title0 + 0.7), bloomRadius: 0.6, bloomThreshold: 0.14, exposure: 1.0, vignette: 0.95, aberration: 0.0018, flash: 0, fade };
  }

  return { scene, camera, update: (t) => (TITLE_ONLY ? updateTitle(t) : updateProof(t)) };
}
