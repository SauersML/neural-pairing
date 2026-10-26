// a5c_talk, global 88–92 s (authored in global time): after chat training the twins actually talk.
// The same real test chat twice (episode 4242:co_chat:0; Quill is privately in Oslo, Ember holds nothing):
//  88.0–89.5  before (nl20, results/magic-nl20/results/chat_before_0): all six turns are the same sentence, echoed
//              word for word. The first lands on 88.25 and holds a beat, five repeats fire on 16ths (88.5–89.0), each greys
//              into an echo stack while a counter ticks ×2 … ×6;
//              the creatures bob in sync, half-lidded.
//  89.5–91.25  after (nl22c, chat_after_0, link on): one joint pass where Quill's private Oslo crosses into Ember while
//              Ember's state crosses back (89.5–89.875); Ember opens "I noticed you mentioned you're in Oslo." (90.0),
//              then new lines each turn (real prefixes): Quill "Oslo is a beautiful city…" (90.375), Ember "That sounds
//              wonderful!" (90.75), Quill "Akerselva is a lovely spot…" (91.0). A pass both ways precedes every turn.
//  91.25–91.875 hold: "repeated turns  17% → 1%" (the 20 test chats). The camera returns to a6's first framing at 92.
// Labels: "before chat training" / "after chat training". Lines are shown as chat bubbles (whole text, no token tiles).
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { el, FONT } from '../lib/hud.js';
import { C } from '../lib/palette.js';
import { clamp, lerp, prog, win, beatPulse, kick, easeInOutCubic, easeOutCubic, easeOutExpo } from '../lib/util.js';
import { hops, blink } from './s3_play_kit.js';
import { cityCardTexture, cardPlane } from './a5b_teacher_art.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const UX = XBLOCKS.map((b) => (b + 0.5) / NBLOCKS);
const VIO = new THREE.Color('#b794ff'), ORG = new THREE.Color('#ff9a3c');
const LOOP = 'The city of Oslo is known for its stunning…';
const T_B = [88.25, 88.5, 88.625, 88.75, 88.875, 89.0];             // six identical turns, Ember first: the first holds a beat, five repeats on 16ths
const PASS0 = [89.5, 89.875];                                        // the pass that brings Oslo over
const T_A = [90.0, 90.375, 90.75, 91.0, 91.25, 91.5];               // after: six new lines (the real transcript, cut short)
const LINES = [
  ['ember', [['I noticed you mentioned you’re in '], ['Oslo', '#c9b2ff'], ['.']]],
  ['quill', [['Oslo is a beautiful city…']]],
  ['ember', [['That sounds wonderful!']]],
  ['quill', [['Akerselva is a lovely spot…']]],
  ['ember', [['The local seafood sounds like a must-try.']]],
  ['quill', [['There’s a small seafood market near the harbor…']]],
];
const LBL = { before: [88.0, 89.5], after: [89.5, 90.7] }, STAT = [999, 999];   // no stat: the conversation itself shows it

function rr(x, a, b, w, h, r) { r = Math.max(0, Math.min(r, w / 2, h / 2)); x.beginPath(); x.moveTo(a + r, b); x.arcTo(a + w, b, a + w, b + h, r); x.arcTo(a + w, b + h, a, b + h, r); x.arcTo(a, b + h, a, b, r); x.arcTo(a, b, a + w, b, r); x.closePath(); }
// a chat bubble in the speaker's colour, tail toward the speaker's side; runs = [[text, colour?], ...]
function bubbleTexture(runs, side, { grey = false } = {}) {
  const H = 132, F = 64, pad = 46, tail = 26;
  const m = document.createElement('canvas').getContext('2d'); m.font = `600 ${F}px Inter`;
  const tw = runs.reduce((a, [s]) => a + m.measureText(s).width, 0);
  const W = Math.ceil(tw + 2 * pad);
  const c = document.createElement('canvas'); c.width = W; c.height = H + tail; const x = c.getContext('2d');
  const k = side === 'quill' ? { bg: 'rgba(52,22,112,0.94)', bd: '#a78bfa', fg: '#f3efff' } : { bg: 'rgba(110,40,12,0.94)', bd: '#fb923c', fg: '#fff1e2' };
  const bg = grey ? 'rgba(44,46,58,0.92)' : k.bg, bd = grey ? '#8b8fa3' : k.bd, fg = grey ? '#c9ccd8' : k.fg;
  rr(x, 4, 4, W - 8, H - 8, 40); x.fillStyle = bg; x.fill();
  // tail
  const tx = side === 'quill' ? 70 : W - 70;
  x.beginPath(); x.moveTo(tx - 22, H - 10); x.lineTo(tx + (side === 'quill' ? -16 : 16), H + tail - 4); x.lineTo(tx + 22, H - 10); x.closePath(); x.fill();
  x.lineWidth = 5; x.strokeStyle = bd; rr(x, 4, 4, W - 8, H - 8, 40); x.stroke();
  x.font = `600 ${F}px Inter`; x.textBaseline = 'middle';
  let px = pad; for (const [s, col] of runs) { x.fillStyle = grey ? fg : (col || fg); x.fillText(s, px, H / 2 + 3); px += x.measureText(s).width; }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.userData = { aspect: c.width / c.height }; return t;
}
function plane(tex, h) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(h * tex.userData.aspect, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false, fog: false }));
  m.renderOrder = 26; m.visible = false; m.userData.w = h * tex.userData.aspect; return m;
}

export async function create({ hud }) {
  try { await document.fonts.load('600 64px Inter'); await document.fonts.load('700 84px Inter'); } catch (_) { /* optional */ }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  scene.add(camera);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group); q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  link.pulsePts.material.size = 0.8;

  const BH = 1.12; // bubble height incl. tail (world units): about 104 px at the chat framing, text about 42 px
  const loopQ = bubbleTexture([[LOOP]], 'quill'), loopE = bubbleTexture([[LOOP]], 'ember');
  const loopG = { quill: bubbleTexture([[LOOP]], 'quill', { grey: true }), ember: bubbleTexture([[LOOP]], 'ember', { grey: true }) };
  const before = T_B.map((_, i) => { const side = i % 2 ? 'quill' : 'ember'; const m = plane(side === 'quill' ? loopQ : loopE, BH); const g = plane(loopG[side], BH); scene.add(m, g); return { side, m, g }; });
  const after = LINES.map(([side, runs]) => { const m = plane(bubbleTexture(runs, side), BH); scene.add(m); return { side, m }; });
  const osloCard = cardPlane(cityCardTexture('Oslo', 'quill'), 0.72); scene.add(osloCard);
  const spr = (col) => { const m = new THREE.Sprite(new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })); m.visible = false; scene.add(m); return m; };
  const osloFlash = spr(new THREE.Color('#cdb8ff')), feed = spr(VIO);

  // HUD: labels and the stat (screen-fixed, big)
  const lab = (txt, col) => el(hud, 'div', { position: 'absolute', left: '110px', top: '64px', font: `650 84px ${FONT.display}`, letterSpacing: '-0.02em', color: col, textShadow: '0 2px 30px rgba(0,0,0,0.7)', display: 'none' }, txt);
  const lBefore = lab('before chat training', 'rgba(220,222,232,0.92)'), lAfter = lab('after chat training', '#f5f3ff');
  // the echo counter: ×2 … ×6, one tick per identical repeat
  const counter = el(hud, 'div', { position: 'absolute', left: '1490px', top: '150px', font: `800 132px ${FONT.display}`, letterSpacing: '-0.03em', color: 'rgba(214,218,232,0.95)', textShadow: '0 2px 30px rgba(0,0,0,0.8)', display: 'none', transformOrigin: '0% 50%' }, '');
  const stat = el(hud, 'div', { position: 'absolute', left: '0px', top: '96px', width: '1920px', textAlign: 'center', whiteSpace: 'nowrap', display: 'none', textShadow: '0 2px 30px rgba(0,0,0,0.85)' },
    `<div style="font:600 52px ${FONT.text};color:rgba(245,243,255,0.86);margin-bottom:6px">repeated turns</div>` +
    `<span style="font:800 120px ${FONT.display};color:rgba(200,204,220,0.75);letter-spacing:-0.035em">17%</span>` +
    `<svg width="120" height="56" viewBox="0 0 120 56" style="vertical-align:middle;margin:0 30px"><path d="M6 28 H100 M80 8 L108 28 L80 48" fill="none" stroke="#34f5c0" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>` +
    `<span style="font:800 120px ${FONT.display};color:#34f5c0;letter-spacing:-0.035em">1%</span>`);

  const reset = (c) => { Object.assign(c.pose, Creature.defaultPose(c.side)); c.blockGlow.fill(0); for (let i = 0; i < NBLOCKS; i++) c.blockColor[i] = null; c.inject = []; };
  const camR = new THREE.Vector3(), camU = new THREE.Vector3();
  const place = (m, pos, s, op) => { m.visible = s > 0.002 && op > 0.002; if (!m.visible) return; m.position.copy(pos); m.quaternion.copy(camera.quaternion); m.scale.setScalar(s); m.material.opacity = op; };
  const showDiv = (d, o, dy = 0) => { d.style.display = o > 0.002 ? 'block' : 'none'; d.style.opacity = o; d.style.transform = `translateY(${dy}px)`; };

  return {
    scene, camera,
    update(t) {
      world.update(t);
      const post = { bloomStrength: 0.85, bloomRadius: 0.5, bloomThreshold: 0.3, vignette: 0.95 };

      // ---------------- camera: a6's framing -> the chat framing (room above the heads) -> a6's framing at 92 ----------------
      const cF = { p: V(0.7, 3.4, 15.6), l: V(0.25, 3.0, 0) };
      const cC = { p: V(0.3, 4.95, 18.6), l: V(0.2, 4.45, 0) };
      const k1 = easeInOutCubic(prog(t, 88.0, 88.375, (x) => x)), k2 = easeInOutCubic(prog(t, 91.3, 92.0, (x) => x));
      camera.position.copy(cF.p).lerp(cC.p, k1).lerp(cF.p, k2);
      camera.lookAt(cF.l.clone().lerp(cC.l, k1).lerp(cF.l, k2));
      camera.updateMatrixWorld(true);
      camR.set(1, 0, 0).applyQuaternion(camera.quaternion); camU.set(0, 1, 0).applyQuaternion(camera.quaternion);

      // ---------------- creatures ----------------
      reset(q); reset(e);
      const isBefore = t < 89.5, liveK = prog(t, 89.5, 89.75);
      const bp = beatPulse(t, 6);
      for (const [c, seed] of [[q, 0], [e, 1.4]]) {
        const p = c.pose;
        p.xray = 0.6; p.portGlow = 1.1; p.groundGlow = 0.45;
        p.sway = lerp(0.02, 0.1, liveK); p.swayPhase = t * 2.3 + seed * 2.1; p.swayFreq = 0.8;
        p.squash = 1 - 0.022 * bp;
        p.blink = Math.max(blink(t, seed + 7), 0.45 * (1 - liveK) * prog(t, 88.2, 88.4)); // half-lidded while echoing
        p.headTilt = lerp(0, 0.06 * Math.sin(t * 1.3 + seed), liveK);
      }
      // before: a stiff bob on every echoed turn, both at once
      const bob = T_B.reduce((a, tb) => a + Math.max(0, kick(t, tb, 4, 9)), 0);
      for (const c of [q, e]) { c.pose.headNod += 0.12 * bob * (1 - liveK); c.pose.squash *= 1 - 0.03 * bob * (1 - liveK); }
      let qLook = e.headWorld(), eLook = q.headWorld();
      // the speaker of the latest bubble looks at it, the other at the speaker
      const hq = [], he = [];
      let qHappy = 0, eHappy = 0;
      // after: lively, they lean in, react to each new line
      T_A.forEach((ta, i) => { const sp = LINES[i][0]; (sp === 'ember' ? hq : he).push([ta + 0.125, 0.18, 0.25]); });
      eHappy = win(t, 90.05, 91.8, 0.12); qHappy = win(t, 90.1, 91.8, 0.12);
      q.pose.lean = 0.06 * liveK * (1 - prog(t, 91.6, 92)); e.pose.lean = -0.06 * liveK * (1 - prog(t, 91.6, 92));
      if (t > 89.4 && t < 89.95) { eLook = e.blockWorld(20).add(V(0, 0, 1)); }
      if (t > 91.8) { qLook = e.headWorld(); eLook = q.headWorld(); qHappy *= 1; }
      const hQ = hops(t, hq), hE = hops(t, he);
      q.pose.position.y += hQ.y; q.pose.squash *= hQ.sq; e.pose.position.y += hE.y; e.pose.squash *= hE.sq;
      q.pose.happy = qHappy; e.pose.happy = eHappy;
      for (const c of [q, e]) if (c.pose.happy > 0.5) c.pose.blink = 0;
      q.pose.lookAt = qLook; e.pose.lookAt = eLook;

      // ---------------- the link: both ways always; a pass before every new line ----------------
      const ls = link.state;
      ls.depth = 1; ls.grow = 1; ls.sag = -0.2; ls.twist = 0.07; ls.highlight = -1; ls.tintAB = null; ls.tintBA = null; ls.pulses = [];
      ls.aToB = 1; ls.bToA = 1;
      ls.intensity = lerp(0.55, 0.42, prog(t, 88, 88.4)) + 0.58 * prog(t, 91.375, 92); ls.packets = t > 91.5 ? 4 : 3; ls.flow = lerp(1, 1.2, prog(t, 91.4, 92)); ls.packetSize = 0.16;
      const passes = [[PASS0[0], PASS0[1] - PASS0[0], 1.6, true]].concat(T_A.slice(1).map((ta) => [ta - 0.25, 0.22, 0.9, false]));
      let wave = -1;
      for (const [t0, dur, amp, oslo] of passes) {
        const w = (t - t0) / dur; if (w >= 0 && w <= 1.08) wave = w;
        XBLOCKS.forEach((b, k) => {
          const tc = t0 + dur * UX[k];
          if (t > tc - 0.04 && t < tc + 0.26) {
            const uu = clamp((t - tc + 0.04) / 0.22);
            ls.pulses.push({ e: k, dir: 0, u: uu, color: VIO, amount: oslo ? 2.6 : 1.3 }, { e: k, dir: 1, u: uu, color: ORG, amount: oslo ? 1.8 : 1.3 });
          }
          // Oslo is written into Ember's core at each depth and rises; Ember's state goes into Quill's
          if (oslo && t > tc + 0.08) {
            const a = prog(t, tc + 0.08, tc + 0.2) * (1 - prog(t, 91.2, 91.8)), top = t < t0 + dur ? Math.max(UX[k] + 0.02, w) : 1.01;
            e.inject.push({ block: b, color: VIO, amount: 0.95 * a, top }); q.inject.push({ block: b, color: ORG, amount: 0.45 * a, top });
          }
        });
      }
      q.pose.wave = e.pose.wave = wave;
      if (wave > -0.5) ls.intensity = UX.map((u) => 0.42 + 0.9 * Math.exp(-Math.pow((u - wave) * 9, 2)));
      q.update(t); e.update(t);
      link.update(t);

      // ---------------- Quill's private city at its tail (both chats; Ember's prompt holds nothing) ----------------
      const cardOp = prog(t, 88.125, 88.3) * (1 - prog(t, 91.5, 91.85));
      const cardPos = V(-2.3, 0.62, 1.1);
      const rd = t > PASS0[0] ? Math.exp(-(t - PASS0[0]) * 5) : 0;
      osloCard.material.color.setScalar(0.9 + 0.5 * rd);
      place(osloCard, cardPos, (t > 88.125 ? Math.min(1, easeOutExpo(prog(t, 88.125, 88.4))) : 0) * (1 + 0.08 * rd), cardOp);
      { const u = clamp((t - PASS0[0] + 0.06) / 0.2), on = t > PASS0[0] - 0.06 && t < PASS0[0] + 0.14; feed.visible = on; if (on) { feed.position.copy(cardPos).lerp(q.tailWorld(), u); feed.scale.setScalar(0.7); feed.material.opacity = 1; } }

      // ---------------- bubbles ----------------
      const sideX = (side, w) => (side === 'quill' ? -5.6 + w / 2 : 5.6 - w / 2);
      // before: the same sentence ping-pongs; each repeat greys and joins an echo stack that climbs and recedes
      const bOut = prog(t, 89.375, 89.625);
      before.forEach((b, i) => {
        const tb = T_B[i];
        if (t < tb) { b.m.visible = b.g.visible = false; return; }
        const age = T_B.slice(i + 1).reduce((acc, tn) => acc + easeOutCubic(prog(t, tn, tn + 0.1)), 0); // repeats since this one
        const base = V(sideX(b.side, b.m.userData.w), 6.75, 0.6);
        const pos = base.clone().add(V(0.25 * (age % 2 ? 1 : -1) * Math.min(1, age), 0.4 * age, -0.9 * age));
        const s = (t < tb + 0.1 ? easeOutCubic(prog(t, tb, tb + 0.1)) * 1.06 : 1) * (1 - 0.05 * age);
        const greyK = clamp(age);
        const fade = (1 - bOut) * (1 - 0.1 * age);
        place(b.m, pos, s, fade * (1 - greyK));
        place(b.g, pos.clone().add(V(0, 0, -0.01)), s, fade * greyK * 0.9);
      });
      // after: four new lines, each new, stacking like a chat
      const aOut = prog(t, 91.65, 91.9);
      after.forEach((a, i) => {
        const ta = T_A[i];
        if (t < ta) { a.m.visible = false; return; }
        const lift = T_A.slice(i + 1).reduce((acc, tn) => acc + easeOutCubic(prog(t, tn, tn + 0.15)), 0);
        const pos = V(sideX(a.side, a.m.userData.w), 6.75 + 1.2 * lift, 0.6);
        const s = easeOutExpo(prog(t, ta, ta + 0.18)) * (1 + 0.06 * Math.max(0, kick(t, ta + 0.1, 2.4, 5)));
        place(a.m, pos, s, (1 - aOut) * (1 - 0.2 * Math.min(2, lift)) * (1 - clamp(lift - 2)));
      });
      // Oslo lands in Ember's first line: a violet burst on the word
      {
        const f = t > T_A[0] ? Math.exp(-(t - T_A[0]) * 3.5) : 0, a0 = after[0].m;
        osloFlash.visible = f > 0.02 && a0.visible;
        if (osloFlash.visible) { osloFlash.position.copy(a0.position).addScaledVector(camR, a0.userData.w * 0.33); osloFlash.scale.setScalar(1.0 + 2.2 * (1 - f)); osloFlash.material.opacity = 0.75 * f; }
      }

      // ---------------- HUD ----------------
      showDiv(lBefore, win(t, LBL.before[0], LBL.before[1], 0.15) * (t > 88.0 ? 1 : 0));
      {
        const n = T_B.filter((x) => x <= t).length, last = T_B[Math.max(0, n - 1)];
        const o = n >= 2 ? win(t, T_B[1], 89.375, 0.08) : 0;
        counter.style.display = o > 0.002 ? 'block' : 'none';
        if (o > 0.002) { counter.textContent = `\u00d7${n}`; counter.style.opacity = o; counter.style.transform = `scale(${1 + 0.18 * Math.max(0, kick(t, last, 3, 9))})`; }
      }
      showDiv(lAfter, win(t, LBL.after[0], LBL.after[1], 0.15));
      { const o = win(t, STAT[0], STAT[1], 0.12); showDiv(stat, o, (1 - easeOutCubic(prog(t, STAT[0], STAT[0] + 0.3))) * -16); }
      return post;
    },
  };
}
