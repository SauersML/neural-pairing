// a1_meet · 0–12 s (visual-first cold open)
// 0–4   the void wakes, two sparks ignite, vertebrae (= the 36 blocks) fly in and stack, heads pop on, eyes open. No captions;
//       name pins only once the bodies are built.
// 4–7   "a model", without words: a prompt enters each tail, one forward-pass wave climbs both spines in lockstep
//       (pose.wave, same value for both), and a token pops out of each head at the same instant.
// 7–10  Quill tries to toss its token to Ember: a hex glass field flashes up and the token shatters (hit at 8.0).
// 10–12 hold: the bodies turn glassy (xray 0 -> 0.7), cores revealed, the 8 port rings light, and only now: "neuralese".
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { Creature, NBLOCKS } from '../lib/creature.js';
import { makeWorld } from '../lib/env.js';
import { el, toScreen, FONT } from '../lib/hud.js';
import { tokenTile } from '../lib/tokens.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, smoother, hash, noise1, easeInCubic, easeOutCubic, easeInOutCubic } from '../lib/util.js';
import { Particles, Shield, Shatter, Callout, makeSpline, flareTexture } from './s1_meet_fx.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const SIDES = ['quill', 'ember'];

// ---- timeline (global seconds)
const QA = [0.4, 2.8], EA = [0.8, 3.2];          // assembly windows (assemble 0 -> 1)
const SPARK = { quill: 0.2, ember: 0.6 };         // spark ignition
const EYES = { quill: 2.95, ember: 3.15 };        // eyes open
const PROMPT = { quill: ['The', 'sky', 'is'], ember: ['Once', 'upon', 'a'] };
const OUT = { quill: 'blue', ember: 'time' };
const T_PROMPT = 4.2, T_SWALLOW = 4.6;           // prompt tiles appear, then enter the tails (both together)
const W_ARM = 5.0, W0 = 5.3, W1 = 6.4;         // forward pass: core dims (armed), wave climbs W0 -> W1
const T_OUT = 6.45;                                // both heads emit their next token at the same instant
const T_THROW = 7.15, T_HIT = 8.0;                // Quill tosses its token; it shatters on the field
const T_SHIELD = 7.62;                            // field flashes up
const T_GLASS = 9.9, T_PORTS = 10.15;             // hold: glassy bodies, port rings ignite
const T_WORD = 10.35;                             // "neuralese"

const landT = (i, A) => A[0] + (A[1] - A[0]) * (1 + 0.7 * i / NBLOCKS) / 1.7; // when block i settles (lib's stagger)
const swallowT = (i) => T_SWALLOW + i * 0.16;
const waveAt = (t) => (t < W_ARM ? -1 : t < W0 ? 0 : t < 7.0 ? Math.min(1.03, 1.03 * (t - W0) / (W1 - W0)) : -1);

// ---- camera: push-in & rise, orbit during the pass, 3/4 toss view, wide settle for the hold
const CAM_KEYS = [
  { t: 0.0, p: V(0.0, 1.2, 21.5), g: V(0, 2.2, 0) },
  { t: 1.8, p: V(0.4, 1.9, 18.4), g: V(0, 2.25, 0) },
  { t: 3.6, p: V(1.3, 2.7, 16.0), g: V(0, 2.5, 0) },
  { t: 5.4, p: V(-0.4, 3.0, 14.9), g: V(0, 2.65, 0) },
  { t: 6.8, p: V(-2.6, 3.4, 15.0), g: V(-0.3, 2.85, 0) },
  { t: 7.7, p: V(-2.6, 3.5, 14.2), g: V(-0.2, 3.0, 0) },
  { t: 8.6, p: V(-2.0, 3.3, 14.2), g: V(-0.2, 2.85, 0) },
  { t: 10.0, p: V(-0.6, 3.0, 16.0), g: V(0, 2.35, 0) },
  { t: 12.0, p: V(0.5, 3.0, 15.2), g: V(0, 2.4, 0) },
];
const camSpline = makeSpline(CAM_KEYS);
function cameraAt(t, cam) {
  const p = camSpline(t, (k) => k.p), g = camSpline(t, (k) => k.g);
  p.x += noise1(t * 0.55, 1) * 0.07; p.y += noise1(t * 0.5, 2) * 0.05; g.x += noise1(t * 0.4, 3) * 0.03;
  const sh = kick(t, T_HIT, 9, 6.5); // impact shake
  p.x += sh * 0.09; p.y += kick(t, T_HIT, 12, 7) * 0.06; g.y += sh * 0.03;
  cam.position.copy(p); cam.up.set(Math.sin(kick(t, T_HIT, 7, 6) * 0.012), 1, 0).normalize(); cam.lookAt(g);
  cam.fov = 35 - 1.2 * prog(t, 7.2, 8.0) + 1.2 * prog(t, 8.0, 9.4);
  cam.updateProjectionMatrix(); cam.updateMatrixWorld(true);
}

export async function create({ hud, scale = 1 }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const probe = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200); // evaluates the camera at other times (pure)
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  const C = { quill: q, ember: e };
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const pxScale = 1080 * scale / (2 * Math.tan(THREE.MathUtils.degToRad(17.5)));
  const LAV = new THREE.Color('#d9ceff');

  // ---- sparks + ignition bursts + bokeh
  const flareTex = flareTexture();
  const sparks = {};
  for (const side of SIDES) {
    const m = new THREE.SpriteMaterial({ map: flareTex, color: side === 'quill' ? new THREE.Color('#d9ccff') : new THREE.Color('#ffd3a8'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false });
    const s = new THREE.Sprite(m); scene.add(s); sparks[side] = s;
  }
  const burst = new Particles(2 * 48, pxScale); scene.add(burst.points);
  const bokeh = new Particles(70, pxScale); scene.add(bokeh.points);

  // ---- prompt tiles (in at the tail) and output tiles (out of the head); Quill's output is the token it tosses
  const prompts = {}, outs = {};
  for (const side of SIDES) {
    prompts[side] = PROMPT[side].map((w) => { const m = tokenTile(w, side, 0.68); m.renderOrder = 5; scene.add(m); return m; });
    const o = tokenTile(OUT[side], side, 0.76); o.renderOrder = 6; scene.add(o); outs[side] = o;
  }
  const tile = outs.quill, tileW = tile.userData.width, tileH = 0.76;
  const tileGlowMat = new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: new THREE.Color('#a78bfa'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const tileGlow = new THREE.Sprite(tileGlowMat); scene.add(tileGlow);
  const trail = new Particles(40, pxScale); scene.add(trail.points);
  // each block writes into the core as the wave passes it
  const feed = {};
  for (const side of SIDES) { feed[side] = new Particles(NBLOCKS * 4, pxScale); scene.add(feed[side].points); }

  // ---- glass field and shatter
  const shield = new Shield({ W: 4.2, H: 5.3 });
  shield.group.position.set(0.8, 0, 0.1); shield.group.rotation.y = -0.85; shield.group.updateMatrixWorld(true);
  shield.mat.uniforms.uHot.value.set('#e6dcff');
  scene.add(shield.group);
  const IMP = new THREE.Vector2(0.0, 3.8);
  shield.mat.uniforms.uImp.value.copy(IMP);
  const impactW = shield.localToWorld(IMP.x, IMP.y);
  const wallN = shield.normalWorld(); // faces Quill
  const shatter = new Shatter(tile.material.map, tileW, tileH, { nx: 8, ny: 3, seed: 11 });
  shatter.mat.uniforms.uHot.value.set('#c9b6ff');
  scene.add(shatter.mesh);
  const dust = new Particles(460, pxScale); scene.add(dust.points);
  const hitSparks = new Particles(110, pxScale); scene.add(hitSparks.points);
  const hitFlareMat = new THREE.SpriteMaterial({ map: flareTex, color: new THREE.Color('#e4dcff'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false });
  const hitFlare = new THREE.Sprite(hitFlareMat); scene.add(hitFlare);

  // ---- HUD: name pins, and the one word
  const nameQ = new Callout(hud, 'Quill', { color: '#c4b5fd', size: 52, font: FONT.display, weight: 700, side: 'left', sub: 'Qwen3-4B', subColor: 'rgba(221,214,254,0.9)', subSize: 32 });
  const nameE = new Callout(hud, 'Ember', { color: '#fdba74', size: 52, font: FONT.display, weight: 700, side: 'right', sub: 'AFM-4.5B', subColor: 'rgba(254,215,170,0.9)', subSize: 32 });
  const word = el(hud, 'div', { position: 'absolute', left: '0px', width: '1920px', bottom: '66px', textAlign: 'center', pointerEvents: 'none' });
  const wordMain = el(word, 'div', { font: `700 136px ${FONT.display}`, letterSpacing: '-0.015em', lineHeight: 1.0, color: '#34f5c0', textShadow: '0 0 46px rgba(52,245,192,0.45), 0 2px 26px rgba(0,0,0,0.85)' });
  const letters = [...'neuralese'].map((ch) => el(wordMain, 'span', { display: 'inline-block' }, ch));
  const wordSub = el(word, 'div', { font: `500 40px ${FONT.text}`, color: 'rgba(245,243,255,0.82)', marginTop: '10px', textShadow: '0 2px 20px rgba(0,0,0,0.9)' }, '');
  wordSub.style.display = 'none';

  // ---------------------------------------------------------------- helpers
  const headApprox = (side) => V(side === 'quill' ? -2.3 : 2.3, 4.85, 0);
  function gaze(t, sched, pts) {
    let i = 0; while (i < sched.length - 1 && t >= sched[i + 1][0]) i++;
    const cur = pts[sched[i][1]], prev = pts[sched[Math.max(0, i - 1)][1]];
    return prev.clone().lerp(cur, i === 0 ? 1 : smoother((t - sched[i][0]) / 0.24));
  }
  function aimFace(c, target, k = 0.7) {
    const d = target.clone().sub(V(c.pose.position.x, 4.8, c.pose.position.z));
    let fy = Math.atan2(d.x, d.z) - c.pose.yaw; fy = Math.atan2(Math.sin(fy), Math.cos(fy));
    c.pose.faceYaw = clamp(fy * k, -0.85, 0.85);
    c.pose.headNod += clamp(Math.atan2(-d.y, Math.hypot(d.x, d.z)) * 0.55, -0.3, 0.45);
  }
  // Quill's token: written out of its head, held up, drawn back, then lobbed at Ember (all fixed points: pure in t)
  const outPos = (side, t) => headApprox(side).add(V(0, 0.55 + 0.45 * clamp(spring(t, T_OUT, 2.3, 0.42), 0, 1.3), 0.75));
  const hoverPoint = (t) => V(-1.95, 5.6 + Math.sin(t * 6) * 0.04, 0.75);
  const WIND = V(-0.85, -0.25, -0.3);
  function tilePos(t) {
    if (t < T_THROW) {
      const toHover = prog(t, 6.75, 6.98, easeInOutCubic), wind = prog(t, 6.92, T_THROW, easeInOutCubic);
      return outPos('quill', t).lerp(hoverPoint(t), toHover).addScaledVector(WIND, wind);
    }
    const u = clamp((t - T_THROW) / (T_HIT - T_THROW));
    const a = hoverPoint(T_THROW).add(WIND), b = impactW.clone().addScaledVector(wallN, 0.12);
    const mid = a.clone().lerp(b, 0.45); mid.y += 1.5;
    const uu = u * (0.85 + 0.15 * u);
    return a.clone().multiplyScalar((1 - uu) * (1 - uu)).addScaledVector(mid, 2 * uu * (1 - uu)).addScaledVector(b, uu * uu);
  }
  const tileSpin = (t) => (t < T_THROW ? Math.sin(t * 5) * 0.06 : -0.28 * Math.sin(Math.PI * 2 * clamp((t - T_THROW) / (T_HIT - T_THROW))) * (1 - clamp((t - T_THROW) / (T_HIT - T_THROW))));
  // shatter frame at impact (from the camera at T_HIT)
  cameraAt(T_HIT, probe);
  const camQ8 = probe.quaternion.clone();
  const frame = {
    origin: tilePos(T_HIT - 1e-4),
    right: V(1, 0, 0).applyQuaternion(camQ8), up: V(0, 1, 0).applyQuaternion(camQ8),
    normal: wallN.clone().lerp(V(0, 0, 1).applyQuaternion(camQ8), 0.35).normalize(),
  };
  const shardCenter = (k, age, out) => {
    const d = shatter.dyn[k]; const hx = d.cx / tileW + 0.5;
    const radial = V(0, 0, 0).addScaledVector(frame.right, d.cx * 2.2 + (d.r3 - 0.5) * 0.6).addScaledVector(frame.up, d.cy * 3.0 + (d.r4 - 0.3) * 1.2);
    const v = radial.multiplyScalar(2.0 + 2.8 * d.r1).addScaledVector(frame.normal, 2.0 + 4.0 * d.r2 * (0.4 + hx));
    const fall = (1 - Math.exp(-2.6 * age)) / 2.6;
    out.copy(frame.origin).addScaledVector(frame.right, d.cx).addScaledVector(frame.up, d.cy).addScaledVector(v, fall);
    out.y += 0.5 * -3.2 * age * age * 0.45;
    return { out, v };
  };

  // ---------------------------------------------------------------- update
  return {
    scene, camera,
    update(t) {
      cameraAt(t, camera);
      world.update(t);
      // the void wakes up
      const wake = prog(t, 0.0, 2.6);
      world.dustMat.opacity = 0.55 * prog(t, 0.1, 1.6);
      world.sky.material.uniforms.uAmp.value = lerp(0.25, 1, wake);
      world.key.intensity = lerp(0.25, 1.6, prog(t, 0.6, 3.2)) * lerp(1, 0.75, prog(t, T_GLASS, T_GLASS + 1));
      world.hemi.intensity = lerp(0.1, 0.5, wake);
      for (const [side, L] of [['quill', world.rimQ], ['ember', world.rimE]]) {
        const s0 = SPARK[side];
        L.intensity = 30 * prog(t, s0, s0 + 1.4) + (t > s0 ? 25 * Math.exp(-(t - s0) * 4) : 0);
      }
      if (world.floor) world.floor.material.uniforms.uAmp.value = prog(t, 0.6, 3.0);

      const wave = waveAt(t);
      const tileP = tilePos(t);
      const pts = {
        cam: camera.position.clone(),
        partnerQ: headApprox('ember'), partnerE: headApprox('quill'),
        qPrompt: V(-2.3, 0.4, 1.2), ePrompt: V(2.3, 0.4, 1.2),
        qOut: V(-2.3, 6.0, 1.4), eOut: V(2.3, 6.0, 1.4),
        tile: tileP, shards: impactW.clone().addScaledVector(wallN, 0.9).add(V(0, -0.6, 0)),
        qPort: V(-2.3, 1.8, 1.5), ePort: V(2.3, 1.8, 1.5), hit: impactW.clone(),
      };
      const schedQ = [[0, 'cam'], [3.55, 'partnerQ'], [4.25, 'qPrompt'], [5.3, 'partnerQ'], [6.45, 'qOut'], [6.75, 'tile'], [T_HIT + 0.05, 'shards'], [8.5, 'partnerQ'], [10.25, 'qPort'], [10.95, 'partnerQ']];
      const schedE = [[0, 'cam'], [3.7, 'partnerE'], [4.3, 'ePrompt'], [5.35, 'partnerE'], [6.45, 'eOut'], [6.85, 'tile'], [T_HIT, 'hit'], [8.55, 'partnerE'], [10.35, 'ePort'], [11.05, 'partnerE']];
      const glass = prog(t, T_GLASS, T_GLASS + 1.0, smooth);

      // ------------------------------------------------ creature poses
      for (const side of SIDES) {
        const c = C[side], p = c.pose, A = side === 'quill' ? QA : EA;
        Object.assign(p, Creature.defaultPose(side));
        p.assemble = clamp((t - A[0]) / (A[1] - A[0]));
        const settled = prog(t, A[1] - 0.4, A[1] + 0.8);
        p.squash = 1 + 0.02 * settled * (beatPulse(t, 5) - 0.35) + 0.07 * kick(t, A[1] - 0.05, 2.6, 5);
        p.sway = 0.1 * settled; p.swayPhase = t * (side === 'quill' ? 2.4 : 2.1) + (side === 'quill' ? 0 : 1);
        p.portGlow = 0.12; p.groundGlow = 0.5 * prog(t, SPARK[side], SPARK[side] + 1.2);
        p.xray = 0.7 * glass; p.wave = wave;
        p.coreGlow = 1.0 + 0.45 * glass; p.glow = 1 + 0.5 * glass;
        // eyes: closed until they open, then blinks
        const eo = EYES[side];
        const blinks = side === 'quill' ? [eo + 0.32, 4.1, 9.1, 11.35] : [eo + 0.42, 5.1, 9.5, 11.55];
        let bl = t < eo ? 1 : 1 - prog(t, eo, eo + 0.14);
        for (const b of blinks) bl = Math.max(bl, win(t, b, b + 0.17, 0.07));
        p.blink = bl;
        const g = gaze(t, side === 'quill' ? schedQ : schedE, pts);
        p.lookAt = g; aimFace(c, g, 0.75);
        // gulps as prompt tiles enter; a little stretch as the token pops out of the head
        for (let i = 0; i < 3; i++) p.squash -= 0.035 * Math.max(0, kick(t, swallowT(i) + 0.32, 3, 7));
        p.squash += 0.04 * kick(t, T_OUT - 0.05, 2.4, 5);
        p.headNod += -0.12 * kick(t, T_OUT - 0.05, 2.4, 5);
        // wonder at the end: a soft smile while the cores glow
        p.happy = 0.75 * win(t, 11.15, 12.4, 0.25);
      }
      // ---- Quill: grin, wind up, toss, startled
      {
        const p = q.pose;
        const wind = prog(t, 6.92, T_THROW, easeInOutCubic) * (1 - prog(t, T_THROW, T_THROW + 0.09));
        const flick = kick(t, T_THROW, 1.7, 4.2);
        p.lean += -0.13 * wind + 0.2 * flick;
        p.headNod += -0.18 * wind + 0.22 * flick;
        p.squash += -0.045 * wind + 0.06 * flick;
        p.happy = Math.max(p.happy, win(t, 6.7, 7.3, 0.15));
        const st = kick(t, T_HIT, 2.2, 4.5);
        p.lean += -0.09 * Math.abs(st); p.squash += 0.05 * st;
        p.headTilt = 0.2 * win(t, 8.6, 9.8, 0.25);
      }
      // ---- Ember: ready to catch, flinch, puzzled
      {
        const p = e.pose;
        const ready = win(t, 7.35, T_HIT, 0.25);
        p.lean += -0.06 * ready; p.squash += 0.025 * ready; p.happy = Math.max(p.happy, 0.85 * win(t, 7.3, T_HIT + 0.02, 0.2));
        const fl = spring(t, T_HIT, 1.6, 0.42) * (1 - prog(t, 8.45, 9.3));
        p.lean += 0.2 * fl; p.leanFwd = -0.08 * fl;
        p.squash += (t >= T_HIT ? -0.11 * Math.exp(-(t - T_HIT) * 5) : 0) + 0.03 * kick(t, T_HIT + 0.12, 2.5, 4);
        p.blink = Math.max(p.blink, win(t, T_HIT - 0.02, T_HIT + 0.36, 0.05));
        p.headNod += 0.18 * fl * 0.6;
        p.headTilt = -0.22 * win(t, 8.62, 9.8, 0.25);
      }
      // per-block landing flashes during assembly (the forward-pass glow comes from pose.wave)
      for (const side of SIDES) {
        const c = C[side], A = side === 'quill' ? QA : EA;
        for (let i = 0; i < NBLOCKS; i++) {
          const tl = landT(i, A);
          c.blockGlow[i] = t >= tl - 0.15 ? 0.9 * Math.exp(-(t - tl + 0.15) * 4) : (t > tl - 0.9 * (A[1] - A[0]) / 1.7 ? 0.35 : 0);
          c.blockGlow[i] += 0.9 * glass * (0.85 + 0.15 * Math.sin(t * 3 - i * 0.35));
          c.blockColor[i] = null;
        }
        c.inject = [];
      }

      q.pose.hair = e.pose.hair = 0.62; // relaxed before the link; it lifts when the link connects (a2_link)
      q.update(t); e.update(t);

      // ---- post-update touches (rebuilt every frame from t, so still pure)
      for (const side of SIDES) {
        const c = C[side], A = side === 'quill' ? QA : EA, asmT = c.pose.assemble;
        // residual-stream core grows block by block through landed vertebrae only (lib would string it through flying ones)
        const mf = clamp((1.7 * asmT - 0.97) * NBLOCKS / 0.7 + 1, 0, NBLOCKS);
        if (asmT < 0.999) {
          if (mf >= 1.6) {
            const n = Math.floor(mf), pts2 = [];
            for (let i = 0; i < n; i++) pts2.push(c._spineLocal((i + 0.5) / NBLOCKS, new THREE.Vector3()));
            if (n < NBLOCKS) pts2.push(c._spineLocal((n - 0.5 + (mf - n)) / NBLOCKS, new THREE.Vector3()));
            c.core.geometry.dispose();
            c.core.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts2), Math.max(8, pts2.length * 3), 0.075, 8, false);
            c.core.visible = true;
          } else c.core.visible = false;
        }
        // head pop wobble
        const wob = kick(t, A[1] - 0.12, 3.2, 5.5);
        c.head.scale.multiply(V(1 - 0.12 * wob, 1 + 0.18 * wob, 1 - 0.12 * wob));
        // tail light gulps
        let gulp = 0; for (let i = 0; i < 3; i++) { const ts = swallowT(i) + 0.32; if (t > ts) gulp += Math.exp(-(t - ts) * 6); }
        c.tail.scale.setScalar(0.9 * (1 + 0.8 * gulp));
        c.tailMat.opacity *= lerp(1, 0.45, prog(t, 5.3, 6.0));
        // port rings: dim until the hold, then ignite bottom-to-top and pulse (handoff to a2)
        c.ports.forEach((pt, k) => {
          const tk = T_PORTS + 0.08 * k + (side === 'ember' ? 0.03 : 0);
          const on = prog(t, tk, tk + 0.3);
          const fl = t >= tk ? Math.exp(-(t - tk) * 5) : 0;
          const pulse = 0.18 * on * beatPulse(t, 4);
          pt.mat.opacity = (lerp(0.12, 0.7, on) + 1.1 * fl + pulse) * (pt.mesh.visible ? 1 : 0);
          pt.mesh.scale.multiplyScalar(1 + 0.35 * fl + 0.05 * pulse);
        });
      }

      // ---- each block writes into the core as the wave passes it (sparks from the block into the stream)
      for (const side of SIDES) {
        const c = C[side], F = feed[side];
        const col = side === 'quill' ? new THREE.Color('#efe9ff') : new THREE.Color('#fff0dc');
        for (let i = 0; i < NBLOCKS; i++) {
          const center = c.blockWorld(i), r = c.blockRadius(i) * 1.3;
          const tw = W0 + ((i + 0.5) / NBLOCKS / 1.03) * (W1 - W0);
          for (let j = 0; j < 4; j++) {
            const k = i * 4 + j, ti = tw - 0.16 - 0.03 * j, age = t - ti, dur = 0.24;
            if (age < 0 || age > dur) { F.hide(k); continue; }
            const a = j * Math.PI / 2 + i * 0.9;
            const pp = center.clone().add(V(Math.cos(a) * r, 0.04, Math.sin(a) * r)).lerp(center, easeInCubic(age / dur));
            F.set(k, pp, col, 0.075, Math.sin(Math.PI * age / dur) * 1.1);
          }
        }
        F.commit();
      }

      // ---- sparks: ignite, ride the top of the growing spine, merge into the head
      for (const side of SIDES) {
        const c = C[side], A = side === 'quill' ? QA : EA, s = sparks[side], t0 = SPARK[side];
        const mf = clamp((1.7 * c.pose.assemble - 0.97) * NBLOCKS / 0.7 + 1, 0, NBLOCKS);
        const base = c.pose.position.clone().add(V(0, 0.18, 0));
        const top = mf >= 1 ? c.blockWorld(Math.min(NBLOCKS - 1, Math.floor(mf) - 1)) : base;
        s.position.copy(base.lerp(top, smooth(clamp(mf / 2))));
        const ign = t >= t0 ? spring(t, t0, 2.2, 0.32) : 0;
        const flash = t >= t0 ? Math.exp(-(t - t0) * 3.5) : 0;
        const fadeOut = 1 - prog(t, A[0] + 0.78 * (A[1] - A[0]), A[1] + 0.1);
        const flick = 0.85 + 0.15 * Math.sin(t * 37 + (side === 'quill' ? 0 : 2));
        s.scale.setScalar((0.8 + 1.3 * ign + 3.8 * flash) * flick * fadeOut + 0.001);
        s.material.opacity = clamp(ign + flash) * fadeOut;
        s.visible = t >= t0 && fadeOut > 0.001;
        for (let k = 0; k < 48; k++) {
          const i = (side === 'quill' ? 0 : 48) + k, age = t - t0;
          if (age < 0 || age > 1.3) { burst.hide(i); continue; }
          const a = hash(k * 3.3 + t0) * Math.PI * 2, el2 = (hash(k * 7.1 + t0) - 0.3) * 1.6, sp = 1.2 + 2.6 * hash(k * 1.7 + t0);
          const dir = V(Math.cos(a) * Math.cos(el2), Math.sin(el2), Math.sin(a) * Math.cos(el2));
          const pp = c.pose.position.clone().add(V(0, 0.18, 0)).addScaledVector(dir, sp * (1 - Math.exp(-3 * age)) / 3); pp.y -= 0.6 * age * age;
          burst.set(i, pp, side === 'quill' ? new THREE.Color('#d6c8ff') : new THREE.Color('#ffc48a'), 0.05 + 0.04 * hash(k), (1 - age / 1.3) * 1.2);
        }
      }
      burst.commit();

      // ---- bokeh motes for depth
      for (let k = 0; k < 70; k++) {
        const x = (hash(k * 1.1 + 4) - 0.5) * 22, y = hash(k * 2.3 + 1) * 9 - 0.5, z = (hash(k * 3.9 + 2) - 0.5) * 16 + 2;
        const pp = V(x + Math.sin(t * 0.3 + k) * 0.3, y + ((t * 0.08 * (0.5 + hash(k))) % 3), z);
        const hue = hash(k * 5.5);
        const col = hue < 0.4 ? new THREE.Color('#7c5cff') : hue < 0.8 ? new THREE.Color('#ff8a3d') : new THREE.Color('#c9c3ff');
        bokeh.set(k, pp, col, 0.25 + 0.5 * hash(k * 9.1), 0.11 * prog(t, 0.6, 2.4) * (0.5 + 0.5 * Math.sin(t * 0.9 + k * 2.1)));
      }
      bokeh.commit();

      // ---- prompt tiles: a row beside each tail, entering one by one (both creatures together)
      const camRight = V(1, 0, 0).applyQuaternion(camera.quaternion);
      for (const side of SIDES) {
        const c = C[side], tiles = prompts[side];
        const tail = c.pose.position.clone().add(V(0, 0.12, 0.1));
        const widths = tiles.map((m) => m.userData.width), gap = 0.08;
        const total = widths.reduce((a, b) => a + b, 0) + gap * (tiles.length - 1);
        const center = c.pose.position.clone().add(V(side === 'quill' ? -0.35 : 0.35, 1.25, 1.6));
        let acc = -total / 2;
        tiles.forEach((m, i) => {
          const home = center.clone().addScaledVector(camRight, acc + widths[i] / 2); acc += widths[i] + gap;
          const ta = T_PROMPT + i * 0.09, ts = swallowT(i);
          const app = spring(t, ta, 2.4, 0.45), sw = prog(t, ts, ts + 0.34, easeInCubic);
          m.visible = t >= ta && sw < 0.999;
          const pos = home.clone().add(V(0, 0.12 * Math.sin(t * 3 + i), 0)).lerp(tail, sw);
          pos.y += 0.25 * Math.sin(Math.PI * sw);
          m.position.copy(pos); m.quaternion.copy(camera.quaternion);
          m.scale.setScalar(Math.max(0.001, app * (1 - 0.85 * sw)));
          m.material.opacity = clamp(app) * (1 - sw * 0.3);
        });
      }
      // Ember's output token: pops out of the head, then drifts up and fades
      {
        const o = outs.ember, pop = spring(t, T_OUT, 2.3, 0.42), gone = prog(t, 7.25, 7.6);
        o.visible = t >= T_OUT && gone < 0.999;
        o.position.copy(outPos('ember', t)).add(V(0, 0.4 * gone, 0));
        o.quaternion.copy(camera.quaternion);
        o.scale.setScalar(Math.max(0.001, pop * (1 - 0.3 * gone)));
        o.material.opacity = 1 - gone;
      }
      // Quill's output token = the one it tosses
      {
        const vis = t >= T_OUT && t < T_HIT;
        tile.visible = vis; tileGlow.visible = vis;
        const pop = spring(t, T_OUT, 2.3, 0.42);
        tile.position.copy(tileP);
        tile.quaternion.copy(camera.quaternion); tile.rotateZ(tileSpin(t));
        const stretch = t > T_THROW ? 1 + 0.1 * Math.sin(Math.PI * clamp((t - T_THROW) / (T_HIT - T_THROW))) : 1;
        tile.scale.set(Math.max(0.001, pop * stretch), Math.max(0.001, pop / stretch), 1);
        tile.material.opacity = 1;
        tileGlow.position.copy(tileP); tileGlow.scale.setScalar(1.3 * clamp(pop)); tileGlowMat.opacity = 0.2 * prog(t, 6.75, 7.0) + 0.3 * prog(t, 7.6, T_HIT);
        for (let k = 0; k < 40; k++) {
          const lag = k * 0.018, tt = t - lag;
          if (t < T_THROW || tt < T_THROW || t >= T_HIT) { trail.hide(k); continue; }
          const pp = tilePos(tt).add(V((hash(k) - 0.5) * 0.25, (hash(k * 2.2) - 0.5) * 0.25 - lag * 0.6, (hash(k * 3.1) - 0.5) * 0.2));
          trail.set(k, pp, LAV, 0.045 + 0.03 * hash(k * 4), (1 - k / 40) * 0.9);
        }
        trail.commit();
      }

      // ---- the glass field
      {
        const u = shield.mat.uniforms;
        u.uT.value = t; u.uOn.value = t - T_SHIELD; u.uAge.value = t - T_HIT;
        u.uOff.value = prog(t, 8.75, 9.55, (x) => x);
        u.uAmp.value = 0.5;
        shield.group.visible = t >= T_SHIELD - 0.01 && t < 9.6;
        const age = t - T_HIT;
        hitFlare.visible = age >= 0 && age < 0.8;
        hitFlare.position.copy(impactW).addScaledVector(wallN, 0.15);
        hitFlare.scale.setScalar(age >= 0 ? (1.4 * Math.exp(-age * 9) + 0.3) : 0.001);
        hitFlareMat.opacity = age >= 0 ? 0.6 * Math.exp(-age * 7) : 0;
      }

      // ---- shatter: shards + dust + sparks
      {
        const age = t - T_HIT;
        shatter.update(age, frame, { fadeStart: 0.35, fadeEnd: 1.5 });
        const nS = shatter.n;
        for (let j = 0; j < 460; j++) {
          const k = j % nS, born = 0.06 + 0.95 * hash(j * 1.37 + 3), life = 0.9 + 1.0 * hash(j * 2.9 + 1);
          const a = age - born;
          if (age < 0 || a < 0 || a > life) { dust.hide(j); continue; }
          const { out: o0, v } = shardCenter(k, born, new THREE.Vector3());
          const rnd = V(hash(j * 3.1) - 0.5, hash(j * 4.7) - 0.3, hash(j * 6.3) - 0.5).multiplyScalar(1.4);
          const pp = o0.addScaledVector(v.clone().multiplyScalar(0.2).add(rnd), (1 - Math.exp(-1.8 * a)) / 1.8); pp.y += 0.18 * a; pp.x += Math.sin(a * 3 + j) * 0.05 * a;
          const h = hash(j * 8.8);
          dust.set(j, pp, new THREE.Color().setRGB(0.78 + 0.22 * h, 0.7 + 0.25 * h, 1.0), 0.03 + 0.04 * hash(j * 5.5), Math.sin(Math.PI * clamp(a / life)) ** 0.7 * 1.1 * (1 - prog(t, 9.1, 9.8)));
        }
        dust.commit();
        const tang = V(0, 1, 0).cross(wallN).normalize();
        for (let j = 0; j < 110; j++) {
          const life = 0.25 + 0.5 * hash(j * 7.7);
          if (age < 0 || age > life) { hitSparks.hide(j); continue; }
          const a = hash(j * 1.9) * Math.PI * 2;
          const dir = tang.clone().multiplyScalar(Math.cos(a)).add(V(0, Math.sin(a), 0)).addScaledVector(wallN, 0.3 + 0.9 * hash(j * 3.3)).normalize();
          const pp = impactW.clone().addScaledVector(dir, (3 + 6 * hash(j * 2.6)) * (1 - Math.exp(-4 * age)) / 4); pp.y -= 1.5 * age * age;
          hitSparks.set(j, pp, new THREE.Color('#f1ebff'), 0.035, 1.4 * (1 - age / life));
        }
        hitSparks.commit();
      }

      // ------------------------------------------------ HUD
      const nameOn = (a, b) => ({ o: win(t, a, b, 0.25), d: prog(t, a, a + 0.7, (x) => x) });
      const nq = nameOn(3.0, 4.55), ne = nameOn(3.2, 4.6);
      nameQ.update(toScreen(q.headWorld().add(V(-0.55, 0.25, 0)), camera), { x: -110, y: -40 }, nq.o, nq.d);
      nameE.update(toScreen(e.headWorld().add(V(0.55, 0.25, 0)), camera), { x: 110, y: -40 }, ne.o, ne.d);
      // "neuralese": letters rise in one by one during the hold, then everything fades by 12
      const wOn = t >= T_WORD - 0.01 && t <= 12.05;
      word.style.display = wOn ? 'block' : 'none';
      if (wOn) {
        const out = prog(t, 11.82, 12.0);
        word.style.opacity = 1 - out; word.style.transform = `translateY(${-out * 14}px)`;
        letters.forEach((s, i) => { const p = easeOutCubic(clamp((t - T_WORD - i * 0.035) / 0.45)); s.style.opacity = p; s.style.transform = `translateY(${(1 - p) * 0.35}em)`; s.style.filter = `blur(${(1 - p) * 10}px)`; });
        const sp = easeOutCubic(clamp((t - T_WORD - 0.35) / 0.4));
        wordSub.style.opacity = sp; wordSub.style.transform = `translateY(${(1 - sp) * 12}px)`;
      }

      // ------------------------------------------------ post
      const hit = t >= T_HIT ? Math.exp(-(t - T_HIT) * 7) : 0;
      return {
        fade: 1 - prog(t, 0.0, 0.8, (x) => 1 - Math.pow(1 - x, 2)),
        bloomStrength: 0.95 + 0.45 * hit + (t > SPARK.quill ? 0.5 * Math.exp(-(t - SPARK.quill) * 2.5) : 0) + 0.15 * glass,
        bloomRadius: 0.55, bloomThreshold: 0.18,
        exposure: 1.0,
        vignette: lerp(1.25, 0.9, prog(t, 0, 3)),
        aberration: 0.0018 + 0.007 * hit,
        flash: 0.14 * hit, flashColor: '#f1ebff',
        grain: 0.045,
      };
    },
  };
}
