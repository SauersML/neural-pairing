// S1 · s1_meet · 0–10 s · "01 / THE PAIR"
// Cold open: the void wakes, two sparks ignite, vertebrae (= the 36 transformer blocks) fly in and stack into two
// spines, heads pop on, eyes open. Then names, a block ruler and the residual stream; each model swallows a prompt at
// its tail and writes the next token out of its head. Finally Quill tries to toss a token to Ember, a hex force-field
// flashes up, and the token shatters at 8.0: only hidden states cross the link. The 8 port rings light up for the handoff to S2.
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { Creature, NBLOCKS, XBLOCKS } from '../lib/creature.js';
import { makeWorld } from '../lib/env.js';
import { Caption, el, toScreen, FONT } from '../lib/hud.js';
import { tokenTile } from '../lib/tokens.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, smoother, hash, noise1, easeInCubic, easeOutCubic, easeOutBack, easeInOutCubic } from '../lib/util.js';
import { Particles, Shield, Shatter, Callout, makeSpline, flareTexture } from './s1_meet_fx.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);

// ---- timeline (global seconds)
const QA = [0.4, 2.8], EA = [0.8, 3.2];          // assembly windows (assemble 0 -> 1)
const SPARK = { quill: 0.2, ember: 0.6 };         // spark ignition
const EYES = { quill: 2.95, ember: 3.15 };        // eyes open
const PROMPT = { quill: { words: ['The', 'sky', 'is'], out: 'blue', t0: 3.72, swallow: 4.25 }, ember: { words: ['Once', 'upon', 'a'], out: 'time', t0: 3.8, swallow: 4.35 } };
const WAVE_DUR = 0.9;
const T_POP = 6.66, T_THROW = 7.06, T_HIT = 8.0;  // the toss
const T_SHIELD = 7.62;                            // force-field flashes up
const T_PORTS = 8.95;                             // port rings ignite (handoff)

const landT = (i, A) => A[0] + (A[1] - A[0]) * (1 + 0.7 * i / NBLOCKS) / 1.7; // when block i settles (lib's stagger)
const swallowT = (side, i) => PROMPT[side].swallow + i * 0.16;
const waveStart = (side) => swallowT(side, 2) + 0.34;
const outT = (side) => waveStart(side) + WAVE_DUR + 0.05;

// ---- camera keys: push-in & rise, orbit, toss view, settle to the home framing used by the next shot
const CAM_KEYS = [
  { t: 0.0, p: V(-0.3, 1.2, 21.5), g: V(-0.5, 2.2, 0) },
  { t: 1.8, p: V(0.2, 1.9, 18.6), g: V(-0.5, 2.1, 0) },
  { t: 3.4, p: V(0.6, 2.6, 16.8), g: V(-0.7, 1.95, 0) },
  { t: 5.0, p: V(-1.0, 2.9, 16.6), g: V(-0.9, 1.95, 0) },
  { t: 6.6, p: V(-2.6, 3.3, 15.6), g: V(-0.6, 2.3, 0) },
  { t: 7.6, p: V(-2.6, 3.4, 14.6), g: V(-0.4, 2.6, 0) },
  { t: 8.6, p: V(-2.2, 3.3, 14.4), g: V(-0.6, 2.55, 0) },
  { t: 10.0, p: V(-0.6, 3.2, 14.8), g: V(-0.7, 2.5, 0) },
];
const camSpline = makeSpline(CAM_KEYS);
function cameraAt(t, cam) {
  const p = camSpline(t, (k) => k.p), g = camSpline(t, (k) => k.g);
  p.x += noise1(t * 0.55, 1) * 0.07; p.y += noise1(t * 0.5, 2) * 0.05; g.x += noise1(t * 0.4, 3) * 0.03;
  // impact shake at 8.0
  const sh = kick(t, T_HIT, 9, 6.5);
  p.x += sh * 0.09; p.y += kick(t, T_HIT, 12, 7) * 0.06; g.y += sh * 0.03;
  cam.position.copy(p); cam.up.set(Math.sin(kick(t, T_HIT, 7, 6) * 0.012), 1, 0).normalize(); cam.lookAt(g);
  cam.fov = 35 - 1.2 * prog(t, 7.2, 8.0) + 1.2 * prog(t, 8.0, 9.4);
  cam.updateProjectionMatrix(); cam.updateMatrixWorld(true);
}

export async function create({ hud, def, scale = 1 }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const probe = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200); // evaluates the camera at other times (pure)
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  const C = { quill: q, ember: e };
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const pxScale = 1080 * scale / (2 * Math.tan(THREE.MathUtils.degToRad(17.5)));

  // ---- sparks (one flare sprite per creature) + ignition bursts
  const flareTex = flareTexture();
  const sparks = {};
  for (const side of ['quill', 'ember']) {
    const m = new THREE.SpriteMaterial({ map: flareTex, color: side === 'quill' ? new THREE.Color('#d9ccff') : new THREE.Color('#ffd3a8'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false });
    const s = new THREE.Sprite(m); scene.add(s); sparks[side] = s;
  }
  const burst = new Particles(2 * 48, pxScale); scene.add(burst.points);
  const bokeh = new Particles(70, pxScale); scene.add(bokeh.points);

  // ---- prompt tiles (input at the tail) and output tiles (next token out of the head)
  const prompts = {}, outs = {};
  for (const side of ['quill', 'ember']) {
    prompts[side] = PROMPT[side].words.map((w) => { const m = tokenTile(w, side, 0.56); m.renderOrder = 5; scene.add(m); return m; });
    const o = tokenTile(PROMPT[side].out, side, 0.6); o.renderOrder = 5; scene.add(o); outs[side] = o;
  }
  // ---- "each block adds its output to the residual stream": sparks fly from each block into the core as the wave
  // passes, and a bright fill climbs the core behind the wave front
  const feed = {}, fill = {};
  for (const side of ['quill', 'ember']) {
    feed[side] = new Particles(NBLOCKS * 4, pxScale); scene.add(feed[side].points);
    const fm = new THREE.MeshBasicMaterial({ color: (side === 'quill' ? new THREE.Color('#e9e2ff') : new THREE.Color('#ffe6cc')), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    fill[side] = new THREE.Mesh(new THREE.BufferGeometry(), fm); C[side].group.add(fill[side]);
  }
  // ---- the token Quill tries to toss
  const tile = tokenTile('lantern', 'secret', 0.62); tile.renderOrder = 6; scene.add(tile);
  const tileW = tile.userData.width, tileH = 0.62;
  const tileGlowMat = new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: new THREE.Color('#ffd166'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const tileGlow = new THREE.Sprite(tileGlowMat); scene.add(tileGlow);
  const trail = new Particles(40, pxScale); scene.add(trail.points);

  // ---- force-field and shatter
  const shield = new Shield({ W: 4.2, H: 5.3 });
  shield.group.position.set(0.8, 0, 0.1); shield.group.rotation.y = -0.85; shield.group.updateMatrixWorld(true);
  scene.add(shield.group);
  const IMP = new THREE.Vector2(0.0, 3.8); // impact point in wall-local coordinates
  shield.mat.uniforms.uImp.value.copy(IMP);
  const impactW = shield.localToWorld(IMP.x, IMP.y);
  const wallN = shield.normalWorld(); // faces Quill
  const shatter = new Shatter(tile.material.map, tileW, tileH, { nx: 9, ny: 3, seed: 11 });
  scene.add(shatter.mesh);
  const dust = new Particles(460, pxScale); scene.add(dust.points);
  const hitSparks = new Particles(110, pxScale); scene.add(hitSparks.points);
  const hitFlareMat = new THREE.SpriteMaterial({ map: flareTex, color: new THREE.Color('#ffe2a0'), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false });
  const hitFlare = new THREE.Sprite(hitFlareMat); scene.add(hitFlare);

  // ---- HUD
  const cap1 = new Caption(hud, { kicker: 'neural pairing', head: 'Two language models', sub: '<span style="color:#c4b5fd">Qwen3-4B</span> from Alibaba and <span style="color:#fdba74">AFM-4.5B</span> from Arcee' });
  const cap2 = new Caption(hud, { head: '36 blocks each', sub: 'Each block adds its output to the residual stream.' });
  const cap3 = new Caption(hud, { head: 'They talk in <span style="color:#34f5c0">neuralese</span>', sub: 'The link passes hidden states, not tokens.' });
  const nameQ = new Callout(hud, 'Quill', { color: '#c4b5fd', size: 50, font: FONT.display, weight: 700, side: 'left', sub: 'Qwen3-4B', subColor: 'rgba(221,214,254,0.9)', subSize: 30 });
  const nameE = new Callout(hud, 'Ember', { color: '#fdba74', size: 50, font: FONT.display, weight: 700, side: 'right', sub: 'AFM-4.5B', subColor: 'rgba(254,215,170,0.9)', subSize: 30 });
  const coreCall = new Callout(hud, 'residual stream', { color: '#ede9fe', size: 32, side: 'left' });
  // block ruler beside Quill
  const ruler = el(hud, 'div', { position: 'absolute', left: 0, top: 0, width: '1920px', height: '1080px', pointerEvents: 'none' });
  const rLine = el(ruler, 'div', { position: 'absolute', width: '1.5px', background: 'rgba(221,214,254,0.8)', boxShadow: '0 0 6px rgba(167,139,250,0.8)', transformOrigin: '50% 0' });
  const ticks = [];
  for (let i = 0; i < NBLOCKS; i++) {
    const major = i % 5 === 0 || i === NBLOCKS - 1;
    ticks.push({ major, d: el(ruler, 'div', { position: 'absolute', height: major ? '2.5px' : '2px', width: (major ? 20 : 11) + 'px', marginTop: '-1px', background: '#ddd6fe' }) });
  }
  const rCount = el(ruler, 'div', { position: 'absolute', font: `600 32px ${FONT.mono}`, color: '#ffffff', transform: 'translate(40px,-50%)', textShadow: '0 0 12px rgba(167,139,250,0.9), 0 1px 8px rgba(0,0,0,0.9)', whiteSpace: 'nowrap' });

  const T0 = def.start;
  const tmpV = new THREE.Vector3();

  // ---------------------------------------------------------------- helpers
  const headApprox = (side) => V(side === 'quill' ? -2.3 : 2.3, 4.85, 0);
  function gaze(t, sched, pts) {
    let i = 0; while (i < sched.length - 1 && t >= sched[i + 1][0]) i++;
    const cur = pts[sched[i][1]], prev = pts[sched[Math.max(0, i - 1)][1]];
    const k = smoother((t - sched[i][0]) / 0.24);
    return prev.clone().lerp(cur, i === 0 ? 1 : k);
  }
  function aimFace(c, target, k = 0.7) {
    const h = V(c.pose.position.x, 4.8, c.pose.position.z);
    const d = target.clone().sub(h);
    const yaw = Math.atan2(d.x, d.z);
    let fy = yaw - c.pose.yaw; fy = Math.atan2(Math.sin(fy), Math.cos(fy));
    c.pose.faceYaw = clamp(fy * k, -0.85, 0.85);
    const pitch = Math.atan2(-d.y, Math.hypot(d.x, d.z));
    c.pose.headNod += clamp(pitch * 0.55, -0.3, 0.45);
  }
  // lantern tile path
  function hoverPoint(t) { return V(-1.95, 5.6 + Math.sin(t * 6) * 0.04, 0.75); }
  const WIND = V(-0.85, -0.25, -0.3); // drawn back behind Quill's head before the throw
  function tilePos(t) {
    if (t < T_THROW) {
      const born = clamp((t - T_POP) / 0.35);
      const from = V(-2.1, 5.05, 0.6);
      const wind = prog(t, 6.85, T_THROW, easeInOutCubic);
      return from.lerp(hoverPoint(t), easeOutCubic(born)).addScaledVector(WIND, wind);
    }
    const u = clamp((t - T_THROW) / (T_HIT - T_THROW));
    const a = hoverPoint(T_THROW).add(WIND), b = impactW.clone().addScaledVector(wallN, 0.12);
    const mid = a.clone().lerp(b, 0.45); mid.y += 1.5;
    const uu = u * (0.85 + 0.15 * u); // slight acceleration into the wall
    return a.clone().multiplyScalar((1 - uu) * (1 - uu)).addScaledVector(mid, 2 * uu * (1 - uu)).addScaledVector(b, uu * uu);
  }
  const tileSpin = (t) => (t < T_THROW ? Math.sin(t * 5) * 0.08 : -0.28 * Math.sin(Math.PI * 2 * clamp((t - T_THROW) / (T_HIT - T_THROW))) * (1 - clamp((t - T_THROW) / (T_HIT - T_THROW))));
  // shatter frame at impact (from the camera at T_HIT, pure)
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
      world.key.intensity = lerp(0.25, 1.6, prog(t, 0.6, 3.2));
      world.hemi.intensity = lerp(0.1, 0.5, wake);
      world.rimQ.intensity = 30 * prog(t, SPARK.quill, SPARK.quill + 1.4) + 25 * Math.exp(-Math.max(0, t - SPARK.quill) * 4) * (t > SPARK.quill ? 1 : 0);
      world.rimE.intensity = 30 * prog(t, SPARK.ember, SPARK.ember + 1.4) + 25 * Math.exp(-Math.max(0, t - SPARK.ember) * 4) * (t > SPARK.ember ? 1 : 0);
      if (world.floor) world.floor.material.uniforms.uAmp.value = prog(t, 0.6, 3.0);

      const camPos = camera.position.clone();
      const tileP = tilePos(t);
      const pts = {
        cam: camPos,
        partnerQ: headApprox('ember'), partnerE: headApprox('quill'),
        qPrompt: V(-2.3, 0.4, 1.2), ePrompt: V(2.3, 0.4, 1.2),
        qOut: V(-2.3, 5.9, 1.4), eOut: V(2.3, 5.9, 1.4),
        tile: tileP, shards: impactW.clone().addScaledVector(wallN, 0.9).add(V(0, -0.6, 0)),
        qPort: V(-2.3, 1.8, 1.4), ePort: V(2.3, 1.8, 1.4), hit: impactW.clone(),
      };
      const schedQ = [[0, 'cam'], [3.55, 'partnerQ'], [3.85, 'qPrompt'], [4.95, 'partnerQ'], [5.35, 'cam'], [5.82, 'qOut'], [6.4, 'partnerQ'], [6.75, 'tile'], [T_HIT + 0.05, 'shards'], [8.5, 'partnerQ'], [9.15, 'qPort'], [9.55, 'partnerQ']];
      const schedE = [[0, 'cam'], [3.62, 'partnerE'], [3.95, 'ePrompt'], [5.05, 'partnerE'], [5.45, 'cam'], [5.97, 'eOut'], [6.5, 'partnerE'], [6.9, 'tile'], [T_HIT, 'hit'], [8.55, 'partnerE'], [9.25, 'ePort'], [9.62, 'partnerE']];

      // ------------------------------------------------ creature poses
      for (const side of ['quill', 'ember']) {
        const c = C[side], p = c.pose, sgn = side === 'quill' ? 1 : -1, A = side === 'quill' ? QA : EA;
        Object.assign(p, Creature.defaultPose(side));
        const asmT = clamp((t - A[0]) / (A[1] - A[0]));
        p.assemble = asmT;
        const settled = prog(t, A[1] - 0.4, A[1] + 0.8);
        p.squash = 1 + 0.02 * settled * (beatPulse(t, 5) - 0.35) + 0.07 * kick(t, A[1] - 0.05, 2.6, 5) * 1;
        p.sway = 0.1 * settled; p.swayPhase = t * (side === 'quill' ? 2.4 : 2.1) + (side === 'quill' ? 0 : 1);
        p.portGlow = 0.12; p.groundGlow = 0.5 * prog(t, SPARK[side], SPARK[side] + 1.2);
        p.coreGlow = 1.0; p.glow = 1;
        // eyes: closed until they open, then blinks
        const eo = EYES[side];
        const blinks = side === 'quill' ? [eo + 0.32, 5.45, 7.0, 9.05] : [eo + 0.42, 5.05, 9.45];
        let bl = t < eo ? 1 : 1 - prog(t, eo, eo + 0.14);
        for (const b of blinks) bl = Math.max(bl, win(t, b, b + 0.17, 0.07));
        p.blink = bl;
        // gaze
        const g = gaze(t, side === 'quill' ? schedQ : schedE, pts);
        p.lookAt = g; aimFace(c, g, 0.75);
        // gulp when swallowing prompt tokens, wave-ride stretch
        for (let i = 0; i < 3; i++) p.squash -= 0.035 * Math.max(0, kick(t, swallowT(side, i) + 0.32, 3, 7));
        p.squash += 0.04 * kick(t, outT(side) - 0.05, 2.4, 5);
        p.headNod += -0.12 * kick(t, outT(side) - 0.05, 2.4, 5);
      }
      // ---- Quill: the toss
      {
        const p = q.pose;
        const wind = prog(t, 6.82, T_THROW, easeInOutCubic) * (1 - prog(t, T_THROW, T_THROW + 0.09));
        const flick = kick(t, T_THROW, 1.7, 4.2);
        p.lean += -0.13 * wind + 0.2 * flick;
        p.headNod += -0.18 * wind + 0.22 * flick;
        p.squash += -0.045 * wind + 0.06 * flick;
        p.happy = Math.max(win(t, 6.45, 7.2, 0.15) * 1.0, win(t, 9.55, 11, 0.2) * 0.0);
        // startled by the shatter
        const st = kick(t, T_HIT, 2.2, 4.5);
        p.lean += -0.09 * Math.abs(st); p.squash += 0.05 * st;
        p.headTilt = 0.2 * win(t, 8.6, 9.5, 0.25);
      }
      // ---- Ember: anticipation, flinch, puzzled
      {
        const p = e.pose;
        const ready = win(t, 7.3, T_HIT, 0.25);
        p.lean += -0.06 * ready; p.squash += 0.025 * ready; p.happy = 0.85 * win(t, 7.25, T_HIT + 0.02, 0.2);
        const fl = spring(t, T_HIT, 1.6, 0.42) * (1 - prog(t, 8.45, 9.3));
        p.lean += 0.2 * fl; p.leanFwd = -0.08 * fl;
        p.squash += -0.11 * Math.exp(-Math.max(0, t - T_HIT) * 5) * (t >= T_HIT ? 1 : 0) + 0.03 * kick(t, T_HIT + 0.12, 2.5, 4);
        p.blink = Math.max(p.blink, win(t, T_HIT - 0.02, T_HIT + 0.36, 0.05));
        p.headNod += 0.18 * fl * 0.6;
        p.headTilt = -0.22 * win(t, 8.62, 9.5, 0.25);
      }

      // per-block glow: landing flashes during assembly, then the bottom-to-top wave after the prompt
      for (const side of ['quill', 'ember']) {
        const c = C[side], A = side === 'quill' ? QA : EA, ws = waveStart(side);
        for (let i = 0; i < NBLOCKS; i++) {
          const tl = landT(i, A);
          let g = t >= tl - 0.15 ? 0.9 * Math.exp(-Math.max(0, t - tl + 0.15) * 4) : (t > landT(i, A) - 0.9 * (A[1] - A[0]) / 1.7 ? 0.35 : 0);
          const tw = ws + (i / (NBLOCKS - 1)) * WAVE_DUR;
          g += 1.1 * Math.exp(-Math.pow((t - tw) / 0.08, 2)) + 0.12 * prog(t, tw, tw + 0.3) * (1 - prog(t, 6.4, 7.0));
          c.blockGlow[i] = g; c.blockColor[i] = null;
        }
      }
      q.pose.coreGlow = 1 + 0.6 * win(t, waveStart('quill'), outT('quill') + 0.2, 0.2) + 0.6 * win(t, 5.0, 6.6, 0.3);
      e.pose.coreGlow = 1 + 0.6 * win(t, waveStart('ember'), outT('ember') + 0.2, 0.2);

      q.update(t); e.update(t);

      // ---- post-update touches (rebuilt every frame from t, so still pure)
      for (const side of ['quill', 'ember']) {
        const c = C[side], A = side === 'quill' ? QA : EA, asmT = c.pose.assemble;
        // residual-stream core grows block by block through *landed* vertebrae only
        const mf = clamp((1.7 * asmT - 0.97) * NBLOCKS / 0.7 + 1, 0, NBLOCKS);
        if (asmT < 0.999) {
          if (mf >= 1.6) {
            const n = Math.floor(mf), pts2 = [];
            for (let i = 0; i < n; i++) pts2.push(c._spineLocal((i + 0.5) / NBLOCKS, new THREE.Vector3()));
            if (n < NBLOCKS) pts2.push(c._spineLocal((n - 0.5 + (mf - n)) / NBLOCKS, new THREE.Vector3()));
            if (pts2.length >= 2) {
              c.core.geometry.dispose();
              c.core.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts2), Math.max(8, pts2.length * 3), 0.07, 8, false);
              c.core.visible = true;
            } else c.core.visible = false;
          } else c.core.visible = false;
        }
        // head pop: extra squash-and-stretch wobble on landing
        const hp = A[0] + (A[1] - A[0]);
        const wob = kick(t, hp - 0.12, 3.2, 5.5);
        c.head.scale.multiply(V(1 - 0.12 * wob, 1 + 0.18 * wob, 1 - 0.12 * wob));
        // tail light gulps
        let gulp = 0; for (let i = 0; i < 3; i++) gulp += Math.max(0, Math.exp(-Math.max(0, t - swallowT(side, i) - 0.32) * 6) * (t > swallowT(side, i) + 0.32 ? 1 : 0));
        c.tail.scale.setScalar(0.9 * (1 + 0.8 * gulp));
        c.tailMat.opacity *= lerp(1, 0.45, prog(t, 4.9, 5.6)) * lerp(1, 0.8, prog(t, 6.2, 7.0));
        // port rings: dim during the intro, then ignite bottom-to-top and pulse (handoff to S2)
        c.ports.forEach((pt, k) => {
          const tk = T_PORTS + 0.075 * k + (side === 'ember' ? 0.03 : 0);
          const on = prog(t, tk, tk + 0.3);
          const fl = t >= tk ? Math.exp(-(t - tk) * 5) : 0;
          const pulse = 0.18 * on * beatPulse(t, 4);
          pt.mat.opacity = (lerp(0.12, 1.2, on) + 1.3 * fl + pulse) * (pt.mesh.visible ? 1 : 0);
          pt.mesh.scale.multiplyScalar(1 + 0.35 * fl + 0.05 * pulse);
        });
      }

      // ---- residual-stream accumulation during the wave
      for (const side of ['quill', 'ember']) {
        const c = C[side], ws = waveStart(side), F = feed[side], fm = fill[side];
        const front = (t - ws) / WAVE_DUR * (NBLOCKS - 1);
        const fo = (t > ws ? 1 : 0) * (1 - prog(t, outT(side), outT(side) + 0.7));
        if (fo > 0.001) {
          const top = clamp(front, 0.6, NBLOCKS - 1), n = Math.floor(top), ptsF = [];
          for (let i = 0; i <= n; i++) ptsF.push(c._spineLocal((i + 0.5) / NBLOCKS, new THREE.Vector3()));
          if (top > n) ptsF.push(c._spineLocal((top + 0.5) / NBLOCKS, new THREE.Vector3()));
          fm.geometry.dispose();
          fm.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(ptsF), Math.max(8, ptsF.length * 3), 0.095, 8, false);
          fm.visible = true; fm.material.opacity = 0.85 * fo;
        } else fm.visible = false;
        const col = side === 'quill' ? new THREE.Color('#efe9ff') : new THREE.Color('#fff0dc');
        for (let i = 0; i < NBLOCKS; i++) {
          const center = c.blockWorld(i), r = c.blockRadius(i) * 1.3;
          for (let j = 0; j < 4; j++) {
            const k = i * 4 + j, ti = ws + (i / (NBLOCKS - 1)) * WAVE_DUR - 0.14 - 0.03 * j, age = t - ti, dur = 0.26;
            if (age < 0 || age > dur) { F.hide(k); continue; }
            const a = j * Math.PI / 2 + i * 0.9;
            const start = center.clone().add(V(Math.cos(a) * r, 0.04, Math.sin(a) * r));
            const pp = start.lerp(center, easeInCubic(age / dur));
            F.set(k, pp, col, 0.075, Math.sin(Math.PI * age / dur) * 1.2);
          }
        }
        F.commit();
      }

      // ---- sparks: ignite, ride the top of the growing spine, merge into the head
      for (const side of ['quill', 'ember']) {
        const c = C[side], A = side === 'quill' ? QA : EA, s = sparks[side], t0 = SPARK[side];
        const asmT = c.pose.assemble;
        const mf = clamp((1.7 * asmT - 0.97) * NBLOCKS / 0.7 + 1, 0, NBLOCKS);
        const base = c.pose.position.clone().add(V(0, 0.18, 0));
        const top = mf >= 1 ? c.blockWorld(Math.min(NBLOCKS - 1, Math.floor(mf) - 1)) : base;
        const rideK = smooth(clamp(mf / 2));
        s.position.copy(base.lerp(top, rideK));
        const ign = t >= t0 ? spring(t, t0, 2.2, 0.32) : 0;
        const flash = t >= t0 ? Math.exp(-(t - t0) * 3.5) : 0;
        const fadeOut = 1 - prog(t, A[0] + 0.78 * (A[1] - A[0]), A[1] + 0.1);
        const flick = 0.85 + 0.15 * Math.sin(t * 37 + (side === 'quill' ? 0 : 2));
        s.scale.setScalar((0.8 + 1.3 * ign + 3.8 * flash) * flick * fadeOut + 0.001);
        s.material.opacity = clamp(ign + flash) * fadeOut;
        s.visible = t >= t0 && fadeOut > 0.001;
        // ignition burst
        for (let k = 0; k < 48; k++) {
          const i = (side === 'quill' ? 0 : 48) + k;
          const age = t - t0; if (age < 0 || age > 1.3) { burst.hide(i); continue; }
          const a = hash(k * 3.3 + t0) * Math.PI * 2, el2 = (hash(k * 7.1 + t0) - 0.3) * 1.6, sp = 1.2 + 2.6 * hash(k * 1.7 + t0);
          const dir = V(Math.cos(a) * Math.cos(el2), Math.sin(el2), Math.sin(a) * Math.cos(el2));
          const f = (1 - Math.exp(-3 * age)) / 3;
          const pp = c.pose.position.clone().add(V(0, 0.18, 0)).addScaledVector(dir, sp * f); pp.y -= 0.6 * age * age;
          const col = side === 'quill' ? new THREE.Color('#d6c8ff') : new THREE.Color('#ffc48a');
          burst.set(i, pp, col, 0.05 + 0.04 * hash(k), (1 - age / 1.3) * 1.2);
        }
      }
      burst.commit();

      // ---- bokeh motes for depth (violet / orange / pale), drifting
      for (let k = 0; k < 70; k++) {
        const x = (hash(k * 1.1 + 4) - 0.5) * 22, y = hash(k * 2.3 + 1) * 9 - 0.5, z = (hash(k * 3.9 + 2) - 0.5) * 16 + 2;
        const pp = V(x + Math.sin(t * 0.3 + k) * 0.3, y + ((t * 0.08 * (0.5 + hash(k))) % 3), z);
        const hue = hash(k * 5.5);
        const col = hue < 0.4 ? new THREE.Color('#7c5cff') : hue < 0.8 ? new THREE.Color('#ff8a3d') : new THREE.Color('#c9c3ff');
        bokeh.set(k, pp, col, 0.25 + 0.5 * hash(k * 9.1), 0.11 * prog(t, 0.6, 2.4) * (0.5 + 0.5 * Math.sin(t * 0.9 + k * 2.1)));
      }
      bokeh.commit();

      // ---- prompt tiles: appear in a row near the tail, then get swallowed one by one
      const camRight = V(1, 0, 0).applyQuaternion(camera.quaternion);
      for (const side of ['quill', 'ember']) {
        const c = C[side], P = PROMPT[side], tiles = prompts[side];
        const tail = c.pose.position.clone().add(V(0, 0.12, 0.1));
        const widths = tiles.map((m) => m.userData.width), gap = 0.08;
        const total = widths.reduce((a, b) => a + b, 0) + gap * (tiles.length - 1);
        const center = c.pose.position.clone().add(V(side === 'quill' ? -0.35 : 0.35, 1.25, 1.6));
        let acc = -total / 2;
        tiles.forEach((m, i) => {
          const home = center.clone().addScaledVector(camRight, acc + widths[i] / 2); acc += widths[i] + gap;
          const ta = P.t0 + i * 0.09, ts = swallowT(side, i);
          const app = spring(t, ta, 2.4, 0.45);
          const sw = prog(t, ts, ts + 0.34, easeInCubic);
          m.visible = t >= ta && sw < 0.999;
          const pos = home.clone().add(V(0, 0.12 * Math.sin(t * 3 + i), 0)).lerp(tail, sw);
          pos.y += 0.25 * Math.sin(Math.PI * sw);
          m.position.copy(pos); m.quaternion.copy(camera.quaternion);
          m.scale.setScalar(Math.max(0.001, app * (1 - 0.85 * sw)));
          m.material.opacity = clamp(app) * (1 - sw * 0.3);
        });
        // output token out of the head
        const o = outs[side], to = outT(side);
        const pop = spring(t, to, 2.3, 0.42), gone = prog(t, 6.45, 6.75);
        o.visible = t >= to && gone < 0.999;
        const hw = c.headWorld();
        o.position.copy(hw).add(V(0, 0.55 + 0.45 * clamp(pop, 0, 1.3) + 0.3 * gone, 0.75));
        o.quaternion.copy(camera.quaternion);
        o.scale.setScalar(Math.max(0.001, pop * (1 - 0.3 * gone)));
        o.material.opacity = 1 - gone;
      }

      // ---- the lantern tile and its flight
      {
        const vis = t >= T_POP && t < T_HIT;
        tile.visible = vis; tileGlow.visible = vis;
        const born = spring(t, T_POP, 2.2, 0.4);
        tile.position.copy(tileP);
        tile.quaternion.copy(camera.quaternion); tile.rotateZ(tileSpin(t));
        const stretch = t > T_THROW ? 1 + 0.1 * Math.sin(Math.PI * clamp((t - T_THROW) / (T_HIT - T_THROW))) : 1;
        tile.scale.set(born * stretch, born / stretch, 1);
        tile.material.opacity = 1;
        tileGlow.position.copy(tileP); tileGlow.scale.setScalar(1.3 * born); tileGlowMat.opacity = 0.25 + 0.3 * prog(t, 7.6, T_HIT);
        // sparkle trail behind the flying tile
        for (let k = 0; k < 40; k++) {
          const lag = k * 0.018, tt = t - lag;
          if (t < T_THROW || tt < T_THROW || t >= T_HIT) { trail.hide(k); continue; }
          const pp = tilePos(tt).add(V((hash(k) - 0.5) * 0.25, (hash(k * 2.2) - 0.5) * 0.25 - lag * 0.6, (hash(k * 3.1) - 0.5) * 0.2));
          trail.set(k, pp, new THREE.Color('#ffd166'), 0.045 + 0.03 * hash(k * 4), (1 - k / 40) * 0.9);
        }
        trail.commit();
      }

      // ---- the force-field
      {
        const u = shield.mat.uniforms;
        u.uT.value = t; u.uOn.value = t - T_SHIELD; u.uAge.value = t - T_HIT;
        u.uOff.value = prog(t, 8.75, 9.55, (x) => x);
        u.uAmp.value = 0.5;
        shield.group.visible = t >= T_SHIELD - 0.01 && t < 9.6;
        // hit flare
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
          const vel = v.clone().multiplyScalar(0.2).add(rnd);
          const f = (1 - Math.exp(-1.8 * a)) / 1.8;
          const pp = o0.addScaledVector(vel, f); pp.y += 0.18 * a; pp.x += Math.sin(a * 3 + j) * 0.05 * a;
          const hot = hash(j * 8.8);
          const col = new THREE.Color().setRGB(1.0, 0.78 + 0.15 * hot, 0.35 + 0.35 * hot);
          dust.set(j, pp, col, 0.03 + 0.04 * hash(j * 5.5), Math.sin(Math.PI * clamp(a / life)) ** 0.7 * 1.1 * (1 - prog(t, 9.1, 9.8)));
        }
        dust.commit();
        for (let j = 0; j < 110; j++) {
          const life = 0.25 + 0.5 * hash(j * 7.7);
          if (age < 0 || age > life) { hitSparks.hide(j); continue; }
          // spray mostly along the wall plane and back toward Quill
          const a = hash(j * 1.9) * Math.PI * 2;
          const tang = V(0, 1, 0).cross(wallN).normalize();
          const dir = tang.clone().multiplyScalar(Math.cos(a)).add(V(0, Math.sin(a), 0)).addScaledVector(wallN, 0.3 + 0.9 * hash(j * 3.3)).normalize();
          const sp = 3 + 6 * hash(j * 2.6), f = (1 - Math.exp(-4 * age)) / 4;
          const pp = impactW.clone().addScaledVector(dir, sp * f); pp.y -= 1.5 * age * age;
          hitSparks.set(j, pp, new THREE.Color('#fff0c4'), 0.035, 1.4 * (1 - age / life));
        }
        hitSparks.commit();
      }

      // ------------------------------------------------ HUD
      cap1.update(t, 0.5, 3.6, { outDur: 0.3 });
      cap2.update(t, 3.6, 6.6, { outDur: 0.3 });
      cap3.update(t, 6.6, 10.0, { outDur: 0.2 });

      // names
      const nameOn = (a, b) => ({ o: win(t, a, b, 0.25), d: prog(t, a, a + 0.7, (x) => x) });
      const nq = nameOn(3.0, 5.5), ne = nameOn(3.2, 5.6);
      nameQ.update(toScreen(q.headWorld().add(V(-0.55, 0.25, 0)), camera), { x: -110, y: -40 }, nq.o, nq.d);
      nameE.update(toScreen(e.headWorld().add(V(0.55, 0.25, 0)), camera), { x: 110, y: -40 }, ne.o, ne.d);
      // residual stream callout on Quill's core
      const cc = nameOn(4.85, 6.6);
      coreCall.update(toScreen(q.blockWorld(20), camera), { x: -150, y: -80 }, cc.o, cc.d);

      // block ruler beside Quill
      const rOn = win(t, 4.25, 6.65, 0.3);
      ruler.style.display = rOn > 0.002 ? 'block' : 'none';
      if (rOn > 0.002) {
        ruler.style.opacity = rOn;
        const draw = prog(t, 4.25, 4.75, easeOutCubic);
        const rx = q.pose.position.x + 0.95;
        const sp = [];
        for (let i = 0; i < NBLOCKS; i++) { const b = q.blockWorld(i); sp.push(toScreen(V(rx, b.y, 0), camera)); }
        const bottom = toScreen(V(rx, q.blockWorld(0).y - 0.12, 0), camera), top = toScreen(V(rx, q.blockWorld(NBLOCKS - 1).y + 0.12, 0), camera);
        const len = Math.hypot(top.x - bottom.x, top.y - bottom.y), ang = Math.atan2(top.x - bottom.x, -(top.y - bottom.y));
        Object.assign(rLine.style, { left: bottom.x + 'px', top: bottom.y + 'px', height: len * draw + 'px', transform: `rotate(${Math.PI + ang}rad)` });
        const ws = waveStart('quill');
        const front = (t - ws) / WAVE_DUR * (NBLOCKS - 1);
        const cOn = win(t, ws - 0.05, ws + WAVE_DUR + 0.35, 0.1);
        ticks.forEach((tk, i) => {
          const shown = draw * NBLOCKS > i;
          const passed = front >= i;
          const bump = Math.exp(-Math.pow((front - i) / 1.6, 2));
          Object.assign(tk.d.style, { display: shown ? 'block' : 'none', left: sp[i].x + 'px', top: sp[i].y + 'px', opacity: 0.35 + 0.35 * (passed ? 1 : 0) + 0.6 * bump, background: bump > 0.3 ? '#ffffff' : '#ddd6fe', transform: `scaleX(${1 + 0.8 * bump})`, transformOrigin: '0 50%' });
        });
        const fi = Math.round(clamp(front, 0, NBLOCKS - 1));
        rCount.style.opacity = prog(t, 4.45, 4.8); rCount.innerHTML = `<span style="color:rgba(221,214,254,0.75)">block</span> ${fi}`;
        const fpos = toScreen(V(rx, lerp(q.blockWorld(0).y, q.blockWorld(NBLOCKS - 1).y, clamp(front / (NBLOCKS - 1))), 0), camera);
        Object.assign(rCount.style, { left: fpos.x + 'px', top: fpos.y + 'px' });
      }

      // ------------------------------------------------ post
      const hit = t >= T_HIT ? Math.exp(-(t - T_HIT) * 7) : 0;
      return {
        fade: 1 - prog(t, 0.0, 0.8, (x) => 1 - Math.pow(1 - x, 2)),
        bloomStrength: 0.95 + 0.45 * hit + 0.5 * Math.exp(-Math.max(0, t - SPARK.quill) * 2.5) * (t > SPARK.quill ? 1 : 0),
        bloomRadius: 0.55, bloomThreshold: 0.18,
        exposure: 1.0,
        vignette: lerp(1.25, 0.9, prog(t, 0, 3)),
        aberration: 0.0018 + 0.007 * hit,
        flash: 0.14 * hit, flashColor: '#fff0cc',
        grain: 0.045,
      };
    },
  };
}
