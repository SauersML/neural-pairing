// ab_plate (40–48 s, authored in global time): the vibe-break. The film turns into a hand-coloured anatomical atlas
// plate (19th-century copperplate engraving) and back, by morphing the same 3D scene, never by cutting.
//  40.00        a2's last frame (the two linked creatures, a3's home framing); the link brightens
//  40.25–41.00  ink blooms out of the link, runs along the strands and the bodies, the glow drains into wet ink,
//               the sky dries into paper and the floor grid into ruled construction lines; perspective flattens
//  41.00        the plate is complete (the plate mark presses into the paper)
//  41.0–43.0    view 1, a turntable: the two bodies as engraved specimens. The spine is 36 blocks (a ruler of 36
//               ticks), the red cord through it the residual stream (token in at the tail, out at the head), and the
//               link: at each of the 8 exchange depths a nerve of two fibres, one each way
//  43.0–46.0    view 2: Quill's vertebra 15 lifts out, enlarged, and parts in two (self-attention, then feed-forward,
//               each adding into the stream); a roundel magnifies the link nerve at that depth: one read
//               (the reader's query, the partner's keys and values, a gate on each value, weighted sum, write size
//               10 %, added to the reader's stream after the block)
//  46.0–46.5    the plate folds its diagrams away; the camera pulls back home
//  46.5–47.8    the link re-ignites and the paper soaks back into light, out from the link to the edges
//  47.8–48.0    the film exactly as a3 opens (its local 32)
// Pure in t.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { toScreen } from '../lib/hud.js';
import { clamp, lerp, prog, win, beatPulse, smoother, smooth, easeInOutCubic, easeOutCubic, easeInCubic, kick } from '../lib/util.js';
import { makeSpline } from './s1_meet_fx.js';
import { blink, yawTo, lerpAng } from './s3_play_kit.js';
import { PlateRenderer, PU, boneMat, ndMat, flatMat, featherInk, featherND, flameInk, flameND, inkTube, inkFlat, floorInk } from '../lib/plate_ink.js';
import { PlateHud, loadPlateFonts } from './ab_plate_hud.js';
import { PlateArt } from './ab_plate_art.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);

// ---------------------------------------------------------------- timeline (global seconds, 16th grid)
export const TL = {
  inkA: 40.25, inkB: 41.0,          // the ink front
  press: 41.0,                      // plate mark
  v1: 42.0,                         // view 1 settled
  v2a: 43.0, v2: 44.0,              // to view 2
  back: 46.0,                       // diagrams fold, camera returns
  outA: 46.5, outB: 47.75,          // the soak back into light
  end: 48.0,
};
const LIFE_SWITCH = 45.0;           // a2's clock (t - 8) -> a3's clock (t - 16), hidden inside the plate

// ---------------------------------------------------------------- cameras
// a2's camera keys around 40 (its spline, the segment 40..41 depends on 39.3, 40, 41 only)
const K = (t, g, az, el, d, fov = 35) => ({ t, g, a: V(az, el, d), f: V(fov, 0, 0) });
const A2K = [K(38.2, V(0, 2.3, -1.0), -1.1, 0.55, 13.6), K(39.3, V(0.1, 3.1, 0), -0.06, 0.02, 14.6), K(40.0, V(0.1, 3.2, 0), -0.0203, 0.00676, 14.80), K(41.0, V(0.1, 3.2, 0), -0.0203, 0.00676, 14.62)];
const a2Spline = makeSpline(A2K);
const sph = (g, a) => g.clone().add(V(Math.cos(a.y) * Math.sin(a.x), Math.sin(a.y), Math.cos(a.y) * Math.cos(a.x)).multiplyScalar(a.z));
function a2Cam(t) { const g = a2Spline(t, (k) => k.g), a = a2Spline(t, (k) => k.a); return { pos: sph(g, a), look: g, fov: 35 }; }
function a3Cam(t) { const u = t - 16 - 32; return { pos: V(-0.2 + 0.05 * u, 3.3, 14.8 - 0.18 * u), look: V(0.1, 3.2, 0), fov: 35 }; }
// the plate's own path: [t, target, azimuth, elevation, distance (at fov 35)]; eased between keys, with a drift
const HOME = { g: V(0.1, 3.2, 0), az: -0.0203, el: 0.00676, d: 14.80 };
const CK = [
  [40.0, HOME.g, HOME.az, HOME.el, HOME.d],
  [42.0, V(0.0, 3.0, 0), -0.62, 0.14, 14.0],
  [42.75, V(0.0, 3.0, 0), -0.69, 0.15, 13.8],
  [43.375, V(-1.45, 2.1, 0.2), -0.24, 0.2, 7.3],
  [46.0, V(-1.45, 2.1, 0.2), -0.20, 0.19, 7.05],
  [47.5, a3Cam(47.5).look, HOME.az, HOME.el, 14.8 + 0.18 * 0.5],
];
function plateCam(t) {
  let i = 0; while (i < CK.length - 2 && t > CK[i + 1][0]) i++;
  const A = CK[i], B = CK[i + 1];
  const u = clamp((t - A[0]) / (B[0] - A[0]));
  const s = i === CK.length - 2 ? 0.5 - 0.5 * Math.cos(Math.PI * u) : easeInOutCubic(u);   // the pull-back leaves at once
  // into view 2: tilt down first (the heads leave the top of the frame early and cleanly), then move in
  const sg = A[0] === 42.75 ? easeOutCubic(u) : s;
  const g = A[1].clone().lerp(B[1], sg);
  const a = V(lerp(A[2], B[2], s), lerp(A[3], B[3], s), lerp(A[4], B[4], s));
  return { g, a };
}
// flatten the perspective while it is a plate (dolly-zoom, the subject keeps its size)
const FOV_PLATE = 26;
function flatK(t) { return smoother(prog(t, 40.3, 41.4)) * (1 - smoother(prog(t, 46.2, 47.5))); }

export async function create({ hud, renderer, scale, W, H }) {
  await loadPlateFonts();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200); scene.add(camera);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group); q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);

  // ---------------- the engraving: plate materials for the same meshes
  const pr = new PlateRenderer(renderer, W, H, scale);
  const nd = ndMat(), ndDouble = ndMat({ side: THREE.DoubleSide });
  const inkBody = (c, wash) => {
    const bone = boneMat({ wash, s1: 'vUv.y', s2: 'vUv.x' });
    c.blocks.forEach((m) => pr.add(m, bone, nd));
    pr.add(c.core, inkTube('red', { base: 0.4, lineTone: 0.5, uEnd: 0.962 }), null);
    for (const pt of c.ports) pr.add(pt.mesh, inkTube('teal', { base: 0.55, lineTone: 0.4, depthTest: true }), null);
    pr.add(c.headMesh, boneMat({ wash, s1: 'vUv.y', s2: 'vUv.x', toneMul: 0.82, cross: 0.0 }), nd);
    const white = boneMat({ wash: 1, s1: 'vUv.y', s2: 'vUv.x', toneMul: 0.55, cross: 0 }), dark = flatMat(0.94), paper = flatMat(0.0);
    for (const ey of c.eyes) { pr.add(ey.white, white, nd); pr.add(ey.pupil, dark, nd); pr.add(ey.glint, paper, null); pr.add(ey.arc, dark, nd); }
    if (c.side === 'quill') { const fi = featherInk(wash), fn = featherND(); c.feathers.forEach((f) => pr.add(f, fi, fn)); }
    else { pr.add(c.flame, flameInk(c.crestMat.uniforms.uTime, wash), flameND(c.crestMat.uniforms.uTime));
      for (const f of c.flameLocks) pr.add(f, flameInk(c.crestMat.uniforms.uTime, wash), flameND(c.crestMat.uniforms.uTime)); }
  };
  inkBody(q, 0.25); inkBody(e, 0.5);
  pr.add(world.floor, floorInk([V(-2.3, 0, 0), V(2.3, 0, 0)]), null);
  world.floor.renderOrder = 0;
  // the link, engraved: one nerve of two fibres at each depth (thicker than the film's strands), and an arrowhead on
  // each fibre for its direction
  const nerveMat = inkTube('teal', { base: 0.5, lineTone: 0.6, tol: 0.7 });
  const nerves = link.strands.map(() => { const m = new THREE.Mesh(new THREE.BufferGeometry(), nerveMat); scene.add(m); pr.add(m, nerveMat, null, true); return m; });
  const arrowGeo = new THREE.ConeGeometry(0.075, 0.24, 14); const arrowMat = inkFlat('teal', 0.95);
  const arrows = link.strands.map(() => { const m = new THREE.Mesh(arrowGeo, arrowMat); scene.add(m); pr.add(m, arrowMat, null, true); return m; });

  // diagrams of the plate (the lifted vertebra, the roundel, labels)
  const ui = new PlateHud(hud);
  const art = new PlateArt({ scene, pr, q, e, link, ui, camera, camAt: (tt, cam) => setCam(tt, cam) });

  const reset = (c) => { Object.assign(c.pose, Creature.defaultPose(c.side)); c.blockGlow.fill(0); for (let i = 0; i < NBLOCKS; i++) c.blockColor[i] = null; c.inject = []; };
  const homeCam = sph(HOME.g, V(HOME.az, HOME.el, HOME.d));
  const linkMid = V(0, (15 + 0.5) / NBLOCKS * 4.2, 0);

  function setCam(t, cam) {
    const pc = plateCam(t);
    const fk = flatK(t), fov = lerp(35, FOV_PLATE, fk);
    const dComp = Math.tan(THREE.MathUtils.degToRad(35 / 2)) / Math.tan(THREE.MathUtils.degToRad(fov / 2));
    let pos = sph(pc.g, V(pc.a.x, pc.a.y, pc.a.z * dComp)), look = pc.g.clone();
    { const w = smoother(prog(t, 40.0, 40.7)); if (w < 1) { const c2 = a2Cam(t); const p2 = c2.look.clone().add(c2.pos.clone().sub(c2.look).multiplyScalar(dComp)); pos = p2.lerp(pos, w); look = c2.look.clone().lerp(look, w); } }
    { const w = smoother(prog(t, 47.3, 48.0)); if (w > 0) { const c3 = a3Cam(t); const p3 = c3.look.clone().add(c3.pos.clone().sub(c3.look).multiplyScalar(dComp)); pos.lerp(p3, w); look.lerp(c3.look, w); } }
    cam.position.copy(pos); cam.lookAt(look); cam.fov = fov; cam.aspect = 16 / 9; cam.updateProjectionMatrix(); cam.updateMatrixWorld(true);
  }
  function update(t) {
    setCam(t, camera);

    // ---------------- clocks: a2's before the switch, a3's after (the switch is hidden inside the plate)
    const late = t >= LIFE_SWITCH;
    const T = late ? t - 16 : t - 8;          // creature life
    const TW = late ? t - 16 : t;             // world, link, creature shaders (a2 passed global t, a3 its local t)
    world.update(TW);
    const still = smoother(prog(t, 40.4, 41.2)) * (1 - smoother(prog(t, 46.5, 47.6)));   // specimens hold still

    // ---------------- base life (a3's, which a2 hands off to)
    reset(q); reset(e);
    const qB = V(-2.3, 0, 0), eB = V(2.3, 0, 0);
    q.pose.position.copy(qB); e.pose.position.copy(eB);
    for (const [c, seed, B, P] of [[q, 0, qB, eB], [e, 1.7, eB, qB]]) {
      const p = c.pose, live = 1 - still;
      p.swayPhase = T * 2.3 + seed * 2; p.sway = 0.08 * live;
      p.squash = 1 - 0.022 * beatPulse(T, 6) * live;
      p.blink = blink(T, seed + 3); p.lean = 0.03 * Math.sin(T * 1.2 + seed) * live; p.headTilt = 0.06 * Math.sin(T * 0.9 + seed * 2.1) * live;
      p.portGlow = 0.28; p.groundGlow = 0.45; p.xray = 0.5;
      p.coreGlow = lerp(0.8, 1.0, smoother(prog(t, 42.0, 45.0)));
      // the turntable: the bodies keep the pose they had for the home camera while the camera orbits
      const yLive = lerpAng(yawTo(B, camera.position), yawTo(B, P), 0.35), yHome = lerpAng(yawTo(B, homeCam), yawTo(B, P), 0.35);
      p.yaw = lerpAng(yLive, yHome, still);
    }
    q.pose.lookAt = e.headWorld(); e.pose.lookAt = q.headWorld();
    // link
    const ls = link.state;
    ls.depth = 1; ls.grow = 1; ls.flow = 1; ls.packets = 3; ls.aToB = 1; ls.bToA = 1; ls.sag = -0.2; ls.twist = 0.07;
    ls.tintAB = null; ls.tintBA = null; ls.pulses = []; ls.packetSize = 0.16; ls.highlight = -1;
    const surge = win(t, 40.0, 40.75, 0.35) + win(t, 46.25, 48.0, 0.6);
    ls.intensity = 0.32 + 0.9 * surge;
    art.pre(t);
    q.update(TW); e.update(TW);
    link.update(TW);

    // ---------------- plate geometry (only drawn in the plate)
    const inPlate = t > TL.inkA - 0.01 && t < TL.outB + 0.06;
    if (inPlate) {
      link.strands.forEach((st, i) => {
        const m = nerves[i];
        m.geometry.dispose(); m.geometry = new THREE.TubeGeometry(st.curve, 40, 0.034, 8, false); m.userData.plateVisible = true;
        const a = arrows[i], u = 0.36, p = st.curve.getPointAt(u), tg = st.curve.getTangentAt(u);
        a.position.copy(p); a.quaternion.setFromUnitVectors(V(0, 1, 0), tg); a.userData.plateVisible = true;
      });
      // shadows on the floor fall away from the engraver's light (it stays with the viewer)
      const r = V(1, 0, 0).applyQuaternion(camera.quaternion), f = V(0, 0, -1).applyQuaternion(camera.quaternion);
      PU.uShadowDir.value.set(r.x * 0.75 + f.x * 0.55, r.z * 0.75 + f.z * 0.55).normalize();
    }
    const artPost = art.update(t, inPlate);

    // ---------------- the morph
    const post = { bloomStrength: 0.95 + 0.05 * beatPulse(t, 6) * (1 - prog(t, 40.25, 41)), bloomRadius: 0.5, bloomThreshold: 0.62, vignette: 0.95 };
    const ci = toScreen(linkMid.clone().applyMatrix4(new THREE.Matrix4()), camera);
    const center = [ci.x / 1920, 1 - ci.y / 1080];
    if (inPlate) {
      const tex = pr.render(scene, camera);
      const plate = { ink: tex.ink, nd: tex.nd, center, band: 0.13, lane: 0.72, contour: 1, wash: 0.3, ...artPost };
      plate.press = smoother(prog(t, TL.press - 0.125, TL.press + 0.06)) * (1 - prog(t, TL.outA, TL.outA + 0.5));
      if (t < TL.inkB) {
        const k = prog(t, TL.inkA, TL.inkB, (x) => x);
        plate.mode = 'in'; plate.radius = 1.42 * Math.pow(k, 1.55); plate.all = 0;
        post.bloomStrength *= 1 - 0.55 * k;
      } else if (t < TL.outA) { plate.mode = 'in'; plate.radius = 3; plate.all = 1; post.bloomStrength *= 0.45; }
      else {
        const k = prog(t, TL.outA, TL.outB, (x) => x);
        plate.mode = 'out'; plate.band = 0.32; plate.radius = 1.75 * Math.pow(prog(t, TL.outA, 47.2, (x) => x), 1.1);
        plate.dim = 0.84 * prog(t, 46.98, 47.72, (x) => x * (0.6 + 0.4 * x)); plate.all = 0;
        post.bloomStrength = lerp(0.95 * 0.45, 0.95, smoother(prog(t, TL.outA + 0.3, TL.outB))) + 0.22 * win(t, 46.75, 47.75, 0.4);
      }
      if (t >= TL.outB + 0.05) delete post.plate; else post.plate = plate;
    }
    return post;
  }
  return { scene, camera, update };
}
