// S2 "THE LINK" (10–22 s). Hero technical shot.
//  10.0–14.0  the 8 bridges grow one per beat (exchange k at 10.0 + 0.5k, blocks 6, 9, ..., 27), packets both ways
//  13.85–14.5 the pair unrolls into two model sheets (one residual stream per token position, lib/grid.js), the
//             cross-attention edges of the exchange after block 24 light up, and the camera dives into that row
//  14.5–21.2  one read, dissected (separate scene: shots/s2_link_schem.js)
//  21.2–22.0  snap back out to the full pair, link alive, flash at 22.0
import * as THREE from 'three';
import { Creature, XBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Chapter, Pin, el, FONT } from '../lib/hud.js';
import { C } from '../lib/palette.js';
import { clamp, lerp, prog, spring, kick, win, beatPulse, smooth, easeOutCubic, easeInCubic, easeOutBack, easeInOutCubic, noise1 } from '../lib/util.js';
import { buildSchematic, TS } from './s2_link_schem.js';
import { ModelSheet, LinkBeams, XROWS } from '../lib/grid.js';

const T_GROW = (k) => 10.0 + 0.5 * k;
const T_DIVE = 14.0, T_CUT1 = 14.5, T_CUT2 = 21.2, T_END = 22.0;
const DIVE_E = 6; // exchange index of block 24

export async function create({ hud, def }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);

  // port flashes: a burst sprite + an expanding ring at both ports of each exchange as its bridge grows
  const dot = Creature.dotTexture();
  const flashes = XBLOCKS.map(() => [0, 1].map(() => {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: dot, color: C.linkHi.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.02, 8, 48), new THREE.MeshBasicMaterial({ color: C.link.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    ring.rotation.x = Math.PI / 2;
    scene.add(sp, ring); return { sp, ring };
  }));

  // the unrolled view (same tokens as the dissection): one residual-stream column per token, rows = blocks
  const DY = q.H / 36, SY = 0.55, DX = 0.62;
  const qTok = ['…', 'lantern', '▸', 'Sky', 'looks', 'blue'], eTok = ['…', 'word?', '▸', 'The', 'word', 'is'];
  const QT = [-1, -1, 0, 1, 2, 3], ET = [-1, -1, 0, 1, 2, 3]; // ticks
  const qs = new ModelSheet('quill', { tokens: qTok, dx: DX, dy: DY, tileH: 0.34 }), es = new ModelSheet('ember', { tokens: eTok, dx: DX, dy: DY, tileH: 0.34 });
  qs.group.position.set(-2.3 - 2.5 * DX, SY, 0); es.group.position.set(2.3 - 2.5 * DX, SY, 0);
  scene.add(qs.group, es.group);
  for (const sh of [qs, es]) for (const tl of sh.inTiles) tl.renderOrder = 10; // draw the token tiles after the floor
  const beams = new LinkBeams(scene, { pool: 64 });
  const ROW24 = XROWS[DIVE_E]; // row after block 24
  const ATTR = [0.05, 0.36, 0.1, 0.07, 0.08, 0.2]; // Ember "is" reading Quill (same illustrative weights as the dissection)

  const schem = buildSchematic({ hud });

  // HUD (main scene)
  const mainRoot = el(hud, 'div', { position: 'absolute', inset: '0' });
  const chap = new Chapter(hud, '02', 'The link');
  const cap = new Caption(mainRoot, {
    kicker: 'the link',
    head: '<span style="color:#34f5c0">Cross-attention</span> at 8 depths',
    sub: 'After blocks 6, 9, …, 27, each model attends to the other.',
    pos: { left: 110, top: 64 },
  });
  const cap2 = new Caption(hud, {
    head: '<span style="color:#fb923c">Ember</span> reads <span style="color:#b9a2ff">Quill</span>',
    sub: '<span style="color:#b9a2ff">Quill</span> reads <span style="color:#fb923c">Ember</span> the same way, at the same time.',
  });
  const labels = XBLOCKS.map((b, k) => new Pin(mainRoot,
    k === 0 ? `<span style="color:rgba(245,243,255,0.7)">after block</span> <span style="color:#34f5c0;font-weight:700">${b}</span>` : `<span style="color:#34f5c0;font-weight:700">${b}</span>`,
    { size: 32, color: '#34f5c0', dot: false }));

  const inst = { scene, camera, update };
  const home = { q: new THREE.Vector3(-2.3, 0, 0), e: new THREE.Vector3(2.3, 0, 0) };
  const tmp = new THREE.Vector3(), far = new THREE.Vector3();

  const blockY = (i) => q.H * (i + 0.5) / 36; // analytic (pure): block height on the rest pose
  function blinkAt(t, times) { let b = 0; for (const tb of times) b = Math.max(b, win(t, tb, tb + 0.17, 0.07)); return b; }

  function poseCreatures(t, phase3) {
    const bp = beatPulse(t, 6);
    for (const [c, sgn] of [[q, -1], [e, 1]]) {
      const p = c.pose;
      p.position.copy(sgn < 0 ? home.q : home.e);
      p.yaw = sgn < 0 ? 0.5 : -0.5;
      // breathe on the beat, and a small squash kick as each bridge lands
      let sq = 1 - 0.02 * bp;
      for (let k = 0; k < 8; k++) sq += 0.035 * kick(t, T_GROW(k) + 0.05, 3.2, 6);
      p.sway = 0.1; p.swayPhase = t * (sgn < 0 ? 2.3 : 2.0) + (sgn < 0 ? 0 : 1.3);
      p.portGlow = 0.95 + 0.3 * bp;
      p.coreGlow = 1.1 + 0.2 * bp;
      p.happy = 0; p.headTilt = 0.08 * Math.sin(t * 1.3 + sgn); p.lean = 0; p.leanFwd = 0; p.crest = 1;
      p.blink = blinkAt(t, sgn < 0 ? [11.1, 13.2] : [11.6, 12.9]);
      p.headNod = 0.12; p.faceYaw = 0;
      if (!phase3) {
        // watch the bridges as they grow, bottom to top; then look at block 24 before the dive
        const kNow = clamp((t - 10.0) / 0.5, 0, 7);
        const y0 = blockY(XBLOCKS[Math.floor(kNow)]);
        const y1 = blockY(XBLOCKS[Math.min(7, Math.floor(kNow) + 1)]);
        let ly = lerp(y0, y1, smooth(kNow - Math.floor(kNow)));
        const yd = blockY(XBLOCKS[DIVE_E]);
        ly = lerp(ly, yd, prog(t, 13.7, 14.0));
        p.lookAt = new THREE.Vector3(0, ly, 0.4);
        p.headNod = lerp(0.32, 0.12, prog(t, 10, 14));
        p.faceYaw = sgn < 0 ? 0.25 : -0.25;
        p.leanFwd = 0.06 * prog(t, 13.4, 14.2);
        p.lean = -sgn * 0.05 * prog(t, 13.4, 14.2);
      } else {
        // delighted: happy eyes, a hop with squash and stretch, crest flare, lean toward each other
        const lt = t - T_CUT2;
        p.happy = prog(t, T_CUT2, T_CUT2 + 0.2);
        const hopT = (lt - 0.12 - (sgn < 0 ? 0 : 0.06)) / 0.42;
        const hop = hopT > 0 && hopT < 1 ? Math.sin(Math.PI * hopT) : 0;
        p.position.y += 0.32 * hop;
        sq += (hopT > 0 && hopT < 1 ? 0.08 * Math.sin(Math.PI * hopT) : 0) - 0.06 * win(lt, 0.0, 0.16, 0.06) - 0.07 * (hopT >= 1 ? Math.exp(-8 * (hopT - 1) * 0.42) : 0);
        p.crest = 1.35;
        p.lean = -sgn * 0.06; p.faceYaw = sgn < 0 ? 0.35 : -0.35;
        p.lookAt = new THREE.Vector3(-sgn * 2.3, 4.9, 0.3);
        p.headNod = 0.02;
      }
      p.squash = sq;
      p.opacity = phase3 ? 1 : 1 - prog(t, 13.78, 13.95);
      p.groundGlow = 0.5 * p.opacity;
    }
  }

  function bridgeMid(out) {
    const cv = link.curveFor(DIVE_E, 0); return out.copy(cv.getPoint(0.5));
  }

  // wide 3/4 sweep for part 1
  function sweepCam(t, out, tgt) {
    const lt = clamp(t - 10, 0, 4.2);
    const az = lerp(0.55, 0.0, easeInOutCubic(lt / 4.2)) + 0.03 * Math.sin(t * 0.9);
    const r = lerp(16.4, 15.0, smooth(lt / 4));
    const h = lerp(5.6, 4.3, smooth(lt / 4));
    tgt.set(lerp(-1.3, -1.0, smooth(lt / 4)), 4.15, 0);
    out.set(tgt.x + 1.0 + Math.sin(az) * r, h, Math.cos(az) * r);
  }

  function update(t) {
    const post = { bloomStrength: 0.75, bloomRadius: 0.5, bloomThreshold: 0.22 };
    chap.update(t, def.start, def.end);
    // ===================== Part 2: the schematic =====================
    if (t >= T_CUT1 && t < T_CUT2) {
      mainRoot.style.display = 'none';
      schem.update(t);
      inst.scene = schem.scene; inst.camera = schem.camera;
      cap2.update(t, 14.8, 21.0);
      post.bloomStrength = 0.55; post.bloomThreshold = 0.35; post.bloomRadius = 0.4; post.vignette = 0.7;
      // flash fading in from the dive, and a flash out on the "+" before the snap back
      post.flash = 0.85 * Math.pow(1 - prog(t, T_CUT1, T_CUT1 + 0.22, easeOutCubic), 2) + 0.55 * prog(t, T_CUT2 - 0.12, T_CUT2, easeInCubic);
      post.flashColor = t < 18 ? '#dffcf2' : '#ffe2c4';
      return post;
    }
    // ===================== Parts 1 and 3: the pair =====================
    schem.hide(); mainRoot.style.display = 'block';
    cap2.update(t, 14.8, 21.0);
    inst.scene = scene; inst.camera = camera;
    const phase3 = t >= T_CUT2;
    world.update(t);
    poseCreatures(t, phase3);
    q.update(t); e.update(t);
    q.group.visible = e.group.visible = q.pose.opacity > 0.06;

    // link state
    const ls = link.state;
    ls.grow = XBLOCKS.map((_, k) => phase3 ? 1 : easeOutCubic(clamp((t - T_GROW(k)) / 0.42)));
    ls.intensity = XBLOCKS.map((_, k) => {
      const tg = T_GROW(k); const f = t > tg ? Math.exp(-3.5 * (t - tg)) : 0;
      return (phase3 ? 1.15 + 0.25 * beatPulse(t) : 0.85 + 1.4 * f);
    });
    ls.packets = phase3 ? 5 : 3; ls.flow = phase3 ? 1.7 : 1.1; ls.packetSize = phase3 ? 0.2 : 0.17;
    ls.highlight = !phase3 && t > 13.7 ? DIVE_E : -1;
    ls.dim = 1 - 0.75 * prog(t, 13.7, 14.2);
    if (!phase3) { const fo = 1 - prog(t, 13.85, 14.05); ls.intensity = ls.intensity.map((v) => v * fo); ls.aToB = ls.bToA = fo; }
    else { ls.aToB = ls.bToA = 1; }
    // per-block glow on the exchange ports as they light
    for (const c of [q, e]) {
      c.blockGlow.fill(0);
      XBLOCKS.forEach((b, k) => { const tg = T_GROW(k); const f = t > tg ? Math.exp(-4 * (t - tg)) : 0; c.blockGlow[b] = 2.2 * f + (t > tg || phase3 ? 0.25 : 0); });
      if (phase3) XBLOCKS.forEach((b) => { c.blockGlow[b] += 0.3 * beatPulse(t); });
    }

    // camera
    const cp = new THREE.Vector3(), ct = new THREE.Vector3();
    sweepCam(Math.min(t, 13.8), cp, ct);
    let fov = 35;
    const M = bridgeMid(new THREE.Vector3());
    const MR = new THREE.Vector3(0.6, SY + ROW24 * DY, 0); // the exchange row after block 24, between the sheets
    if (!phase3 && t >= 13.8) {
      // move in to the unrolled sheets (13.8–14.1), then dive into the exchange row after block 24 (14.25–14.5)
      const a = easeInOutCubic(prog(t, 13.8, 14.12, (x) => x));
      cp.lerp(new THREE.Vector3(0.35 + 0.2 * prog(t, 14.0, 14.3), 4.7, 9.0), a); ct.lerp(new THREE.Vector3(0.05, 2.5, 0), a);
      const k = easeInCubic(prog(t, 14.22, T_CUT1, (x) => x));
      cp.lerp(MR.clone().add(new THREE.Vector3(0.05, 0.02, 0.3)), k); ct.lerp(MR, k);
      fov = lerp(35, 55, k);
    }
    if (phase3) {
      // snap back out from inside the bridge to a wide, slightly low 3/4 view
      const s = spring(t, T_CUT2, 2.1, 0.62);
      const lt = t - T_CUT2;
      const az = 0.12 + 0.05 * lt;
      const wideT = new THREE.Vector3(0, 2.55, 0);
      const wideP = new THREE.Vector3(Math.sin(az) * 12.2, 3.0, Math.cos(az) * 12.2);
      const near = M.clone().add(new THREE.Vector3(0.0, 0.1, 1.6));
      cp.copy(near).lerp(wideP, s); ct.copy(M).lerp(wideT, Math.min(1, s * 1.1));
      fov = lerp(50, 35, clamp(s));
    }
    camera.position.copy(cp); camera.lookAt(ct); camera.fov = fov;
    camera.aspect = 16 / 9; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
    // the flame billboards toward the camera: redo it now that this frame's camera is set (keeps update pure)
    { e.group.updateMatrixWorld(true); const wq = new THREE.Quaternion(); e.crest.getWorldQuaternion(wq); e.flame.quaternion.copy(wq.invert().multiply(camera.quaternion)); }

    link.update(t);

    // the unrolled sheets (13.85–14.5): same model, one residual stream per token position
    const un = phase3 ? 0 : prog(t, 13.8, 14.05);
    for (const [sh, cx] of [[qs, -2.3], [es, 2.3]]) {
      sh.group.visible = un > 0.002;
      sh.group.scale.set(lerp(0.12, 1, easeOutCubic(un)), 1, 1);
      sh.group.position.x = cx - 2.5 * DX * sh.group.scale.x;
      sh.state.opacity = un; sh.state.reveal = prog(t, 13.8, 14.05); sh.state.exchangeGlow = 1.0; sh.state.nodeGlow = 1.0; sh.state.streamGlow = 1.8;
      sh.begin();
    }
    beams.begin(); beams.opacity = un * prog(t, 13.95, 14.1);
    if (un > 0.002) {
      // Ember "is" (tick 3) reads Quill's positions at ticks <= 3, after block 24 (bright); every other position reads
      // the partner's positions of its tick or earlier, in both directions (faint)
      for (let s2 = 0; s2 < 6; s2++) beams.beam(qs, s2, es, 5, ROW24, s2 === 1 ? 1.0 : ATTR[s2] * 1.6, t, s2 === 1 ? C.quillHi : null, 0.05, 0.9 + 0.45 * (5 - s2));
      for (let r = 0; r < 5; r++) for (let s2 = 0; s2 < 6; s2++) if (QT[s2] <= ET[r]) beams.beam(qs, s2, es, r, ROW24, 0.05, t, null, 0, 0.5);
      for (let r = 0; r < 6; r++) for (let s2 = 0; s2 < 6; s2++) if (ET[s2] <= QT[r] && beams.n < 62) beams.beam(es, s2, qs, r, ROW24, 0.04, t, null, 0, -0.5);
      es.addInfo(5, ROW24 - 0.5, ROW24 + 2.5, C.link, 0.8 * prog(t, 14.0, 14.2));
      qs.addInfo(1, 0, ROW24, C.quillHi, 0.35 * prog(t, 13.95, 14.15)); // the "lantern" column, read most strongly
    }
    qs.end(t); es.end(t); beams.end();

    // port flashes
    XBLOCKS.forEach((b, k) => {
      const tg = T_GROW(k); const f = !phase3 && t > tg ? Math.exp(-5 * (t - tg)) : 0;
      [q, e].forEach((c, s) => {
        const other = s === 0 ? e : q;
        const P = c.blockSurface(b, other.blockWorld(b, tmp), new THREE.Vector3());
        const { sp, ring } = flashes[k][s];
        sp.position.copy(P); sp.scale.setScalar(0.3 + 1.6 * f); sp.material.opacity = f; sp.visible = f > 0.01;
        const rt = t - tg; const ron = !phase3 && rt > 0 && rt < 0.6;
        ring.visible = ron; if (ron) { ring.position.copy(c.blockWorld(b, tmp)); ring.scale.setScalar(c.blockRadius(b) * (1.1 + rt * 2.2)); ring.material.opacity = (1 - rt / 0.6) * 0.9; }
      });
    });

    // labels: a column beside Ember, each appearing with its bridge
    far.copy(e.pose.position).add(new THREE.Vector3(6, 0, 0));
    labels.forEach((L, k) => {
      const o = phase3 ? 0 : prog(t, T_GROW(k) + 0.05, T_GROW(k) + 0.35) * (1 - prog(t, 13.7, 13.88));
      const P = e.blockSurface(XBLOCKS[k], far, new THREE.Vector3());
      L.update(P, camera, o, 26 + (1 - o) * 14, 0);
      if (k === DIVE_E) L.root.style.textShadow = t > 13.6 ? '0 0 14px rgba(52,245,192,0.9)' : '0 1px 12px rgba(0,0,0,0.9)';
    });
    cap.update(t, 10.3, 13.95);
    // the sheets' 3D token tiles stay sharp: keep bloom off them

    // post
    if (!phase3) {
      post.bloomStrength = 0.75 + 0.2 * beatPulse(t, 5) * (t < 14 ? 1 : 0) + 1.2 * prog(t, 14.2, 14.5, easeInCubic);
      post.bloomThreshold = 0.22 + 0.3 * win(t, 13.85, 14.4, 0.1);
      post.flash = 0.9 * prog(t, 14.3, 14.5, easeInCubic) + 0.18 * Math.exp(-6 * (t - 10)) ;
      post.flashColor = '#dffcf2';
      post.aberration = 0.0018 + 0.006 * prog(t, 14.0, 14.5, easeInCubic);
    } else {
      const lt = t - T_CUT2;
      post.flash = 0.3 * Math.exp(-10 * lt) + 0.9 * prog(t, 21.82, 22.0, easeInCubic);
      post.flashColor = lt < 0.4 ? '#ffe2c4' : '#ffffff';
      post.bloomStrength = 0.85 + 0.4 * Math.exp(-5 * lt);
      post.aberration = 0.0018 + 0.004 * Math.exp(-6 * lt);
    }
    return post;
  }
  return inst;
}
