// a4_games, global 50–64 s (authored in global time). The games, as nine glass windows in 3D; each window is a live
// 3D diorama (shots/games_win3d.js): the two creatures, the link running both ways, the private item at the holder's
// tail (only that side has it), the other side writing the real answer token by token, then a check.
//   wave A 50–53.5: relay a word (Quill -> Ember), a number (Ember -> Quill), a phrase (Ember -> Quill)
//   wave B 53.5–58: relay digits (Quill -> Ember), add two numbers (both write), name and city (both write)
//   wave C 58–62: look up a code (both write), shared reading (passage / question), free chat (Taipei)
//   Each wave: windows slam in on 8ths (W, W+.25, W+.5), the read pass at slam+.375, one token per 16th from W+1
//   (the sum from W+.75), a check on the 16th after each writer's last token; waves A and B fly up and out at
//   53.125 and 57.625. See cues/a4_games.md.
//   62.0–62.5 every window comes back and folds into a deck (the training data), lands on 62.5, compresses on 63.0,
//   flies off into the dark 63.25–63.875 with a teal spark; a5 takes over at 64.
// update(t) is a pure function of global t.
import * as THREE from 'three';
import { makeWorld } from '../lib/env.js';
import { C } from '../lib/palette.js';
import { clamp, lerp, prog, spring, kick, hash, easeInCubic, easeOutCubic, easeInOutCubic } from '../lib/util.js';
import { Diorama, Pane } from './games_win3d.js';
import { GAMES } from './kit_gamewin.js';

const ORDER = ['word', 'number', 'phrase', 'digits', 'sum', 'fact', 'lookup', 'reading', 'chat'];
const WAVE = [50, 53.5, 58];
const SLAM = (i) => WAVE[Math.floor(i / 3)] + 0.25 * (i % 3);
const GEN = (i) => WAVE[Math.floor(i / 3)] + (ORDER[i] === 'sum' ? 0.75 : 1.0);
const EXIT = (i) => [53.125, 57.625][Math.floor(i / 3)];
const FOLD0 = 62.0, LAND = 62.5, PRESS = 63.0, FLY0 = 63.25, FLY1 = 63.875;
const PW = 560, PH = 760, HDR = 78, BD = 14, GAP = 34;
const PPU = 150;                       // outer px per world unit at z = 0
const FOV = 35, DIST = 1080 / (2 * Math.tan((FOV / 2) * Math.PI / 180) * PPU);
const CY = 3.2;                        // window centre height
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

export async function create({ renderer, scale, def }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 16 / 9, 0.1, 200);
  const world = makeWorld(scene, { fogDensity: 0.022 });
  // big soft colour fields behind the wall
  const glowTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 128; const x = c.getContext('2d'); const gr = x.createRadialGradient(64, 64, 0, 64, 64, 64); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.4, 'rgba(255,255,255,0.35)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); x.fillStyle = gr; x.fillRect(0, 0, 128, 128); const tx = new THREE.CanvasTexture(c); tx.colorSpace = THREE.SRGBColorSpace; return tx; })();
  const glows = [['#14e3a8', 0, 3.5, -12, 20, 0.18], ['#8b5cf6', -11, 4.5, -14, 20, 0.26], ['#f97316', 11, 4, -14, 20, 0.2]].map(([col, x, y, z, s, o]) => {
    const m = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: new THREE.Color(col), transparent: true, opacity: o, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
    m.position.set(x, y, z); m.scale.set(s, s, 1); m.userData.o = o; scene.add(m); return m;
  });
  const spark = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: new THREE.Color('#bfffe9'), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, fog: false, toneMapped: false }));
  scene.add(spark);

  const W = PW + 2 * BD, H = PH + HDR + 2 * BD;
  const wins = ORDER.map((key, i) => {
    const spec = { ...GAMES[key] };
    if (spec.write && spec.write.length > 1 && spec.cs == null) spec.cs = 0.62;
    const d = new Diorama(renderer, spec, { w: PW, h: PH, scale, ppu: 70, tileH: 0.84 });
    const pane = new Pane({ w: PW, h: PH, header: HDR, border: BD, title: spec.title, titlePx: 50, tex: d.rt.texture, ppu: PPU });
    scene.add(pane.group);
    const j = i % 3;
    const T = Diorama.timeline(spec, SLAM(i), { gen: GEN(i) - SLAM(i) });
    const home = { pos: V(((j - 1) * (W + GAP)) / PPU, CY, -Math.abs(j - 1) * 0.12), rot: new THREE.Euler(0, (j - 1) * -0.05, 0) };
    return { key, spec, d, pane, j, i, T, home };
  });
  const DECK = V(0, 1.15, 0.6);

  function camAt(t) {
    const L = t - 50;
    const pos = V(Math.sin(L * 0.21) * 0.35, CY + 0.12 * Math.sin(L * 0.17), DIST + 0.15 * Math.sin(L * 0.13));
    const tgt = V(Math.sin(L * 0.21) * 0.12, CY, 0);
    // crane down onto the deck
    const k = easeInOutCubic(prog(t, FOLD0 - 0.1, LAND + 0.2, (x) => x));
    pos.lerp(V(0, 5.6, 8.6), k); tgt.lerp(V(0, 1.35, 0), k);
    // follow the deck as it leaves
    tgt.lerp(V(0, 3.4, -30), 0.85 * easeInOutCubic(prog(t, FLY0 - 0.1, FLY1, (x) => x)));
    // shake on slams and the landing
    let sh = 0; for (let i = 0; i < 9; i++) sh += kick(t, SLAM(i), 6, 10) * 0.035;
    sh += kick(t, LAND, 5, 7) * 0.06 + kick(t, PRESS, 5, 8) * 0.03;
    pos.y += sh; tgt.y += sh * 0.5;
    return { pos, tgt };
  }

  return {
    scene, camera,
    update(t) {
      world.update(t);
      const cam = camAt(t);
      camera.position.copy(cam.pos); camera.lookAt(cam.tgt); camera.updateMatrixWorld(true);
      const fly = easeInCubic(prog(t, FLY0, FLY1, (x) => x));
      const deckGlow = (t >= LAND ? Math.exp(-(t - LAND) * 2.5) : 0) + (t >= PRESS ? 0.8 * Math.exp(-(t - PRESS) * 3) : 0);
      glows.forEach((m, k) => { m.material.opacity = m.userData.o * (0.85 + 0.15 * Math.sin(t * 0.9 + k * 2)) * (1 + (k === 0 ? 0.4 * deckGlow : 0)) * (1 - 0.8 * fly); });

      for (const w of wins) {
        const { pane, i, T, home } = w; const g = pane.group;
        const s0 = SLAM(i), ex = EXIT(i);
        let vis = t >= s0 - 0.22;
        const pos = home.pos.clone(), rot = home.rot.clone(); let sc = 1, a = 1, hit = 0, dim = 1;
        // slam in from in front of the camera
        const pIn = easeInCubic(clamp((t - (s0 - 0.22)) / 0.22));
        const imp = t >= s0 ? kick(t, s0, 3.2, 7) : 0;
        pos.z += (1 - pIn) * 3.2; pos.y += (1 - pIn) * -0.9; rot.x += (1 - pIn) * 0.55; rot.z += (1 - pIn) * (w.j - 1) * -0.08;
        sc *= 1 + 0.025 * imp;
        a = clamp((t - (s0 - 0.22)) / 0.1);
        hit = Math.max(t >= s0 ? Math.exp(-(t - s0) * 4.5) : 0, ...T.check.map((c) => (t >= c ? 0.8 * Math.exp(-(t - c) * 5) : 0)));
        // exit: up and back (waves A and B), parked out of frame until the fold
        const parked = { pos: home.pos.clone().add(V(0, 7.5, -4)), rot: new THREE.Euler(-0.5, home.rot.y, 0) };
        if (i < 6 && t >= ex) {
          const u = easeInCubic(clamp((t - ex - (w.j) * 0.04) / 0.32));
          pos.lerp(parked.pos, u); rot.x = lerp(rot.x, parked.rot.x, u);
          if (u >= 1 && t < FOLD0) vis = false;
        }
        // fold into the deck
        if (t >= FOLD0) {
          const start = i < 6 ? parked : { pos: home.pos, rot: home.rot };
          const u = easeInOutCubic(clamp((t - FOLD0 - i * 0.035) / (LAND - FOLD0 - 0.28)));
          const k = i; // stack order: first game at the bottom
          const press = 1 - 0.55 * easeInOutCubic(prog(t, PRESS, PRESS + 0.2, (x) => x));
          const slot = DECK.clone().add(V((hash(k * 3.1) - 0.5) * 0.12, k * 0.075 * press, (hash(k * 1.7) - 0.5) * 0.08));
          pos.copy(start.pos).lerp(slot, u); pos.y += Math.sin(Math.PI * u) * 0.8;
          rot.set(lerp(start.rot.x, -1.32, u), lerp(start.rot.y, (hash(k * 5.3) - 0.5) * 0.18, u), lerp(0, (hash(k * 2.9) - 0.5) * 0.1, u));
          sc = lerp(1, 0.62, u);
          vis = true;
          dim = lerp(1, 0.75, u);
          hit = Math.max(hit * (1 - u), (t >= LAND ? 0.45 * Math.exp(-(t - LAND) * 3) : 0) + (t >= PRESS ? 0.35 * Math.exp(-(t - PRESS) * 4) : 0) + 0.12 * u);
          // fly off
          if (t >= FLY0) {
            const f = fly;
            pos.add(V(0, f * 3, -f * 40)); sc *= lerp(1, 0.6, f);
            a = 1 - prog(t, FLY1 - 0.18, FLY1);
            hit = Math.max(hit, 0.25 + 0.5 * f);
          }
        }
        g.visible = vis;
        if (!vis) continue;
        g.position.copy(pos); g.rotation.copy(rot); g.scale.setScalar(sc / PPU);
        pane.set(a, hit, dim);
        if (t >= FOLD0) { pane.glowM.opacity *= 0.12; pane.edgeM.uniforms.uOp.value *= 0.5; }
        // the diorama: live until its wave leaves, frozen after
        const tEff = Math.min(t, i < 6 ? ex : FOLD0);
        const off = V((camera.position.x - g.position.x) * 0.5, (camera.position.y - g.position.y) * 0.25, 0);
        if (t >= FOLD0) off.set(0, 0, 0);
        w.d.render(tEff, T, { camOff: t < FOLD0 ? off : null });
      }
      // spark where the deck vanishes
      const sp = Math.min(clamp((t - (FLY1 - 0.2)) / 0.15), 1 - clamp((t - FLY1 - 0.05) / 0.1));
      spark.position.copy(DECK).add(V(0, 3 * fly, -40 * fly));
      spark.scale.setScalar(1.5 + 6 * sp); spark.material.opacity = sp;

      const slamE = Array.from({ length: 9 }, (_, i) => (t >= SLAM(i) ? Math.exp(-(t - SLAM(i)) * 10) : 0)).reduce((x, y) => x + y, 0);
      return {
        bloomStrength: 0.5 + 0.2 * deckGlow, bloomRadius: 0.45, bloomThreshold: 0.34, vignette: 0.9,
        flash: 0.03 * slamE + 0.05 * (t >= LAND ? Math.exp(-(t - LAND) * 6) : 0) + 0.12 * sp, flashColor: '#34f5c0',
        aberration: 0.0016 + 0.003 * slamE,
        fade: 0.85 * prog(t, FLY1, 64.0),
      };
    },
  };
}
