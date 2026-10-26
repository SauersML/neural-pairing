// S3 "THE GAMES" (22–36 s, bars 11–17).
//  22.0–30.5 the two models unrolled (shots/s3_play_sheets.js, lib/grid.js): ticks, then the word relay
//            (creatures bookend it: they unroll at 22.1 and roll back up at 30.0, Quill hopping with joy).
//  30.5–33.0 value-game montage, one vignette per beat; written answers collect on a shelf.
//  33.0–36.0 play: "where I'm on holiday". Ember: `Ah, Bangkok! … mango smoothie by the Chao Phraya River…`
// Exactness: tokens only enter at the tail (prompt) or leave the head (written text). Only packets (dots) travel
// on the link. update(t) is a pure function of t.
import * as THREE from 'three';
import { Creature, XBLOCKS, NBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Chapter, Pin } from '../lib/hud.js';
import { tokenTile, TOKEN_STYLE } from '../lib/tokens.js';
import { C, CSS } from '../lib/palette.js';
import { clamp, lerp, prog, win, beatPulse, kick, easeInOutCubic, easeOutCubic, easeInCubic, hash } from '../lib/util.js';
import { GOLD, GOLD_C, hops, nod, blink, pop, bez, yawTo, lerpAng, bubbleTexture, tableTexture, plane } from './s3_play_kit.js';
import { makeSheets } from './s3_play_sheets.js';

const G = (s) => `<span style="color:${GOLD}">${s}</span>`;
const Q = (s) => `<span style="color:${CSS.quill}">${s}</span>`;
const E = (s) => `<span style="color:${CSS.ember}">${s}</span>`;
const L = (s) => `<span style="color:${CSS.link}">${s}</span>`;

const ST = {
  q: TOKEN_STYLE.quill, e: TOKEN_STYLE.ember, secret: TOKEN_STYLE.secret,
  gq: { color: '#fff7d6', bg: 'rgba(84,58,0,0.92)', border: GOLD, radius: 10, glow: GOLD },
  ge: { color: '#fff7d6', bg: 'rgba(84,58,0,0.92)', border: GOLD, radius: 50, glow: GOLD },
};

export async function create({ hud, def }) {
  try { await document.fonts.load('600 64px Inter'); } catch (_) { /* optional */ }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  scene.add(camera);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  const sheets = makeSheets(scene, hud);

  const T = (text, style, h = 0.36) => { const m = tokenTile(text, ST[style] || style, h); m.material.fog = false; m.material.color.setScalar(0.78); m.renderOrder = 20; m.visible = false; scene.add(m); return m; };
  const dot = Creature.dotTexture();

  // gold packets riding the Quill->Ember strands (a value's information; dots only)
  const NGO = XBLOCKS.length * 3; const goldGeo = new THREE.BufferGeometry();
  goldGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NGO * 3), 3));
  goldGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NGO * 3), 3));
  const goldPts = new THREE.Points(goldGeo, new THREE.PointsMaterial({ size: 0.5, map: dot, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  scene.add(goldPts);
  // goldPts carries Quill->Ember; goldBack carries Ember->Quill (montage games where Ember holds the value)
  const goldGeo2 = goldGeo.clone(); const goldBack = new THREE.Points(goldGeo2, goldPts.material); scene.add(goldBack);

  // ================= value-game montage =================
  const VIG = [
    { t: 30.5, chips: { q: ['5203'] }, to: 'e', out: [['520', 'ge'], ['3', 'ge']] },
    { t: 31.0, chips: { e: ['silver whale'] }, to: 'q', out: [['silver', 'gq'], ['whale', 'gq']] },
    { t: 31.5, chips: { q: ['31'], e: ['42'] }, to: 'e', out: [['31', 'ge'], ['+', 'e'], ['42', 'e'], ['=', 'e'], ['73', 'ge']] },
    { t: 32.0, chips: { q: ['Keiko'], e: ['Geneva'] }, to: 'q', out: [['Keiko', 'q'], ['lives', 'q'], ['in', 'q'], ['Geneva', 'gq'], ['.', 'q']] },
    { t: 32.5, chips: { q: ['code M2'], e: ['@table'] }, to: 'q', out: [['code', 'q'], ['M2', 'q'], ['→', 'q'], ['breeze', 'gq']] },
  ];
  const OH = 0.5;
  for (const v of VIG) {
    v.chipMeshes = [];
    for (const s of ['q', 'e']) for (const c of (v.chips[s] || [])) {
      const m = c === '@table' ? (() => { const p = plane(tableTexture([['code K7', 'candle'], ['code M2', 'breeze']], 1), 1.3); p.renderOrder = 20; p.visible = false; scene.add(p); return p; })() : T(c, 'secret', 0.46);
      v.chipMeshes.push({ s, m });
    }
    v.tiles = v.out.map(([s, st]) => T(s, st, OH));
  }
  // shelf (screen-anchored, right column): row 0 = "lantern" from the relay, rows 1..5 = the vignettes
  const shelfLantern = T('lantern', 'ge', OH);
  const SHELF_X = 1235, SHELF_Y0 = 250, SHELF_DY = 70, SHELF_HPX = 48;
  const SHELF_D = 5, PXU = 540 / (SHELF_D * Math.tan((35 / 2) * Math.PI / 180));

  // ================= play =================
  const bangkok = T('Bangkok', 'secret', 0.5);
  const bubTex = bubbleTexture(1100, 640);
  const BT = 0.44, BH = 2.6;
  const bubble = plane(bubTex, BH); bubble.material.color.setScalar(0.85); bubble.renderOrder = 15; bubble.visible = false; scene.add(bubble);
  const BUB = [
    [['Ah,', 'e'], ['Bangkok!', 'ge'], ['…', 'e']],
    [['mango', 'e'], ['smoothie', 'e'], ['by', 'e'], ['the', 'e']],
    [['Chao', 'e'], ['Phraya', 'e'], ['River…', 'e']],
  ];
  const bubTiles = []; let bi = 0;
  BUB.forEach((row, r) => { let x = 0; const ms = row.map(([s, st]) => T(s, st, BT)); const w = ms.reduce((a, m) => a + m.userData.width, 0) + 0.06 * (ms.length - 1); ms.forEach((m) => { bubTiles.push({ m, r, x: x + m.userData.width / 2, w, i: bi++ }); x += m.userData.width + 0.06; }); });
  const bubLand = (i) => 34.0 + 0.1 * i; // "Bangkok!" (i=1) lands at 34.1
  const secretPin = new Pin(hud, `Quill's secret`, { color: GOLD, size: 34, dot: false, anchor: 'center' });

  // ================= HUD =================
  const chap = new Chapter(hud, '', 'The games');
  const capA = new Caption(hud, { kicker: 'time', head: `One token per ${L('tick')}`, sub: 'A token attends to partner tokens from its tick or earlier.' });
  const capB = new Caption(hud, { kicker: 'relay a word', head: `Only ${Q("Quill's")} prompt has the word`, sub: `${E('Ember')} reads ${Q("Quill's")} hidden states and writes it.` });
  const capC = new Caption(hud, { head: `Seven ${G('value')} games`, sub: 'words, numbers, phrases, digit strings, sums, name + city, lookups' });
  const capD = new Caption(hud, { kicker: 'play', head: 'Six-turn games', sub: "Each model names a secret that appears only in the other's prompt." });

  // ================= helpers =================
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const camR = new THREE.Vector3(), camU = new THREE.Vector3();
  const bbAt = (anchor, dx, dy, out = new THREE.Vector3()) => out.copy(anchor).addScaledVector(camR, dx).addScaledVector(camU, dy);
  const place = (m, pos, s = 1, op = 1) => { m.visible = s > 0.002 && op > 0.002; if (!m.visible) return; m.position.copy(pos); m.quaternion.copy(camera.quaternion); m.scale.setScalar(s); m.material.opacity = op; };
  const shelfWorld = (row, xpx, out = new THREE.Vector3()) => {
    const y = SHELF_Y0 + row * SHELF_DY; const lx = (xpx - 960) / PXU, ly = -(y - 540) / PXU;
    return camera.localToWorld(out.set(lx, ly, -SHELF_D));
  };
  const SHELF_S = (SHELF_HPX / PXU) / OH;
  function resetPose(c) { Object.assign(c.pose, Creature.defaultPose(c.side)); c.blockGlow.fill(0); for (let i = 0; i < NBLOCKS; i++) c.blockColor[i] = null; }
  // a gold pulse climbing the spine (value information entering at the tail and rising)
  const goldWave = (c, t, t0, dur, from = 0, to = NBLOCKS - 1, hold = 1) => {
    if (t < t0) return; const front = from + (to - from) * clamp((t - t0) / dur);
    const after = clamp(1 - (t - t0 - dur - hold) / 0.6);
    for (let i = from; i <= to; i++) {
      const pk = Math.exp(-Math.pow((i - front) / 1.8, 2)) * clamp(1 - (t - t0 - dur) / 0.25);
      const lit = (i <= front ? 1 : 0) * after;
      if (lit > 0.02 || pk > 0.02) { c.blockColor[i] = c.pal.col.clone().lerp(GOLD_C, Math.min(1, lit * 0.3 + pk)); c.blockGlow[i] = Math.max(c.blockGlow[i], 0.8 * pk); }
    }
  };
  const gold = new THREE.Color(GOLD);

  return {
    scene, camera,
    update(t) {
      world.update(t);
      const post = { bloomStrength: 0.85, bloomRadius: 0.55, bloomThreshold: 0.22, vignette: 0.95, flash: 0.32 * (1 - prog(t, 22.0, 22.25)) };

      // ---------------- camera ----------------
      const cS = () => { const u = t - 22; return { p: V(0.5 * Math.sin(u * 0.25), 2.5, 15.9 - 0.06 * u), l: V(0, 1.62, 0) }; };
      const cC = () => { const u = t - 30.5; return { p: V(2.2 + 0.3 * u, 3.7, 17.6 - 0.3 * u), l: V(3.1, 2.15, 0) }; };
      const cD = () => { const u = t - 33; const a = 0.1 - u * 0.07; return { p: V(1.9 + Math.sin(a) * 17.6, 3.6 + 0.06 * u, Math.cos(a) * 17.6 - 0.3 * u), l: V(1.9, 2.15, 0) }; };
      const seq = [[cS, 29.95, 30.6], [cC, 33.0, 33.6], [cD, 99, 100]];
      let cp = seq[seq.findIndex((s) => t < s[2])][0]();
      for (let i = 0; i < seq.length - 1; i++) {
        const [, s1, s2] = seq[i];
        if (t >= s1 && t < s2) { const k = easeInOutCubic((t - s1) / (s2 - s1)); const a = seq[i][0](), b = seq[i + 1][0](); cp = { p: a.p.lerp(b.p, k), l: a.l.lerp(b.l, k) }; }
      }
      const shake = (t > 30.4 && t < 33.0 ? 0.035 : 0.01) * beatPulse(t, 9);
      camera.position.copy(cp.p).add(V(0, shake, 0)); camera.lookAt(cp.l);
      camera.fov = 35; camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
      camR.set(1, 0, 0).applyQuaternion(camera.quaternion); camU.set(0, 1, 0).applyQuaternion(camera.quaternion);

      // ---------------- creatures ----------------
      // present at the very start (they unroll into the sheets) and from 30.0 (the sheets roll back up)
      const cOp = clamp(1 - prog(t, 22.15, 22.42) + prog(t, 30.05, 30.32));
      // the body thins into one column as it unrolls into the sheet (and back)
      const squeeze = clamp(prog(t, 22.08, 22.4) * (1 - prog(t, 29.98, 30.3)));
      resetPose(q); resetPose(e);
      const qBase = V(-2.3, 0, 0), eBase = V(2.3, 0, 0);
      q.pose.position.copy(qBase); e.pose.position.copy(eBase);
      for (const [c, seed] of [[q, 0], [e, 1.7]]) {
        const p = c.pose;
        p.swayPhase = t * 2.3 + seed * 2; p.sway = 0.09; p.swayFreq = 1;
        p.squash = 1 - 0.022 * beatPulse(t, 6);
        p.blink = blink(t, seed);
        p.lean = 0.035 * Math.sin(t * 1.2 + seed);
        p.headTilt = 0.07 * Math.sin(t * 0.9 + seed * 2.1);
        p.portGlow = 1.05 + 0.25 * beatPulse(t, 4); p.groundGlow = 0.45;
        p.opacity = cOp; p.glow = 1 + 1.4 * squeeze;
      }
      q.pose.yaw = lerpAng(yawTo(qBase, camera.position), yawTo(qBase, eBase), 0.35);
      e.pose.yaw = lerpAng(yawTo(eBase, camera.position), yawTo(eBase, qBase), 0.35);
      let qLook = V(2.3, 5.3, 0), eLook = V(-2.3, 5.3, 0);
      const hq = [], he = [];
      let qHappy = 0, eHappy = 0, qCrest = 1, eCrest = 1, qNod = 0, eNod = 0;
      const ls = link.state;
      ls.grow = 1; ls.intensity = 0.6; ls.flow = 1; ls.packets = 3; ls.aToB = 1; ls.bToA = 1; ls.highlight = -1; ls.dim = 1; ls.sag = -0.25; ls.twist = 0.07; ls.colorBySender = 0.55; ls.packetSize = 0.16;
      let goldQE = 0, goldEQ = 0;

      // bookend in: delighted, a stretch as they unroll
      if (t < 22.6) { qHappy = eHappy = 1 - prog(t, 22.2, 22.4); q.pose.squash *= 1 + 0.1 * prog(t, 22.0, 22.3); e.pose.squash *= 1 + 0.1 * prog(t, 22.0, 22.3); }
      // bookend out: the relay worked. Quill hops with joy, Ember beams.
      if (t > 29.9 && t < 30.7) {
        qHappy = Math.max(qHappy, prog(t, 30.05, 30.2)); eHappy = Math.max(eHappy, prog(t, 30.0, 30.15));
        qCrest = 1 + 1.3 * win(t, 30.1, 30.9, 0.15); qLook = e.headWorld();
      }
      hq.push([30.18, 0.8, 0.5]); he.push([30.28, 0.3, 0.36]);

      // ---------------- montage (30.5–33) ----------------
      if (t >= 30.3 && t < 33.4) {
        for (const v of VIG) {
          const rec = v.to, snd = rec === 'q' ? 'e' : 'q';
          (snd === 'q' ? hq : he).push([v.t - 0.02, 0.16, 0.3]);
          (rec === 'q' ? hq : he).push([v.t + 0.14, 0.38, 0.34]);
          if (rec === 'e') eNod += nod(t, v.t + 0.06, 0.22); else qNod += nod(t, v.t + 0.06, 0.22);
          if (t > v.t && t < v.t + 0.5) {
            if (rec === 'e') eHappy = Math.max(eHappy, win(t, v.t + 0.16, v.t + 0.5, 0.06)); else qHappy = Math.max(qHappy, win(t, v.t + 0.16, v.t + 0.5, 0.06));
          }
          // value information crosses toward the receiver (gold dots on the strands)
          const g = 0.9 * Math.exp(-Math.max(0, t - v.t) * 3) * (t > v.t - 0.1 ? 1 : 0);
          if (rec === 'e') goldQE = Math.max(goldQE, g); else goldEQ = Math.max(goldEQ, g);
          if (v.chips.q) goldWave(q, t, v.t - 0.05, 0.22, 0, 27, -0.5);
          if (v.chips.e) goldWave(e, t, v.t - 0.05, 0.22, 0, 27, -0.5);
        }
        ls.intensity = 0.7 + 0.3 * beatPulse(t, 5); ls.flow = 1.5; ls.packets = 5; ls.packetSize = 0.2;
        qLook = e.headWorld(); eLook = q.headWorld();
        qCrest = Math.max(qCrest, 1 + 0.5 * beatPulse(t, 5)); eCrest = 1 + 0.4 * beatPulse(t, 5);
        q.pose.sway = 0.13; e.pose.sway = 0.13;
      }

      // ---------------- play (33–36) ----------------
      if (t >= 33.0) {
        goldWave(q, t, 33.55, 0.45, 0, NBLOCKS - 1, 1.0);
        goldQE = Math.max(goldQE, prog(t, 33.6, 33.9) * (0.75 + 0.25 * beatPulse(t, 4)));
        ls.intensity = 0.65 + 0.3 * goldQE; ls.flow = 1.3; ls.packets = 5; ls.packetSize = 0.22;
        const portIn = prog(t, 33.75, 33.95);
        for (const b of XBLOCKS) if (portIn > 0.01) { e.blockColor[b] = GOLD_C; e.blockGlow[b] = Math.max(e.blockGlow[b], 0.5 * portIn); }
        goldWave(e, t, 33.8, 0.25, 6, NBLOCKS - 1, 0.6);
        for (let i = 0; i < bubTiles.length; i++) eNod += nod(t, bubLand(i) - 0.18, 0.12);
        if (t < 33.6) qLook = V(-2.3, 0.5, 0.9);
        if (t > 33.9) eLook = V(4.4, 6.0, 0);
        if (t > 34.6) eLook = q.headWorld();
        const laugh = prog(t, 34.1, 34.3);
        qHappy = Math.max(qHappy, laugh); eHappy = Math.max(eHappy, prog(t, 33.95, 34.15));
        qCrest = 1 + 1.2 * laugh * (0.7 + 0.3 * beatPulse(t, 4)); eCrest = 1 + 0.8 * laugh;
        const gig = laugh * Math.sin(t * 2 * Math.PI * 6.5);
        q.pose.squash *= 1 + 0.025 * gig; q.pose.headTilt += 0.06 * gig + 0.15 * laugh * Math.sin(t * 3.1);
        e.pose.squash *= 1 - 0.02 * gig; e.pose.headTilt -= 0.05 * gig;
        hq.push([34.12, 0.75, 0.48], [35.0, 0.45, 0.42]); he.push([34.55, 0.4, 0.4], [35.5, 0.42, 0.42]);
        q.pose.lean += 0.08 * laugh; e.pose.lean -= 0.06 * laugh;
        if (t > 34.3) qLook = e.headWorld();
      }

      // ---------------- apply character layers ----------------
      const hQ = hops(t, hq), hE = hops(t, he);
      q.pose.position.y += hQ.y; q.pose.squash *= hQ.sq;
      e.pose.position.y += hE.y; e.pose.squash *= hE.sq;
      q.pose.headNod += qNod; e.pose.headNod += eNod;
      q.pose.happy = qHappy; e.pose.happy = eHappy;
      if (qHappy > 0.5) q.pose.blink = 0; if (eHappy > 0.5) e.pose.blink = 0;
      q.pose.crest = Math.min(1.8, qCrest); e.pose.crest = Math.min(1.6, eCrest);
      q.pose.blush = 0.35 + 0.4 * qHappy; e.pose.blush = 0.35 + 0.4 * eHappy;
      q.pose.lookAt = qLook; e.pose.lookAt = eLook;
      q.update(t); e.update(t);
      q.crest.scale.setScalar(1 + 0.22 * (qCrest - 1)); e.crest.scale.setScalar(1 + 0.18 * (eCrest - 1));
      for (const c of [q, e]) { c.group.scale.x = c.group.scale.z = lerp(1, 0.12, squeeze); c.group.updateMatrixWorld(true); }
      q.group.visible = e.group.visible = cOp > 0.01;
      ls.intensity *= cOp; link.group.visible = cOp > 0.01;
      link.update(t);
      // gold tint on the packets of the direction that carries the value, plus larger gold dots
      for (const st of link.strands) {
        const g = st.dir === 0 ? goldQE : goldEQ; if (g <= 0.001) continue;
        const col = st.pts.geometry.attributes.color.array;
        for (let k = 0; k < col.length; k += 3) { const m = (col[k] + col[k + 1] + col[k + 2]) / 3 * 1.2; col[k] = lerp(col[k], gold.r * m * 1.3, g); col[k + 1] = lerp(col[k + 1], gold.g * m * 1.3, g); col[k + 2] = lerp(col[k + 2], gold.b * m * 1.3, g); }
        st.pts.geometry.attributes.color.needsUpdate = true;
      }
      for (const [geo, pts, dir, g] of [[goldGeo, goldPts, 0, goldQE], [goldGeo2, goldBack, 1, goldEQ]]) {
        const pa = geo.attributes.position.array, ca = geo.attributes.color.array; let n = 0;
        for (const st of link.strands) {
          if (st.dir !== dir) continue;
          for (let k = 0; k < 3; k++, n++) {
            const u = (t * 0.8 + k / 3 + hash(st.e * 3.3 + dir)) % 1;
            const p = st.curve.getPoint(u); pa[n * 3] = p.x; pa[n * 3 + 1] = p.y; pa[n * 3 + 2] = p.z;
            const f = Math.pow(Math.sin(Math.PI * u), 0.6) * g * cOp;
            ca[n * 3] = gold.r * f * 1.4; ca[n * 3 + 1] = gold.g * f * 1.4; ca[n * 3 + 2] = gold.b * f * 1.4;
          }
        }
        geo.attributes.position.needsUpdate = true; geo.attributes.color.needsUpdate = true;
        pts.visible = g * cOp > 0.002;
      }

      // ---------------- the unrolled sheets (22–30.5) ----------------
      sheets.update(t, camera, post);

      const qTail = q.tailWorld(), eTail = e.tailWorld(), eMouth = e.mouthWorld();
      // ---------------- montage objects ----------------
      const shelfOut = 1 - prog(t, 33.25, 33.5);
      const shelfOn = t >= 30.3 && t < 33.55;
      place(shelfLantern, shelfWorld(0, SHELF_X + shelfLantern.userData.width * SHELF_S * PXU / 2), shelfOn ? pop(t, 30.42, 2.6, 0.45) * SHELF_S : 0, shelfOut);
      VIG.forEach((v, vi) => {
        const rec = v.to === 'q' ? q : e;
        // the values enter the holders' tails as prompt tokens
        v.chipMeshes.forEach(({ s, m }) => {
          const tail = s === 'q' ? qTail : eTail;
          const t0 = v.t - 0.2; const k = pop(t, t0, 3.0, 0.45); const sw = prog(t, v.t + 0.08, v.t + 0.3, easeInCubic);
          const base = V(tail.x + (s === 'q' ? -0.45 : 0.45), (m.geometry.parameters.height / 2) + 0.2, 1.1);
          if (t < t0 + 0.12) base.y += (1 - easeOutCubic(clamp((t - t0) / 0.12))) * 1.2;
          const p = base.clone().lerp(V(tail.x, tail.y, tail.z), sw);
          place(m, p, shelfOn && t > t0 ? k * (1 - sw) : 0, 1);
        });
        // the receiver writes: tokens leave its head into a clean row between the heads, then the row flies to the shelf
        const mouth = rec.mouthWorld();
        const rowW = v.tiles.reduce((acc, m) => acc + m.userData.width, 0) + 0.14 * (v.tiles.length - 1);
        const rowAnchor = V(v.to === 'q' ? -0.2 : 0.4, 6.15, 0.6);
        let xs = SHELF_X, xr = -rowW / 2;
        v.tiles.forEach((m, i) => {
          const w = m.userData.width; const t0 = v.t + 0.04 + 0.035 * i;
          const wpx = w * SHELF_S * PXU; const dst = shelfWorld(vi + 1, xs + wpx / 2); xs += wpx + 7;
          const slot = bbAt(rowAnchor, xr + w / 2, 0); xr += w + 0.14;
          const u1 = easeOutCubic(clamp((t - t0) / 0.16));            // mouth -> row
          const tF = v.t + 0.3, u2 = easeInOutCubic(clamp((t - tF) / 0.24)); // row -> shelf
          // each token pops out at its own slot, nudged in from the direction of the head (no tile crosses another)
          const p = slot.clone().addScaledVector(mouth.clone().sub(slot).normalize(), 0.45 * (1 - u1));
          const c2 = slot.clone().lerp(dst, 0.5); c2.y += 0.25;
          if (u2 > 0) p.copy(bez(slot, c2, dst, u2));
          const land = tF + 0.24;
          let sc = lerp(0.3, 1, u1) * (1 + 0.06 * Math.max(0, kick(t, t0 + 0.16, 3, 6)));
          if (u2 > 0) sc = t < land ? lerp(1, SHELF_S, u2) : SHELF_S * (1 + 0.35 * Math.max(0, kick(t, land, 3, 6)));
          place(m, p, shelfOn && t > t0 ? sc : 0, shelfOut);
        });
      });

      // ---------------- play objects ----------------
      {
        const on = t >= 33.0;
        const cp3 = V(-2.3, 0.36, 0.95);
        if (t < 33.35) cp3.y = lerp(2.0, 0.36, easeInCubic(clamp((t - 33.08) / 0.27))); else cp3.y += 0.14 * Math.max(0, kick(t, 33.35, 2.5, 5));
        const sw = prog(t, 33.5, 33.72, easeInCubic); cp3.lerp(V(qTail.x, qTail.y, qTail.z), sw);
        place(bangkok, cp3, on && t > 33.05 ? (1 - sw) : 0, 1);
        secretPin.update(V(-2.3, 1.05, 0.95), camera, on ? win(t, 33.15, 34.0, 0.15) : 0, 0, -10);
        const eh = e.headWorld();
        const bAnchor = bbAt(eh, 2.75, 0.75);
        const bk = pop(t, 33.82, 2.2, 0.5);
        place(bubble, bAnchor, on && t > 33.82 ? bk : 0, 1);
        bubTiles.forEach((bt) => {
          const slotP = bbAt(bAnchor, (-bt.w / 2 + bt.x) * bk, (0.086 * BH + (1 - bt.r) * BT * 1.25) * bk);
          const land = bubLand(bt.i), t0 = land - 0.22;
          if (!on || t < t0) { bt.m.visible = false; return; }
          const u = easeOutCubic(clamp((t - t0) / 0.22));
          const ctrl = eMouth.clone().lerp(slotP, 0.5); ctrl.y += 0.4;
          const p = bez(eMouth, ctrl, slotP, u);
          p.y += 0.05 * Math.max(0, kick(t, land, 3, 6));
          const big = bt.i === 1 ? 1 + 0.18 * Math.max(0, kick(t, land, 2, 4)) : 1;
          place(bt.m, p, lerp(0.3, 1, u) * big, 1);
        });
      }

      // keep text tiles under the bloom threshold in the creature sections
      if (t > 30.2) { post.bloomThreshold = 0.86; post.bloomStrength = 1.25; }
      // ---------------- HUD ----------------
      chap.update(t, def.start, def.end + 0.3);
      capA.update(t, 22.35, 25.9);
      capB.update(t, 26.05, 30.3);
      capC.update(t, 30.32, 33.1);
      capD.update(t, 33.12, 35.97);
      return post;
    },
  };
}
