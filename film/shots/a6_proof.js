// a6_proof (v10). Two instances of this file (shots/index.js):
//  a6_proof (global 92–100, no shift: authored in global time)
//    92.0–94.5  one real test (nl20 relay_word, episode 18): Quill holds "cipher"; one pass both ways; Ember writes
//               "cipher"; its chance of writing it with the real partner: 99.9% (main hit 93.0)
//    94.0–96.5  the real Quill slides out, a Quill from another test (grey, holding a different word, not shown) slides
//               in and links; one pass; Ember's "cipher" dims: 1 in 14 billion, under the 99.9% (main hit 95.5)
//    96.5–100   all 180 held-out tests (medians, shots/proof180_v2_data.js): real partner 99.7%, partner from another
//               test 1 in 120 billion; "real partner higher in 180 / 180" (main hit 98.5); two small pairs below
//  a6_title (global 117–121, shift 37 -> authored 80–84, internally "S5 time" = authored - 20 = 60–64): the title,
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
import { TESTS } from './proof180_v2_data.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const vlerp = (a, b, k) => a.clone().lerp(b, k);
const GOLD = new THREE.Color('#ffd166'), SILVER = new THREE.Color('#c3c8da');
const TEALC = '#34f5c0', TEAL_SOFT = '#a7f3d0', GREYC = '#c3c7d6';
const SH = '0 2px 22px rgba(0,0,0,0.85)';

// medians over the 180 tests (log10 of the chance of the whole right answer)
const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const MED_REAL = Math.pow(10, median(TESTS.map((r) => r[0])));   // 0.9974 -> "99.7%"
const MED_OTHER = Math.pow(10, median(TESTS.map((r) => r[1])));  // 8.3e-12 -> "1 in 120 billion"
const N_HIGHER = TESTS.filter((r) => r[0] > r[1]).length;        // 180
const TXT_REAL = (MED_REAL * 100).toFixed(1) + '%';
const TXT_OTHER = `1 in ${Math.round(1 / MED_OTHER / 1e10) * 10} billion`;

// proof instance (global seconds)
const P = {
  passOK: [92.05, 92.55], okWrite: 92.625, okRead: 93.0,
  linkCut: [93.875, 94.125], qOut: [94.0, 94.5], gIn: 94.375, gLink: 94.625,
  passBad: [94.75, 95.175], badWrite: 95.25, badRead: 95.5,
  wide: [96.5, 97.125], head: 96.625, medReal: 97.0, medOther: 97.5, count: 98.5,
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
  const q = new Creature('quill'), e = new Creature('ember'), g = new Creature('quill'), e2 = new Creature('ember');
  scene.add(q.group, e.group, g.group, e2.group);
  q.camera = e.camera = g.camera = e2.camera = camera;
  const link = new Link(q, e), linkG = new Link(g, e), link2 = new Link(q, e2);
  scene.add(link.group, linkG.group, link2.group);
  const pxScale = (1080 * scale) / (2 * Math.tan((35 / 2) * Math.PI / 180));
  const ring = makeRing({ pxScale, radius: 4.5 }); scene.add(ring.group);
  const title = makeTitle(hud);

  // tiles: inputs at the tails, the word Ember writes (nothing here rides the link)
  const secQ = tokenTile('cipher', 'secret', 0.66);
  const secG = tokenTile('another word', 'ghost', 0.66);
  const outOK = tokenTile('cipher', { ...TOKEN_STYLE.ember, color: '#ffe3a1', border: '#ffd166', radius: 50, glow: '#ffd166' }, 0.74);
  const outBad = tokenTile('cipher', { color: '#8d90a0', bg: 'rgba(30,31,40,0.85)', border: '#5d6070', radius: 50 }, 0.74);
  for (const m of [secQ, secG, outOK, outBad]) { scene.add(m); m.material.color.setScalar(0.74); m.visible = false; }
  outBad.material.color.setScalar(0.62);

  // light columns behind the two summary numbers (bright teal over the real pair, faint grey over the other)
  const beamMat = (col) => new THREE.ShaderMaterial({
    uniforms: { uCol: { value: col.clone() }, uAmp: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `varying vec2 vUv; uniform vec3 uCol; uniform float uAmp; void main(){ float x = abs(vUv.x-0.5)*2.0;
      float a = pow(1.0-x, 2.6) * smoothstep(0.0, 0.25, vUv.y) * (1.0 - smoothstep(0.55, 1.0, vUv.y)); gl_FragColor = vec4(uCol*a*uAmp, 1.0); }`,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const beams = [new THREE.Color('#14e3a8'), new THREE.Color('#8b8fa3')].map((c) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 15), beamMat(c)); scene.add(m); m.visible = false; return m; });

  // ---- HUD
  const abs = (css) => ({ position: 'absolute', left: '0px', top: '0px', whiteSpace: 'nowrap', pointerEvents: 'none', display: 'none', ...css });
  const headA = new Caption(hud, { head: 'Ember’s chance of writing <span style="color:#ffd166">cipher</span>', size: 72, pos: { left: 110, top: 64 }, width: 1700 });
  const numReal = el(hud, 'div', abs({ font: `700 132px ${FONT.display}`, letterSpacing: '-0.035em', color: TEALC, textShadow: `0 0 40px rgba(52,245,192,0.55), ${SH}` }), '99.9%');
  const numOther = el(hud, 'div', abs({ font: `700 88px ${FONT.display}`, letterSpacing: '-0.025em', color: GREYC, textShadow: SH }), '1 in 14 billion');
  const lab = (txt, col) => el(hud, 'div', abs({ font: `600 44px ${FONT.text}`, color: col, textShadow: SH }), txt);
  const labReal = lab('real partner', TEAL_SOFT), labOther = lab('partner from another test', GREYC);
  const pinReal = lab('real partner', TEAL_SOFT), pinOther = lab('partner from another test', GREYC);
  // the summary over all 180 tests
  const headB = new Caption(hud, { head: 'Median chance of the right answer, over 180 tests', size: 56, pos: { left: 0, top: 70 }, width: 1920, align: 'center' });
  const bigReal = el(hud, 'div', abs({ font: `700 196px ${FONT.display}`, letterSpacing: '-0.045em', color: TEALC, textShadow: `0 0 60px rgba(52,245,192,0.6), ${SH}` }), TXT_REAL);
  const bigOther = el(hud, 'div', abs({ font: `700 112px ${FONT.display}`, letterSpacing: '-0.03em', color: GREYC, textShadow: SH }), TXT_OTHER);
  const subReal = lab('real partner', TEAL_SOFT), subOther = lab('partner from another test', GREYC);
  const count = el(hud, 'div', abs({ font: `600 60px ${FONT.text}`, color: '#f5f3ff', textShadow: SH }),
    `real partner higher in <span style="font-weight:800;color:#fff;text-shadow:0 0 30px rgba(52,245,192,0.7)">${N_HIGHER} / ${TESTS.length}</span>`);

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
    const inj = [];
    if (amt > 0.01) XBLOCKS.forEach((blk) => { const u = (blk + 0.5) / 36, top = t <= b ? w : 1.01; if (u < top) inj.push({ block: blk, color, amount: amt, top: Math.min(1.01, top) }); });
    return { wave, inj };
  }

  // ---- cameras (proof)
  const PAN = 2.3;
  // A: starts on a5c_talk's last framing, eases right to make room for the readouts
  const camA = (t) => { const k = prog(t, 92, 96.5, (x) => x), p = PAN * prog(t, 92.1, 93.1, smooth); return { pos: V(lerp(0.7, 0.2, k) + p, lerp(3.4, 3.3, k), lerp(15.6, 14.9, k)), tgt: V(lerp(0.25, 0.1, k) + p, 3.0, 0) }; };
  // W: wide, two small pairs low in the frame under the two summary numbers
  const camW = (t) => { const k = prog(t, 97, 100, (x) => x); return { pos: V(0.4 * Math.sin((t - 97) * 0.4), lerp(9.6, 9.4, k), lerp(33.5, 32.5, k)), tgt: V(0, 7.3, 0) }; };
  const ROW = [V(2.05, 1.0, 0.55), V(2.05, -0.55, 0.55)];
  const PAIR = 8.2; // summary pairs centred at x = -PAIR (real) and +PAIR (other)
  // the summary columns sit over the pairs as seen by the settled wide camera (fixed screen x, so numbers never slide)
  const COLX = (() => { const c = camera.clone(), w = camW(98.5); c.position.copy(w.pos); c.lookAt(w.tgt); c.aspect = 16 / 9; c.updateProjectionMatrix(); c.updateMatrixWorld(true);
    return [-PAIR, PAIR].map((x) => toScreen(V(x, 4.0, 0), c).x); })();

  function updateProof(t) {
    world.update(t);
    const kw = easeInOutCubic(prog(t, P.wide[0], P.wide[1], (x) => x));
    { const a = camA(t), b = camW(t); camera.position.copy(vlerp(a.pos, b.pos, kw)); camera.lookAt(vlerp(a.tgt, b.tgt, kw)); }
    const sh = kick(t, P.okRead, 7, 7) * 0.025 + kick(t, P.count, 6, 7) * 0.03;
    camera.position.x += sh; camera.updateMatrixWorld(true);
    const wideOn = t >= P.wide[0];

    alive(q, t, 0.3); alive(e, t, 1.7); alive(g, t, 4.1, 0.7); alive(e2, t, 2.9);
    // ---- Quill (real): glad at 93.0, slides out left, comes back for the summary (left pair)
    const qo = easeInCubic(prog(t, P.qOut[0], P.qOut[1], (x) => x));
    const qBack = easeOutCubic(prog(t, P.wide[0] + 0.125, P.wide[1] + 0.125, (x) => x));
    q.pose.position.x = t < P.wide[0] ? -2.3 - 9 * qo : lerp(-20, -PAIR - 2.3, qBack);
    q.pose.lean = t < P.wide[0] ? -0.28 * Math.sin(Math.PI * clamp(qo * 1.2)) : 0.2 * (1 - qBack);
    q.pose.happy = Math.max(win(t, P.okRead - 0.05, P.qOut[0] + 0.3, 0.12), wideOn ? 0.8 : 0);
    q.pose.squash -= 0.07 * kick(t, P.okRead, 3, 6);
    q.pose.lookAt = (wideOn ? e2 : e).headWorld();
    q.group.visible = t < P.qOut[1] || (wideOn && qBack > 0.001);
    // ---- the second Ember (left pair, summary only)
    e2.group.visible = wideOn && qBack > 0.001;
    e2.pose.position.x = lerp(-15.4, -PAIR + 2.3, qBack); e2.pose.happy = 0.8; e2.pose.lookAt = q.headWorld();
    // ---- the partner from another test: a grey Quill; Ember stays with it and both move to the right pair
    const mv = easeInOutCubic(prog(t, P.wide[0], P.wide[1], (x) => x)) * PAIR;
    g.group.visible = t > P.gIn - 0.05;
    { const s = spring(t, P.gIn, 1.5, 0.72), s2 = spring(t - 0.04, P.gIn, 1.5, 0.72);
      g.pose.position.x = -11.3 + 9 * s + mv; g.pose.lean = clamp((s - s2) * 6, -0.3, 0.3); }
    g.pose.ghost = 1; g.pose.portGlow = 0.9; g.pose.glow = 0.8; g.pose.lookAt = camera.position.clone().add(V(-4, 0, 0));
    const pBad = pass(t, P.passBad, SILVER, P.wide[1]); g.pose.wave = pBad.wave;
    // ---- Ember
    const ep = e.pose; ep.position.x = 2.3 + mv;
    ep.lookAt = t < P.gIn + 0.15 ? q.headWorld() : g.headWorld();
    if (t > P.okWrite - 0.1 && t < P.okWrite + 0.3) ep.lookAt = outOK.position.clone();
    if (t > P.badWrite - 0.1 && t < P.wide[0]) ep.lookAt = outBad.position.clone();
    ep.happy = win(t, P.okRead - 0.05, P.qOut[0] + 0.1, 0.12);
    ep.squash -= 0.06 * kick(t, P.okRead, 3, 6) + 0.05 * kick(t, P.badWrite, 3, 6);
    ep.headNod -= 0.12 * win(t, P.okWrite - 0.1, P.okWrite + 0.35, 0.12) + 0.08 * win(t, P.badWrite - 0.1, P.badWrite + 0.35, 0.12);
    const r = prog(t, P.badRead, P.badRead + 0.35, easeOutBack);
    ep.lean += 0.06 * r; ep.headTilt += 0.24 * r; ep.blink = Math.max(ep.blink, clamp(1 - Math.abs((t - P.badRead - 0.12) / 0.07)));
    const pOK = pass(t, P.passOK, GOLD, P.qOut[0] + 0.1);
    ep.wave = pOK.wave > -0.5 ? pOK.wave : pBad.wave; q.pose.wave = pOK.wave;
    e.inject = pOK.inj.length ? pOK.inj : pBad.inj; ep.coreGlow = e.inject.length ? 0.55 : 1;
    // the summary pairs: a steady pass every beat so both stay alive
    if (wideOn) { const wv = ((t - 96) * 1.0) % 1 * 1.1 - 0.05; q.pose.wave = e2.pose.wave = wv; }
    // links
    link.group.visible = t < P.linkCut[1] + 0.05;
    link.state.grow = 1 - prog(t, P.linkCut[0], P.linkCut[1]);
    link.state.intensity = 1 + 0.35 * win(t, 92, P.okWrite, 0.2); link.state.aToB = 1; link.state.bToA = 0.8; link.state.packets = 4; link.state.flow = 1.2;
    link.state.tintAB = t < P.okRead + 0.2 ? GOLD : null;
    linkG.group.visible = g.group.visible;
    { const gg = []; for (let k = 0; k < 8; k++) gg.push(prog(t, P.gLink + k * 0.035, P.gLink + 0.25 + k * 0.035)); linkG.state.grow = gg; }
    linkG.state.intensity = 1; linkG.state.bToA = 0.8; linkG.state.colorBySender = 0; linkG.state.packets = 4; linkG.state.flow = 1.2; linkG.state.tintAB = SILVER;
    link2.group.visible = e2.group.visible;
    { const gg = []; for (let k = 0; k < 8; k++) gg.push(prog(t, P.wide[1] + k * 0.03, P.wide[1] + 0.2 + k * 0.03)); link2.state.grow = gg; }
    link2.state.intensity = 1.15; link2.state.aToB = 1; link2.state.bToA = 1; link2.state.packets = 4; link2.state.flow = 1.3; link2.state.tintAB = null;
    for (const c of [q, e, g, e2]) if (c.group.visible) c.update(t);
    if (link.group.visible) link.update(t);
    if (linkG.group.visible) linkG.update(t);
    if (link2.group.visible) link2.update(t);

    // ---- tiles (billboards)
    const camQ = camera.quaternion, camRight = V(1, 0, 0).applyQuaternion(camQ);
    const card = (m, base, off, a) => { m.visible = a > 0.002; if (!m.visible) return; m.position.copy(base).add(off); m.position.y += 0.05 * Math.sin(t * 2 + off.x); m.quaternion.copy(camQ); m.material.opacity = a; };
    card(secQ, q.pose.position, V(1.55, 0.5, 0.95), t < P.wide[0] && q.group.visible ? win(t, 91.9, P.qOut[0] + 0.12, 0.12) : 0);
    card(secG, g.pose.position, V(1.5, 0.5, 0.95), g.group.visible ? prog(t, P.gIn + 0.3, P.gIn + 0.5) * (1 - prog(t, P.wide[0], P.wide[0] + 0.25)) : 0);
    const writeTile = (m, t0, row) => {
      const dst = V(2.3, 3.75, 0).add(row), anchor = toScreen(dst.clone().add(camRight.clone().multiplyScalar(m.userData.width / 2)), camera);
      const a = t >= t0 ? win(t, t0, P.wide[0] + 0.25, 0.18) : 0;
      m.visible = a > 0.002; if (!m.visible) return t >= t0 ? anchor : null;
      const k = easeOutBack(prog(t, t0, t0 + 0.38, (x) => x), 1.3);
      m.position.copy(vlerp(e.mouthWorld(), dst, k)); m.position.y += 0.04 * Math.sin(t * 2.4 + row.y);
      m.quaternion.copy(camQ); m.scale.setScalar(Math.max(0.01, spring(t, t0, 2.4, 0.45))); m.material.opacity = a;
      return anchor;
    };
    const rowA = writeTile(outOK, P.okWrite, ROW[0]), rowB = writeTile(outBad, P.badWrite, ROW[1]);

    // ---- light columns over the summary pairs
    beams.forEach((m, i) => {
      const a = prog(t, i ? P.medOther - 0.25 : P.medReal - 0.25, i ? P.medOther + 0.25 : P.medReal + 0.25);
      m.visible = a > 0.002; if (!m.visible) return;
      m.position.set(i ? PAIR : -PAIR, 8.0, -1.5); m.quaternion.copy(camQ);
      m.material.uniforms.uAmp.value = a * (i ? 0.22 : 0.75) * (1 + 0.6 * (t > P.count ? Math.exp(-(t - P.count) * 3) : 0) * (i ? 0 : 1));
    });

    // ---- HUD: the example
    const out = 1 - prog(t, P.wide[0], P.wide[0] + 0.25);
    headA.update(t, 92.125, P.wide[0] + 0.25, { outDur: 0.25 });
    const readout = (node, t0, row) => {
      if (!row || t < t0) { place(node, 0, 0, { a: 0 }); return; }
      const pop = spring(t, t0, 2.6, 0.55), a0 = clamp((t - t0) / 0.1);
      place(node, row.x + 34, row.y, { a: a0 * out, s: 0.82 + 0.18 * pop, blur: (1 - a0) * 10 + (1 - out) * 10 });
    };
    readout(numReal, P.okRead, rowA); readout(numOther, P.badRead, rowB);
    const pin = (node, t0, t1, c) => {
      const a = c.group.visible && t < P.wide[0] ? easeOutCubic(clamp((t - t0) / 0.25)) * (1 - prog(t, t1, t1 + 0.2)) : 0;
      const h = toScreen(c.headWorld(), camera); place(node, h.x, h.y - 165, { a, ax: -0.5, blur: (1 - clamp((t - t0) / 0.2)) * 8 });
    };
    pin(pinReal, 92.25, P.okRead, q); pin(pinOther, P.gIn + 0.4, P.badRead, g);
    const under = (node, t0, row, dy) => {
      if (!row || t < t0) { place(node, 0, 0, { a: 0 }); return; }
      const a = easeOutCubic(clamp((t - t0) / 0.25)) * out;
      place(node, row.x + 40, row.y + dy + (1 - easeOutCubic(clamp((t - t0) / 0.3))) * 16, { a, blur: (1 - clamp((t - t0) / 0.2)) * 8 });
    };
    under(labReal, P.okRead + 0.125, rowA, 84); under(labOther, P.badRead + 0.125, rowB, 68);

    // ---- HUD: the summary over all 180 tests
    headB.update(t, P.head, 101, {});
    const colX = (i) => COLX[i];
    const big = (node, sub, t0, i) => {
      const a = clamp((t - t0 + 0.04) / 0.1), s = spring(t, t0, 2.4, 0.5);
      place(node, colX(i), 395, { a, ax: -0.5, s: 0.86 + 0.14 * s, blur: (1 - a) * 12 });
      const as = easeOutCubic(clamp((t - t0 - 0.125) / 0.3));
      place(sub, colX(i), 540 + (1 - as) * 14, { a: as, ax: -0.5 });
    };
    big(bigReal, subReal, P.medReal, 0); big(bigOther, subOther, P.medOther, 1);
    { const a = clamp((t - P.count + 0.04) / 0.1), s = spring(t, P.count, 2.4, 0.5); place(count, 960, 985, { a, ax: -0.5, s: 0.9 + 0.1 * s, blur: (1 - a) * 12 }); }

    // hidden in this instance
    ring.update(t, { center: V(0, 0.04, 0), radius: 5.2, yaw: 0, amp: 0, unlit: 0, size: 0.2 });
    title.update(t, { top: 664, tTitle: 1e9, tTag: [1e9], tCredit: 1e9 });
    hud.style.opacity = '1';
    const hitE = [P.okRead, P.badRead, P.medReal, P.count].reduce((s, h, k) => s + (t > h ? [0.6, 0.3, 0.5, 0.8][k] * Math.exp(-(t - h) * 3) : 0), 0);
    return {
      bloomStrength: 0.85 + 0.35 * hitE, bloomRadius: 0.6, bloomThreshold: 0.16,
      exposure: 0.95 + 0.06 * hitE, vignette: 0.95, aberration: 0.0013 + 0.002 * hitE,
      flash: 0.1 * (t > P.count ? Math.exp(-(t - P.count) * 9) : 0) + 0.06 * (t > P.okRead ? Math.exp(-(t - P.okRead) * 9) : 0), flashColor: '#c9fff0', fade: 0,
    };
  }

  // ---------------------------------------------------------------- title instance (S5 time)
  function camC(t) { const k = prog(t, 59, 64, (x) => x); return { pos: V(Math.sin((t - 59) * 0.12) * 1.2, lerp(2.5, 2.8, k), lerp(23.0, 25.0, k)), tgt: V(0, lerp(0.75, 0.85, k), 0) }; }
  function updateTitle(tg) {
    const t = tg - SHIFT;
    world.update(tg);
    for (const m of [secQ, secG, outOK, outBad, ...beams]) m.visible = false;
    g.group.visible = e2.group.visible = linkG.group.visible = link2.group.visible = false;
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
    for (const n of [numReal, numOther, labReal, labOther, pinReal, pinOther, bigReal, bigOther, subReal, subOther, count]) place(n, 0, 0, { a: 0 });
    headA.update(t, -9, -8); headB.update(t, -9, -8);
    const fade = prog(t, K.fade0, K.fade1, smooth);
    hud.style.opacity = String(1 - fade);
    const titleE = t > K.titleText ? Math.exp(-(t - K.titleText) * 1.5) : 0;
    return { bloomStrength: 0.9 + 0.55 * titleE - 0.1 * prog(t, K.title0 + 0.3, K.title0 + 0.7), bloomRadius: 0.6, bloomThreshold: 0.14, exposure: 1.0, vignette: 0.95, aberration: 0.0018, flash: 0, fade };
  }

  return { scene, camera, update: (t) => (TITLE_ONLY ? updateTitle(t) : updateProof(t)) };
}
