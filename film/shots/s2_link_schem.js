// S2 part 2: one read, dissected. Ember (reader) reads Quill (sender) at the exchange after block 24.
//   h_E += tanh(α) · size · O( attention( Q(norm h_E), [null; K(norm h_Q)], [0; s · V(norm h_Q)] ) )
// The reader is Ember's hidden state at ONE token position ("is", tick 3). It reads Quill's hidden states at the token
// positions of tick <= 3; the write is added into that position's residual stream, which continues to block 25.
// Ember's other positions do the same read at the same time (faint streams behind).
// Orange = the reader Ember's weights (Q, O, α, null key). Violet = the sender Quill's weights (K, V, gate s).
// Teal = attention (as in the repo's method figure). Everything is a pure function of global t.
import * as THREE from 'three';
import { C } from '../lib/palette.js';
import { Creature } from '../lib/creature.js';
import { makeWorld } from '../lib/env.js';
import { Pin, el, FONT } from '../lib/hud.js';
import { clamp, lerp, prog, kick, win, beatPulse, easeOutBack, easeOutCubic, easeInOutCubic, hash } from '../lib/util.js';

const HEX = { quill: '#8b5cf6', quillHi: '#c4b5fd', ember: '#f97316', emberHi: '#fdba74', link: '#14e3a8', linkHi: '#a7f3d0' };
const CSSC = { quill: '#b9a2ff', quillHi: '#ddd6fe', ember: '#fb923c', emberHi: '#fed7aa', link: '#34f5c0', text: '#f5f3ff' };

// ---- layout (world units; the schematic lies in the XY plane, exploded a little in z)
const DXC = 1.35;
const COLS = [-8.1, -6.75, -5.4, -4.05, -2.7, -1.35];          // Quill's token positions
const TOK = ['…', 'lantern', '▸', 'Sky', 'looks', 'blue'];
const TICKS = ['−1', '−1', '0', '1', '2', '3'];
const GHOST_X = 0.5;                                           // "because", tick 4: the future, hidden
const NULL_X = 1.6;                                            // the null key column
const Q_X = 3.25;                                              // reader's query projection
const IS_X = 6.2;                                              // Ember's residual stream at the token "is"
const BEHIND = [{ tok: 'word', dx: 0.8, z: -1.3 }, { tok: 'The', dx: 1.6, z: -2.6 }, { tok: '▸', dx: 2.4, z: -3.9 }];
const Y_TICK = -3.0, Y_TOK = -2.38, Y_STATE = -1.55, Y_VAL = 0.0, Y_KEY = 1.3, Y_BAR = 1.74, BAR_SCALE = 4.0;
const BX = 2.75, BY = 2.9;                                     // blend orb (attention-weighted sum of values)
const O_X = 3.95, SHRINK_X = 5.05;
const STREAM_TOP = 4.0, STREAM_BOT = -2.05;
// illustrative values (as in the method figure; not measured)
const GATE = [0.84, 0.7, 0.9, 0.6, 0.55, 0.88];
const CLOSE = 2;                                               // the gate at "▸" closes
const ATT = { nul: 0.14, c: [0.05, 0.36, 0.1, 0.07, 0.08, 0.2] }; // softmax over [null; 6 visible keys], sums to 1

// ---- timeline (global seconds)
export const TS = {
  start: 14.5, orbs: 14.52, kv: 15.0, gates: 16.0, close: 16.45, nul: 16.85, q: 17.4, beam: 17.8, bars: 18.1,
  pour: 19.0, toO: 19.95, shrink: 20.3, plus: 20.75, end: 21.2,
};

// ring gauge shader: track + filled arc (clockwise from 12 o'clock)
const gaugeMat = (col) => new THREE.ShaderMaterial({
  uniforms: { uFrac: { value: 0 }, uCol: { value: new THREE.Color(col) }, uAmp: { value: 1 }, uTrack: { value: 0.3 } },
  vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader: `uniform float uFrac; uniform vec3 uCol; uniform float uAmp; uniform float uTrack; varying vec2 vUv;
    void main(){ vec2 p = vUv*2.0-1.0; float r = length(p);
      float a = atan(p.x, p.y); a = a < 0.0 ? a + 6.28318 : a; float f = a / 6.28318;
      float band = smoothstep(0.72, 0.76, r) * smoothstep(0.98, 0.94, r);
      float thin = smoothstep(0.82, 0.84, r) * smoothstep(0.9, 0.88, r);
      float on = step(f, uFrac);
      float edge = smoothstep(0.025, 0.0, abs(f - uFrac)) * band;
      vec3 c = uCol * (band * on * 1.2 + thin * uTrack * (1.0 - on) + edge * 1.4);
      float alpha = clamp(band*on + thin*uTrack*(1.0-on) + edge, 0.0, 1.0);
      gl_FragColor = vec4(c*uAmp, alpha*uAmp); }`,
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
});

export function buildSchematic({ hud }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene, { floor: false, fog: false, dust: 500, skyAmp: 0.8 });
  scene.background = C.bg0.clone();
  const dot = Creature.dotTexture();
  const root = el(hud, 'div', { position: 'absolute', inset: '0', display: 'none' });

  // blueprint backdrop: fine grid plane behind the schematic
  const gridMat = new THREE.ShaderMaterial({
    uniforms: { uAmp: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
    fragmentShader: `uniform float uAmp; varying vec3 vW;
      float line(float x, float w){ float f = abs(fract(x)-0.5); return smoothstep(w, 0.0, 0.5-f); }
      void main(){ vec2 p = vW.xy - vec2(-0.5, 0.6); float fade = smoothstep(13.0, 2.0, length(p*vec2(0.75,1.1)));
        float g = max(line(vW.x*2.0, 0.03), line(vW.y*2.0, 0.03))*0.35 + max(line(vW.x*0.5, 0.012), line(vW.y*0.5, 0.012))*0.6;
        gl_FragColor = vec4(vec3(0.35,0.3,0.75)*g*0.2*uAmp*fade, 1.0); }`,
  });
  const grid = new THREE.Mesh(new THREE.PlaneGeometry(48, 30), gridMat); grid.position.set(0, 0.6, -6.5); scene.add(grid);

  const G = new THREE.Group(); scene.add(G);
  const glow = (col, s = 1) => { const m = new THREE.SpriteMaterial({ map: dot, color: new THREE.Color(col), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }); const sp = new THREE.Sprite(m); sp.scale.setScalar(s); G.add(sp); return sp; };
  const stdMat = (col, emi = 0.6) => new THREE.MeshStandardMaterial({ color: new THREE.Color(col).multiplyScalar(0.55), emissive: new THREE.Color(col), emissiveIntensity: emi, roughness: 0.3, metalness: 0.25, transparent: true });
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const edgeGeo = new THREE.EdgesGeometry(boxGeo);
  const sphereGeo = new THREE.SphereGeometry(1, 40, 24);
  const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
  const thin = (col, r = 0.014) => { const m = new THREE.Mesh(cylGeo, new THREE.MeshBasicMaterial({ color: new THREE.Color(col), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); m.userData.r = r; G.add(m); return m; };
  const UP = new THREE.Vector3(0, 1, 0);
  // place a thin cylinder from a toward b, grown by k (0..1)
  const span = (m, a, b, k) => { const d = new THREE.Vector3().subVectors(b, a); const L = d.length() * Math.max(1e-4, k); m.position.copy(a).addScaledVector(d.normalize(), L / 2); m.quaternion.setFromUnitVectors(UP, d); m.scale.set(m.userData.r, L, m.userData.r); m.visible = k > 0.002; };
  const V3 = (x, y, z = 0) => new THREE.Vector3(x, y, z);

  function makeCube(col, size = 0.56, edgeCol = '#ffffff') {
    const g = new THREE.Group();
    const mat = stdMat(col, 0.55);
    g.add(new THREE.Mesh(boxGeo, mat));
    const em = new THREE.LineBasicMaterial({ color: new THREE.Color(edgeCol), transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false });
    g.add(new THREE.LineSegments(edgeGeo, em));
    g.scale.setScalar(size); G.add(g);
    return { g, mat, em, size };
  }
  const setCube = (c, s, op = 1, emi = 0.55) => { c.g.scale.setScalar(Math.max(1e-3, c.size * s)); c.g.visible = s > 0.003 && op > 0.003; c.mat.opacity = op; c.mat.emissiveIntensity = emi; c.em.opacity = 0.55 * op; };

  // ---- Quill's hidden states (one per token position), their values (inside the gate dial) and keys
  const cols = COLS.map((x, j) => {
    const orbMat = stdMat(HEX.quill, 0.9);
    const orb = new THREE.Mesh(sphereGeo, orbMat); orb.position.set(x, Y_STATE, 0); G.add(orb);
    const halo = glow(HEX.quill, 1.4); halo.position.copy(orb.position);
    const key = makeCube(HEX.quill, 0.62, HEX.quillHi); key.g.position.set(x, Y_KEY, 0);
    const val = makeCube(HEX.quillHi, 0.56, '#ffffff'); val.g.position.set(x, Y_VAL, 0);
    const gauge = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 1.3), gaugeMat(HEX.quillHi)); gauge.position.set(x, Y_VAL, 0.0); G.add(gauge);
    const lineV = thin(HEX.quillHi, 0.012);                      // state -> value (through the gate)
    const lineK = [thin(HEX.quillHi, 0.012), thin(HEX.quillHi, 0.012), thin(HEX.quillHi, 0.012)]; // state -> key, around the gate
    const bar = new THREE.Mesh(boxGeo, new THREE.MeshStandardMaterial({ color: new THREE.Color(HEX.link).multiplyScalar(0.5), emissive: new THREE.Color(HEX.link), emissiveIntensity: 0.8, roughness: 0.35, metalness: 0.1, transparent: true }));
    G.add(bar);
    const barCap = glow(HEX.linkHi, 0.5);
    const flash = glow(HEX.emberHi, 1.0); flash.position.set(x, Y_KEY, 0.45);
    return { x, j, orb, orbMat, halo, key, val, gauge, lineV, lineK, bar, barCap, flash };
  });
  // the future position (tick 4): hidden
  const ghostMat = new THREE.MeshStandardMaterial({ color: 0x2a2440, emissive: new THREE.Color('#6b6390'), emissiveIntensity: 0.25, transparent: true, opacity: 0.3, roughness: 0.5 });
  const ghostOrb = new THREE.Mesh(sphereGeo, ghostMat); ghostOrb.position.set(GHOST_X, Y_STATE, 0); G.add(ghostOrb);
  const ghostEdgeMat = new THREE.LineDashedMaterial({ color: new THREE.Color('#8b85a8'), dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.35 });
  const ghostK = new THREE.LineSegments(edgeGeo, ghostEdgeMat); ghostK.computeLineDistances(); ghostK.scale.setScalar(0.62); ghostK.position.set(GHOST_X, Y_KEY, 0); G.add(ghostK);
  const ghostV = new THREE.LineSegments(edgeGeo, ghostEdgeMat); ghostV.computeLineDistances(); ghostV.scale.setScalar(0.56); ghostV.position.set(GHOST_X, Y_VAL, 0); G.add(ghostV);

  // null key (the reader's learned key, orange) with a zero value
  const nullKey = makeCube(HEX.ember, 0.62, HEX.emberHi); nullKey.g.position.set(NULL_X, Y_KEY, 0);
  const nullVal = makeCube(HEX.ember, 0.56, HEX.ember); nullVal.g.position.set(NULL_X, Y_VAL, 0);
  const nullBar = new THREE.Mesh(boxGeo, cols[0].bar.material.clone()); G.add(nullBar);
  const nullCap = glow(HEX.linkHi, 0.5);
  const nullFlash = glow(HEX.emberHi, 1.0); nullFlash.position.set(NULL_X, Y_KEY, 0.45);

  // ---- Ember: residual streams per token position; "is" (tick 3) is dissected, the others read at the same time
  const mkStream = (x, z, amp) => {
    const mat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uCol: { value: new THREE.Color(HEX.emberHi) }, uAmp: { value: amp }, uHot: { value: 0 }, uHotY: { value: 0 } },
      vertexShader: `varying float vY; void main(){ vY=(modelMatrix*vec4(position,1.)).y; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
      fragmentShader: `uniform float uTime; uniform vec3 uCol; uniform float uAmp; uniform float uHot; uniform float uHotY; varying float vY;
        void main(){ float p = 0.55 + 0.45*sin(vY*7.0 - uTime*6.0); float hot = uHot*smoothstep(uHotY-0.1, uHotY+0.3, vY)*step(${BY.toFixed(2)} - 0.05, vY);
          gl_FragColor = vec4(uCol*uAmp*(0.8+0.7*p) + vec3(1.0,0.75,0.45)*hot, 1.0);} `,
    });
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, STREAM_TOP - STREAM_BOT, 16, 1, true), mat);
    m.position.set(x, (STREAM_TOP + STREAM_BOT) / 2, z); G.add(m);
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.36, 18), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.emberHi), transparent: true }));
    head.position.set(x, STREAM_TOP + 0.15, z); G.add(head);
    const orb = new THREE.Mesh(sphereGeo, stdMat(HEX.ember, 0.8)); orb.position.set(x, Y_STATE, z); G.add(orb);
    const node = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.03, 10, 40), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.emberHi), transparent: true })); node.position.set(x, BY, z); G.add(node);
    return { mat, m, head, orb, node, x, z };
  };
  const isS = mkStream(IS_X, 0, 0.12);
  const halo = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, STREAM_TOP - STREAM_BOT, 16, 1, true), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.ember), transparent: true, opacity: 0.05, blending: THREE.AdditiveBlending, depthWrite: false }));
  halo.position.copy(isS.m.position); G.add(halo);
  const eHalo = glow(HEX.ember, 1.8); eHalo.position.copy(isS.orb.position);
  const behind = BEHIND.map((b) => ({ ...b, s: mkStream(IS_X + b.dx, b.z, 0.08), rip: (() => { const r = new THREE.Mesh(new THREE.TorusGeometry(1, 0.025, 8, 48), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.emberHi), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); r.rotation.x = Math.PI / 2; G.add(r); return r; })() }));

  const qCube = makeCube(HEX.ember, 0.72, HEX.emberHi); qCube.g.position.set(Q_X, Y_KEY, 0);
  const qLine = thin(HEX.emberHi, 0.02);
  const oCube = makeCube(HEX.ember, 0.8, HEX.emberHi); oCube.g.position.set(O_X, BY, 0);
  // × tanh(α)·size: an orange iris the write passes through
  const iris = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.035, 10, 64), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.emberHi), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  iris.position.set(SHRINK_X, BY, 0); iris.rotation.y = 1.05; G.add(iris);
  const ripples = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.TorusGeometry(1, 0.03, 8, 64), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.emberHi), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); m.rotation.x = Math.PI / 2; m.position.set(IS_X, BY, 0); G.add(m); return m; });
  const plusFlash = glow('#ffd2a0', 2); plusFlash.position.set(IS_X, BY, 0.3);
  const writeLine = thin(HEX.emberHi, 0.02);

  // the query beam: orange, along the key row, from Q leftwards
  const beam = thin(HEX.ember, 0.03);
  const beamCore = thin('#fff1e0', 0.011);
  const beamHead = glow('#ffc58a', 0.9);

  // the write orb (blend -> O -> shrink -> plus)
  const wMat = new THREE.MeshStandardMaterial({ color: new THREE.Color('#2a1a4a'), emissive: new THREE.Color(HEX.quillHi), emissiveIntensity: 0.55, roughness: 0.2, metalness: 0.1, transparent: true });
  const wOrb = new THREE.Mesh(sphereGeo, wMat); G.add(wOrb);
  const wHalo = glow(HEX.quillHi, 1.5);
  const wRing = new THREE.Mesh(new THREE.TorusGeometry(1, 0.02, 8, 64), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.link), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); G.add(wRing);

  // pour paths: value j -> blend orb, arcing toward the camera (in front of keys and bars)
  const pourCurves = COLS.map((x) => new THREE.CubicBezierCurve3(V3(x, Y_VAL + 0.1, 0.35), V3(x + 0.4, Y_VAL + 1.3, 2.8), V3(BX - 2.6, BY + 0.7, 2.0), V3(BX - 0.3, BY, 0.15)));
  const pourTubes = pourCurves.map((cv) => { const m = new THREE.Mesh(new THREE.TubeGeometry(cv, 48, 0.012, 6, false), new THREE.MeshBasicMaterial({ color: new THREE.Color(HEX.quillHi), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })); G.add(m); return m; });
  const NPP = 22, NP = NPP * COLS.length;
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
  pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
  const pMat = new THREE.PointsMaterial({ size: 0.75, map: dot, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  G.add(new THREE.Points(pg, pMat));

  // speed lines: we have just flown into the exchange after block 24
  const NSL = 90; const slPos = new Float32Array(NSL * 6), slCol = new Float32Array(NSL * 6);
  const slGeo = new THREE.BufferGeometry(); slGeo.setAttribute('position', new THREE.BufferAttribute(slPos, 3)); slGeo.setAttribute('color', new THREE.BufferAttribute(slCol, 3));
  const speed = new THREE.LineSegments(slGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(speed);
  function speedLines(t) {
    const lt = t - TS.start, on = 1 - prog(t, TS.start + 0.12, TS.start + 0.55);
    speed.visible = on > 0.003; if (!speed.visible) return;
    const cT = new THREE.Color(HEX.link), cV = new THREE.Color(HEX.quillHi), c = new THREE.Color();
    for (let i = 0; i < NSL; i++) {
      const a = hash(i * 1.7) * Math.PI * 2, r = 0.6 + hash(i * 3.1) * 4.5;
      const zh = -30 + ((lt * 70 + hash(i * 7.3) * 36) % 36), L = 2.5 + hash(i * 2.2) * 3;
      const x = -4.7 + Math.cos(a) * r, y = -1.0 + Math.sin(a) * r * 0.7;
      slPos.set([x, y, zh, x, y, zh - L], i * 6);
      c.copy(i % 3 ? cT : cV).multiplyScalar(on * (0.5 + hash(i * 9.9)));
      slCol.set([c.r, c.g, c.b, 0, 0, 0], i * 6);
    }
    slGeo.attributes.position.needsUpdate = true; slGeo.attributes.color.needsUpdate = true;
  }

  // ================= HUD =================
  // the equation (2 lines, 46 px) across the top
  const eqWrap = el(root, 'div', { position: 'absolute', left: '0', right: '0', top: '118px', display: 'flex', justifyContent: 'center' });
  const eq = el(eqWrap, 'div', { font: `500 46px ${FONT.display}`, whiteSpace: 'nowrap', textAlign: 'center', lineHeight: '1.22', padding: '10px 34px 14px', background: 'rgba(9,7,20,0.66)', border: '1px solid rgba(245,243,255,0.13)', borderRadius: '18px', boxShadow: '0 10px 40px rgba(0,0,0,0.45)' });
  const sub = (s) => `<sub style="font-size:0.62em;vertical-align:-0.28em">${s}</sub>`;
  const P = 'p';
  // [html, color, time it lights, active-until]
  // plain-language line (no single-letter symbols): [html, color, time it lights, active windows]
  const TERMS = [
    ["Ember's hidden state", CSSC.ember, TS.plus, [[TS.plus, TS.end]]], ['  +=  ', P, TS.plus, []],
    ['scale', CSSC.ember, TS.shrink, [[TS.shrink, TS.plus]]], [' × ', P, TS.shrink, []],
    ['output(', CSSC.ember, TS.toO, [[TS.toO, TS.shrink]]], ['BR'],
    ['attention(', CSSC.link, TS.nul, [[TS.nul, TS.q], [TS.bars, TS.toO]]], [' ', P, TS.kv, []],
    ['query', CSSC.ember, TS.q, [[TS.q, TS.bars]]], [', ', P, TS.kv, []],
    ['keys', CSSC.quill, TS.kv, [[TS.kv, TS.gates]]], [', ', P, TS.kv, []],
    ['gated values', CSSC.quill, TS.kv, [[TS.kv, TS.nul]]],
    [' )', CSSC.link, TS.nul, [[TS.nul, TS.q], [TS.bars, TS.toO]]], [' )', CSSC.ember, TS.toO, [[TS.toO, TS.shrink]]],
  ];
  const termEls = [];
  for (const [html, col, on, wins] of TERMS) {
    if (html === 'BR') { el(eq, 'br'); continue; }
    termEls.push({ s: el(eq, 'span', { display: 'inline-block', whiteSpace: 'pre', transformOrigin: '50% 70%', borderRadius: '8px', padding: col === P ? '0' : '0 4px', marginLeft: html.startsWith(',') ? '-4px' : '0' }, html), col, on, wins });
  }

  // a soft dark band behind the caption and the key: text never sits on bright geometry
  const band = el(root, 'div', { position: 'absolute', left: '0', right: '0', bottom: '0', height: '330px', background: 'linear-gradient(to top, rgba(4,3,10,0.92) 0%, rgba(4,3,10,0.82) 55%, rgba(4,3,10,0) 100%)' });
  // ownership key (bottom right), 30 px
  const legend = el(root, 'div', { position: 'absolute', right: '110px', bottom: '100px', font: `600 30px ${FONT.text}`, color: 'rgba(245,243,255,0.8)', lineHeight: '1.4', textAlign: 'left', textShadow: '0 1px 12px rgba(0,0,0,0.9)' });
  const sq = (c) => `<span style="display:inline-block;width:18px;height:18px;border-radius:4px;background:${c};margin-right:14px;vertical-align:-1px;box-shadow:0 0 12px ${c}"></span>`;
  const legS = el(legend, 'div', {}, `${sq(HEX.quill)}<span style="color:${CSSC.quill}">Quill's weights</span>: key, value, gate`);
  const legR = el(legend, 'div', {}, `${sq(HEX.ember)}<span style="color:${CSSC.ember}">Ember's weights</span>: query, output, scale`);

  // pins (all >= 30 px)
  const mk = (html, o = {}) => {
    const pin = new Pin(root, html, { size: 32, dot: false, ...o });
    const up = pin.update.bind(pin);
    // labels never enter the caption band at the bottom of the frame
    pin.update = (w, cam, op = 1, dx = 0, dy = 0) => { const v = w.clone().project(cam); const y = (-v.y * 0.5 + 0.5) * 1080; up(w, cam, op * clamp((805 - y) / 25), dx, dy); };
    return pin;
  };
  const chipQ = (txt, op = 1) => `<span style="display:inline-block;padding:2px 14px 5px;border-radius:7px;background:rgba(76,29,149,${0.85 * op});border:2px solid rgba(167,139,250,${op});color:rgba(237,233,254,${op})">${txt}</span>`;
  const chipE = (txt, op = 1) => `<span style="display:inline-block;padding:2px 18px 5px;border-radius:999px;background:rgba(124,45,18,${0.85 * op});border:2px solid rgba(251,146,60,${op});color:rgba(255,237,213,${op})">${txt}</span>`;
  const dimW = 'rgba(245,243,255,0.6)';
  const pins = {
    rowH: mk(`Quill's hidden states`, { color: CSSC.quill, anchor: 'right', size: 32 }),
    rowV: mk('gated value', { color: CSSC.quill, anchor: 'right', size: 32 }),
    rowK: mk('key', { color: CSSC.quill, anchor: 'right', size: 32 }),
    rowTick: mk('tick', { color: dimW, anchor: 'right', size: 30, font: FONT.mono }),
    toks: TOK.map((tk) => mk(chipQ(tk), { anchor: 'center', size: 30, weight: 500 })),
    ticks: TICKS.map((tk) => mk(tk, { color: dimW, anchor: 'center', size: 32, font: FONT.mono })),
    ghostTok: mk(chipQ('because', 0.35), { anchor: 'center', size: 30, weight: 500 }),
    ghostTick: mk('4', { color: 'rgba(245,243,255,0.4)', anchor: 'center', size: 32, font: FONT.mono }),
    ghost: mk('future: hidden', { color: 'rgba(245,243,255,0.75)', anchor: 'center', size: 34, weight: 600, bg: 'rgba(8,6,18,0.75)' }),
    gate: mk('gate', { color: CSSC.quillHi, anchor: 'right', size: 32 }),
    closed: mk('gate closed', { color: CSSC.quillHi, anchor: 'center', size: 36, weight: 700, bg: 'rgba(8,6,18,0.8)' }),
    nullOn: mk('nothing', { color: '#fff7ed', anchor: 'center', size: 30, weight: 700 }),
    nullL: mk('attend to nothing', { color: CSSC.ember, anchor: 'center', size: 32, bg: 'rgba(8,6,18,0.75)' }),
    Q: mk('query', { color: CSSC.ember, anchor: 'center', size: 32, weight: 700 }),
    O: mk('output', { color: '#fff7ed', anchor: 'center', size: 32, weight: 700 }),
    plus: mk('+', { color: '#fff7ed', anchor: 'center', size: 44, weight: 700, font: FONT.display }),
    hE: mk(`<div style="text-align:right">Ember's hidden state<br><span style="color:rgba(245,243,255,0.65)">block 24</span></div>`, { color: CSSC.ember, anchor: 'right', size: 30 }),
    isTok: mk(chipE('is'), { anchor: 'center', size: 30, weight: 500 }),
    isTick: mk('3', { color: dimW, anchor: 'center', size: 32, font: FONT.mono }),
    b25: mk('block 25', { color: dimW, anchor: 'left', size: 30 }),
    shrink: mk('× scale', { color: CSSC.ember, anchor: 'center', size: 34, weight: 700, bg: 'rgba(8,6,18,0.7)' }),
    pct: mk('≈ 5–10% of the residual', { color: CSSC.emberHi, anchor: 'center', size: 30, bg: 'rgba(8,6,18,0.7)' }),
  };

  // ---- camera path: keyframes (each segment eased) plus a slow drift so it never sits still
  const KF = [
    { t: 14.5, p: [-5.0, -1.3, 6.0], q: [-4.7, -1.1, 0], f: 48 },
    { t: 15.3, p: [-4.4, -0.4, 17.1], q: [-4.2, -0.75, 0], f: 36 },
    { t: 16.3, p: [-3.0, -0.4, 17.4], q: [-3.1, -0.75, 0], f: 36 },
    { t: 17.5, p: [-0.4, 0.8, 23.4], q: [-0.9, 0.4, 0], f: 36 },
    { t: 18.7, p: [-0.4, 0.9, 23.6], q: [-0.9, 0.4, 0], f: 36 },
    { t: 19.7, p: [0.6, 0.95, 23.4], q: [-0.4, 0.4, 0], f: 36 },
    { t: 20.55, p: [4.7, 3.0, 11.4], q: [3.9, 2.55, 0], f: 36 },
    { t: 21.2, p: [0.2, 0.95, 23.6], q: [-0.6, 0.4, 0], f: 36 },
  ];
  const tmpP = new THREE.Vector3(), tmpQ = new THREE.Vector3();
  function camAt(t) {
    let i = 0; while (i < KF.length - 2 && t > KF[i + 1].t) i++;
    const a = KF[i], b = KF[i + 1];
    const k = easeInOutCubic(clamp((t - a.t) / (b.t - a.t)));
    tmpP.set(lerp(a.p[0], b.p[0], k), lerp(a.p[1], b.p[1], k), lerp(a.p[2], b.p[2], k));
    tmpQ.set(lerp(a.q[0], b.q[0], k), lerp(a.q[1], b.q[1], k), lerp(a.q[2], b.q[2], k));
    tmpP.x += Math.sin(t * 0.7) * 0.25; tmpP.y += Math.sin(t * 0.53 + 1) * 0.12;
    camera.fov = lerp(a.f, b.f, k);
    camera.position.copy(tmpP); camera.lookAt(tmpQ);
  }
  const B0 = Q_X - 0.4, B1 = COLS[0] - 0.5, BDUR = 0.8;
  const beamPass = (x) => TS.beam + BDUR * (B0 - x) / (B0 - B1);
  const beamHeadX = (t) => lerp(B0, B1, prog(t, TS.beam, TS.beam + BDUR, (x) => x));
  const barRise = (t, tb, h) => h * BAR_SCALE * prog(t, tb, tb + 0.6, (x) => easeOutBack(x, 1.6));

  function update(t) {
    world.update(t);
    camAt(t); speedLines(t);
    camera.aspect = 16 / 9; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
    const bp = beatPulse(t, 7);
    gridMat.uniforms.uAmp.value = prog(t, TS.start, TS.start + 0.6);

    // ---------- Quill's states (14.52+), their keys and values (15.0+), gates (16.0+), attention bars (18.1+)
    const closeK = prog(t, TS.close, TS.close + 0.45);
    const gateOn = prog(t, TS.gates, TS.gates + 0.5);
    const qd = (1 - 0.5 * prog(t, TS.toO, TS.toO + 0.45)) * (1 - 0.6 * win(t, 20.05, 21.0, 0.25)); // Quill's side recedes once its values are read
    for (const c of cols) {
      const j = c.j;
      const tPop = TS.orbs + j * 0.1;
      const sp = easeOutBack(clamp((t - tPop) / 0.4), 2.2);
      c.orb.scale.setScalar(0.3 * Math.max(1e-3, sp) * (1 + 0.04 * bp)); c.orb.visible = sp > 0.002;
      c.orbMat.opacity = qd;
      c.orbMat.emissiveIntensity = 0.8 + (t > tPop ? 1.8 * Math.exp(-6 * (t - tPop)) : 0);
      c.halo.scale.setScalar(1.3 * clamp(sp)); c.halo.material.opacity = 0.7 * clamp(sp) * qd;
      const tk = TS.kv + j * 0.09;
      const lk = prog(t, tk, tk + 0.35, easeOutCubic);
      span(c.lineV, V3(c.x, Y_STATE + 0.3, 0), V3(c.x, Y_VAL - 0.28, 0), lk);
      const bx = c.x - DXC / 2;
      const k1 = clamp(lk * 3), k2 = clamp(lk * 3 - 1), k3 = clamp(lk * 3 - 2);
      span(c.lineK[0], V3(c.x - 0.2, Y_STATE + 0.24, 0), V3(bx, Y_STATE + 0.7, 0), k1);
      span(c.lineK[1], V3(bx, Y_STATE + 0.7, 0), V3(bx, Y_KEY - 0.18, 0), k2);
      span(c.lineK[2], V3(bx, Y_KEY - 0.18, 0), V3(c.x - 0.31, Y_KEY - 0.18, 0), k3);
      const lineOp = 0.55 * lk * (1 - 0.5 * prog(t, TS.pour, TS.pour + 0.5)) * qd;
      for (const l of c.lineK) l.material.opacity = lineOp;
      const sj = j === CLOSE ? lerp(GATE[j], 0.02, closeK) : GATE[j];
      c.lineV.material.opacity = lineOp * (j === CLOSE ? lerp(1, 0.3, closeK) : 1);
      const g0 = TS.gates + j * 0.06;
      const gShow = prog(t, g0, g0 + 0.35);
      c.gauge.material.uniforms.uFrac.value = sj * prog(t, g0, g0 + 0.55, easeOutCubic);
      c.gauge.material.uniforms.uAmp.value = gShow * qd;
      c.gauge.scale.setScalar(lerp(0.6, 1, easeOutBack(gShow, 2))); c.gauge.visible = gShow > 0.002;
      const ck = easeOutBack(clamp((t - tk - 0.22) / 0.4), 2.4);
      const kFlash = t > tk + 0.22 ? Math.exp(-5 * (t - tk - 0.22)) : 0;
      const tp = beamPass(c.x);
      const hit = t > tp ? Math.exp(-3.5 * (t - tp)) : 0;
      const keyLit = t > tp ? (0.4 + ATT.c[j] * 4) * (1 - prog(t, TS.pour, TS.pour + 0.6)) : 0;
      setCube(c.key, ck, qd, 0.55 + 1.0 * kFlash + 0.8 * hit + 0.35 * keyLit);
      c.flash.material.opacity = 0.35 * hit; c.flash.scale.setScalar(0.9 + ATT.c[j] * 3);
      const vs = lerp(1, sj, gateOn); // values scale by their gate
      const pourDim = 1 - 0.55 * prog(t, TS.pour + j * 0.05, TS.pour + 0.7);
      setCube(c.val, ck * vs, qd, (0.28 + 0.9 * kFlash) * vs * pourDim + 0.05);
      const tb = TS.bars + 0.36 - j * 0.06;
      const bh = barRise(t, tb, ATT.c[j]);
      c.bar.visible = bh > 0.003;
      c.bar.scale.set(0.42, Math.max(1e-3, bh), 0.42); c.bar.position.set(c.x, Y_BAR + bh / 2, 0);
      c.bar.material.emissiveIntensity = 0.7 + (t > tb ? 0.6 * Math.exp(-4 * (t - tb)) : 0) + 0.12 * bp;
      c.bar.material.opacity = 0.95 * (1 - 0.3 * prog(t, TS.toO, TS.toO + 0.5));
      c.barCap.visible = c.bar.visible; c.barCap.position.set(c.x, Y_BAR + bh, 0.1); c.barCap.material.opacity = 0.6 * clamp(bh * 2);
    }
    const gk = prog(t, TS.orbs + 0.7, TS.orbs + 1.1);
    ghostOrb.scale.setScalar(0.3 * Math.max(1e-3, gk)); ghostMat.opacity = 0.32 * gk;
    ghostEdgeMat.opacity = 0.32 * prog(t, TS.kv + 0.6, TS.kv + 0.9);
    ghostK.visible = ghostV.visible = ghostEdgeMat.opacity > 0.003;

    // null key + zero value
    const nk = easeOutBack(clamp((t - TS.nul) / 0.45), 2.4);
    const nFlash = t > TS.nul ? Math.exp(-4 * (t - TS.nul)) : 0;
    const ntp = beamPass(NULL_X);
    const nHit = t > ntp ? Math.exp(-3.5 * (t - ntp)) : 0;
    setCube(nullKey, nk, 1, 0.6 + 1.4 * nFlash + 0.8 * nHit);
    const nv = easeOutBack(clamp((t - TS.nul - 0.15) / 0.45), 2.4);
    setCube(nullVal, nv, 0.1, 0.2); nullVal.em.opacity = 0.9 * clamp((t - TS.nul - 0.15) / 0.3);
    nullFlash.material.opacity = 0.35 * nHit;
    {
      const tb = TS.bars + 0.42; const bh = barRise(t, tb, ATT.nul);
      nullBar.visible = bh > 0.003; nullBar.scale.set(0.42, Math.max(1e-3, bh), 0.42); nullBar.position.set(NULL_X, Y_BAR + bh / 2, 0);
      nullBar.material.emissiveIntensity = 0.7 + (t > tb ? 0.6 * Math.exp(-4 * (t - tb)) : 0) + 0.12 * bp;
      nullBar.material.opacity = 0.95 * (1 - 0.3 * prog(t, TS.toO, TS.toO + 0.5));
      nullCap.visible = nullBar.visible; nullCap.position.set(NULL_X, Y_BAR + bh, 0.1); nullCap.material.opacity = 0.6 * clamp(bh * 2);
    }

    // ---------- Ember: the residual stream at "is", its state h_E, the query
    const eOn = prog(t, TS.q - 0.3, TS.q + 0.2);
    const eVis = prog(t, TS.q - 0.6, TS.q - 0.1);
    const hot = t > TS.plus ? Math.exp(-2.2 * (t - TS.plus - 0.12)) * clamp((t - TS.plus) / 0.12) : 0;
    for (const s of [isS, ...behind.map((b) => b.s)]) {
      const main = s === isS;
      s.m.visible = s.head.visible = s.orb.visible = eVis > 0.002 && (main || eOn > 0.5);
      const u = s.mat.uniforms; u.uTime.value = t;
      u.uAmp.value = main ? 0.12 + 0.5 * eOn : 0.05 + 0.12 * eOn;
      u.uHot.value = (main ? 1.3 : 0.45) * hot; u.uHotY.value = BY + (t - TS.plus) * 3.0;
      s.head.material.opacity = main ? 0.3 + 0.65 * eOn : 0.1 + 0.2 * eOn;
      s.orb.scale.setScalar((main ? 0.38 : 0.26) * (1 + 0.05 * bp + (main ? 0.25 * kick(t, TS.q, 3, 5) : 0)));
      s.orb.material.emissiveIntensity = main ? 0.25 + 0.8 * eOn : 0.15 + 0.25 * eOn; s.orb.material.opacity = main ? 1 : 0.6;
      const plusOn = prog(t, TS.toO, TS.toO + 0.5), pk = t > TS.plus ? Math.exp(-3 * (t - TS.plus)) : 0;
      s.node.visible = plusOn > 0.003; s.node.scale.setScalar(easeOutBack(plusOn, 2) * (1 + 0.35 * pk));
      s.node.material.opacity = plusOn * (main ? 1 : 0.4);
    }
    halo.material.opacity = (0.02 + 0.06 * eOn + 0.25 * hot) * eVis;
    eHalo.material.opacity = (0.12 + 0.45 * eOn) * eVis; eHalo.scale.setScalar(1.8 + 0.4 * eOn);
    behind.forEach((b) => { const rt = t - TS.plus - 0.05; const on = rt > 0 && rt < 0.8; b.rip.visible = on; if (on) { b.rip.position.set(b.s.x, BY, b.z); b.rip.scale.setScalar(0.25 + rt * 1.6); b.rip.material.opacity = (1 - rt / 0.8) * 0.45; } });
    const ql = prog(t, TS.q, TS.q + 0.3, easeOutCubic);
    span(qLine, V3(IS_X - 0.1, Y_KEY, 0), V3(Q_X + 0.37, Y_KEY, 0), ql);
    qLine.material.opacity = 0.9 * ql;
    const qk = easeOutBack(clamp((t - TS.q - 0.2) / 0.4), 2.4);
    const qFire = t > TS.beam ? Math.exp(-5 * (t - TS.beam)) : 0;
    setCube(qCube, qk * (1 + 0.15 * qFire), 1, 0.7 + 1.4 * qFire + 0.15 * bp);
    const bxh = beamHeadX(t);
    const beamOn = t > TS.beam ? 1 - prog(t, TS.pour - 0.1, TS.pour + 0.3) : 0;
    span(beam, V3(B0, Y_KEY, 0.45), V3(bxh, Y_KEY, 0.45), t > TS.beam ? 1 : 0);
    span(beamCore, V3(B0, Y_KEY, 0.45), V3(bxh, Y_KEY, 0.45), t > TS.beam ? 1 : 0);
    beam.material.opacity = 0.6 * beamOn; beamCore.material.opacity = 0.9 * beamOn;
    beam.visible = beamCore.visible = beam.visible && beamOn > 0.003;
    beamHead.position.set(bxh, Y_KEY, 0.48); beamHead.material.opacity = t > TS.beam && t < TS.beam + BDUR + 0.1 ? 0.9 : 0; beamHead.scale.setScalar(0.9 + 0.3 * Math.sin(t * 40));

    // ---------- pour: values (weighted by attention × gate) flow into the blend orb
    const pos = pg.attributes.position.array, col = pg.attributes.color.array;
    const w = COLS.map((_, j) => ATT.c[j] * (j === CLOSE ? lerp(GATE[j], 0.02, closeK) : GATE[j]));
    const wMax = Math.max(...w);
    const pc = new THREE.Color(), vc = new THREE.Color(HEX.quillHi), lc = new THREE.Color(HEX.linkHi);
    for (let j = 0; j < COLS.length; j++) {
      const ww = w[j] / wMax;
      pourTubes[j].material.opacity = 0.35 * ww * win(t, TS.pour, TS.toO + 0.1, 0.25);
      for (let k = 0; k < NPP; k++) {
        const idx = j * NPP + k;
        const te = TS.pour + (k / NPP) * 0.65 + hash(idx) * 0.04;
        const u = (t - te) / 0.55;
        const show = u > 0 && u < 1 && ww > 0.05;
        const p = pourCurves[j].getPoint(clamp(u));
        pos[idx * 3] = p.x; pos[idx * 3 + 1] = p.y; pos[idx * 3 + 2] = show ? p.z : 1e5;
        const b = (0.6 + 1.2 * ww) * Math.sin(Math.PI * clamp(u)) ** 0.4;
        pc.copy(vc).lerp(lc, clamp(u * 0.8)).multiplyScalar(b * 1.4);
        col[idx * 3] = pc.r; col[idx * 3 + 1] = pc.g; col[idx * 3 + 2] = pc.b;
      }
    }
    pg.attributes.position.needsUpdate = true; pg.attributes.color.needsUpdate = true;

    // ---------- the write: blend -> O -> × tanh(α)·size -> + into the residual stream at "is"
    const fill = prog(t, TS.pour + 0.2, TS.toO - 0.05);
    const wp = V3(BX, BY);
    let wr = 0.42 * Math.sqrt(fill);
    const wcol = new THREE.Color('#a78bfa');
    const mO = prog(t, TS.toO, TS.toO + 0.3, easeInOutCubic);
    const mS = prog(t, TS.shrink - 0.02, TS.shrink + 0.32, easeInOutCubic);
    const mP = prog(t, TS.plus - 0.12, TS.plus + 0.05, (x) => x * x);
    wp.x = lerp(BX, O_X, mO); if (mS > 0) wp.x = lerp(O_X, SHRINK_X, mS); if (mP > 0) wp.x = lerp(SHRINK_X, IS_X, mP);
    const throughO = prog(t, TS.toO + 0.2, TS.toO + 0.34);
    wcol.lerp(new THREE.Color(HEX.ember), throughO);
    wr *= lerp(1, 0.3, prog(t, TS.shrink + 0.1, TS.shrink + 0.4, easeOutCubic));
    const inO = win(t, TS.toO + 0.17, TS.toO + 0.37, 0.06);
    wr *= 1 - 0.6 * inO;
    const alive = t > TS.pour + 0.2 && t < TS.plus + 0.06;
    wOrb.visible = alive && wr > 0.003; wOrb.position.copy(wp); wOrb.scale.setScalar(Math.max(1e-3, wr) * (1 + 0.05 * bp));
    wMat.emissive.copy(wcol);
    wHalo.visible = wOrb.visible; wHalo.position.copy(wp); wHalo.scale.setScalar(wr * 4.2); wHalo.material.color.copy(wcol); wHalo.material.opacity = 0.45;
    wRing.visible = wOrb.visible && throughO < 0.5; wRing.position.copy(wp); wRing.scale.setScalar(wr * 1.35 + 0.05); wRing.rotation.set(0.4, t * 1.5, 0); wRing.material.opacity = 0.7 * (1 - throughO * 2);
    const oPop = easeOutBack(clamp((t - (TS.pour + 0.6)) / 0.45), 2.4);
    const oFire = inO + (t > TS.toO + 0.3 ? 0.6 * Math.exp(-5 * (t - TS.toO - 0.3)) : 0);
    setCube(oCube, oPop * (1 + 0.12 * oFire), 1, 0.6 + 1.6 * oFire + 0.1 * bp);
    const irisOn = prog(t, TS.toO + 0.1, TS.toO + 0.45);
    iris.visible = irisOn > 0.003;
    iris.scale.setScalar(lerp(0.6, 1, easeOutBack(irisOn, 2)) * lerp(1, 0.42, prog(t, TS.shrink + 0.1, TS.shrink + 0.4)));
    iris.material.opacity = irisOn * (0.75 + 0.6 * win(t, TS.shrink, TS.shrink + 0.5, 0.1)) * (1 - 0.5 * prog(t, TS.plus + 0.1, TS.end));
    span(writeLine, V3(O_X + 0.42, BY, 0), V3(IS_X - 0.32, BY, 0), prog(t, TS.toO + 0.3, TS.shrink + 0.1));
    writeLine.material.opacity = 0.35;
    const pk = t > TS.plus ? Math.exp(-3 * (t - TS.plus)) : 0;
    ripples.forEach((r, i) => { const rt = t - TS.plus - i * 0.12; const on = rt > 0 && rt < 0.8; r.visible = on; if (on) { r.scale.setScalar(0.3 + rt * 2.4); r.material.opacity = (1 - rt / 0.8) * 0.9; r.position.y = BY + rt * 0.5 * i; } });
    plusFlash.material.opacity = pk; plusFlash.scale.setScalar(1.0 + 3.0 * (1 - pk));

    // ---------- HUD
    root.style.display = 'block';
    const hudIn = prog(t, TS.start + 0.05, TS.start + 0.45);
    const fo = 1 - prog(t, TS.end - 0.12, TS.end);
    eqWrap.style.opacity = hudIn * fo;
    eqWrap.style.transform = `translateY(${(1 - hudIn) * -16}px)`;
    const finalAll = prog(t, TS.plus + 0.15, TS.end - 0.05);
    for (const T of termEls) {
      const lit = prog(t, T.on, T.on + 0.25);
      const active = T.wins.reduce((m, [a, b]) => Math.max(m, win(t, a, b + 0.1, 0.15)), 0);
      if (T.col === P) { T.s.style.color = `rgba(245,243,255,${(0.3 + 0.45 * lit).toFixed(3)})`; continue; }
      const c = new THREE.Color(T.col);
      const mixed = new THREE.Color('#57536b').lerp(c, clamp(lit * (0.72 + 0.28 * Math.max(active, finalAll))));
      T.s.style.color = '#' + mixed.getHexString();
      const g = Math.max(active, 0.5 * finalAll);
      T.s.style.textShadow = g > 0.01 ? `0 0 ${(18 * g).toFixed(1)}px ${T.col}` : 'none';
      T.s.style.transform = `translateY(${(-2 * active).toFixed(2)}px) scale(${(1 + 0.05 * active).toFixed(3)})`;
      const rgb = `${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)}`;
      T.s.style.background = active > 0.01 ? `rgba(${rgb},${(0.2 * active).toFixed(3)})` : 'transparent';
      T.s.style.boxShadow = active > 0.01 ? `0 0 0 1px rgba(${rgb},${(0.45 * active).toFixed(3)})` : 'none';
    }
    legend.style.opacity = hudIn * fo;
    band.style.opacity = hudIn * (1 - prog(t, 21.0, TS.end));
    legS.style.opacity = prog(t, TS.kv, TS.kv + 0.4); legR.style.opacity = prog(t, TS.q, TS.q + 0.4);

    const rowsX = COLS[0] - 0.62;
    const cu = 1 - win(t, 20.05, 21.0, 0.25); // labels near the bottom step aside during the close-up on the write
    pins.rowH.update(V3(rowsX, Y_STATE), camera, prog(t, TS.orbs, TS.orbs + 0.4) * fo * cu);
    pins.rowV.update(V3(rowsX - 0.12, Y_VAL), camera, prog(t, TS.kv, TS.kv + 0.4) * fo);
    pins.rowK.update(V3(rowsX, Y_KEY), camera, prog(t, TS.kv, TS.kv + 0.4) * fo);
    pins.rowTick.update(V3(rowsX, Y_TICK), camera, prog(t, TS.orbs + 0.2, TS.orbs + 0.6) * fo * cu);
    TOK.forEach((_, j) => {
      const o = prog(t, TS.orbs + j * 0.1, TS.orbs + j * 0.1 + 0.3) * fo;
      pins.toks[j].update(V3(COLS[j], Y_TOK), camera, o * lerp(1, 0.6, 1 - qd) * cu);
      pins.ticks[j].update(V3(COLS[j], Y_TICK), camera, o * lerp(1, 0.6, 1 - qd) * cu);
    });
    pins.ghostTok.update(V3(GHOST_X, Y_TOK), camera, gk * fo * cu);
    pins.ghostTick.update(V3(GHOST_X, Y_TICK), camera, gk * fo * cu);
    pins.ghost.update(V3(GHOST_X, Y_VAL + 0.62), camera, gk * fo * (1 - prog(t, TS.gates - 0.3, TS.gates)));
    pins.gate.update(V3(COLS[0] - 0.7, Y_VAL + 0.55), camera, win(t, TS.gates + 0.1, TS.close + 0.3, 0.2));
    pins.closed.update(V3(COLS[CLOSE], (Y_VAL + Y_KEY) / 2 + 0.02), camera, win(t, TS.close + 0.25, TS.bars - 0.1, 0.25));
    pins.nullOn.update(V3(NULL_X, Y_KEY), camera, prog(t, TS.nul + 0.1, TS.nul + 0.35) * fo);
    pins.nullL.update(V3(NULL_X, Y_VAL - 0.75), camera, win(t, TS.nul + 0.2, TS.q + 0.9, 0.25));
    pins.Q.update(V3(Q_X, Y_KEY - 0.62), camera, prog(t, TS.q + 0.3, TS.q + 0.55) * fo);
    pins.O.update(V3(O_X, BY), camera, prog(t, TS.pour + 0.75, TS.pour + 1.0) * fo * (1 - 0.6 * inO));
    pins.plus.update(V3(IS_X, BY), camera, prog(t, TS.toO + 0.1, TS.toO + 0.5) * fo);
    pins.hE.update(V3(IS_X - 0.75, Y_STATE - 0.62), camera, prog(t, TS.q - 0.3, TS.q + 0.1) * fo * cu);
    pins.isTok.update(V3(IS_X, Y_TOK), camera, prog(t, TS.q - 0.3, TS.q + 0.1) * fo * cu);
    pins.isTick.update(V3(IS_X, Y_TICK), camera, prog(t, TS.q - 0.3, TS.q + 0.1) * fo * cu);
    pins.b25.update(V3(IS_X + 0.4, STREAM_TOP + 0.1), camera, prog(t, TS.toO, TS.toO + 0.4) * fo);
    pins.shrink.update(V3(SHRINK_X - 0.1, BY + 0.8), camera, prog(t, TS.toO + 0.2, TS.toO + 0.5) * fo);
    pins.pct.update(V3(SHRINK_X - 0.1, BY - 0.8), camera, prog(t, TS.shrink + 0.15, TS.shrink + 0.4) * fo);
  }

  function hide() { root.style.display = 'none'; }
  return { scene, camera, update, hide };
}
