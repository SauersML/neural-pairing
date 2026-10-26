// The anatomical-plate post pass (used by shots/ab_plate.js through main.js; every other shot leaves it disabled).
//
// It composites three images of the same 3D scene, all rendered by the shot from the same camera:
//   tDiffuse  the film's normal render (after bloom, tone mapping, vignette, grain)
//   tInk      the engraving: r = sepia ink density (hatching that follows the form), g = red ink (the residual stream),
//             b = blue-green ink (the link), a = which body (hand-colour wash: 0.25 Quill, 0.5 Ember, 1 other)
//   tND       view-space normals (rgb) and linear depth (a): the contour lines are found from it (depth and crease edges)
// over a generated sheet of aged paper (fibres, foxing, plate tone, the pressed plate mark).
//
// The morph between the film and the plate is per pixel: a front with a ragged, fibrous edge spreads from a point
// (the link), faster along the bodies and the strands (ink wicks along them). Behind the front the glow drains into
// wet ink (a negative of the same frame, tinted by its own colours), then dries into the engraving. On the way back the
// paper soaks through with the light behind it and the film returns. With mix 0 everywhere the pass is a copy, and
// main.js disables it entirely unless a shot returns post.plate.
import * as THREE from 'three';

const VS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`;

const FS = `
uniform sampler2D tDiffuse, tInk, tND, tPaper, tNoise, tGrain;
uniform vec2 uRes; uniform float uScale;
uniform vec2 uCenter; uniform float uR, uBand, uMode, uLane, uAll;
uniform float uPress, uMark, uTime, uContour, uWash, uDim;
uniform vec2 uMis; uniform vec4 uClear, uClearR, uClearR2; uniform float uClearRK, uClearRK2;
varying vec2 vUv;

const vec3 INK = vec3(0.155, 0.098, 0.066);       // iron-gall / sepia
const vec3 RED = vec3(0.74, 0.27, 0.22);           // muted vermilion (the residual stream)
const vec3 TEAL = vec3(0.21, 0.50, 0.47);          // muted blue-green (the link)
const vec3 WASH_Q = vec3(0.80, 0.76, 0.93);        // pale violet wash (Quill)
const vec3 WASH_E = vec3(0.99, 0.84, 0.66);        // pale ochre wash (Ember)

float lum(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

// contour lines from depth and normal discontinuities (one-sided for silhouettes: the nearer side carries the line)
float contour(vec2 uv){
  vec2 px = 1.0/uRes;
  // a slow hand tremor in the line (under a pixel)
  vec2 wob = (texture2D(tNoise, uv*2.3).gb - 0.5) * px * 1.1 * uScale;
  uv += wob;
  vec4 c = texture2D(tND, uv);
  float dc = c.a > 0.0 ? c.a : 1e4; vec3 nc = c.rgb*2.0 - 1.0;
  float r = max(1.0, 1.15*uScale);
  float e = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i)*0.7853982;
    vec2 o = vec2(cos(a), sin(a)) * r * px;
    vec4 s = texture2D(tND, uv + o);
    float ds = s.a > 0.0 ? s.a : 1e4;
    // silhouette: the neighbour is much farther (or background)
    float jump = (ds - dc) / max(dc, 1e-3);
    e = max(e, smoothstep(0.035, 0.09, jump) * step(c.a, 1e3) * step(0.0001, c.a));
    // crease: same surface depth, normals turn sharply
    if (s.a > 0.0 && c.a > 0.0 && abs(ds - dc) / dc < 0.08) {
      float bend = 1.0 - dot(nc, s.rgb*2.0 - 1.0);
      e = max(e, 0.85*smoothstep(0.22, 0.5, bend));
    }
  }
  return e;
}

vec3 plateColor(vec2 uv, vec4 ink, vec4 paperT, out float inside, out float Dout){
  vec3 paper = paperT.rgb;
  vec2 p = uv*vec2(1920.0, 1080.0);
  p.y = 1080.0 - p.y;                                   // top-left origin, like the HUD
  float inset = 44.0;
  vec2 q = vec2(min(p.x - inset, 1920.0 - inset - p.x), min(p.y - inset, 1080.0 - inset - p.y));
  float dm = min(q.x, q.y);
  inside = smoothstep(-0.8, 0.8, dm);
  // the plate mark: the area the copper plate pressed is a touch darker and smoother (plate tone), the pressed
  // edge is a thin shadow on the top and left walls and a thin highlight on the bottom and right walls
  float press = uPress;
  paper = mix(paper, paper*vec3(0.965, 0.95, 0.925), inside*press*uMark);
  float wall = smoothstep(4.0, 0.0, abs(dm - 1.5)) * press * uMark;
  bool tl = (q.y < q.x) ? (p.y < 540.0) : (p.x < 960.0);
  paper *= 1.0 - wall*(tl ? 0.16 : -0.07);
  paper = mix(paper, paper*0.86, smoothstep(1.4, 0.0, abs(dm + 0.6)) * press * uMark * 0.6);
  // inks (the colour plates are printed slightly out of register)
  vec2 mis = uMis/vec2(1920.0, 1080.0);
  float red = texture2D(tInk, uv + mis).g;
  float teal = texture2D(tInk, uv - mis*0.8).b;
  // the wash: soft, a little outside the lines, offset like a hand-applied colour
  vec2 wo = vec2(2.5, -1.8)/vec2(1920.0, 1080.0);
  float wq = 0.0, we = 0.0;
  for (int i = 0; i < 5; i++) {
    float a = float(i)*1.2566 + 0.4;
    vec2 o = wo + vec2(cos(a), sin(a))*3.0/vec2(1920.0, 1080.0)*(i == 0 ? 0.0 : 1.0);
    float id = texture2D(tInk, uv + o).a;
    wq += step(0.15, id)*step(id, 0.35); we += step(0.4, id)*step(id, 0.6);
  }
  wq /= 5.0; we /= 5.0;
  float grain = paperT.a;                                // fine tooth of the paper
  // an inset (the roundel) is printed on bare paper: the drawing under it is not printed
  float clr = uClear.w*smoothstep(uClear.z + 1.5, uClear.z - 1.5, length(p - uClear.xy));
  { vec2 lo = uClearR.xy, hi = uClearR.zw; vec2 dq = max(max(lo - p, p - hi), 0.0);
    clr = max(clr, uClearRK*smoothstep(190.0, 10.0, length(dq))); }      // a broad, even vignette
  { vec2 dq = max(max(uClearR2.xy - p, p - uClearR2.zw), 0.0);
    clr = max(clr, uClearRK2*smoothstep(60.0, 4.0, length(dq))); }
  clr = max(clr, 1.0 - inside);
  wq *= 1.0 - clr; we *= 1.0 - clr;
  float D = max(ink.r, contour(uv)*uContour)*(1.0 - clr);
  red *= 1.0 - clr; teal *= 1.0 - clr;
  D = clamp(D*(0.86 + 0.28*grain) - 0.04*(1.0 - grain), 0.0, 1.0);   // ink sits unevenly on the tooth
  vec3 c = paper;
  c *= mix(vec3(1.0), WASH_Q, wq*uWash*(0.75 + 0.5*grain));
  c *= mix(vec3(1.0), WASH_E, we*uWash*(0.75 + 0.5*grain));
  c *= mix(vec3(1.0), RED, clamp(red, 0.0, 1.0)*(0.86 + 0.2*grain));
  c *= mix(vec3(1.0), TEAL, clamp(teal, 0.0, 1.0)*(0.86 + 0.2*grain));
  c = mix(c, INK, D*inside);
  Dout = D*inside;
  return c;
}

void main(){
  vec2 uv = vUv;
  vec3 live = texture2D(tDiffuse, uv).rgb;
  vec4 ink = texture2D(tInk, uv);
  vec4 nz = texture2D(tNoise, uv);
  vec4 paperT = vec4(texture2D(tPaper, uv).rgb, texture2D(tGrain, uv).r);
  // ---- the front
  float lane = max(step(0.01, ink.a), max(clamp(ink.g*2.0, 0.0, 1.0), clamp(ink.b*2.0, 0.0, 1.0)));
  vec2 dd = (uv - uCenter)*vec2(1.7778, 1.0);
  float d = length(dd);
  float de = d*mix(1.0, uLane, lane) + (nz.r - 0.5)*0.14 + (texture2D(tNoise, uv*2.9 + 0.31).g - 0.5)*0.05
             + (texture2D(tNoise, uv*7.3 + 0.17).b - 0.5)*0.014 + (paperT.a - 0.5)*0.008;
  if (uMode > 0.5) de = d*mix(1.0, uLane, lane) + (nz.r - 0.5)*0.08;
  float m = clamp((uR - de)/uBand, 0.0, 1.0);
  if (uMode > 0.5) m = 1.0 - m;
  m = clamp(max(m, uAll), 0.0, 1.0) * step(0.0001, uAll + uR + uMode);
  // the return's second stage: the glowing paper dims into the film, softly, from the link outward
  float sd = clamp((uDim - d*0.22)/0.55, 0.0, 1.0);
  if (uMode < 0.5 && m <= 0.0) { gl_FragColor = vec4(live, 1.0); return; }
  if (uMode > 0.5 && sd >= 1.0) { gl_FragColor = vec4(live, 1.0); return; }
  float inside, Dp;
  vec3 plate = plateColor(uv, ink, paperT, inside, Dp);
  vec3 paper = paperT.rgb;
  float L = lum(live);
  vec3 hue = live/max(1e-3, max(live.r, max(live.g, live.b)));
  if (uMode < 0.5) {
    // background: the glow drains into wet ink (a negative of this very frame, in its own hues) behind a tide line
    float dens = smoothstep(0.12, 0.7, L);
    vec3 inkTint = mix(INK, INK*0.55 + hue*0.32, 0.55);
    vec3 neg = mix(paper, inkTint, dens*0.92);
    float k1 = smoothstep(0.04, 0.24, m), k2 = smoothstep(0.4, 1.0, m);
    vec3 cb = mix(live, neg, k1);
    cb = mix(cb, plate, k2);
    float tide = exp(-pow((m - 0.27)/0.07, 2.0)) * (0.6 + 0.8*nz.g);
    cb = mix(cb, INK*0.85 + paper*0.15, clamp(tide*0.5, 0.0, 0.75));
    // the bodies and the strands: the engraver's lines are printed over the glow first, then the glow gives way
    // to paper under them (clean wet lines, no blotches)
    vec3 co = mix(live, INK, Dp*smoothstep(0.0, 0.32, m)*0.94);
    co = mix(co, plate, smoothstep(0.28, 0.9, m));
    gl_FragColor = vec4(mix(cb, co, lane), 1.0);
  } else {
    // light soaks through the paper: it glows warm as if lit from behind (the drawing stays on it); then the
    // glow gives way to the film's world, the drawing's lines turning to light in the film's own colours
    float s1 = 1.0 - m;
    vec3 warm = vec3(1.0, 0.87, 0.64);
    vec3 lit = mix(plate, plate*0.8 + warm*0.4 + live*0.12, smoothstep(0.0, 1.0, s1));
    // the drawing's lines start to glow before the paper dims, so it is the drawing that becomes the film's light
    vec3 glowL = (warm*0.45 + live*1.1)*Dp;
    vec3 litG = mix(lit, lit + glowL*0.5, smoothstep(0.0, 0.35, sd));
    vec3 night = live + glowL*0.8*(1.0 - smoothstep(0.45, 1.0, sd));
    vec3 c = mix(litG, night, smoothstep(0.12, 0.8, sd));
    c = mix(c, live, smoothstep(0.6, 1.0, sd));
    float edge = exp(-pow((s1 - 0.5)/0.2, 2.0))*(1.0 - sd);
    c += warm*edge*0.14;
    gl_FragColor = vec4(c, 1.0);
  }
}`;

// ---- generated textures (built once per page)
function paperCanvas(W = 1920, H = 1080) {
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const x = cv.getContext('2d');
  let s = 1234567; const R = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  x.fillStyle = '#ece0c3'; x.fillRect(0, 0, W, H);
  // broad mottling (uneven sizing and age)
  for (let i = 0; i < 260; i++) {
    const cx = R() * W, cy = R() * H, r = 60 + R() * 320, dark = R() < 0.55;
    const g = x.createRadialGradient(cx, cy, 0, cx, cy, r);
    const a = 0.018 + R() * 0.035;
    g.addColorStop(0, dark ? `rgba(150,110,60,${a})` : `rgba(255,250,235,${a * 1.2})`); g.addColorStop(1, 'rgba(0,0,0,0)');
    x.fillStyle = g; x.fillRect(cx - r, cy - r, 2 * r, 2 * r);
  }
  // age toward the edges
  { const g = x.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.62);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(140,96,40,0.16)'); x.fillStyle = g; x.fillRect(0, 0, W, H); }
  for (const [x0, y0, w, h] of [[0, 0, W, 26], [0, H - 26, W, 26], [0, 0, 26, H], [W - 26, 0, 26, H]]) {
    const g = w > h ? x.createLinearGradient(0, y0 === 0 ? 0 : H, 0, y0 === 0 ? 26 : H - 26) : x.createLinearGradient(x0 === 0 ? 0 : W, 0, x0 === 0 ? 26 : W - 26, 0);
    g.addColorStop(0, 'rgba(130,88,40,0.14)'); g.addColorStop(1, 'rgba(130,88,40,0)'); x.fillStyle = g; x.fillRect(x0, y0, w, h);
  }
  // laid lines (faint chain lines of hand-made paper)
  x.globalAlpha = 0.022; x.strokeStyle = '#7a5a30'; x.lineWidth = 1.4;
  for (let cx = 37; cx < W; cx += 92 + R() * 6) { x.beginPath(); x.moveTo(cx, 0); for (let y = 0; y <= H; y += 60) x.lineTo(cx + (R() - 0.5) * 1.5, y); x.stroke(); }
  x.globalAlpha = 1;
  // fibres
  for (let i = 0; i < 9000; i++) {
    const cx = R() * W, cy = R() * H, len = 5 + R() * R() * 34, a = R() * Math.PI * 2, bend = (R() - 0.5) * len * 0.8;
    const dark = R() < 0.6;
    x.strokeStyle = dark ? `rgba(110,80,40,${0.04 + R() * 0.07})` : `rgba(255,252,240,${0.08 + R() * 0.12})`;
    x.lineWidth = 0.5 + R() * 0.8;
    const ex = cx + Math.cos(a) * len, ey = cy + Math.sin(a) * len;
    x.beginPath(); x.moveTo(cx, cy); x.quadraticCurveTo((cx + ex) / 2 - Math.sin(a) * bend, (cy + ey) / 2 + Math.cos(a) * bend, ex, ey); x.stroke();
  }
  // foxing: rust spots, more of them near the margins, each a cluster with a soft halo and a darker core
  for (let i = 0; i < 70; i++) {
    let cx = R() * W, cy = R() * H;
    if (R() < 0.6) { const side = Math.floor(R() * 4); if (side === 0) cy = R() * 140; else if (side === 1) cy = H - R() * 140; else if (side === 2) cx = R() * 170; else cx = W - R() * 170; }
    const big = R() < 0.25, n = big ? 6 : 2 + Math.floor(R() * 3);
    for (let k = 0; k < n; k++) {
      const px = cx + (R() - 0.5) * (big ? 40 : 14), py = cy + (R() - 0.5) * (big ? 40 : 14), r = (big ? 6 : 2) + R() * (big ? 18 : 7);
      const g = x.createRadialGradient(px, py, 0, px, py, r);
      const a = 0.05 + R() * (big ? 0.12 : 0.16);
      g.addColorStop(0, `rgba(150,85,30,${a * 1.4})`); g.addColorStop(0.45, `rgba(165,105,45,${a})`); g.addColorStop(1, 'rgba(170,120,60,0)');
      x.fillStyle = g; x.beginPath(); x.arc(px, py, r, 0, Math.PI * 2); x.fill();
    }
    if (R() < 0.5) { x.fillStyle = `rgba(105,60,25,${0.25 + R() * 0.3})`; x.beginPath(); x.arc(cx, cy, 0.6 + R() * 1.2, 0, Math.PI * 2); x.fill(); }
  }
  // fine tooth: a separate grey sheet (the ink breaks on it); a little low-pass so it reads as paper, not as noise
  const img = x.getImageData(0, 0, W, H), d = img.data;
  const gc = document.createElement('canvas'); gc.width = W; gc.height = H; const gx = gc.getContext('2d');
  const gimg = gx.createImageData(W, H), gd = gimg.data;
  let hs = 99; const hr = () => { hs = (hs * 1103515245 + 12345) & 0x7fffffff; return hs / 0x7fffffff; };
  const g1 = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) g1[i] = hr();
  for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
    const i = y * W + xx;
    const v = (y > 0 && y < H - 1 && xx > 0 && xx < W - 1) ? (g1[i] * 4 + g1[i - 1] + g1[i + 1] + g1[i - W] + g1[i + W]) / 8 : 0.5;
    const j = i * 4, n = (v - 0.5) * 9;
    d[j] = Math.max(0, Math.min(255, d[j] + n)); d[j + 1] = Math.max(0, Math.min(255, d[j + 1] + n)); d[j + 2] = Math.max(0, Math.min(255, d[j + 2] + n * 0.9)); d[j + 3] = 255;
    const gv = Math.max(0, Math.min(255, 128 + (v - 0.5) * 420)); gd[j] = gd[j + 1] = gd[j + 2] = gv; gd[j + 3] = 255;
  }
  x.putImageData(img, 0, 0); gx.putImageData(gimg, 0, 0);
  cv.grain = gc;
  return cv;
}

function noiseCanvas(W = 1024, H = 576) {
  // r: fbm-ish blotches (the front's ragged edge); g, b: finer variation
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H; const x = cv.getContext('2d');
  let s = 77; const R = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const layer = (n, rmin, rmax, blur) => {
    const c = document.createElement('canvas'); c.width = W; c.height = H; const y = c.getContext('2d');
    y.fillStyle = '#808080'; y.fillRect(0, 0, W, H);
    for (let i = 0; i < n; i++) { const v = Math.floor(R() * 255); y.fillStyle = `rgb(${v},${v},${v})`; y.beginPath(); y.arc(R() * W, R() * H, rmin + R() * (rmax - rmin), 0, Math.PI * 2); y.fill(); }
    const o = document.createElement('canvas'); o.width = W; o.height = H; const z = o.getContext('2d'); z.filter = `blur(${blur}px)`; z.drawImage(c, 0, 0);
    return z.getImageData(0, 0, W, H).data;
  };
  const a = layer(1400, 10, 60, 16), b = layer(5000, 3, 14, 4), c2 = layer(6000, 1.5, 6, 1.6);
  const img = x.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const j = i * 4;
    img.data[j] = Math.min(255, Math.max(0, (a[j] - 128) * 1.6 + (b[j] - 128) * 0.5 + 128));
    img.data[j + 1] = b[j]; img.data[j + 2] = c2[j]; img.data[j + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  return cv;
}

export function createPlatePass(W, H, scale) {
  const pc = paperCanvas(); const paper = new THREE.CanvasTexture(pc); const grain = new THREE.CanvasTexture(pc.grain); grain.colorSpace = THREE.NoColorSpace; grain.generateMipmaps = false; grain.minFilter = THREE.LinearFilter; paper.colorSpace = THREE.NoColorSpace; paper.minFilter = THREE.LinearFilter; paper.generateMipmaps = false;
  const noise = new THREE.CanvasTexture(noiseCanvas()); noise.wrapS = noise.wrapT = THREE.RepeatWrapping; noise.colorSpace = THREE.NoColorSpace;
  const black = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1); black.needsUpdate = true;
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: null }, tInk: { value: black }, tND: { value: black }, tPaper: { value: paper }, tGrain: { value: grain }, tNoise: { value: noise },
      uRes: { value: new THREE.Vector2(W * scale, H * scale) }, uScale: { value: scale },
      uCenter: { value: new THREE.Vector2(0.5, 0.5) }, uR: { value: 0 }, uBand: { value: 0.12 }, uMode: { value: 0 }, uLane: { value: 0.6 }, uAll: { value: 0 },
      uPress: { value: 0 }, uMark: { value: 1 }, uTime: { value: 0 }, uContour: { value: 1 }, uWash: { value: 0.32 }, uDim: { value: 0 },
      uMis: { value: new THREE.Vector2(1.6, -1.1) }, uClear: { value: new THREE.Vector4(0, 0, 0, 0) }, uClearR: { value: new THREE.Vector4(0, 0, 0, 0) }, uClearRK: { value: 0 }, uClearR2: { value: new THREE.Vector4(0, 0, 0, 0) }, uClearRK2: { value: 0 },
    },
    vertexShader: VS, fragmentShader: FS,
  });
  // a ShaderPass-compatible object built on the composer's own full-screen quad pattern
  return { mat, paper };
}

// apply the plate fields a shot returned (post.plate) to the pass's uniforms
export function applyPlate(mat, plate, t) {
  const u = mat.uniforms;
  u.tInk.value = plate.ink; u.tND.value = plate.nd;
  u.uCenter.value.set(plate.center?.[0] ?? 0.5, plate.center?.[1] ?? 0.5);
  u.uR.value = plate.radius ?? 0; u.uBand.value = plate.band ?? 0.12; u.uMode.value = plate.mode === 'out' ? 1 : 0;
  u.uLane.value = plate.lane ?? 0.6; u.uAll.value = plate.all ?? 0; u.uPress.value = plate.press ?? 0; u.uMark.value = plate.mark ?? 1;
  u.uContour.value = plate.contour ?? 1; u.uDim.value = plate.dim ?? 0; u.uWash.value = plate.wash ?? 0.32; u.uTime.value = t;
  const c = plate.clear; if (c) u.uClear.value.set(c[0], c[1], c[2], c[3] ?? 1); else u.uClear.value.set(0, 0, 0, 0);
  const cr = plate.clearRect; if (cr) { u.uClearR.value.set(cr[0], cr[1], cr[2], cr[3]); u.uClearRK.value = cr[4] ?? 1; } else u.uClearRK.value = 0;
  const c2 = plate.clearRect2; if (c2) { u.uClearR2.value.set(c2[0], c2[1], c2[2], c2[3]); u.uClearRK2.value = c2[4] ?? 1; } else u.uClearRK2.value = 0;
}
