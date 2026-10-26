// Engraving materials and the plate's G-buffer renderer (shots/ab_plate.js).
//
// The same meshes as the film are drawn a second time with these materials into two render targets:
//   ND   view-space normal (rgb) and linear depth (a) of the solid shapes (bones, heads, eyes, crests): the post pass
//        finds the contour lines in it.
//   INK  r = sepia line density (engraved hatching that follows the form: line families laid along the surface's own
//        coordinates, widening with the shade, crossed by a second family in the darks), g = red ink (the residual
//        stream), b = blue-green ink (the link), a = which body (for the hand-colour wash).
// Line spacing is held in screen pixels (measured at 1920×1080), choosing an octave of the surface coordinate per
// pixel and blending to the next, so lines keep an engraver's spacing as the camera moves.
import * as THREE from 'three';

export const PU = {   // shared uniforms of every plate material
  uScale: { value: 1 }, uSpacing: { value: 5.2 }, tND: { value: null }, uRes: { value: new THREE.Vector2(1920, 1080) },
  uShadowDir: { value: new THREE.Vector2(0.6, -0.4) }, uFloorFade: { value: 1 },
};

const VS = `
varying vec3 vN; varying vec3 vV; varying vec2 vUv; varying vec3 vW; varying vec3 vO;
void main(){
  vUv = uv; vO = position;
  vec4 w = modelMatrix*vec4(position, 1.0); vW = w.xyz;
  vec4 mv = viewMatrix*w; vV = mv.xyz;
  vN = normalize(normalMatrix*normal);
  gl_Position = projectionMatrix*mv;
}`;

const HATCH = `
uniform float uScale, uSpacing; uniform sampler2D tND; uniform vec2 uRes;
// one family of engraved lines along iso-lines of s; coverage follows tone; spacing held in 1080p pixels
float lines(float s, float tone, float spacing, float phase){
  if (tone <= 0.004) return 0.0;
  vec2 g = vec2(dFdx(s), dFdy(s));
  float gpx = max(length(g), 1e-7) / uScale;          // change of s per 1080p pixel
  float L = log2(1.0/(spacing*gpx));
  float L0 = floor(L), f = L - L0;
  float aa = 0.62/uScale;
  float ink = 0.0;
  for (int k = 0; k < 2; k++) {
    float F = exp2(L0 + float(k));
    float sp = 1.0/(F*gpx);
    float dpx = abs(fract(s*F + phase) - 0.5)*sp;
    float hw = max(0.5*sp*tone, 0.42*smoothstep(0.0, 0.12, tone));
    float c = clamp((hw - dpx)/aa + 0.5, 0.0, 1.0);
    ink += c*(k == 0 ? 1.0 - f : f);
  }
  return clamp(ink, 0.0, 1.0);
}
// fixed-frequency lines (shape-bound patterns: feather barbs, flame tongues)
float flines(float s, float tone){
  float w = fwidth(s);
  float d = abs(fract(s) - 0.5);
  float hw = 0.5*tone;
  return clamp((hw - d)/max(w, 1e-4) + 0.5, 0.0, 1.0);
}
// screen-space diagonal lines (the darkest shade, and sections)
float slines(float tone, float ang, float spacing){
  vec2 p = gl_FragCoord.xy/uScale;
  float s = dot(p, vec2(cos(ang), sin(ang)))/spacing;
  float d = abs(fract(s) - 0.5)*spacing;
  float hw = 0.5*spacing*tone;
  return clamp((hw - d)*uScale + 0.5, 0.0, 1.0);
}
// light from the upper left, held to the viewer (an engraver's light), with lines gathering toward the silhouette
float toneOf(vec3 N, vec3 V){
  vec3 Ld = normalize(vec3(-0.55, 0.62, 0.56));
  float dif = dot(N, Ld);
  float t = 1.0 - smoothstep(-0.3, 0.92, dif);
  float rim = pow(1.0 - clamp(abs(dot(N, normalize(-V))), 0.0, 1.0), 2.2);
  return clamp(t*0.82 + rim*0.4, 0.0, 1.0);
}
uniform float uCross;
float engrave(float s1, float s2, float tone){
  float a = lines(s1, smoothstep(0.1, 1.0, tone)*0.62, uSpacing, 0.0);
  float b = lines(s2, smoothstep(0.52, 1.0, tone)*0.5*uCross, uSpacing*1.05, 0.25);
  float c = slines(smoothstep(0.82, 1.0, tone)*0.45*uCross, 0.62, uSpacing*0.9);
  return 1.0 - (1.0 - a)*(1.0 - b)*(1.0 - c);
}
// depth of the solid shapes at this pixel (0 = none)
float solidDepth(){ vec4 s = texture2D(tND, gl_FragCoord.xy/uRes); return s.a; }
`;

function mat(fs, uniforms = {}, o = {}) {
  return new THREE.ShaderMaterial({ uniforms: { ...PU, ...uniforms }, vertexShader: o.vs || VS, fragmentShader: fs, side: o.side ?? THREE.FrontSide,
    depthTest: o.depthTest ?? true, depthWrite: o.depthWrite ?? true, transparent: !!o.transparent,
    blending: o.blending ?? THREE.NoBlending, ...(o.max ? { blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor } : {}) });
}

// ---------------------------------------------------------------- ND (normal + depth) materials
const ND_FS = `varying vec3 vN; varying vec3 vV; varying vec3 vW; uniform vec4 uCut; uniform float uCutOn;
void main(){ if (uCutOn > 0.5 && dot(vec4(vW, 1.0), uCut) > 0.0) discard;
  vec3 N = normalize(vN); if (!gl_FrontFacing) N = -N; gl_FragColor = vec4(N*0.5 + 0.5, -vV.z); }`;
export function ndMat(o = {}) { return mat(ND_FS, { uCut: { value: o.cut || new THREE.Vector4(0, 0, 0, 0) }, uCutOn: { value: o.cut ? 1 : 0 } }, { side: o.side ?? THREE.FrontSide }); }

// ---------------------------------------------------------------- bones (vertebrae, head)
// s1 / s2: GLSL expressions for the two line families (surface coordinates); wash: body id in the alpha channel
export function boneMat({ wash = 1, s1 = 'vUv.y', s2 = 'vUv.x', toneAdd = 0, toneMul = 1, cut = null, side = THREE.FrontSide, cross = 1 } = {}) {
  const fs = `varying vec3 vN; varying vec3 vV; varying vec2 vUv; varying vec3 vW; varying vec3 vO;
    uniform float uWash, uToneAdd, uToneMul, uCutOn; uniform vec4 uCut; ${HATCH}
    void main(){
      if (uCutOn > 0.5 && dot(vec4(vW, 1.0), uCut) > 0.0) discard;
      vec3 N = normalize(vN);
      float ink;
      if (!gl_FrontFacing) {
        // the cut face of a sectioned bone: close parallel section lines with a stipple of spongy bone
        float st = step(0.82, fract(sin(dot(floor(gl_FragCoord.xy/uScale/2.0), vec2(12.9898, 78.233)))*43758.5453));
        ink = max(slines(0.24, 0.785, 4.6), st*0.6);
      } else {
        float tone = clamp(toneOf(N, vV)*uToneMul + uToneAdd, 0.0, 1.0);
        ink = engrave(${s1}, ${s2}, tone);
      }
      gl_FragColor = vec4(ink, 0.0, 0.0, uWash);
    }`;
  return mat(fs, { uWash: { value: wash }, uToneAdd: { value: toneAdd }, uToneMul: { value: toneMul }, uCross: { value: cross }, uCut: { value: cut || new THREE.Vector4() }, uCutOn: { value: cut ? 1 : 0 } }, { side });
}

// flat ink (pupils), or paper (glints): density d
export function flatMat(d, wash = 1) {
  return mat(`uniform float uD, uWash; void main(){ gl_FragColor = vec4(uD, 0.0, 0.0, uWash); }`, { uD: { value: d }, uWash: { value: wash } });
}

// ---------------------------------------------------------------- crest shapes (mirror lib/creature.js shaders)
const FEATHER_SHAPE = `float y = vUv.y; float x = abs(vUv.x - 0.5)*2.0;
  float tw = sin(3.14159*pow(y, 0.8))*0.95 + 0.05;
  float body = smoothstep(tw, tw - 0.12, x);
  float shaft = smoothstep(0.08, 0.0, x);
  float a = body*smoothstep(1.0, 0.85, y) + shaft*0.9*step(y, 0.97);`;
export function featherInk(wash) {
  return mat(`varying vec2 vUv; varying vec3 vN; varying vec3 vV; uniform float uWash; ${HATCH}
    void main(){ ${FEATHER_SHAPE}
      if (a < 0.5) discard;
      // barbs: the feather's own pattern, as engraved lines; darker toward the edge of the vane and the base
      float tone = 0.32 + 0.4*smoothstep(0.5, 1.0, x/max(tw, 0.05)) + 0.25*(1.0 - y);
      float barb = flines((y*90.0 - x*14.0)/6.2832, clamp(tone, 0.0, 0.9)*0.75);
      float sh = smoothstep(0.05, 0.02, x);
      gl_FragColor = vec4(max(barb, sh*0.95), 0.0, 0.0, uWash); }`, { uWash: { value: wash } }, { side: THREE.DoubleSide });
}
export function featherND() {
  return mat(`varying vec2 vUv; varying vec3 vN; varying vec3 vV; void main(){ ${FEATHER_SHAPE} if (a < 0.5) discard;
    vec3 N = normalize(vN); if (!gl_FrontFacing) N = -N; gl_FragColor = vec4(N*0.5 + 0.5, -vV.z); }`, {}, { side: THREE.DoubleSide });
}
const FLAME_SHAPE = `
float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float n(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);return mix(mix(h(i),h(i+vec2(1,0)),u.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),u.x),u.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*n(p);p*=2.03;a*=.5;}return v;}
float flameShape(vec2 uv, out float xo, out float w){ vec2 p = uv; p.x = (p.x - .5)*2.; float y = p.y;
  // an engraver's flame: a teardrop whose tip sways (same place and size as the film's flame)
  w = 0.92*pow(sin(3.14159*pow(clamp(y, 0.0, 1.0), 0.62)), 0.9)*(1.0 - 0.25*y);
  float d = fbm(vec2(y*2.2 - uTime*1.6, 1.7));
  xo = p.x - (0.22*sin(y*3.6 - uTime*3.1) + (d - 0.5)*0.4)*y*y;
  return smoothstep(w, w*0.86, abs(xo))*smoothstep(0.98, 0.9, y)*smoothstep(0.0, 0.05, y); }`;
export function flameInk(timeUniform, wash) {
  return mat(`varying vec2 vUv; varying vec3 vN; varying vec3 vV; uniform float uTime, uWash; ${HATCH} ${FLAME_SHAPE}
    void main(){ float xo, w; float sh = flameShape(vUv, xo, w); if (sh < 0.26) discard;
      // tongues of flame: lines that run with the flame's own sway, closer and heavier toward its edges
      float r = xo/max(w, 0.04);
      float tone = 0.16 + 0.6*smoothstep(0.3, 1.0, abs(r)) + 0.15*smoothstep(0.5, 0.0, vUv.y);
      float l = flines(r*5.0*(0.6 + 0.4*vUv.y), clamp(tone, 0.0, 0.85)*0.42);
      gl_FragColor = vec4(l, 0.0, 0.0, uWash); }`, { uTime: timeUniform, uWash: { value: wash } }, { side: THREE.DoubleSide });
}
export function flameND(timeUniform) {
  return mat(`varying vec2 vUv; varying vec3 vN; varying vec3 vV; uniform float uTime; ${FLAME_SHAPE}
    void main(){ float xo, w; if (flameShape(vUv, xo, w) < 0.26) discard; vec3 N = normalize(vN); if (!gl_FrontFacing) N = -N; gl_FragColor = vec4(N*0.5 + 0.5, -vV.z); }`,
  { uTime: timeUniform }, { side: THREE.DoubleSide });
}

// ---------------------------------------------------------------- coloured inks: the residual stream (red), the link (blue-green)
// Drawn over the bones (as a hand-coloured plate would be): full strength where the shape lies inside a bone or in
// the open, faint where something solid stands well in front of it. channel 'red' | 'teal'.
export function inkTube(channel, { base = 0.42, lineTone = 0.55, uEnd = 1.0, depthTest = false, along = 'vUv.y', across = 'vUv.x', tol = 0.75 } = {}) {
  const out = channel === 'red' ? 'vec4(0.0, v, 0.0, 0.0)' : 'vec4(0.0, 0.0, v, 0.0)';
  return mat(`varying vec3 vN; varying vec3 vV; varying vec2 vUv; uniform float uBase, uLT, uEnd, uAmt, uTol; ${HATCH}
    void main(){
      if (vUv.x > uEnd) discard;
      vec3 N = normalize(vN);
      float nv = abs(dot(N, normalize(-vV)));
      float edge = smoothstep(0.42, 0.18, nv);                       // the two edges of the tube, as ink lines
      float tone = toneOf(N, vV);
      float l = lines(${along}, smoothstep(0.2, 1.0, tone)*uLT, uSpacing*0.85, 0.0);   // fibres along the length
      float v = max(max(uBase*(0.6 + 0.4*tone), l), edge);
      float sd = solidDepth();
      float hidden = (sd > 0.0 && -vV.z > sd + uTol) ? 1.0 : 0.0;
      v *= mix(1.0, 0.28, hidden) * uAmt;
      gl_FragColor = ${out};
    }`, { uBase: { value: base }, uLT: { value: lineTone }, uEnd: { value: uEnd }, uAmt: { value: 1 }, uTol: { value: tol } },
  { depthTest, depthWrite: false, transparent: true, max: true });
}
// flat coloured ink (arrowheads, marks)
export function inkFlat(channel, amt = 0.95) {
  const out = channel === 'red' ? 'vec4(0.0, uAmt, 0.0, 0.0)' : channel === 'teal' ? 'vec4(0.0, 0.0, uAmt, 0.0)' : 'vec4(uAmt, 0.0, 0.0, 0.0)';
  return mat(`uniform float uAmt; void main(){ gl_FragColor = ${out}; }`, { uAmt: { value: amt } }, { depthTest: false, depthWrite: false, transparent: true, max: true, side: THREE.DoubleSide });
}

// ---------------------------------------------------------------- the floor: ruled construction lines and engraved shadows
export function floorInk(bases) {
  return mat(`varying vec3 vW; uniform vec2 uShadowDir; uniform float uFloorFade; uniform vec3 uB0, uB1; ${HATCH}
    float px(float c, float wpx){ float d = abs(fract(c + 0.5) - 0.5); float fw = max(fwidth(c), 1e-5)/uScale; return clamp((wpx*0.5 - d/fw)*uScale + 0.5, 0.0, 1.0); }
    void main(){
      float r = length(vW.xz);
      float fade = smoothstep(10.5, 2.5, r) * uFloorFade;
      float g = max(px(vW.x, 0.8), px(vW.z, 0.8)) * 0.36;
      // the mid-line between the two bodies and the compass circle through both feet
      float mid = px(vW.x*0.5, 1.4)*step(abs(vW.x), 0.6)*0.75;
      float ang = atan(vW.z, vW.x);
      float circ = px((r - 2.3)/9.0, 1.2) * step(0.45, fract(ang*24.0/6.2832)) * 0.8;
      float circ2 = px((r - 4.6)/9.0, 1.0) * step(0.5, fract(ang*40.0/6.2832)) * 0.5;
      // cast shadows: engraved horizontal lines, densest under the body
      float sh = 0.0;
      for (int i = 0; i < 2; i++) {
        vec3 b = i == 0 ? uB0 : uB1;
        vec2 q = vW.xz - (b.xz + uShadowDir*0.55);
        vec2 dir = normalize(uShadowDir); vec2 nrm = vec2(-dir.y, dir.x);
        float e = length(vec2(dot(q, dir)/1.05, dot(q, nrm)/0.5));
        sh = max(sh, smoothstep(1.0, 0.15, e));
      }
      float shl = lines(vW.z*1.0 + vW.x*0.12, sh*0.55, uSpacing*0.95, 0.0);
      float v = max(max(g, mid), max(circ, circ2))*fade;
      v = max(v, shl*smoothstep(14.0, 4.0, r)*uFloorFade);
      gl_FragColor = vec4(v, 0.0, 0.0, 0.0);
    }`, { uB0: { value: bases[0] }, uB1: { value: bases[1] } }, { depthWrite: false });
}

// ---------------------------------------------------------------- the renderer
export class PlateRenderer {
  constructor(renderer, W, H, scale) {
    this.r = renderer; this.items = new Map();
    const w = Math.round(W * scale), h = Math.round(H * scale);
    this.nd = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true });
    this.ink = new THREE.WebGLRenderTarget(w, h, { type: THREE.UnsignedByteType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: true });
    PU.uScale.value = scale; PU.uRes.value.set(w, h); PU.tND.value = this.nd.texture;
    this._c = new THREE.Color();
  }
  // obj: a mesh of the scene; ink / nd: its plate materials (null = not drawn in that pass); plateOnly: drawn only
  // in the plate (obj.userData.plateVisible decides)
  add(obj, ink, nd = null, plateOnly = false) { this.items.set(obj, { ink, nd, plateOnly }); if (plateOnly) obj.visible = false; return obj; }
  render(scene, camera) {
    const r = this.r;
    const prevRT = r.getRenderTarget(), prevAlpha = r.getClearAlpha(); r.getClearColor(this._c);
    const bg = scene.background, fog = scene.fog; scene.background = null; scene.fog = null;
    const saved = [];
    scene.traverse((o) => { if (o.isMesh || o.isPoints || o.isSprite || o.isLine) saved.push([o, o.visible, o.material]); });
    const pass = (key, rt, clear) => {
      for (const [o, vis, m] of saved) {
        const it = this.items.get(o);
        if (!it || !it[key] || o.userData.plateHide) { o.visible = false; continue; }
        o.visible = it.plateOnly ? !!o.userData.plateVisible : vis; o.material = it[key];
      }
      r.setRenderTarget(rt); r.setClearColor(clear[0], clear[1]); r.clear(true, true, true); r.render(scene, camera);
    };
    pass('nd', this.nd, [new THREE.Color(0.5, 0.5, 1.0), 0]);
    pass('ink', this.ink, [new THREE.Color(0, 0, 0), 0]);
    for (const [o, vis, m] of saved) { o.visible = vis; o.material = m; }
    scene.background = bg; scene.fog = fog;
    r.setRenderTarget(prevRT); r.setClearColor(this._c, prevAlpha);
    return { ink: this.ink.texture, nd: this.nd.texture };
  }
}

// a printed mark on a camera-facing quad: 'plus' (a circled plus: added into the stream)
export function inkSymbol(channel, amt = 1) {
  const out = channel === 'red' ? 'vec4(0.0, v, 0.0, 0.0)' : channel === 'teal' ? 'vec4(0.0, 0.0, v, 0.0)' : 'vec4(v, 0.0, 0.0, 0.0)';
  return mat(`varying vec2 vUv; uniform float uAmt, uDraw;
    void main(){ vec2 p = vUv*2.0 - 1.0; float r = length(p); float a = atan(p.y, p.x)/6.2832 + 0.5;
      float aa = fwidth(r)*1.2;
      float ring = smoothstep(0.13 + aa, 0.13 - aa, abs(r - 0.8))*step(a, uDraw);
      float bar = max(step(abs(p.x), 0.12)*step(abs(p.y), 0.5), step(abs(p.y), 0.12)*step(abs(p.x), 0.5))*step(0.6, uDraw);
      float disc = step(r, 0.93);
      float v = max(ring, bar)*uAmt;
      if (disc < 0.5) discard;
      gl_FragColor = ${out}; }`, { uAmt: { value: amt }, uDraw: { value: 1 } }, { depthTest: false, depthWrite: false, transparent: true, max: true, side: THREE.DoubleSide });
}
