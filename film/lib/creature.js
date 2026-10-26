import * as THREE from 'three';
import { C } from './palette.js';
import { clamp, lerp, smooth, easeOutBack, hash, noise1 } from './util.js';

// A model as a creature. The body is the model itself, drawn accurately:
//  - 36 vertebrae = the 36 transformer blocks (block 0 at the bottom, block 35 under the head)
//  - a glowing core running through them = the residual stream (2,560 wide)
//  - teal port rings on blocks 6, 9, 12, 15, 18, 21, 24, 27 = the 8 exchange points of the link
//  - input enters at the tail (block 0), the next token leaves from the head (after block 35)
// Quill (Qwen3-4B) has a feather crest; Ember (AFM-4.5B) has a flame crest.
export const NBLOCKS = 36;
export const XBLOCKS = [6, 9, 12, 15, 18, 21, 24, 27];

const SIDES = {
  quill: { col: C.quill, hi: C.quillHi, deep: C.quillDeep },
  ember: { col: C.ember, hi: C.emberHi, deep: C.emberDeep },
};

function puckGeometry(r, h) {
  // rounded puck via lathe of a rounded-rectangle half profile
  const pts = []; const b = h * 0.48; const n = 6;
  pts.push(new THREE.Vector2(0, -h / 2));
  for (let k = 0; k <= n; k++) { const a = -Math.PI / 2 + (k / n) * (Math.PI / 2); pts.push(new THREE.Vector2(r - b + Math.cos(a) * b, -h / 2 + b + Math.sin(a) * b)); }
  for (let k = 0; k <= n; k++) { const a = (k / n) * (Math.PI / 2); pts.push(new THREE.Vector2(r - b + Math.cos(a) * b, h / 2 - b + Math.sin(a) * b)); }
  pts.push(new THREE.Vector2(0, h / 2));
  return new THREE.LatheGeometry(pts, 40);
}

const _flameVS = `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);} `;
const _flameFS = `
uniform float uTime; uniform vec3 uCol; uniform vec3 uHi; uniform float uAmp; uniform float uSeed; varying vec2 vUv;
float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float n(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);return mix(mix(h(i),h(i+vec2(1,0)),u.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),u.x),u.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*n(p);p*=2.03;a*=.5;}return v;}
void main(){ vec2 p=vUv; p.x=(p.x-.5)*2.; float y=p.y;
 float w = (1.0-y)*0.85*(0.4+0.6*smoothstep(0.,.25,y));
 float d = fbm(vec2(p.x*2.0 + uSeed*7.3, y*3.0 - uTime*(2.4 + uSeed*0.3)));
 float shape = smoothstep(w, w*0.2, abs(p.x + (d-.5)*0.6*y)) * smoothstep(1.0, 0.25+0.35*d, y) * smoothstep(0.0,0.08,y);
 vec3 col = mix(uCol, uHi, smoothstep(.3,1.,shape)*(1.0-y*0.6));
 gl_FragColor = vec4(col*(0.75+0.6*shape)*uAmp, shape*uAmp*0.9); }`;

const _featherFS = `
uniform vec3 uCol; uniform vec3 uHi; uniform float uAmp; varying vec2 vUv;
void main(){ float y=vUv.y; float x=abs(vUv.x-.5)*2.;
 float tw = sin(3.14159*pow(y,0.8))*0.95 + 0.05;
 float body = smoothstep(tw, tw-0.12, x);
 float barb = 0.65 + 0.35*sin(y*90.0 - x*14.0);
 float shaft = smoothstep(0.08, 0.0, x);
 float a = body*barb*smoothstep(1.0,0.85,y) + shaft*0.9;
 vec3 col = mix(uCol, uHi, y*0.7 + shaft*0.6);
 gl_FragColor = vec4(col*(1.2+shaft)*uAmp, a*uAmp*0.95); }`;

// A lock of crest hair: a ribbon with uv.x across and uv.y from root (0) to tip (1); its centreline is set per frame.
const LOCK_N = 22;
function lockGeometry() {
  const g = new THREE.BufferGeometry(), n = LOCK_N + 1;
  const pos = new Float32Array(n * 2 * 3), uv = new Float32Array(n * 2 * 2), idx = [];
  for (let k = 0; k < n; k++) { uv[k * 4] = 0; uv[k * 4 + 1] = k / LOCK_N; uv[k * 4 + 2] = 1; uv[k * 4 + 3] = k / LOCK_N; }
  for (let k = 0; k < LOCK_N; k++) { const a = k * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(n * 2 * 3).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)), 3));
  g.setIndex(idx); return g;
}

export class Creature {
  constructor(side = 'quill', opts = {}) {
    this.side = side;
    const pal = SIDES[side];
    this.pal = pal;
    this.group = new THREE.Group();
    this.H = opts.height ?? 4.2; // spine height in world units
    this.R = opts.radius ?? 0.52;
    this.pose = Creature.defaultPose(side);
    this.blockGlow = new Float32Array(NBLOCKS); // extra per-block emissive (shots set this)
    this.blockColor = Array.from({ length: NBLOCKS }, () => null); // optional per-block color override (THREE.Color)
    // writes the partner adds into this model's residual stream: [{block, color: THREE.Color, amount, top?}] (shots set it every frame)
    this.inject = [];

    // --- vertebrae (blocks)
    this.blocks = []; this.blockMats = []; this.iceMeshes = []; this.loraMeshes = []; this.ports = [];
    const iceMat = new THREE.MeshStandardMaterial({ color: C.ice, emissive: C.ice, emissiveIntensity: 0.15, roughness: 0.25, metalness: 0.0, transparent: true, opacity: 0, depthWrite: false });
    this.iceMat = iceMat;
    this.loraMat = new THREE.MeshBasicMaterial({ color: C.link, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
    const hb = this.H / NBLOCKS;
    for (let i = 0; i < NBLOCKS; i++) {
      const r = this.blockRadius(i);
      const g = puckGeometry(r, hb * 0.62);
      const m = new THREE.MeshStandardMaterial({ color: pal.deep.clone(), emissive: pal.col.clone(), emissiveIntensity: 0.35, roughness: 0.32, metalness: 0.35, transparent: true, opacity: 1 });
      const mesh = new THREE.Mesh(g, m);
      this.group.add(mesh); this.blocks.push(mesh); this.blockMats.push(m);
      const ice = new THREE.Mesh(puckGeometry(r * 1.13, hb * 0.86), iceMat); ice.visible = false;
      this.group.add(ice); this.iceMeshes.push(ice);
      // LoRA adapter: a small bright chip on the front of each block (rank-32 adapters train; base weights frozen)
      const lora = new THREE.Mesh(new THREE.BoxGeometry(r * 0.55, hb * 0.32, 0.03), this.loraMat); lora.visible = false;
      this.group.add(lora); this.loraMeshes.push(lora);
      if (XBLOCKS.includes(i)) {
        const pm = new THREE.MeshBasicMaterial({ color: C.link.clone(), transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false });
        const port = new THREE.Mesh(new THREE.TorusGeometry(r * 1.06, 0.022, 8, 48), pm);
        this.group.add(port); this.ports.push({ block: i, mesh: port, mat: pm });
      }
    }
    // --- residual stream core: a glowing tube through all the blocks
    this.coreMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uCol: { value: pal.hi.clone() }, uAmp: { value: 1 }, uFlow: { value: 1 },
        uWave: { value: -1 }, uWaveAmp: { value: 1 },
        uInj: { value: Array.from({ length: 16 }, () => new THREE.Vector4(2, 0, 0, 0)) }, uInjAmt: { value: new Float32Array(16) }, uInjTop: { value: new Float32Array(16).fill(1) } },
      vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
      fragmentShader: `uniform float uTime; uniform vec3 uCol; uniform float uAmp; uniform float uFlow; uniform float uWave; uniform float uWaveAmp;
        uniform vec4 uInj[16]; uniform float uInjAmt[16]; uniform float uInjTop[16]; varying vec2 vUv;
        void main(){ float u = vUv.x; float p = 0.55 + 0.45*sin(u*60.0 - uTime*6.0*uFlow);
          vec3 c = uCol*(0.5+0.35*p);
          // forward-pass wavefront: computed part below the front glows, the front itself flares, above it is dim
          if (uWave > -0.5) { float below = step(u, uWave); float front = exp(-pow((u-uWave)*28.0, 2.0));
            c *= mix(0.3, 1.0, below); c += uCol*front*1.6*uWaveAmp; }
          // writes from the partner: added into the residual stream at u0, then carried upward (up to uInjTop)
          for (int k=0;k<16;k++){ float a = uInjAmt[k]; if (a <= 0.0) continue; float u0 = uInj[k].x;
            float on = smoothstep(u0-0.012, u0+0.004, u) * (1.0 - smoothstep(uInjTop[k]-0.01, uInjTop[k]+0.01, u));
            float hit = exp(-pow((u-u0)*40.0, 2.0));
            c = mix(c, uInj[k].yzw*(0.8+0.3*p), clamp(on*a*0.85, 0.0, 0.85)) + uInj[k].yzw * a * hit*2.0; }
          gl_FragColor=vec4(c*uAmp, 1.0);} `,
      transparent: false,
    });
    this.core = new THREE.Mesh(new THREE.BufferGeometry(), this.coreMat);
    this.group.add(this.core);

    // --- head
    const headMat = new THREE.MeshStandardMaterial({ color: pal.deep.clone().lerp(pal.col, 0.35), emissive: pal.col.clone(), emissiveIntensity: 0.45, roughness: 0.25, metalness: 0.2, transparent: true });
    this.headMat = headMat;
    this.head = new THREE.Group(); this.group.add(this.head);
    this.headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.62, 48, 32), headMat);
    this.headMesh.scale.set(1.05, 0.9, 0.98);
    this.head.add(this.headMesh);
    // face pivot: rotates the eyes toward faceYaw
    this.face = new THREE.Group(); this.head.add(this.face);
    const eyeWhiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.55, roughness: 0.2, transparent: true });
    const pupilMat = new THREE.MeshStandardMaterial({ color: 0x0a0612, roughness: 0.15, metalness: 0.1, transparent: true });
    const glintMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true });
    const arcMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true });
    this.eyeMats = [eyeWhiteMat, pupilMat, glintMat, arcMat];
    this.eyes = [];
    for (const s of [-1, 1]) {
      const eye = new THREE.Group();
      const az = s * 0.4, el = 0.08, rr = 0.6;
      eye.position.set(Math.sin(az) * Math.cos(el) * rr, Math.sin(el) * rr * 0.9, Math.cos(az) * Math.cos(el) * rr);
      eye.lookAt(eye.position.clone().multiplyScalar(2));
      const white = new THREE.Mesh(new THREE.SphereGeometry(0.19, 28, 18), eyeWhiteMat); white.scale.set(1, 1.12, 0.45);
      const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.11, 22, 16), pupilMat); pupil.scale.set(1, 1.1, 0.5); pupil.position.z = 0.06;
      const glint = new THREE.Mesh(new THREE.SphereGeometry(0.028, 10, 8), glintMat); glint.position.set(0.04, 0.05, 0.12);
      const arc = new THREE.Mesh(new THREE.TorusGeometry(0.125, 0.03, 8, 24, Math.PI), arcMat); arc.position.z = 0.06; arc.visible = false;
      eye.add(white); eye.add(pupil); pupil.add(glint); eye.add(arc);
      this.face.add(eye);
      this.eyes.push({ group: eye, white, pupil, glint, arc });
    }
    // cheeks
    const blushMat = new THREE.MeshBasicMaterial({ color: side === 'quill' ? 0xf0abfc : 0xfda4af, transparent: true, opacity: 0.35, depthWrite: false });
    this.blushMat = blushMat;
    for (const s of [-1, 1]) {
      const az = s * 0.62, el = -0.12, rr = 0.615;
      const b = new THREE.Mesh(new THREE.CircleGeometry(0.075, 20), blushMat);
      b.position.set(Math.sin(az) * Math.cos(el) * rr, Math.sin(el) * rr * 0.9, Math.cos(az) * Math.cos(el) * rr);
      b.lookAt(b.position.clone().multiplyScalar(2)); b.scale.set(1.4, 0.8, 1);
      this.face.add(b);
    }
    // crest
    this.crest = new THREE.Group(); this.head.add(this.crest);
    if (side === 'quill') {
      this.feathers = [];
      const fm = new THREE.ShaderMaterial({ uniforms: { uCol: { value: pal.col.clone() }, uHi: { value: pal.hi.clone() }, uAmp: { value: 1 } }, vertexShader: _flameVS, fragmentShader: _featherFS, transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
      this.crestMat = fm;
      // root (head-local), length, width, spread around the back (az, 0 = forward), first angle from vertical, phase
      const specs = [{ r: [0, 0.47, -0.1], len: 1.6, w: 0.44, az: Math.PI, th: 0.32, ph: 0 },
        { r: [0.14, 0.45, -0.15], len: 1.22, w: 0.33, az: Math.PI - 0.55, th: 0.5, ph: 1.3 }, { r: [-0.14, 0.45, -0.15], len: 1.18, w: 0.33, az: Math.PI + 0.55, th: 0.5, ph: 2.1 },
        { r: [0.25, 0.38, -0.2], len: 0.9, w: 0.27, az: Math.PI - 1.0, th: 0.72, ph: 3.4 }, { r: [-0.25, 0.38, -0.2], len: 0.88, w: 0.27, az: Math.PI + 1.0, th: 0.72, ph: 4.2 }];
      for (const sp of specs) {
        const f = new THREE.Mesh(lockGeometry(), fm); f.frustumCulled = false;
        f.userData.lock = sp; this.crest.add(f); this.feathers.push(f);
      }
    } else {
      const fm = new THREE.ShaderMaterial({ uniforms: { uTime: { value: 0 }, uCol: { value: pal.col.clone() }, uHi: { value: new THREE.Color('#ffe7a3') }, uAmp: { value: 1 } }, vertexShader: _flameVS, fragmentShader: _flameFS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
      this.crestMat = fm;
      this.flame = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 1.35), fm);
      fm.uniforms.uSeed = { value: 0 };
      this.flame.position.set(0, 0.95, -0.05);
      this.crest.add(this.flame);
      // flame locks: tongues of the same fire that stream like hair
      this.flameLocks = [];
      const specs = [{ r: [0, 0.5, -0.05], len: 1.5, w: 0.62, az: Math.PI, th: 0.1, ph: 0 },
        { r: [0.17, 0.47, -0.08], len: 1.18, w: 0.5, az: Math.PI - 0.6, th: 0.28, ph: 1.7 }, { r: [-0.17, 0.47, -0.08], len: 1.12, w: 0.5, az: Math.PI + 0.6, th: 0.28, ph: 2.6 },
        { r: [0.29, 0.4, -0.13], len: 0.86, w: 0.42, az: Math.PI - 1.1, th: 0.5, ph: 3.3 }, { r: [-0.29, 0.4, -0.13], len: 0.84, w: 0.42, az: Math.PI + 1.1, th: 0.5, ph: 4.4 },
        { r: [0, 0.42, -0.26], len: 1.0, w: 0.5, az: Math.PI, th: 0.55, ph: 5.1 }];
      specs.forEach((sp, j) => {
        const m = new THREE.ShaderMaterial({ uniforms: { uTime: fm.uniforms.uTime, uCol: fm.uniforms.uCol, uHi: fm.uniforms.uHi, uAmp: fm.uniforms.uAmp, uSeed: { value: 1 + j } }, vertexShader: _flameVS, fragmentShader: _flameFS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
        const f = new THREE.Mesh(lockGeometry(), m); f.frustumCulled = false;
        f.userData.lock = sp; this.crest.add(f); this.flameLocks.push(f);
      });
      // sparks
      const n = 18; const pos = new Float32Array(n * 3);
      const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      this.sparkMat = new THREE.PointsMaterial({ color: new THREE.Color('#ffc078'), size: 0.07, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, map: Creature.dotTexture() });
      this.sparks = new THREE.Points(sg, this.sparkMat); this.sparks.frustumCulled = false; this.group.add(this.sparks);
    }
    // tail light: where the prompt enters (block 0)
    this.tailMat = new THREE.SpriteMaterial({ map: Creature.dotTexture(), color: pal.hi.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.tail = new THREE.Sprite(this.tailMat); this.tail.scale.set(0.9, 0.9, 1); this.group.add(this.tail);
    // soft ground glow
    this.groundMat = new THREE.MeshBasicMaterial({ map: Creature.dotTexture(), color: pal.col.clone(), transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false });
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 3.2), this.groundMat); this.ground.rotation.x = -Math.PI / 2;
    this.group.add(this.ground);

    // scatter positions for assembly
    this.scatter = [];
    for (let i = 0; i <= NBLOCKS; i++) {
      const a = hash(i * 3.1 + (side === 'quill' ? 1 : 7)) * Math.PI * 2, e = (hash(i * 7.7 + 2) - 0.5) * 1.6, d = 5 + hash(i * 1.3 + 5) * 5;
      this.scatter.push(new THREE.Vector3(Math.cos(a) * Math.cos(e) * d, 2 + Math.sin(e) * d, Math.sin(a) * Math.cos(e) * d));
    }
    this._centers = Array.from({ length: NBLOCKS }, () => new THREE.Vector3());
    this._tangents = Array.from({ length: NBLOCKS }, () => new THREE.Vector3(0, 1, 0));
    this._headPos = new THREE.Vector3();
  }

  static defaultPose(side) {
    return {
      position: new THREE.Vector3(side === 'quill' ? -2.3 : 2.3, 0, 0), // base of the tail, world
      yaw: side === 'quill' ? 0.5 : -0.5,  // whole-body yaw (radians); default: turned a bit toward the partner, still facing camera (+z)
      lean: 0,        // tilt sideways (radians, + = toward +x)
      leanFwd: 0,     // tilt toward camera (+) / away (-)
      bend: 0,        // C-curve of the spine along +x (world units at top)
      sway: 0, swayPhase: 0, swayFreq: 1, // snake wiggle amplitude, phase (radians), waves along the body
      squash: 1,      // <1 squashes vertically (and fattens), >1 stretches
      headTilt: 0,    // head roll (radians)
      headNod: 0,     // head pitch (radians, + = looks down)
      faceYaw: 0,     // extra head yaw (radians) relative to body
      lookAt: null,   // THREE.Vector3 world point the pupils track (null = straight ahead)
      blink: 0,       // 0 open .. 1 closed
      happy: 0,       // 0 normal eyes .. 1 happy arc eyes (^ ^)
      blush: 0.35,
      glow: 1,        // overall emissive multiplier
      coreGlow: 1,    // residual stream brightness
      coreFlow: 1,    // residual stream flow speed
      portGlow: 0.5,  // brightness of the 8 exchange port rings
      ice: 0,         // 0..1 frost on the blocks (frozen original weights)
      lora: 0,        // 0..1 glowing LoRA adapter chips
      assemble: 1,    // 0 = vertebrae scattered in space .. 1 = assembled body
      opacity: 1,
      ghost: 0,       // 0..1 desaturate to a grey stranger (e.g. the shuffled partner of another episode)
      crest: 1,       // crest visibility/intensity
      hair: null,     // crest lift, 0 = fallen (link cut) .. 1 = up and flowing toward lookAt (linked); null = from crest
      scale: 1,
      groundGlow: 0.5,
      xray: 0.5,      // 0 = solid blocks .. 1 = glassy blocks (the residual stream core and link strands inside are visible)
      wave: -1,       // forward-pass wavefront height 0..1 (-1 = off): blocks below are computed, the front flares
      body: 1,        // 0..1 visibility of the body only (vertebrae, core, ports, tail, ground glow); the head stays (a2 unfolds the body)
    };
  }

  blockRadius(i) { const u = i / (NBLOCKS - 1); return this.R * (0.58 + 0.42 * Math.sin(Math.PI * (0.1 + 0.82 * u))); }

  // spine point in creature-local frame (before world transform), u in [0,1] along height
  _spineLocal(u, out) {
    const p = this.pose, H = this.H * p.squash;
    const sw = p.sway * Math.sin(Math.PI * 2 * p.swayFreq * u - p.swayPhase) * Math.min(1, u * 2.5);
    out.set(p.bend * u * u + sw, H * u, 0);
    return out;
  }

  // call once per frame after setting pose; t = global time (s)
  update(t) {
    const p = this.pose, g = this.group;
    g.position.copy(p.position);
    g.rotation.set(p.leanFwd, p.yaw, -p.lean, 'YXZ');
    g.scale.setScalar(p.scale);
    const widen = 1 / Math.sqrt(Math.max(0.3, p.squash));
    const ghostCol = new THREE.Color('#8b8fa3');
    const colOf = (c) => c.clone().lerp(ghostCol, p.ghost);
    const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();
    const hb = this.H / NBLOCKS;
    const quat = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < NBLOCKS; i++) {
      const u = (i + 0.5) / NBLOCKS;
      this._spineLocal(u, tmp); this._spineLocal(u + 0.01, tmp2);
      const tan = tmp2.sub(tmp).normalize();
      quat.setFromUnitVectors(up, tan);
      // assembly: fly in from scatter, staggered bottom-up
      const k = clamp(p.assemble * 1.7 - (i / NBLOCKS) * 0.7);
      const ka = easeOutBack(k, 1.4);
      const pos = this._centers[i].copy(tmp);
      if (k < 1) {
        pos.lerpVectors(this.scatter[i], tmp, ka);
      }
      this._tangents[i].copy(tan);
      const m = this.blocks[i];
      m.position.copy(pos); m.quaternion.copy(quat);
      if (k < 1) m.rotateX((1 - k) * 6 + i);
      const vis = k > 0.001; m.visible = vis;
      m.scale.set(widen, 1, widen).multiplyScalar(lerp(0.2, 1, Math.min(1, k * 1.3)));
      const mat = this.blockMats[i];
      const base = this.blockColor[i] || this.pal.col;
      mat.emissive.copy(colOf(base));
      mat.color.copy(colOf(this.pal.deep));
      let waveGlow = 0;
      if (p.wave > -0.5) { const ub = (i + 0.5) / NBLOCKS; waveGlow = 1.6 * Math.exp(-Math.pow((ub - p.wave) * 22, 2)) + (ub < p.wave ? 0.12 : -0.12); }
      mat.emissiveIntensity = Math.max(0.05, (0.3 + this.blockGlow[i] + waveGlow) * p.glow * (1 - p.ice * 0.45) * (1 - 0.35 * p.xray));
      mat.opacity = p.opacity * (1 - 0.72 * p.xray) * (p.body ?? 1);
      mat.depthWrite = p.xray < 0.2 && (p.body ?? 1) > 0.999;
      if ((p.body ?? 1) < 0.003) m.visible = false;
      // ice & lora
      const ice = this.iceMeshes[i]; ice.visible = vis && p.ice > 0.001;
      ice.position.copy(pos); ice.quaternion.copy(quat); ice.scale.copy(m.scale);
      const lo = this.loraMeshes[i]; lo.visible = vis && p.lora > 0.001;
      const r = this.blockRadius(i) * widen;
      // chip on the face of the block that points toward the camera-ish (+z local)
      lo.position.copy(pos).add(new THREE.Vector3(0, 0, r * 1.16).applyQuaternion(quat));
      lo.quaternion.copy(quat);
    }
    this.iceMat.opacity = 0.55 * p.ice * p.opacity;
    this.loraMat.opacity = p.lora * (0.75 + 0.25 * Math.sin(t * 9)) * p.opacity;
    for (const pt of this.ports) {
      const i = pt.block, c = this._centers[i];
      pt.mesh.position.copy(c); pt.mesh.quaternion.copy(this.blocks[i].quaternion); pt.mesh.rotateX(Math.PI / 2);
      pt.mesh.scale.copy(this.blocks[i].scale);
      pt.mesh.visible = this.blocks[i].visible;
      pt.mat.opacity = p.portGlow * p.opacity * (1 - p.ghost * 0.6) * (p.body ?? 1);
    }
    // residual stream core along the assembled spine
    const pts = [];
    const nA = Math.round(clamp(p.assemble * 1.2 - 0.15) * NBLOCKS);
    if (nA >= 2) {
      for (let i = 0; i < nA; i++) pts.push(this._centers[i].clone());
      const s = this._spineLocal(1.0, new THREE.Vector3()); if (nA === NBLOCKS) pts.push(s.clone().add(new THREE.Vector3(0, 0.2, 0)));
      const curve = new THREE.CatmullRomCurve3(pts);
      this.core.geometry.dispose();
      this.core.geometry = new THREE.TubeGeometry(curve, Math.max(8, pts.length * 3), 0.085 * widen, 10, false);
      this.core.visible = true;
    } else this.core.visible = false;
    this.coreMat.uniforms.uTime.value = t; this.coreMat.uniforms.uFlow.value = p.coreFlow;
    this.coreMat.uniforms.uAmp.value = p.coreGlow * p.opacity * (1 - 0.4 * p.ghost) * (p.body ?? 1);
    if ((p.body ?? 1) < 0.003) this.core.visible = false;
    this.coreMat.uniforms.uCol.value.copy(colOf(this.pal.hi));
    { const U = this.coreMat.uniforms; U.uWave.value = p.wave > -0.5 ? p.wave : -1;
      const n = Math.min(16, this.inject.length);
      for (let k = 0; k < 16; k++) {
        if (k < n) { const j = this.inject[k]; U.uInj.value[k].set((j.block + 0.5) / (NBLOCKS + 1), j.color.r, j.color.g, j.color.b); U.uInjAmt.value[k] = j.amount ?? 1; U.uInjTop.value[k] = j.top ?? 1.01; }
        else U.uInjAmt.value[k] = 0;
      } }

    // head on top
    const top = this._spineLocal(1, new THREE.Vector3()), top2 = this._spineLocal(0.98, new THREE.Vector3());
    const tan = top.clone().sub(top2).normalize();
    const hk = clamp((p.assemble - 0.75) / 0.25);
    this.head.visible = hk > 0.001;
    this.head.position.copy(top).add(tan.clone().multiplyScalar(0.55));
    this.head.quaternion.setFromUnitVectors(up, tan);
    this.head.rotateZ(p.headTilt); this.head.rotateX(p.headNod);
    this.head.scale.setScalar(easeOutBack(hk, 2.2) * Math.max(0.01, 1)).multiply(new THREE.Vector3(widen, 1 / widen, widen));
    this.face.rotation.y = p.faceYaw;
    this.headMat.opacity = p.opacity; this.headMat.color.copy(colOf(this.pal.deep.clone().lerp(this.pal.col, 0.35)));
    this.headMat.emissive.copy(colOf(this.pal.col)); this.headMat.emissiveIntensity = 0.42 * p.glow;
    for (const m of this.eyeMats) m.opacity = p.opacity;
    this.blushMat.opacity = 0; // no cheek blush (director's call); pose.blush is ignored
    // eyes: blink, happy, look-at
    const blink = clamp(p.blink);
    for (const e of this.eyes) {
      const normal = 1 - clamp(p.happy * 2 - 0.6);
      e.white.visible = normal > 0.02; e.pupil.visible = normal > 0.02;
      e.white.scale.set(1, 1.15 * Math.max(0.06, 1 - blink) * normal, 0.45);
      e.pupil.scale.set(1, 1.1 * Math.max(0.05, 1 - blink) * normal, 0.5);
      e.arc.visible = p.happy > 0.4; e.arc.scale.setScalar(clamp((p.happy - 0.3) * 1.6));
      // pupils track lookAt
      e.pupil.position.set(0, 0, 0.06);
      if (p.lookAt) {
        this.group.updateMatrixWorld(true);
        const inv = new THREE.Matrix4().copy(e.group.matrixWorld).invert();
        const local = p.lookAt.clone().applyMatrix4(inv).normalize();
        e.pupil.position.x = clamp(local.x, -1, 1) * 0.075; e.pupil.position.y = clamp(local.y, -1, 1) * 0.075;
      }
    }
    // crest animation
    this.crest.visible = p.crest > 0.01 && this.head.visible;
    if (this.side === 'quill') {
      this.crestMat.uniforms.uAmp.value = p.crest * p.opacity;
      this.crestMat.uniforms.uCol.value.copy(colOf(this.pal.col)); this.crestMat.uniforms.uHi.value.copy(colOf(this.pal.hi));
      this._hair(t, this.feathers, false);
    } else {
      this.crestMat.uniforms.uTime.value = t; this.crestMat.uniforms.uAmp.value = p.crest * p.opacity;
      this.crestMat.uniforms.uCol.value.copy(colOf(this.pal.col));
      const lift = this._hair(t, this.flameLocks, true);
      this.flame.scale.setScalar(0.62 * (0.55 + 0.45 * Math.min(1, lift)));
      // flame billboards toward the camera (set by shots via creature.camera)
      if (this.camera) { this.group.updateMatrixWorld(true); const wq = new THREE.Quaternion(); this.crest.getWorldQuaternion(wq); this.flame.quaternion.copy(wq.invert().multiply(this.camera.quaternion)); }
      // sparks rise from the flame
      this.group.updateMatrixWorld(true);
      const hp = this.head.getWorldPosition(new THREE.Vector3());
      const inv = new THREE.Matrix4().copy(this.group.matrixWorld).invert();
      const arr = this.sparks.geometry.attributes.position.array; const n = arr.length / 3;
      for (let k = 0; k < n; k++) {
        const ph = (t * (0.35 + hash(k) * 0.3) + hash(k * 3.3)) % 1;
        const w = new THREE.Vector3(hp.x + (hash(k * 5.1) - 0.5) * 0.7 + Math.sin(t * 2 + k) * 0.15 * ph, hp.y + 0.9 + ph * 1.8, hp.z + (hash(k * 2.7) - 0.5) * 0.5).applyMatrix4(inv);
        arr[k * 3] = w.x; arr[k * 3 + 1] = w.y; arr[k * 3 + 2] = w.z;
      }
      this.sparks.geometry.attributes.position.needsUpdate = true;
      this.sparks.visible = this.crest.visible;
      this.sparkMat.opacity = 0.85 * p.crest * p.opacity * clamp(0.2 + 0.8 * lift);
    }
    // tail + ground
    this.tail.position.copy(this._centers[0]).add(new THREE.Vector3(0, -0.25, 0));
    this.tailMat.opacity = 0.7 * p.opacity * clamp(p.assemble * 3) * (p.body ?? 1);
    this.tailMat.color.copy(colOf(this.pal.hi));
    this.ground.position.set(0, 0.01 - p.position.y, 0); // stays on the floor y=0 (world) when the creature hops
    this.groundMat.opacity = p.groundGlow * p.opacity * clamp(1 - p.position.y * 0.4) * (p.body ?? 1);
    this.groundMat.color.copy(colOf(this.pal.col));
    g.updateMatrixWorld(true);
  }

  // Crest hair (pure in t). Each lock's centreline starts upright at its root on the crown and bends along its length;
  // linked (lift 1) it rises and its tips stream toward pose.lookAt (usually the partner), with a wave running to the
  // tips; cut (lift 0) it falls back and droops. Ribbons turn their face to the camera. Returns the lift.
  _hair(t, locks, fire) {
    const p = this.pose;
    const lift = p.hair ?? clamp((p.crest - 0.45) / 0.55, 0, 1.4);
    const L = Math.min(1, lift), over = Math.max(0, lift - 1);
    this.group.updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(this.crest.matrixWorld).invert();
    // flow: the horizontal direction of lookAt in crest space (default: straight back)
    let flowAz = Math.PI;
    if (p.lookAt) { const l = p.lookAt.clone().applyMatrix4(inv); if (Math.hypot(l.x, l.z) > 1e-3) flowAz = Math.atan2(l.x, l.z); }
    const cam = this.camera ? this.camera.position.clone().applyMatrix4(inv) : new THREE.Vector3(0, 0.5, 10);
    const angMix = (a, b, k) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * k; };
    const P = new THREE.Vector3(), d = new THREE.Vector3(), side = new THREE.Vector3(), view = new THREE.Vector3();
    for (const f of locks) {
      const sp = f.userData.lock, arr = f.geometry.attributes.position.array, seg = sp.len * (fire ? lerp(0.62, 1, L) + 0.12 * over : lerp(0.88, 1, L)) / LOCK_N;
      P.set(sp.r[0], sp.r[1], sp.r[2]);
      for (let k = 0; k <= LOCK_N; k++) {
        const s = k / LOCK_N;
        // angle from vertical: linked, a gentle arc (fire rises straighter); cut, falls over and hangs
        const thUp = sp.th + s * (fire ? 0.55 : 1.05) - 0.12 * over;
        const thDown = sp.th + (fire ? 0.75 : 0.5) + s * (fire ? 1.5 : 1.45);
        const wave = Math.sin(t * (fire ? 3.1 : 2.2) - s * 4.2 + sp.ph) * (fire ? 0.16 : 0.11) * s * (0.6 + 0.4 * L)
          + 0.07 * noise1(t * 0.9 + sp.ph, sp.ph) * s;
        const th = lerp(thDown, thUp, L) + wave;
        const az0 = sp.az + (sp.az - Math.PI) * 0.9 * (1 - L);   // wilted locks spread to the sides
        const az = angMix(az0, flowAz, clamp(0.1 + s * 0.85) * (0.25 + 0.6 * L)) + 0.16 * s * Math.sin(t * 1.7 - s * 3.5 + sp.ph * 1.3);
        d.set(Math.sin(th) * Math.sin(az), Math.cos(th), Math.sin(th) * Math.cos(az));
        // the ribbon faces the camera
        view.copy(cam).sub(P).normalize(); side.crossVectors(d, view); if (side.lengthSq() < 1e-6) side.set(1, 0, 0); side.normalize();
        const w = sp.w * 0.5;
        arr[k * 6] = P.x - side.x * w; arr[k * 6 + 1] = P.y - side.y * w; arr[k * 6 + 2] = P.z - side.z * w;
        arr[k * 6 + 3] = P.x + side.x * w; arr[k * 6 + 4] = P.y + side.y * w; arr[k * 6 + 5] = P.z + side.z * w;
        P.addScaledVector(d, seg);
      }
      f.geometry.attributes.position.needsUpdate = true;
    }
    return lift;
  }

  // ---- world-space accessors for shots (valid after update(t))
  blockWorld(i, out = new THREE.Vector3()) { return out.copy(this._centers[i]).applyMatrix4(this.group.matrixWorld); }
  // point on the surface of block i, on the side facing a world point (e.g. the partner)
  blockSurface(i, toward, out = new THREE.Vector3()) {
    const c = this.blockWorld(i, new THREE.Vector3());
    const d = toward.clone().sub(c); d.y = 0; d.normalize();
    return out.copy(c).add(d.multiplyScalar(this.blockRadius(i) * 1.08 * this.pose.scale));
  }
  headWorld(out = new THREE.Vector3()) { return this.head.getWorldPosition(out); }
  // a point just above/in front of the head where output tokens appear
  mouthWorld(out = new THREE.Vector3()) { return this.head.localToWorld(out.set(0, 0.15, 0.75)); }
  tailWorld(out = new THREE.Vector3()) { return this.tail.getWorldPosition(out); }

  static dotTexture() {
    if (Creature._dot) return Creature._dot;
    const c = document.createElement('canvas'); c.width = c.height = 128; const x = c.getContext('2d');
    const g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.25, 'rgba(255,255,255,0.55)'); g.addColorStop(0.6, 'rgba(255,255,255,0.12)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g; x.fillRect(0, 0, 128, 128);
    Creature._dot = new THREE.CanvasTexture(c); Creature._dot.colorSpace = THREE.SRGBColorSpace; return Creature._dot;
  }
}
