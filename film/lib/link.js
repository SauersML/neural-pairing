import * as THREE from 'three';
import { C } from './palette.js';
import { XBLOCKS, Creature } from './creature.js';
import { clamp, hash } from './util.js';

// The link: at each of the 8 exchange blocks, two strands (Quill -> Ember and Ember -> Quill) twisted around each other.
// Glowing packets ride the strands: they are hidden states (continuous vectors), never letters or tokens.
// Both directions run at once, at the same depth.
export class Link {
  constructor(a, b, opts = {}) {
    this.a = a; this.b = b; // a = quill, b = ember (creatures)
    this.blocks = opts.blocks || XBLOCKS;
    this.group = new THREE.Group();
    this.segs = 64; this.radial = 6;
    this.state = {
      grow: 1,            // 0..1 strands grow out from the ports (scalar or array per exchange)
      intensity: 1,       // scalar or array per exchange
      flow: 1,            // packet speed multiplier
      packets: 3,         // packets per strand visible
      packetSize: 0.16,
      aToB: 1, bToA: 1,   // strength of each direction (0 hides that strand's packets)
      sag: -0.25,         // vertical sag of the strands (world units)
      twist: 0.07,        // helix radius between the two strands
      highlight: -1,      // exchange index (0..7) to highlight, -1 none
      dim: 1,             // multiplier for non-highlighted exchanges when highlight>=0
      colorBySender: 0.55,// how much packets take their sender's color (0 = pure link teal)
      depth: 1,           // 0 = strands end at the body surface, 1 = strands plug into the partner's residual stream (core)
      tintAB: null, tintBA: null, // optional packet color override per direction (THREE.Color), e.g. gold for a secret
      pulses: [],         // extra bright packets placed by the shot: [{e, dir, u (0..1 along the strand), size, color}]
    };
    this.strands = [];
    for (let e = 0; e < this.blocks.length; e++) {
      for (const dir of [0, 1]) {
        const mat = new THREE.ShaderMaterial({
          uniforms: { uTime: { value: 0 }, uCol: { value: C.link.clone() }, uAmp: { value: 1 }, uDir: { value: dir }, uGrow: { value: 1 } },
          vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
          fragmentShader: `uniform float uTime; uniform vec3 uCol; uniform float uAmp; uniform float uGrow; varying vec2 vUv;
            void main(){ float x=vUv.x; if(x>uGrow) discard; float tip = smoothstep(uGrow-0.06, uGrow, x)*step(uGrow,0.999);
              float s = 0.55 + 0.45*sin(x*70.0 - uTime*9.0);
              float edge = 1.0 - abs(vUv.y-0.5)*1.2;
              gl_FragColor = vec4(uCol*uAmp*(0.55+0.6*s+2.5*tip)*edge, 1.0);} `,
          transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
        this.group.add(mesh);
        // packets for this strand
        const pg = new THREE.BufferGeometry(); const NP = 8;
        pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
        pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
        const pm = new THREE.PointsMaterial({ size: 0.16, map: Creature.dotTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true });
        const pts = new THREE.Points(pg, pm); pts.frustumCulled = false; this.group.add(pts);
        this.strands.push({ e, dir, mesh, mat, pts, pm, NP, curve: null });
      }
    }
    const NPU = 48; const ug = new THREE.BufferGeometry();
    ug.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NPU * 3), 3));
    ug.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NPU * 3), 3));
    this.pulsePts = new THREE.Points(ug, new THREE.PointsMaterial({ size: 0.34, map: Creature.dotTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); this.pulsePts.frustumCulled = false;
    this.group.add(this.pulsePts);
  }

  _val(v, e) { return Array.isArray(v) || ArrayBuffer.isView(v) ? v[e] : v; }

  // curve for exchange e, dir 0 = a->b, 1 = b->a (curve starts at the sender)
  curveFor(e, dir) {
    const i = this.blocks[e];
    const d = this.state.depth;
    const pa = this.a.blockSurface(i, this.b.blockWorld(i)).lerp(this.a.blockWorld(i), d), pb = this.b.blockSurface(i, this.a.blockWorld(i)).lerp(this.b.blockWorld(i), d);
    const from = dir === 0 ? pa : pb, to = dir === 0 ? pb : pa;
    const mid = from.clone().lerp(to, 0.5); mid.y += this.state.sag;
    const span = from.distanceTo(to);
    const side = new THREE.Vector3().subVectors(to, from).normalize();
    const n1 = new THREE.Vector3(0, 1, 0).cross(side).normalize(); // horizontal normal
    const n2 = side.clone().cross(n1).normalize();
    const pts = [];
    const tw = this.state.twist, ph0 = dir === 0 ? 0 : Math.PI;
    for (let k = 0; k <= 24; k++) {
      const u = k / 24;
      // quadratic bezier from -> mid -> to
      const p = new THREE.Vector3().addScaledVector(from, (1 - u) * (1 - u)).addScaledVector(mid, 2 * u * (1 - u)).addScaledVector(to, u * u);
      const env = Math.sin(Math.PI * u); // twist is zero at both ports
      const ang = ph0 + u * span * 2.2 + (dir === 0 ? 1 : -1) * 0; // fixed helix (not rotating)
      const fu = dir === 0 ? u : 1 - u; // keep both helices in phase in world space
      const a2 = ph0 + fu * span * 2.2;
      p.addScaledVector(n1, Math.cos(a2) * tw * env).addScaledVector(n2, Math.sin(a2) * tw * env);
      pts.push(p);
    }
    return new THREE.CatmullRomCurve3(pts);
  }

  update(t) {
    const s = this.state;
    for (const st of this.strands) {
      const { e, dir } = st;
      let amp = this._val(s.intensity, e);
      if (s.highlight >= 0 && s.highlight !== e) amp *= s.dim;
      const grow = clamp(this._val(s.grow, e));
      const dirAmp = this._val(dir === 0 ? s.aToB : s.bToA, e);
      const curve = this.curveFor(e, dir); st.curve = curve;
      st.mesh.geometry.dispose();
      st.mesh.geometry = new THREE.TubeGeometry(curve, this.segs, 0.018 + (s.highlight === e ? 0.01 : 0), this.radial, false);
      st.mesh.visible = grow > 0.001 && amp > 0.001;
      st.mat.uniforms.uTime.value = t * s.flow * (dir === 0 ? 1 : 1);
      st.mat.uniforms.uGrow.value = grow;
      st.mat.uniforms.uAmp.value = amp * (0.5 + 0.5 * dirAmp);
      // packets
      const pos = st.pts.geometry.attributes.position.array, col = st.pts.geometry.attributes.color.array;
      const sender = dir === 0 ? C.quillHi : C.emberHi;
      const n = Math.min(st.NP, Math.max(0, Math.round(s.packets)));
      for (let k = 0; k < st.NP; k++) {
        let u = ((t * 0.55 * s.flow) + k / Math.max(1, n) + hash(e * 13 + dir * 7) ) % 1;
        const show = k < n && u < grow && dirAmp > 0.001 && amp > 0.001;
        const p = curve.getPoint(clamp(u));
        pos[k * 3] = p.x; pos[k * 3 + 1] = p.y; pos[k * 3 + 2] = show ? p.z : 1e5;
        const fade = Math.sin(Math.PI * u) ** 0.5 * amp * dirAmp;
        const tint = dir === 0 ? s.tintAB : s.tintBA;
        const c = (tint ? tint.clone() : C.link.clone().lerp(sender, s.colorBySender * (1 - u))).multiplyScalar(fade * 1.6);
        col[k * 3] = c.r; col[k * 3 + 1] = c.g; col[k * 3 + 2] = c.b;
      }
      st.pts.geometry.attributes.position.needsUpdate = true; st.pts.geometry.attributes.color.needsUpdate = true;
      st.pm.size = s.packetSize * (s.highlight === e ? 1.5 : 1);
    }
    // shot-placed pulses
    const pa = this.pulsePts.geometry.attributes.position.array, ca = this.pulsePts.geometry.attributes.color.array;
    const NPU = pa.length / 3;
    for (let k = 0; k < NPU; k++) {
      const pu = s.pulses[k];
      if (!pu) { pa[k * 3 + 2] = 1e5; continue; }
      const st = this.strands.find((x) => x.e === pu.e && x.dir === pu.dir);
      const p = st.curve.getPoint(clamp(pu.u));
      pa[k * 3] = p.x; pa[k * 3 + 1] = p.y; pa[k * 3 + 2] = p.z;
      const c = (pu.color || C.linkHi).clone().multiplyScalar(pu.amount ?? 1.6);
      ca[k * 3] = c.r; ca[k * 3 + 1] = c.g; ca[k * 3 + 2] = c.b;
    }
    this.pulsePts.geometry.attributes.position.needsUpdate = true; this.pulsePts.geometry.attributes.color.needsUpdate = true;
  }
}
