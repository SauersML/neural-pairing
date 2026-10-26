// a2: the neck that joins a head to the newest column of its lattice. It is built like the creature's body: the same
// glowing residual-stream core as the lattice columns (same colour, same flowing stripes) plus a few vertebra pucks and
// rings. It leaves the top of a column going straight up (tangent-continuous with the column) and curves into the
// base of the head (a cubic Bezier). `g1` draws only the part from the column up to that fraction (to grow or
// retract); `amp` fades it. Pure: the shot calls set() with values computed from t every frame.
import * as THREE from 'three';
import { clamp, lerp } from '../lib/util.js';

const NV = 4; // vertebrae per neck
export class Neck {
  constructor(scene, col, hi) {
    this.col = col; this.hi = hi;
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uCol: { value: col.clone().lerp(hi, 0.45) }, uAmp: { value: 1 }, uTime: { value: 0 }, uG1: { value: 1 }, uLen: { value: 1 } },
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `uniform vec3 uCol; uniform float uAmp, uTime, uG1, uLen; varying vec2 vUv;
        void main(){ if (vUv.x > uG1) discard;
          float s = vUv.x * uLen;                                   // arc length from the column top
          float flow = 0.72 + 0.28*sin(s*26.0 - uTime*7.0);
          float tip = smoothstep(uG1 - 0.08, uG1, vUv.x) * step(uG1, 0.995);
          gl_FragColor = vec4(uCol * flow * (0.62 + 1.4*tip) * uAmp, 1.0); }`,
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this.core = new THREE.Mesh(new THREE.BufferGeometry(), this.mat); this.core.renderOrder = 2; this.core.frustumCulled = false; scene.add(this.core);
    const add = (geo, color) => { const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: color.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); m.renderOrder = 1; scene.add(m); return m; };
    const puckG = new THREE.CylinderGeometry(1, 1, 1, 28, 1, false), ringG = new THREE.TorusGeometry(1, 0.05, 6, 40);
    this.pucks = Array.from({ length: NV }, () => add(puckG, col));
    this.rings = Array.from({ length: NV }, () => add(ringG, hi));
    this._q = new THREE.Quaternion(); this._up = new THREE.Vector3(0, 1, 0); this._rx = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  }
  hide() { this.core.visible = false; for (const m of [...this.pucks, ...this.rings]) m.visible = false; }
  // curve: THREE.Curve from the column top to the head base. r0/r1: puck radius at the column / at the head.
  set(curve, g1, amp, t, r0 = 0.15, r1 = 0.1) {
    if (amp <= 0.004 || g1 <= 0.01) { this.hide(); return; }
    const len = curve.getLength();
    this.core.geometry.dispose();
    this.core.geometry = new THREE.TubeGeometry(curve, 40, 0.032, 10, false);
    this.core.visible = true;
    const U = this.mat.uniforms; U.uAmp.value = amp; U.uTime.value = t; U.uG1.value = clamp(g1); U.uLen.value = len;
    for (let i = 0; i < NV; i++) {
      const u = (i + 0.6) / (NV + 0.4), on = clamp((g1 - u) / 0.06) * amp;
      const p = this.pucks[i], r = this.rings[i];
      p.visible = r.visible = on > 0.004; if (!p.visible) continue;
      const pos = curve.getPointAt(u), tan = curve.getTangentAt(u).normalize();
      this._q.setFromUnitVectors(this._up, tan);
      const rad = lerp(r0, r1, u) * (0.6 + 0.4 * on);
      p.position.copy(pos); p.quaternion.copy(this._q); p.scale.set(rad, 4.2 / 36 * 0.34, rad);
      r.position.copy(pos); r.quaternion.copy(this._q).multiply(this._rx); r.scale.set(rad, rad, rad * 1.6);
      p.material.color.copy(this.col).multiplyScalar(0.12 * on); r.material.color.copy(this.hi).multiplyScalar(0.3 * on);
    }
  }
}
