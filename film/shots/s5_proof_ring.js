// S5 helper: the field of 180 held-out episodes as a ring of glowing dots.
// Layout follows figures/transfer.png exactly: two halves (Ember -> Quill = AFM -> Qwen on the left, Quill -> Ember on
// the right), each with the 7 value games in the figure's order and their real episode counts (10,10,10,10,20,20,10).
// The single loss is AFM -> Qwen, "join a name and a city" (19 of 20). Everything is a pure function of t.
import * as THREE from 'three';
import { clamp, hash } from '../lib/util.js';

export const GAMES = [10, 10, 10, 10, 20, 20, 10]; // per direction, figure order
// nl20, full training mix, step 800: the real partner wins all 180 held-out episodes (no grey dot)
export const LOSS = { half: -1, game: -1, k: -1 };

const TEAL = new THREE.Color('#14e3a8'), TEAL_HI = new THREE.Color('#b8ffe9'), GREY = new THREE.Color('#8b8fa3');
const UNLIT = new THREE.Color('#3a3358');

export function makeRing({ pxScale = 1, radius = 4.5 } = {}) {
  const group = new THREE.Group();
  const N = 180;
  const dots = []; // {half, game, k, j (order along half 0..89), ang, grey}
  const endGap = 0.07, gameGap = 0.055;
  const usable = Math.PI - 2 * endGap - 6 * gameGap, step = usable / 90;
  for (let half = 0; half < 2; half++) {
    let a = endGap + step / 2, j = 0;
    GAMES.forEach((n, g) => {
      for (let k = 0; k < n; k++) {
        const grey = half === LOSS.half && g === LOSS.game && k === LOSS.k;
        dots.push({ half, game: g, k, j, ang: a, grey });
        a += step; j++;
      }
      a += gameGap;
    });
  }
  // position: angle measured from the back (-z) toward the front (+z); half 0 on -x, half 1 on +x
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), size = new Float32Array(N);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  const unitPos = dots.map((d) => new THREE.Vector3((d.half === 0 ? -1 : 1) * Math.sin(d.ang), 0, -Math.cos(d.ang)));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uScale: { value: pxScale }, uAmp: { value: 1 } },
    vertexShader: `attribute vec3 aColor; attribute float aSize; uniform float uScale; varying vec3 vC;
      void main(){ vC = aColor; vec4 mv = modelViewMatrix*vec4(position,1.); gl_PointSize = aSize*uScale/(-mv.z); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `varying vec3 vC; uniform float uAmp;
      void main(){ float d = length(gl_PointCoord-0.5)*2.0; if(d>1.0) discard;
        float core = 1.0 - smoothstep(0.30, 0.42, d); float halo = pow(1.0-d, 2.2);
        gl_FragColor = vec4(vC*uAmp*(core*1.0 + halo*0.35), 1.0); }`,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const points = new THREE.Points(geo, mat); group.add(points);

  // a hair-thin guide circle under the dots, broken into the 14 game arcs
  const arcMat = new THREE.LineBasicMaterial({ color: new THREE.Color('#5b4f99'), transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false });
  const arcs = [];
  for (let half = 0; half < 2; half++) {
    for (let g = 0; g < 7; g++) {
      const ds = dots.filter((d) => d.half === half && d.game === g);
      const a0 = ds[0].ang - step * 0.35, a1 = ds[ds.length - 1].ang + step * 0.35;
      const pts = [];
      for (let s = 0; s <= 24; s++) { const a = a0 + (a1 - a0) * s / 24; pts.push(new THREE.Vector3((half === 0 ? -1 : 1) * Math.sin(a), 0, -Math.cos(a)).multiplyScalar(1.075)); }
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), arcMat); group.add(l); arcs.push(l);
    }
  }
  // shockwave annulus in the ring plane
  const shockMat = new THREE.ShaderMaterial({
    uniforms: { uAmp: { value: 0 }, uCol: { value: TEAL.clone() } },
    vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader: `varying vec2 vUv; uniform float uAmp; uniform vec3 uCol; void main(){ float r = length(vUv-0.5)*2.0;
      float a = smoothstep(0.90, 0.985, r)*(1.0 - smoothstep(0.985, 1.0, r)); gl_FragColor = vec4(uCol*a*uAmp, 1.0);} `,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
  const shock = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), shockMat); shock.rotation.x = -Math.PI / 2; group.add(shock);

  // ignition time of each dot: both halves sweep from the back to the front at once
  const T_IGN = 50.0, T_END = 51.45;
  dots.forEach((d) => { const f = d.j / 89; d.t = T_IGN + (T_END - T_IGN) * Math.pow(f, 1.35) + (hash(d.j * 3.1 + d.half) - 0.5) * 0.02; });

  const tmpC = new THREE.Color();
  const ring = {
    group, dots, radius, T_IGN, T_END,
    // number of teal (won) dots lit at time t
    count(t) { let n = 0; for (const d of dots) if (!d.grey && t >= d.t) n++; return n; },
    // world position of the dot (valid after update)
    dotWorld(i, out = new THREE.Vector3()) { return out.copy(unitPos[i]).multiplyScalar(ring.radius).applyMatrix4(group.matrixWorld); },
    lossIndex: dots.findIndex((d) => d.grey),
    // o: {center, radius, yaw, amp (overall brightness), pulse (extra brightness 0..), unlit (visibility of unlit dots), size}
    update(t, o) {
      group.position.copy(o.center); group.rotation.set(o.tilt || 0, o.yaw || 0, 0, 'YXZ');
      ring.radius = o.radius;
      const R = o.radius, sz = o.size ?? 0.13;
      for (let i = 0; i < N; i++) {
        const d = dots[i], u = unitPos[i];
        const age = t - d.t;
        // gentle shimmer along the ring once lit
        const shimmer = 0.85 + 0.15 * Math.sin(t * 3.0 - d.ang * 6 + d.half * 1.7);
        let s, c;
        if (age < 0) { // unlit placeholder
          s = 0.55; c = tmpC.copy(UNLIT).multiplyScalar(o.unlit ?? 0.6);
        } else if (d.grey) {
          s = 1.3; c = tmpC.copy(GREY).multiplyScalar(1.1 * clamp(age / 0.15));
        } else {
          const fl = Math.exp(-age * 5.0);
          s = 1 + 0.9 * fl * clamp(age / 0.04);
          c = tmpC.copy(TEAL).lerp(TEAL_HI, fl).multiplyScalar((0.8 + 1.4 * fl) * shimmer * (1 + (o.pulse || 0)));
        }
        const yb = (o.wave || 0) * Math.sin(d.ang * 3 - t * 4) * 0.06;
        pos[i * 3] = u.x * R; pos[i * 3 + 1] = yb; pos[i * 3 + 2] = u.z * R;
        size[i] = s * sz * (o.sizeMul ?? 1);
        col[i * 3] = c.r * o.amp; col[i * 3 + 1] = c.g * o.amp; col[i * 3 + 2] = c.b * o.amp;
      }
      geo.attributes.position.needsUpdate = true; geo.attributes.aColor.needsUpdate = true; geo.attributes.aSize.needsUpdate = true;
      for (const l of arcs) l.scale.setScalar(R);
      arcMat.opacity = (o.arcs ?? 0) * o.amp;
      // shock
      const sa = o.shock || 0;
      shock.visible = sa > 0.002; shock.scale.setScalar(o.shockR || R); shockMat.uniforms.uAmp.value = sa;
      group.visible = o.amp > 0.002;
      group.updateMatrixWorld(true);
    },
  };
  return ring;
}
