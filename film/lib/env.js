import * as THREE from 'three';
import { C } from './palette.js';
import { Creature } from './creature.js';
import { hash } from './util.js';

// The shared world: deep violet-black void, a soft nebula gradient, drifting dust, a dark glossy floor with a faint grid,
// and the standard three-light rig (cool key, violet rim from the left, orange rim from the right).
export function makeWorld(scene, o = {}) {
  const w = { };
  scene.background = C.bg0.clone();
  if (o.fog !== false) scene.fog = new THREE.FogExp2(0x05040a, o.fogDensity ?? 0.035);
  // sky dome
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { uTime: { value: 0 }, uA: { value: new THREE.Color('#120c2a') }, uB: { value: new THREE.Color('#030208') }, uQ: { value: C.quillDeep.clone() }, uE: { value: C.emberDeep.clone() }, uAmp: { value: o.skyAmp ?? 1 } },
    vertexShader: `varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uTime; uniform vec3 uA,uB,uQ,uE; uniform float uAmp; varying vec3 vP;
      float h(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);} 
      float n(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z);} 
      void main(){ float y = vP.y*0.5+0.5; vec3 col = mix(uB, uA, smoothstep(0.35, 0.75, y));
        float neb = n(vP*3.0 + vec3(0., uTime*0.02, 0.)) * n(vP*7.0 - vec3(uTime*0.03));
        col += uQ * 0.55 * smoothstep(0.2, 0.9, neb) * smoothstep(0.2, -0.8, vP.x);
        col += uE * 0.45 * smoothstep(0.2, 0.9, neb) * smoothstep(-0.2, 0.8, vP.x);
        gl_FragColor = vec4(col*uAmp, 1.0); }`,
  });
  w.sky = new THREE.Mesh(new THREE.SphereGeometry(80, 48, 24), skyMat); scene.add(w.sky);
  // floor
  if (o.floor !== false) {
    const floorMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uTime: { value: 0 }, uCol: { value: C.link.clone() }, uAmp: { value: o.gridAmp ?? 1 } },
      vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
      fragmentShader: `uniform float uTime; uniform vec3 uCol; uniform float uAmp; varying vec3 vW;
        float line(float x, float w){ float f = abs(fract(x)-0.5); return smoothstep(w, 0.0, 0.5-f); }
        void main(){ float r = length(vW.xz); float fade = smoothstep(16.0, 2.0, r);
          float g = max(line(vW.x*1.0, 0.025), line(vW.z*1.0, 0.025));
          float ring = smoothstep(0.03, 0.0, abs(fract(r*0.25 - uTime*0.05)-0.5)-0.47);
          vec3 col = vec3(0.02,0.015,0.04) + uCol*(g*0.10 + ring*0.04)*uAmp;
          gl_FragColor = vec4(col, 0.92*fade); }`,
    });
    w.floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), floorMat); w.floor.rotation.x = -Math.PI / 2; scene.add(w.floor);
  }
  // dust
  const N = o.dust ?? 900; const pos = new Float32Array(N * 3); w.dustBase = [];
  for (let i = 0; i < N; i++) { w.dustBase.push([(hash(i) - 0.5) * 30, hash(i * 2.1) * 14 - 2, (hash(i * 3.7) - 0.5) * 30, hash(i * 5.3)]); }
  const dg = new THREE.BufferGeometry(); dg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  w.dustMat = new THREE.PointsMaterial({ size: 0.05, map: Creature.dotTexture(), color: new THREE.Color('#b9a8ff'), transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false });
  w.dust = new THREE.Points(dg, w.dustMat); w.dust.frustumCulled = false; scene.add(w.dust);
  // lights
  w.hemi = new THREE.HemisphereLight(0x9b8cff, 0x120a08, 0.5); scene.add(w.hemi);
  w.key = new THREE.DirectionalLight(0xdfe6ff, 1.6); w.key.position.set(2, 6, 8); scene.add(w.key);
  w.rimQ = new THREE.PointLight(C.quill, 30, 30, 2); w.rimQ.position.set(-6, 4, -3); scene.add(w.rimQ);
  w.rimE = new THREE.PointLight(C.ember, 30, 30, 2); w.rimE.position.set(6, 4, -3); scene.add(w.rimE);
  w.update = (t) => {
    skyMat.uniforms.uTime.value = t; if (w.floor) w.floor.material.uniforms.uTime.value = t;
    const a = w.dust.geometry.attributes.position.array;
    for (let i = 0; i < N; i++) { const b = w.dustBase[i]; a[i * 3] = b[0] + Math.sin(t * 0.2 + b[3] * 6) * 0.4; a[i * 3 + 1] = ((b[1] + t * 0.12 * (0.3 + b[3]) + 2) % 14) - 2; a[i * 3 + 2] = b[2] + Math.cos(t * 0.15 + b[3] * 9) * 0.4; }
    w.dust.geometry.attributes.position.needsUpdate = true;
  };
  return w;
}
