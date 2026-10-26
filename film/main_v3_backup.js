import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { SHOTS } from './shots/index.js';

// The film runtime. window.renderAt(t) draws the frame at global time t (seconds). Pure: any t, any order.
const W = 1920, H = 1080;
const params = new URLSearchParams(location.search);
const scale = parseFloat(params.get('scale') || '1');
const only = params.get('shots') ? params.get('shots').split(',') : null;

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(scale); renderer.setSize(W, H, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.getElementById('stage').prepend(renderer.domElement);
const hud = document.getElementById('hud');

const composer = new EffectComposer(renderer);
composer.setPixelRatio(scale); composer.setSize(W, H);
const renderPass = new RenderPass(new THREE.Scene(), new THREE.PerspectiveCamera());
composer.addPass(renderPass);
const bloom = new UnrealBloomPass(new THREE.Vector2(W * scale, H * scale), 0.9, 0.55, 0.18);
composer.addPass(bloom);
composer.addPass(new OutputPass());
const finalPass = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uVig: { value: 0.9 }, uGrain: { value: 0.05 }, uCA: { value: 0.0018 }, uFade: { value: 0 }, uFlash: { value: 0 }, uFlashCol: { value: new THREE.Color(1, 1, 1) } },
  vertexShader: `varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uTime,uVig,uGrain,uCA,uFade,uFlash; uniform vec3 uFlashCol; varying vec2 vUv;
    float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);} 
    void main(){ vec2 d = vUv-0.5; float r2 = dot(d,d);
      vec2 o = d*uCA*(1.0+r2*4.0);
      vec3 c = vec3(texture2D(tDiffuse, vUv+o).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv-o).b);
      c *= mix(1.0, smoothstep(0.95, 0.15, r2*2.2), uVig*0.55);
      c += (h(vUv*1000.0 + fract(uTime*7.31))-0.5)*uGrain;
      c = mix(c, uFlashCol, uFlash); c *= (1.0-uFade);
      gl_FragColor = vec4(c,1.0);} `,
});
composer.addPass(finalPass);

const POST_DEFAULT = { bloomStrength: 0.9, bloomRadius: 0.55, bloomThreshold: 0.18, exposure: 1.0, vignette: 0.9, grain: 0.045, aberration: 0.0018, fade: 0, flash: 0, flashColor: null };
const live = new Map(); // id -> instance
const ctx = { THREE, renderer, hud, W, H, scale };

async function getShot(def) {
  if (live.has(def.id)) return live.get(def.id);
  const mod = await import(`./shots/${def.id}.js`);
  const root = document.createElement('div'); root.style.cssText = 'position:absolute;inset:0;display:none'; hud.appendChild(root);
  const inst = await mod.create({ ...ctx, hud: root, def });
  inst._root = root; live.set(def.id, inst); return inst;
}

const defs = params.get('test') ? [{ id: params.get('test'), start: 0, end: 1000 }] : SHOTS.filter((s) => !only || only.includes(s.id));
window.renderAt = async (t) => {
  let def = defs.find((s) => t >= s.start && t < s.end) || (t >= defs[defs.length - 1].end ? defs[defs.length - 1] : defs[0]);
  const inst = await getShot(def);
  for (const [id, s] of live) s._root.style.display = id === def.id ? 'block' : 'none';
  const post = { ...POST_DEFAULT, ...(inst.update(t) || {}) };
  renderPass.scene = inst.scene; renderPass.camera = inst.camera;
  inst.camera.aspect = W / H; inst.camera.updateProjectionMatrix();
  bloom.strength = post.bloomStrength; bloom.radius = post.bloomRadius; bloom.threshold = post.bloomThreshold;
  renderer.toneMappingExposure = post.exposure;
  const u = finalPass.uniforms; u.uTime.value = t; u.uVig.value = post.vignette; u.uGrain.value = post.grain; u.uCA.value = post.aberration; u.uFade.value = post.fade; u.uFlash.value = post.flash;
  if (post.flashColor) u.uFlashCol.value.set(post.flashColor); else u.uFlashCol.value.setRGB(1, 1, 1);
  composer.render();
  await document.fonts.ready;
  return def.id;
};
window.FILM = { SHOTS, duration: SHOTS[SHOTS.length - 1].end };
window.filmReady = true;
