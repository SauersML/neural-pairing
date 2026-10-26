import * as THREE from 'three';
import { Creature, XBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { C } from '../lib/palette.js';
const GOLD = new THREE.Color('#ffd166');
export async function create() {
  const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(32, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember'); scene.add(q.group, e.group); q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  return { scene, camera, update(t) {
    camera.position.set(0.6, 3.0, 11); camera.lookAt(0, 2.3, 0); world.update(t);
    const wave = (t % 2) / 1.6; q.pose.wave = e.pose.wave = wave;
    q.pose.xray = e.pose.xray = 0.6;
    e.inject = []; q.inject = [];
    XBLOCKS.forEach((b, k) => { const u = (b + 0.5) / 36; if (u < wave) { (k % 2 ? q : e).inject.push({ block: b, color: k % 2 ? C.ember : C.quill, amount: 0.9, top: Math.min(1.01, wave) }); } });
    q.update(t); e.update(t);
    link.state.intensity = XBLOCKS.map((b) => Math.exp(-Math.pow(((b + 0.5) / 36 - wave) * 10, 2)) + 0.15);
    link.update(t);
    return {};
  } };
}
