// Hair test: t 0-1 linked (lift 1), 1-2 flare (1.35), 2-3 cut (lift 0); camera close on the heads.
import * as THREE from 'three';
import { Creature } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
export async function create() {
  const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene); const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group); q.camera = e.camera = camera; const link = new Link(q, e); scene.add(link.group);
  return { scene, camera, update(t) {
    const side = Math.floor(t / 3) % 2;  // 0 front, 1 three-quarter
    camera.position.set(side ? 4 : 0, 5.6, side ? 6.5 : 8); camera.lookAt(0, 4.9, 0); world.update(t);
    const lift = t % 3 < 1 ? 1 : t % 3 < 2 ? 1.35 : 0;
    for (const [c, o] of [[q, e], [e, q]]) { c.pose.hair = lift; c.pose.lookAt = o.headWorld(); c.pose.xray = 0.6; c.pose.crest = lift ? 1 : 0.45; c.pose.happy = lift ? 0.8 : 0; c.pose.blink = lift ? 0 : 0.58; }
    q.update(t); e.update(t); link.update(t); return { bloomStrength: 0.8, bloomThreshold: 0.4 };
  } };
}
