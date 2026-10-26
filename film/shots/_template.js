// Reference shot: copy this structure. create() builds once; update(t) is a pure function of global time t.
import * as THREE from 'three';
import { Creature, XBLOCKS } from '../lib/creature.js';
import { Link } from '../lib/link.js';
import { makeWorld } from '../lib/env.js';
import { Caption, Chapter, Pin } from '../lib/hud.js';
import { prog, spring, kick, win, beatPulse, smooth } from '../lib/util.js';

export async function create({ hud, def }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene);
  const q = new Creature('quill'), e = new Creature('ember');
  scene.add(q.group, e.group);
  q.camera = e.camera = camera;
  const link = new Link(q, e); scene.add(link.group);
  const chap = new Chapter(hud, '00', 'Template');
  const cap = new Caption(hud, { kicker: 'template', head: 'Two models. <span style="color:#34f5c0">One link.</span>', sub: 'No tokens cross.' });
  const pinQ = new Pin(hud, 'QUILL · Qwen3-4B', { color: '#c4b5fd' });
  const T0 = def.start;
  return {
    scene, camera,
    update(t) {
      const lt = t - T0;
      camera.position.set(Math.sin(lt * 0.2) * 2, 3.2, 11); camera.lookAt(0, 2.6, 0);
      world.update(t);
      q.pose.swayPhase = t * 2.4; q.pose.sway = 0.12; q.pose.lookAt = e.headWorld();
      e.pose.swayPhase = t * 2.1 + 1; e.pose.sway = 0.12; e.pose.lookAt = q.headWorld();
      q.pose.blink = win(t, T0 + 2, T0 + 2.18, 0.08); e.pose.happy = 1;
      q.update(t); e.update(t); link.update(t);
      chap.update(t, T0, def.end); cap.update(t, T0 + 0.3, def.end - 0.2);
      pinQ.update(q.headWorld().add(new THREE.Vector3(-0.9, 0.9, 0)), camera, 1);
      return { bloomStrength: 0.9 };
    },
  };
}
