import * as THREE from 'three';
import { makeWorld } from '../lib/env.js';
import { ModelSheet, LinkBeams, ROWS, XROWS } from '../lib/grid.js';
import { C } from '../lib/palette.js';
const GOLD = new THREE.Color('#ffd166');
export async function create() {
  const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.1, 200);
  const world = makeWorld(scene, { floor: false });
  const qt = ['lantern', '▸', 'Sky', 'looks', 'blue', 'because'].map((t) => ({ text: t }));
  const et = ['word?', '▸', 'The', 'word', 'is', 'lantern'].map((t) => ({ text: t }));
  const q = new ModelSheet('quill', { tokens: qt }), e = new ModelSheet('ember', { tokens: et, outputs: [null, 'The', 'word', 'is', 'lantern', '.'] });
  q.group.position.set(-4.6, 0, 0); e.group.position.set(1.4, 0, 0);
  scene.add(q.group, e.group); const beams = new LinkBeams(scene);
  return { scene, camera, update(t) {
    camera.position.set(1.5, 3.2, 11.5); camera.lookAt(0, 2.4, 0); world.update(t);
    q.begin(); e.begin(); beams.begin();
    q.addInfo(0, 0, 36, GOLD, 0.8);
    for (let p = 1; p < 6; p++) q.arc(10, 0, p, 0.5, GOLD, t);
    for (const r of XROWS) for (let s = 0; s <= 4; s++) beams.beam(q, s, e, 4, r, s === 0 ? 0.9 : 0.15, t, s === 0 ? GOLD : null, 0, 0.8);
    e.addInfo(4, 7, 36, GOLD, 0.7);
    q.end(t); e.end(t); beams.end(t);
    return {};
  } };
}
