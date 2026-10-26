// The games, as data for the 3D game windows (shots/games_win3d.js), checked against pairing/games.py, and the a6
// result windows. Values are from the held-out split (data/pools.json "test" parts; numbers and digit strings with
// digit sum 3 mod 7; sums 50-99 per number). Token splits are the two models' real tokenizers (Qwen3-4B for Quill,
// AFM-4.5B for Ember), e.g. Quill writes 5203 as 5|2|0|3 and Ember as 520|3.
//   relay: the receiver writes "The secret word is lantern." etc. (the sender meanwhile writes its own reply to another
//          question, which the windows leave out to keep them clean; both bodies still run every pass in lockstep).
//   merge_sum: both write "Mine is X, <partner>'s is Y, and the sum is Z."     merge_fact: both write "Keiko lives in Geneva."
//   lookup: the asker has the code, the holder the table (8 rows); both write "Code M2 holds the word breeze."
//   shared reading (results/shared_reading.json, sciq test example): passage on one side, question on the other.
//   free chat (results/hero_grid.json, nl22c episode 4242:co_chat:15): Ember's private city is Taipei; Quill opens
//          "Hey Ember, how's the weather in Taipei? I hope you're keeping warm."
import * as THREE from 'three';
import { Diorama, Pane } from './games_win3d.js';
import { clamp, prog, spring, kick, lerp, easeInCubic, easeOutCubic } from '../lib/util.js';

const S = (text) => ({ text, kind: 'secret' }), Q = { text: '?', kind: 'ghost' }, P = (text) => ({ text, kind: 'plain' }), BR = { br: true };

export const GAMES = {
  word: { cs: 0.95, title: 'relay a word', holder: 'quill', inputs: { quill: [S('lantern')], ember: [Q] }, step: 0.125,
    write: [{ side: 'ember', tokens: ['The', ' secret', ' word', ' is', ' lantern', '.'], gold: [4] }] },
  number: { cs: 0.95, title: 'relay a number', holder: 'ember', inputs: { ember: [S('5203')], quill: [Q] }, step: 0.125,
    write: [{ side: 'quill', tokens: ['The', ' number', ' is', ' ', '5', '2', '0', '3', '.'], gold: [4, 5, 6, 7] }] },
  phrase: { cs: 0.95, title: 'relay a phrase', holder: 'ember', inputs: { ember: [S('silver whale')], quill: [Q] }, step: 0.125,
    write: [{ side: 'quill', tokens: ['The', ' phrase', ' is', ' "', 'silver', ' whale', '".'], gold: [4, 5] }] },
  digits: { cs: 0.95, title: 'relay digits', holder: 'quill', inputs: { quill: [S('7 0 1 9 7')], ember: [Q] }, step: 0.125,
    write: [{ side: 'ember', tokens: ['The', ' digits', ' are', ' ', '7', ' ', '0', ' ', '1', ' ', '9', ' ', '7', '.'], gold: [4, 6, 8, 10, 12], breaks: [3] }] },
  sum: { title: 'add two numbers', holder: 'both', inputs: { quill: [S('58')], ember: [S('67')] }, step: 0.125,
    write: [{ side: 'quill', tokens: ['Mine', ' is', ' ', '5', '8', ',', ' Ember', "'s", ' is', ' ', '6', '7', ',', ' and', ' the', ' sum', ' is', ' ', '1', '2', '5', '.'], gold: [3, 4, 10, 11, 18, 19, 20] },
      { side: 'ember', tokens: ['Mine', ' is', ' ', '67', ',', ' Qu', 'ill', "'s", ' is', ' ', '58', ',', ' and', ' the', ' sum', ' is', ' ', '125', '.'], gold: [3, 10, 17] }] },
  fact: { title: 'name and city', holder: 'both', inputs: { quill: [S('Keiko')], ember: [S('Geneva')] }, step: 0.125,
    write: [{ side: 'quill', tokens: ['Ke', 'iko', ' lives', ' in', ' Geneva', '.'], gold: [0, 1, 4] },
      { side: 'ember', tokens: ['Ke', 'iko', ' lives', ' in', ' Geneva', '.'], gold: [0, 1, 4] }] },
  lookup: { title: 'code lookup', holder: 'both', pin: { quill: 'left', ember: 'right' }, step: 0.125,
    inputs: { quill: [S('M2')], ember: [S('K7 alley'), BR, S('M2 breeze')] },
    write: [{ side: 'quill', tokens: ['Code', ' M', '2', ' holds', ' the', ' word', ' breeze', '.'], gold: [1, 2, 6] },
      { side: 'ember', tokens: ['Code', ' M', '2', ' holds', ' the', ' word', ' breeze', '.'], gold: [1, 2, 6] }] },
  reading: { cs: 0.95, title: 'shared reading', holder: 'quill', pin: { quill: 'left', ember: 'right' }, step: 0.125,
    inputs: { quill: [S('…dormant,'), BR, S('or extinct.')], ember: [P('…dormant,'), BR, P('or what else?')] },
    write: [{ side: 'ember', tokens: ['Ext', 'inct', '.'], gold: [0, 1] }] },
  chat: { cs: 0.68, title: 'free chat', holder: 'ember', inputs: { ember: [S('Taipei')] }, step: 0.125,
    // the opening line up to its question mark (a real prefix of the real line), so the hold shows the whole question
    write: [{ side: 'quill', tokens: ['Hey', ' Ember', ',', ' how', "'s", ' the', ' weather', ' in', ' Taipei', '?'], gold: [8], box: { lines: 4 } }] },
};

// ---------------------------------------------------------------- a6: four result windows, camera-locked in a6's scene
// The real never-trained lookup episode (results/games_untrained.json, nl21 episode 36): Ember asks for G7, Quill holds
// the table (G7 nome, N6 weight, F8 hypocrisy, H7 buzz, F2 brewing, S0 honey, B5 tasted, F1 dagger); both write
// "Code G7 holds the word nome.". Shared pen (pairing/games.py pen_secret, one of the three untrained turn formats,
// 128/128 each in nl16 turnbench): words alternate, the receiver writes the final private word (vortex, a turnbench item).
const PEN = ['Together', ' we', ' follow', ' the', ' little', ' lantern', ' toward', ' vortex'];
export const RESULTS = {
  lookup: { title: 'code lookup', holder: 'both', pin: { quill: 'left', ember: 'right' }, step: 0.125, cs: 0.48,
    inputs: { ember: [S('G7')], quill: [S('H7 buzz'), BR, S('G7 nome')] },
    // fixed line breaks (Code G7 / holds the / word nome.): at the hold both show "holds the / word nome."
    write: [{ side: 'quill', tokens: ['Code', ' G', '7', ' holds', ' the', ' word', ' nome', '.'], gold: [1, 2, 6], breaks: [3, 5], box: { lines: 2 } },
      { side: 'ember', tokens: ['Code', ' G', '7', ' holds', ' the', ' word', ' nome', '.'], gold: [1, 2, 6], breaks: [3, 5], box: { lines: 2 } }],
    footer: [[['35', '#34f5c0', 750, 92], [' / 40', 'rgba(245,243,255,0.62)', 600, 64]], 'never trained'] },
  formats: { title: 'shared pen', holder: 'quill', inputs: { quill: [S('vortex')], ember: [Q] }, step: 0.125, cs: 0.6,
    write: [{ side: 'ember', tokens: PEN, sides: PEN.map((_, i) => (i % 2 ? 'ember' : 'quill')), gold: [7], box: { lines: 3, align: 'center' } }],
    footer: [[['128', '#34f5c0', 750, 92], [' / 128', 'rgba(245,243,255,0.62)', 600, 64]], 'never trained'] },
  reading: { ...GAMES.reading, cs: 0.6, inputs: { quill: [S('…or extinct.'), BR], ember: [P('…what else?')] },
    footer: [[['410', '#34f5c0', 750, 92], [' / 460', 'rgba(245,243,255,0.62)', 600, 64]]] },
  chat: { title: 'free chat', holder: 'ember', inputs: { ember: [S('Taipei')] }, step: 0.125, cs: 0.5,
    write: [{ side: 'quill', tokens: ['Hey', ' Ember', ',', ' how', "'s", ' the', ' weather', ' in', ' Taipei', '?'], gold: [8], box: { lines: 4, align: 'center' } }],
    footer: [[['13', '#34f5c0', 750, 92], [' / 23', 'rgba(245,243,255,0.62)', 600, 64]], [['cut ', 'rgba(245,243,255,0.62)', 600, 36], ['1', '#f5f3ff', 700, 36], ['   stranger ', 'rgba(245,243,255,0.62)', 600, 36], ['0', '#f5f3ff', 700, 36]]] },
};

// footer rows from the spec: [statRuns, caption, extra?]
function footerRows(f) {
  const rows = [{ runs: f[0], px: 92, h: 104, align: 'left' }];
  for (const r of f.slice(1)) rows.push({ runs: typeof r === 'string' ? [[r, 'rgba(245,243,255,0.86)', 600, 38]] : r, px: 38, h: 52, align: 'left' });
  return rows;
}

// makeResultWindows({renderer, scene, camera, scale}).update(t, hits, tOut): window i slams in on hits[i] (a6 local
// time); all leave from tOut. The windows are 3D frames held in front of a6's camera.
export function makeResultWindows({ renderer, scene, camera, scale = 1 }, { top = 50, gap = 26 } = {}) {
  const keys = ['lookup', 'formats', 'reading', 'chat'];
  const PW = 418, PH = 650, HDR = 76, FTR = 166, BD = 12;
  const DIST = 6, ppu = 1080 / (2 * DIST * Math.tan((35 / 2) * Math.PI / 180));
  const root = new THREE.Group(); scene.add(root);
  const W = PW + 2 * BD, H = PH + HDR + FTR + 2 * BD;
  const x0 = -(4 * W + 3 * gap) / 2 + W / 2;
  const wins = keys.map((k, i) => {
    const spec = RESULTS[k];
    const d = new Diorama(renderer, spec, { w: PW, h: PH, scale, ppu: 66, tileH: 0.8 });
    const pane = new Pane({ w: PW, h: PH, header: HDR, footer: FTR, border: BD, title: spec.title, titlePx: 46, tex: d.rt.texture, ppu, footerRows: footerRows(spec.footer) });
    root.add(pane.group);
    const cx = x0 + i * (W + gap), cy = 540 - top - H / 2;
    return { d, pane, spec, cx, cy };
  });
  return {
    update(t, hits, tOut) {
      tOut = Math.round(tOut * 8) / 8; // leave on the 16th grid
      const vis = t >= hits[0] - 0.25 && t < tOut + 0.6;
      root.visible = vis; if (!vis) return;
      camera.updateMatrixWorld(true);
      root.position.copy(camera.position); root.quaternion.copy(camera.quaternion);
      wins.forEach((w, i) => {
        const s0 = hits[i];
        const T = w.T || (w.T = Diorama.timeline(w.spec, s0, { gen: 0.5 }));
        if (t < s0 - 0.2) { w.pane.group.visible = false; return; }
        const pIn = easeInCubic(clamp((t - (s0 - 0.2)) / 0.2));
        const out = easeInCubic(clamp((t - tOut) / 0.3));
        const imp = t >= s0 ? kick(t, s0, 3.0, 7) : 0;
        const g = w.pane.group;
        g.position.set((w.cx) / ppu, (w.cy + out * 120) / ppu - imp * 0.012, -DIST + (1 - pIn) * 1.6 + out * -0.8);
        g.rotation.set((1 - pIn) * -0.5 + out * 0.25, (1 - pIn) * (i - 1.5) * -0.12, 0);
        const sc = 1 / ppu * (1 + 0.03 * imp);
        g.scale.setScalar(sc);
        const hit = Math.max(t >= s0 ? Math.exp(-(t - s0) * 4) : 0, ...w.T.check.map((c) => (t >= c ? 0.7 * Math.exp(-(t - c) * 5) : 0)));
        w.pane.set(clamp((t - (s0 - 0.2)) / 0.1) * (1 - out), hit);
        if (g.visible) w.d.render(Math.min(t, tOut), T);
      });
    },
  };
}
