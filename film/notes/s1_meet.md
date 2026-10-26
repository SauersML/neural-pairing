# S1 `s1_meet` (0–10 s) · "01 / THE PAIR"

Files: `shots/s1_meet.js` (the shot) and `shots/s1_meet_fx.js` (hex shield shader, tile shatter, soft particles,
leader-line callouts, time-keyed Catmull-Rom camera spline). Captions follow REVISION 2 word for word. No chapter tag: the `Chapter` call was removed, and the shot has no numbered tags or links.
Final contact sheet: `out/preview/s1/final_sheet.jpg` (times 0, 0.3, 1.5, 2.5, 3.3, 5, 6.2, 7.4, 8.05, 8.6, 9.8, 9.95).

## Second by second
- **0.0–0.8** Fade up from black (the frame at 0.0 is pure black). A violet spark ignites at Quill's base (0.2) and an
  orange one at Ember's (0.6). Each has a flare, a burst of 48 embers and a rim-light swell. Dust, sky, floor grid and
  lights ramp up.
- **0.4–3.2** Vertebrae (= blocks) fly in glowing from the dark and stack bottom-up with the lib's overshoot
  (Quill 0.4→2.8, Ember 0.8→3.2). Each block flashes as it lands. The residual-stream core grows block by block
  through *landed* blocks only, and the spark rides the top of the growing spine. Heads pop on with an extra
  squash-and-stretch wobble. Camera: a slow push-in and rise.
  Caption 0.5–3.6: `neural pairing` / `Two language models` / `Qwen3-4B from Alibaba and AFM-4.5B from Arcee`
  (model names tinted violet and orange).
- **2.95 / 3.15** Eyes open (closed until then), with a blink. The creatures look at the camera, then at each other at ~3.6.
- **3.0–5.6** Name callouts with leader lines: `Quill` / `Qwen3-4B` (violet, left) and `Ember` / `AFM-4.5B` (orange, right).
- **3.7–4.9** Prompt tiles in each model's own token style float beside the tails (Quill square tiles: `The` `sky` `is`;
  Ember pills: `Once` `upon` `a`) and are swallowed one by one into the tail, each with a gulp (squash dip and tail flash).
  The creatures look down at them.
  Caption 3.6–6.6: `36 blocks each` / `Each block adds its output to the residual stream.`
- **4.25–6.65** A block ruler (36 ticks) draws up beside Quill. A `block N` counter (32 px) rides the wave from `block 0`
  to `block 35`.
- **4.9–5.8 (Quill) / 5.0–5.9 (Ember)** A glow wave climbs the blocks bottom to top. At each block, sparks fly from
  the block into the core, and a bright fill climbs the core behind the wave front: each block adds its output to the
  residual stream. Callout `residual stream` (32 px) on Quill's core from 4.85.
- **5.86 / 5.96** The next token pops out of the head with a spring (`blue` from Quill, `time` from Ember). It fades by 6.75.
- **6.6–7.06** The camera swings to a front-left 3/4 view. Quill grins (happy eyes) and writes a gold `lantern` tile out of
  its head, then winds up (lean back, crouch).
  Caption 6.6–10.0: `Only hidden states cross the link` (head only; "hidden states" in link teal).
- **7.06–8.0** Quill flicks its head and the tile flies in a lob toward Ember with a sparkle trail. Ember leans in
  happily to catch it. At 7.62 a curved hex force field flashes up cell by cell, starting from the impact point.
- **8.0 (hit)** The tile hits the field and shatters into 54 textured shards. Each shard carries its piece of the word,
  sprays back toward Quill and dissolves into about 460 gold dust motes, alongside about 110 white-gold sparks. The
  field dents, ripples and sends a bright ring across its cells. There is also a short warm flash, a bloom and chromatic
  aberration kick, and a camera shake. Ember flinches: eyes squeezed shut, lean away, squash. Quill is startled.
- **8.5–9.5** Both look at each other with puzzled head tilts, and the field's cells blink out one by one (8.75–9.55).
  The dust fades by 9.8.
- **8.95–9.5** The 8 port rings on both bodies ignite bottom to top, each with a flash, then pulse on the beat at
  ~1.2. Quill glances down at its ports at 9.15, Ember at 9.25, and then they look back at each other.
- **9.95 / end state** Home poses (x = ∓2.3), fully assembled, ports glowing ~1.2, no link, the shield and dust gone.
  The caption is fading out. Camera at `(-0.6, 3.2, 14.8)` looking at `(-0.7, 2.5, 0)` and still drifting.

Framing: because the captions are now very large, the camera keeps the creatures' feet above the caption block
(feet at y ≈ 740 px while the captions sit lower) and frames the pair a little right of center, so no text sits over
bright geometry. All on-screen text is at least 30 px. 3D token tiles are 0.56–0.62 world units tall, so their glyphs
render above 30 px.

## Lib issues worked around (lib untouched)
- `Creature.update` builds the core tube through `_centers` of *all* blocks up to `nA`, including blocks still in
  flight, so during assembly the core whips out to flying vertebrae. While `assemble < 1`, I rebuild `creature.core.geometry`
  after `update()` through rest-spine points of landed blocks only, using `_spineLocal`.
- The head pop and port flashes are applied after `update()` by multiplying `head.scale` and `port.mesh.scale` and
  setting each `port.mat.opacity` (the lib exposes only one `portGlow` for all 8 ports).
- Gaze targets for the partner use a fixed approximate head position, never the partner's `headWorld()` from a
  previous frame, so the result does not depend on update order.
- The tail sprite opacity is scaled down after the prompt section so the bright tail glow does not compete later.

Purity: rendering t = 8.5 and t = 5.6 fresh and after other times gives max pixel difference 1/255 (float noise).
Render cost at full res: about 1.1–1.8 s per frame.
