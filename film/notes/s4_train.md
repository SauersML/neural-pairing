# S4 `s4_train` (36–46 s): training

## REVISION 2 (current state; overrides the details below where they differ)
- Final contact sheet: `out/preview/s4/final_r2_sheet.jpg` (36.3, 37.3, 38.4, 39.5, 40.6, 41.8, 43.2, 44.4, 45.7).
  Full-res checks: `out/preview/s4/r2full_39.60.jpg`, `r2full_45.80.jpg`, `r2full2_41.80.jpg`, `r2full2_44.20.jpg`.
- Captions use the replacement wording exactly, at the lib defaults (30/92/46 px):
  36.25–39.95 `training` · `Fitted start` · `Tokens that end at the same character are matched, and a ridge regression maps their states.`;
  40.1–42.95 `Base weights frozen` · `Training updates the link and rank-32 LoRA adapters.`;
  43.1–46.3 `900 warm-up steps, then the full mix` · `value games first; then reading, play, dialogue, silent rounds, shared pen`.
  A dark gradient sits under the caption so it stays legible over the glowing link.
- The token rows show part of the sentence, ` Phraya, 5203 lanterns` (characters 8–30), with the real splits:
  Quill ` Ph|ray|a|,| |5|2|0|3| lantern|s` and Ember ` Ph|raya|,| |520|3| lantern|s`. The 8 real matches, re-indexed from
  `matches_quill_ember`, get gold connectors. `ray`, `5` and `2` get a red ×. On screen the tiles are about 105 px tall,
  with tile text about 58 px, and the narrowest one-character tiles are about 58 px wide.
- All small labels are removed: the tokenizer names, the match/no-match pins, the M pin, the "Quill reads" pin and the
  three legend pins in the frozen part. The ridge map is drawn as a slab with a large "M" on it (about 45 px). The
  only label besides the caption is the stat at 38.75–40.1, one line centred above the rows:
  `attention on the matching token: 89–97% (chance 1.2%)` (36 px text, 64 px number).
- Attention steps Ph→Ph, a→raya, 0→520, then dwells on lantern→lantern.
- Frozen part: the camera starts low at the tails and rises with the frost, so the frost is seen above the caption.
  Only the caption is on screen (no pins). The padlocks on the port rings remain as icons.
- Curriculum: the counter reads `STEP` (30 px), the number (84 px) and `warm-up` → `full mix` (30 px). The value tiles
  are blank gold tiles with no text. The ✓ pops beside the receiver's head. There are 5 full-mix cards (`shared reading`, `play`,
  `dialogue`, `silent rounds`, `shared pen`), each 0.74 units tall, so the text is about 40–60 px. The near side of the
  orbit is low and the far side is high, so the cards stay off the counter and the caption.
- There are no numbered tags (`Chapter` is a no-op in lib) and no links.

---

Files: `shots/s4_train.js` (the shot), `shots/s4_train_tex.js` (canvas textures: char-aligned token tiles, outlines,
sparkle star, matrix slab M, padlock open/closed, red x, curriculum cards).
Final contact sheet: `out/preview/s4/final_sheet.jpg` (times 36.3, 37.3, 38.4, 39.5, 40.6, 41.8, 43.2, 44.4, 45.7).
Render cost: about 1.6–2.4 s per full-res frame with the machine under load (first frame about 6 s for shader compile).

## Second by second

**36.0–40.0 The fitted start** (camera: slow dolly right, 10.4 to 9.7 units out; creatures pulled wider to x = ±3.85
and turned in to read; their eyes follow the action along the rows)
- 36.06–36.6: the real tokens from `data/tokens.json` pop in with springs: Quill's square violet tiles above (15 tokens),
  Ember's orange pills below (12). Tiles are laid out **by character position** (0.152 world units per character), so a
  tile's right edge is its end character. Spaces show as faint dots (`·the`, `·lantern`), so the token boundaries are exact.
  Small labels `Qwen3 tokens` / `AFM tokens`.
- 36.75–37.45: gold connectors snap in left to right at each shared end character (all 12 real matches), each with a
  flash and gold outlines on both tiles. Where the splits differ (Quill `ray`, `5`, `2`), a grey stub and a red × mark
  "no partner". Pins: `same end character → match` (36.95–37.8), `splits differ → no match` under `520` (37.15–38.45).
- 37.5–38.6: the ridge map **M** (a glass slab with a heat-map grid, teal edges) slides between the rows. As it passes
  each matched pair, an orange state orb rises from Ember's token through M, turns violet, and docks at Quill's
  matched token with a sparkle. Pin follows it: `M · ridge map: Ember states → Quill space`.
- 38.62–40.0: attention. A reader token in Quill's row lights violet and fans beams down to every Ember token, with
  bars under Ember's row (the peak bar and beam are gold). It steps `Past→Past`, `Ph→Ph`, `a→raya`, `0→520`, then
  dwells on `lantern→lantern`. Stat box (38.75–40.1): `TOP ATTENTION ON THE MATCHING TOKEN 89–97% (chance 1.2%)`.
  The link brightens from 0.12 to full as attention lands (the link now has its fitted start).
- Caption 36.25–39.95: `training · step 0` / `A fitted start.` / `Align tokens that end at the same character, then fit a map between their states.`

**40.0–43.0 Frozen and adapted** (camera cuts closer to the pair, 8.6 units out, and rises with the frost front, then
eases down to the ports)
- 39.95–40.35: the rows dissolve upward. The creatures slide home (±2.3, yaw ±0.5) by 40.65.
- 40.0 (Quill) / 40.15 (Ember): frost crawls **up** the spine block by block over 1.3 s. Each block's ice shell pops on
  with easeOutBack, faceted ice crystals sprout at the front, star sparkles flash as it passes, and a glow rides the
  front. A pale-blue flash at 40.0. The creatures **shiver** (high-frequency lean, jitter, head shake, squint, blush),
  look down at their own bodies, and the core dims a little.
- 40.95–41.75: LoRA chips light teal on every block, bottom to top, with a flash on each. The link comes back to full.
  Both creatures do a happy hop with squash at 41.85 / 41.95.
- 41.5–42.1: padlocks appear on all 16 port rings, each on the side that faces the partner, open, then snap shut with a
  punch and an ice ring flash.
- Legend pins between the heads: `original weights · frozen` (ice, from 40.85), `link + LoRA adapters (rank 32) · learn`
  (teal, from 41.05), and a padlock with `gates held (not trained)` (from 41.7). This matches `--hold`, which freezes the gates and α in the mix.
- Caption 40.1–42.95: `Original weights: frozen.` / `Only the link and rank-32 LoRA adapters learn.`

**43.0–46.0 Curriculum** (camera: wide orbit pushing in from 13.6 to 10.6 and accelerating; the fov tightens and a
small shake grows into the cut)
- A step counter at top center races 0 → 900 (43.15–44.55, ease in-out) with a gold progress bar, labelled
  `warm-up · 7 value games`. It punches at 900, turns teal (`warm-up done · full mix`), and keeps climbing faster,
  to about 1,250 at the cut.
- Warm-up: 14 value-game cycles, accelerating. A gold chip naming the game type (`word`, `number`, `phrase`, `digits`,
  `sum`, `name + city`, `lookup`) drops into the **sender's tail** (its prompt). Gold packets race across all 8
  bridges in that direction, and a gold ✓ pops from the **receiver's head**. The direction alternates. The creatures'
  eyes track the receiver.
- 44.6: a star burst at the center. The six `full mix` cards (`shared reading`, `distillation`, `play`, `dialogue`,
  `silent rounds`, `shared pen`) burst out with easeOutBack into a tilted orbit around the pair, whose speed grows
  quadratically. Both creatures hop with happy eyes.
- Tension into 46.0: link intensity, packet count and flow, core flow, sway and crest all ramp up, along with bloom
  pumping on beats, slight chromatic aberration, vignette and a creature crouch. The cut lands at the peak (no flash,
  so S5's near-silent drop gets the contrast).
- Caption 43.1–46.3: `curriculum` / `Warm up on value games. Then the full mix.` / `900 warm-up steps, then reading, play, dialogue and more. One A100.`

## Accuracy choices
- **Nothing discrete crosses the link.** The storyboard says "value-game glyphs zip through the link", which would
  break exactness rule 1. So the glyphs enter as prompts at the sender's tail, only gold packets ride the strands,
  and the receiver writes the result (✓) at its head.
- The warm-up chips name game **types**, not values, because the held-out test values (lantern, 5203, …) are never
  training values.
- The alignment is the real one: matches are the 12 pairs from `matches_quill_ember`. Quill `ray`, `5` and `2` have no
  partner; `a↔raya`, `0↔520` and `3↔3` match.
- The direction shown at step 0 is Quill reading Ember: M maps the sender's (Ember's) states into the reader's (Quill's)
  space, and the reader's top attention lands on the matching token. The weights shown are illustrative.

## Lib issues worked around (lib not edited)
- `creature.pose.ice` / `pose.lora` are single scalars (one shared material for all blocks). To get a bottom-up crawl,
  after `creature.update(t)` the shot hides or scales each `iceMeshes[i]` / `loraMeshes[i]` by its own front progress.
  It also overrides `iceMat` opacity/colour/emissive and `loraMat` opacity on its own creature instances, because the
  defaults bloomed the frozen bodies to white-teal and hid the LoRA chips.
- `camera.matrixWorld` is only refreshed at render, so a `Pin` or `toScreen` call inside `update` projects with the
  **previous frame's** camera, which breaks purity (pins jumped between sequential and random-order renders). The shot calls
  `camera.updateMatrixWorld(true)` after placing the camera. `_template.js` (and any shot copied from it) has the same issue.
- In the template, `lookAt = other.headWorld()` before `update` reads last frame's matrices, which is also impure.
  This shot computes eye targets from the current poses instead.
- Caption `wordsInto` splits on spaces even inside a `<span>`, so multi-word colored phrases must be wrapped
  word by word (the `paint()` helper does this).
