# S2 `s2_link`: THE LINK (10–22 s)

This shot follows **REVISION 2 and REVISION 3** (no single-letter symbols on screen): large type (no text under 30 px), the replacement captions word for word, no
chapter numbers, and exact per-token-position architecture.

Files: `shots/s2_link.js` holds the pair, the bridges, the unroll into `lib/grid.js` sheets, the dive and the snap back.
`shots/s2_link_schem.js` holds the dissected read, which has its own `THREE.Scene`, camera and HUD layer. `update(t)`
reassigns `inst.scene` / `inst.camera`: the schematic runs for 14.5 ≤ t < 21.2, and the pair scene runs for the rest.

Final contact sheet: **`out/preview/s2/final4_sheet.jpg`** (REVISION 3 text). Full-res REVISION 3 checks: `out/preview/s2/r7_15.50.jpg`, `r7_16.80.jpg`, `r7_18.20.jpg`, `r7_19.60.jpg`, `r7_20.60.jpg`, `r7_21.00.jpg`.
Previous sheet: `out/preview/s2/final3_sheet.jpg` (11.5, 13.6, 14.05, 14.25, 14.6, 15.5, 16.8, 18.2, 19.6, 20.6,
21.0, 21.6). Full-res checks: `out/preview/s2/r3c_16.80.jpg`, `r3c_17.60.jpg`, `r3c_19.60.jpg`, `r3c_20.60.jpg`, `r3c_21.10.jpg`.

Director-review fixes:
1. In the dissection's wide framings, all geometry now sits above about y = 780. A dark gradient band (330 px) sits behind
   the caption and the key. Every schematic label fades out if it would land below y ≈ 790. In the close-up on the write,
   the camera is placed so that the residual stream stays right of the caption text, and Quill's side dims.
2. `gate closed` is 36 px bold on a chip. `future: hidden` is 34 px on a chip. Tick numbers are 32 px mono. All were
   checked at full res.
3. The unroll is brighter and closer. The camera moves in to the sheets during 13.8–14.12 and dives at 14.22–14.5. The
   sheets have stream glow ×1.8 and full node glow, and the 3D token tiles show (about 36 px text). The `lantern` column
   is lit, its beam to `is` is the strongest (violet packets), and the beams fan out in depth.
   The caption now ends at 13.95.

## Second by second
**10.0–13.85: the bridges grow.** The beat drop brings a soft teal flash. The camera sweeps in 3/4 view (az 0.55 → 0),
with the pair framed low-right and the caption top-left over the sky. Exchange k (blocks 6, 9, …, 27) grows at
10.0 + 0.5k over 0.42 s. Each one brings a burst sprite and an expanding ring at both ports, a block-glow flash, a
brightness spike on the strand and a squash kick in both creatures. Packets ride both directions (the Link lib's glowing
points, no letters). A column of block numbers beside Ember (`after block 6`, then `9 … 27`, 32 px) adds one number per
bridge. The creatures breathe on the beat, sway and blink, and their eyes follow the bridges upward. They lean in at
13.4 and look at block 24 from 13.7. Caption 10.3–14.3: `the link` / `Cross-attention at 8 depths` /
`After blocks 6, 9, …, 27, each model attends to the other.`

**13.85–14.5: unroll and dive.** The bridges other than block 24 dim. The creatures fade, and two `ModelSheet`s
(lib/grid.js) unroll in place, rows aligned to the vertebrae. Quill's columns are `… lantern ▸ Sky looks blue` and
Ember's are `… word? ▸ The word is`, so each column is one token position's residual stream. On the row after block 24,
`LinkBeams` draw the cross-attention edges. Ember's `is` (tick 3) reads Quill's positions with the same weights used in
the dissection, so `lantern` is strongest. Every other position reads the partner's positions of its tick or earlier,
in both directions (faint). The camera accelerates into that row (FOV 35 → 55), and the bloom and a mint flash build to
the cut at 14.5. The camera moves in close, so the 3D token tiles render at about 36 px.

**14.5–21.2: one read, dissected (Ember reads Quill, the exchange after block 24).** The section opens on teal speed
lines over a blueprint grid. A 2-line HUD equation at 46 px sits across the top:
`h_E += tanh(α) · size · O( attention(` / `Q(norm h_E), [null; K(norm h_Q)], [0; s · V(norm h_Q)] ) )`.
Each term lights in its owner's colour, with a pill and glow, when its 3D part animates. An ownership key sits bottom
right (30 px): `K, V, gate s: Quill` (violet) and `Q, O, α, null: Ember` (orange). Caption 14.8–21.0 (92/46 px):
`Ember reads Quill` / `Quill reads Ember the same way, at the same time.`
- 14.52–15.9: Quill's hidden states pop as violet orbs, one per token position. Below them are token chips
  (`… lantern ▸ Sky looks blue`) and a tick row (`−1 −1 0 1 2 3`). The future position `because` (tick 4) is ghosted,
  with the label `future: hidden` and empty dashed K/V. Lines extrude to a violet **K** cube (it bypasses the gate,
  because keys are not gated) and a light-violet value cube. Row labels: `h_Q`, `s · V`, `K`.
- 16.0–16.85: a violet ring-gauge gate dial (`gate s`) fills around each value, and each value scales by its gate
  (illustrative s = 0.84 / 0.70 / 0.90 / 0.60 / 0.55 / 0.88). At 16.45 the gate at `▸` closes: its value shrinks to a
  dark speck (`gate closed`) while its key stays.
- 16.85–17.4: an orange **null** key pops with an empty value **0** (`null: attend to nothing`).
- 17.4–18.1: Ember's residual stream at the token `is` (tick 3) brightens, along with its state orb `h_E · block 24`.
  Three faint streams behind it are Ember's other positions. An orange line carries `h_E` to the **Q** cube. At 17.8 an
  orange query beam sweeps along the key row, and each key flashes as the beam passes.
- 18.1–19.0: teal attention bars rise over null and the 6 visible keys (0.14 / 0.05 / 0.36 / 0.10 / 0.07 / 0.08 / 0.20,
  which sum to 1). There is no bar over the hidden future.
- 19.0–19.95: the values pour along arcs into one blended orb. Brightness ∝ bar × gate, so the closed gate and the null
  `0` contribute nothing. Quill's side then recedes.
- 19.95–20.3: the orb passes through **O** and comes out orange. 20.3–20.75: it shrinks through an iris
  (`× tanh(α) · size`, then `≈ 5–10% of the residual`). The camera pushes in on the write, and the bottom labels step
  aside.
- 20.75–21.2: the orb lands on the **+** node of the `is` stream, ripples run, and a hot pulse runs up to `block 25`.
  The faint streams of the other positions get their own writes at the same moment. `h_E +=` lights. The camera pulls
  wide, and a warm flash closes the section.

**21.2–22.0: snap back.** A spring camera pulls out from beside the block-24 bridge to the wide pair. All 8 bridges are
bright, with more packets. Both creatures go happy-eyed, flare their crests, lean in and hop (squash, stretch, Ember
60 ms after Quill). A white flash ramps over 21.82–22.0.

## Accuracy (against `pairing/link.py` and the storyboard)
- Orange (reader, Ember): Q, O, α (tanh(α)), size, the null key. Violet (sender, Quill): K, V, gate s. Attention is teal.
- The gate multiplies the values, not the keys. A closed position keeps its key and can still take attention mass, but
  it adds nothing. The null key has a zero value. Positions after the reader's tick are masked.
- The reader is one token position. The write is added to that position's residual stream after block 24, and that
  stream continues to block 25. All positions read at the same time, and Quill reads Ember in the same way (the sheet
  beams in both directions, plus the caption).
- Bar heights and gate values are illustrative. Revision 2 removes footnotes, so this is stated here instead of on
  screen.

## Lib notes / workarounds (lib not edited)
- `Creature.update` billboards Ember's flame with the camera from before the shot moves it. I recompute
  `e.flame.quaternion` after setting the camera.
- No `blockWorld()` / `headWorld()` calls happen before `creature.update`, because those matrices would be stale.
  Look-at targets are analytic.
- The creature's residual-stream core is opaque (`transparent: false`), so with `opacity → 0` it turns black instead of
  fading. I hide the creature groups below 6% opacity during the unroll.
- The `ModelSheet` token tiles get `renderOrder = 10`, so they draw after the floor plane.
- Determinism: all JS state (camera, creatures, link) is identical when the same t is rendered after different frames.
  Pixel output drifts slightly with render order (a few levels on average, around bloom and DOM text). `s1_meet` drifts
  the same way, so the cause is engine or compositor level.
- Render cost at full res: about 0.7–1.3 s per frame at steady state, and about 2–2.6 s for the first frame after a
  scene switch (shader compile).

## REVISION 3: plain words instead of symbols
- The equation line (46 px, 2 lines) now reads `Ember's hidden state  +=  scale × output(` /
  `attention( query, keys, gated values ) )`.
  - Orange: `Ember's hidden state`, `scale`, `output`, `query`. Violet: `keys`, `gated values`. Teal: `attention`.
  - Lighting order: keys and gated values at 15.0 (gated values stay active through the gates), attention at 16.85 (the
    empty slot appears) and again at 18.1 (the bars), query at 17.4, output at 19.95, scale at 20.3, and
    Ember's hidden state += at 20.75.
- The 3D labels are now words: `key`, `gated value`, `Quill's hidden states`, `gate`, `gate closed`, `query` (below the
  cube), `output`, `nothing` (the null key; the zero-value cube has no label), `attend to nothing`, `× scale`,
  `≈ 5–10% of the residual`, `Ember's hidden state / block 24`, and `block 25`.
- The ownership key reads `Quill's weights: key, value, gate` / `Ember's weights: query, output, scale`.
- The top caption sits well without a kicker. There is no all-caps text.
