# S5 `s5_proof`: 46–60 s, "05 / THE PROOF"

Files: `shots/s5_proof.js` (the shot), `shots/s5_proof_ring.js` (the ring of 180 episode dots),
`shots/s5_proof_hud.js` (number, stat cards, check and cross badges, title card). `update(t)` is a pure function of t.
I checked this by rendering a frame cold and again after other frames. The two images match byte for byte.
Full-res render takes about 1.0–1.5 s per frame when warm, and about 2.2 s for the first frame in a fresh page.

Final contact sheet (Revision 2): `out/preview/s5/final_r2_sheet.jpg`. Full-res Revision 2 checks: `out/preview/s5/r2full_*.jpg`.
Pre-revision sheet: `out/preview/s5/final_sheet.jpg` (half-res frames at 46.3, 47.5, 48.6, 49.6, 50.3, 51.5, 53.2, 54.9, 56.5, 58.2, 59.5).
Full-res checks: `out/preview/s5/full1_*.jpg`, `full2_*.jpg`, `full3_49.30.jpg`, `fulltime_*.jpg`.

## Second by second
- **46.0–46.6** The camera dollies slowly in on the pair, framed right of centre so the caption has room. The real Quill is linked to
  Ember, and packets flow. A gold secret tile `lantern` sits at Quill's tail. A prompt tile `What is Quill's secret word?` sits at Ember's
  tail and stays the same for the whole test (same game, same roles). The pin `REAL PARTNER` is above Quill. The caption comes in at 46.2:
  `the test` / `Swap in a stranger.` / `Same game, same roles, another episode's partner. Does the secret vanish?` (on screen until 49.6).
- **46.6** Ember writes `lantern` (an Ember pill with gold text). It rises out of its head with a spring.
- **47.0** (beat) A teal ✓ badge pops in and its stroke draws on. Both creatures get happy eyes and a squash.
- **47.22–47.5** The link retracts. **47.32–47.82** Quill slides out to the left, and its secret tile fades.
- **47.72–48.15** A grey ghost Quill (`ghost: 1`, no blush, oblivious gaze) springs in from the left. Its own tile `galaxy`
  (ghost style) fades in at its tail. The pin `STRANGER · another episode` appears.
- **48.0–48.5** The link regrows exchange by exchange, bottom to top, to the stranger. These packets are pure teal.
- **48.5** Ember confidently writes `galaxy` (a red "wrong" pill) with happy eyes. **49.0** (beat) A red ✗ pops in. Ember recoils,
  double-blinks and tilts its head, puzzled. The stranger's copy is the confident wrong answer the README describes for the shuffle control.
- **49.48–50.0** The camera cranes fast up into the sky to a ring of 180 unlit dots.
- **50.0** (hit) A flash, a bloom surge, a camera kick and a shockwave. The dots ignite in a cascade. Both halves sweep from the back of the ring
  to the front at once, fast at first, then slower, and finish at 51.45. The counter counts the lit teal dots. The big
  `179 / 180` springs in, and it gets a pop and a second softer pulse when the cascade completes.
  The ring's layout follows `figures/transfer.png` exactly: the left half is EMBER → QUILL (AFM → Qwen) and the right half is QUILL → EMBER, 90 episodes each.
  Each half holds the 7 games as separate arcs, with their real episode counts 10/10/10/10/20/20/10. The one grey dot sits in the AFM → Qwen
  "join a name and a city" arc (19 of 20), which is where the loss really is. The direction labels show from 50.7. The small label `1 not won` marks the grey dot from 51.7.
  The sub runs 50.35–53.0: `held-out episodes won by the real partner · all 7 games, both directions, p < .05`.
- **52.75–53.25** The number glides up and shrinks into a header, and the ring dims to 40%.
- **53.0 / 53.8 / 54.6** Three stat cards punch in on the beats, each with a spring, an un-blur, a teal border flash, a ring pulse and a bloom bump. Each card also fills in its own small graphic:
  - `31 / 32`: 32 dots, 1 grey · "shared-reading groups better with the real partner"
  - `128 / 128`: 128 dots · "passed in turn structures it never trained on" · `DIALOGUE · SILENT ROUNDS · SHARED PEN`
  - `6–9%`: 100 gate ticks, 6 open and 3 half-open · "of gates left open after training:" *they learned to whisper.*
- **55.3–55.6** The cards and the header leave. **55.6–56.7** The camera cranes back down on a spring to a wide shot of the pair, and the ring of 180
  descends like a halo to lie on the floor around them.
- **55.9–56.9** The creatures lean and curve toward each other, and the link blazes. They get happy eyes at 56.1–56.4.
- **56.0** (bar) **NEURAL PAIRING** resolves from blur, with letters converging from wide tracking and a teal glow plus a bloom surge.
- **57.0 / 57.5 / 58.0** (beats) `Two models.` · `One link.` (teal) · `No words.`
- **57.3–58.0** The shared hop: anticipation squash, airborne 57.5–58.0, landing squash on 58.0.
- **58.45** A hairline rule draws, then `Qwen3-4B × AFM-4.5B · github.com/SauersML/neural-pairing` appears.
- **59.2–60.0** Fade to black. The post `fade` darkens the canvas, and the HUD opacity fades with it.


## Revision 2 changes (applied)
- All text uses the replacement wording word for word. The smallest text in the shot is 30 px; the chapter tag's 28 px comes from the lib `Chapter`.
  - 46–50: Caption at the new defaults. Kicker `control`, head `Swap in another episode's partner` (one line at 92 px), sub `Same game and roles, but a different secret.`
  - 50–53: `179 / 180` (196 px), then the sub at 46 px, set on two centred lines: `held-out episodes where the real partner wins` and `all 7 games, both directions, p < 0.05`. The "·" separator becomes the line break.
  - Cards: `31 / 32` + `shared-reading groups`, `128 / 128` + `turn structures never trained`, `6–9%` + `of gates open after training`. The stats are 84 px and the descriptions 36 px, each broken into two balanced lines. "they learned to whisper" and the detail line are gone.
  - Title: `NEURAL PAIRING` (106 px), then `Two models linked through their hidden states` (50 px, words revealed from 57.0), then `github.com/SauersML/neural-pairing` (32 px mono, from 57.85).
- Removed: the REAL PARTNER and STRANGER pins, the ring's direction labels, the `1 not won` label, and Ember's long prompt tile (`What is Quill's secret word?`), which would have been under 30 px or added words. The grey dot is now larger and brighter, so it reads without a label.
- The control: Quill's tail tile `lantern` is the real episode's secret. When the stranger arrives, its own tail tile `galaxy` (another episode's secret) fades in, the link regrows to it, and Ember writes `galaxy`. The viewer sees the receiver taking in the hidden states of a sender that was given a different secret.
- 3D tiles are 0.66–0.7 world units tall at about 95 px per unit, so their text renders about 40 px tall. The ✓/✗ badges are now 76 px.
- The pair is centred and framed above the full-width caption block (camera at d ≈ 18).

The second-by-second section above describes the pre-revision wording. The timings are unchanged except that there are no pins, the tagline starts at 57.0 and the credit at 57.85.

## Lib issues worked around (lib left untouched)
- **The post `fade` only darkens the WebGL canvas, not the DOM HUD.** I set `hud.style.opacity = 1 - fade` in my own root.
- **The link colours packets with `C.quillHi` whatever the sender's `ghost` value is.** A grey stranger would have sent violet-tinted packets, so the ghost's link uses `colorBySender = 0`.
- **Token tiles use `toneMapped: false` with a bloom threshold of 0.16, so white text blooms and goes soft.** I scaled the tile material colour down (0.62 for the prompt tile, 0.74 for the others).
- **Non-exact font weights can resolve differently on a cold first frame.** My `/ 180` was set at weight 250, and the first frame after page load rendered it differently from later frames. I now use only weights that exist as font files (300 and 700). The lib `Caption` uses weight 650. I did not see it change between frames, but other shots should keep this in mind.
- **Reversed `smoothstep(edge0 > edge1)` is undefined in GLSL.** My first shockwave shader used it and lit a huge teal flood under SwiftShader. My shaders now use `1.0 - smoothstep(a, b, x)`. Lib shaders (creature, env) still use the reversed form, and it seems harmless there.

## Revision 3 (result windows, no symbols)
- No single-letter symbols. The director replaced `p < 0.05` with `statistically significant in all 7 games, both directions` (shown on two lines). Codes appear only as labelled data (`code M2`).
- **53.0 / 53.8 / 54.6:** three 572 × 744 windows (`shots/s5_proof_demos.js`) pop in on the beats with a spring and a teal border flash. They replace the stat cards. The `179 / 180` header is dropped and fades out at 52.7–53.0, and the ring dims to 30% behind the windows. Each window plays a small 2D demo: mini Quill (violet, feather) and Ember (orange, flame) heads, token pills (Quill square-cornered, Ember round), and a teal link line that carries only dots. All text is at least 32 px, the numbers are 104 px and the captions 36 px.
  1. **Code lookup:** Ember holds `K7 candle` and `M2 breeze`, and Quill holds `code M2`. Dots cross Ember → Quill while the `M2 breeze` row glows. Quill writes `breeze` ✓ (about 55.0 / 55.3). The window shows `17 / 20` and `a game it never trained on`.
  2. **Turn structures:** the window cycles `dialogue` (alternating speech pills with grey bars in place of words), then `silent rounds` (no text, stronger dots both ways), then `shared pen` (`the boat leaves at dawn`, alternating violet and orange words). Each format gets a ✓ by its label. The window shows `128 / 128` and `secrets passed in all three`.
  3. **Shared reading:** Quill holds `…the door was red…` and Ember holds `door colour?`. Dots cross Quill → Ember and Ember writes `red` ✓ (about 56.1 / 56.35). The window shows `31 / 32` and `reading tasks improved by the link`.
- **56.85–57.15:** the windows exit with blur. **56.9:** the camera cranes down and the ring descends to the floor. **57.2:** `Neural pairing`. **57.7:** the tagline, word by word. **58.2:** `Qwen3-4B × AFM-4.5B`. The hop takes off at 58.0 and lands at 58.5. The fade runs 59.2–60.0, unchanged.
- Final sheet: `out/preview/s5/final_r3_sheet.jpg`. Full-res checks: `out/preview/s5/r3b_*.jpg` (53.4, 54.4, 55.4, 56.4, 57.8, 58.8).
- Render cost: 2.1–3.2 s per full-res frame and 4.2–4.6 s for the first frame after page load or a scene cut, measured with a load average of about 11 on 4 CPUs. Before the windows existed, the same shot measured 1–1.5 s per frame on a quiet machine.

## a6_proof (v2, 66–82 s): built on these modules
- `shots/a6_proof.js` runs the S5 logic on S5 time (t − 20). Bodies are glassy (xray 0.6), and the link plugs into the cores. During each forward pass, the injected color rises in Ember's core: gold for the real secret, silver for the other episode's secret. The only on-screen label in the control is `another episode's partner`. The counter shows `180 / 180`, with `won by the real partner` underneath.
- `makeDemoWindows(parent, opts)` takes optional overrides: `captions`, `stats`, layout (`width`, `gap`, `top`, `statSize`, `pad`, `height`), and a fourth window, `chat: {stat, caption, note, city}`. With no `opts` it gives exactly the old S5 layout.
- In a6 there are four windows, each 436 px wide, popping in on 73.0, 73.8, 74.6 and 75.4:
  - `35 / 40` untrained game (nl21)
  - `128 / 128` untrained formats
  - `410 / 456` shared reading (nl20, step 1,320)
  - `14 / 16` free chat, with `0 / 16 with a stranger` underneath (nl20)
- The windows exit at 77.8. The title runs 78.2 / 78.7 / 79.2, and the fade runs 80.8–82.
- Previews: `out/preview/a6/final_sheet.jpg` (whole shot) and `out/preview/a6/r5_sheet.jpg` (full-res frames at 73.4, 75.0, 75.9 and 77.0 for the four windows).
