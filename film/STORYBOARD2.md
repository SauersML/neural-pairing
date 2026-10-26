# NEURAL PAIRING v2: visual-first (about 82 s)

This replaces STORYBOARD.md. The old file is still useful for the engine API and exactness rules, but **this file wins
wherever they differ.**

## The director's notes that drive v2 (from the user)
1. **Show, don't write.** A viewer who can't read must still follow the film. Every idea is carried by motion,
   color, character acting and icons. Words are optional garnish.
2. **The viewer is either watching or reading, never both.** Text appears only while the picture holds still, and
   never over important motion. At most about 5 words on screen at a time, plus big numbers. No sub lines unless
   essential. No kickers, no ALL CAPS, no single-letter symbols, no "tick" (nobody knows what a tick is).
3. **The link writes INSIDE the partner.** The strands plug into the partner's glowing core (its residual stream) at
   the 8 exchange depths, and the sender's color is injected into the reader's core and then rises with the computation.
   The models are deeply integrated, not standing side by side. Bodies are glassy (x-ray) whenever the link matters.
4. **One forward pass already has a rich back-and-forth.** Both models compute block by block in lockstep. At each of
   the 8 exchange depths, both write into each other at once, so an idea can bounce Quill → Ember → Quill → … 8 times
   on its way up, inside a single forward pass for a single token. This repeats for every token both models write.
5. **Not only facts.** Because the link writes into the internal state, it can carry anything a hidden state holds:
   facts and concepts, but also mood and tone. Show both: a secret word, and a feeling.
6. **Games, training data and results are fully animated windows** (picture-in-picture panels with mini creatures,
   icons and token pills) that explain themselves.
7. **Neuralese.** Use this word: the language of hidden states. It's the film's key word.
8. **Look:** keep the current world, palette and creatures (eyes are now bigger and the blush is gone). The creatures
   are characters who play and react.

## New engine features (lib/, by the director)
- `creature.pose.xray` (0..1, default 0.5): glassy blocks, so the core and the strands inside are visible.
- `creature.pose.wave` (0..1, −1 = off): a forward-pass wavefront. Blocks and core below it are computed (lit), the
  front flares, and everything above is dim. Drive both creatures with the **same** wave value: they compute in lockstep.
- `creature.inject = [{block, color, amount, top}]`: the partner's writes, added into this creature's core at `block`
  (an exchange block) and carried upward to height `top` (0..1, e.g. the current wave). Use the **sender's** color
  (violet into Ember, orange into Quill), or gold for a secret, or a mood color. Set it every frame.
- `link.state.depth` (default 1): strands end in the partner's core. `link.state.aToB` / `bToA` accept per-exchange
  arrays. `tintAB` / `tintBA` override packet color. `pulses: [{e, dir, u, color, amount}]` places big bright packets
  exactly where you want them along a strand (dir 0 = Quill → Ember).
- See `shots/_xraytest.js` (`node tools/preview.mjs --test _xraytest --times 0.9`).
- Exchange heights: block b sits at u = (b+0.5)/36; exchanges are after blocks 6, 9, 12, 15, 18, 21, 24, 27.

## Cut list (120 bpm, 1 bar = 2 s)
| id | time | what |
|---|---|---|
| a1_meet | 0–12 | two creatures assemble; text goes in at the tail and a token comes out the head; a token can't cross (the shatter); bodies turn glassy, revealing the cores; the word **neuralese** |
| a2_link | 12–32 | strands grow into each other's cores; ONE forward pass slowed down, with the zig-zag of an idea; then many passes fast (tokens streaming); then one read up close (query, keys, gated values poured into the core) |
| a3_cross | 32–42 | what crosses: a hidden secret word (information), then a feeling (mood) |
| a4_games | 42–54 | animated windows: the games they train and play on |
| a5_train | 54–66 | training: fitted start (token alignment), rank-32 LoRA with frozen original weights, game windows streaming in as data, success rising from chance to near-perfect |
| a6_proof | 66–82 | the control (another episode's partner fails), 180 / 180, the never-trained windows, title |

### a1_meet, 0–12
- 0–4: assembly as before (it's lovely), with **no captions**. Names only as clean pins once the bodies are built:
  `Quill` (`Qwen3-4B`) and `Ember` (`AFM-4.5B`).
- 4–7: show "a model" without words. A prompt tile enters each tail, a light climbs the 36 blocks (use `pose.wave`),
  and a token pops out of each head. Both do it together.
- 7–10: Quill tosses a token at Ember. The glass field flashes, the token shatters (hit at **8.0**) and Ember flinches.
- 10–12: hold. Both bodies turn glassy (`xray` 0 → 0.7), revealing the glowing cores, and the 8 port rings light.
  Only now, during the hold, does the one word **neuralese** appear (large, teal, centered or low), perhaps with a small
  `the language of hidden states` under it. It fades by 12.

### a2_link, 12–32 (the heart of the film)
- 12–15: strands grow from port to port **into each other's cores** (bottom to top, one per half beat). At the moment
  of each connection, the receiving core flashes with the sender's color.
- 15–23: **ONE forward pass, slowed down.** A tiny prompt tile enters both tails. A single wave climbs both spines
  together (same `wave` value). As it reaches each exchange block, both strands at that depth fire **at once**
  (`pulses` crossing both ways) and plunge into the partner's core, and the injected colors then rise with the wave
  (`inject` with `top = wave`).
  - Make the zig-zag visible: one gold "idea" starts in Quill's core near the bottom, crosses into Ember at block 6,
    rises with Ember's wave, crosses back into Quill at block 9 (a slightly changed hue = processed), and so on, a bright
    zig-zag path up the ladder.
  - At the top, both heads emit their next token at the same instant.
  - Optional on-screen, during a brief hold at the end only: `one token, 8 exchanges`.
- 23–26: then fast. Pass after pass (about 4 per second), tokens streaming out of both heads, strands flickering both
  ways at every depth: a constant, dense conversation.
- 26–32: one read up close, from the old S2 dissection, trimmed:
  - Ember's core position sends a query beam into Quill's hidden states.
  - Quill's states show keys, and their gated values (gate dials; one closes).
  - The weighted blend is poured INTO Ember's core and rises.
  - Labels are tiny in count and large in size: `query`, `key`, `value`, `gate`. No equation.
  - End by pulling back to the pair.

### a3_cross, 32–42
- 32–37, **information.** Quill gets a gold secret card (a lantern pictogram plus the word `lantern`) and hides it
  behind its back or in its tail, away from Ember. Ember peeks but can't see. A forward pass runs: gold is injected
  into Ember's core at the exchange depths, rises to Ember's head, and Ember writes `lantern` (gold tile, with a lantern
  pictogram above it). Quill hops.
- 37–42, **mood.** Something delightful happens to Quill only (e.g. a burst of fireworks or confetti appears in
  front of Quill, with a dark curtain on Ember's side). Quill bursts with joy: happy eyes, crest flare, warm pink-gold
  glow in its core. The warm glow flows through all strands INTO Ember's core and rises. Ember, who saw nothing,
  starts smiling (happy eyes, crest flare). Optional word at the end hold: `facts, and feelings`.

### a4_games, 42–54
A wall of small animated windows (3×3 or similar), each one game, playing at once after they fly in one by one (about
0.6 s apart). Mini 2D Quill and Ember icons as in `shots/s5_proof_demos.js` (reuse that kit and look), a teal link
line with dots only, pictograms and token pills. Each window shows who holds what (only one side has it), the link
firing, and the other side writing the answer ✓. Games (use the real held-out values):
1. relay a word: `lantern` (lantern pictogram)
2. relay a number: `5203`
3. relay a phrase: `silver whale`
4. relay a digit string: a long run of digits
5. add two numbers: Quill has `31`, Ember has `42`, and both write `73`
6. name and city: Quill has `Keiko` (person icon), Ember has `Geneva` (city icon), and both write `Keiko lives in Geneva`
7. code lookup: Ember has a table, Quill asks `code M2`, and Quill writes `breeze`
8. shared reading: Quill has a passage (`…the door was red…`), Ember has `door colour?`, and Ember answers `red`
9. play: a chat with speech bubbles where each names the other's secret (`Bangkok!` with a palm icon)
Then the camera pushes into one window, or the wall shrinks into a "dataset" stack that flies to a5. No caption, or
one 2-word title during a hold.

### a5_train, 54–66
- 54–58: fitted start (old S4 part 1, trimmed). Two token rows snap together where tokens end at the same character,
  the mismatches get no connector, and a matrix slab maps one model's states onto the other's. Stat during the hold
  at the end only: `89–97%` vs `1.2% chance`.
- 58–61: frost freezes the original blocks while the teal rank-32 LoRA chips light on every block, and the link stays
  bright. One label during the hold: `rank-32 LoRA` (the chips are what trains; the frost is what stays fixed).
- 61–66: training montage. The game windows from a4 stream in as data (small cards flying into the pair), and a big
  success meter or curve climbs from coin-flip 50% up to 100% (the shape of `figures/opened.png`). Value games come
  first, then the rest of the mix joins (more card types). Build to the cut at 66.

### a6_proof, 66–82
- 66–70: the control (old S5), visual: the real partner gives `lantern` ✓; another episode's partner (grey ghost,
  holding a different card) gives `galaxy` ✗. No caption, or one short one during the hold.
- 70–73: the ring of 180 dots ignites, all teal: **180 / 180** (nl20, full training mix).
- 73–78: the three result windows (old S5 demos): lookup never trained `17 / 20`, three turn formats never trained
  `128 / 128`, shared reading `407 / 456`. Fewer words.
- 78–82: title. `Neural pairing`, then `two models that talk in neuralese`, then `Qwen3-4B × AFM-4.5B`. Fade to black
  80.8 → 82.

## Music (audio agent)
A new score, about 82 s, 120 bpm in half-time. It should be dark and cool, not upbeat. Requirements are in the audio
agent's brief.
