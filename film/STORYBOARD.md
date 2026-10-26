# REVISION 3: no single-letter symbols on screen

Viewers know LLMs but are not fluent in math notation. **No single-letter variable names or symbols anywhere on
screen**: no Q, K, V, O, s, α, M, h_E, h_Q, p, and no equations written in symbols. Use plain words: "query",
"key", "value", "gate", "output projection", "Ember's hidden state", "small scale (starts at 5% of the residual)",
"linear map", "statistically significant". Data values such as the lookup game's code "M2" are fine when labelled
(e.g. "code M2").

# REVISION 2: these rules override everything below

**No numbered chapter tags (01, 05, …) and no GitHub links anywhere.** `Chapter` is now a no-op in lib/hud.js. Don't add step numbers in any other form.

The director has given three rules that apply to every shot.

## A. Very large type and no small text
- `Caption` defaults are now kicker 30 px, headline 92 px and sub 46 px (in `lib/hud.js`). Keep them or go bigger.
  Never go below **30 px** for any text on screen, including pins, labels, tiles seen on screen, counters and credit lines.
- Cut every label that is not essential. The viewer reads at most about 12 words at a time. Remove decorative
  micro-labels, small direction labels and footnotes. Text in 3D tiles must also render at least 30 px tall on screen.
- A headline is at most 6 words. A sub is one short sentence (at most about 12 words) or nothing.

## B. Plain, direct wording, written the way ML papers write
- Use standard terms: token, hidden state, residual stream, block, attention, query, key, value, gate, cross-attention,
  LoRA, frozen, held-out.
- Use no metaphors and no anthropomorphic verbs on non-agents. "Carry", "whisper", "heart", "secret only the link can
  carry" and "learned to whisper" are all out. The models can be subjects ("Ember writes the word") because in this
  film they are characters.
- Use no punchy negation fragments ("No tokens. No text."), no rhetorical questions ("Can they talk without words?"),
  no "rather than" contrasts, no "exactly" for emphasis, and no meta-commentary.
- State facts directly.

## C. The architecture must be exact
Show what really happens inside the models: tokens, one residual stream **per token position**, causal self-attention
within each model, cross-attention between the models at the 8 exchange depths, the write added into the reader's
residual stream at that position, and the next token decoded at the top. **`lib/grid.js`** (coming from the director)
draws an "unrolled" model: a sheet of token columns × 36 blocks, with causal attention arcs and cross-model link beams.
S2 and S3 use it for their technical moments. The creature body is the compact view of the same model, so a shot can
transform creature ↔ unrolled sheet.

## Replacement captions (use these words exactly)
- S1 0.5–3.6: kicker `neural pairing` · head `Two language models` · sub `Qwen3-4B from Alibaba and AFM-4.5B from Arcee`
- S1 3.6–6.6: head `36 blocks each` · sub `Each block adds its output to the residual stream.`
- S1 6.6–10.0: head `Only hidden states cross the link` (no sub). Note: in dialogue games the models also read each other's text as normal chat messages, but the link itself takes hidden states only.
- S2 10.3–14.3: kicker `the link` · head `Cross-attention at 8 depths` · sub `After blocks 6, 9, …, 27, each model attends to the other.`
- S2 14.5–21.2: the equation (large, at least 44 px) plus head `Ember reads Quill` · sub `Quill reads Ember the same way, at the same time.`
- S3 22–26: kicker `time` · head `One token per tick` · sub `A token attends to partner tokens from its tick or earlier.`
- S3 26–30.5: kicker `relay a word` · head `Only Quill's prompt has the word` · sub `Ember reads Quill's hidden states and writes it.`
- S3 30.5–33: head `Seven value games` · sub `words, numbers, phrases, digit strings, sums, name + city, lookups`
- S3 33–36: kicker `play` · head `Six-turn games` · sub `Each model names a secret that appears only in the other's prompt.`
- S4 36–40: kicker `training` · head `Fitted start` · sub `Tokens that end at the same character are matched, and a ridge regression maps their states.` · stat `attention on the matching token: 89–97% (chance 1.2%)`
- S4 40–43: head `Base weights frozen` · sub `Training updates the link and rank-32 LoRA adapters.`
- S4 43–46: head `900 warm-up steps, then the full mix` · sub `value games first; then reading, play, dialogue, silent rounds, shared pen`
- S5 46–50: kicker `control` · head `Swap in another episode's partner` · sub `Same game and roles, but a different secret.`
- S5 50–53: `179 / 180` · sub `held-out episodes where the real partner wins · all 7 games, both directions, p < 0.05`
- S5 53–55.6 cards: `31 / 32` shared-reading groups · `128 / 128` turn structures never trained · `6–9%` of gates open after training
- S5 title: `NEURAL PAIRING` · `Two models linked through their hidden states` · `github.com/SauersML/neural-pairing` (≥ 30 px)

# NEURAL PAIRING: a 60-second film

Two language models from different companies (Quill = Qwen3-4B by Alibaba, Ember = AFM-4.5B by Arcee) get a private
channel between their hidden states and use it to pass information the other was never shown. The film must be
**beautiful, alive and technically exact**, and it must make complete sense **with the sound off**. On-screen text
carries the story, and the motion shows what the text says.

Render target: 1920×1080, 30 fps, 60.0 s. Music: 120 bpm, so 1 beat = 0.5 s and 1 bar = 2 s (bar n starts at 2n s).
Cuts land on bar lines.

## The engine (read these files)

- `main.js`: runtime. `window.renderAt(t)` draws global time t. Each shot is `shots/<id>.js` exporting
  `async create({hud, def, renderer})` that returns `{scene, camera, update(t)}`. `update(t)` gets **global** t and may
  return post settings `{bloomStrength, bloomRadius, bloomThreshold, exposure, vignette, grain, aberration, fade, flash, flashColor}`.
- **Everything must be a pure function of t.** The renderer may jump to any t in any order, and frames render in
  parallel workers. Keep no state that accumulates between frames, use no `Math.random()` (use `rng(seed)`/`hash` from
  `lib/util.js`), and never assume `update` was called for earlier frames.
- `lib/creature.js`: `new Creature('quill'|'ember')`. Set `creature.pose.*` every frame, then `creature.update(t)`.
  The body **is** the model: 36 vertebrae = 36 blocks (block 0 at the tail, block 35 under the head), a glowing core =
  the residual stream, teal port rings = the 8 exchange blocks (6, 9, 12, 15, 18, 21, 24, 27). Quill has a feather
  crest and Ember a flame crest. The pose fields are documented in `Creature.defaultPose`: position, yaw, lean,
  leanFwd, bend, sway/swayPhase, squash, headTilt, headNod, faceYaw, lookAt, blink, happy, glow, coreGlow, portGlow,
  ice, lora, assemble, opacity, ghost, crest, scale. `blockGlow[i]` and `blockColor[i]` light individual blocks.
  The accessors `blockWorld(i)`, `blockSurface(i, toward)`, `headWorld()`, `mouthWorld()` and `tailWorld()` are valid
  after `update`. Set `creature.camera = camera` so the flame billboards.
- `lib/link.js`: `new Link(quill, ember)`. `link.state` holds grow (scalar or per-exchange array), intensity, flow,
  packets, aToB, bToA, highlight, dim, sag and twist. Call `link.update(t)` **after** both creatures update. The packets
  are hidden-state vectors, so they must never carry letters.
- `lib/tokens.js`: `tokenTile(text, style, height)` makes a 3D token tile (styles: quill, ember, prompt, secret,
  ghost, wrong, link). Quill tiles are square-cornered and Ember tiles are round pills, because the two tokenizers differ.
- `lib/hud.js`: DOM typography over the canvas. `Caption` (kicker / headline / sub, with a word-by-word kinetic
  reveal), `Pin` (a label pinned to a 3D point), `Chapter` (top-left chapter tag), `el()` and `toScreen()`.
- `lib/env.js`: `makeWorld(scene)` builds the void, nebula sky, floor grid, dust and light rig. Call `world.update(t)`.
- `lib/util.js`: easing, `prog(t,a,b)`, `win(t,a,b,fade)`, `spring(t,t0,freq,damp)`, `kick()`, `beatPulse(t)`, `rng`, `hash`, `noise1`.
- `shots/_template.js`: a minimal working shot.
- Preview: `node tools/preview.mjs --shots <id> --times 10.5,12,14 --scale 0.5 --out out/preview/<id>/a --sheet`,
  then look at the JPGs with the Read tool. A full-res frame takes about 1–2 s, half-res about 0.5 s. The machine has
  4 CPUs **shared by everyone**, so keep previews at `--scale 0.5` with a handful of times each, and use full res only to
  check text sharpness occasionally.

**Do not edit `lib/`, `main.js`, `index.html` or another shot's file.** If you need more, build it inside your own shot
file (or a new file `shots/<id>_*.js`). If a lib bug blocks you, work around it locally and note it in `notes/<id>.md`.

## Look and feel (all shots)

- Palette: Quill violet `#8b5cf6` / `#c4b5fd`; Ember orange `#f97316` / `#fdba74`; link teal-green `#14e3a8`; secret/value
  gold `#ffd166`; frozen weights ice `#9fd8ff`; wrong red `#f87171`. Background: deep violet-black void.
- The camera never sits still: slow dollies and orbits, with spring-eased moves on cuts. The motion is cinematic, not
  PowerPoint.
- The creatures are always alive. They breathe (squash ±2% on the beat via `beatPulse`), sway, blink every few
  seconds, look at each other and at objects, and react to things (happy eyes, a hop with squash and stretch, a lean-in).
  They are characters who are playing together.
- Typography: one `Caption` at a time (kicker + headline ≤ 7 words + sub ≤ 14 words), bottom-left by default at
  left 120 / bottom 110. A headline needs at least 2 s on screen and a sub needs at least 2.5 s. Use `Pin`s for short
  3D labels. Highlight key words in color with inline `<span style="color:…">`. Every shot shows its `Chapter` tag.
- On-beat accents: use `beatPulse(t)` for subtle pulses. Big hits are at the times in the cue sheet below; land
  motion on them.

## Exactness rules (non-negotiable)

1. **Nothing discrete crosses the link.** No letters, token tiles or text ever travel along the strands. Only
   glowing packets (continuous vectors) do. Tokens appear only as prompts entering a tail, or as text a model writes
   out of its head.
2. Exchanges sit **after blocks 6, 9, 12, 15, 18, 21, 24, 27** of 36. Both directions run at once, at the same depth,
   in one joint forward pass. The link is not attached to the first 6 or the last 8 blocks.
3. One read (reader x, sender y): `h_x += tanh(α) · size · O( attention( Q(norm h_x), [null; K(norm h_y)], [0; s · V(norm h_y)] ) )`.
   Q, O and α belong to the reader. K, V and the gate s = σ(w·norm(h_y)+b) belong to the sender. s is one number per
   sender token per exchange, and it **multiplies the values**, so a closed gate makes that position invisible. The
   null key has a zero value, so the reader can attend to nothing. There are 8 heads × 128 dims, in float32. tanh(α)
   starts at 0.1, so a write starts at about 5% of the residual's size. Trained writes change the residual by about 7–10%.
4. Time runs in ticks. A token at tick t reads the partner's tokens from tick ≤ t. Both prompts sit at tick −1. Each
   head has a learned bias by how long ago the partner token was written. Information flows only up in depth and
   forward in time. The joint pass equals tick-by-tick generation (difference 5×10⁻⁷).
5. What trains: the link plus rank-32 LoRA adapters on both models. The original weights are **frozen**.
6. Values in examples are real held-out test values: the word **lantern**, the number **5203**, the phrase
   **silver whale**, **Keiko lives in Geneva**, the sum **31 + 42 = 73**, the holiday city **Bangkok** (real model line:
   "Ah, Bangkok! … mango smoothie by the Chao Phraya River…").
7. Real tokenizations are in `data/tokens.json`: "Past the Phraya, 5203 lanterns glow." Qwen splits digits one by
   one, and AFM splits them "520"+"3" and "Phraya" as "raya". Matches are tokens that end at the same character.

## Cue sheet and shots

### S1 `s1_meet`, 0–10 s: "01 / THE PAIR" (bars 0–4)
- **0.0–3.4 Assembly.** Fade up from black. The void comes alive with dust. Two sparks ignite (violet left, orange
  right). Vertebrae fly in from the dark and stack bottom-up into spines (`assemble` 0→1; Quill 0.4→2.8, Ember
  0.8→3.2). The residual-stream cores light up block by block, and the heads pop on with a spring. Eyes open with a blink
  at about 3.0, look at the camera, then at each other. Camera: a slow push-in and rise.
  Caption 0.5–3.6: kicker `neural pairing` · head `Two language models.` · sub `Different companies. Different tokenizers. Different weights.`
- **3.6–6.6 Who they are.** The camera orbits. Pins: `QUILL · Qwen3-4B · Alibaba` (violet) and
  `EMBER · AFM-4.5B · Arcee` (orange). A block ruler beside one body ticks 0…35 as the blocks light bottom to top (a
  wave of `blockGlow`). Pin on the core: `residual stream · 2,560 wide`. Each creature swallows a short prompt in its
  own token style at its tail.
  Caption 3.6–6.6: head `Each is 36 blocks deep.` · sub `Text in at the bottom, the next token out the top.`
- **6.6–10.0 The rule.** Playful: Quill tries to toss a gold token tile (`lantern`) to Ember like a ball. A glassy
  hexagonal force-field flashes up between them, and the tile **shatters** into dust at **8.0** (big hit). Ember
  flinches, then both look at each other, and the 8 port rings on their bodies begin to glow and pulse (handoff to S2).
  Caption 6.6–8.3: head `Can they talk without words?` / caption 8.3–10.0: kicker `the rule` · head `No tokens. No text.` · sub `Only hidden states may pass.`
- End state at 10.0: home poses (Quill at x=−2.3, Ember at x=+2.3, as in the template), assembled, ports glowing
  (`portGlow` ~1.2), link not yet grown.

### S2 `s2_link`, 10–22 s: "02 / THE LINK" (bars 5–10). **Hero technical shot.**
- **10.0–14.0 The bridges grow.** The beat drops at 10.0. The strands of exchange k grow at **10.0 + 0.5k**
  (k=0..7, bottom to top: blocks 6, 9, …, 27), each with a flash at its ports. A vertical label column beside Ember
  reads `after block 6 · 9 · 12 · 15 · 18 · 21 · 24 · 27` (each label appears with its bridge). Packets flow both ways
  once a bridge is up. The camera sweeps around in 3/4 view.
  Caption 10.3–14.3: kicker `the link` · head `8 bridges between hidden states.` · sub `Same depth, both directions at once, inside one joint forward pass.`
- **14.0–14.5** The camera dives into the block-24 bridge (whoosh at 14.5).
- **14.5–21.2 One read, dissected (Ember reads Quill).** An exploded 3D schematic floats in space, built from the
  same visual language, and each piece animates as the matching term of the equation lights up in a HUD equation
  across the top:
  `h_E += tanh(α) · size · O( attention( Q(norm h_E), [null; K(norm h_Q)], [0; s · V(norm h_Q)] ) )`
  - 14.5–16.0: Quill's states for ticks −1, 0, 1, 2, 3 (violet orbs; tick 4 is ghosted and labelled `future: hidden`)
    each project a **key** cube and a **value** cube (violet = sender's weights K, V).
  - 16.0–17.4: each state has a **gate dial s** (a ring gauge). The values scale by their gate, and one gate closes so
    its value goes dark (label `gate closed → invisible`). A **null key** appears with an empty value `0`
    (label `null: attend to nothing`).
  - 17.4–19.0: Ember's state (orange orb) fires a **query** beam (orange = reader's weights Q). Attention bars
    rise over the keys with different heights (illustrative), including some on the null key.
  - 19.0–21.2: the values pour together, weighted by the bars, into one blended orb, pass through **O** (orange), and
    shrink by **× tanh(α) · size** (label `≈ 5–10% of the residual`). The orb then drops into Ember's residual stream
    with a "+" ripple. Small labels: `Q, O, α: reader` / `K, V, gate s: sender` / `8 heads × 128 dims`.
  Caption 14.8–21.0 (small, bottom): head `Ember reads Quill.` · sub `…and Quill reads Ember the same way, at all 8 depths.`
- **21.2–22.0** Snap back out to the full pair, with the link fully alive and both creatures delighted. Flash cut at 22.0.

### S3 `s3_play`, 22–36 s: "03 / THE GAMES" (bars 11–17)
- **22.0–26.0 Time runs in ticks.** Low side camera. Each creature writes tokens out of its head, one per tick, and
  they settle onto its own lane on the floor (Quill lane, Ember lane), running away from the camera along a tick
  ruler `−1 0 1 2 3 4 5`. The prompts sit at tick −1. A glowing "now" plane sweeps forward. At tick 3, an Ember token
  shoots thin teal read-lines back to Quill's tokens at ticks −1…3, while Quill's tick 4+ positions stay dark.
  Caption: kicker `time` · head `They write in ticks.` · sub `A token reads its partner's states up to its own tick. Never the future.`
- **26.0–30.5 Relay a word.** A gold secret card `lantern` drops into **Quill's** tail (Quill's prompt). Ember's
  prompt card at its tail reads `What is Quill's secret word?`. Ember cannot see the card: show this clearly with a
  dark occluder or blind spot. The link lights up and gold-tinted **packets** (not letters) stream Quill → Ember.
  Ember's eyes go wide, then happy, and it writes `The secret word is lantern.` as Ember-style token tiles from its
  head (`lantern` in gold) at about 29.5. Quill hops with joy.
  Caption: kicker `game · relay a word` · head `The word is only in Quill's prompt.` · sub `Ember writes it anyway.`
- **30.5–33.0 Seven value games (montage).** Fast and playful, one vignette per beat (30.5, 31.0, 31.5, 32.0, 32.5).
  The creatures bounce, and the output tiles pop with springs: `5203` (number) · `silver whale` (phrase) ·
  `31 + 42 = 73` (sum, one number from each) · `Keiko lives in Geneva.` (name + city, one from each) · a code-word
  lookup table. Caption head `Seven value games.` · sub `words · numbers · phrases · digit strings · sums · name + city · lookups`
- **33.0–36.0 Play.** A six-turn game, "where I'm on holiday". Quill's secret is Bangkok, carried by the link. Ember
  answers in a speech-bubble-like row of tiles: `Ah, Bangkok! … mango smoothie by the Chao Phraya River…`. Both are
  bouncing and laughing (happy eyes, crest flare).
  Caption: kicker `play` · head `They play games together.` · sub `Six-turn chats. Each holds a secret only the link can carry.`

### S4 `s4_train`, 36–46 s: "04 / TRAINING" (bars 18–22)
- **36.0–40.0 The fitted start.** Both creatures read the same sentence, which floats between them as two rows of
  tiles from `data/tokens.json` (Quill's square tiles above, Ember's pills below; real splits, so `5 2 0 3` vs `520 3`
  and `ray a` vs `raya`). Gold connectors snap between tokens that end at the same character. Show where they *don't*
  match too (no connector). Then a `ridge map` lens or matrix-plane `M` slides through, mapping states across. The
  reader's attention then lands on the matching token.
  Caption: kicker `training · step 0` · head `A fitted start.` · sub `Align tokens that end at the same character, then fit a map between their states.`
  Stat pin: `top attention on the matching token: 89–97%  (chance 1.2%)`.
- **40.0–43.0 Frozen and adapted.** Frost crystallizes up both bodies (`ice` 0→1) with crackle and sparkle, as
  the original weights freeze. LoRA chips light up teal on every block (`lora` 0→1), and the link stays bright. Gates
  lock (a small lock icon on the port rings, `gates held`).
  Caption: head `Original weights: frozen.` · sub `Only the link and rank-32 LoRA adapters learn.`
- **43.0–46.0 Curriculum.** A training-step counter races 0 → 900 (warm-up: tiny value-game glyphs zip through the link),
  then `full mix` cards burst into orbit around the pair: `shared reading` · `distillation` · `play` · `dialogue` ·
  `silent rounds` · `shared pen`. The tension builds to the hit at 46.0.
  Caption: head `Warm up on value games. Then the full mix.` · sub `900 warm-up steps, then reading, play, dialogue and more. One A100.`

### S5 `s5_proof`, 46–60 s: "05 / THE PROOF" (bars 23–29)
- **46.0–50.0 The shuffle control.** Ember is paired with the real Quill and writes `lantern` (teal check ✓). Then
  Quill slides out and a **grey ghost Quill** (`ghost: 1`, the partner from another episode, same game, same roles)
  slides in, and the link reconnects to it. Ember confidently writes a wrong word, `galaxy` (red ✗).
  Caption: kicker `the test` · head `Swap in a stranger.` · sub `Same game, same roles, another episode's partner. Does the secret vanish?`
- **50.0–53.0 The number.** The hit lands at 50.0. A field of **180 held-out episodes** (dots in a grid or ring)
  ignites in a cascade: 179 teal ("the real partner wins") and 1 grey. A huge counter reads `179 / 180`.
  Sub: `held-out episodes won by the real partner · all 7 games, both directions, p < .05`
- **53.0–55.6 More wins.** Three stat cards punch in on beats 53.0, 53.8 and 54.6:
  `31 / 32` shared-reading groups better with the real partner · `128 / 128` passed in turn structures it never trained on (dialogue, silent rounds, shared pen) ·
  `6–9%` of gates left open after training: they learned to whisper.
- **55.6–60.0 Title.** Pull back wide. The two creatures lean together, the link blazes, happy eyes, a little shared
  hop. Title card: **NEURAL PAIRING**, then `Two models. One link. No words.`, then a small
  `Qwen3-4B × AFM-4.5B · github.com/SauersML/neural-pairing`. Fade to black from 59.2 to 60.0.

## Audio (the `audio/` agent)
The score is made with the Patina synth (`/home/user/patina`). It follows the cue sheet above: a 120 bpm,
60.0 s cold open swell, the assembly arpeggio, the shatter hit at 8.0, the drop at 10.0, eight rising plucks at
10.0 + 0.5k, a whoosh at 14.5, a mechanical/glassy dissection section, an impact at 22.0, playful games, a resolve at
about 29.5 ("lantern"), staccato hits on 30.5–32.5, ice shimmer at 40, a riser to 46, a near-silent tension drop at 46,
a sour note at about 48 (the stranger), the big hit at 50, stat hits at 53.0/53.8/54.6, and a warm resolving outro
fading out by 60.
