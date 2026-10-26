# Neural pairing film, v4 brief (shared by every agent)

Project: /home/user/video (git repo; commit your own files often with clear messages, never `git add -A` other agents'
work-in-progress, never reset/checkout files you don't own). Research repo (read-only reference):
/home/user/np_src3/neural-pairing-main (README.md, pairing/link.py, pairing/session.py, pairing/games.py, results/,
figures/*.png).

Engine: Three.js 0.170 in headless Chromium (SwiftShader) via Playwright. A frame is a pure function of t
(`window.renderAt(t)`), 1920×1080, DOM HUD overlay. Read main.js, lib/*.js, STORYBOARD2.md, and the existing shots
before writing code. Preview: `node tools/preview.mjs --shots ID --times a,b,c --scale 0.5 --out out/preview/NAME --sheet`
(the machine has only 4 cores shared by several agents: preview a few frames at scale 0.5, look at them with Read,
iterate; never render whole sequences; never run tools/render.mjs).

## What the user said about v3 (the reasons for v4)
- The game windows look like cheap toys, lower quality and oversimplified compared with the rest. They must be as good
  as everything else.
- The film does not show how the system actually works. The architecture is not described well enough.
- Misleading visuals:
  1. They suggest **one-way** message passing. The link is **two-way and simultaneous** at every exchange depth:
     Quill reads Ember and Ember reads Quill at the same moment, using the states of that block from before the exchange.
  2. They often suggest **one message per token**. For every token there are **8 exchanges** (after blocks 6, 9, 12,
     15, 18, 21, 24, 27 of 36), each in both directions. And each read is attention over **all** visible partner
     positions, not one token to one token.
  3. They do not show **propagation of information through the layers**: what one block writes into the residual stream
     is there for every later block (it rises), and self-attention spreads it to later positions of the same model.
  4. The attention mechanism is weird and vague. It is not obvious what things represent.
- Some text is fine. Visual communication first, but use words when they help. Big type only.
- The word "neuralese" stays (no subtitle under it).

## Exact mechanism (pairing/link.py). Everything shown must agree with this.
- Quill = Qwen3-4B, Ember = AFM-4.5B. 36 transformer blocks each. They run block by block in lockstep in one joint pass.
- Exchanges after blocks 6, 9, 12, 15, 18, 21, 24, 27 (none in the first six blocks or the last eight).
- At an exchange, each model (reader x) reads the other (sender y) at that same depth:
  `h_x += tanh(alpha) * size * O( attention( Q(norm h_x), K(norm h_y), gate * V(norm h_y) ) )`
  - Reader owns: query projection, output projection, write size (tanh(alpha) × the residual's typical size; starts at
    0, trained to about 7–10% of the residual stream), a null key (lets a position read nothing), and a recency bias
    on attention scores by how long ago the partner token was produced.
  - Sender owns: key projection, value projection, and a visibility gate per position (sigmoid of the state it
    shows; it scales that position's value, starts about 88% open).
  - 8 heads × 128 dims per exchange.
  - Both reads at one exchange use states from before the exchange: simultaneous, symmetric.
  - The write is ADDED to the reader's residual stream (deep integration, not a message beside it). Later blocks of the
    reader compute on it, and at the next exchange depth the reader's (now changed) states are what the partner reads:
    so an idea can go Quill → Ember → Quill … up to 8 times inside one forward pass, for one token.
- Time: a position may read partner positions produced at the same time or earlier, and the partner's whole prompt.
  The two models have their own tokenizers, so positions don't line up one to one. Never say "tick" on screen.
- Training: rank-32 LoRA on each model (the original weights stay frozen; the LoRA update and the link train), from a
  fitted start (tokens matched where they end at the same character; a linear map fitted between hidden states: 89–97%
  vs 1.2% chance).
- Chat training (run nl22c): a privileged teacher. The original Ember, given both cities in its prompt, writes the
  reply ("How's the weather in Stockholm?"). The linked Ember sees only its own city and is trained to write the same
  reply, so the partner's city must come through the link.

## Results the film may use (real numbers only)
- 180 / 180 held-out values (nl20, all 7 games, both directions). nl22c keeps value games at 360 / 360.
- Shared reading 410 / 456. Never-trained games: lookups 35 / 40 (18/20 into Qwen, 17/20 into AFM); three untrained
  turn formats 128 / 128.
- Free chats (nl22c, cities never seen in training): partner's city said first in 13 / 23 chat sides with the link on,
  1 / 23 with the link cut, 0 / 23 linked to a stranger (another episode's partner); with a stranger, the stranger's
  city comes out in 14 / 36. Real opening lines (figures/hero_grid.png, results/hero_grid.json), e.g. Quill (whose
  partner Ember is in Taipei): "Hey Ember, how's the weather in Taipei? I hope you're keeping warm."
- Training curve nl20 (% real partner wins by step): 0: 52.8, 20: 66.0, 100: 88.9, 200: 96.5, 300: 97.9, 400: 96.5,
  500: 98.6, 600: 99.3, 700: 99.3, 800: 98.6, 900: 99.3.

## Writing rules for anything on screen (strict; the user has flagged each one)
- Big type: nothing under 34 px at 1920×1080; headings 64–110 px. No tiny labels, no tiny subtitles.
- Plain, direct words. No "carry", no metaphors, no anthropomorphic verbs, no punchy negations ("not X, but Y"), no
  rhetorical questions, no "rather than", no meta-commentary.
- No single-letter variable names (Q, K, V, α, h …): write "query", "key", "value", "gate", "write size".
- No ALL CAPS titles or kickers, no step/chapter numbers ("05"), no GitHub links, no "tick".
- The viewer either watches or reads: text appears when the picture holds, never over important motion. Few words.
- Creatures: cute, large eyes, no blush, nothing cringe.

## Visual language (keep it consistent across all shots)
- Quill violet (C.quill / C.quillHi), Ember orange (C.ember / C.emberHi), the link teal (C.link), a secret gold.
- A token is a tile with text (lib/tokens.js). A hidden state is a glowing orb or a short bar-code vector (never a
  letter). The residual stream is the vertical glowing column through the blocks (the creature's core). Blocks are the
  36 vertebrae. A write from the partner is the sender's colour injected into the reader's core, then rising.
- Both link directions always run at once. When a story beat follows one piece of content going one way (a secret
  going Quill → Ember), the other direction must still be visibly active.

## Music sync
120 bpm, 1 bar = 2 s, 16th = 0.125 s. Put every visual hit (impacts, pops, ✓ marks, cuts, flashes) on the 16th grid,
main hits on beats (multiples of 0.5 s). Document your hit times in a cue list (see Deliverables).

## Cut list v4 (shots/index.js, 96 s)
| id | global time | owner |
|---|---|---|
| a1_meet | 0–12 | done (director) |
| a2_link | 12–40 | architecture agent (new, authored in global time) |
| a3_cross | 40–50 (shift 8: authored for 32–42) | training agent (audit) |
| a4_games | 50–64 | games agent (new, authored in global time) |
| a5_train | 64–76 (shift 10: authored for 54–66) | training agent |
| a5b_teacher | 76–80 | training agent (new, authored in global time) |
| a6_proof | 80–96 (shift 14: authored for 66–82) | games agent (result windows only) |
`shift` means the shot's update(t) receives `t - shift` and its def.start/def.end are shifted too (main.js).

## File ownership (to avoid clobbering each other)
- Architecture agent: shots/a2_link*.js, and the shared libs lib/creature.js, lib/link.js, lib/grid.js (changes there
  must be additive: new options default to current behaviour, so other shots render exactly as before).
- Games agent: shots/a4_games*.js, shots/kit_gamewin.js, shots/a6_proof.js (only its result-window part),
  shots/s5_proof_demos.js, plus any new shots/games_*.js files.
- Training agent: shots/a3_cross*.js, shots/a5_train*.js, shots/a5b_teacher*.js.
- Anyone may add new files under lib/ with a unique prefix (e.g. lib/games_*.js). Don't edit files you don't own; if
  you need a change in one, write it in your report.

## Deliverables (each video agent)
1. Working shot code, committed.
2. A contact sheet of preview frames for your range (out/preview/<you>/sheet.png) at 0.5 scale.
3. `cues/<shot id>.md`: every timed event (global time, on the 16th grid) and what the music should do there.
4. A short report: what you built, how each user complaint is answered, known issues.
