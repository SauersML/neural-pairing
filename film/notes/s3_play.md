# S3 `s3_play` (22–36 s): notes

Files: `shots/s3_play.js` (camera, creatures, montage, play, captions), `shots/s3_play_sheets.js` (22–30.5 unrolled
view on `lib/grid.js`), `shots/s3_play_kit.js` (pure character curves: hop with anticipation/squash/stretch, nod,
blink, spring pop; bubble and lookup-table textures).
Final contact sheet: `out/preview/s3/final2_sheet.jpg` (times 22.3 … 35.95). Full-res checks after the director's review:
`out/preview/s3/v18full_35.60.jpg`, `out/preview/s3/v23full_32.70.jpg`, `out/preview/s3/v23full_33.20.jpg`.
No uppercase text or text-transform anywhere in the shot's files; captions use no kickers (lib switches them off).
Render cost: about 1.1–1.5 s per full-res frame (the first frame of a session is about 3 s because of warm-up).

Follows REVISION 2: captions word for word, Caption defaults (30/92/46), all pins ≥ 30 px, token tiles ≥ about 40 px
tall on screen, no chapter numbers, no links.

## Second by second
- **22.0–22.45** Flash-in on the pair from S2. Each body thins into a single column (one residual stream) and fades
  as the two sheets spread out of it: Quill's sheet on the left, Ember's on the right, rows aligned (row r = after block r−1;
  the row spacing equals the creature's block spacing). Teal rows are the 8 exchange rows.
- **22.35–25.9** Caption `time` / `One token per tick` / `A token attends to partner tokens from its tick or earlier.`
  Columns are labelled by tick (−1 for the prompt tokens: Quill `… lantern`, Ember `… word?`). Ticks 0–3 at 22.9 / 23.4 /
  23.9 / 24.4: both models write at once. Each new column is computed bottom to top (0.38 s), with causal self-attention
  arcs to earlier columns of the same model. The top is decoded and the token drops to the next column's input
  (Sky, looks, blue / The, word, is). A teal "now" band and the teal tick label follow the current tick.
  From 24.4, Ember's tick-3 position (`is`) attends to Quill's positions at ticks −1…3 at every exchange row (one row
  drawn as a full fan, the others faint). Quill's tick-4 column is a dark ghost labelled `not yet written`, and no beam
  reaches it.
- **26.05–30.3** Caption `relay a word` / `Only Quill's prompt has the word` / `Ember reads Quill's hidden states and writes it.`
  26.25–27.0: gold (the information of the prompt token `lantern`) rises up Quill's `lantern` column. Self-attention
  spreads a little of it to Quill's later columns. 27.05–29.15: Ember's `is` column is computed again, slowly (a white scan).
  At each exchange row the scan passes, a gold-tinted beam from Quill's `lantern` column (plus faint beams to Quill's
  other columns with tick ≤ 3) writes into Ember's column, and the gold is carried up by the residual stream.
  29.5: the top decodes to the token **lantern** (large gold tile, ring burst). 29.6–29.9: the tile drops into Ember's
  tick-4 column as its input.
- **29.95–30.45** The sheets roll back up into the creatures (bodies widen back from one column). Quill hops with joy
  (crouch, stretch, land squash, crest flare), and Ember beams. The camera swings to the montage framing.
- **30.32–33.1** Caption `Seven value games` / `words, numbers, phrases, digit strings, sums, name + city, lookups`.
  One vignette per beat. The value chips drop into the holder's tail (gold pulse climbs that spine), and gold dots run
  along the strands toward the receiver. The receiver hops, nods and goes happy-eyed. Its answer tokens leave its head
  into one clean row above the pair (each token pops out at its own slot, nudged in from the head, so no tile
  crosses another), then the row flies onto a shelf on the right (the shelf starts with `lantern`
  from the relay and holds all six rows until 33.25):
  30.5 Quill `5203` → Ember `520`·`3` (AFM digit split) · 31.0 Ember `silver whale` → Quill `silver`·`whale` ·
  31.5 Quill `31`, Ember `42` → Ember `31 + 42 = 73` · 32.0 Quill `Keiko`, Ember `Geneva` → Quill `Keiko lives in Geneva.` ·
  32.5 Ember holds a code table (`code K7  candle`, `code M2  breeze`, held-out words) and Quill holds the key `code M2`
  in its tail → Quill writes `code M2 → breeze` (shortened from the real answer `Code M2 holds the word breeze.`, at the
  director's request; codes are always labelled "code").
  Partner-sourced values are gold.
- **33.12–35.97** Caption `play` / `Six-turn games` / `Each model names a secret that appears only in the other's prompt.`
  33.1: gold `Bangkok` card drops into Quill's tail (pin `Quill's secret`), and a gold pulse climbs its spine. 33.6+: gold
  dots stream Quill → Ember, and Ember's ports catch it. 33.82: speech bubble springs from Ember's head. Tokens fly out of its
  mouth one by one: `Ah, Bangkok! …` / `mango smoothie by the` / `Chao Phraya River…` (`Phraya` shown as one tile at the director's request; AFM actually splits it `Ph`+`raya`).
  From 34.1 both are laughing: happy eyes, giggle wobble, crest flare, leaning in, alternating hops on 34.1/34.55/35.0/35.5.

## Lib notes and workarounds (no lib files edited)
- `grid.js` has no per-column reveal. After `sheet.end(t)` I set each stream's `uReveal`/`uAmp` and black out or dim node
  instance colors per column (future columns are ghosts). I also show and scale the sheet's input tiles per tick, and
  compensate `group.scale.x` (used for the unroll).
- `LinkBeams` always colors the beam tube teal. I record the pool index of the gold beams and recolor those meshes
  after `beams.end()`. I also enlarged `beams.packets.material.size` (0.16 → 0.3) on my instance.
- Bloom blurs tile text. I return `bloomThreshold` 0.88 in the sheet section and 0.86 in the creature sections
  (bloomStrength 1.25), and draw tiles at about 0.8 color, so all token text stays sharp.
- `Creature` has no per-axis scale or "wide eyes" field. The unroll squeezes `group.scale.x/z` after `update()`, and
  the crest scale is set after `update()` too. Both are re-set every frame, so the shot stays pure.
- `Link` packets have no color input. The gold tint is applied to the packet color buffer after `link.update()`, plus my
  own larger gold dot layer that rides `strand.curve`. Dots only; no glyph ever travels on the link or on the beams.
