# Game windows: a4_games (42–54) and the a6_proof result windows (73.0–77.8)

## Files
- `shots/kit_gamewin.js` is the shared kit, used by both shots. It holds the mini bodies, the strands, the prompt cards,
  the typed tiles, one prop scene per game, `makeGameWindow`, and `makeResultWindows` for a6.
- `shots/a4_games.js` is the wall.
- `shots/a6_proof.js` changed in two lines only: the import, and the `cards = makeResultWindows(hud, { top: 90 })`
  constructor. `cards.update(t, K.cards, K.cardsOut)` is unchanged, so the pops stay on 73.0 / 73.8 / 74.6 / 75.4 and
  the exit at 77.8. `s5_proof_demos.js` is no longer used by a6 and is untouched.

## The window grammar (kit)
- **Mini bodies:** 36 glassy blocks around a glowing core (the residual stream), with teal port rings on blocks 6, 9, …, 27.
  The heads have the 3D creatures' eyes and crests. Every token is one forward pass: a wave lights the blocks bottom to
  top, in lockstep on both bodies.
- **Link:** 8 strands from core to core at the exchange depths. Only dots ride them. The value pass sends gold packets in
  the holder's direction (the other direction carries small sender-coloured packets, because the link is duplex). Each
  written token fires a pass both ways. Gold enters the receiver's core at the first exchange and rises to its head,
  then the head glows.
- **Prompt cards:** each card sits at its model's tail, with a feed line into the tail. The value appears only on the
  holder's card, in gold, and that card glows while it is read.
- **Answers:** the real target sentence types out of the writer's head token by token. Quill uses square tiles and splits
  digits one per token. Ember uses round pills. Value tokens are gold, then a ✓ punches in. In the relay games, the
  sender types a grey reply in lockstep (its own answer to its own question).
- **Props:** one scene per game. The value prop shows only on the holder's side until it arrives:
  - word: a lantern cradled by Quill; an identical lantern ignites over Ember's head.
  - number: Quill's gold `5203` slip; Ember's padlock dials roll 5-2-0-3 on sixteenths, then the shackle springs open.
  - phrase: Ember's silver whale; on Quill's side, a whale outline draws itself and fills with liquid silver.
  - digits: Ember's tape; a tape prints out of Quill's side digit by digit, in sync with Quill's digit tokens, and the
    holder's matching digit lights at the same moment.
  - sum: column addition on both sides. The partner's number arrives as a dashed ghost, then the rule draws and `73`
    lands with the `7 3` tokens.
  - fact: a portrait and a map on both sides. Keiko walks a dotted path into the Geneva pin.
  - lookup: the key's packets arrive and Ember's code drawer is scanned down to the `M2: breeze` row. Wind gusts fly
    back and a breeze swirl appears over Quill's key tag. BOTH models write `Code M2 holds the word breeze.` in lockstep,
    and both get a ✓ (`games.py` gives both the same target: the asker needs the word, the holder needs the code).
    The narrow a6 window shows a 2-row drawer and has a taller answer area.
  - reading: Quill's open book; Ember's door with a question mark fills red.
  - play / free chat: postcards. Each side has its own card face up and the partner's face down; each face-down card
    flips when that side's chat bubble names the city.
  - formats (a6): dialogue, then silent rounds, then shared pen, cycling in one window, each ending with `anchor` and a ✓.

## Accuracy
- Targets are the real `games.py` targets:
  - relays: `The secret word is lantern.`, `The number is 5203.`, `The phrase is "silver whale".`, and
    `The digits are 7 3 0 1 9 4 8 6.` (digits space-separated, as in `games.py`).
  - sum: `Mine is 31, Ember's is 42, and the sum is 73.` from Quill, mirrored from Ember.
  - fact: `Keiko lives in Geneva.` from both.
  - lookup: `Code M2 holds the word breeze.` (both the asker and the holder).
  - dialogue (long_relay): `You sent me the word anchor.`
- Values come from the held-out pools: lantern, silver (held-out adjective), whale, Keiko, Geneva, Bangkok, Cairo,
  Seattle, Rome, breeze, aroma, alley, atom and anchor.
  - Numbers obey the held-out digit-sum rule: 5203 and 73019486 both sum to 3 modulo 7.
  - The sum uses held-out values (50–99).
  - `73019486` is a stand-in, because the results files store no digit strings.
- Prompt-card wording is shortened from `games.py` (lookup keys use the real letter+digit format and the `K7: word`
  row format). The play line `Ah, Bangkok! … mango smoothie by the Chao Phraya River…` is the real model line. The
  short openers `You're in Cairo!`, `You're in Rome!` and `And you're in Seattle!` are written for the film.
- Token splits: Quill's digits are split one per token (Qwen behaviour), and Ember's `5203` is `520` + `3`. `M2` stays one
  tile, to avoid a single-letter tile.

## a4 timeline (120 bpm)
Each wave shows one row of three 600 × 940 windows. They slam in on eighths (W, W+0.25, W+0.5) with a camera shake,
every link ignites on W+1.0, the values cross and the props act, tokens type on sixteenths (thirty-seconds for the long
digit and sum sentences), and the checks punch on W+2.5 + 0.125·column. The camera whips down a row at W+2.8–3.0.
- wave A 42–45: word, number, phrase.
- wave B 45–48: digits, sum, fact.
- wave C 48–51: lookup, reading, play. The last check lands about 50.85.
- 51.0–51.5 pull back to the whole wall. All text hides while it is small; the bodies, props and checks stay.
- 51.55–52.4 `Training games`, over a scrim.
- 52.45 the windows fold into a deck, which pulses on 53.0 and flies off by 53.92, leaving a teal spark.

## a6
Four 440 × 900 windows, with the stat (76 px) and caption under each scene:
- `35 / 40` untrained game (lookup).
- `128 / 128` untrained formats.
- `410 / 456` shared reading.
- `14 / 16` free chat, with the grey `0 / 16 with a stranger`.

Each link fires at hit+0.45. Every window finishes its ✓ before the 77.8 exit (free chat about 77.1).

## Checks
- All text is at least 30 px (tiles, cards and bubbles 30, titles 34–38, stats 76).
- Purity: a4 at 47.7 and a6 at 76.9 render identically before and after jumping to other times.
- Full-res frames: about 3.5–7.5 s under a load average of 6–8 (shared machine).
- Sheets: `out/preview/a4/final2_sheet.jpg` (a4), `out/preview/a4/a6final_sheet.jpg` (a6).
- Full-res checks: `out/preview/a4/k1full_44.60.jpg`, `k1full_47.65.jpg`, `k1full_50.85.jpg`, `a6full_77.30.jpg`.
