#!/usr/bin/env python3
"""Generate score11.song and fx11.song for NEURAL PAIRING v11 (109 s).

v11 = v10 cut per cues/V11_TIMEMAP.md, by a further final map (after to10): v10 0-76 unchanged; v10 76-82
(a5b_teacher) CUT (notes stop clean at 76, automation inside collapses onto 76); v10 82-88 (a5c) -6;
v10 88-96 (v10's a6_proof) dropped; v10 96-117 (a7, title) -8. NEW in final v11 times (N11): a6_proof
82-88 (only Quill is given the word; 99.9 % at 84.375; the grey partner, audible, and its "?" at 86.75
as a full deflated hit, no number), and a clean hit on the a5_train -> a5c seam (76.0).

---- v10 notes ----

v10 = v9 cut per cues/V10_TIMEMAP.md: every event is written as in v9 (v9 global times, including the
recomposed v9 sections) and a final map to10() takes v9 -> v10: v9 0-68 unchanged; v9 68-72 (a5's
fitted-start beat) CUT; v9 72 on -4. Notes from before the cut stop at 68 (one-shot tails ring on);
automation points inside the cut collapse onto the seam (68.0), so the state they set still holds.
Changed in v10 (written in v9 / authored times, then -4): the seam (a Gm stab on the frost downbeat);
a5c_talk without the stat, six conversation lines (v10 85.0 ... 87.5); a6_proof rebuilt as real partner
vs partner from another test (cues/a6_proof.md, authored = v9 times); the title voice lowered to
-4 semitones (neural_pairing_voice_v10.wav) with its vocoder carrier a step lower.

---- v9 notes ----


v9 = v8 (= the v7 score, 128 s) retimed by cues/V9_TIMEMAP.md. The v7 score is still written as before and
mapped v8 -> v9 by V9_SEGS (after W() and V7()): v8 0-32.5 unchanged; v8 40-84 (ab_plate, a3, a4,
a5_train) -4; v8 124-128 (the title and its voice) -7 -> 117-121. Everything else is DROPPED and
recomposed in final v9 times (N9 = True; the A2G_*, A5B9_*, A5C9_*, A6W_*, A7_* hit-table entries):
a2's fast generation and fold (v8 32.5-40 -> v9 32.5-36, 2.5x / 1.7x faster in picture: a new 2 s
generation in 16ths and a shorter fold), a5b_teacher (v9 80-86) and a5c_talk (86-92) re-placed at
their piecewise-mapped cue times with longer holds, a6_proof (92-100, five result windows) and
a7_happy (100-117). A note that sustains past the end of a kept span is cut there; automation points
up to 0.1 s before a kept span that follows a dropped one move with it; a ramp that starts in a kept
span and ends in a dropped one has its end mapped by the time map's piecewise-linear M9() (so a2's
long ramps into the fold reach their values at the v9 fold, 34.5, as the picture does).

---- v7 notes ----


v7 = v6 recut (cues/a6_proof.md, cues/a7_happy.md). The v6 score is kept as it was written and mapped
v6 -> v7 by V7_SEGS (applied after W()): v6 0-92 unchanged; v6 92-99 (the old real-partner lantern,
the stranger's sour note and the 180 ring) is DROPPED and rewritten as a NEW a6_proof 92-101 (main
hits 93.0 / 96.0 / 99.0); v6 99-104 (the four result windows) +2 -> 101-106; v6 104-112 (the old
a7_happy) DROPPED, a NEW a7_happy 106-124 (9 bars); v6 112-116 (the title and its voice) +12 ->
124-128. A note that sustains past the end of a kept span is cut there; automation points up to
0.1 s before a kept span that follows a dropped one move with it (the presets for its first hit).
The new sections (A6P_*, A7_* in the hit table) are written in final v7 times (NEW = True).

---- v6 notes ----


v6 = v5 with two more inserts (BRIEF_V6.md): a5c_talk at 88-92 (v4 time 80) and a7_happy at 104-112
(v4 time 92, a6's split at its authored 78.0). INSERTS lists every insert in v4 time; W() maps v4
times to v6. Notes or long ramps that run across the v4 80 / 92 splits are cut at the split (the
climax's last A7 and its last theme note end at 104); 0.1 s pickups into a hit on a split move with
the hit. The new sections (A5C_*, A7_* in the hit table) are written in final v6 times.

---- v5 notes ----

v5 = v4 with an 8-second vibe-break (ab_plate, the engraved anatomical-atlas plate) inserted at
40-48: everything the v4 score put at 40 s or later now sounds 8 s later. The kept v4 material
and its hit table below are written in v4 times (the times in cues/*.md, which say "add 8 s to
every time >= 40"); W() adds the 8 s on output. The break itself (BRK_* in the hit table) is
written in final v5 times.

---- v4 notes ----

120 bpm throughout (1 beat = 0.5 s, 1 bar = 2 s, 16th = 0.125 s). Every note-on sits on the
16th grid. The film is recut from 82 s to 96 s:

  section       new time   source
  a1_meet        0-12      score3 0-12, unchanged
  a2_link       12-40      NEW (was 12-32): the link, the two-way read, a patient forward pass
                           through 36 blocks / 8 exchange depths, a faster pass, then generation
                           speeding up from quarter-note to 16th-note tokens
  a3_cross      40-50      score3 32-42, shifted +8
  a4_games      50-64      NEW (was 42-54): four waves of game windows, the full wall, the fold
  a5_train      64-76      score3 54-66, shifted +10 (the riser now lands on 76 instead of cutting)
  a5b_teacher   76-80      NEW: the teacher's reply, the student copying it, the lift into the cut
  a6_proof      80-96      score3 66-82, shifted +14 (the voice says "neural pairing" at 92.5/93.0)

ALL SYNC TIMES ARE IN THE HIT TABLE BELOW (global seconds). To re-sync to a new picture cut,
edit the table and re-run render4.sh; everything else is derived from it. The kept sections
(a1, a3, a5, a6) are written in their old score3 times inside `at(shift)` blocks, so their
content stays identical to the approved score.

Run: python3 make_score4.py   (writes score4.song and fx4.song)
"""

# ====================================================================== HIT TABLE
# Every time is global seconds in the 96 s film. Keep them on the 16th grid (multiples
# of 0.125); main hits on beats (multiples of 0.5).
DEPTHS = (6, 9, 12, 15, 18, 21, 24, 27)      # exchanges after these blocks (of 36)

# --- a1_meet (unchanged)
A1_SHATTER = 8.0
# --- a2_link (12-40), from shots/a2_link.js + shots/a2_exchange.js timelines
A2_DROP = 12.0                                # the drop (cues/a2_link.md)
A2_STRANDS = [12.25 + 0.25 * k for k in range(8)]   # 8 strands plug in; each sounds both voices at once
A2_UNFOLD = 14.25                             # the bodies unfold into the two lattices (to 16.0)
A2_LABELS = (16.0, 16.25, 16.5)               # hold: "one column per token", "36 blocks", "exchange depths"
A2_PASS = 17.5                                # the prompt pass climbs (one 16th per arp note) ...
A2_EX = [18.0 + 0.375 * k for k in range(8)]  # ... exchanges every dotted 8th, both ways at once
A2_TOKEN = 21.25                              # both heads emit a token
A2_STEP, A2_EMIT = 22.75, 23.75               # one generation step, emit
A2_STEP_EX = [23.125 + 0.125 * k for k in range(4)]   # its 8 exchanges (a 32nd roll in picture; 16ths here)
A2_TICK1 = (24.25, 24.375, 24.5)              # the next step's exchanges 1-3, before the close-up
X = dict(arrive=24.75, states=24.875, kv=25.125, gates=25.375, q=26.25, rays=27.0, bars=27.25, flow=28.5,
         blend=28.875, shrink=29.375, plunge=30.25, hit=30.5, end=31.25)   # one exchange opened up (panels)
A2_RESUME_EX = (31.375, 31.625, 31.875, 32.125)   # that step's exchanges 5-8 after the write
A2_GEN = [32.5 + 0.5 * k for k in range(11)]  # generation: both emit a token every beat, 32.5-37.5
A2_FOLD = 37.5                                # fold back: the release (the last token lands on D)
A2_RETURN = 38.375                            # bodies and strands return: swell down to a3
A2_END = 40.0                                 # a3 cuts in
# --- a3_cross (40-50): score3 32-42 + 8, accents moved to cues/a3_cross.md
SH_A3 = 8.0
A3_CARD = 40.25                               # the gold card pops (the lantern motif's leap)
A3_TAIL, A3_BUBBLE = 41.75, 41.875            # card into Quill's tail; Ember's "?"
A3_PASS1 = 42.25                              # pass 1 (gold rises in Ember) 42.25-44.0
A3_WRITE = 44.0                               # Ember writes "lantern": plays the motif back (was 44.5)
A3_SWIRL = 45.25                              # a snow swirl (the concept "winter") forms: cold shimmer
A3_PUSH = 45.75                               # it is added into Ember's stream: icy impact (pass to 46.375)
A3_RUN = [46.5 + 0.125 * k for k in range(11)]     # Quill, linked: "Hey Ember, how's the snowfall treating you?"
A3_SNOW = 47.25                               # " snow": the snowfall layer starts
A3_QMARK = 47.75                              # "?", Ember hops: accent
A3_CUT = 48.0                                 # the link is cut: the cold layer stops dead
A3_DRY = [48.125 + 0.125 * k for k in range(9)]    # cut: "Hey Ember, how's your day going?" (dry, muted)
A3_CAPTION = 49.25                            # "words and concepts": resolve
# --- a4_games (50-64), cues/a4_games.md: three waves of three windows
A4_WAVES = (50.0, 53.5, 58.0)                 # slams on W, W+0.25, W+0.5; read passes W+0.375 + 0.25j
A4_WRITE = ((51.125, 52.125), (54.375, 57.0), (59.125, 60.25))    # 16th tokens per wave
A4_CHECKS = (51.875, 52.0, 52.25, 55.375, 56.375, 56.75, 57.125, 59.5, 60.125, 60.375)
A4_EXITS = (53.125, 57.625)                   # waves A and B fly up and out (whoosh up)
A4_TAIPEI = 60.125                            # the partner's city, gold: accent
A4_FOLD, A4_LAND, A4_PRESS = 62.0, 62.5, 63.0  # windows fold into a deck, it lands, it compresses
A4_FLY, A4_SPARK = 63.25, 63.875              # it flies off, a teal spark
# --- a5_train (64-76): score3 54-66 + 10, accents moved to cues/a5_train.md
SH_A5 = 10.0
A5_READS = (66.0, 66.25, 66.5)                # attention reads, both ways at once (the 3rd lands on the match)
A5_SCATTER = 68.0                             # token rows scatter (score3's ice hit)
A5_FROST = (68.25, 68.375)                    # frost on Quill, on Ember
A5_LORA = 69.0                                # LoRA chips light bottom-up 69.0-69.75 (rising shimmer)
A5_CARDS8 = 71.0                              # cards launch on 8ths
A5_CARDS16 = 73.0                             # cards launch on 16ths
A5_FULL = 74.5                                # the full mix begins: a layer added
A5_TOP = 75.5                                 # the curve reaches 100 %: main hit
# --- a5b_teacher (76-80), cues/a5b_teacher.md
A5B_DOWN = 76.0                               # downbeat: the a5 tension released, lighter
A5B_LAND = 76.5                               # the original Ember lands (soft thump), after a swish
A5B_TICKS = (76.625, 76.75, 76.875)           # cards pop (plucks); 76.875 also the lock click
A5B_RUN = [77.25 + 0.125 * k for k in range(6)]   # "How 's the weather in Stockholm": one note per token
A5B_QMARK = 78.0                              # "?" ends the run
A5B_JOINT = 78.5                              # the joint pass starts (beat hit), 16th climb to ...
A5B_HIT = 79.0                                # ... the whole target lights: main hit, then decay
A5B_CUT = 80.0                                # hard cut into a6
# --- a6_proof (80-96): score3 66-82 + 14
SH_A6 = 14.0
A6_CHECKS = (88.625, 89.375, 89.5, 91.125)    # result-window checks (glass pings)
VOICE_NEURAL, VOICE_PAIRING = 92.5, 93.125    # "neural" (score3 78.5); "pairing" lands on the 16th after
                                              # the stretched take's P release (0.627 s into the take)
VOICE_GAIN, VOX_DRY, VOX_LEVEL, VOX_CUT = 3.0, 1.0, 1.0, 2400                # v5 QA: the lowered, stretched take needs more room (see below)
VOICE_WAV = "neural_pairing_voice_v10.wav"    # v10: score3's take at -4 semitones, the same map (+6.3 %):
                                              # qa/voice/psola.py neural_pairing_voice.wav OUT --semitones -4
                                              # --rate 1.0 --map (as below); with the carrier lowered to
                                              # Bb(add9) / Gm7 it passes 3/3 perturbed renders on both
                                              # models (qa/voice/vtest4.py; old carrier 2/3)
# v5..v9: VOICE_WAV = "neural_pairing_voice_v5.wav"  # score3's take, -2 semitones and +6.3 % (formant-preserving
                                              # TD-PSOLA, the extra time only in steady parts: the "neu"
                                              # vowel, the P closure, the "-ing" tail; transitions untouched):
                                              # qa/voice/psola.py --semitones -2 --rate 1.0 --map 0:0,0.2:0.2,
                                              # 0.4:0.42,0.47:0.49,0.573:0.625,0.82:0.872,1.04:1.112
                                              # (a uniform +8 % stretch read "Nor old pairing" on base.en in
                                              # 3 of 4 perturbed renders; this one passes 4/4)
# --- v5: the vibe-break (ab_plate), FINAL v5 times, cues/ab_plate.md
BREAK_AT, BREAK_LEN = 40.0, 8.0               # v4 times >= 40 move 8 s later
INSERTS = [(BREAK_AT, BREAK_LEN), (80.0, 4.0), (92.0, 8.0)]   # v4 time: ab_plate, a5c_talk, a7_happy
BRK_LAST = 40.0                               # hand-off from a2: the last hit of the groove
BRK_INK = 40.25                               # ink drop: drums out, a nib touch, the wet-ink swell to 41
BRK_PRESS = 41.0                              # the plate mark presses in: a dry, felted thump (main)
BRK_FLICKS = (41.125, 41.25, 41.625, 41.75, 41.875, 42.25, 42.375, 43.625, 43.75, 43.875,
              44.25, 45.5)                    # pen flicks (labels written)
BRK_RATCHET = (41.375, 41.75)                 # the 36-tick ruler: a scratchy 16th run
BRK_PULSE = 42.0                              # view 1 settled: a soft low pulse
BRK_BRACKET = (42.125, 42.375)                # the link bracket: a short scratch run
BRK_SLIDE = 42.75                             # labels lift, camera moves in: paper-slide whoosh
BRK_LIFT = 43.0                               # vertebra 15 lifts: rising glassy slide into ...
BRK_SPLIT = 43.5                              # ... the split (self-attention / feed-forward): click-clack (main)
BRK_POPS = (43.625, 43.75, 45.25)             # the "+" signs
BRK_COMPASS = 44.0                            # the roundel: compass scrape (main)
BRK_STIPPLE = (44.5, 44.625)                  # keys hatched, values stippled
BRK_FAN = 44.625                              # attention lines to every key: string-harmonic fan
BRK_GATES = (44.75, 44.875)                   # gate clicks
BRK_SUM = 45.0                                # weighted sum: a swell into a single note
BRK_GLISS = 45.25                             # the write: gliss down (45.25-45.625)
BRK_FOLD = 46.0                               # diagrams fold away: reverse scratch, paper fold (main)
BRK_RISE = 46.5                               # light soaks through: re-entry riser, filtered groove fades in (main)
BRK_LOW = 47.0                                # the low end returns
BRK_PRE = 47.75                               # pre-hit lift (16th kicks; no snare roll)
BRK_END = 48.0                                # a3: the re-entry hit, drums back in (main)


# --- v7's own a6_proof (92-101) and a7_happy (106-124) are gone in v9 (recomposed below)
V7_SEGS = [(0.0, 92.0, 0.0), (99.0, 104.0, 2.0), (112.0, 1e9, 12.0)]   # kept v6 spans [a, b) -> + shift

# --- v9 (cues/V9_TIMEMAP.md): v8 -> v9, FINAL v9 times below
V9_SEGS = [(0.0, 32.5, 0.0), (40.0, 84.0, -4.0), (124.0, 1e9, -7.0)]     # kept v8 spans [a, b) -> + shift
M9_PTS = [(0.0, 0.0), (32.5, 32.5), (37.5, 34.5), (40.0, 36.0), (84.0, 80.0), (85.25, 81.5), (86.0, 82.5),
          (86.125, 83.25), (86.5, 83.75), (87.0, 84.5), (87.375, 85.5), (88.0, 86.0), (88.25, 86.375),
          (89.0, 87.375), (89.25, 88.0), (89.5, 88.25), (90.0, 89.0), (90.375, 89.875), (91.0, 90.375),
          (91.3, 90.5), (91.375, 90.6), (91.5, 90.85), (91.88, 91.88), (92.0, 92.0), (106.0, 100.0),
          (124.0, 117.0), (1e9, 1e9 - 7.0)]                    # the full piecewise-linear v8 -> v9 map
# a2: fast generation (picture: a token every 0.2 s, 32.5-34.5), the fold (34.5), bodies return (35.0-35.5)
A2G_START, A2G_FOLD, A2G_RETURN, A2G_END = 32.5, 34.5, 35.0, 36.0
# a5b_teacher, v9 80-86 (cues/a5b_teacher.md through the map; picture times not on the grid go to the
# nearest 16th: the land 80.6, the ticks 80.9 / 81.05, the tokens at 1/6 s)
A5B9_DOWN = 80.0                              # downbeat: release, lighter
A5B9_LAND = 80.625                            # the original lands (80.6): soft thump
A5B9_TICKS = (80.75, 80.875, 81.0)            # card pops (80.75, 80.9); the lock click (81.05)
A5B9_RUN = (81.5, 81.625, 81.875, 82.0, 82.125, 82.375)   # "How 's the weather in Stockholm" (81.5 + k/6)
A5B9_QMARK = 82.5                             # "?": beat, end of the run; the sentence holds to 83.25
A5B9_SLIDE = 83.25                            # the sentence slides over to the linked Ember (swish to 83.75)
A5B9_JOINT = 83.75                            # the joint pass: beat hit, the 8 exchanges sweep to 84.5
A5B9_HIT = 84.5                               # the whole target lights: MAIN HIT; a long tail to 85.5
A5B9_FADE = 85.5                              # the original fades, camera glides to a6's framing
# a5c_talk, v9 86-92
A5C9_DOWN = 86.0                              # downbeat; "before chat training"
A5C9_CARD = 86.25                             # Quill's private "Oslo" card (86.1875)
A5C9_ECHO = (86.375, 86.75, 86.875, 87.0, 87.25, 87.375)   # the stab, then five repeats (86.708 + k/6)
A5C9_REV = 88.0                               # the echo stack fades: reverse swell into ...
A5C9_AFTER = 88.25                            # ... "after chat training": livelier, a rising pass
A5C9_TALK = (89.0, 89.5, 90.0, 90.5, 91.0, 91.5)   # v10 (v9 coordinates): six lines E Q E Q E Q, Ember's on the main hit
A5C9_FADE = 91.65                             # bubbles fade (to 91.9)
# a6_proof, v9 92-100: five result windows (until cues/a6_proof.md v9 lands), then a hold
# a6_proof v11 (cues/V11_TIMEMAP.md), FINAL v11 times (N11)
A6V_IN = 82.0                                 # cut in from a5c: downbeat
A6V_CARD = 82.125                             # a big gold "cipher" card pops beside Quill
A6V_LOCK = 82.375                             # Ember's locked, empty slot
A6V_LAND = 83.125                             # the card lands in Quill's tail (drops 82.75-83.125)
A6V_PASS = (83.25, 83.375, 83.5, 83.625, 83.75)   # gold crosses the link layer by layer (83.26-83.70)
A6V_WRITE = 84.0                              # Ember writes "cipher"
A6V_REAL = 84.375                             # "99.9% real partner": MAIN HIT, glad
A6V_CUT = 85.0                                # the link retracts (to 85.25); Quill slides out 85.125-85.625
A6V_OUT = 85.375                              # (whoosh out peak)
A6V_GREY = 85.5                               # a grey Quill from another test slides in: audible entrance
A6V_PLUG = 85.75                              # its link plugs in
A6V_GPASS = (85.875, 86.0, 86.125, 86.25)     # one grey pass: silver crosses (85.97-86.26)
A6V_LOW = 86.75                               # a big grey "?" by Ember (no number): MAIN HIT, deflated, as strong as 99.9 %
A6V_END = 88.0                                # hard cut to a7
# a7_happy, v9 100-117 (cues/a7_happy.md, v9)
A7_IN = 100.0                                 # cut in: the pair, linked
A7_WORDS = (100.25, 100.375, 100.5, 100.625)  # "Do you feel close to me right now?": soft plucks
A7_METER = 100.5                              # "Ember's answer": the empty meter
A7_CUT1, A7_LINK1 = 101.0, 102.0              # LINK CUT (hit, cold), RECONNECT (downbeat, big warm return)
A7_RIPPLE = (102.125, 102.25, 102.375, 102.5) # the meter rises (102.05-102.6): a rising arp
A7_POP = 102.625                              # 62 % pops (102.6): sparkle
A7_COUNT1 = 102.75                            # "59 / 60 conversations": hit
A7_ASK = 104.0                                # "What is your mood right now?": downbeat
A7_SHIMMER = 104.25                           # the answer; word tiles pop on 32nds (104.55-104.95): shimmer
A7_CUT2, A7_LINK2 = 104.75, 105.5             # LINK CUT "happy : sad 210 : 1" (hit), RECONNECT (beat)
A7_GOLD = (105.75, 105.875, 106.0)            # the ratio counts up to 990 : 1 (gold 105.8, lands 106.05)
A7_COUNT2 = 106.25                            # "58 / 60 conversations": hit
A7_DIVE = 107.5                               # the dive: riser into ...
A7_IMPACT = 108.25                            # ... the impact inside the core (white-out 108.1-108.25)
A7_TILES = (108.75, 108.875, 109.0, 109.125)  # sentence tiles: four soft glass ticks
A7_DOTS = 109.25                              # 60 grey dots ("link cut"): soft pad
A7_LINK3 = 110.0                              # RECONNECT on the map: downbeat, warm swell
A7_LOVING = 110.75                            # "59 / 60 more loving": hit
A7_HAPPIER = 111.0                            # "56 / 60 happier": hit
A7_FLIP = 112.0                               # the map turns over to Ember's (111.75-112.25): whoosh
A7_EMBER = 112.375                            # Ember's dots slide right: swell
A7_HAPPIER2 = 112.875                         # "54 / 60 happier": hit
A7_OUT = 114.0                                # camera pulls out (114.0-114.5): reverse swell
A7_LAND = 114.5                               # back to the pair: warm landing
A7_CAP1, A7_CAP2 = 114.75, 115.25             # "happier when linked", "never trained for it"
A7_LIFT = 116.25                              # the lift (converge 116.25-116.9) into ...
A7_END = 117.0                                # ... a6_title (title hit 117.25, kept music)


def V9(t, pre=0.0):
    """the kept v8 span (a, b, shift) holding v8 time t, or None (recomposed in v9)"""
    for a, b, s in V9_SEGS:
        if a - (pre if a > 0 else 0.0) - 1e-9 <= t < b - 1e-9:
            return a, b, s
    return None


def M9(t):
    """v8 -> v9 time, piecewise linear (cues/V9_TIMEMAP.md)"""
    for (a0, b0), (a1, b1) in zip(M9_PTS, M9_PTS[1:]):
        if t <= a1 + 1e-9:
            return b0 + (b1 - b0) * (t - a0) / (a1 - a0)
    return t - 7.0


CUT10 = (68.0, 72.0)                          # v9 span cut in v10


def to10(t):
    """v9 -> v10 time for automation (points inside the cut collapse onto the seam)"""
    return t if t < CUT10[0] else (CUT10[0] if t < CUT10[1] else t - (CUT10[1] - CUT10[0]))


CUT11 = ((76.0, 82.0, None), (82.0, 88.0, -6.0), (88.0, 96.0, None), (96.0, 1e9, -8.0))   # v10 -> v11


def to11(t):
    """v10 -> v11 time for an automation point (None: dropped; points in the a5b cut collapse onto 76)"""
    if t < 76.0:
        return t
    if t < 82.0:
        return 76.0
    if t < 88.0:
        return t - 6.0
    if t < 96.0:
        return None
    return t - 8.0


def to11_end(t):
    """v10 -> v11 for a ramp's end (a ramp running into a dropped span ends where that span begins)"""
    if 76.0 <= t < 82.0:
        return 76.0
    if 88.0 <= t < 96.0:
        return 82.0
    return to11(t)


def V7(t, pre=0.0):
    """the kept span (a, b, shift) holding v6 time t, or None (dropped); `pre` lets automation
    points just before a kept span that follows a dropped one move with it"""
    for a, b, s in V7_SEGS:
        if a - (pre if a > 0 else 0.0) - 1e-9 <= t < b - 1e-9:
            return a, b, s
    return None


# --- v6: a5c_talk (cues/a5c_talk.md) and a7_happy 104-112 (cues/a7_happy.md), FINAL v6 times
A5C_CARD = 88.125                             # Quill's private "Oslo" card: a tick
A5C_ECHO = [88.25, 88.5, 88.625, 88.75, 88.875, 89.0]   # the same sentence six times: the same stab
A5C_REV = 89.25                               # the echo stack fades: reverse swell into ...
A5C_AFTER = 89.5                              # ... "after chat training": livelier, a rising pass
A5C_TALK = (90.0, 90.375, 90.75, 91.0)        # Ember (main hit), Quill, Ember, Quill: four different phrases
A5C_STAT = 91.375                             # "repeated turns 17% -> 1%": hold, resolve
# (v6's A7_* times for its old a7_happy 104-112 are gone: that section is rewritten below)


def W(t):
    """v4 time -> v6 time"""
    return t + sum(n for a, n in INSERTS if t >= a - 1e-9)


LENGTH = 109.0

# ====================================================================== helpers
OFF = 0.0          # time offset added to every event (for the kept, shifted sections)
RAW = False        # True while writing the break (final v5 times, no warp)
NEW = False        # True while writing the v7 sections (final v7 times, no warp, no v6 -> v7 map)
N9 = False         # True while writing the v9 sections (v9 times: only the v9 -> v10 -> v11 cut maps)
N11 = False        # True while writing the v11 sections (final v11 times: no maps at all)


def out_t(t):
    return t if RAW else W(t)


def at(shift):
    """write the following events in score3 time, shifted by `shift` seconds"""
    global OFF
    OFF = shift


def B(t):
    """seconds -> song beats (120 bpm), as compact text"""
    return f"{round(2 * t, 5):g}"


def grid(t0, bpm):
    return lambda b: t0 + b * 60.0 / bpm


PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']


def midi(n):
    p = PC[n[0]] + (1 if '#' in n else 0) - (1 if 'b' in n[1:] else 0)
    return 12 * (int(n[-1]) + 1) + p


def name(m):
    return f"{NAMES[m % 12]}{m // 12 - 1}"


class Track:
    def __init__(self, name, opts):
        self.name, self.opts, self.ev = name, opts, []

    def n(self, t, dur, notes, vel):
        """a note or chord at t seconds (+OFF), held dur seconds"""
        t0 = t + OFF
        if N11:                                            # final v11 times: no maps at all
            if isinstance(notes, (list, tuple)):
                notes = "[" + " ".join(notes) + "]"
            self.ev.append((t0, f">{B(t0)} {notes}:{round(2 * dur, 5):g}@{vel:.2f}"))
            return
        if N9:
            t = t0
        elif NEW:
            t = t0
        else:
            t = out_t(t0)
            if not RAW and self.name not in ("fx",):
                for a, _ in INSERTS:
                    if t0 < a - 1e-9 < t0 + dur - 0.02:
                        if a == BREAK_AT:
                            raise ValueError(f"{self.name}: a note at {t0} sustains across the break")
                        dur = a - t0                       # cut at the split
            seg = V7(t)
            if seg is None:
                return                                     # v6 material the v7 cut replaces
            if t + dur - 0.02 > seg[1]:
                dur = seg[1] - t                           # cut at the end of its kept span
            t += seg[2]
        if not N9:                                         # v8 -> v9
            seg = V9(t)
            if seg is None:
                return
            if t + dur - 0.02 > seg[1] and self.name not in ("boom", "burst"):   # (one-shot tails ring on)
                dur = seg[1] - t
            t += seg[2]
        if isinstance(notes, (list, tuple)):
            notes = "[" + " ".join(notes) + "]"
        if self.name != "fx":                               # v9 -> v10
            if CUT10[0] - 1e-9 <= t < CUT10[1] - 1e-9:
                return
            if t < CUT10[0] and t + dur - 0.02 > CUT10[0] and self.name not in ("boom", "burst"):
                dur = CUT10[0] - t
            t = to10(t)
            if 76.0 - 1e-9 <= t < 82.0 - 1e-9 or 88.0 - 1e-9 <= t < 96.0 - 1e-9:    # v10 -> v11
                return
            for end in (76.0, 88.0):
                if t < end - 1e-9 and t + dur - 0.02 > end:
                    dur = end - t                          # (cut clean at the a5b cut and at a5c's end)
            t = t if t < 76.0 else (t - 6.0 if t < 88.0 else t - 8.0)
        self.ev.append((t, f">{B(t)} {notes}:{round(2 * dur, 5):g}@{vel:.2f}"))

    def text(self):
        out = [f"track {self.name} {self.opts}".rstrip()]
        ev = sorted(self.ev, key=lambda e: e[0])
        for i in range(0, len(ev), 6):
            out.append(" ".join(e[1] for e in ev[i:i + 6]))
        return "\n".join(out)


class Lane:
    """automation: set(t, v) jumps (5 ms ramp), ramp(t0, t1, v, shape) glides (times + OFF)"""

    def __init__(self, target, v0):
        self.target, self.v0, self.pts = target, v0, []

    def set(self, t, v):
        if N11:
            t = t + OFF
            self.pts.append((t, f">{B(t)} {v:g}:0.01"))
            return self
        if N9:
            t = to11(to10(t + OFF))
            if t is not None:
                self.pts.append((t, f">{B(t)} {v:g}:0.01"))
            return self
        if NEW:
            t = t + OFF
        else:
            t = out_t(t + OFF)
            seg = V7(t, 0.1)
            if seg is None:
                return self
            t += seg[2]
        seg = V9(t, 0.1)                                   # v8 -> v9
        if seg is None:
            return self
        t += seg[2]
        t = to11(to10(t))
        if t is not None:
            self.pts.append((t, f">{B(t)} {v:g}:0.01"))
        return self

    def ramp(self, t0, t1, v, shape="lin"):
        t0, t1 = t0 + OFF, t1 + OFF
        if N11:
            self.pts.append((t0, f">{B(t0)} {v:g}:{round(2 * (t1 - t0), 5):g}@{shape}"))
            return self
        if N9:
            return self._put(to10(t0), to10(t1), v, shape)
        if NEW:
            return self._ramp9(t0, t1, v, shape)
        cut = None if RAW else next((a for a, _ in INSERTS if t0 < a - 1e-9 < t1 + 2e-9), None)
        if cut is not None and t1 - t0 <= 0.1 + 1e-9:
            t0, t1 = W(cut) - (cut - t0), W(cut) - (cut - t1)   # a pickup into a hit on a split moves with it
        elif cut is not None and t1 > cut + 1e-9:
            if cut == BREAK_AT:
                raise ValueError(f"{self.target}: a ramp {t0}-{t1} crosses the break")
            t0, t1 = out_t(t0), W(cut - 1e-6)                    # cut at the split (reaches its value there)
        else:
            t0, t1 = out_t(t0), out_t(t1)
        seg = V7(t0, 0.1)
        if seg is None:
            return self
        t1 = min(t1, seg[1])
        t0, t1 = t0 + seg[2], t1 + seg[2]
        return self._ramp9(t0, t1, v, shape)

    def _ramp9(self, t0, t1, v, shape):
        """v8 -> v9: a ramp starting in a kept span keeps its start; its end is mapped by M9 when it
        reaches past the span (else shifted with it)"""
        seg = V9(t0, 0.1)
        if seg is None:
            return self
        a, b, s = seg
        t1 = t1 + s if t1 <= b + 1e-9 else M9(t1)
        t0 = t0 + s
        return self._put(to10(t0), to10(t1), v, shape)            # (a ramp inside a cut becomes a jump)

    def _put(self, t0, t1, v, shape):
        """v10 -> v11 and write"""
        t0, t1 = to11(t0), to11_end(t1)
        if t0 is None:
            return self
        t1 = max(t1 if t1 is not None else t0, t0)
        self.pts.append((t0, f">{B(t0)} {v:g}:{max(0.01, round(2 * (t1 - t0), 5)):g}@{shape}"))
        return self

    def text(self):
        body = " ".join(p[1] for p in sorted(self.pts, key=lambda p: p[0]))
        return f"automate {self.target}\n{self.v0:g} {body}".rstrip()


tracks, lanes, LN = [], [], {}


def track(name, opts=""):
    t = Track(name, opts)
    tracks.append(t)
    return t


def lane(target, v0=None):
    """create a lane, or return the existing one (so each section adds its own points)"""
    if target in LN:
        return LN[target]
    l = Lane(target, v0)
    lanes.append(l)
    LN[target] = l
    return l


g = grid(0.0, 120)

# ====================================================================== harmony
# (start, end, Ember's pad = wide saws (low), Quill's pad = PWM strings (mid)), global seconds
H = [
    # a1 (score3)
    (0.0, 4.0, "D2 A2 B2 F3", "D4 F4 Bb4"),
    (4.0, 8.0, "D2 A2 F3", "D4 F4 A4"),
    (8.0, 10.0, "D2 A2 D3", None),
    (10.0, 12.0, "D2 A2 E3 F3", "A3 E4"),
    # a2 (new): the link on Dm9; the labels on Bbmaj7(#11); the prompt pass climbs over Gm9 to
    # A7sus4 and the token lands on Dm9 (pushed, on the 16th before the beat); the opened-up
    # exchange in cold colours (Bbmaj7(#11) -> Gm(add9) -> A7sus4), the write lands on Dm9 at
    # 30.5; generation walks VI - iv - ii(half-dim) - V and releases on i at 37.5
    (12.0, 16.0, "D2 A2 F3 C4", "E4 A4"),          # the link: Dm9
    (16.0, 18.0, "Bb1 F2 D3 A3 E4", "D4 E4 A4"),   # labels, pass start: Bbmaj7(#11)
    (18.0, 20.0, "G1 D2 Bb2 F3", "A3 D4"),         # exchanges 1-6: Gm9
    (20.0, 21.25, "A1 E2 G2 D3", "D4 E4"),         # exchanges 7-8, the top: A7sus4
    (21.25, 22.75, "D2 A2 F3 C4", "E4 A4"),        # token: Dm9
    (22.75, 27.0, "Bb1 F2 D3 A3 E4", "D4 E4 A4"),  # a step, then the exchange opened: Bbmaj7(#11)
    (27.0, 29.0, "G1 D2 Bb2 A3", "Bb3 D4 A4"),     # rays, weights, values: Gm(add9)
    (29.0, 30.5, "A1 E2 G2 D3", "D4 E4 G4"),       # blend, output, write size: A7sus4
    (30.5, 32.0, "D2 A2 F3 C4", "E4 A4"),          # the write lands: Dm9
    (32.0, 34.0, "Bb1 F2 D3 A3", "D4 F4 A4"),      # generation: Bbmaj7
    (34.0, 36.0, "G1 D2 Bb2 F3", "A3 D4 F4"),      # Gm9
    (36.0, 37.0, "E2 Bb2 D3 G3", "D4 G4 Bb4"),     # Em7b5
    (37.0, 37.5, "A1 E2 G2 C#3", "C#4 E4 G4"),     # A7
    (37.5, 40.0, "D2 A2 E3 F3", "A3 E4 F4"),       # the fold: release on Dm(add9), into a3's pedal
    # a4 (new): the games walk in D dorian, with the iv turning minor (B -> Bb) in wave B
    (50.0, 52.0, "D2 A2 F3 C4", "E4 A4"),          # wave A: Dm9
    (52.0, 53.5, "Bb1 F2 D3 A3", "D4 F4 A4"),      # checks, fly out: Bbmaj7
    (53.5, 56.0, "G1 D2 B2 E3", "D4 B4"),          # wave B: G6/9 (dorian B)
    (56.0, 58.0, "G1 D2 Bb2 F3", "Bb3 D4 A4"),     # Gm9 (the B darkens to Bb)
    (58.0, 60.0, "Bb1 F2 D3 A3", "D4 F4 A4"),      # wave C: Bbmaj7
    (60.0, 61.0, "C2 G2 D3 E3", "E4 G4 D5"),       # Taipei: C(add9)
    (61.0, 62.5, "A1 E2 G2 D3", "D4 E4 G4"),       # A7sus4: the fold
    (62.5, 64.0, "A1 E2 G2 C#3", "C#4 E4 G4"),     # A7: the deck lands, compresses, flies
    # a5b (new): the a5 dominant released onto VI (lighter), the teacher writes over iv, the
    # joint pass on V sus, and the target lights on the relative major (F, not D: the tonic
    # is saved for the proof)
    (76.0, 77.0, "Bb1 F2 D3 A3 E4", "D4 F4 A4"),   # Bbmaj7(#11): release
    (77.0, 78.0, "G1 D2 Bb2 F3", "Bb3 D4 A4"),     # Gm9: the teacher writes
    (78.0, 79.0, "A1 E2 G2 D3", "D4 E4 G4"),       # A7sus4: the joint pass
    (79.0, 80.0, "F2 C3 E3 G3", "A3 C4 G4"),       # Fmaj9: the whole target lights
]
# kept sections, in score3 time + shift
H += [(a + SH_A3, b + SH_A3, e, q) for a, b, e, q in [
    (32.0, 36.0, "D2 A2 E3", "F4 A4"),
    (36.0, 37.0, "Bb1 F2 D3", None),
    ]]
H += [  # a3 second half (new, global time): the pushed concept, cold colours
    (45.0, 45.75, "D2 A2 E3", "A3 E4"),            # Dsus2: Ember has nothing yet
    (45.75, 47.25, "G1 D2 Bb2 A3", "Bb3 D4 A4"),   # winter pushed in: Gm(add9)
    (47.25, 48.0, "Bb1 F2 D3 A3 E4", "D4 E4 A4"),  # the snowfall: Bbmaj7(#11)
    (48.0, 49.25, "D2 A2", None),                  # link cut: a bare, dry fifth
    (49.25, 50.0, "D2 A2 E3 F3", "A3 E4 F4")]      # "words and concepts": Dm(add9)
H += [(a + SH_A5, b + SH_A5, e, q) for a, b, e, q in [
    (54.0, 58.0, "D2 A2 F3 C4", "A3 D4 F4"),
    (58.0, 61.0, "D2 G2 Bb2 A3", "Bb3 D4 A4"),
    (61.0, 63.0, "Bb1 F2 D3 A3", "D4 F4 A4"),
    (63.0, 66.0, "A1 E2 C#3 G3", "C#4 E4 G4")]]
H += [(a + SH_A6, b + SH_A6, e, q) for a, b, e, q in [
    (70.0, 72.0, "D2 A2 F3 C4", "D4 F4 A4"),
    (72.0, 73.0, "Bb1 F2 D3 A3", "D4 F4 A4"),
    (73.0, 73.75, "Bb1 F2 D3 F3", "D4 F4 Bb4"),
    (73.75, 74.5, "C2 G2 C3 E3", "E4 G4 C5"),
    (74.5, 75.25, "A1 E2 C#3 G3", "C#4 E4 A4"),
    (75.25, 76.0, "D2 A2 F3 C4", "D4 F4 A4"),
    (76.0, 77.5, "G1 D2 Bb2 F3", "Bb3 D4 A4"),
    (77.5, 78.25, "A1 E2 C#3 G3", "C#4 E4 G4"),
    (78.25, 79.0, "D2 A2 E3 F3", "A3 E4 F4"),
    (79.0, 80.0, "Bb1 F2 D3 A3", "D4 F4 A4"),
    (80.0, 82.0, "D2 A2 E3 G3", "A3 D4 E4")]]
H.sort()


def chord_pcs(t):
    for a, b, e, q in H:
        if a <= t < b:
            return {midi(n) % 12 for n in (e + " " + (q or "")).split()}
    return {2, 5, 9}


# ====================================================================== bus (initial values)
for p, v in [("reverb_decay", 0.86), ("reverb_wet", 0.26), ("reverb_tone", 4200), ("reverb_pre", 0.03),
             ("echo_time", 0.5), ("echo_feedback", 0.32), ("echo_tone", 2600), ("echo_pingpong", 0.7),
             ("chorus_mode", 2), ("chorus_mix", 0), ("chorus_depth", 0.5), ("spring", 0.0),
             ("tape_drive", 0.42), ("tape_wow", 0.0), ("tape_flutter", 0.004), ("tape_age", 0.1),
             ("width", 1.25), ("lfo_rate", 0.12), ("comp_in", 1), ("comp_threshold", -18), ("comp_ratio", 1),
             ("comp_attack", 3), ("comp_release", 4), ("comp_makeup", 3), ("comp_mix", 0.35),
             ("comp_sc_hpf", 60)]:
    lane(p, v)
# the 909 board (score3's tuned settings)
for p, v in [("bd_level", 1.0), ("bd_tune", 0.32), ("bd_attack", 0.7), ("bd_sweep", 0.4),
             ("bd_drive", 0.75), ("sd_level", 0.9), ("sd_tune", 0.32), ("sd_tone", 0.35), ("sd_snappy", 0.6),
             ("sd_decay", 0.55), ("rs_level", 0.5), ("rs_tune", 0.52), ("cp_level", 0.9), ("cp_decay", 0.6),
             ("hh_level", 0.45), ("hh_tune", 0.4), ("hh_metal", 0.5), ("ch_decay", 0.14),
             ("dr_drive", 0.9), ("dr_tone", 0.6)]:
    lane(p, v)

# ====================================================================== tracks
fx = track("fx", "sample=fx11.wav root=C4 gain=1.0 reverb_send=0.2")
N11 = True; fx.n(0.0, LENGTH, "C4", 0.8); N11 = False
dr = track("drums", "kit=909 comp=-28 gain=2.0 spring_send=0.12 reverb_send=0.08")
kb = track("kickbody", "patch=s3-kickbody gain=1.1")
lane("kickbody.pitch_shift", -0.17)
xh = track("xhit", "kit=909")                       # rim knocks on the exchanges
boom = track("boom", "patch=boom gain=1.0")
sub = track("sub", "patch=sub2 duck=0.2 duck_release=0.12")
gr = track("growl", "patch=s3-growl duck=0.2 duck_release=0.1 reverb_send=0.04")
ep = track("epad", "patch=s3-pad reverb_send=0.38 chorus_send=0.35 duck=0.4 duck_release=0.25 pan=0.12")
qp = track("qpad", "patch=s3-glasspad reverb_send=0.45 chorus_send=0.4 duck=0.3 duck_release=0.25 pan=-0.12")
sw = track("swell", "patch=s3-pad reverb_send=0.4 chorus_send=0.3")
Q = track("quill", "patch=s3-quill reverb_send=0.35 echo_send=0.22 chorus_send=0.2 pan=-0.25")
E = track("ember", "patch=s3-ember reverb_send=0.28 echo_send=0.16 chorus_send=0.12 pan=0.25")
sg = track("sig", "patch=glasslead reverb_send=0.4 echo_send=0.2 pan=-0.35 gain=1.2")
ar = track("arp", "patch=s3-arp reverb_send=0.15 echo_send=0.15 duck=0.35 duck_release=0.2 pan=0.08")
br = track("brass", "patch=s3-brass reverb_send=0.25 duck=0.0")
gl = track("glass", "patch=s3-glass reverb_send=0.4 echo_send=0.15 pan=0.25")
gq = track("glassq", "patch=s3-glass reverb_send=0.45 echo_send=0.2 pan=-0.35")   # Quill's side
gh = track("ghost", "patch=sour pan=-0.2 reverb_send=0.3 gain=1.1")
vx = track("voice", f"vox wav={VOICE_WAV} wav_at={B(to11(to10(W(VOICE_NEURAL) + V7(W(VOICE_NEURAL))[2] + V9(W(VOICE_NEURAL) + V7(W(VOICE_NEURAL))[2])[2])))} reverb_send=0 gain={VOICE_GAIN}")

for a, b, e, q in H:                                 # the pads follow the harmony table
    ep.n(a, b - a, e.split(), 0.5)
    if q:
        qp.n(a, b - a, q.split(), 0.45)


# ---------------------------------------------------------------- drum helpers (score3)
def K(t, v=0.95, body=True):
    dr.n(t, 0.1, "BD", v)
    if body:
        kb.n(t, 0.3, "A1", min(1.0, v))


def S(t, v=0.95, clap=True):
    dr.n(t, 0.1, "[SD CP]" if clap else "SD", v)


def Hh(t, v=0.12, open_=False):
    dr.n(t, 0.1, "OH" if open_ else "CH", v)


def HIT(t, v=1.0, kit="[BD CP OH]"):
    dr.n(t, 0.2, kit, v)
    kb.n(t, 0.3, "A1", min(1.0, v))


def half_bar(g, b0, fills=True):
    K(g(b0)); S(g(b0 + 2))
    for e, v in [(0.5, 0.1), (1.0, 0.07), (1.5, 0.12), (2.5, 0.1), (3.5, 0.14)]:
        Hh(g(b0 + e), v)
    if fills:
        K(g(b0 + 3.0), 0.55, body=False)


def full_bar(g, b0, four=False, oh=True):
    for k in range(4):
        if four or k in (0, 2):
            K(g(b0 + k))
        if k in (1, 3):
            S(g(b0 + k))
        Hh(g(b0 + k + 0.5), 0.2 if not oh else 0.16, open_=oh and k == 3)
    if not four:
        K(g(b0 + 2.5), 0.6, body=False)


def hats16(t0, t1, v0=0.06, acc=0.04, every=2):
    k = 0
    while t0 + 0.125 * k < t1 - 1e-6:
        Hh(t0 + 0.125 * k, v0 + acc * (k % every == 0))
        k += 1


def bassnote(t, dur, root, v=0.85, growl=True, octave=1, subv=0.8):
    sub.n(t, dur, f"{root}{octave + 1}", subv)
    if growl:
        gr.n(t, dur, f"{root}{octave}", v)


def half_bass(g, b0, root, octave=1):
    bassnote(g(b0), g(b0 + 1.5) - g(b0), root, 0.95, octave=octave)
    bassnote(g(b0 + 1.5), g(b0 + 2.5) - g(b0 + 1.5), root, 0.6, octave=octave)
    gr.n(g(b0 + 2.5), g(b0 + 3) - g(b0 + 2.5), f"{root}{octave + 1}", 0.7)
    sub.n(g(b0 + 2.5), g(b0 + 3) - g(b0 + 2.5), f"{root}{octave + 1}", 0.7)
    bassnote(g(b0 + 3), g(b0 + 4) - g(b0 + 3), root, 0.8, octave=octave)


def eighths(spans, accent_oct=4, v1=1.0, v2=0.7):
    """driving bass 8ths over [(root, t0, t1)], octave pop every `accent_oct`-th 8th"""
    for r, a, b in spans:
        t, k = a, 0
        while t < b - 1e-6:
            bassnote(t, 0.24, r, v1 if k % 2 == 0 else v2, octave=2 if k % accent_oct == accent_oct - 1 else 1)
            t += 0.25; k += 1


ARP = {"Dm9": "D2 A2 C3 F3 E3 C3 A2 F2", "Bb": "Bb1 F2 A2 D3 C3 A2 F2 D2", "Gm9": "G1 D2 F2 Bb2 A2 F2 D2 Bb1",
       "A7": "A1 E2 G2 C#3 E3 C#3 G2 E2", "G6": "G1 D2 E2 B2 A2 E2 D2 B1", "D": "D2 D3 A2 D3 F3 D3 A2 D3",
       "Dice": "D2 D3 G2 D3 Bb2 D3 G2 D3", "BbR": "Bb1 Bb2 F2 Bb2 D3 Bb2 F2 Bb2", "AR": "A1 A2 E2 A2 C#3 A2 E2 A2",
       "C9": "C2 G2 D3 E3 G3 E3 D3 G2", "Eh": "E2 Bb2 D3 G3 Bb2 G2 D3 Bb2"}


def arp(g, b0, b1, chord, step, vel=0.5):
    notes = ARP[chord].split(); k = 0; b = b0
    while b < b1 - 1e-6:
        ar.n(g(b), (g(b + step) - g(b)) * 0.95, notes[k % len(notes)], vel + (0.12 if k % 4 == 0 else 0))
        b += step; k += 1


def arp_t(t0, t1, chord, dt, vel=0.5):
    """arp in global seconds"""
    arp(lambda b: t0 + b * dt, 0, round((t1 - t0) / dt), chord, 1, vel)


def climb(t0, t1, dt, lo="D2", hi="D5", v0=0.34, v1=0.55, accents=()):
    """a forward pass on the arp: one note per step from t0 to t1, climbing from lo to hi through
    the chord tones sounding at each step (a small zig-zag inside the climb); notes at the
    `accents` times (the exchanges) are louder"""
    lo_, hi_ = midi(lo), midi(hi)
    n = round((t1 - t0) / dt)
    for b in range(n):
        t = t0 + dt * b
        target = lo_ + (hi_ - lo_) * b / max(1, n - 1) + (2, -2, 0)[b % 3]
        pcs = chord_pcs(t)
        m = min((x for x in range(lo_ - 3, hi_ + 4) if x % 12 in pcs), key=lambda x: abs(x - target))
        acc = 0.12 if any(abs(t - a) < 1e-6 for a in accents) else 0.0
        ar.n(t, dt * 0.95, name(m), v0 + (v1 - v0) * b / max(1, n - 1) + acc)


# lanes used by many sections (initial values = score3's)
L_drg = lane("drums.gain", 2.0)
L_ohd = lane("oh_decay", 0.45)
L_bdd = lane("bd_decay", 0.24)
L_bpi = lane("boom.pitch_shift", 0)
L_subg = lane("sub.gain", 0)
L_grc = lane("growl.cutoff", 170)
L_grd = lane("growl.drive", 4.0)
L_grg = lane("growl.gain", 1.8)
for p_, v_ in [("sustain", 0.35), ("decay", 0.22), ("attack", 0.001), ("filter_sustain", 0.1),
               ("filter_decay", 0.3)]:
    lane(f"growl.{p_}", v_)
lane("growl.pitch_shift", 0.08)
lane("sub.glide", 0)
L_epg = lane("epad.gain", 0)
L_qpg = lane("qpad.gain", 0)
L_epc = lane("epad.cutoff", 250)
L_epr = lane("epad.resonance", 0.9)
L_qpc = lane("qpad.cutoff", 900)
L_eprl = lane("epad.release", 1.6)
L_qprl = lane("qpad.release", 2.4)
L_swg = lane("swell.gain", 0.0)
L_swc = lane("swell.cutoff", 400)
lane("swell.release", 0.05)
L_qg = lane("quill.gain", 0.7)
L_eg = lane("ember.gain", 0.65)
L_qc = lane("quill.cutoff", 1300)
L_ec = lane("ember.cutoff", 520)
L_em = lane("ember.mono", 0)
L_egl = lane("ember.glide", 0)
L_qm = lane("quill.mono", 0)
L_qgl = lane("quill.glide", 0)
lane("sig.mono", 1)
lane("sig.glide", 0.2)
L_arc = lane("arp.cutoff", 600)
L_arr = lane("arp.resonance", 1.3)
L_arg = lane("arp.gain", 0.8)
lane("glass.gain", 1.0)
L_rvd = LN["reverb_decay"]
L_rvw = LN["reverb_wet"]


def booms(*tv):
    for t, v in tv:
        boom.n(t, 1.5, "D2", v)
        L_bpi.ramp(t - 0.05, t, 12).ramp(t, t + 0.6, 0, "log")


def swell(a, b, chord, v=0.6, c0=400, c1=4000):
    """a reverse-like pad swell from a to b, cut on the hit"""
    sw.n(a, b + 0.02 - a, chord, v)
    L_swg.set(a, 0.02).ramp(a + 0.02, b, 0.9, "exp").set(b + 0.01, 0.0)
    L_swc.set(a, c0).ramp(a + 0.02, b, c1, "exp")


def down(n):
    return n[:-1] + str(int(n[-1]) - 1)


# ====================================================================== A1 (0-12): score3, unchanged
at(0.0)
for t in (4.0, 5.0, 6.0):
    K(t, 0.42)
dr.n(8.0, 0.2, "[BD SD CP OH]", 1.0); kb.n(8.0, 0.3, "A1", 1.0)
L_drg.set(4.0, 0.8).set(8.0, 2.0)
L_ohd.set(7.95, 0.95).set(9.0, 0.45)
L_bdd.set(7.95, 0.8).set(9.0, 0.24)
booms((8.0, 1.0))
sub.n(0.0, 8.0, "D2", 0.55); sub.n(8.0, 2.0, "D2", 0.95); sub.n(10.0, 2.0, "D2", 0.6)
gr.n(8.0, 1.6, "D1", 1.0)
L_subg.ramp(0.0, 4.0, 0.6, "smooth").set(8.0, 1.0).set(10.0, 0.7)
L_epg.ramp(0.0, 3.5, 0.32, "smooth").set(8.0, 0.4)
L_qpg.ramp(0.0, 3.5, 0.28, "smooth")
L_epc.ramp(0.0, 4.0, 900, "exp").set(10.0, 700)
L_qpc.ramp(0.0, 4.0, 1700, "exp")
sw.n(10.25, 1.73, ["D3", "A3", "E4", "F4"], 0.6)
sw.n(6.75, 1.23, ["D3", "G#3", "D4"], 0.5)
for a, b in [(6.75, 7.98), (10.25, 11.98)]:
    L_swg.set(a, 0.02).ramp(a + 0.02, b, 0.9, "exp").set(b + 0.01, 0.0)
L_swc.ramp(6.75, 7.98, 3000, "exp").set(10.25, 400).ramp(10.27, 11.98, 4000, "exp")
for t, n, d in [(0.5, "D4", 0.5), (1.0, "F4", 0.5), (1.5, "A4", 0.5), (2.0, "Bb4", 1.0), (3.0, "A4", 1.0)]:
    Q.n(t, d, n, 0.5)
for t, n, d in [(2.0, "D3", 0.5), (2.5, "A3", 0.5), (3.0, "B3", 0.5), (3.5, "C4", 0.25), (3.75, "A3", 0.25)]:
    E.n(t, d, n, 0.5)
for k, (q, e) in enumerate(zip("D4 F4 G4 A4 Bb4 C5".split(), "D3 F3 G3 A3 B3 C4".split())):
    Q.n(4.0 + 0.5 * k, 0.5, q, 0.42 + 0.03 * k)
    E.n(4.0 + 0.5 * k, 0.5, e, 0.42 + 0.03 * k)
Q.n(7.0, 0.5, "D5", 0.62); E.n(7.0, 0.5, "D4", 0.62)
E.n(10.0, 2.0, "A3", 0.45); Q.n(10.5, 1.5, "E4", 0.42)
br.n(8.0, 1.5, "D2 G#2 D3 G#3".split(), 1.0)
for k, n in enumerate("C6 F#5 A5 D#5 G5 C#5 E5 A#4 D5 G#4 C5".split()):   # shards (deliberately atonal)
    gl.n(8.0 + 0.125 * k, 0.15, n, 0.55 - 0.035 * k)

# ====================================================================== A2 (12-40): NEW
# The architecture, following the picture: the link plugs in (12-14), the bodies unfold into
# two lattices (14.25-16) and hold for the labels; the prompt pass climbs (a 16th arp through the
# stack) with an exchange every dotted 8th, 3 against the beat; a token on Dm9; one quick
# generation step; then ONE exchange opened up, patient, precise and cold (kick on the bar, rim
# 8ths, glass): keys/values, gates, the two queries, attention over every key, the weighted values,
# the write made small, plunged in: the hit at 30.5. Then the write rises and generation runs a
# token a beat over a groove that fills up to four on the floor with 16th arps, driving into the
# release on D at 37.5 (the fold), which swells down into a3's quiet opening at 40.
# Story in the notes: at every exchange BOTH voices sound together (the link is two-way and
# simultaneous); the idea's note (accented) hops between them up the stack.
at(0.0)

# --- drums
dr.n(A2_DROP, 0.2, "OH", 0.5)
half_bar(g, 2 * A2_DROP, fills=True); half_bar(g, 2 * A2_DROP + 4, fills=False)   # 12-16
K(A2_LABELS[0], 0.6)                                      # the hold: kick and rim ticks, precise
for e in range(int((A2_PASS - A2_LABELS[0]) / 0.25)):
    dr.n(A2_LABELS[0] + 0.25 * e, 0.1, "RS", 0.2 if e % 2 else 0.28)
K(A2_PASS, 0.7, body=False)
half_bar(g, 36, fills=False)                              # the pass: half time 18-20
K(20.0); Hh(20.5, 0.1); Hh(20.75, 0.12); K(20.75, 0.5, body=False); Hh(21.0, 0.14)
for t in A2_EX:                                           # a rim knock rides every exchange
    xh.n(t, 0.1, "RS", 0.6)
HIT(A2_TOKEN, 0.95)                                       # the token
K(22.25, 0.55); Hh(22.0, 0.1); Hh(22.5, 0.12)
K(A2_STEP, 0.6, body=False)                               # one step: 16th hats, rims on the sweep
hats16(A2_STEP, A2_EMIT, 0.06, 0.05)
for k, t in enumerate(A2_STEP_EX):
    xh.n(t, 0.1, "RS", 0.4 + 0.05 * k)
HIT(A2_EMIT, 0.75, "[BD CP]")
for t in A2_TICK1:
    xh.n(t, 0.1, "RS", 0.4)
for t in (24.0, 26.0, 28.0):                              # the exchange opened: cold, tight
    K(t, 0.75)
for e in range(int((X["hit"] - 24.0) / 0.25)):
    dr.n(24.0 + 0.25 * e, 0.1, "RS", 0.2 if e % 2 else 0.3)
K(X["rays"], 0.6, body=False)
hats16(28.5, X["plunge"], 0.04, 0.04)
K(30.0, 0.6, body=False); K(X["plunge"], 0.7, body=False)
HIT(X["hit"], 1.0)                                        # the write is added: hit
K(31.0, 0.7); S(31.5, 0.8); Hh(31.25, 0.12); Hh(31.75, 0.14, open_=True)
for t in A2_RESUME_EX:
    xh.n(t, 0.1, "RS", 0.55)
full_bar(g, 64, four=False, oh=False); hats16(32.0, 34.0, 0.05, 0.04)          # generation
full_bar(g, 68, four=True, oh=True); hats16(34.0, 36.0, 0.05, 0.05, 4)
for k in range(3):                                        # 36-37.5: four on the floor, 16th hats,
    K(36.0 + 0.5 * k)                                     # kicks doubling into the release
S(36.5); S(37.0, 0.85)
K(37.25, 0.7, body=False); K(37.375, 0.75, body=False)
hats16(36.0, A2_FOLD, 0.07, 0.05)
HIT(A2_FOLD, 1.0, "[BD SD CP OH]")                        # the release: the fold
K(38.5, 0.5); Hh(38.75, 0.08); K(39.5, 0.4)               # swell down: a heartbeat
L_drg.set(12.0, 2.0).set(A2_LABELS[0], 1.3).set(A2_PASS, 1.8).set(A2_EMIT + 0.2, 1.3).set(X["hit"] - 0.1, 2.0) \
    .set(38.25, 1.2)
L_ohd.set(A2_FOLD - 0.05, 0.95).set(38.4, 0.45)
L_bdd.set(A2_FOLD - 0.05, 0.75).set(38.4, 0.24)

# --- impacts and swells
booms((A2_DROP, 0.85), (A2_TOKEN, 0.5), (X["hit"], 0.75), (A2_FOLD, 0.7))
br.n(A2_DROP, 1.5, "D2 A2 D3 F3".split(), 0.85)
br.n(A2_TOKEN, 0.5, "D3 A3 D4".split(), 0.6)
br.n(X["hit"], 0.75, "D2 A2 D3 F3 A3".split(), 0.85)
swell(X["blend"], X["hit"] - 0.02, ["D3", "A3", "C4", "E4"], 0.5, 400, 4500)     # into the write
br.n(A2_FOLD, 0.75, "D2 A2 D3 E3 A3".split(), 0.8)
swell(36.5, A2_FOLD - 0.02, ["A2", "E3", "G3", "C#4"], 0.45, 400, 4500)          # into the release

# --- bass
half_bass(g, 24, "D"); half_bass(g, 28, "D")             # 12-16
sub.n(A2_LABELS[0], 1.5, "Bb1", 0.65)
bassnote(A2_PASS, 0.5, "Bb", 0.8)
half_bass(g, 36, "G")                                     # 18-20
bassnote(20.0, 1.25, "A", 0.85)
bassnote(A2_TOKEN, 0.5, "D", 1.0); sub.n(A2_TOKEN + 0.5, 1.0, "D2", 0.6)
sub.n(A2_STEP, A2_EMIT - A2_STEP, "Bb1", 0.65)
bassnote(A2_EMIT, 0.25, "Bb", 0.8, octave=2)
sub.n(24.0, 3.0, "Bb1", 0.65); sub.n(27.0, 2.0, "G1", 0.65); sub.n(29.0, 1.5, "A1", 0.7)
bassnote(X["hit"], 0.5, "D", 1.0)
eighths([("D", 31.0, 32.0)], accent_oct=4)
eighths([("Bb", 32.0, 34.0), ("G", 34.0, 36.0)], accent_oct=8)
eighths([("E", 36.0, 37.0), ("A", 37.0, A2_FOLD)], accent_oct=3)
bassnote(A2_FOLD, 0.5, "D", 1.0); sub.n(A2_FOLD + 0.5, A2_END - A2_FOLD - 0.5, "D2", 0.55)
L_subg.set(12.0, 1.0)
L_grc.set(A2_LABELS[0], 140).set(A2_PASS, 170).set(A2_EMIT + 0.2, 140).set(X["hit"], 200) \
    .ramp(32.0, A2_FOLD, 330, "exp")
L_grd.set(36.0, 4.6).set(38.0, 4.0)

# --- pads: open across the link, cooler for the labels, opening with the climb, cold in the
# opened exchange, opening again with the write and the generation
L_epg.set(12.0, 0.45).set(A2_LABELS[0], 0.36).set(A2_PASS, 0.42).set(A2_EMIT + 0.2, 0.3) \
    .set(X["hit"], 0.45).set(36.0, 0.5).ramp(A2_RETURN, 39.25, 0.3)
L_qpg.set(12.0, 0.36).set(A2_LABELS[0], 0.3).set(A2_PASS, 0.34).set(A2_EMIT + 0.2, 0.26).set(X["hit"], 0.36)
L_epc.ramp(12.0, A2_UNFOLD, 1000, "exp").ramp(A2_UNFOLD, 16.0, 1400, "exp").set(A2_LABELS[0], 900) \
    .ramp(A2_PASS, A2_TOKEN, 1700, "exp").set(A2_TOKEN, 1300).set(A2_EMIT + 0.2, 550) \
    .ramp(X["blend"], X["hit"], 1200, "exp").set(X["hit"], 1300).ramp(X["hit"], A2_FOLD, 2400, "exp") \
    .ramp(A2_RETURN, 39.25, 700, "exp")
L_epr.ramp(12.0, 16.0, 1.1).set(A2_LABELS[0], 0.9).ramp(A2_PASS, A2_TOKEN, 1.2).set(A2_EMIT + 0.2, 0.8) \
    .set(X["hit"], 1.0).ramp(32.0, A2_FOLD, 1.25).ramp(A2_RETURN, 39.25, 0.9)
L_qpc.ramp(A2_UNFOLD, 16.0, 2000, "exp").set(A2_LABELS[0], 1500).ramp(A2_PASS, A2_TOKEN, 2400, "exp") \
    .set(A2_EMIT + 0.2, 1200).set(X["hit"], 1800).ramp(X["hit"], A2_FOLD, 2800, "exp") \
    .ramp(A2_RETURN, 39.25, 1300, "exp")
L_rvd.set(A2_LABELS[0], 0.75).set(A2_PASS, 0.82).set(A2_EMIT + 0.2, 0.62).set(X["hit"], 0.86)

# --- Quill and Ember
# the link: 8 strands, 8 rising plucks, each sounding both voices at once (one per direction)
for t, e, q in zip(A2_STRANDS, "D3 E3 F3 A3 C4 D4 E4 F4".split(), "A3 C4 D4 E4 F4 A4 C5 D5".split()):
    E.n(t, 0.24, e, 0.62); Q.n(t, 0.24, q, 0.58)
Q.n(A2_UNFOLD + 0.25, 1.75, "E4", 0.42); E.n(A2_UNFOLD + 0.25, 1.75, "A3", 0.45)   # the unfold: their fifth
# the prompt pass: at every exchange both voices sound; the idea's note (accented) zig-zags up
# Ember / Quill, the other voice answering softly at the same instant
IDEA = "G3 A4 Bb3 C5 D4 D5 E4 E5".split()
OTHER = "Bb4 D3 F4 F3 A4 G3 A4 A3".split()
for k, t in enumerate(A2_EX):
    idea_e = k % 2 == 0
    (E if idea_e else Q).n(t, 0.34, IDEA[k], 0.85)
    (Q if idea_e else E).n(t, 0.34, OTHER[k], 0.5)
Q.n(A2_TOKEN, 0.75, "D4", 0.9); E.n(A2_TOKEN, 0.75, "D4", 0.9)      # the token: the same D
# one generation step: a quick sweep in contrary motion, then the emitted tokens
for k, (t, q, e) in enumerate(zip(A2_STEP_EX, "D4 F4 A4 C5".split(), "Bb3 A3 F3 D3".split())):
    Q.n(t, 0.12, q, 0.5 + 0.04 * k); E.n(t, 0.12, e, 0.52 + 0.04 * k)
Q.n(A2_EMIT, 0.45, "E5", 0.65); E.n(A2_EMIT, 0.45, "D4", 0.65)
for k, (t, q, e) in enumerate(zip(A2_TICK1, "D4 F4 A4".split(), "A3 F3 D3".split())):   # the next step starts
    Q.n(t, 0.12, q, 0.42); E.n(t, 0.12, e, 0.45)
# the exchange opened up, both directions at once
gl.n(X["states"], 0.2, "A4", 0.24); gq.n(X["states"], 0.2, "D5", 0.24)              # partner states land
gl.n(X["kv"], 0.3, ["D5", "A5"], 0.3); gq.n(X["kv"], 0.3, ["A4", "E5"], 0.3)        # keys and values appear
sg.n(X["gates"], 0.125, "A4", 0.45); sg.n(X["gates"] + 0.125, 0.6, "D4", 0.4)    # a gate slides shut
E.n(X["q"], 0.25, "D3", 0.6); E.n(X["q"] + 0.25, 0.6, "A3", 0.62)                # two queries glide,
Q.n(X["q"], 0.25, "A4", 0.55); Q.n(X["q"] + 0.25, 0.6, "D4", 0.58)               # mirror images
L_em.set(X["q"] - 0.05, 1).set(X["rays"] - 0.02, 0)
L_egl.set(X["q"] + 0.1, 0.3).set(X["rays"] - 0.02, 0)
L_qm.set(X["q"] - 0.05, 1).set(X["rays"] - 0.02, 0)
L_qgl.set(X["q"] + 0.1, 0.3).set(X["rays"] - 0.02, 0)
for k, (q, e) in enumerate(zip("D5 C5 Bb4 A4".split(), "D3 F3 A3 C4".split())):   # rays to every key
    Q.n(X["rays"] + 0.125 * k, 0.12, q, 0.5); E.n(X["rays"] + 0.125 * k, 0.12, e, 0.52)
for k, n in enumerate("G4 Bb4 D5 A5".split()):                    # the weights rise
    gl.n(X["bars"] + 0.25 * k, 0.3, n, 0.2 + 0.03 * k)
for k, (q, e) in enumerate(zip("Bb4 A4 G4 F4".split(), "D4 C4 Bb3 A3".split())):  # values pour, both ways
    Q.n(X["flow"] + 0.125 * k, 0.12, q, 0.5); E.n(X["flow"] + 0.125 * k, 0.12, e, 0.55)
Q.n(X["blend"], 0.5, "D4", 0.5); E.n(X["blend"], 0.5, "G3", 0.55); xh.n(X["blend"], 0.1, "RS", 0.45)   # the sum
gl.n(X["shrink"], 0.15, "D5", 0.24); gl.n(X["shrink"] + 0.125, 0.4, "A4", 0.2)     # made small (write size)
Q.n(X["hit"], 0.75, "D4", 0.85); E.n(X["hit"], 0.75, "D4", 0.85)                  # added: the write lands
for t, q, e in zip(A2_RESUME_EX, "F4 A4 C5 E5".split(), "A3 F3 D4 C4".split()):     # exchanges 5-8 resume
    Q.n(t, 0.2, q, 0.6); E.n(t, 0.2, e, 0.62)
# generation: every token is written by both at once (dyads), then the fold in 16th runs
GEN_Q = "D5 F5 E5 D5 Bb4 A4 C5 D5 Bb4 C#5 D5".split()
GEN_E = "Bb3 A3 D4 Bb3 G3 F3 A3 Bb3 G3 A3 D4".split()
for k, t in enumerate(A2_GEN):
    last = t == A2_FOLD
    Q.n(t, 1.0 if last else 0.45, GEN_Q[k], 0.8 if last else 0.7)
    E.n(t, 1.5 if last else 0.45, GEN_E[k], 0.8 if last else 0.72)
Q.n(38.5, 1.0, "A4", 0.5); Q.n(39.5, 0.45, "E4", 0.42)    # the swell down, into a3's lantern on D4
E.n(39.0, 1.0, "A3", 0.45)
L_qg.set(12.0, 1.1).set(A2_LABELS[0], 0.95).set(A2_PASS, 1.05).set(A2_EMIT + 0.2, 0.9).set(X["hit"], 1.05)
L_eg.set(12.0, 1.25).set(A2_LABELS[0], 1.05).set(A2_PASS, 1.2).set(A2_EMIT + 0.2, 1.0).set(X["hit"], 1.2)
L_qc.set(A2_EMIT + 0.2, 1000).set(X["hit"], 1300).ramp(X["hit"], A2_FOLD, 1800, "exp")
L_ec.set(A2_EMIT + 0.2, 420).set(X["hit"], 520).ramp(X["hit"], A2_FOLD, 700, "exp")

# --- arp: the prompt pass climbs the stack in 16ths; the step climbs in 16ths; the write
# rises after the hit; generation in 16ths
climb(A2_PASS, A2_TOKEN, 0.125, "D2", "D5", 0.32, 0.52, A2_EX)
climb(A2_STEP, A2_EMIT, 0.125, "Bb1", "Bb4", 0.32, 0.5)
climb(31.0, 32.5, 0.125, "D3", "A5", 0.3, 0.45)            # the write rises
for a, b, c in [(32.5, 34.0, "Bb"), (34.0, 36.0, "Gm9"), (36.0, 37.0, "Eh"), (37.0, A2_FOLD, "A7"),
                (A2_FOLD, A2_END, "D")]:
    arp_t(a, b, c, 0.125, 0.38 + 0.04 * (a >= 36.0))
L_arc.set(A2_PASS - 0.05, 450).ramp(A2_PASS, A2_TOKEN, 1500, "exp").set(A2_STEP, 900) \
    .set(A2_EMIT + 0.2, 600).set(X["hit"], 800).ramp(X["hit"], A2_FOLD, 2600, "exp") \
    .ramp(A2_RETURN, 39.75, 500, "exp")
L_arg.set(A2_PASS, 0.8).set(36.0, 0.9).ramp(A2_RETURN, 39.9, 0.35).set(A2_END, 0.8)

# --- glass: the labels
for t, n in zip(A2_LABELS, ("A4", "E5", "D5")):
    gl.n(t, 0.3, n, 0.26)

# ====================================================================== A3 (40-50): score3 32-42, +8
at(SH_A3)
HIT(32.0, 1.0, "[BD CP OH]")                     # v5: the hard re-entry after the break (48.0)
booms((32.0, 0.8))
K(34.0, 0.55)
L_drg.set(32.3, 1.0).set(A3_WRITE - SH_A3, 1.6)   # (v5: after the re-entry hit at 48.0)
sub.n(32.0, 4.0, "D2", 0.55); sub.n(36.0, 1.0, "Bb1", 0.55)
L_grc.set(32.0, 140)
L_grd.set(32.0, 4.0)
L_epg.set(32.0, 0.25)
L_qpg.set(32.0, 0.22)
L_epc.set(32.0, 550)
L_epr.set(32.0, 0.8)
L_qpc.set(32.0, 1200)
L_rvd.set(32.0, 0.86)
for t0, last in [(32.0, 1.25), (34.0, 0.75)]:           # Quill's lantern, twice
    for k, n in enumerate("D4 A4 G4".split()):
        Q.n(t0 + 0.25 * k, 0.25, n, 0.75)
    Q.n(t0 + 0.75, last, "Bb4", 0.78)
L_qg.set(32.0, 0.8)
L_eg.set(32.0, 0.95)
L_qc.set(32.0, 1400)
L_ec.set(32.0, 560)
for k, n in enumerate("D2 F2 A2 D3 F3 A3 D4".split()):  # the gold rising
    ar.n(34.5 + 0.25 * k, 0.24, n, 0.35 + 0.04 * k)
L_arc.set(34.45, 500)
# accents moved to the a3 cue list (global time)
at(0.0)
HIT(A3_WRITE, 0.8, "[BD CP]")                           # Ember writes "lantern" and plays it back
for k, n in enumerate("D4 A4 G4".split()):
    E.n(A3_WRITE + 0.25 * k, 0.25, n, 0.75)
E.n(A3_WRITE + 0.75, 0.75, "Bb4", 0.78)
gl.n(A3_CARD, 0.4, ["D5", "A5"], 0.28)                  # the gold card pops
K(A3_TAIL, 0.45); gl.n(A3_TAIL, 0.5, "D5", 0.32)        # the card goes into Quill's tail
dr.n(A3_BUBBLE, 0.1, "RS", 0.35)                        # Ember's "?"

# --- the second half (new, cues/a3_cross.md): a concept pushed into Ember's hidden states.
# Cold and quiet: a glass shimmer as the snow swirl forms, a swell into the push, an icy impact
# on Gm(add9), the pass as a rising 16th arpeggio, Quill's linked opener one note per token,
# with "snow" on the cold #11 of Bbmaj7 where a glass-and-noise snowfall starts; the link cut
# stops it dead (pads and reverb cut, filters shut), the cut opener runs dry and muted over a
# bare fifth, and the caption resolves on Dm(add9).
gl.n(A3_SWIRL, 0.6, ["E5", "A5"], 0.18); gq.n(A3_SWIRL + 0.125, 0.6, ["D5", "A5"], 0.16)
swell(A3_SWIRL, A3_PUSH - 0.02, ["D4", "E4", "A4"], 0.35, 600, 6000)
HIT(A3_PUSH, 0.9)
br.n(A3_PUSH, 0.3, "G1 D2 G2 D3 A3".split(), 0.7)
booms((A3_PUSH, 0.5))
gl.n(A3_PUSH, 0.6, ["A4", "E5", "A5"], 0.38)
E.n(A3_PUSH, 0.5, "D3", 0.7)
climb(A3_PUSH, A3_RUN[0], 0.125, "G2", "D5", 0.36, 0.5)          # the pass: rising arpeggio
bassnote(A3_PUSH, 0.5, "G", 0.9); sub.n(A3_PUSH + 0.5, A3_SNOW - A3_PUSH - 0.5, "G1", 0.6)
sub.n(45.0, A3_PUSH - 45.0, "D2", 0.5)
for t, n in zip(A3_RUN, "D5 A4 Bb4 G4 A4 Bb4 E5 D5 A4 F4 D5".split()):   # Quill's opener
    Q.n(t, 0.24 if t == A3_QMARK else 0.12, n, 0.8 if t in (A3_SNOW, A3_QMARK) else 0.6)
K(A3_RUN[0], 0.6); hats16(A3_RUN[0], A3_CUT, 0.04, 0.03)
K(A3_SNOW, 0.7); Hh(A3_SNOW, 0.14, open_=True)                   # the snowfall
E.n(A3_SNOW, 0.5, "A3", 0.5)                                     # Ember lights up
for k, n in enumerate("A5 E5 D5 A5 F5 E5".split()):
    (gq if k % 2 else gl).n(A3_SNOW + 0.125 * k, 0.3, n, 0.14 + 0.02 * (k % 3))
sub.n(A3_SNOW, A3_CUT - A3_SNOW, "Bb1", 0.6)
dr.n(A3_QMARK, 0.2, "[BD CP]", 0.75); kb.n(A3_QMARK, 0.3, "A1", 0.75)    # "?", Ember hops
E.n(A3_QMARK, 0.25, "D4", 0.6); gl.n(A3_QMARK, 0.4, "D5", 0.3)
K(A3_CUT, 0.8); dr.n(A3_CUT, 0.1, "RS", 0.5)                     # the link is cut: a dry snip
sub.n(A3_CUT, 0.5, "D2", 0.55)
for k, (t, n) in enumerate(zip(A3_DRY, "D5 A4 Bb4 G4 A4 F4 E4 D4 E4".split())):   # the cut opener, dry
    Q.n(t, 0.11, n, 0.5)
    if k % 2 == 0:
        Hh(t, 0.05)
K(A3_CAPTION, 0.5); Hh(A3_CAPTION, 0.12, open_=True)             # "words and concepts"
sub.n(A3_CAPTION, 50.0 - A3_CAPTION, "D2", 0.5)
Q.n(A3_CAPTION, 0.7, "A4", 0.4); E.n(A3_CAPTION, 0.7, "D4", 0.42)
Hh(49.75, 0.3, open_=True)                                       # pickup into a4
# the cold colours, the hard stop at the cut, the resolve
L_drg.set(45.7, 1.7)
L_epg.set(45.0, 0.3).set(A3_PUSH, 0.4).set(A3_CUT, 0.24).set(A3_CAPTION, 0.34)
L_qpg.set(45.0, 0.26).set(A3_PUSH, 0.34).set(A3_CUT, 0.16).set(A3_CAPTION, 0.3)
L_epc.set(45.0, 600).ramp(A3_PUSH, A3_SNOW, 1400, "exp").set(A3_SNOW, 1800).set(A3_CUT, 450).set(A3_CAPTION, 800)
L_qpc.set(45.0, 1400).ramp(A3_PUSH, A3_QMARK, 3200, "exp").set(A3_CUT, 800).set(A3_CAPTION, 1500)
L_epr.set(45.0, 0.8)
L_eprl.set(47.9, 0.08).set(48.1, 1.6)
L_qprl.set(47.9, 0.08).set(48.1, 2.4)
L_rvd.set(45.0, 0.92).set(A3_CUT, 0.6).set(A3_CAPTION, 0.8)
L_rvw.set(47.97, 0.1).set(A3_CAPTION - 0.05, 0.26)
L_qg.set(45.0, 1.0).set(A3_CUT, 0.75).set(A3_CAPTION, 1.0)
L_eg.set(45.0, 1.15)
L_qc.set(45.0, 1400).ramp(A3_RUN[0], A3_QMARK, 1900, "exp").set(A3_CUT, 650).set(A3_CAPTION, 1300)
L_ec.set(45.0, 560)
lane("quill.echo_send", 0.22).set(A3_CUT, 0.0).set(A3_CAPTION, 0.22)
L_arc.set(A3_PUSH - 0.05, 900).ramp(A3_PUSH, A3_RUN[0], 2400, "exp")

# ====================================================================== A4 (50-64): NEW
# Three waves of game windows (cues/a4_games.md), each denser than the last. A wave: three
# window slams on 8ths with brass stabs, a gold glass sparkle for each window's read pass, then
# the answers typed in 16ths in the writers' voices (both voices at once when both write), a
# glass ping for every check. Wave A on Dm9 -> Bbmaj7; wave B, the long one, both models write
# the sum in a 16th drive over G6/9 turning to Gm9; wave C on Bbmaj7 -> C(add9) with the
# partner's city (Taipei) as a gold accent; then the windows fold into a deck that lands and
# compresses on A7 and flies off.
at(0.0)
W_ROOT = ("D", "G", "Bb")
W_BRASS = ("D2 A2 D3 F3", "G1 D2 G2 B2", "Bb1 F2 Bb2 D3")
W_SPARK = (("A4", "D5", "F5"), ("B4", "D5", "E5"), ("A4", "D5", "F5"))
for i, w in enumerate(A4_WAVES):
    big = 1.0 if i != 1 else 0.95
    dr.n(w, 0.2, "[BD CP OH]", big); kb.n(w, 0.3, "A1", big)          # slam 1
    K(w + 0.25, 0.8)                                                   # slam 2
    dr.n(w + 0.5, 0.2, "[BD SD]", 0.9); kb.n(w + 0.5, 0.3, "A1", 0.9)  # slam 3
    for dt, v in ((0.0, 0.85), (0.25, 0.7), (0.5, 0.8)):
        br.n(w + dt, 0.15, W_BRASS[i].split(), v)
    for dt, d, v in ((0.0, 0.2, 1.0), (0.25, 0.2, 0.8), (0.5, 0.4, 0.95)):
        bassnote(w + dt, d, W_ROOT[i], v)
    for j_, n in enumerate(W_SPARK[i]):                                 # read passes: gold sparkle
        gq.n(w + 0.375 + 0.25 * j_, 0.25, n, 0.3)
    a, b = A4_WRITE[i]
    hats16(a, b + 0.125, 0.08 + 0.02 * i, 0.06)
booms((A4_WAVES[0], 0.6), (A4_WAVES[2], 0.65))
for t in A4_CHECKS:                                                    # every check: rim + glass ping
    dr.n(t, 0.1, "RS", 0.5)
for t, n in zip(A4_CHECKS, "A4 D5 F5 B4 D5 F5 G5 A4 E5 D5".split()):
    gl.n(t, 0.3, n, 0.36)

# wave A (50-53.5): Ember writes the word, Quill the number and the phrase, at once
K(51.0, 0.85); Hh(51.0, 0.2, open_=True); K(51.5, 0.8); S(52.0, 0.85); K(52.5, 0.85); K(52.75, 0.5, body=False)
K(53.0, 0.75); Hh(53.25, 0.2, open_=True)
for k, n in enumerate("D3 F3 A3 B3 C4 A3".split()):                    # Ember (dorian B)
    E.n(51.125 + 0.125 * k, 0.12, n, 0.6)
for k, n in enumerate("A4 C5 D5 E5 F5 E5 D5 C5 Bb4".split()):          # Quill
    Q.n(51.125 + 0.125 * k, 0.12, n, 0.58)
eighths([("D", 51.0, 52.0), ("Bb", 52.0, 53.0)], accent_oct=4)
arp_t(51.0, 52.0, "Dm9", 0.25, 0.42); arp_t(52.0, 53.5, "Bb", 0.25, 0.4)

# wave B (53.5-58): both write the sum, a 16th drive in both voices
S(54.5); K(55.0); K(55.25, 0.6, body=False); S(55.5); Hh(54.75, 0.2); Hh(55.75, 0.2)   # 54-56
full_bar(g, 112, four=True, oh=True)                                    # 56-58
QB = "D5 E5 D5 B4 A4 B4 D5 E5 G5 E5 D5 B4 A4 Bb4 D5 F5 D5 Bb4 A4 G4 A4 D5".split()
EB = "G3 A3 B3 D4 E4 D4 B3 A3 G3 A3 B3 D4 E4 D4 Bb3 A3 G3 F3 G3".split()
for k, n in enumerate(QB):
    Q.n(54.375 + 0.125 * k, 0.12, n, 0.52 + 0.06 * (k % 4 == 0))
for k, n in enumerate(EB):
    E.n(54.375 + 0.125 * k, 0.12, n, 0.55 + 0.06 * (k % 4 == 0))
eighths([("G", 54.5, 56.0), ("G", 56.0, 57.5)], accent_oct=4)
arp_t(54.0, 56.0, "G6", 0.125, 0.4); arp_t(56.0, 57.5, "Gm9", 0.125, 0.42)

# wave C (58-62): Ember "Extinct." and the lookup, Quill the chat line with Taipei in gold
K(59.0, 0.85); S(59.5, 0.85); K(60.5, 0.85); K(60.75, 0.55, body=False)
K(61.0, 0.85); S(61.5, 0.85); Hh(61.75, 0.2, open_=True)
HIT(A4_TAIPEI, 0.95)                                                    # the city: accent (no kick on 60.0)
br.n(A4_TAIPEI, 0.3, "C2 G2 C3 E3 G3".split(), 0.8)
gq.n(A4_TAIPEI, 0.5, ["E5", "G5"], 0.36)
for k, n in enumerate("D4 C4 A3 Bb3 D4 F4 D4 C4".split()):              # Ember
    E.n(59.125 + 0.125 * k, 0.12, n, 0.6)
QC = "F4 A4 Bb4 C5 D5 C5 A4 G4 E5 D5".split()                         # Quill (E5 = Taipei, D5 = "?")
for k, n in enumerate(QC):
    t = 59.125 + 0.125 * k
    Q.n(t, 0.24 if t == A4_TAIPEI else 0.12, n, 0.85 if t == A4_TAIPEI else 0.56)
eighths([("Bb", 58.75, 60.0), ("C", 60.0, 61.0), ("A", 61.0, 62.0)], accent_oct=4)
arp_t(58.75, 60.0, "Bb", 0.125, 0.42); arp_t(60.0, 61.0, "C9", 0.125, 0.45); arp_t(61.0, 62.0, "AR", 0.25, 0.42)
Q.n(61.5, 0.9, "E4", 0.45); E.n(61.5, 0.9, "A3", 0.45)                  # a fifth under the fold

# the deck: fold (riser), lands (impact), compresses (thump), flies off, a teal spark
K(A4_FOLD, 0.6, body=False); K(A4_FOLD + 0.25, 0.65, body=False)
HIT(A4_LAND, 0.95)
br.n(A4_LAND, 0.4, "A1 E2 G2 D3 E3".split(), 0.75)
bassnote(A4_LAND, 0.5, "A", 0.9)
Hh(A4_LAND + 0.25, 0.12)
dr.n(A4_PRESS, 0.2, "[BD CP OH]", 0.95); kb.n(A4_PRESS, 0.3, "A1", 0.95)
br.n(A4_PRESS, 0.25, "A1 E2 A2 C#3 E3".split(), 0.8)
bassnote(A4_PRESS, 0.25, "A", 1.0)
Q.n(A4_PRESS, 0.25, "C#5", 0.62); E.n(A4_PRESS, 0.25, "A3", 0.62)
gl.n(A4_SPARK, 0.4, ["E5", "A5"], 0.26); Hh(A4_SPARK, 0.12)
booms((A4_LAND, 0.45), (A4_PRESS, 0.6))

L_drg.set(50.0, 2.0)
L_grc.set(50.0, 220).ramp(54.0, 57.5, 320, "exp").set(58.0, 240).set(A4_LAND, 220)
L_epg.set(50.0, 0.45)
L_qpg.set(50.0, 0.34)
L_epc.set(50.0, 1400).ramp(54.0, 57.5, 2200, "exp").set(58.0, 1600).ramp(58.0, A4_LAND, 2200, "exp")
L_epr.set(50.0, 0.9)
L_qpc.set(50.0, 2800)
L_rvd.set(50.0, 0.86)
L_qg.set(50.0, 1.1); L_eg.set(50.0, 1.25)
L_qc.set(50.0, 2000); L_ec.set(50.0, 800)
L_arc.set(50.0, 700).ramp(50.0, 62.0, 1700, "exp").set(62.0, 1200)

# ====================================================================== A5 (64-76): score3 54-66, +10
at(SH_A5)
for bar in range(2):
    t0 = 54.0 + 2 * bar
    full_bar(g, 2 * t0, four=False, oh=False)
    for k in range(16):
        Hh(t0 + 0.125 * k, 0.05 + 0.05 * (k % 4 == 2))
K(58.0, 0.9); dr.n(58.0, 0.2, "OH", 0.4); K(60.0, 0.6)
# (the riser's drums are written below from the a5 cue list)
booms((58.0, 0.6))
for k in range(16):
    t = 54.0 + 0.25 * k
    bassnote(t, 0.25, "D", 0.9 if k % 2 == 0 else 0.55, growl=True, subv=0.75 if k % 2 == 0 else 0.0001)
sub.n(58.0, 3.0, "G1", 0.75)
for k in range(18):
    t = 61.0 + 0.25 * k
    bassnote(t, 0.25, "Bb" if t < 63.0 else "A", 0.7 + 0.015 * k)
L_grc.set(54.0, 150).ramp(61.0, 66.0, 600, "exp")
L_epg.set(54.0, 0.32).set(58.0, 0.4).ramp(61.0, 66.0, 0.55, "exp")
L_qpg.set(54.0, 0.26).set(58.0, 0.38).ramp(61.0, 66.0, 0.45, "exp")
L_epc.set(54.0, 600).set(58.0, 900).ramp(61.0, 66.0, 3200, "exp")
L_epr.set(54.0, 0.8).ramp(61.0, 66.0, 1.3)
L_qpc.set(54.0, 1300).set(58.0, 2600).ramp(61.0, 66.0, 3500, "exp")
L_rvd.set(54.0, 0.66).set(58.0, 0.93).set(61.0, 0.8)
for t, v, n, d in [(55.0, Q, "A4", 1.0), (57.0, Q, "F4", 1.0), (57.5, E, "F3", 0.5),
                   (59.0, Q, "A4", 2.0), (59.0, E, "D4", 2.0)]:
    v.n(t, d, n, 0.55)
for k, (q, e) in enumerate(zip("A3 C#4 E4 G4 A4 C#5".split(), "A2 C#3 E3 G3 A3 C#4".split())):
    Q.n(63.0 + 0.5 * k, 0.48, q, 0.62 + 0.04 * k)
    E.n(63.0 + 0.5 * k, 0.48, e, 0.62 + 0.04 * k)
L_qc.set(54.0, 1100); L_ec.set(54.0, 450)
arp(g, 108, 116, "D", 0.25, 0.42)
arp(g, 116, 122, "Dice", 0.25, 0.38)
arp(g, 122, 126, "BbR", 0.25, 0.42); arp(g, 126, 132, "AR", 0.25, 0.45)
L_arc.set(54.0, 560).ramp(58.0, 61.0, 500, "exp").ramp(61.0, 66.0, 3200, "exp")
L_arr.ramp(54.0, 66.0, 1.6)
L_arg.ramp(61.0, 66.0, 1.2, "exp")
gl.n(58.0, 2.0, ["D5", "A5"], 0.45)
# accents moved to the a5 cue list (global time)
at(0.0)
for k, (t, q, e) in enumerate(zip(A5_READS, ("F4", "E4", "A4"), ("A3", "C4", "A3"))):
    Q.n(t, 0.5 if k == 2 else 0.24, q, 0.6); E.n(t, 0.5 if k == 2 else 0.24, e, 0.62)   # both ways at once;
    xh.n(t, 0.1, "RS", 0.5 + 0.1 * (k == 2))                                            # the 3rd is the match
gq.n(A5_FROST[0], 0.5, "A5", 0.3); xh.n(A5_FROST[0], 0.1, "RS", 0.4)    # frost on Quill (left) ...
gl.n(A5_FROST[1], 0.5, "E5", 0.3); xh.n(A5_FROST[1], 0.1, "RS", 0.35)   # ... and on Ember (right)
for k, n in enumerate("D5 G5 A5 D6".split()):                          # LoRA chips light bottom-up
    gl.n(A5_LORA + 0.25 * k, 0.4, n, 0.16 + 0.03 * k); gq.n(A5_LORA + 0.25 * k, 0.4, n, 0.14 + 0.03 * k)
dr.n(69.875, 0.1, "RS", 0.3)                                           # the hops
# the build: cards launch on 8ths (kicks in quarters, 8th hats), then on 16ths (kicks in 8ths,
# 16th hats), a layer of open hats from the full mix, the main hit at 100 %
for k in range(4):
    K(A5_CARDS8 + 0.5 * k, 0.6 + 0.04 * k)
for k in range(8):
    Hh(A5_CARDS8 + 0.25 * k, 0.1 + 0.06 * (k % 2))
for k in range(int((A5_TOP - A5_CARDS16) / 0.25)):
    K(A5_CARDS16 + 0.25 * k, 0.72 + 0.02 * k, body=k % 2 == 0)
hats16(A5_CARDS16, A5_TOP, 0.07, 0.05)
for t in (A5_FULL + 0.25, A5_FULL + 0.75):
    Hh(t, 0.22, open_=True)
HIT(A5_TOP, 1.0)
br.n(A5_TOP, 0.4, "A1 E2 A2 C#3 E3".split(), 0.85)
bassnote(A5_TOP, 0.45, "A", 1.0)
booms((A5_TOP, 0.65))

# ====================================================================== A5b (76-80): NEW
# Chat training with a privileged teacher (cues/a5b_teacher.md). The a5 dominant is released
# on the downbeat onto a quiet Bbmaj7(#11) (VI): lighter, filters down. Card pops are glass
# plucks, the lock a rim click; Quill's colour swells into the original Ember, which writes
# the reply one note per token in 16ths (Ember's voice: D F G A Bb C, "?" on A). Then the joint
# pass of Quill and the linked Ember: a beat hit, a 16th climb in both voices at once, and the
# main hit on Fmaj9, the reply's own notes F-A-C(-G) sounding together; it decays into the cut.
at(0.0)
T = A5B_DOWN
K(T, 0.6)                                               # release: a soft downbeat, no crash
hats16(T, A5B_LAND, 0.05, 0.03)
K(A5B_LAND, 0.55)                                       # the original lands: soft thump
sub.n(T, 1.0, "Bb1", 0.6); sub.n(T + 1.0, 1.0, "G1", 0.6)
gl.n(A5B_TICKS[0], 0.3, "F5", 0.32); gq.n(A5B_TICKS[0], 0.3, "D5", 0.32)   # two Stockholm cards
gl.n(A5B_TICKS[1], 0.3, "A4", 0.3)                                          # Ember's Miami card
xh.n(A5B_TICKS[2], 0.1, "RS", 0.45); K(A5B_TICKS[2], 0.4, body=False)        # the locked slot
swell(A5B_TICKS[2], A5B_RUN[0] - 0.02, ["Bb3", "D4", "F4", "A4"], 0.35, 500, 3000)  # Quill's colour rises
for k, (t, n) in enumerate(zip(A5B_RUN, "D4 F4 G4 A4 Bb4 C5".split())):    # the teacher writes
    E.n(t, 0.12, n, 0.62 + 0.03 * k)
    Hh(t, 0.07 + 0.03 * (k % 2 == 0))
E.n(A5B_QMARK, 0.5, "A4", 0.7)                          # "?"
K(A5B_QMARK, 0.7); sub.n(A5B_QMARK, 0.5, "A1", 0.7)
# the joint pass: a beat hit, then both voices climb in 16ths at once (the 8 exchanges are a
# sweep inside this half beat), the kick doubling in 16ths underneath
HIT(A5B_JOINT, 0.85, "[BD CP]"); xh.n(A5B_JOINT, 0.1, "RS", 0.5)
br.n(A5B_JOINT, 0.25, "A1 E2 A2 D3".split(), 0.6)
bassnote(A5B_JOINT, 0.24, "A", 0.85); bassnote(A5B_JOINT + 0.25, 0.24, "A", 0.9, octave=2)
for k, (q, e, a) in enumerate(zip("E4 G4 A4 D5".split(), "A3 D4 E4 G4".split(), "A2 D3 E3 G3".split())):
    t = A5B_JOINT + 0.125 * k
    Q.n(t, 0.12, q, 0.6 + 0.05 * k); E.n(t, 0.12, e, 0.6 + 0.05 * k); ar.n(t, 0.12, a, 0.45 + 0.05 * k)
    if k:
        K(t, 0.6 + 0.08 * k, body=False)
# the whole target lights at once
HIT(A5B_HIT, 1.0)
br.n(A5B_HIT, 0.6, "F1 C2 F2 A2 C3 E3".split(), 0.9)
booms((A5B_HIT, 0.7))
bassnote(A5B_HIT, 0.75, "F", 1.0); sub.n(A5B_HIT + 0.75, 0.9, "F1", 0.5)
E.n(A5B_HIT, 0.75, ["F3", "C4"], 0.8); Q.n(A5B_HIT, 0.75, ["A4", "C5"], 0.75)
gl.n(A5B_HIT, 0.6, ["A4", "C5", "F5"], 0.32); gq.n(A5B_HIT + 0.125, 0.5, ["G4", "C5"], 0.26)
L_bdd.set(A5B_HIT - 0.05, 0.7).set(79.9, 0.24)
L_ohd.set(A5B_HIT - 0.05, 0.9).set(79.9, 0.45)
# arp: light 16ths, then 8ths on the sus chord (the climb and the hit are written above)
arp_t(T, 77.0, "Bb", 0.125, 0.3)
arp_t(77.0, A5B_QMARK, "Gm9", 0.125, 0.32)
arp_t(A5B_QMARK, A5B_JOINT, "AR", 0.25, 0.4)
# lighter on the downbeat, then everything opens towards the hit
L_epg.set(T, 0.36).ramp(A5B_RUN[0], A5B_HIT, 0.52, "exp")
L_qpg.set(T, 0.3).ramp(A5B_RUN[0], A5B_HIT, 0.45, "exp")
L_epc.set(T, 1300).ramp(A5B_TICKS[2], A5B_HIT, 3400, "exp")
L_qpc.set(T, 1800).ramp(A5B_TICKS[2], A5B_HIT, 3600, "exp")
L_epr.set(T, 0.9).ramp(A5B_JOINT, A5B_HIT, 1.25)
L_arc.set(T, 800).ramp(A5B_QMARK, A5B_HIT, 3000, "exp")
L_arg.set(T, 0.8)
L_grc.set(T, 200).ramp(A5B_QMARK, A5B_HIT, 600, "exp")
L_qc.set(T, 1300).ramp(A5B_JOINT, A5B_HIT, 1800, "exp")
L_ec.set(T, 560).ramp(A5B_RUN[0], A5B_HIT, 760, "exp")
L_rvd.set(T, 0.86)
# the hard cut into a6 (score3's 66.0)
L_eprl.ramp(79.8, 79.9, 0.08)
L_qprl.ramp(79.8, 79.9, 0.08)
L_rvw.ramp(79.95, A5B_CUT, 0.0)

# ====================================================================== A6 (80-96): score3 66-82, +14
at(SH_A6)
dr.n(70.0, 0.2, "[BD SD CP OH]", 1.0); kb.n(70.0, 0.3, "A1", 1.0)
for k in range(1, 6):
    K(70.0 + 0.5 * k)
    if k % 2:
        S(70.0 + 0.5 * k)
    if k < 5:
        Hh(70.0 + 0.5 * k + 0.25, 0.22, open_=True)
Hh(70.25, 0.22, open_=True)
RES = (73.0, 73.75, 74.5, 75.25)
for k, t in enumerate(RES):
    dr.n(t, 0.2, "[BD SD CP OH]", 0.85 if k < 3 else 1.0); kb.n(t, 0.3, "A1", 0.85 if k < 3 else 1.0)
    if k < 3:
        Hh(t + 0.375, 0.16)
for t in (75.75, 76.0, 76.5, 77.0, 77.5):
    K(t)
for t in (76.5, 77.5):
    S(t)
for t in (75.5, 76.25, 76.75, 77.25, 77.75):
    Hh(t, 0.24, open_=True)
K(77.875, 0.8, body=False)
dr.n(78.25, 0.3, "[BD OH]", 0.95); kb.n(78.25, 0.3, "A1", 0.95)
K(79.5, 0.45); K(80.0, 0.5)
L_ohd.set(69.95, 0.95).set(71.0, 0.5).set(72.95, 0.6).set(73.5, 0.5).set(75.2, 0.95).set(76.0, 0.5) \
    .set(78.2, 0.2)
L_bdd.set(69.95, 0.75).set(71.0, 0.24).set(75.2, 0.75).set(75.7, 0.24).set(78.2, 0.4).set(79.0, 0.24)
booms((70.0, 1.0), (73.0, 0.6), (75.25, 1.0), (78.25, 0.75))
for r, a, b in [("D", 70.0, 72.0), ("Bb", 72.0, 72.75)]:
    t = a; k = 0
    while t < b - 1e-6:
        bassnote(t, 0.25, r, 1.0 if k % 2 == 0 else 0.7, octave=2 if k % 4 == 3 else 1)
        t += 0.25; k += 1
for t, r in zip(RES[:3], ("Bb", "C", "A")):
    bassnote(t, 0.36, r, 1.0)
bassnote(75.25, 0.75, "D", 1.0)
for r, a, b in [("G", 76.0, 77.5), ("A", 77.5, 78.0)]:
    t = a; k = 0
    while t < b - 1e-6:
        bassnote(t, 0.24, r, 1.0 if k % 2 == 0 else 0.72, octave=2 if k % 4 == 3 else 1)
        t += 0.25; k += 1
bassnote(78.25, 0.25, "D", 0.95); sub.n(78.5, VOICE_PAIRING - VOICE_NEURAL, "D2", 0.55)
sub.n(VOICE_PAIRING - SH_A6, 80.0 - (VOICE_PAIRING - SH_A6), "A1", 0.55)   # under the Am7 carrier
bassnote(80.0, 2.0, "D", 0.7)
L_grc.set(70.0, 260).ramp(73.0, 78.25, 380, "exp").set(78.25, 200)
L_grd.set(70.0, 5.0).set(78.25, 4.0)
L_epg.set(70.0, 0.55).set(73.0, 0.3).set(75.25, 0.6).set(78.25, 0.4).set(78.4, 0.03).set(79.85, 0.45)
L_qpg.set(70.0, 0.45).set(73.0, 0.3).set(75.25, 0.5).set(78.25, 0.36).set(78.4, 0.03).set(79.85, 0.4)
L_epc.set(70.0, 2400).ramp(73.0, 78.25, 3200, "exp").ramp(78.25, 82.0, 1500, "smooth")
L_epr.set(70.0, 1.0)
L_qpc.set(70.0, 3000)
L_eprl.set(70.0, 1.0).set(73.0, 0.6).set(78.2, 0.15).set(79.9, 2.0)
L_qprl.set(70.0, 1.2).set(73.0, 0.7).set(78.2, 0.15).set(79.9, 2.4)
sw.n(68.75, 1.23, ["A2", "E3", "A3", "C#4", "G4"], 0.6)
sw.n(74.75, 0.48, ["D3", "A3", "D4", "F4"], 0.6)
for a, b in [(68.75, 69.98), (74.75, 75.23)]:
    L_swg.set(a, 0.02).ramp(a + 0.02, b, 0.9, "exp").set(b + 0.01, 0.0)
L_swc.set(68.75, 400).ramp(68.77, 69.98, 5000, "exp").set(74.75, 600).ramp(74.77, 75.23, 5000, "exp")
for k, n in enumerate("D4 A4 G4".split()):               # the real partner's lantern
    E.n(66.5 + 0.25 * k, 0.25, n, 0.5)
E.n(67.25, 0.7, "Bb4", 0.52)
theme = [(70.0, 0.5, "D4"), (70.5, 0.5, "A4"), (71.0, 0.5, "G4"), (71.5, 0.5, "Bb4"), (72.0, 0.7, "A4"),
         (73.0, 0.5, "F4"), (73.75, 0.5, "G4"), (74.5, 0.5, "A4"),
         (75.25, 0.5, "D5"), (75.75, 0.25, "C5"), (76.0, 0.75, "Bb4"), (76.75, 0.25, "A4"), (77.0, 0.5, "G4"),
         (77.5, 0.75, "C#5"), (78.25, 0.25, "D5")]
for t, d, n in theme:
    Q.n(t, d * 0.97, n, 0.88)
    E.n(t, d * 0.97, down(n), 0.88)
Q.n(80.0, 2.0, "D4", 0.75); E.n(80.0, 2.0, "D4", 0.75)
L_qg.set(70.0, 1.2)
L_eg.set(66.0, 0.9).set(70.0, 1.35)
L_qc.set(66.0, 1100).set(70.0, 2000)
L_ec.set(66.0, 450).set(70.0, 750)
arp(g, 140, 144, "Dm9", 0.5, 0.5); arp(g, 144, 145.5, "Bb", 0.5, 0.5)
arp(g, 151, 152, "Dm9", 0.5, 0.5); arp(g, 152, 155, "Gm9", 0.5, 0.5); arp(g, 155, 156, "A7", 0.5, 0.5)
L_arc.set(66.0, 3200).set(70.0, 1500).ramp(73.0, 78.25, 2600, "exp")
L_arr.set(70.0, 1.3)
L_arg.set(70.0, 0.9)
for t, c, d, v in [(70.0, "D2 A2 D3 F3 A3", 2.0, 1.0), (73.0, "Bb1 F2 Bb2 D3 F3", 0.4, 0.85),
                   (73.75, "C2 G2 C3 E3 G3", 0.4, 0.85), (74.5, "A1 E2 A2 C#3 E3", 0.4, 0.85),
                   (75.25, "D2 A2 D3 F3 A3 D4", 1.2, 1.0), (78.25, "D2 A2 E3 F3 A3", 0.25, 0.85)]:
    br.n(t, d, c.split(), v)
gl.n(79.5, 0.6, ["D5", "A5"], 0.3)
# the result windows' checks (cues/a4_games.md, a6 part), global time
at(0.0)
for t, n in zip(A6_CHECKS, ("E5", "D5", "F5", "D5")):
    gl.n(t, 0.3, n, 0.3)
at(SH_A6)
for t, n, d in [(68.0, "G#3", 0.75), (68.75, "D#4", 0.25), (69.0, "C#4", 0.25), (69.25, "E4", 0.7)]:
    gh.n(t, d, n, 0.6)
L_rvd.set(70.0, 0.86).set(78.3, 0.55).set(79.85, 0.86)
L_rvw.ramp(66.5, 67.5, 0.2, "smooth").ramp(69.9, 70.0, 0.26).ramp(78.3, 78.4, 0.08) \
    .ramp(79.85, 80.5, 0.26, "smooth")
LN["echo_feedback"].set(78.3, 0.0).set(79.85, 0.32)
# the voice: "neural" (78.5 -> 92.5) on Dm(add9), "pairing" (79.0 -> 93.0) on Bbmaj7, through
# Patina's vocoder, with score3's QA room around it (see NOTES3.md)
assert abs(78.5 + SH_A6 - VOICE_NEURAL) < 1e-9
lane("vox_mode", 1)
lane("vox_level", VOX_LEVEL)
lane("vox_dry", VOX_DRY)
# the carrier sits a scale step lower than score3's (Dm(add9) -> C(add9), Bbmaj7 -> Am7: each note down
# 2 semitones, or 1 where -2 would leave D minor), locked with the take's -2 semitones
_vp = VOICE_PAIRING - SH_A6
vx.n(78.5, _vp - 78.5, ["Bb2", "F3", "C4", "D4"], 0.9)          # v10: Bb(add9) (v5-v9: C(add9))
vx.n(_vp, 80.0 - _vp, ["G2", "D3", "F3", "Bb3"], 0.9)            # v10: Gm7 (v5-v9: Am7)
for p, val in [("waveform", 2), ("osc2_pitch", -12), ("osc2_level", 0.5), ("detune", 4), ("cutoff", VOX_CUT),
               ("resonance", 0.4), ("sub", 0.3)]:
    lane(f"voice.{p}", val)
lane("quill.echo_send", 0.22).set(77.45, 0.0).set(79.85, 0.22)
lane("ember.echo_send", 0.16).set(77.45, 0.0).set(79.85, 0.16)
L_subg.set(78.42, 0.12).set(79.85, 1.0)
L_qg.set(78.42, 0.08).set(79.85, 1.2)
L_eg.set(78.42, 0.08).set(79.85, 1.35)
L_grg.set(78.42, 0.3).set(79.85, 1.8)
lane("boom.gain", 1.0).set(78.42, 0.06).set(79.85, 1.0)
lane("brass.gain", 1.0).set(78.42, 0.05).set(79.85, 1.0)
lane("brass.release", 0.7).set(78.2, 0.1).set(79.9, 0.7)
# the pads' (inaudible, ducked) chords on "pairing" are not played: the vocoder carrier needs the cards
ep.ev = [e for e in ep.ev if abs(e[0] - 93.0) > 1e-6]
qp.ev = [e for e in qp.ev if abs(e[0] - 93.0) > 1e-6]
at(0.0)

# ====================================================================== BREAK (v5 40-48): NEW
# ab_plate (cues/ab_plate.md): the film turns into an engraved anatomical-atlas plate and back.
# The groove's last hit on 40.0, then everything drops on 40.25: the bus goes intimate and analog
# (tape wow and age up, a small dark room), the pads close to a murmur, and a felt piano, low bowed
# strings and a soft cassette pad play four bars, Dm(add9) - Bbmaj7 - Gm(add9) - A7sus4/A7. The
# piano's upper line spells D-A-G-Bb across bars 2-3, landing G on the split (43.5) and Bb on the
# compass (44.0): the lantern motif, drawn into the atlas before a3 plays it. The drawing is heard
# in the FX stem (nib, pen flicks, ruler ratchet, compass scrape, stipple, paper slides and folds,
# dusty room tone); a felted thump presses the plate (41.0), glass slides lift the vertebra and
# write the write; from 46.5 light soaks through: a riser, a filtered groove fading in, the low
# end back at 47.0, 16th kicks at 47.75 and the hard re-entry on 48.0.
RAW = True; at(0.0)
fe = track("felt", "patch=feltkeys reverb_send=0.3 echo_send=0.08 pan=-0.12")
st = track("strings", "patch=glasswing reverb_send=0.35 chorus_send=0.25 pan=0.1")
vl = track("vellum", "patch=vellum reverb_send=0.4 pan=0.05")
# the last hit of a2's groove, then nothing but the plate
HIT(BRK_LAST, 0.8, "[BD OH]"); sub.n(BRK_LAST, 0.25, "D2", 0.6)
BRK_BARS = [  # (start, end, epad (murmur), strings (low, from the press), cassette pad, piano left hand)
    (40.0, 42.0, "D2 A2 E3", "D3 A3", "F4 A4 E5", "D2 A2"),
    (42.0, 44.0, "Bb1 F2 D3", "Bb2 F3", "D4 F4 A4", "Bb1 F2"),
    (44.0, 46.0, "G1 D2 Bb2", "G2 D3", "Bb3 D4 A4", "G1 D2"),
    (46.0, 47.0, "A1 E2 D3", "A2 E3", "D4 E4 G4", "A1 E2"),
    (47.0, 48.0, "A1 E2 C#3", "A2 E3", "C#4 E4 G4", None)]
for a, b, e_, s_, v_, lh in BRK_BARS:
    ep.n(a, b - a, e_.split(), 0.4)
    a_s = max(a, BRK_PRESS)
    st.n(a_s, b - a_s - 0.02, s_.split(), 0.42)
    a_v = max(a, BRK_INK)
    vl.n(a_v, b - a_v - 0.02, v_.split(), 0.32)
    if lh:
        fe.n(max(a, BRK_PRESS), 1.0 if a < 46 else 1.0, lh.split(), 0.45)
for t, n, d, v in [(41.5, "E4", 0.5, 0.36), (BRK_PULSE + 0.5, "D4", 0.5, 0.4), (BRK_LIFT, "A4", 0.5, 0.42),
                   (BRK_SPLIT, "G4", 0.5, 0.48), (BRK_COMPASS, "Bb4", 1.0, 0.48), (BRK_SUM, "A4", 0.75, 0.42),
                   (BRK_FOLD + 0.5, "E4", 0.5, 0.38)]:
    fe.n(t, d, n, v)
for k, n in enumerate("A3 C#4 E4 G4".split()):                  # rising into the re-entry
    fe.n(BRK_LOW + 0.25 * k, 0.25, n, 0.42 + 0.05 * k)
# the press, the pulse, the split, the gates: felted, dry, small
K(BRK_PRESS, 0.6); sub.n(BRK_PRESS, 0.5, "D2", 0.6)
sub.n(BRK_PULSE, 0.4, "Bb1", 0.5)
xh.n(BRK_SPLIT, 0.1, "RS", 0.55); dr.n(BRK_SPLIT, 0.1, "CP", 0.3)
gl.n(BRK_SPLIT, 0.3, ["D5", "G5"], 0.3)
for t, n in zip(BRK_POPS, ("A5", "E5", "D5")):
    (gq if t < 45 else gl).n(t, 0.25, n, 0.26)
for t in BRK_GATES:
    xh.n(t, 0.1, "RS", 0.25)
gq.n(BRK_FAN, 0.5, ["D5", "A5"], 0.2); gl.n(BRK_FAN, 0.5, ["G5", "D6"], 0.16)   # the attention fan
# glass slides: the vertebra lifts (up), the write goes into the stream (down); deliberate glides
sg.n(BRK_LIFT, 0.25, "A4", 0.4); sg.n(BRK_LIFT + 0.25, 0.25, "D5", 0.42)
sg.n(BRK_GLISS, 0.125, "D5", 0.38); sg.n(BRK_GLISS + 0.125, 0.25, "A4", 0.36); sg.n(BRK_GLISS + 0.375, 0.25, "F4", 0.32)
swell(44.5, BRK_SUM - 0.02, ["G3", "D4", "A4"], 0.35, 400, 3000)  # the values flow into one node
# from 46.5: light soaks through; a filtered groove fades in, the low end returns, the lift
arp_t(BRK_RISE, BRK_END, "AR", 0.125, 0.32)
hats16(BRK_RISE, BRK_END, 0.03, 0.03)
K(BRK_LOW, 0.5); K(BRK_LOW + 0.5, 0.6, body=False)
sub.n(BRK_LOW, BRK_END - BRK_LOW - 0.02, "A1", 0.6)
K(BRK_PRE, 0.45, body=False); K(BRK_PRE + 0.125, 0.55, body=False)
br.n(BRK_END, 0.4, "D2 A2 D3 F3 A3".split(), 0.85)              # the re-entry: brass on Dm, ...
swell(46.75, BRK_END - 0.02, ["A2", "E3", "G3", "C#4", "E4"], 0.5, 400, 5000)
L_epg.set(BREAK_AT, 0.22); L_qpg.set(BREAK_AT, 0.0)
L_epc.set(BRK_INK, 380); L_epr.set(BREAK_AT, 0.7)
L_drg.set(BRK_LAST, 1.6).set(BRK_INK, 1.0).set(BRK_PRE - 0.05, 2.0)   # the re-entry hits at full level
L_bdd.set(BRK_PRESS - 0.05, 0.14).set(BRK_PRESS + 0.3, 0.24)
L_arc.set(BRK_RISE - 0.05, 300).ramp(BRK_RISE, BRK_END - 0.05, 2400, "exp")
L_arg.set(BRK_RISE - 0.05, 0.6)
L_subg.set(BREAK_AT, 0.8).ramp(BRK_LOW, BRK_END - 0.05, 1.0)
for p_, v0_, v_ in [("tape_wow", 0.0, None), ("tape_age", 0.1, 0.45), ("reverb_decay", None, 0.55),
                    ("reverb_tone", 4200, 2400), ("reverb_wet", None, 0.3)]:
    l_ = lane(p_, v0_)
    if p_ == "tape_wow":
        l_.set(BRK_INK, 0.0).ramp(BRK_INK, BRK_PRESS, 0.22, "smooth").ramp(BRK_RISE, 47.7, 0.0, "smooth")
    else:
        l_.set(BRK_INK, v_)
    l_.set(BRK_END - 0.05, {"tape_wow": 0.0, "tape_age": 0.1, "reverb_decay": 0.86, "reverb_tone": 4200,
                             "reverb_wet": 0.26}[p_])
lane("felt.gain", 1.0); lane("strings.gain", 0.8); lane("vellum.gain", 0.7)
lane("strings.cutoff", 1200)
RAW = False

# ====================================================================== A5C (v6 88-92): NEW
# a5c_talk (cues/a5c_talk.md). Before chat training: one sentence six times, word for word. It is
# one dry stab, the same dyad in both voices with a rim, on a static bare fifth in a dried-out
# room: on 88.25, then on five 16ths in a row (88.5-89.0), mechanical and a bit comic; the last one
# rings and stutters in the echo, a reverse swell clears it. After training (89.5): livelier, a
# rising pass, and a real conversation, each line a different phrase in its speaker's voice
# (Ember 90.0 on the main hit, Quill 90.375, Ember 90.75, Quill 91.0) over Bbmaj9 -> C(add9),
# resolving on F(add9) for the stat; a6's hard cut at 92.
RAW = True; at(0.0)
for a, b, e_, q_ in [(88.0, A5C_AFTER, "D2 A2 D3", None), (A5C_AFTER, 90.0, "A1 E2 G2 D3", "D4 E4 G4"),
                     (90.0, 91.0, "Bb1 F2 D3 A3 C4", "D4 F4 A4"), (91.0, A5C_STAT, "C2 G2 D3 E3", "E4 G4 D5"),
                     (A5C_STAT, 92.0, "F1 C2 A2 G3", "A3 C4 G4")]:
    ep.n(a, b - a, e_.split(), 0.45)
    if q_:
        qp.n(a, b - a, q_.split(), 0.4)
K(88.0, 0.5); gq.n(A5C_CARD, 0.2, "A5", 0.25)
for k, t in enumerate(A5C_ECHO):                                   # the same stab, every turn
    v_ = E if k % 2 == 0 else Q
    v_.n(t, 0.25 if k == 5 else 0.1, ["D4", "A4"], 0.62)
    xh.n(t, 0.1, "RS", 0.45)
lane("ember.echo_send").set(88.9, 0.35).set(89.4, 0.16)          # the last one stutters
lane("quill.echo_send").set(88.9, 0.4).set(89.4, 0.22)
swell(A5C_REV, A5C_AFTER - 0.02, ["D3", "A3", "D4"], 0.4, 400, 3500)
sub.n(88.0, 1.5, "D2", 0.5)
climb(A5C_AFTER, 90.0, 0.125, "A2", "E4", 0.32, 0.45)               # the pass: a rising arpeggio
hats16(A5C_AFTER, 91.875, 0.05, 0.04)
HIT(A5C_TALK[0], 0.85, "[BD CP]"); gq.n(A5C_TALK[0] + 0.125, 0.3, "A5", 0.3)   # "Oslo": the violet burst
TALK = [(E, "D4 F4 A4 C5"), (Q, "D5 C5 A4"), (E, "F4 G4 A4"), (Q, "E5 D5 C5 G4")]   # four different lines
for t, (v_, line) in zip(A5C_TALK, TALK):
    ns = line.split()
    for j_, n in enumerate(ns):
        v_.n(t + 0.125 * j_, 0.12 if j_ < len(ns) - 1 else 0.25, n, 0.66 if j_ == 0 else 0.6)
K(90.5, 0.6); dr.n(A5C_TALK[2], 0.1, "RS", 0.4); S(A5C_TALK[3], 0.6); K(91.25, 0.5)
Q.n(A5C_STAT, 0.5, "A4", 0.5); E.n(A5C_STAT, 0.5, "C4", 0.5)    # the stat: resolve
bassnote(A5C_AFTER, 0.5, "A", 0.6); bassnote(90.0, 1.0, "Bb", 0.8); bassnote(91.0, 0.375, "C", 0.75)
sub.n(A5C_STAT, 92.0 - A5C_STAT - 0.1, "F1", 0.55)
arp_t(90.0, 91.0, "Bb", 0.125, 0.32); arp_t(91.0, A5C_STAT, "C9", 0.125, 0.32)
L_eprl.set(88.0, 1.0).ramp(91.8, 91.9, 0.08)
L_qprl.set(88.0, 1.2).ramp(91.8, 91.9, 0.08)
L_epg.set(88.0, 0.3).set(A5C_AFTER, 0.42); L_qpg.set(88.0, 0.0).set(A5C_AFTER, 0.34)
L_epc.set(88.0, 500).set(A5C_AFTER, 1600); L_qpc.set(A5C_AFTER, 2400)
L_rvw.set(88.0, 0.08).set(A5C_AFTER, 0.24)
L_rvd.set(88.0, 0.5).set(A5C_AFTER, 0.8)
L_qg.set(88.0, 1.0); L_eg.set(88.0, 1.1)
L_qc.set(88.0, 1200); L_ec.set(88.0, 520)
L_drg.set(88.0, 1.4)
L_arc.set(A5C_AFTER, 700).ramp(A5C_AFTER, 90.0, 1600, "exp")
RAW = False

# ====================================================================== v7 helpers (final v7 times)
ARP.update({"F9": "F1 C2 G2 A2 C3 A2 G2 C2", "Asus": "E3 A3 D4 A3 E4 A3 D4 A3",
            "BbH": "Bb2 F3 A3 D4 C4 A3 F3 D3", "GmH": "G2 D3 F3 Bb3 A3 F3 D3 Bb2",
            "F9H": "F2 C3 G3 A3 C4 A3 G3 C3"})   # (Asus / BbH sit above C3: low arp notes read flat in tuning QA)


def pads(rows, ve=0.5, vq=0.45):
    for a, b, e_, q_ in rows:
        ep.n(a, b - a, e_.split(), ve)
        if q_:
            qp.n(a, b - a, q_.split(), vq)


def pcs_of(rows, t):
    for a, b, e_, q_ in rows:
        if a <= t < b:
            return {midi(n) % 12 for n in (e_ + " " + (q_ or "")).split()}
    return {2, 9}


def run(trk, times, rows, lo, hi, v0, v1, dur=0.12, zig=True):
    """a rising run through the chord tones sounding at each step (a small zig-zag inside)"""
    lo_, hi_, n = midi(lo), midi(hi), len(times)
    for b, t in enumerate(times):
        target = lo_ + (hi_ - lo_) * b / max(1, n - 1) + ((2, -2, 0)[b % 3] if zig else 0)
        pcs = pcs_of(rows, t)
        m = min((x for x in range(lo_ - 3, hi_ + 4) if x % 12 in pcs), key=lambda x: abs(x - target))
        trk.n(t, dur, name(m), v0 + (v1 - v0) * b / max(1, n - 1))


def steps(t0, t1, dt):
    return [t0 + dt * k for k in range(round((t1 - t0) / dt))]


# ====================================================================== A2 generation (v9 32.5-36): RECOMPOSED
# v8's 32.5-40 (a token every beat over Bbmaj7 - Gm9 - Em7b5 - A7, the fold on Dm at 37.5, a slow swell down)
# runs 2.5x / 1.7x faster in the v9 picture (a token every 0.2 s). The same walk in one bar: a chord
# per beat, both voices writing dyads on 8ths and then on 16ths over A7 (the A7 arpeggio rising into the
# fold), kicks doubling, 16th arp and hats; the fold hits on Dm at 34.5 and the swell down (a heartbeat,
# A4 / A3 / E4) is shortened to 35-36, into ab_plate's last groove hit.
N9 = True; at(0.0)
G9 = [(A2G_START, 33.0, "Bb1 F2 D3 A3", "D4 F4 A4"),            # Bbmaj7
      (33.0, 33.5, "G1 D2 Bb2 F3", "A3 D4 F4"),                  # Gm9
      (33.5, 34.0, "E2 Bb2 D3 G3", "D4 G4 Bb4"),                 # Em7b5
      (34.0, A2G_FOLD, "A1 E2 G2 C#3", "C#4 E4 G4"),             # A7
      (A2G_FOLD, A2G_END, "D2 A2 E3 F3", "A3 E4 F4")]            # the fold: Dm(add9)
pads(G9)
GEN9 = [(32.5, "D5", "Bb3"), (32.75, "F5", "A3"), (33.0, "D5", "Bb3"), (33.25, "A4", "F3"),
        (33.5, "Bb4", "G3"), (33.75, "D5", "Bb3"), (34.0, "A4", "E3"), (34.125, "C#5", "G3"),
        (34.25, "E5", "A3"), (34.375, "G5", "C#4")]
for k, (t, q, e) in enumerate(GEN9):
    d_ = 0.24 if t < 34.0 else 0.12
    Q.n(t, d_, q, 0.66 + 0.012 * k); E.n(t, d_, e, 0.68 + 0.012 * k)
Q.n(A2G_FOLD, 0.5, "D5", 0.8); E.n(A2G_FOLD, 0.75, "D4", 0.8)      # the fold: both on D
Q.n(A2G_RETURN, 0.5, "A4", 0.5); E.n(A2G_RETURN + 0.25, 0.5, "A3", 0.45); Q.n(35.5, 0.45, "E4", 0.42)
for t in (32.5, 33.0, 33.5):
    K(t)
S(33.0); S(34.0, 0.85)
K(33.75, 0.75, body=False)
for k, t in enumerate(steps(34.0, A2G_FOLD, 0.125)):
    K(t, 0.7 + 0.05 * k, body=k % 2 == 0)
hats16(A2G_START, A2G_FOLD, 0.06, 0.05)
HIT(A2G_FOLD, 1.0, "[BD SD CP OH]")                                 # the fold
K(A2G_RETURN, 0.5); Hh(35.25, 0.08); K(35.75, 0.4)                  # swell down: a heartbeat
booms((A2G_FOLD, 0.7))
br.n(A2G_FOLD, 0.75, "D2 A2 D3 E3 A3".split(), 0.8)
swell(33.75, A2G_FOLD - 0.02, ["A2", "E3", "G3", "C#4"], 0.45, 400, 4500)
eighths([("Bb", A2G_START, 33.0), ("G", 33.0, 33.5), ("E", 33.5, 34.0), ("A", 34.0, A2G_FOLD)], accent_oct=4)
bassnote(A2G_FOLD, 0.5, "D", 1.0); sub.n(A2G_FOLD + 0.5, A2G_END - A2G_FOLD - 0.5, "D2", 0.55)
for a, b, c in [(A2G_START, 33.0, "Bb"), (33.0, 33.5, "Gm9"), (33.5, 34.0, "Eh"), (34.0, A2G_FOLD, "A7"),
                (A2G_FOLD, A2G_END, "D")]:
    arp_t(a, b, c, 0.125, 0.4 + 0.04 * (a >= 33.5))
L_drg.set(A2G_START, 2.0).set(A2G_FOLD + 0.375, 1.2)
L_ohd.set(A2G_FOLD - 0.05, 0.95).set(A2G_RETURN, 0.45)
L_bdd.set(A2G_FOLD - 0.05, 0.75).set(A2G_RETURN, 0.24)
L_grd.set(33.5, 4.6).set(34.875, 4.0)
L_epg.set(33.5, 0.5).ramp(A2G_RETURN, 35.5, 0.3)
L_epc.ramp(A2G_RETURN, 35.5, 700, "exp"); L_epr.ramp(A2G_RETURN, 35.5, 0.9)
L_qpc.ramp(A2G_RETURN, 35.5, 1300, "exp")
L_arc.ramp(A2G_RETURN, 35.875, 500, "exp")
L_arg.set(33.5, 0.9).ramp(A2G_RETURN, 35.9, 0.35)
N9 = False

# ====================================================================== A5b (v9 80-86): RE-PLACED
# a5b_teacher at its v9 times (cues/a5b_teacher.md through the time map): the a5 dominant released on
# the downbeat onto Bbmaj7(#11), lighter; the original lands (soft thump), card plucks, the lock click,
# Quill's colour rises; the teacher's run on 16ths placed at the picture's tokens (D F G A Bb C; the
# picture writes them at 1/6 s, each note within 42 ms); "?" on A (82.5) now rings over A7sus4 for
# three quarters of a bar while the sentence slides over (an Asus arp in 8ths and a soft swish); the
# joint pass (83.75) climbs in 16ths in both voices for a beat and a half; the main hit on Fmaj9 (84.5)
# gets a full bar of tail (a slow glass descent), cut at 86 for a5c.
N9 = True; at(0.0)
T = A5B9_DOWN
B9 = [(T, A5B9_RUN[0], "Bb1 F2 D3 A3 E4", "D4 F4 A4"),           # Bbmaj7(#11): release
      (A5B9_RUN[0], A5B9_QMARK, "G1 D2 Bb2 F3", "Bb3 D4 A4"),      # Gm9: the teacher writes
      (A5B9_QMARK, A5B9_HIT, "A1 E2 G2 D3", "D4 E4 G4"),           # A7sus4: "?", the slide, the joint pass
      (A5B9_HIT, 86.0, "F2 C3 E3 G3", "A3 C4 G4")]                 # Fmaj9: the whole target lights
pads(B9)
K(T, 0.8); hats16(T, A5B9_LAND, 0.05, 0.03)
K(A5B9_LAND, 0.55)
sub.n(T, A5B9_RUN[0] - T, "Bb1", 0.6); sub.n(A5B9_RUN[0], A5B9_QMARK - A5B9_RUN[0], "G1", 0.6)
gl.n(A5B9_TICKS[0], 0.3, "F5", 0.32); gq.n(A5B9_TICKS[0], 0.3, "D5", 0.32)   # two Stockholm cards
gl.n(A5B9_TICKS[1], 0.3, "A4", 0.3)                                          # Ember's Miami card
xh.n(A5B9_TICKS[2], 0.1, "RS", 0.45); K(A5B9_TICKS[2], 0.4, body=False)        # the locked slot
swell(A5B9_TICKS[2], A5B9_RUN[0] - 0.02, ["Bb3", "D4", "F4", "A4"], 0.35, 500, 3000)   # Quill's colour rises
for k, (t, n) in enumerate(zip(A5B9_RUN, "D4 F4 G4 A4 Bb4 C5".split())):    # the teacher writes
    E.n(t, 0.12, n, 0.62 + 0.03 * k)
    Hh(t, 0.07 + 0.03 * (k % 2 == 0))
E.n(A5B9_QMARK, 0.7, "A4", 0.7)                                     # "?": it rings
K(A5B9_QMARK, 0.7); sub.n(A5B9_QMARK, A5B9_JOINT - A5B9_QMARK - 0.02, "A1", 0.65)
Q.n(83.0, 0.5, "E4", 0.4)                                           # Quill answers softly in the hold
swell(A5B9_SLIDE, A5B9_JOINT - 0.02, ["A2", "E3", "G3", "D4"], 0.35, 400, 3500)   # the slide (swish)
HIT(A5B9_JOINT, 0.85, "[BD CP]"); xh.n(A5B9_JOINT, 0.1, "RS", 0.5)  # the joint pass
br.n(A5B9_JOINT, 0.25, "A1 E2 A2 D3".split(), 0.6)
bassnote(A5B9_JOINT, 0.24, "A", 0.85); bassnote(A5B9_JOINT + 0.25, 0.24, "A", 0.9, octave=2)
bassnote(A5B9_JOINT + 0.5, 0.24, "A", 0.8)
for k, (q, e, a) in enumerate(zip("D4 E4 G4 A4 D5 E5".split(), "G3 A3 D4 E4 G4 A4".split(),
                                  "A2 D3 E3 G3 A3 D4".split())):
    t = A5B9_JOINT + 0.125 * k
    Q.n(t, 0.12, q, 0.56 + 0.04 * k); E.n(t, 0.12, e, 0.56 + 0.04 * k); ar.n(t, 0.12, a, 0.42 + 0.04 * k)
    if k:
        K(t, 0.55 + 0.06 * k, body=False)
HIT(A5B9_HIT, 1.0)                                                  # the whole target lights
br.n(A5B9_HIT, 0.6, "F1 C2 F2 A2 C3 E3".split(), 0.9)
booms((A5B9_HIT, 0.7))
bassnote(A5B9_HIT, 0.75, "F", 1.0); sub.n(A5B9_HIT + 0.75, 86.0 - A5B9_HIT - 0.85, "F1", 0.5)
E.n(A5B9_HIT, 1.0, ["F3", "C4"], 0.8); Q.n(A5B9_HIT, 1.0, ["A4", "C5"], 0.75)
gl.n(A5B9_HIT, 0.6, ["A4", "C5", "F5"], 0.32); gq.n(A5B9_HIT + 0.125, 0.5, ["G4", "C5"], 0.26)
for k, n in enumerate("C6 A5 G5 F5".split()):                     # the tail: a slow glass descent
    (gl if k % 2 == 0 else gq).n(85.0 + 0.25 * k, 0.4, n, 0.2 - 0.02 * k)
K(85.5, 0.4, body=False)
L_bdd.set(A5B9_HIT - 0.05, 0.7).set(85.9, 0.24)
L_ohd.set(A5B9_HIT - 0.05, 0.9).set(85.9, 0.45)
arp_t(T, A5B9_RUN[0], "Bb", 0.125, 0.3)
arp_t(A5B9_RUN[0], A5B9_QMARK, "GmH", 0.125, 0.3)
arp_t(A5B9_QMARK, A5B9_JOINT, "Asus", 0.25, 0.36)
L_epg.set(T, 0.36).ramp(A5B9_RUN[0], A5B9_HIT, 0.52, "exp")
L_qpg.set(T, 0.3).ramp(A5B9_RUN[0], A5B9_HIT, 0.45, "exp")
L_epc.set(T, 1300).ramp(A5B9_TICKS[2], A5B9_HIT, 3400, "exp").ramp(85.0, 85.9, 1800, "exp")
L_qpc.set(T, 1800).ramp(A5B9_TICKS[2], A5B9_HIT, 3600, "exp")
L_epr.set(T, 0.9).ramp(A5B9_JOINT, A5B9_HIT, 1.25)
L_arc.set(T, 800).ramp(A5B9_QMARK, A5B9_HIT, 3000, "exp")
L_arg.set(T, 0.8); L_arr.set(T, 1.3)
L_grc.set(T, 200).ramp(A5B9_QMARK, A5B9_HIT, 600, "exp")
L_qc.set(T, 1300).ramp(A5B9_JOINT, A5B9_HIT, 1800, "exp")
L_ec.set(T, 560).ramp(A5B9_RUN[0], A5B9_HIT, 760, "exp")
L_qg.set(T, 1.0); L_eg.set(T, 1.1)
L_rvd.set(T, 0.86); L_rvw.set(T, 0.26)
L_drg.set(T, 1.6)
L_eprl.set(T, 1.6).ramp(85.8, 85.9, 0.08)                          # the hard cut into a5c
L_qprl.set(T, 2.4).ramp(85.8, 85.9, 0.08)
L_rvw.ramp(85.95, A5C9_DOWN, 0.0)
N9 = False

# ====================================================================== A5C (v9 86-92 = v10 82-88): RE-PLACED
# v10: no stat; after the main hit the conversation runs six lines on half-beats (Ember, Quill, ...),
# each its own phrase, over Bbmaj9 -> C(add9) -> F(add9); the bubbles fade (v10 87.65-87.9) on F.
# a5c_talk at its v9 times: one dry stab on a bare fifth in a dried-out room (86.375, ringing for
# three 16ths), five repeats placed at the picture's turns (each within 42 ms), a longer hold that
# lets the last one stutter in the echo (87.375-88.0), the reverse swell, then after training (88.25)
# a rising pass over A7sus4 and the conversation: Ember on the main hit (89.0, Bbmaj9) and its line
# holding through "I noticed you mentioned you're in Oslo.", Quill (89.875), Ember (90.125), Quill
# (90.375, over C(add9)), and the stat resolving on F(add9) with a long, quiet hold to the cut.
N9 = True; at(0.0)
C9 = [(A5C9_DOWN, A5C9_AFTER, "D2 A2 D3", None), (A5C9_AFTER, A5C9_TALK[0], "A1 E2 G2 D3", "D4 E4 G4"),
      (A5C9_TALK[0], A5C9_TALK[2], "Bb1 F2 D3 A3 C4", "D4 F4 A4"),   # Bbmaj9: "...you're in Oslo." / Quill
      (A5C9_TALK[2], A5C9_TALK[4], "C2 G2 D3 E3", "E4 G4 D5"),        # C(add9): the seafood turns
      (A5C9_TALK[4], 92.0, "F1 C2 A2 G3", "A3 C4 G4")]                # F(add9): the last two lines, fade
for a, b, e_, q_ in C9:
    ep.n(a, b - a, e_.split(), 0.45)
    if q_:
        qp.n(a, b - a, q_.split(), 0.4)
K(A5C9_DOWN, 0.5); gq.n(A5C9_CARD, 0.2, "A5", 0.25)
for k, t in enumerate(A5C9_ECHO):                                  # the same stab, every turn
    v_ = E if k % 2 == 0 else Q
    v_.n(t, 0.3 if k == 0 else (0.25 if k == 5 else 0.1), ["D4", "A4"], 0.62)
    xh.n(t, 0.1, "RS", 0.45)
lane("ember.echo_send").set(87.3, 0.35).set(87.9, 0.16)           # the last one stutters
lane("quill.echo_send").set(87.3, 0.4).set(87.9, 0.22)
swell(87.75, A5C9_AFTER - 0.02, ["D3", "A3", "D4"], 0.4, 400, 3500)
sub.n(A5C9_DOWN, A5C9_AFTER - A5C9_DOWN - 0.02, "D2", 0.5)
run(ar, steps(A5C9_AFTER, A5C9_TALK[0], 0.125), C9, "A2", "E4", 0.32, 0.45)   # the pass: a rising arpeggio
hats16(A5C9_AFTER, A5C9_TALK[4], 0.05, 0.04)
for t in steps(A5C9_TALK[4], 91.75, 0.25):
    Hh(t, 0.06)
HIT(A5C9_TALK[0], 0.85, "[BD CP]"); gq.n(A5C9_TALK[0] + 0.125, 0.3, "A5", 0.3)   # "Oslo": the violet burst
TALK10 = [(E, "D4 F4 A4 C5"), (Q, "D5 C5 A4"), (E, "G4 E4 D4"), (Q, "E5 D5 G4"), (E, "F4 G4 A4"), (Q, "C5 A4")]
for k, (t, (v_, line)) in enumerate(zip(A5C9_TALK, TALK10)):          # six lines, each a different phrase
    ns = line.split()
    for j_, n in enumerate(ns):
        v_.n(t + 0.125 * j_, 0.12 if j_ < len(ns) - 1 else (0.4 if k == 5 else 0.25), n, 0.66 if j_ == 0 else 0.6)
K(A5C9_TALK[1], 0.6); dr.n(A5C9_TALK[2], 0.1, "RS", 0.4); S(A5C9_TALK[3], 0.6)
K(A5C9_TALK[4], 0.6); dr.n(A5C9_TALK[5], 0.1, "RS", 0.35)
bassnote(A5C9_AFTER, 0.5, "A", 0.6); bassnote(A5C9_TALK[0], 0.98, "Bb", 0.8)
bassnote(A5C9_TALK[2], 0.98, "C", 0.75); bassnote(A5C9_TALK[4], 0.48, "F", 0.7)
sub.n(A5C9_TALK[5], 92.0 - A5C9_TALK[5] - 0.1, "F1", 0.5)
arp_t(A5C9_TALK[0], A5C9_TALK[2], "BbH", 0.125, 0.3); arp_t(A5C9_TALK[2], A5C9_TALK[4], "C9", 0.125, 0.3)
arp_t(A5C9_TALK[4], A5C9_FADE + 0.1, "F9H", 0.25, 0.26)
L_qg.set(A5C9_FADE, 1.0).ramp(A5C9_FADE, 91.9, 0.7); L_eg.set(A5C9_FADE, 1.1).ramp(A5C9_FADE, 91.9, 0.8)   # bubbles fade
L_eprl.set(A5C9_DOWN, 1.0).ramp(91.8, 91.9, 0.08)
L_qprl.set(A5C9_DOWN, 1.2).ramp(91.8, 91.9, 0.08)
L_epg.set(A5C9_DOWN, 0.3).set(A5C9_AFTER, 0.42); L_qpg.set(A5C9_DOWN, 0.0).set(A5C9_AFTER, 0.34)
L_epc.set(A5C9_DOWN, 500).set(A5C9_AFTER, 1600); L_qpc.set(A5C9_AFTER, 2400)
L_rvw.set(A5C9_DOWN, 0.08).set(A5C9_AFTER, 0.24)
L_rvd.set(A5C9_DOWN, 0.5).set(A5C9_AFTER, 0.8)
L_qg.set(A5C9_DOWN, 1.0); L_eg.set(A5C9_DOWN, 1.1)
L_qc.set(A5C9_DOWN, 1200); L_ec.set(A5C9_DOWN, 520)
L_drg.set(A5C9_DOWN, 1.4)
L_arc.set(A5C9_AFTER, 700).ramp(A5C9_AFTER, A5C9_TALK[0], 1600, "exp")
N9 = False

# ====================================================================== A6 proof (v11 82-88): NEW
# a6_proof v11 (cues/V11_TIMEMAP.md): only Quill is given the word. A dark Dm(add9) under the gold card
# (a glass ping, 82.125) and Ember's locked slot (a low lock click, 82.375); the card drops into Quill
# (a glass slide, the land 83.125); one pass over Bbmaj7: gold climbs and crosses the link layer by
# layer, a rising 16th figure in both voices and the arp (83.25-83.75); Ember writes "cipher" (G on
# A7sus4, 84.0) and 99.9 % lands as the glad MAIN HIT on F(add9) (84.375). The link retracts (85.0:
# dead stop), whoosh out; the grey Quill enters on a sour, audible stab (85.5: the "sour" voice, a
# low kick and rim), its link plugs in (85.75), its pass is a cold rising figure in glass doubled by
# the sour voice (85.875-86.25); and Ember's grey "?" (86.75, no number) is a full,
# deflated MAIN HIT at least as loud as 99.9 %: kit, a boom that sags, the tritone brass of the film's
# first shatter (D G# D G#), the sour voice and the growl; it hangs over a bare fifth to the hard cut
# (88.0). The 180-test medians are gone.
N11 = True; at(0.0)
# v11: the seam at 76.0. a5_train's 99.7 % (75.5, on A) now cuts straight into "before chat training":
# a clean, dry hit on the downbeat (kick and rim, a short bare-fifth brass, the room already dry) that
# resolves the A onto a5c's bare D fifth
dr.n(76.0, 0.1, "[BD RS]", 0.9); kb.n(76.0, 0.3, "A1", 0.85)
br.n(76.0, 0.25, "D2 A2 D3".split(), 0.75)
P11 = [(A6V_IN, A6V_LAND, "D2 A2 E3 F3", "A3 E4 F4"),          # only Quill is given the word: Dm(add9)
       (A6V_LAND, A6V_WRITE, "Bb1 F2 D3 A3", "D4 F4 A4"),       # the pass: Bbmaj7
       (A6V_WRITE, A6V_REAL, "A1 E2 G2 D3", "D4 E4 G4"),        # Ember writes: A7sus4
       (A6V_REAL, A6V_CUT, "F2 C3 G3 A3", "A3 C4 G4"),          # 99.9 %: F(add9), glad
       (A6V_GREY, A6V_LOW, "D2 A2 E3", "A3 E4"),                # the grey partner: cold Dsus2
       (A6V_LOW, A6V_END, "D2 A2 D3", None)]                    # the grey "?": a bare fifth
pads(P11)
K(A6V_IN, 0.6); sub.n(A6V_IN, A6V_LAND - A6V_IN - 0.02, "D2", 0.55)
gl.n(A6V_CARD, 0.5, ["A5", "D6"], 0.3)                           # the gold card
xh.n(A6V_LOCK, 0.1, "RS", 0.5); K(A6V_LOCK, 0.45, body=False); gr.n(A6V_LOCK, 0.2, "D1", 0.5)   # the locked slot
for k, n in enumerate("D6 A5 F5".split()):                       # the card drops into Quill's tail
    gq.n(82.75 + 0.125 * k, 0.15, n, 0.24 - 0.02 * k)
K(A6V_LAND, 0.6); bassnote(A6V_LAND, A6V_WRITE - A6V_LAND - 0.02, "Bb", 0.8)
for k, (t, q, e) in enumerate(zip(A6V_PASS, "D4 F4 A4 C5 D5".split(), "Bb3 D4 F4 A4 Bb4".split())):
    Q.n(t, 0.12, q, 0.5 + 0.04 * k); E.n(t, 0.12, e, 0.5 + 0.04 * k)   # gold climbs and crosses
run(ar, steps(A6V_LAND, A6V_WRITE, 0.125), P11, "Bb2", "D5", 0.32, 0.5)
hats16(A6V_PASS[0], A6V_WRITE, 0.05, 0.04)
E.n(A6V_WRITE, 0.36, "G4", 0.66); K(A6V_WRITE, 0.55); bassnote(A6V_WRITE, 0.36, "A", 0.75)   # Ember writes
arp_t(A6V_WRITE, A6V_REAL, "Asus", 0.125, 0.36)
HIT(A6V_REAL, 0.95, "[BD CP OH]")                                # 99.9 %: glad
br.n(A6V_REAL, 0.5, "F1 C2 F2 A2 C3".split(), 0.8)
booms((A6V_REAL, 0.6)); bassnote(A6V_REAL, A6V_CUT - A6V_REAL - 0.02, "F", 0.95)
Q.n(A6V_REAL, 0.6, "C5", 0.75); E.n(A6V_REAL, 0.6, "A4", 0.75)
gl.n(A6V_REAL, 0.3, ["A5", "C6"], 0.3)
Hh(84.625, 0.1); K(84.75, 0.5, body=False)
# the grey partner: audible
dr.n(A6V_GREY, 0.1, "[BD RS]", 0.8); kb.n(A6V_GREY, 0.3, "A1", 0.7)
gh.n(A6V_GREY, 0.45, ["D4", "A4"], 0.8)                          # its entrance: a sour stab
sub.n(A6V_GREY, A6V_LOW - A6V_GREY - 0.02, "D2", 0.55)
xh.n(A6V_PLUG, 0.1, "RS", 0.6); gl.n(A6V_PLUG, 0.12, "D6", 0.28)  # its link plugs in
for k, (t, n, g_) in enumerate(zip(A6V_GPASS, "A4 D5 E5 A5".split(), "A3 D4 E4 A4".split())):
    gq.n(t, 0.15, n, 0.34 + 0.04 * k); gl.n(t, 0.15, n, 0.26 + 0.03 * k)   # silver crosses (cold, rising)
    gh.n(t, 0.15, g_, 0.6 + 0.05 * k)                            # (doubled by the sour voice)
    Hh(t, 0.1 + 0.03 * k)
dr.n(A6V_LOW, 0.2, "[BD SD CP]", 1.0); kb.n(A6V_LOW, 0.3, "A1", 1.0)   # the grey "?": deflated, full
boom.n(A6V_LOW, 1.5, "D2", 1.0); L_bpi.set(A6V_LOW - 0.05, 0).ramp(A6V_LOW, A6V_LOW + 0.9, -3, "log")
br.n(A6V_LOW, 1.0, "D2 G#2 D3 G#3".split(), 1.0)                # (the first shatter's tritone)
gh.n(A6V_LOW, 1.0, ["G#3", "D4"], 1.0)
gr.n(A6V_LOW, 0.9, "D1", 1.0); sub.n(A6V_LOW, A6V_END - A6V_LOW - 0.05, "D2", 0.9)
E.n(A6V_LOW, 0.75, "D3", 0.7); Q.n(A6V_LOW, 0.75, "G#3", 0.6)
K(87.25, 0.5); K(87.75, 0.4, body=False)                          # the hold: a heartbeat
L_bpi.set(87.9, 0)
# lanes
L_drg.set(A6V_IN, 1.6).set(A6V_GREY - 0.05, 1.8).set(A6V_LOW - 0.05, 2.0)
L_epg.set(A6V_IN, 0.4).set(A6V_REAL, 0.5).set(A6V_GREY, 0.35).set(A6V_LOW, 0.4)
L_qpg.set(A6V_IN, 0.3).set(A6V_REAL, 0.42).set(A6V_GREY, 0.25).set(A6V_LOW, 0.0)
L_epc.set(A6V_IN, 900).ramp(A6V_LAND, A6V_REAL, 2600, "exp").set(A6V_GREY, 700).set(A6V_LOW, 500)
L_qpc.set(A6V_IN, 1800).set(A6V_LAND, 2400).set(A6V_REAL, 3200).set(A6V_GREY, 1400)
L_epr.set(A6V_IN, 1.0)
L_eprl.set(A6V_IN, 1.0).set(A6V_CUT - 0.1, 0.08).set(A6V_GREY - 0.05, 1.0)
L_qprl.set(A6V_IN, 1.2).set(A6V_CUT - 0.1, 0.08).set(A6V_GREY - 0.05, 1.2)
L_rvd.set(A6V_IN, 0.8).set(A6V_GREY, 0.7).set(A6V_LOW, 0.9)
L_rvw.set(A6V_IN, 0.24).set(A6V_CUT, 0.05).set(A6V_GREY - 0.05, 0.2).set(A6V_LOW, 0.28)
lane("quill.echo_send").set(A6V_IN, 0.22).set(A6V_CUT - 0.05, 0.0).set(A6V_GREY - 0.05, 0.16)
lane("ember.echo_send").set(A6V_IN, 0.16).set(A6V_CUT - 0.05, 0.0).set(A6V_GREY - 0.05, 0.12)
L_qg.set(A6V_IN, 1.1); L_eg.set(A6V_IN, 1.2)
L_qc.set(A6V_IN, 1600).set(A6V_REAL, 2000).set(A6V_LOW, 1200)
L_ec.set(A6V_IN, 650).set(A6V_LOW - 0.07, 600)
L_arc.set(A6V_IN, 1200).ramp(A6V_LAND, A6V_REAL, 3000, "exp").set(A6V_REAL + 0.5, 1400)
L_arr.set(A6V_IN, 1.3); L_arg.set(A6V_IN, 0.85)
L_grc.set(A6V_IN, 260).set(A6V_LOW, 360); L_grd.set(A6V_IN, 5.0)
L_ohd.set(A6V_IN, 0.5).set(A6V_REAL - 0.05, 0.7).set(84.9, 0.45)
L_bdd.set(A6V_IN, 0.24).set(A6V_LOW - 0.05, 0.8).set(87.2, 0.24)
lane("ghost.gain", 1.0).set(A6V_GREY - 0.05, 1.6).set(A6V_LOW - 0.05, 2.0)
N11 = False

# ====================================================================== A5 train chart (v9 75.3-80): ADDED
# a5_train v9 (cues/a5_train.md): the kept a5 build is unchanged; the new training chart's S-curve is
# carried by a rising glass figure on the grid under its eased readouts: the lines start at 76.0, a
# slow start (76.625), the take-off (77.0, accent), 77.25, the peak of the rise (77.5, accent), 77.75,
# and a glint on step 900 (78.5).
N9 = True; at(0.0)
for t, n, v in [(76.0, "A4", 0.14), (76.625, "Bb4", 0.16), (77.0, "C#5", 0.24), (77.125, "E5", 0.2),
                (77.25, "G5", 0.26), (77.375, "A5", 0.22), (77.5, "C#6", 0.3), (77.75, "E6", 0.22)]:
    (gq if t in (77.125, 77.375) else gl).n(t, 0.3 if t < 77.5 else 0.5, n, v)
gq.n(78.5, 0.5, ["A5", "E6"], 0.24)
# v10: the seam. a4's last bar (A7) now cuts straight into the frost downbeat (v9 72.0 = v10 68.0, score3's
# 58.0 hit over Gm/D): a short Gm(add9) brass stab and a crash make it a hit of its own
br.n(72.0, 0.4, "G1 D2 G2 Bb2 D3 A3".split(), 0.8)
N9 = False

# ====================================================================== A7 (v9 100-117): RECOMPOSED
# a7_happy (cues/a7_happy.md, v9), v7's a7 music re-laid on the shorter picture. Linked and warm
# (Bbmaj9), Quill's question as soft plucks; LINK CUT 1 (101.0) a dull drop to a bare cold fifth;
# RECONNECT 1 (102.0) on D(add9) with the lantern in major, the meter's ripple, the 62 % sparkle and the
# count hit (102.75), G(add9); the mood question (104.0, Bbmaj9) with a shimmer; LINK CUT 2 (104.75);
# RECONNECT 2 on the beat (105.5, D(add9)) with three gold sparkles as the ratio counts up and lands
# (106.0), the count hit (106.25) and F(add9); the dive (107.5, A7sus4 riser, 16th kicks, a suck-out
# in the white-out) into the impact (108.25, Dm(add9)) and the glassy inner texture of the map (ring-
# bell 8ths over glasswing, 108.5-114): four glass ticks, grey dots (109.25), the warm swell on the
# reconnect (110.0, Bbmaj9), hits on 110.75 and 111.0 (F(add9)), the flip (Gm9, whoosh at 112.0, swell
# from 112.375), the hit on 112.875 (D(add9)); the reverse swell out (114.0), the warm landing on
# Bbmaj9 (114.5), Gm9 for "never trained for it", and the lift on A7sus4 into the title (117).
N9 = True; at(0.0)
inner = track("inner", "patch=glintbell reverb_send=0.5 echo_send=0.3 chorus_send=0.2 pan=-0.15")
A7R = [(A7_IN, A7_CUT1, "Bb1 F2 D3 A3 C4", "D4 F4 C5"),          # linked: Bbmaj9
       (A7_CUT1, A7_LINK1, "D3 A3", None),                        # link cut: a bare cold fifth
       (A7_LINK1, 103.0, "D2 A2 F#3 E4", "F#4 A4 E5"),            # reconnect: D(add9)
       (103.0, A7_ASK, "G1 D2 B2 A3", "B3 D4 A4"),                # G(add9)
       (A7_ASK, A7_CUT2, "Bb1 F2 D3 A3 C4", "D4 F4 C5"),          # the mood question: Bbmaj9
       (A7_CUT2, A7_LINK2, "D3 A3", None),                        # link cut
       (A7_LINK2, 106.5, "D2 A2 F#3 E4", "F#4 A4 E5"),            # reconnect: D(add9)
       (106.5, A7_DIVE, "F2 C3 G3 A3", "C4 G4 A4"),               # F(add9)
       (A7_DIVE, A7_IMPACT, "A1 E2 G2 D3", "D4 E4 G4"),           # the dive: A7sus4
       (A7_IMPACT, A7_LINK3, "D2 A2 E3 F3", "A3 E4 F4"),          # inside the core: Dm(add9)
       (A7_LINK3, A7_HAPPIER, "Bb1 F2 D3 A3 C4", "D4 F4 C5"),     # linked: Bbmaj9
       (A7_HAPPIER, 111.75, "F2 C3 G3 A3", "C4 G4 A4"),           # happier: F(add9)
       (111.75, A7_HAPPIER2, "G1 D2 Bb2 F3 A3", "Bb3 D4 A4"),     # Ember's map: Gm9
       (A7_HAPPIER2, A7_OUT, "D2 A2 F#3 E4", "F#4 A4 E5"),        # happier: D(add9)
       (A7_OUT, A7_LAND, "A1 E2 G2 D3", "D4 E4 G4"),              # out of the map: A7sus4
       (A7_LAND, 115.5, "Bb1 F2 D3 A3 C4", "D4 F4 C5"),           # back to the pair: Bbmaj9
       (115.5, A7_LIFT, "G1 D2 Bb2 F3 A3", "Bb3 D4 A4"),          # "never trained for it": Gm9
       (A7_LIFT, A7_END, "A1 E2 G2 D3", "D4 E4 G4")]              # the lift: A7sus4
pads(A7R)
# --- linked, the question; the cuts
HIT(A7_IN, 0.75, "[BD OH]"); booms((A7_IN, 0.35)); bassnote(A7_IN, 0.98, "Bb", 0.6)
for k, (t, n) in enumerate(zip(A7_WORDS, "F4 A4 C5 D5".split())):   # soft plucks under the words
    Q.n(t, 0.12, n, 0.5 + 0.02 * k)
gq.n(A7_METER, 0.3, "A5", 0.22); E.n(A7_METER + 0.25, 0.22, "A3", 0.45)   # the empty meter, Ember's answer
swell(100.25, A7_CUT1 - 0.02, ["Bb3", "D4", "F4", "C5"], 0.3, 500, 3000)
for c in (A7_CUT1, A7_CUT2):                                       # the cuts: a dull drop, cold
    HIT(c, 1.0, "[BD RS]"); sub.n(c, 0.5, "D2", 0.7)
Q.n(101.5, 0.4, "A4", 0.38); E.n(101.75, 0.25, "D3", 0.4)
E.n(105.125, 0.25, "D3", 0.4)
for c, sw0 in ((A7_LINK1, 101.25), (A7_LINK2, 105.0)):            # the returns
    swell(sw0, c - 0.02, ["D3", "A3", "F#4", "E4"], 0.45, 400, 5000)
    HIT(c, 1.0, "[BD CP OH]")
    br.n(c, 0.5, "D2 A2 D3 F#3 A3 E4".split(), 0.85)
    booms((c, 0.8))
    L_ohd.set(c - 0.05, 0.9).set(c + 0.6, 0.45)
    L_bdd.set(c - 0.05, 0.7).set(c + 0.6, 0.24)
bassnote(A7_LINK1, 0.98, "D", 0.85); bassnote(A7_LINK2, 0.98, "D", 0.85)
for c in (A7_COUNT1, A7_COUNT2):                                   # the counts
    HIT(c, 1.0, "[BD SD CP]")
    br.n(c, 0.24, "D2 A2 D3 F#3 A3".split(), 0.75)
# --- the first return
for t, d, q in [(A7_LINK1, 0.24, "D5"), (102.25, 0.24, "A4"), (102.5, 0.24, "G4"), (A7_COUNT1, 0.72, "B4"),
                (103.5, 0.5, "A4")]:                               # the lantern, in major
    Q.n(t, d, q, 0.8); E.n(t, d, down(q), 0.8)
run(ar, A7_RIPPLE, A7R, "D3", "D4", 0.38, 0.5, zig=False)          # the meter rises
gl.n(A7_POP, 0.4, ["F#5", "A5"], 0.32)                             # 62 % pops
gq.n(A7_COUNT1, 0.35, "D6", 0.28)
K(103.0, 0.8); Hh(103.25, 0.1); K(103.5, 0.6, body=False); Hh(103.75, 0.14, open_=True)
bassnote(103.0, 0.98, "G", 0.85)
arp_t(103.0, A7_ASK, "G6", 0.25, 0.34)
# --- the mood question, the second cut, the second return
K(A7_ASK, 0.7); bassnote(A7_ASK, A7_CUT2 - A7_ASK - 0.02, "Bb", 0.7)
Q.n(A7_ASK, 0.24, "D5", 0.6); Q.n(A7_SHIMMER, 0.24, "C5", 0.55)
for k, n in enumerate("F5 A5 C6 D6".split()):                     # word tiles pop in: shimmer
    (gl if k % 2 == 0 else gq).n(104.375 + 0.125 * k, 0.3, n, 0.16 + 0.012 * (k % 3))
arp_t(A7_ASK, A7_CUT2, "BbH", 0.25, 0.28)
for k, (t, n) in enumerate(zip(A7_GOLD, "F#5 A5 D6".split())):    # the ratio counts up, gold, lands
    (gl if k % 2 == 0 else gq).n(t, 0.3, n, 0.3 + 0.03 * k)
for t, d, q in [(A7_LINK2, 0.72, "D5"), (A7_COUNT2, 0.24, "E5"), (106.5, 0.72, "C5"), (107.25, 0.24, "A4")]:
    Q.n(t, d, q, 0.75); E.n(t, d, down(q), 0.75)
gl.n(A7_COUNT2, 0.35, ["A5", "E6"], 0.28)
K(106.5, 0.8); Hh(106.75, 0.1); xh.n(107.0, 0.1, "RS", 0.4); Hh(107.25, 0.14, open_=True)
bassnote(106.5, 0.98, "F", 0.8)
arp_t(A7_LINK2 + 0.25, 106.5, "D", 0.25, 0.34); arp_t(106.5, A7_DIVE, "F9H", 0.25, 0.32)
# --- the dive, the impact, the inner texture
K(A7_DIVE, 0.85); sub.n(A7_DIVE, A7_IMPACT - A7_DIVE - 0.02, "A1", 0.75)
K(107.75, 0.6, body=False)
for k, t in enumerate((107.875, 108.0)):                           # (108.125: the white-out, a suck-out)
    K(t, 0.7 + 0.08 * k, body=k == 1)
for k, t in enumerate(steps(A7_DIVE, 108.125, 0.125)):
    Hh(t, 0.04 + 0.015 * k)
run(ar, steps(A7_DIVE, 108.125, 0.125), A7R, "E3", "E5", 0.32, 0.55)
swell(A7_DIVE, A7_IMPACT - 0.02, ["A2", "E3", "G3", "D4", "E4"], 0.55, 300, 6000)
HIT(A7_IMPACT, 1.0, "[BD CP OH]"); booms((A7_IMPACT, 1.0))
br.n(A7_IMPACT, 0.6, "D2 A2 D3 E3 F3 A3".split(), 0.9)
sub.n(A7_IMPACT, A7_LINK3 - A7_IMPACT - 0.02, "D2", 0.7); gr.n(A7_IMPACT, 0.75, "D1", 0.9)
Q.n(A7_IMPACT, 0.5, "D5", 0.75); E.n(A7_IMPACT, 0.5, "D4", 0.75)
gl.n(A7_IMPACT, 0.6, ["D5", "A5", "E6"], 0.3)
for k, (t, n) in enumerate(zip(A7_TILES, "E5 F5 A5 D6".split())): # tiles land: soft glass ticks
    (gq if k % 2 == 0 else gl).n(t, 0.12, n, 0.2 + 0.02 * k)
INNER = {"Dm": "A4 D5 E5 F5", "Bb": "Bb4 C5 D5 F5", "F": "A4 C5 F5 G5", "Gm": "Bb4 D5 F5 A5", "D": "A4 D5 E5 F#5"}
for a, b, ch, v in [(108.5, A7_DOTS, "Dm", 0.3), (A7_DOTS, A7_LINK3, "Dm", 0.22), (A7_LINK3, A7_HAPPIER, "Bb", 0.3),
                    (A7_HAPPIER, 111.75, "F", 0.3), (111.75, A7_HAPPIER2, "Gm", 0.3), (A7_HAPPIER2, A7_OUT, "D", 0.32)]:
    ns = INNER[ch].split()
    for k, t in enumerate(steps(a, b, 0.125)):
        if round((t - 108.5) / 0.125) % 2:
            continue                                               # 8ths on the bar's 8th grid
        inner.n(t, 0.24, ns[(0, 2, 1, 3, 0, 3, 2, 1)[round((t - 108.5) / 0.25) % 8]], v + 0.05 * (k % 8 == 0))
for a, b, c in [(A7_IMPACT, A7_LINK3, "D3 A3 E4"), (A7_LINK3, A7_HAPPIER, "D3 F3 C4"), (A7_HAPPIER, 111.75, "C3 G3 A3"),
                (111.75, A7_HAPPIER2, "D3 Bb3 F4"), (A7_HAPPIER2, A7_OUT, "D3 A3 F#4")]:
    st.n(a, b - a - 0.02, c.split(), 0.3)
# --- grey dots, then linked; happier; Ember's map
K(A7_DOTS, 0.8); xh.n(A7_DOTS, 0.1, "RS", 0.3); xh.n(109.75, 0.1, "RS", 0.25)
HIT(A7_LINK3, 0.8, "[BD CP]"); boom.n(A7_LINK3, 1.5, "D2", 0.45)
br.n(A7_LINK3, 0.5, "Bb1 F2 D3 A3 C4".split(), 0.55)
swell(A7_LINK3, A7_LOVING - 0.02, ["Bb3", "D4", "F4", "A4", "C5"], 0.45, 600, 4500)   # the warm swell
bassnote(A7_LINK3, 0.98, "Bb", 0.75)
HIT(A7_LOVING, 0.75, "[BD SD CP]"); gq.n(A7_LOVING, 0.3, "C6", 0.28)
HIT(A7_HAPPIER, 0.95, "[BD CP OH]"); booms((A7_HAPPIER, 0.5))
br.n(A7_HAPPIER, 0.5, "F1 C2 F2 A2 C3 G3".split(), 0.75); bassnote(A7_HAPPIER, 0.73, "F", 0.8)
gl.n(A7_HAPPIER, 0.35, ["A5", "C6"], 0.28)
for t, d, q, e in [(A7_LINK3, 0.7, "D5", "F4"), (A7_LOVING, 0.24, "C5", "A3"), (A7_HAPPIER, 0.72, "A4", "C4")]:
    Q.n(t, d, q, 0.62); E.n(t, d, e, 0.62)
Hh(111.25, 0.1); K(111.5, 0.6)
K(111.75, 0.5, body=False); bassnote(111.75, A7_HAPPIER2 - 111.75 - 0.02, "G", 0.7)
E.n(111.75, 0.5, "Bb3", 0.6); E.n(A7_EMBER, 0.48, "D4", 0.62)
swell(A7_EMBER, A7_HAPPIER2 - 0.02, ["G3", "Bb3", "D4", "A4"], 0.5, 400, 5000)
# --- the last count; out of the map; back to the pair; the lift
HIT(A7_HAPPIER2, 0.95, "[BD CP OH]"); booms((A7_HAPPIER2, 0.6))
br.n(A7_HAPPIER2, 0.6, "D2 A2 D3 F#3 A3 E4".split(), 0.85); bassnote(A7_HAPPIER2, A7_OUT - A7_HAPPIER2 - 0.02, "D", 0.85)
gl.n(A7_HAPPIER2, 0.4, ["F#5", "A5"], 0.3)
Q.n(A7_HAPPIER2, 0.72, "D5", 0.75); E.n(A7_HAPPIER2, 0.72, "F#4", 0.75)
Q.n(113.625, 0.24, "E5", 0.6); E.n(113.625, 0.24, "A4", 0.6)
Hh(113.125, 0.1); K(113.375, 0.6); Hh(113.625, 0.12)
swell(A7_OUT, A7_LAND - 0.02, ["A2", "E3", "G3", "D4"], 0.45, 400, 6000)   # pull back out (reverse swell)
sub.n(A7_OUT, A7_LAND - A7_OUT - 0.02, "A1", 0.5)
K(A7_LAND, 0.8); booms((A7_LAND, 0.45))
br.n(A7_LAND, 0.75, "Bb1 F2 D3 A3 C4".split(), 0.6)               # the warm landing
bassnote(A7_LAND, 0.98, "Bb", 0.75)
for t, d, q, e in [(A7_LAND, 0.24, "D5", "F4"), (A7_CAP1, 0.5, "C5", "A3"), (A7_CAP2, 0.5, "Bb4", "D4"),
                   (A7_LIFT, 0.36, "E4", "A3"), (116.625, 0.36, "G4", "D4")]:
    Q.n(t, d, q, 0.65); E.n(t, d, e, 0.65)
gq.n(A7_CAP1, 0.3, "A5", 0.2)
K(115.5, 0.7); Hh(116.0, 0.1); bassnote(115.5, A7_LIFT - 115.5 - 0.02, "G", 0.7)
K(A7_LIFT, 0.55); sub.n(A7_LIFT, A7_END - A7_LIFT, "A1", 0.6)
hats16(116.5, 116.875, 0.04, 0.04); K(116.875, 0.75, body=False)
arp_t(A7_LAND, 115.5, "BbR", 0.25, 0.34); arp_t(115.5, A7_LIFT, "Gm9", 0.25, 0.34)
arp_t(A7_LIFT, A7_END, "Asus", 0.25, 0.38)
swell(A7_LIFT, A7_END - 0.02, ["A2", "E3", "G3", "D4"], 0.4, 500, 4500)
# --- lanes
L_drg.set(A7_IN, 1.8).set(A7_LINK1 - 0.05, 2.0).set(108.75, 1.6).set(A7_LINK3 - 0.05, 1.8).set(A7_END - 0.1, 2.0)
L_ohd.set(A7_IMPACT - 0.05, 0.95).set(109.25, 0.45).set(A7_HAPPIER - 0.05, 0.7).set(111.6, 0.45) \
    .set(A7_HAPPIER2 - 0.05, 0.8).set(113.5, 0.45)
L_bdd.set(A7_IMPACT - 0.05, 0.8).set(108.85, 0.24)
L_epg.set(A7_IN, 0.4).set(A7_ASK, 0.45).set(A7_DIVE, 0.45).set(A7_IMPACT, 0.4).set(A7_DOTS, 0.3) \
    .set(A7_LINK3, 0.5).set(A7_LAND, 0.45)
L_qpg.set(A7_IN, 0.32).set(A7_ASK, 0.38).set(A7_DIVE, 0.35).set(A7_IMPACT, 0.3).set(A7_DOTS, 0.2) \
    .set(A7_LINK3, 0.4).set(A7_LAND, 0.38)
L_epc.set(A7_IN, 1600).set(A7_ASK, 2000).ramp(A7_DIVE, A7_IMPACT - 0.05, 3200, "exp").set(A7_IMPACT, 1400) \
    .set(A7_DOTS, 700).ramp(A7_LINK3, A7_LOVING, 2600, "exp").set(A7_LAND, 2400).ramp(115.5, A7_END, 2000, "exp")
L_qpc.set(A7_IN, 2400).set(A7_IMPACT, 2600).set(A7_DOTS, 1400).set(A7_LINK3, 3200).set(A7_LAND, 3000)
L_epr.set(A7_IN, 1.0)
L_eprl.set(A7_IN, 1.6); L_qprl.set(A7_IN, 2.4)
L_rvd.set(A7_IN, 0.86); L_rvw.set(A7_IN, 0.26)
L_qg.set(A7_IN, 1.1); L_eg.set(A7_IN, 1.2)
L_qc.set(A7_IN, 1600); L_ec.set(A7_IN, 650)
L_grc.set(A7_IN, 260); L_grd.set(A7_IN, 5.0)
L_arc.set(A7_IN, 1400).set(A7_DIVE, 1400).ramp(A7_DIVE, 108.15, 3200, "exp").set(A7_IMPACT, 1600) \
    .set(A7_LAND, 1600).set(A7_END - 0.1, 2600)
L_arg.set(A7_IN, 0.8); L_arr.set(A7_IN, 1.3)
for c, l in [(A7_CUT1, A7_LINK1), (A7_CUT2, A7_LINK2)]:            # cold and thin, then the flood
    L_eprl.set(c - 0.1, 0.08).set(c + 0.1, 1.6); L_qprl.set(c - 0.1, 0.08).set(c + 0.1, 2.4)
    L_epg.set(c, 0.12).set(l, 0.58).set(l + 0.75, 0.4); L_qpg.set(c, 0.0).set(l, 0.48).set(l + 0.75, 0.32)
    L_epc.set(c, 380).set(l, 2800); L_qpc.set(c, 800).set(l, 3400)
    L_rvd.set(c, 0.5).set(l, 0.9); L_rvw.set(c, 0.1).set(l, 0.3)
    L_qg.set(c, 0.9).set(l, 1.2); L_eg.set(c, 1.0).set(l, 1.35)
    L_qc.set(c, 800).set(l, 2200); L_ec.set(c, 400).set(l, 800)
L_rvd.set(A7_IMPACT - 0.05, 0.95).set(A7_DOTS, 0.9).set(A7_LAND, 0.86)    # the impact's tail, the inner room
L_rvw.set(A7_IMPACT - 0.05, 0.34).set(A7_DOTS, 0.3).set(A7_LAND, 0.26)
L_qc.set(A7_LAND, 2200); L_ec.set(A7_LAND, 800)
lane("quill.echo_send").set(A7_IN, 0.22).set(A7_END - 0.55, 0.0)
lane("ember.echo_send").set(A7_IN, 0.16).set(A7_END - 0.55, 0.0)
lane("inner.gain", 0.9); lane("inner.cutoff", 4200)
lane("strings.gain").set(A7_IMPACT, 0.6)
L_qpc.set(A7_END - 0.1, 3000); L_rvd.set(A7_END - 0.1, 0.86); L_rvw.set(A7_END - 0.1, 0.26)
N9 = False

# ====================================================================== write score11.song
HEAD = f"""# NEURAL PAIRING v11: the score (109 s). GENERATED by make_score11.py: edit that, not this.
#
# D minor, 120 bpm, every cue on the 16th grid (the hit table is at the top of make_score11.py).
# a1 0-12 | a2 12-36 | ab_plate 36-44 | a3 44-54 | a4 54-68 | a5 68-76 | a5c 76-82 |
# a6_proof 82-88 | a7_happy 88-105 | a6 title 105-109

bpm 120
gate 0.97
tail 2

section ASSEMBLY 0..8
section PASS1    8..16
section SHATTER  16..20
section HOLD     20..24
section LINK     24..32
section READ     32..38
section FWDPASS  38..56
section PASS2    56..65
section GENERATE 65..72
section PLATE    72..88
section SECRET   88..98
section WINTER   98..108
section GAMES    108..128
section WALL     128..136
section ICE      136..142
section RISE     142..152
section TALK     152..164
section PROOF    164..176
section HAPPY    176..192
section MAP      192..210
section TITLE    210..218
"""

with open("score11.song", "w") as f:
    f.write(HEAD + "\n")
    for l in lanes:
        if "." not in l.target:
            f.write(l.text() + "\n")
    f.write("\n")
    for t in tracks:
        f.write(t.text() + "\n")
        for l in lanes:
            if l.target.split(".")[0] == t.name and "." in l.target:
                f.write(l.text() + "\n")
        f.write("\n")

# ====================================================================== fx4.song (noise stem)
FX = """# NEURAL PAIRING v11: the FX stem (GENERATED by make_score11.py). Noise-only voices:
# swells, risers into each hit (cut 50 ms before it), crash washes, whooshes, frost, rumble.
bpm 120
gate 1.0
tail 3

automate noise
1
"""
fxt, fxl = [], []


def ft(name, opts):
    t = Track(name, opts); fxt.append(t); return t


def fl(target, v0):
    l = Lane(target, v0); fxl.append(l); return l


air = ft("air", "patch=fx-swell gain=0.5"); air.n(0.0, 4.0, "0", 0.8)
fl("air.attack", 3.0); fl("air.release", 2.5); fl("air.resonance", 0.8)
fl("air.cutoff", 200).ramp(0.0, 4.0, 1400, "exp")
burst = ft("burst", "patch=fx-burst gain=2.0")
BURSTS = [(8.0, 1.0, 2.0), (A2_DROP, 0.6, 2.0), (A2_TOKEN, 0.35, 2.0), (X["hit"], 0.5, 2.0),
          (A2_FOLD, 0.6, 2.0)] + \
         [(w, 0.45, 2.0) for w in A4_WAVES] + [(A4_TAIPEI, 0.3, 0.3), (A4_LAND, 0.5, 2.0)] + \
         [(A3_PUSH, 0.45, 2.0), (BREAK_AT, 0.85, 2.0), (A4_PRESS, 0.45, 2.0), (68.0, 0.45, 2.0), (A5_TOP, 0.8, 2.0),
          (A5B_HIT, 0.7, 2.0),
          (84.0, 1.0, 2.0), (87.0, 0.6, 0.3), (87.75, 0.6, 0.3), (88.5, 0.65, 0.3), (89.25, 1.0, 2.0),
          (92.25, 0.3, 0.3)]
for t, v, d in BURSTS:
    burst.n(t, d, "0", v)
fl("burst.cutoff", 8000)
rise = ft("rise", "patch=fx-swell gain=2.0")
gl_ = fl("rise.gain", 0.03); cl_ = fl("rise.cutoff", 300)
RISERS = [(6.8, 7.95, 1.0, 7000), (10.0, 11.95, 0.9, 9000), (20.25, A2_TOKEN - 0.05, 0.45, 5000),
          (A2_UNFOLD, 15.95, 0.3, 4000), (29.75, X["hit"] - 0.05, 0.6, 8000), (35.5, A2_FOLD - 0.05, 0.7, 10000), (A3_PASS1, A3_WRITE - 0.05, 0.3, 2500),
          (45.5, A3_PUSH - 0.05, 0.4, 6000),
          (A4_FOLD, A4_LAND - 0.05, 0.6, 8000), (A4_FLY, A4_SPARK, 0.7, 9000), (71.0, A5_TOP - 0.05, 0.9, 10000),
          (A5B_TICKS[2], A5B_RUN[0] - 0.05, 0.3, 4000), (A5B_JOINT, A5B_HIT - 0.05, 0.8, 11000),
          (83.0, 83.95, 0.9, 9000), (88.75, 89.2, 0.7, 7000), (91.0, 91.95, 0.8, 9000)]
# (v7: the last one now stops 50 ms before a7's downbeat, as every riser does; v6 ran it into the cut)
for a, b, top, fc in RISERS:
    rise.n(a, b - a, "0", 0.8)
    gl_.set(a - 0.02, 0.03).ramp(a, b, top, "exp")
    cl_.set(a - 0.02, 300).ramp(a, b, fc, "exp")
wh = ft("whoosh", "patch=fx-swell gain=1.6")
WH = (X["arrive"], A4_WAVES[0], A4_EXITS[0] + 0.25, A4_EXITS[1] + 0.25, A5B_LAND, A5B_JOINT)
# (the close-up, into the games, waves A and B fly out, the a5b swishes)
for peak in WH:
    wh.n(peak - 0.5, 0.5, "0", 0.8)
fl("whoosh.attack", 0.42); fl("whoosh.release", 0.9); fl("whoosh.resonance", 2.0)
wl = fl("whoosh.cutoff", 400)
for peak in WH:
    wl.ramp(peak - 0.5, peak, 4000, "exp").ramp(peak, peak + 1.0, 400, "exp")
ice = ft("ice", "patch=fx-ice gain=1.0"); ice.n(68.0, 3.0, "0", 0.8); fl("ice.lfo_filter", 1.2)
ice.n(A3_SNOW, A3_CUT - A3_SNOW - 0.02, "0", 0.45)            # a3's snowfall, stopped dead at the cut
# the break (v5 time, cues/ab_plate.md): dusty room tone, the nib and pen flicks, the ruler
# ratchet and bracket runs, the compass scrape, stipple, paper settle / slide / fold, the wet-ink
# swell with a reversed cymbal into the press, and the re-entry riser
RAW = True
room = ft("room", "patch=fx-low gain=0.1"); room.n(BRK_INK, BRK_END - 0.05 - BRK_INK, "0", 0.6)
pen = ft("pen", "patch=fx-burst gain=0.5")
PEN = [(BRK_INK, 0.5)] + [(t_, 0.38) for t_ in BRK_FLICKS] + \
      [(BRK_RATCHET[0] + 0.125 * k_, 0.3 + 0.05 * (k_ % 2)) for k_ in range(4)] + \
      [(BRK_BRACKET[0] + 0.125 * k_, 0.3) for k_ in range(3)] + [(t_, 0.32) for t_ in BRK_STIPPLE] + \
      [(BRK_FOLD, 0.4), (BRK_FOLD + 0.125, 0.32)]
for t_, v_ in sorted(set(PEN)):
    pen.n(t_, 0.05, "0", v_)
fl("pen.cutoff", 6500); fl("pen.release", 0.04)
paper = ft("paper", "patch=fx-swell gain=1.2")
for t_, d_ in [(BRK_PRESS, 0.3), (BRK_SLIDE, 0.6), (BRK_FOLD, 0.5)]:
    paper.n(t_, d_, "0", 0.7)
fl("paper.attack", 0.04); fl("paper.release", 0.25); fl("paper.resonance", 0.4); fl("paper.cutoff", 2600)
compass = ft("compass", "patch=fx-swell gain=0.9"); compass.n(BRK_COMPASS, 0.5, "0", 0.6)
fl("compass.attack", 0.02); fl("compass.release", 0.15); fl("compass.resonance", 1.6)
fl("compass.cutoff", 3200).ramp(BRK_COMPASS, BRK_COMPASS + 0.5, 4800, "lin")
ink = ft("ink", "patch=fx-swell gain=0.8"); ink.n(BRK_INK, BRK_PRESS - 0.03 - BRK_INK, "0", 0.6)
fl("ink.attack", 0.5); fl("ink.release", 0.05); fl("ink.cutoff", 400).ramp(BRK_INK, BRK_PRESS - 0.03, 2500, "exp")
for a_, b_, top_, fc_ in [(40.375, BRK_PRESS - 0.03, 0.5, 12000), (BRK_RISE, BRK_END - 0.125, 0.9, 11000)]:   # (ends a 16th early: a suck-out)
    rise.n(a_, b_ - a_, "0", 0.8)
    gl_.set(a_ - 0.02, 0.03).ramp(a_, b_, top_, "exp")
    cl_.set(a_ - 0.02, 300).ramp(a_, b_, fc_, "exp")
RAW = False; N9 = True                                              # v9 recomposed sections (final v9 times)
for t_, v_, d_ in [(A2G_FOLD, 0.6, 2.0), (A5B9_HIT, 0.7, 2.0), (A5C9_TALK[0], 0.35, 0.3), (72.0, 0.45, 0.6), (A7_LINK1, 0.9, 0.9), (A7_LINK2, 0.85, 0.9), (A7_IMPACT, 1.0, 2.0),
                   (A7_HAPPIER2, 0.4, 0.3)]:
    burst.n(t_, d_, "0", v_)
for a_, b_, top_, fc_ in [(33.5, A2G_FOLD - 0.05, 0.7, 10000), (A5B9_TICKS[2], A5B9_RUN[0] - 0.05, 0.3, 4000),
                          (A5B9_JOINT, A5B9_HIT - 0.05, 0.8, 11000), (87.75, A5C9_AFTER - 0.05, 0.7, 7000),
                          (101.25, A7_LINK1 - 0.05, 0.7, 10000), (105.0, A7_LINK2 - 0.05, 0.6, 9000),
                          (A7_DIVE, A7_IMPACT - 0.05, 1.0, 12000), (A7_OUT, A7_LAND - 0.05, 0.6, 9000),
                          (A7_LIFT, A7_END - 0.05, 0.6, 9000)]:
    rise.n(a_, b_ - a_, "0", 0.8)
    gl_.set(a_ - 0.02, 0.03).ramp(a_, b_, top_, "exp")
    cl_.set(a_ - 0.02, 300).ramp(a_, b_, fc_, "exp")
room.n(A7_CUT1, 0.95, "0", 0.6); room.n(A7_CUT2, 0.7, "0", 0.6)

for peak in (A5B9_LAND, A5B9_JOINT, A7_FLIP):                        # swishes (the land, the slide), the flip
    wh.n(peak - 0.5, 0.5, "0", 0.8)
    wl.ramp(peak - 0.5, peak, 4000, "exp").ramp(peak, peak + 0.75, 400, "exp")
N9 = False
N11 = True                                                          # v11 a6_proof (final v11 times)
for t_, v_, d_ in [(A6V_REAL, 0.5, 0.3), (A6V_GREY, 0.45, 0.3), (A6V_LOW, 1.0, 1.5)]:
    burst.n(t_, d_, "0", v_)
for a_, b_, top_, fc_ in [(A6V_PASS[0], A6V_WRITE - 0.05, 0.35, 5000)]:
    rise.n(a_, b_ - a_, "0", 0.8)
    gl_.set(a_ - 0.02, 0.03).ramp(a_, b_, top_, "exp")
    cl_.set(a_ - 0.02, 300).ramp(a_, b_, fc_, "exp")
wh2 = ft("whoosh2", "patch=fx-swell gain=1.4")                      # Quill slides out, the grey one in
wh2.n(A6V_OUT - 0.25, 0.5, "0", 0.8)
fl("whoosh2.attack", 0.15); fl("whoosh2.release", 0.3); fl("whoosh2.resonance", 2.0)
fl("whoosh2.cutoff", 400).ramp(A6V_OUT - 0.25, A6V_OUT, 4000, "exp").ramp(A6V_OUT, A6V_OUT + 0.125, 1500, "exp") \
    .ramp(A6V_OUT + 0.125, A6V_GREY + 0.125, 4000, "exp").ramp(A6V_GREY + 0.125, A6V_GREY + 0.5, 400, "exp")
room.n(A6V_CUT, 0.45, "0", 0.6)
N11 = False; RAW = True
burst.n(A5C_TALK[0], 0.3, "0", 0.35)
RAW = False
low = ft("low", "patch=fx-low gain=0.12"); low.n(80.0, 3.9, "0", 0.8)

with open("fx11.song", "w") as f:
    f.write(FX + "\n")
    for t in fxt:
        f.write(t.text() + "\n")
        for l in fxl:
            if l.target.split(".")[0] == t.name:
                f.write(l.text() + "\n")
        f.write("\n")

# sanity: every note-on on the 16th grid
off = [(t.name, e[0]) for t in tracks for e in t.ev if abs(e[0] * 8 - round(e[0] * 8)) > 1e-6]
assert not off, f"off-grid note-ons: {off[:10]}"
print("wrote score11.song and fx11.song")
