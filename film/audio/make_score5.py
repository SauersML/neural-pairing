#!/usr/bin/env python3
"""Generate score5.song and fx5.song for NEURAL PAIRING v5 (104 s).

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
VOICE_WAV = "neural_pairing_voice_v5.wav"     # score3's take, -2 semitones and +6.3 % (formant-preserving
                                              # TD-PSOLA, the extra time only in steady parts: the "neu"
                                              # vowel, the P closure, the "-ing" tail; transitions untouched):
                                              # qa/voice/psola.py --semitones -2 --rate 1.0 --map 0:0,0.2:0.2,
                                              # 0.4:0.42,0.47:0.49,0.573:0.625,0.82:0.872,1.04:1.112
                                              # (a uniform +8 % stretch read "Nor old pairing" on base.en in
                                              # 3 of 4 perturbed renders; this one passes 4/4)
# --- v5: the vibe-break (ab_plate), FINAL v5 times, cues/ab_plate.md
BREAK_AT, BREAK_LEN = 40.0, 8.0               # v4 times >= 40 move 8 s later
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


def W(t):
    """v4 time -> v5 time"""
    return t + BREAK_LEN if t >= BREAK_AT - 1e-9 else t


LENGTH = 104.0

# ====================================================================== helpers
OFF = 0.0          # time offset added to every event (for the kept, shifted sections)
RAW = False        # True while writing the break (final v5 times, no warp)


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
        t = out_t(t0)
        if not RAW and t0 < BREAK_AT - 1e-9 < t0 + dur - 0.02 and self.name not in ("fx",):
            raise ValueError(f"{self.name}: a note at {t0} sustains across the break")
        if isinstance(notes, (list, tuple)):
            notes = "[" + " ".join(notes) + "]"
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
        t = out_t(t + OFF)
        self.pts.append((t, f">{B(t)} {v:g}:0.01"))
        return self

    def ramp(self, t0, t1, v, shape="lin"):
        t0, t1 = t0 + OFF, t1 + OFF
        if not RAW and t0 < BREAK_AT - 1e-9 < t1 + 2e-9 and t1 - t0 <= 0.1 + 1e-9:
            t0, t1 = t0 + BREAK_LEN, t1 + BREAK_LEN    # a pickup into a hit on 40 (v4) moves with the hit
        elif not RAW and t0 < BREAK_AT - 1e-9 < t1 - 1e-9:
            raise ValueError(f"{self.target}: a ramp {t0}-{t1} crosses the break")
        else:
            t0, t1 = out_t(t0), out_t(t1)
        self.pts.append((t0, f">{B(t0)} {v:g}:{round(2 * (t1 - t0), 5):g}@{shape}"))
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
fx = track("fx", "sample=fx5.wav root=C4 gain=1.0 reverb_send=0.2")
RAW = True; fx.n(0.0, LENGTH, "C4", 0.8); RAW = False
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
vx = track("voice", f"vox wav={VOICE_WAV} wav_at={B(W(VOICE_NEURAL))} reverb_send=0 gain={VOICE_GAIN}")

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
vx.n(78.5, _vp - 78.5, ["C3", "G3", "D4", "E4"], 0.9)
vx.n(_vp, 80.0 - _vp, ["A2", "E3", "G3", "C4"], 0.9)
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

# ====================================================================== write score5.song
HEAD = f"""# NEURAL PAIRING v5: the score (104 s). GENERATED by make_score5.py: edit that, not this.
#
# D minor, 120 bpm, every cue on the 16th grid (the hit table is at the top of make_score5.py).
# a1 0-12 | a2 12-40 | ab_plate 40-48 (break) | a3 48-58 | a4 58-72 | a5 72-84 | a5b 84-88 | a6 88-104

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
section GENERATE 65..80
section PLATE    80..96
section SECRET   96..106
section WINTER   106..116
section GAMES    116..136
section WALL     136..144
section TRAIN    144..152
section ICE      152..158
section RISE     158..168
section TEACHER  168..176
section DROP     176..184
section PROOF    184..190
section CLIMAX   190..200.5
section TITLE    200.5..208
"""

with open("score5.song", "w") as f:
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
FX = """# NEURAL PAIRING v5: the FX stem (GENERATED by make_score5.py). Noise-only voices:
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
          (83.0, 83.95, 0.9, 9000), (88.75, 89.2, 0.7, 7000), (91.0, 92.2, 0.8, 9000)]
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
RAW = False
low = ft("low", "patch=fx-low gain=0.12"); low.n(80.0, 3.9, "0", 0.8)

with open("fx5.song", "w") as f:
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
print("wrote score5.song and fx5.song")
