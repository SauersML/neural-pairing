#!/usr/bin/env python3
"""Generate score3.song and fx3.song for NEURAL PAIRING (82 s).

120 bpm throughout (1 beat = 0.5 s, 16th = 0.125 s). Every cue on the director's
grid-locked cue sheet sits on the 16th grid:
  bridges 12.0 + 0.5k; slow-pass exchanges 15.875 + 0.875k (every 7 sixteenths,
  a 7:4 cross-rhythm against the half-time groove); token 23.0; fast passes from
  23.5 every 0.25; games: three waves at 42.0 / 45.0 / 48.0 (slams +0, +0.25, +0.5;
  links ignite +1.0; answer typing in 16ths from +1.5; checks +2.5 / +2.625 / +2.75),
  pull-back 51.0, title 51.5, fold 52.5, stack pulse 53.0, fly-off 53.25-53.9;
  result windows 73.0 / 73.75 / 74.5 / 75.25 (every 1.5 beats; the 4th is the
  strongest); title 78.25, "neural" 78.5, "pairing" 79.0, shared hop 79.5.

Every event is written with an absolute seek (`>beat`), computed from seconds.

Run: python3 make_score3.py   (writes score3.song and fx3.song)
"""

# ------------------------------------------------------------------ helpers
NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def B(t):
    """seconds -> song beats (120 bpm), as compact text"""
    return f"{round(2 * t, 5):g}"


def grid(t0, bpm):
    """local-beat -> seconds for a section starting at t0 with tempo bpm"""
    return lambda b: t0 + b * 60.0 / bpm


class Track:
    def __init__(self, name, opts):
        self.name, self.opts, self.ev = name, opts, []

    def n(self, t, dur, notes, vel):
        """a note or chord at t seconds, held dur seconds"""
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
    """automation: set(t, v) jumps (5 ms ramp), ramp(t0, t1, v, shape) glides"""

    def __init__(self, target, v0):
        self.target, self.v0, self.pts = target, v0, []

    def set(self, t, v):
        self.pts.append((t, f">{B(t)} {v:g}:0.01"))
        return self

    def ramp(self, t0, t1, v, shape="lin"):
        self.pts.append((t0, f">{B(t0)} {v:g}:{round(2 * (t1 - t0), 5):g}@{shape}"))
        return self

    def text(self):
        body = " ".join(p[1] for p in sorted(self.pts, key=lambda p: p[0]))
        return f"automate {self.target}\n{self.v0:g} {body}".rstrip()


tracks, lanes = [], []


def track(name, opts=""):
    t = Track(name, opts)
    tracks.append(t)
    return t


def lane(target, v0):
    l = Lane(target, v0)
    lanes.append(l)
    return l


# ------------------------------------------------------------------ grid
g = grid(0.0, 120)         # beat -> seconds (one tempo for the whole film)
EX = [15.875 + 0.875 * k for k in range(8)]       # the 8 exchanges
WAVES = (42.0, 45.0, 48.0)                         # the games wall
RES = (73.0, 73.75, 74.5, 75.25)                   # the result windows

# ------------------------------------------------------------------ harmony
# (start, end, Ember's pad = wide saws (low), Quill's pad = PWM strings (mid))
H = [
    (0.0, 4.0, "D2 A2 B2 F3", "D4 F4 Bb4"),      # two dialects: Ember dorian Dm6, Quill aeolian Bb/D
    (4.0, 8.0, "D2 A2 F3", "D4 F4 A4"),
    (8.0, 10.0, "D2 A2 D3", None),
    (10.0, 12.0, "D2 A2 E3 F3", "A3 E4"),         # the hold: Dm(add9)
    (12.0, 18.0, "D2 A2 F3 C4", "E4 A4"),         # Dm9 (bridges and the first exchanges)
    (18.0, 20.0, "Bb1 F2 D3 A3", "D4 F4"),        # Bbmaj7
    (20.0, 22.0, "G1 D2 Bb2 F3", "A3 D4"),        # Gm9
    (22.0, 23.0, "A1 E2 G2 C#3", "E4 G4"),        # A7(b13 in the melody)
    (23.0, 24.0, "D2 A2 F3 C4", "E4 A4"),
    (24.0, 25.0, "Bb1 F2 D3 A3", "D4 F4"),
    (25.0, 25.5, "G1 D2 Bb2 F3", "A3 D4"),
    (25.5, 26.0, "A1 E2 G2 C#3", "C#4 E4"),
    (26.0, 29.0, "Bb1 F2 D3 A3 E4", "D4 E4 A4"),  # the read: Bbmaj7(#11)
    (29.0, 32.0, "G1 D2 Bb2 A3", "Bb3 D4 A4"),    # Gm(add9)
    (32.0, 36.0, "D2 A2 E3", "F4 A4"),            # the secret: Dm(add9) pedal
    (36.0, 37.0, "Bb1 F2 D3", None),
    (37.0, 39.5, "D2 A2 F3", "D4 F#4 A4"),        # Quill turns major first (37.5; its pad at 37.0)
    (39.5, 40.5, "D2 A2 F#3", "D4 F#4 A4"),       # then Ember (39.5)
    (40.5, 42.0, "G1 D2 B2", "D4 G4 B4"),         # G major: dorian IV, the bloom
    (42.0, 45.0, "D2 A2 F3 C4", "E4 A4"),         # games wave 1: Dm9 (D dorian vamp)
    (45.0, 48.0, "G1 D2 B2 E3", "D4 B4"),         # wave 2: G6/9
    (48.0, 51.0, "Bb1 F2 D3 A3", "D4 F4 A4"),     # wave 3: Bbmaj7
    (51.0, 52.5, "A1 E2 G2 D3", "D4 E4 G4"),      # pull-back, title: A7sus4
    (52.5, 54.0, "A1 E2 G2 C#3", "C#4 E4 G4"),    # fold, stack pulse, fly-off: A7
    (54.0, 58.0, "D2 A2 F3 C4", "A3 D4 F4"),      # training
    (58.0, 61.0, "D2 G2 Bb2 A3", "Bb3 D4 A4"),    # ice: frozen Gm(add9)/D
    (61.0, 63.0, "Bb1 F2 D3 A3", "D4 F4 A4"),     # riser
    (63.0, 66.0, "A1 E2 C#3 G3", "C#4 E4 G4"),
    (70.0, 72.0, "D2 A2 F3 C4", "D4 F4 A4"),      # proof
    (72.0, 73.0, "Bb1 F2 D3 A3", "D4 F4 A4"),
    (73.0, 73.75, "Bb1 F2 D3 F3", "D4 F4 Bb4"),   # results: Bb - C - A - Dm (VI VII V i)
    (73.75, 74.5, "C2 G2 C3 E3", "E4 G4 C5"),
    (74.5, 75.25, "A1 E2 C#3 G3", "C#4 E4 A4"),
    (75.25, 76.0, "D2 A2 F3 C4", "D4 F4 A4"),     # the 4th (strongest): Dm9, the arrival
    (76.0, 77.5, "G1 D2 Bb2 F3", "Bb3 D4 A4"),
    (77.5, 78.25, "A1 E2 C#3 G3", "C#4 E4 G4"),
    (78.25, 79.0, "D2 A2 E3 F3", "A3 E4 F4"),     # title: Dm(add9) - Bbmaj7 ("pairing") - Dsus4(add9)
    (79.0, 80.0, "Bb1 F2 D3 A3", "D4 F4 A4"),
    (80.0, 82.0, "D2 A2 E3 G3", "A3 D4 E4"),
]

# ------------------------------------------------------------------ the bus
lane("reverb_decay", 0.86).set(26.0, 0.62).set(32.0, 0.86).set(37.0, 0.9).set(54.0, 0.66) \
    .set(58.0, 0.93).set(61.0, 0.8).set(70.0, 0.86).set(78.3, 0.55).set(79.85, 0.86)
lane("reverb_wet", 0.26).ramp(65.95, 66.0, 0.0).ramp(66.5, 67.5, 0.2, "smooth").ramp(69.9, 70.0, 0.26) \
    .ramp(78.3, 78.4, 0.08).ramp(79.85, 80.5, 0.26, "smooth")   # dry under the spoken title
lane("reverb_tone", 4200)
lane("reverb_pre", 0.03)
lane("echo_time", 0.5)                                # a quarter note, everywhere
lane("echo_feedback", 0.32).set(78.3, 0.0).set(79.85, 0.32)   # QA: clear the echo loop under the words
lane("echo_tone", 2600)
lane("echo_pingpong", 0.7)
lane("chorus_mode", 2)
lane("chorus_mix", 0)
lane("chorus_depth", 0.5)
lane("spring", 0.0)
lane("tape_drive", 0.42)
lane("tape_wow", 0.0)
lane("tape_flutter", 0.004)
lane("tape_age", 0.1)
lane("width", 1.25)
lane("lfo_rate", 0.12)
lane("comp_in", 1)
lane("comp_threshold", -18)
lane("comp_ratio", 1)
lane("comp_attack", 3)
lane("comp_release", 4)
lane("comp_makeup", 3)
lane("comp_mix", 0.35)
lane("comp_sc_hpf", 60)
# the 909 board, tuned to the key under THESE drive settings (QA A/B renders, qa/ab3/drum*.song):
# kick tail on A1 (55 Hz, the same note as the sine kick body), snare body on F3 (175 Hz),
# rim knock on A3. (bd_tune 0.20 / sd_tune 0.17 read G1 +46 c / F3 -50 c with this drive.)
for p, v in [("bd_level", 1.0), ("bd_tune", 0.32), ("bd_attack", 0.7), ("bd_sweep", 0.4),
             ("bd_drive", 0.75), ("sd_level", 0.9), ("sd_tune", 0.32), ("sd_tone", 0.35), ("sd_snappy", 0.6),
             ("sd_decay", 0.55), ("rs_level", 0.5), ("rs_tune", 0.52), ("cp_level", 0.9), ("cp_decay", 0.6),
             ("hh_level", 0.45), ("hh_tune", 0.4), ("hh_metal", 0.5), ("ch_decay", 0.14),
             ("dr_drive", 0.9), ("dr_tone", 0.6)]:
    lane(p, v)

# ------------------------------------------------------------------ FX stem
fx = track("fx", "sample=fx3.wav root=C4 gain=1.0 reverb_send=0.2")
fx.n(0.0, 82.0, "C4", 0.8)

# ------------------------------------------------------------------ drums
dr = track("drums", "kit=909 comp=-28 gain=2.0 spring_send=0.12 reverb_send=0.08")
kb = track("kickbody", "patch=s3-kickbody gain=1.1")
lane("kickbody.pitch_shift", -0.17)   # QA: the short sine body read +17 c sharp


def K(t, v=0.95, body=True):
    dr.n(t, 0.1, "BD", v)
    if body:
        kb.n(t, 0.3, "A1", min(1.0, v))


def S(t, v=0.95, clap=True):
    dr.n(t, 0.1, "[SD CP]" if clap else "SD", v)


def Hh(t, v=0.12, open_=False):
    dr.n(t, 0.1, "OH" if open_ else "CH", v)


def half_bar(g, b0, fills=True):
    """half-time bar on grid g starting at local beat b0: kick 1, snare 3, ghost hats on the 8ths"""
    K(g(b0)); S(g(b0 + 2))
    for e, v in [(0.5, 0.1), (1.0, 0.07), (1.5, 0.12), (2.5, 0.1), (3.5, 0.14)]:
        Hh(g(b0 + e), v)
    if fills:
        K(g(b0 + 3.0), 0.55, body=False)


def full_bar(g, b0, four=False, oh=True):
    """full-time bar: kick 1 & 3 (+ the 3-and), snare+clap 2 & 4, 8th hats; four=True: four on the floor"""
    for k in range(4):
        if four or k in (0, 2):
            K(g(b0 + k))
        if k in (1, 3):
            S(g(b0 + k))
        Hh(g(b0 + k + 0.5), 0.2 if not oh else 0.16, open_=oh and k == 3)
    if not four:
        K(g(b0 + 2.5), 0.6, body=False)


# 4-7: heartbeat under the first forward pass
for t in (4.0, 5.0, 6.0):
    K(t, 0.42)
# 8.0 shatter
dr.n(8.0, 0.2, "[BD SD CP OH]", 1.0); kb.n(8.0, 0.3, "A1", 1.0)
# 12.0-23.0 the drop, the bridges and the slow pass: half time, kick 1, snare+clap 3
for b0 in (24, 28, 32, 36, 40):
    half_bar(g, b0, fills=b0 in (24, 32))
K(22.0); Hh(22.25, 0.1); Hh(22.5, 0.12); Hh(22.75, 0.14)
dr.n(12.0, 0.2, "OH", 0.5)
# the 8 exchanges (every 7 sixteenths): a rim knock rides each, the 7:4 against the groove
xh = track("xhit", "kit=909")
for t in EX:
    xh.n(t, 0.1, "RS", 0.6)
# 23.0 token; fast passes from 23.5 on the 8ths: four on the floor from 24, 16th ghost hats
dr.n(23.0, 0.2, "[BD CP OH]", 1.0); kb.n(23.0, 0.3, "A1", 1.0)
K(23.5, 0.7)
full_bar(g, 48, four=True, oh=False)          # 24-26
for k in range(16):
    Hh(24.0 + 0.125 * k, 0.06 + 0.04 * (k % 2))
# 26-32 the read: tight, cold: kick on the bar, precise rim ticks
for bar in range(3):
    t0 = 26.0 + 2 * bar
    K(t0, 0.8)
    for e in range(8):
        dr.n(t0 + 0.25 * e, 0.1, "RS", 0.22 if e % 2 else 0.32)
# 32-37 secret: heartbeat
for t in (32.0, 34.0):
    K(t, 0.55)
dr.n(36.5, 0.2, "[BD CP]", 0.8); kb.n(36.5, 0.3, "A1", 0.8)
# 37-42 mood: the groove rebuilds
K(37.5, 0.7)
half_bar(g, 76); half_bar(g, 80)               # 38-40, 40-42
Hh(41.75, 0.3, open_=True)
# 42-51 games wall, three waves: three window slams, the links ignite, the answer types
# in 16ths, three checks
for w in WAVES:
    dr.n(w, 0.2, "[BD CP OH]", 1.0); kb.n(w, 0.3, "A1", 1.0)        # slam 1
    K(w + 0.25, 0.8)                                                   # slam 2
    dr.n(w + 0.5, 0.2, "[BD SD]", 0.9); kb.n(w + 0.5, 0.3, "A1", 0.9)  # slam 3
    dr.n(w + 1.0, 0.2, "[BD CP OH]", 0.95); kb.n(w + 1.0, 0.3, "A1", 0.95)   # ignite
    K(w + 1.5, 0.85); S(w + 2.0, 0.9)
    for k in range(8):                                                 # typing: 16th hats
        Hh(w + 1.5 + 0.125 * k, 0.08 + 0.06 * (k % 2 == 0))
    for k, v in enumerate((0.5, 0.55, 0.62)):                          # checks
        dr.n(w + 2.5 + 0.125 * k, 0.1, "RS", v)
# 51.0 pull-back, 51.5 title, 52.5 fold, 53.0 stack pulse, 53.25-53.9 fly-off
K(51.0, 0.85); dr.n(51.0, 0.2, "OH", 0.35)
dr.n(51.5, 0.2, "[BD CP]", 0.75); kb.n(51.5, 0.3, "A1", 0.75)
K(52.5, 0.6); Hh(52.75, 0.12)
dr.n(53.0, 0.2, "[BD CP OH]", 0.95); kb.n(53.0, 0.3, "A1", 0.95)
# 54-58 training: the machine, 16th hats
for bar in range(2):
    t0 = 54.0 + 2 * bar
    full_bar(g, 2 * t0, four=False, oh=False)
    for k in range(16):
        Hh(t0 + 0.125 * k, 0.05 + 0.05 * (k % 4 == 2))
# 58-61 ice
K(58.0, 0.9); dr.n(58.0, 0.2, "OH", 0.4); K(60.0, 0.6)
# 61-66 riser: kicks quarters -> eighths -> sixteenths, then cut dead at 66.0
for k in range(6):
    K(61.0 + 0.5 * k, 0.6 + 0.04 * k)
for k in range(4):
    K(64.0 + 0.25 * k, 0.8, body=False)
for k in range(6):
    K(65.0 + 0.125 * k, 0.85, body=False)
dr.n(65.75, 0.1, "CP", 0.9)
# 70.0 BIG: four on the floor
dr.n(70.0, 0.2, "[BD SD CP OH]", 1.0); kb.n(70.0, 0.3, "A1", 1.0)
for k in range(1, 6):
    K(70.0 + 0.5 * k)
    if k % 2:
        S(70.0 + 0.5 * k)
    if k < 5:
        Hh(70.0 + 0.5 * k + 0.25, 0.22, open_=True)
Hh(70.25, 0.22, open_=True)
# 73.0 / 73.75 / 74.5 / 75.25 result windows: the hits stand alone, a hat between
for k, t in enumerate(RES):
    dr.n(t, 0.2, "[BD SD CP OH]", 0.85 if k < 3 else 1.0); kb.n(t, 0.3, "A1", 0.85 if k < 3 else 1.0)
    if k < 3:
        Hh(t + 0.375, 0.16)
# 75.25-78.25 the climax groove: four on the floor
for t in (75.75, 76.0, 76.5, 77.0, 77.5):
    K(t)
for t in (76.5, 77.5):
    S(t)
for t in (75.5, 76.25, 76.75, 77.25, 77.75):
    Hh(t, 0.24, open_=True)
K(77.875, 0.8, body=False)                       # pickup, then air before the title
# 78.25 title; 79.5 the shared hop lands; 80.0 the last chord
dr.n(78.25, 0.3, "[BD OH]", 0.95); kb.n(78.25, 0.3, "A1", 0.95)
K(79.5, 0.45); K(80.0, 0.5)

lane("drums.gain", 2.0).set(4.0, 0.8).set(8.0, 2.0).set(26.0, 1.4).set(32.0, 1.0).set(36.5, 1.6) \
    .set(37.5, 1.8).set(42.0, 2.0)
lane("oh_decay", 0.45).set(7.95, 0.95).set(9.0, 0.45).set(69.95, 0.95).set(71.0, 0.5).set(72.95, 0.6) \
    .set(73.5, 0.5).set(75.2, 0.95).set(76.0, 0.5).set(78.2, 0.5)
lane("bd_decay", 0.24).set(7.95, 0.8).set(9.0, 0.24).set(69.95, 0.75).set(71.0, 0.24).set(75.2, 0.75) \
    .set(75.7, 0.24).set(78.2, 0.7).set(79.0, 0.24)

# ------------------------------------------------------------------ boom (deliberate pitch-drop impacts)
boom = track("boom", "patch=boom gain=1.0")
BOOMS = [(8.0, 1.0), (12.0, 0.85), (23.0, 0.55), (42.0, 0.6), (51.0, 0.45), (53.0, 0.6), (58.0, 0.6),
         (70.0, 1.0), (73.0, 0.6), (75.25, 1.0), (78.25, 0.75)]
for t, v in BOOMS:
    boom.n(t, 1.5, "D2", v)
bl = lane("boom.pitch_shift", 0)
for t, _ in BOOMS:
    bl.ramp(t - 0.05, t, 12).ramp(t, t + 0.6, 0, "log")

# ------------------------------------------------------------------ bass: sub + growl
sub = track("sub", "patch=sub2 duck=0.2 duck_release=0.12")
gr = track("growl", "patch=s3-growl duck=0.2 duck_release=0.1 reverb_send=0.04")
ROOT = {"D": "D", "Bb": "Bb", "G": "G", "A": "A", "C": "C"}


def bassnote(t, dur, root, v=0.85, growl=True, octave=1, subv=0.8):
    sub.n(t, dur, f"{root}{octave + 1}", subv)
    if growl:
        gr.n(t, dur, f"{root}{octave}", v)


# 0-12: sub drone swelling from silence
sub.n(0.0, 8.0, "D2", 0.55); sub.n(8.0, 2.0, "D2", 0.95); sub.n(10.0, 2.0, "D2", 0.6)
gr.n(8.0, 1.6, "D1", 1.0)


def half_bass(g, b0, root, octave=1):
    # root on 1 (held), a push on the 2-and, octave on 3-and, root on 4
    bassnote(g(b0), g(b0 + 1.5) - g(b0), root, 0.95, octave=octave)
    bassnote(g(b0 + 1.5), g(b0 + 2.5) - g(b0 + 1.5), root, 0.6, octave=octave)
    gr.n(g(b0 + 2.5), g(b0 + 3) - g(b0 + 2.5), f"{root}{octave + 1}", 0.7)
    sub.n(g(b0 + 2.5), g(b0 + 3) - g(b0 + 2.5), f"{root}{octave + 1}", 0.7)
    bassnote(g(b0 + 3), g(b0 + 4) - g(b0 + 3), root, 0.8, octave=octave)


for b0, r in [(24, "D"), (28, "D"), (32, "D"), (36, "Bb"), (40, "G")]:
    half_bass(g, b0, r)
bassnote(22.0, 1.0, "A", 0.9)
# 23-26 fast passes: driving 8ths
seq = [("D", 23.0, 24.0), ("Bb", 24.0, 25.0), ("G", 25.0, 25.5), ("A", 25.5, 26.0)]
for r, a, b in seq:
    t = a
    while t < b - 1e-6:
        bassnote(t, 0.25, r, 0.75 + 0.2 * ((t - a) < 0.01))
        t += 0.25
# 26-32 read: sub only, long
sub.n(26.0, 3.0, "Bb1", 0.7); sub.n(29.0, 3.0, "G1", 0.7)
# 32-37 secret: sub pedal
sub.n(32.0, 4.0, "D2", 0.55); sub.n(36.0, 1.0, "Bb1", 0.55)
# 37-42 mood
sub.n(37.0, 1.0, "D2", 0.7)
half_bass(g, 76, "D"); half_bass(g, 80, "D")
for k in range(4):
    pass
# 42-51 games: slams on the root, then Drive 8ths (root / octave) under the typing
for w, r in zip(WAVES, ("D", "G", "Bb")):
    for dt, d, v in ((0.0, 0.2, 1.0), (0.25, 0.2, 0.8), (0.5, 0.4, 0.95)):
        bassnote(w + dt, d, r, v)
    for k in range(6):
        t = w + 1.0 + 0.25 * k
        oc = 2 if k % 4 == 3 else 1
        gr.n(t, 0.24, f"{r}{oc}", 0.95 if k % 2 == 0 else 0.62)
        if k % 2 == 0:
            sub.n(t, 0.5, f"{r}2" if r == "D" else f"{r}1", 0.8)
# 51-54 pull-back / title / fold / pulse / fly-off on A
sub.n(51.0, 1.5, "A1", 0.75); gr.n(51.0, 0.5, "A1", 0.8)
for k in range(2):
    bassnote(52.5 + 0.25 * k, 0.24, "A", 0.75)
bassnote(53.0, 0.25, "A", 1.0)
# 54-58 training: growl 8ths, root pedal, accents on the beat
for k in range(16):
    t = 54.0 + 0.25 * k
    bassnote(t, 0.25, "D", 0.9 if k % 2 == 0 else 0.55, growl=True, subv=0.75 if k % 2 == 0 else 0.0001)
# 58-61 ice: sub only
sub.n(58.0, 3.0, "G1", 0.75)
# 61-66 riser: 8ths, opening
for k in range(20):
    t = 61.0 + 0.25 * k
    r = "Bb" if t < 63.0 else "A"
    bassnote(t, 0.25, r, 0.7 + 0.015 * k)
# 70-73 proof: 8ths with octave pops
for r, a, b in [("D", 70.0, 72.0), ("Bb", 72.0, 72.75)]:   # a breath before 73.0
    t = a; k = 0
    while t < b - 1e-6:
        bassnote(t, 0.25, r, 1.0 if k % 2 == 0 else 0.7, octave=2 if k % 4 == 3 else 1)
        t += 0.25; k += 1
# 73.0-75.25 result windows: short notes, so each hit stands out of air; then 8ths
for t, r in zip(RES[:3], ("Bb", "C", "A")):
    bassnote(t, 0.36, r, 1.0)
bassnote(75.25, 0.75, "D", 1.0)
for r, a, b in [("G", 76.0, 77.5), ("A", 77.5, 78.0)]:
    t = a; k = 0
    while t < b - 1e-6:
        bassnote(t, 0.24, r, 1.0 if k % 2 == 0 else 0.72, octave=2 if k % 4 == 3 else 1)
        t += 0.25; k += 1
# title
bassnote(78.25, 0.25, "D", 0.95); sub.n(78.5, 0.5, "D2", 0.55); sub.n(79.0, 1.0, "Bb1", 0.55); bassnote(80.0, 2.0, "D", 0.7)
lane("sub.gain", 0).ramp(0.0, 4.0, 0.6, "smooth").set(8.0, 1.0).set(10.0, 0.7).set(12.0, 1.0)
lane("growl.cutoff", 170).set(26.0, 140).set(37.0, 170).set(42.0, 220).set(54.0, 150) \
    .ramp(61.0, 66.0, 600, "exp").set(70.0, 260).ramp(73.0, 78.25, 380, "exp").set(78.25, 200)
lane("growl.drive", 4.0).set(70.0, 5.0).set(78.25, 4.0)
lane("growl.gain", 1.8)            # QA: +5 dB makes up the lower sustain below
# QA: repeated legato notes at sustain 0.85 retriggered from near full level, so the
# onsets smeared (+40..+85 ms, IQR 40 ms); a lower sustain, a faster decay and a 1 ms
# attack give every note a clear attack (A/B qa/ab3/growl_A_att1: +7.5 ms, IQR 6 ms)
for p_, v_ in [("sustain", 0.35), ("decay", 0.22), ("attack", 0.001), ("filter_sustain", 0.1),
               ("filter_decay", 0.3)]:
    lane(f"growl.{p_}", v_)
lane("growl.pitch_shift", 0.08)   # QA: the growl envelope sweep reads ~5-15 c flat
lane("sub.glide", 0)              # no slides between bass notes

# ------------------------------------------------------------------ pads
ep = track("epad", "patch=s3-pad reverb_send=0.38 chorus_send=0.35 duck=0.4 duck_release=0.25 pan=0.12")
qp = track("qpad", "patch=s3-glasspad reverb_send=0.45 chorus_send=0.4 duck=0.3 duck_release=0.25 pan=-0.12")
for a, b, e, q in H:
    ep.n(a, b - a, e.split(), 0.5)
    if q:
        qp.n(a, b - a, q.split(), 0.45)
lane("epad.gain", 0).ramp(0.0, 3.5, 0.32, "smooth").set(8.0, 0.4).set(12.0, 0.45).set(26.0, 0.3) \
    .set(32.0, 0.25).set(37.0, 0.42).set(42.0, 0.45).set(54.0, 0.32).set(58.0, 0.4).ramp(61.0, 66.0, 0.55, "exp") \
    .set(70.0, 0.55).set(73.0, 0.3).set(75.25, 0.6).set(78.25, 0.4).set(78.4, 0.12).set(79.85, 0.45)
lane("qpad.gain", 0).ramp(0.0, 3.5, 0.28, "smooth").set(12.0, 0.36).set(26.0, 0.26).set(32.0, 0.22) \
    .set(37.0, 0.4).set(42.0, 0.34).set(54.0, 0.26).set(58.0, 0.38).ramp(61.0, 66.0, 0.45, "exp") \
    .set(70.0, 0.45).set(73.0, 0.3).set(75.25, 0.5).set(78.25, 0.36).set(78.4, 0.1).set(79.85, 0.4)
# the sound evolves with the story: dark swell, open at the link, cold in the read
# and training, bloom at the mood, frozen at the ice, wide open at the proof
lane("epad.cutoff", 250).ramp(0.0, 4.0, 900, "exp").set(10.0, 700).ramp(12.0, 23.0, 1500, "exp") \
    .set(26.0, 550).ramp(37.0, 42.0, 2200, "exp").set(42.0, 1400).set(54.0, 600).set(58.0, 900) \
    .ramp(61.0, 66.0, 3200, "exp").set(70.0, 2400).ramp(73.0, 78.25, 3200, "exp").ramp(78.25, 82.0, 1500, "smooth")
lane("epad.resonance", 0.9).ramp(12.0, 23.0, 1.2).set(26.0, 0.8).ramp(61.0, 66.0, 1.3).set(70.0, 1.0)
lane("qpad.cutoff", 900).ramp(0.0, 4.0, 1700, "exp").set(26.0, 1200).ramp(37.0, 42.0, 3000, "exp") \
    .set(54.0, 1300).set(58.0, 2600).ramp(61.0, 66.0, 3500, "exp").set(70.0, 3000)
lane("epad.release", 1.6).ramp(65.8, 65.9, 0.08).set(70.0, 1.0).set(73.0, 0.6).set(78.25, 2.0)
lane("qpad.release", 2.4).ramp(65.8, 65.9, 0.08).set(70.0, 1.2).set(73.0, 0.7).set(78.25, 2.4)

# reverse-like swells into the drop and the big hit (and into the shatter)
sw = track("swell", "patch=s3-pad reverb_send=0.4 chorus_send=0.3")
sw.n(10.25, 1.73, ["D3", "A3", "E4", "F4"], 0.6)
sw.n(68.75, 1.23, ["A2", "E3", "A3", "C#4", "G4"], 0.6)
sw.n(6.75, 1.23, ["D3", "G#3", "D4"], 0.5)
sw.n(74.75, 0.48, ["D3", "A3", "D4", "F4"], 0.6)
sl = lane("swell.gain", 0.0)
for a, b in [(6.75, 7.98), (10.25, 11.98), (68.75, 69.98), (74.75, 75.23)]:
    sl.set(a, 0.02).ramp(a + 0.02, b, 0.9, "exp").set(b + 0.01, 0.0)
lane("swell.cutoff", 400).ramp(6.75, 7.98, 3000, "exp").set(10.25, 400).ramp(10.27, 11.98, 4000, "exp") \
    .set(68.75, 400).ramp(68.77, 69.98, 5000, "exp") \
    .set(74.75, 600).ramp(74.77, 75.23, 5000, "exp")
lane("swell.release", 0.05)

# ------------------------------------------------------------------ QUILL (glass PWM, mid, aeolian)
Q = track("quill", "patch=s3-quill reverb_send=0.35 echo_send=0.22 chorus_send=0.2 pan=-0.25")
E = track("ember", "patch=s3-ember reverb_send=0.28 echo_send=0.16 chorus_send=0.12 pan=0.25")
# assembly: two motifs in two modes, on the 8th grid
for t, n, d in [(0.5, "D4", 0.5), (1.0, "F4", 0.5), (1.5, "A4", 0.5), (2.0, "Bb4", 1.0), (3.0, "A4", 1.0)]:
    Q.n(t, d, n, 0.5)
for t, n, d in [(2.0, "D3", 0.5), (2.5, "A3", 0.5), (3.0, "B3", 0.5), (3.5, "C4", 0.25), (3.75, "A3", 0.25)]:
    E.n(t, d, n, 0.5)
# 4-7 the first pass in lockstep, each in its own mode; tokens at 7.0
for k, (q, e) in enumerate(zip("D4 F4 G4 A4 Bb4 C5".split(), "D3 F3 G3 A3 B3 C4".split())):
    Q.n(4.0 + 0.5 * k, 0.5, q, 0.42 + 0.03 * k)
    E.n(4.0 + 0.5 * k, 0.5, e, 0.42 + 0.03 * k)
Q.n(7.0, 0.5, "D5", 0.62); E.n(7.0, 0.5, "D4", 0.62)
# 10-12 the hold: their first fifth
E.n(10.0, 2.0, "A3", 0.45); Q.n(10.5, 1.5, "E4", 0.42)
# 12.0-15.5 bridges: alternating voices stack one Dm9 chord (each held to the first exchange)
for k, n in enumerate("D3 F3 A3 C4 D4 F4 A4 C5".split()):
    t = 12.0 + 0.5 * k
    (E if k % 2 == 0 else Q).n(t, EX[0] - t, n, 0.82)
# 15.875-22.0 slow pass: the zig-zag, one note per exchange, every 7 sixteenths
for k, n in enumerate("F3 G4 A3 Bb4 C4 D5 E4 F5".split()):
    (E if k % 2 == 0 else Q).n(EX[k], 0.85 if k < 7 else 0.95, n, 0.85)
# 23.0 token: the same D4 together
Q.n(23.0, 0.5, "D4", 0.9); E.n(23.0, 0.5, "D4", 0.9)
# 23.5-26 fast passes: one pass per 8th, the line alternating Ember / Quill
for k, n in enumerate("A3 E4 Bb2 D4 F3 A4 G2 Bb3 A2 C#4".split()):
    (E if k % 2 == 0 else Q).n(23.5 + 0.25 * k, 0.24, n, 0.72)
# 26-32 the read (cold): Ember's query glides up (deliberate glide), Quill's keys,
# a gate slides shut, the values pour into Ember
E.n(26.5, 0.25, "D3", 0.6); E.n(26.75, 1.25, "A4", 0.62)
for k, n in enumerate("G4 A4 Bb4 C5 D5".split()):
    Q.n(28.0 + 0.25 * k, 0.2, n, 0.5)
for k, n in enumerate("D4 C4 Bb3 A3".split()):
    E.n(30.0 + 0.25 * k, 0.25, n, 0.55)
E.n(31.0, 1.0, "G3", 0.55)
lane("ember.mono", 0).set(26.45, 1).set(28.0, 0)
lane("ember.glide", 0).set(26.6, 0.3).set(28.0, 0)
# 32-37 the secret: Quill's lantern motif (twice), Ember plays it back exactly at 36.5
for t0, last in [(32.0, 1.25), (34.0, 0.75)]:
    for k, n in enumerate("D4 A4 G4".split()):
        Q.n(t0 + 0.25 * k, 0.25, n, 0.75)
    Q.n(t0 + 0.75, last, "Bb4", 0.78)
for k, n in enumerate("D4 A4 G4".split()):
    E.n(36.5 + 0.25 * k, 0.25, n, 0.75)
E.n(37.25, 0.75, "Bb4", 0.78)
# 37-42 the mood: Quill's third rises F -> F# (37.5), then Ember's (39.5)
for t, n, d, v in [(37.0, "A4", 0.25, 0.55), (37.25, "F4", 0.25, 0.55), (37.5, "F#4", 1.0, 0.8),
                   (38.5, "A4", 0.5, 0.6), (39.0, "D5", 1.5, 0.66), (40.5, "B4", 1.5, 0.62)]:
    Q.n(t, d, n, v)
for t, n, d, v in [(39.0, "A3", 0.25, 0.55), (39.25, "F3", 0.25, 0.55), (39.5, "F#3", 1.0, 0.82),
                   (40.5, "B3", 1.5, 0.62)]:
    E.n(t, d, n, v)
# 42-51 the games wall: at each ignite (+1.0) both voices sound together (a fifth);
# the answer types in 16ths (+1.5..+2.4) in the answering model's voice; three checks
IGN = {42.0: ("D4", "A3"), 45.0: ("B4", "E4"), 48.0: ("A4", "D4")}
TYPE = {42.0: (E, "D3 F3 A3 C4 E4 C4 A3 F3"), 45.0: (Q, "G4 A4 B4 D5 E5 D5 B4 A4"),
        48.0: (None, "Bb2 D4 F3 F4 A3 A4 D4 C5")}
for w in WAVES:
    q, e = IGN[w]
    Q.n(w + 1.0, 0.5, q, 0.7); E.n(w + 1.0, 0.5, e, 0.72)
    who, line = TYPE[w]
    for k, n in enumerate(line.split()):
        v = who if who else (E if k % 2 == 0 else Q)
        v.n(w + 1.5 + 0.125 * k, 0.12, n, 0.6)
# 51.0 pull-back: a held fifth; 51.5 the title; 53.0 the stack pulse
Q.n(51.0, 1.5, "E4", 0.5); E.n(51.0, 1.5, "A3", 0.5)
Q.n(53.0, 0.25, "C#5", 0.62); E.n(53.0, 0.25, "A3", 0.62)
# 54-61 training: they trade single notes
for t, v, n, d in [(55.0, Q, "A4", 1.0), (56.0, E, "A3", 1.0), (57.0, Q, "F4", 1.0), (57.5, E, "F3", 0.5),
                   (59.0, Q, "A4", 2.0), (59.0, E, "D4", 2.0)]:
    v.n(t, d, n, 0.55)
# 61-66 riser: climbing in octaves on A7 (63-66)
for k, (q, e) in enumerate(zip("A3 C#4 E4 G4 A4 C#5".split(), "A2 C#3 E3 G3 A3 C#4".split())):
    Q.n(63.0 + 0.5 * k, 0.48, q, 0.62 + 0.04 * k)
    E.n(63.0 + 0.5 * k, 0.48, e, 0.62 + 0.04 * k)
# 66.5 the real partner answers with the lantern, exactly
for k, n in enumerate("D4 A4 G4".split()):
    E.n(66.5 + 0.25 * k, 0.25, n, 0.5)
E.n(67.25, 0.7, "Bb4", 0.52)
# 70.0 the proof: the theme in octaves (Quill above Ember)
theme = [(70.0, 0.5, "D4"), (70.5, 0.5, "A4"), (71.0, 0.5, "G4"), (71.5, 0.5, "Bb4"), (72.0, 0.7, "A4"),
         (73.0, 0.5, "F4"), (73.75, 0.5, "G4"), (74.5, 0.5, "A4"),
         (75.25, 0.5, "D5"), (75.75, 0.25, "C5"), (76.0, 0.75, "Bb4"), (76.75, 0.25, "A4"), (77.0, 0.5, "G4"),
         (77.5, 0.75, "C#5"), (78.25, 0.25, "D5")]


def down(n):
    return n[:-1] + str(int(n[-1]) - 1)


for t, d, n in theme:
    Q.n(t, d * 0.97, n, 0.88)
    E.n(t, d * 0.97, down(n), 0.88)
Q.n(80.0, 2.0, "D4", 0.75); E.n(80.0, 2.0, "D4", 0.75)       # the final unison
lane("quill.gain", 0.7).set(12.0, 1.1).set(26.0, 0.9).set(32.0, 0.8).set(37.0, 1.1).set(70.0, 1.2)
lane("ember.gain", 0.65).set(12.0, 1.25).set(26.0, 1.0).set(32.0, 0.95).set(37.0, 1.25).set(66.0, 0.9).set(70.0, 1.35)
lane("quill.cutoff", 1300).set(26.0, 1000).set(32.0, 1400).ramp(37.0, 42.0, 2000, "exp").set(54.0, 1100) \
    .set(70.0, 2000)
lane("ember.cutoff", 520).set(26.0, 420).set(32.0, 560).ramp(37.0, 42.0, 800, "exp").set(54.0, 450) \
    .set(70.0, 750)

# gate closing in the read (a deliberate glass slide)
sg = track("sig", "patch=glasslead reverb_send=0.4 echo_send=0.2 pan=-0.35 gain=1.2")
sg.n(29.5, 0.125, "A4", 0.45); sg.n(29.625, 0.6, "D4", 0.4)
lane("sig.mono", 1)
lane("sig.glide", 0.2)

# ------------------------------------------------------------------ arp / ostinato
ar = track("arp", "patch=s3-arp reverb_send=0.15 echo_send=0.15 duck=0.35 duck_release=0.2 pan=0.08")
ARP = {"Dm9": "D2 A2 C3 F3 E3 C3 A2 F2", "Bb": "Bb1 F2 A2 D3 C3 A2 F2 D2", "Gm9": "G1 D2 F2 Bb2 A2 F2 D2 Bb1",
       "A7": "A1 E2 G2 C#3 E3 C#3 G2 E2", "G6": "G1 D2 E2 B2 A2 E2 D2 B1", "D": "D2 D3 A2 D3 F3 D3 A2 D3",
       "Dice": "D2 D3 G2 D3 Bb2 D3 G2 D3", "BbR": "Bb1 Bb2 F2 Bb2 D3 Bb2 F2 Bb2", "AR": "A1 A2 E2 A2 C#3 A2 E2 A2"}


def arp(g, b0, b1, chord, step, vel=0.5):
    notes = ARP[chord].split(); k = 0; b = b0
    while b < b1 - 1e-6:
        ar.n(g(b), (g(b + step) - g(b)) * 0.95, notes[k % len(notes)], vel + (0.12 if k % 4 == 0 else 0))
        b += step; k += 1


for b0, c in [(32, "Dm9"), (36, "Bb"), (40, "Gm9")]:                # slow pass: 8ths
    arp(g, b0, b0 + 4, c, 0.5, 0.42)
arp(g, 44, 46, "A7", 0.5, 0.42)
arp(g, 48, 52, "Bb", 0.25, 0.36)                                   # fast passes: 16ths under the 8th line
for k, n in enumerate("D2 F2 A2 D3 F3 A3 D4".split()):          # 34.5-36.25 the gold rising
    ar.n(34.5 + 0.25 * k, 0.24, n, 0.35 + 0.04 * k)
for w, c in zip(WAVES, ("Dm9", "G6", "Bb")):                       # games: 8ths from the ignite
    arp(g, 2 * w + 2, 2 * w + 5, c, 0.5, 0.42)
arp(g, 108, 116, "D", 0.25, 0.42)                                  # training: 16ths
arp(g, 116, 122, "Dice", 0.25, 0.38)                               # ice
arp(g, 122, 126, "BbR", 0.25, 0.42); arp(g, 126, 132, "AR", 0.25, 0.45)   # riser
arp(g, 140, 144, "Dm9", 0.5, 0.5); arp(g, 144, 145.5, "Bb", 0.5, 0.5)     # proof 70-72.75
arp(g, 151, 152, "Dm9", 0.5, 0.5); arp(g, 152, 155, "Gm9", 0.5, 0.5); arp(g, 155, 156, "A7", 0.5, 0.5)
lane("arp.cutoff", 600).set(15.8, 450).ramp(15.9, 23.0, 1400, "exp").set(34.5, 500).set(42.0, 700) \
    .ramp(42.0, 54.0, 1600, "exp").set(54.0, 560).ramp(58.0, 61.0, 500, "exp").ramp(61.0, 66.0, 3200, "exp") \
    .set(70.0, 1500).ramp(73.0, 78.25, 2600, "exp")
lane("arp.resonance", 1.3).ramp(54.0, 66.0, 1.6).set(70.0, 1.3)
lane("arp.gain", 0.8).ramp(61.0, 66.0, 1.2, "exp").set(70.0, 0.9)

# ------------------------------------------------------------------ brass impacts
br = track("brass", "patch=s3-brass reverb_send=0.25 duck=0.0")
for t, c, d, v in [(8.0, "D2 G#2 D3 G#3", 1.5, 1.0), (12.0, "D2 A2 D3 F3", 1.5, 0.85), (23.0, "D3 A3 D4", 0.5, 0.65),
                   (70.0, "D2 A2 D3 F3 A3", 2.0, 1.0), (73.0, "Bb1 F2 Bb2 D3 F3", 0.4, 0.85),
                   (73.75, "C2 G2 C3 E3 G3", 0.4, 0.85), (74.5, "A1 E2 A2 C#3 E3", 0.4, 0.85),
                   (75.25, "D2 A2 D3 F3 A3 D4", 1.2, 1.0), (78.25, "D2 A2 E3 F3 A3", 0.25, 0.85),
                   (51.5, "A1 E2 G2 D3 E3", 0.5, 0.7), (53.0, "A1 E2 A2 C#3 E3", 0.25, 0.8)] + \
                  [(w + dt, c, 0.15 if dt < 1 else 0.5, v) for w, c in zip(WAVES, ("D2 A2 D3 F3", "G1 D2 G2 B2", "Bb1 F2 Bb2 D3"))
                   for dt, v in ((0.0, 0.85), (0.25, 0.7), (0.5, 0.8), (1.0, 0.9))]:
    br.n(t, d, c.split(), v)

# ------------------------------------------------------------------ glass (shards, read ticks, frost)
gl = track("glass", "patch=s3-glass reverb_send=0.4 echo_send=0.15 pan=0.25")
for k, n in enumerate("C6 F#5 A5 D#5 G5 C#5 E5 A#4 D5 G#4 C5".split()):   # 8.0 shards (deliberately atonal)
    gl.n(8.0 + 0.125 * k, 0.15, n, 0.55 - 0.035 * k)
for bar in range(3):                                                       # read: precise 8th glass
    t0 = 26.0 + 2 * bar
    ns = [("A4", "E5"), ("A4", "D5"), ("Bb4", "D5")][bar]
    for k in range(8):
        if k in (0, 3, 6):
            gl.n(t0 + 0.25 * k, 0.2, ns[k % 2], 0.3)
gl.n(58.0, 2.0, ["D5", "A5"], 0.45)                                        # 58 frost
for k, n in enumerate("E6 A5 D6 G5 Bb5 A5".split()):
    gl.n(58.5 + 0.25 * k, 0.4, n, 0.24 - 0.02 * k)
for w in WAVES:                                                            # the three checks
    for k, n in enumerate(("A4", "D5", "F5")):
        gl.n(w + 2.5 + 0.125 * k, 0.3, n, 0.42 + 0.05 * k)
gl.n(79.5, 0.6, ["D5", "A5"], 0.3)                                         # 79.5 the hop lands
lane("glass.gain", 1.0)

# ------------------------------------------------------------------ the stranger (deliberately sour)
gh = track("ghost", "patch=sour pan=-0.2 reverb_send=0.3 gain=1.1")
for t, n, d in [(68.0, "G#3", 0.75), (68.75, "D#4", 0.25), (69.0, "C#4", 0.25), (69.25, "E4", 0.7)]:
    gh.n(t, d, n, 0.6)

# ------------------------------------------------------------------ the voice: "neural pairing" ("neural" 78.5, "pairing" 79.0)
# The QA agent's verified route: the modulator is a recording of the phrase (piper
# en_US-ryan, length 1.35, copied to neural_pairing_voice.wav); Patina's studio vocoder
# (vox_mode 1) speaks it through a saw carrier playing the title harmony (Dm(add9) on
# "neural", Bbmaj7 on "pairing"), with the room it needs: no reverb send, the reverb
# dipped to 0.08, gain 2.0, vox_level 1.45, nothing on 79.0, melody and pads 6 dB down
# from 78.4 to 79.85. vox_dry 0.7 (raw formant voice under the vocoder) keeps the
# consonants: Whisper hears "Neural pairing." in the mix window. wav_at pins the
# recording to the first carrier note (beat 157 = 78.5 s).
lane("vox_mode", 1)
lane("vox_level", 1.45)
lane("vox_dry", 0.85)   # QA: 0.7 passed only by luck (fails under inaudible perturbations)
vx = track("voice", "vox wav=neural_pairing_voice.wav wav_at=157 reverb_send=0 gain=2.0")
vx.n(78.5, 0.5, ["D3", "A3", "E4", "F4"], 0.9)
vx.n(79.0, 1.0, ["Bb2", "F3", "A3", "D4"], 0.9)
for p, val in [("waveform", 2), ("osc2_pitch", -12), ("osc2_level", 0.5), ("detune", 4), ("cutoff", 1600),
               ("resonance", 0.4), ("sub", 0.3)]:
    lane(f"voice.{p}", val)

# ------------------------------------------------------------------ QA: a robust room for the voice
# The phrase passed Whisper only by luck: an inaudible velocity-0.01 note earlier in the score
# (which reshuffles Patina's voice-card assignment; the score peaks at 64/64 cards) turned it
# into "You're all pairing". Tested over 4 such perturbations (qa/ab3/voice_robust.py), this
# room passes 4/4 on small.en and base.en:
#  - free voice cards so the vocoder carrier is never short of cards: the pads' long releases
#    stop at the title, and the ducked (inaudible) pad chords at 79.0 are not played (the voice's
#    own carrier plays the Bbmaj7)
#  - the title hit (78.25) clears by 78.42: shorter open hat and kick rumble, its brass, melody,
#    boom, sub and growl tails dipped; Quill/Ember stop feeding the quarter-note echo from 77.45
#    (its repeats landed exactly on "neural" and "pairing")
lane("quill.echo_send", 0.22).set(77.45, 0.0).set(79.85, 0.22)
lane("ember.echo_send", 0.16).set(77.45, 0.0).set(79.85, 0.16)
[l for l in lanes if l.target == "sub.gain"][0].set(78.42, 0.3).set(79.85, 1.0)
[l for l in lanes if l.target == "quill.gain"][0].set(78.42, 0.15).set(79.85, 1.2)
[l for l in lanes if l.target == "ember.gain"][0].set(78.42, 0.15).set(79.85, 1.35)
[l for l in lanes if l.target == "growl.gain"][0].set(78.42, 0.3).set(79.85, 1.8)
lane("boom.gain", 1.0).set(78.42, 0.15).set(79.85, 1.0)
lane("brass.gain", 1.0).set(78.42, 0.1).set(79.85, 1.0)
lane("brass.release", 0.7).set(78.2, 0.1).set(79.9, 0.7)
for _name, _short, _back in (("bd_decay", 0.4, None), ("oh_decay", 0.2, None),
                             ("epad.release", 0.15, 2.0), ("qpad.release", 0.15, 2.4)):
    _l = [l for l in lanes if l.target == _name][0]
    _l.pts = [p for p in _l.pts if not (78.19 < p[0] < 78.26)]
    _l.set(78.2, _short)
    if _back:
        _l.set(79.9, _back)
ep.ev = [e for e in ep.ev if abs(e[0] - 79.0) > 1e-6]
qp.ev = [e for e in qp.ev if abs(e[0] - 79.0) > 1e-6]

# ------------------------------------------------------------------ write score3.song
HEAD = """# NEURAL PAIRING v3: the score (82 s). GENERATED by make_score3.py: edit that, not this.
#
# D minor, 120 bpm, every cue on the 16th grid (see make_score3.py for the cue sheet).
# Every event is placed with an absolute seek computed from seconds.
#
# Theory as story (see NOTES3.md): Quill (glass PWM voice, aeolian, B-flat) and Ember
# (driven saw voice, dorian, B-natural) speak apart, then hocket once linked; the slow
# pass zig-zags one note per exchange (every 7 sixteenths, a 7:4 against the groove);
# both land on D at the token; Quill's lantern motif D-A-G-Bb is played back by Ember;
# Picardy thirds spread from Quill to Ember; the stranger answers a tritone away; the
# proof puts both voices in octaves, the title in unison.

bpm 120
gate 0.97
tail 2

# sections (song beats = 2 x seconds), for reports; cue times live in make_score3.py
section ASSEMBLY 0..8
section PASS1    8..16
section SHATTER  16..20
section HOLD     20..24
section BRIDGES  24..31.75
section SLOWPASS 31.75..46
section FAST     46..52
section READ     52..64
section SECRET   64..74
section MOOD     74..84
section GAMES    84..108
section TRAIN    108..116
section ICE      116..122
section RISE     122..132
section DROP     132..140
section PROOF    140..146
section CLIMAX   146..156.5
section TITLE    156.5..164
"""

with open("score3.song", "w") as f:
    f.write(HEAD + "\n")
    # bus lanes first (no track prefix), then each track followed by its lanes
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

# ------------------------------------------------------------------ fx3.song (noise stem)
FX = """# NEURAL PAIRING v3: the FX stem (GENERATED by make_score3.py). Noise-only voices:
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
for t, v in [(8.0, 1.0), (12.0, 0.6), (23.0, 0.35), (42.0, 0.4), (43.0, 0.3), (45.0, 0.4), (46.0, 0.3),
             (48.0, 0.4), (49.0, 0.3), (53.0, 0.45), (58.0, 0.45), (70.0, 1.0), (73.0, 0.6), (73.75, 0.6),
             (74.5, 0.65), (75.25, 1.0), (78.25, 0.3)]:
    burst.n(t, 0.3 if t in (43.0, 46.0, 49.0, 73.0, 73.75, 74.5, 78.25) else 2.0, "0", v)
fl("burst.cutoff", 8000)
rise = ft("rise", "patch=fx-swell gain=2.0")
gl_ = fl("rise.gain", 0.03); cl_ = fl("rise.cutoff", 300)
for a, b, top, fc in [(6.8, 7.95, 1.0, 7000), (10.0, 11.95, 0.9, 9000), (22.1, 22.95, 0.45, 5000),
                      (34.5, 36.45, 0.3, 2500), (53.25, 53.9, 0.7, 9000), (61.0, 65.95, 1.0, 12000),
                      (69.0, 69.95, 0.9, 9000), (74.75, 75.2, 0.7, 7000), (77.0, 78.2, 0.8, 9000)]:
    rise.n(a, b - a, "0", 0.8)
    gl_.set(a - 0.02, 0.03).ramp(a, b, top, "exp")
    cl_.set(a - 0.02, 300).ramp(a, b, fc, "exp")
wh = ft("whoosh", "patch=fx-swell gain=1.6")
WH = (26.0, 42.0, 51.0, 52.5)                 # into the read, into the games, pull-back, fold
for peak in WH:
    wh.n(peak - 0.5, 0.5, "0", 0.8)
fl("whoosh.attack", 0.42); fl("whoosh.release", 0.9); fl("whoosh.resonance", 2.0)
wl = fl("whoosh.cutoff", 400)
for peak in WH:
    wl.ramp(peak - 0.5, peak, 4000, "exp").ramp(peak, peak + 1.0, 400, "exp")
ice = ft("ice", "patch=fx-ice gain=1.0"); ice.n(58.0, 3.0, "0", 0.8); fl("ice.lfo_filter", 1.2)
low = ft("low", "patch=fx-low gain=0.12"); low.n(66.0, 3.9, "0", 0.8)

with open("fx3.song", "w") as f:
    f.write(FX + "\n")
    for t in fxt:
        f.write(t.text() + "\n")
        for l in fxl:
            if l.target.split(".")[0] == t.name:
                f.write(l.text() + "\n")
        f.write("\n")
print("wrote score3.song and fx3.song")
