#!/usr/bin/env python3
"""Apply the QA fixes to a copy of score2.song -> qa/fixes/score2_fixed.song.
Every edit is an exact, asserted text replacement, so the diff is the fix list.
    python3 qa/fixes/make_fixed.py && diff score2.song qa/fixes/score2_fixed.song
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "..", "..", "score2.song")
s = open(src).read()

def rep(old, new, why):
    global s
    assert s.count(old) == 1, f"not found exactly once: {old!r}"
    s = s.replace(old, new)

# ---- TUNING ---------------------------------------------------------------
# T1 kick tail 47.6 Hz (G1 -49 c, a quarter tone) -> 49.0 Hz G1, a fifth under the D2 sub
rep("automate bd_tune\n0.16\n", "automate bd_tune\n0.20   # QA: kick tail on G1 (49.0 Hz), was 0.16 = 47.6 Hz (G1 -49 c)\n", "kick")
# T2 rim knock 203 Hz (Ab3 -36 c, the tritone) -> 220 Hz A3, the fifth of D
rep("automate rs_tune\n0.32\n", "automate rs_tune\n0.52   # QA: rim main mode on A3 (220 Hz), was 0.32 = 203 Hz (Ab3 -36 c)\n", "rim")
# T3 snare body 167 Hz (E3 +24 c) -> E3
rep("automate sd_tune\n0.2\n", "automate sd_tune\n0.17   # QA: snare body on E3 (165 Hz), was 0.2 = 167 Hz (E3 +24 c)\n", "snare")
# T4 the ostinato: the filter-envelope sweep through the low harmonics drags each 16th flat
rep("automate ost.cutoff\n",
    "# QA: a deep fast filter-env sweep at a low cutoff bends every pluck 15-27 cents flat;\n"
    "# a shallower, slower sweep plus a small trim where the cutoff sits lowest\n"
    "automate ost.filter_env\n1.3\nautomate ost.filter_decay\n0.4\nautomate ost.resonance\n1.0\n"
    "automate ost.pitch_shift: 0.12 during TRAIN,ICE base 0\n"
    "automate ost.cutoff\n", "ost")
# T5 Ember's 0.3 s glide (for the 26.5 s query slide) was left on for the values-pour line at 30.25 s
rep("automate ember.glide\n0.025 >READ 0.3:0.1 >SECRET 0.025:0.1",
    "automate ember.glide\n0.025 >READ R:1.2 0.3:0.1 R:6.1 0.025:0.1 >SECRET 0.025:0.1   # QA: 0.3 s glide only for the D3->A4 query (26.5 s), off before the 30.25 s line", "glide")

# ---- TIMING: attacks ------------------------------------------------------
# M1/M2 pads reach -6 dB ~0.27 s after the beat: shorter attacks from the drop on
rep("automate epad.release\n",
    "automate epad.attack: 0.5 during ASSEMBLY,PASS1,HOLD,SECRET,DROP base 0.12   # QA: chord changes land on the beat\n"
    "automate epad.release\n", "epad attack")
rep("automate qpad.release\n",
    "automate qpad.attack: 0.9 during ASSEMBLY,PASS1,HOLD,SECRET,DROP base 0.2   # QA\n"
    "automate qpad.release\n", "qpad attack")
# M3 the stranger (68.0 s cue) peaks 114 ms late on sour's 0.25 s attack
rep(">DROP R:4 G#3:1.5@0.62 D#4:0.5@0.55 C#4:0.5@0.55 E4:1.5@0.6",
    ">DROP R:4 G#3:1.5@0.62 D#4:0.5@0.55 C#4:0.5@0.55 E4:1.5@0.6\nautomate ghost.attack\n0.06   # QA: was 0.25 s", "ghost")

# ---- TIMING: written placement, snapped to 16ths (picture cues: audio <= 75 ms late / <= 50 ms early)
# exchanges 15.8 + 0.9k s (beats 31.6 + 1.8k) -> beats 31.75 33.5 35.25 37 38.75 40.75 42.5 44.25
rep(">SLOWPASS R:1.6 (D2:0.5@0.6 R:1.3)x8",
    ">SLOWPASS R:1.75 D2:0.5@0.6 R:1.25 D2:0.5@0.6 R:1.25 D2:0.5@0.6 R:1.25 D2:0.5@0.6 R:1.25 D2:0.5@0.6 R:1.5 D2:0.5@0.6 R:1.25 D2:0.5@0.6 R:1.25 D2:0.5@0.6", "thump")
rep(">BRIDGES (RS:1@0.55)x7 RS:0.6@0.55 (RS:1.8@0.7)x8",
    ">BRIDGES (RS:1@0.55)x7 RS:0.75@0.55 RS:1.75@0.7 RS:1.75@0.7 RS:1.75@0.7 RS:1.75@0.7 RS:2@0.7 RS:1.75@0.7 RS:1.75@0.7 RS:1.8@0.7", "xhit ex")
rep("R:1.4 G4:1.5@0.78 R:2.1 Bb4:1.5@0.8 R:2.1 D5:1.5@0.8 R:2.1 F5:1.8@0.82",
    "R:1.5 G4:1.5@0.78 R:2 Bb4:1.5@0.8 R:2.25 D5:1.5@0.8 R:2 F5:1.75@0.82", "quill ex")
rep("R:0.6 F3:1.5@0.8 R:2.1 A3:1.5@0.8 R:2.1 C4:1.5@0.82 R:2.1 E4:1.5@0.84 R:2.1",
    "R:0.75 F3:1.5@0.8 R:2 A3:1.5@0.8 R:2 C4:1.5@0.82 R:2.25 E4:1.5@0.84 R:2", "ember ex")
# window pops 42.0 + 0.6k s (beats 84 + 1.2k) -> 84 85.25 86.5 87.75 88.75 90 91.25 92.5 93.75
rep(">GAMES (RS:1.2@0.62)x9",
    ">GAMES RS:1.25@0.62 RS:1.25@0.62 RS:1.25@0.62 RS:1@0.62 RS:1.25@0.62 RS:1.25@0.62 RS:1.25@0.62 RS:1.25@0.62 RS:1.2@0.62", "xhit pops")
rep(">GAMES A4:1.2@0.62 R:1.2 E4:1.2@0.6 R:1.2 B4:1.2@0.62 R:1.2 D5:1.2@0.62 R:1.2 C5:1.2@0.6",
    ">GAMES A4:1.25@0.62 R:1.25 E4:1.25@0.6 R:1 B4:1.25@0.62 R:1.25 D5:1.25@0.62 R:1.25 C5:1.05@0.6", "quill pops")
rep(">GAMES R:1.2 D3:1.2@0.62 R:1.2 A3:1.2@0.6 R:1.2 G3:1.2@0.62 R:1.2 F3:1.2@0.6",
    ">GAMES R:1.25 D3:1.25@0.62 R:1.25 A3:1.25@0.6 R:1 G3:1.25@0.62 R:1.25 F3:1.1@0.6", "ember pops")
# window hits 73.0 / 73.8 / 74.6 s (beats 146, 147.6, 149.2) -> 146, 147.75, 149.25
rep("[BD CP OH]:1.6@1 [BD CP OH]:1.6@1 [BD CP OH]:1.6@1 CH:0.4@0.14 BD:0.4@0.6 CH:0.4@0.14",
    "[BD CP OH]:1.75@1 [BD CP OH]:1.5@1 [BD CP OH]:1.5@1 CH:0.5@0.14 BD:0.25@0.6 CH:0.5@0.14", "drum hits")
rep("[Bb1 F2 Bb2 D3]:0.8@0.92 R:0.8 [C2 G2 C3 E3]:0.8@0.92 R:0.8 [D2 A2 D3 F3]:0.8@0.96",
    "[Bb1 F2 Bb2 D3]:0.8@0.92 R:0.95 [C2 G2 C3 E3]:0.8@0.92 R:0.7 [D2 A2 D3 F3]:0.8@0.96", "stab hits")
rep("Bb1:1.6@0.9 C2:1.6@0.9 D2:1.6@0.95 G1:1.2@0.85", "Bb1:1.75@0.9 C2:1.5@0.9 D2:1.5@0.95 G1:1.25@0.85", "sub hits")
rep("R:4.8 G2:0.6 D2:0.6", "R:4.75 G2:0.5 D2:0.75", "ost proof")
rep("G4:1.2@0.76 Bb4:0.8@0.8", "G4:1.25@0.76 Bb4:0.75@0.8", "quill proof")
rep("G3:1.2@0.76 Bb3:0.8@0.8", "G3:1.25@0.76 Bb3:0.75@0.8", "ember proof")
# assembly motifs on 0.8-beat steps -> dotted eighths
rep(">ASSEMBLY R:0.8 D4:0.8@0.48 F4:0.8@0.5 A4:0.8@0.53 Bb4:1.2@0.58 A4:2@0.45",
    ">ASSEMBLY R:0.75 D4:0.75@0.48 F4:0.75@0.5 A4:0.75@0.53 Bb4:1.25@0.58 A4:2@0.45", "quill asm")
rep(">ASSEMBLY R:1.6 D3:1@0.5 A3:1@0.52 B3:1@0.55 C4:1.2@0.55 A3:2@0.45",
    ">ASSEMBLY R:1.5 D3:1@0.5 A3:1@0.52 B3:1@0.55 C4:1.25@0.55 A3:2@0.45", "ember asm")
rep(">READ R:7 A4:0.3@0.4 D4:1.2@0.36", ">READ R:7 A4:0.25@0.4 D4:1.25@0.36", "sig")
rep("automate sig.glide\n0.2\n", "automate sig.glide\n0.2\nautomate sig.attack\n0.02   # QA: glasslead's 0.06 s attack put the 29.5 s gate 54 ms late\n", "sig attack")

open(os.path.join(HERE, "score2_fixed.song"), "w").write(s)
print("wrote", os.path.join(HERE, "score2_fixed.song"))
