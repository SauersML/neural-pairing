#!/usr/bin/env python3
"""Put the "neural pairing" line into a score: makes room at the title and appends the
vox track (neural_pairing_snippet.song).
    python3 qa/voice/make_voice_score.py [IN.song] [OUT.song]
    (defaults: qa/fixes/score2_fixed.song -> qa/fixes/score2_fixed_voice.song)
Works on score2.song too (the TITLE lines are the same in both)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "fixes", "score2_fixed.song")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "fixes", "score2_fixed_voice.song")
s = open(src).read()

def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

# the snare+clap at 79.0 lands on the P of "pairing": move it to 80.0 with the final kick
rep(">TITLE [BD OH]:2@0.85 [SD CP]:2@0.6 BD:4@0.5", ">TITLE [BD OH]:2@0.85 R:2 [BD SD CP]:4@0.55")
# -6 dB on the melody and the pads for the phrase (78.4 -> 79.85 s), back by 80.0
DIP = " >TITLE R:0.6 {lo}:0.2 R:2.9 {hi}:0.3"
for line, lo, hi in (("automate quill.gain\n0.65 >BRIDGES 1.1:0.1 >READ 0.9:0.5 >SECRET 1.1:0.1", 0.55, 1.1),
                     ("automate ember.gain\n0.6 >BRIDGES 1.4:0.1 >READ 1.15:0.5 >SECRET 1.4:0.1 >DROP 0.8:0.1 >PROOF 1.4:0.1", 0.7, 1.4),
                     ("automate epad.gain\n0 0.26:6@smooth >BRIDGES 0.42:0.1 >READ 0.3:1@smooth >SECRET 0.27:0.5 >MOOD 0.42:2@smooth", 0.21, 0.42),
                     ("automate qpad.gain\n0 0.22:6@smooth >BRIDGES 0.4:0.1 >READ 0.3:1@smooth >SECRET 0.26:0.5 >MOOD 0.44:2@smooth >GAMES 0.36:1", 0.18, 0.36)):
    rep(line, line + DIP.format(lo=lo, hi=hi))
s = s.rstrip("\n") + "\n\n" + open(os.path.join(HERE, "neural_pairing_snippet.song")).read()
open(dst, "w").write(s)
print("wrote", dst)
