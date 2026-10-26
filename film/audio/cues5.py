# Sync verification for score5.wav (v5: the v4 cue list warped by W(), plus the break).
#   python3 cues5.py score5.wav work5/ev5.json work5
# Drum/impact cues: broadband onset (max 30 ms energy contrast, refined to 2.5 ms).
# Melodic cues: the onset of the cue's own pitch. The note numbers are read from the
# exported score events, and the onset is the sharpest rise of narrow-band energy at
# that note's fundamental and 2nd harmonic. A groove kick a few ms away cannot fool it.
import sys, json, numpy as np
sys.path.insert(0, '/home/user/video/audio')
from readwav import readwav
x, sr = readwav(sys.argv[1]); m = x.mean(1).astype(np.float64)
# optional 3rd arg: a stem of just the two voices (+ the stranger), rendered through the
# same engine, for the pitched cues; the full mix is still checked broadband at every cue
# optional 3rd arg: a directory holding per-voice stems s2_<track>.wav (each voice soloed
# through the same engine); pitched cues are then measured on their own voice's stem
import os
stemdir = sys.argv[3] if len(sys.argv) > 3 else None
def voice(track):
    f = os.path.join(stemdir, f"s5_{track}.wav") if stemdir else None
    return readwav(f)[0].mean(1).astype(np.float64) if f and os.path.exists(f) else m
d = json.load(open(sys.argv[2])); tracks = d['tracks']; ev = d['events']
def notes_at(track, t):
    ch = tracks[track]
    return sorted({e['note'] for e in ev if e['type'] == 'on' and e['ch'] == ch and abs(e['t'] - t) < 0.004})
def broadband(t, W=0.12):
    hop = int(0.0025*sr); win = int(0.005*sr)
    a = int((t - W - 0.05)*sr); seg = m[a:a + int((2*W + 0.1)*sr)]
    k = (len(seg) - win)//hop
    E = 10*np.log10(np.array([np.mean(seg[i*hop:i*hop+win]**2) for i in range(k)]) + 1e-12)
    L = 12; best = None
    for i in range(L, k - L):
        tt = (a + i*hop)/sr
        if abs(tt - t) > W: continue
        c = 10*np.log10(np.mean(10**(E[i:i+L]/10))) - 10*np.log10(np.mean(10**(E[i-L:i]/10)))
        if best is None or c > best[0]: best = (c, tt)
    return best[1], best[0]
def sounding(track, t):
    """notes on this track already sounding at t (held, not the new ones)"""
    ch = tracks[track]; on = {}
    for e in ev:
        if e['ch'] != ch or e['t'] > t - 0.004: continue
        if e['type'] == 'on': on[e['note']] = e['t']
        elif e['type'] == 'off': on.pop(e['note'], None)
    return list(on)
def pitched(t, midis, mv, W=0.12, others=()):
    n = 2048; hop = 120; w = np.hanning(n)
    a = int((t - W)*sr) - n//2; frames = int((2*W)*sr)//hop
    f = np.fft.rfftfreq(n, 1/sr); mask = np.zeros_like(f, bool)
    busy = [h*440*2**((o - 69)/12) for o in others for h in range(1, 7)]
    for q in midis:
        f0 = 440*2**((q - 69)/12)
        # the lowest three partials not shared with notes already held on the track
        # (fall back to the fundamental and octave when every partial is shared)
        free = [h for h in range(1, 9) if not any(abs(h*f0 - x) < 0.03*h*f0 for x in busy)][:3] or [1, 2]
        for h in free:
            fh = h*f0
            mask |= np.abs(f - fh) < max(0.03*fh, 12)
    E = np.array([10*np.log10((np.abs(np.fft.rfft(mv[a+i*hop:a+i*hop+n]*w))[mask]**2).sum() + 1e-12) for i in range(frames)])
    r = E[4:] - E[:-4]; j = np.argmax(r) + 2
    return (a + j*hop + n//2)/sr, r.max()
# the cue list comes from the hit table at the top of make_score5.py
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_score5.py")).read()
_h = {}
exec(_src[_src.index("# ====================================================================== HIT TABLE"):
          _src.index("# ====================================================================== helpers")], _h)
globals().update({k: v for k, v in _h.items() if k.isupper()})
C = [("shatter", A1_SHATTER, None), ("a2 drop", A2_DROP, None)] + \
    [(f"a2 strand {k+1}", t, "quill") for k, t in enumerate(A2_STRANDS)] + \
    [("a2 pass starts", A2_PASS, None)] + \
    [(f"a2 exchange {k+1}", t, "ember" if k % 2 == 0 else "quill") for k, t in enumerate(A2_EX)] + \
    [("a2 token (both on D4)", A2_TOKEN, None), ("a2 emit", A2_EMIT, None), ("a2 keys/values", X["kv"], "glassq"),
     ("a2 queries", X["q"], "ember"), ("a2 rays", X["rays"], "quill"), ("a2 values pour", X["flow"], "quill"),
     ("a2 the write lands", X["hit"], None)] + \
    [(f"a2 gen token {k+1}", t, "quill") for k, t in enumerate(A2_GEN)] + \
    [("a2 release (fold)", A2_FOLD, None), ("a3 lantern", A2_END, "quill"),
     ("a3 card pops", A3_CARD, "quill"),
     ("a3 card into the tail", A3_TAIL, None), ("a3 Ember writes lantern", A3_WRITE, "ember"),
     ("a3 push into Ember", A3_PUSH, None)] + \
    [(f"a3 linked token {k+1}", t, "quill") for k, t in enumerate(A3_RUN)] + \
    [("a3 link cut", A3_CUT, None)] + \
    [(f"a3 cut token {k+1}", t, "quill") for k, t in enumerate(A3_DRY)] + \
    [("a3 caption", A3_CAPTION, None)] + \
    [(f"a4 wave {i+1} slam {j+1}", w + 0.25*j, None) for i, w in enumerate(A4_WAVES) for j in range(3)] + \
    [(f"a4 wave {i+1} first token", A4_WRITE[i][0], "quill") for i in range(3)] + \
    [(f"a4 check {k+1}", t, "glass") for k, t in enumerate(A4_CHECKS)] + \
    [("a4 Taipei", A4_TAIPEI, None), ("a4 deck lands", A4_LAND, None), ("a4 deck compresses", A4_PRESS, None),
     ("a5 training", 54.0 + SH_A5, None)] + \
    [(f"a5 read {k+1}", t, "quill") for k, t in enumerate(A5_READS)] + \
    [("a5 scatter", A5_SCATTER, None), ("a5 frost Quill", A5_FROST[0], "glassq"), ("a5 LoRA", A5_LORA, "glass"),
     ("a5 cards 8ths", A5_CARDS8, None), ("a5 cards 16ths", A5_CARDS16, None), ("a5 100 %", A5_TOP, None),
     ("a5b downbeat", A5B_DOWN, None), ("a5b lands", A5B_LAND, None), ("a5b card pluck", A5B_TICKS[0], "glass")] + \
    [(f"a5b token {k+1}", t, "ember") for k, t in enumerate(A5B_RUN)] + \
    [("a5b ?", A5B_QMARK, "ember"), ("a5b joint pass", A5B_JOINT, None), ("a5b target lights", A5B_HIT, None),
     ("a6 real partner's lantern", 66.5 + SH_A6, "ember"), ("a6 stranger's sour note", 68.0 + SH_A6, "ghost"),
     ("a6 BIG hit", 70.0 + SH_A6, None)] + \
    [(f"a6 result window {k+1}", t + SH_A6, None) for k, t in enumerate((73.0, 73.75, 74.5, 75.25))] + \
    [("a6 title", 78.25 + SH_A6, None), ("a6 final unison D", 80.0 + SH_A6, "ember"), ("a6 pairing (carrier)", VOICE_PAIRING, None)]
C = [(n, W(t), tr) for n, t, tr in C if not n.startswith("a3 lantern")]
C += [("break: last groove hit", BRK_LAST, None), ("break: the press", BRK_PRESS, None),
      ("break: the split", BRK_SPLIT, None), ("break: compass (Bb4, felt)", BRK_COMPASS, "felt"),
      ("break: fold (felt)", BRK_FOLD, "felt"), ("break re-entry / a3", BRK_END, None)]
C.sort(key=lambda c: c[1])
worst = 0
for name, t, tr in C:
    if tr:
        q = notes_at(tr, t)
        if not q: print(f"{name}: no event!"); continue
        tt, rise = pitched(t, q, voice(tr), others=[o for o in sounding(tr, t) if o not in q]); kind = f"{tr} {q}"
        tb, rb = broadband(t); kind += f"; full-mix broadband {(tb-t)*1000:+.1f} ms"
    else:
        tt, rise = broadband(t); kind = "broadband"
    off = (tt - t)*1000; worst = max(worst, abs(off))
    print(f"{name:32s} cue {t:6.2f}  onset {tt:7.3f}  {off:+6.1f} ms  ({kind}, rise {rise:4.1f} dB)")
print(f"worst |offset|: {worst:.1f} ms")
