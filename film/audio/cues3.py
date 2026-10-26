# Sync verification for score3.wav.
#   python3 cues3.py score3.wav work3/ev3.json work3
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
    f = os.path.join(stemdir, f"s3_{track}.wav") if stemdir else None
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
EX = [15.875 + 0.875*k for k in range(8)]
C = [("shatter", 8.0, None), ("drop", 12.0, None)] + \
    [(f"bridge {k+1}", 12.0 + 0.5*k, "ember" if k % 2 == 0 else "quill") for k in range(8)] + \
    [(f"exchange {k+1}", EX[k], "ember" if k % 2 == 0 else "quill") for k in range(8)] + \
    [("token (both voices on D4)", 23.0, None), ("fast pass 1", 23.5, "ember"), ("read", 26.0, None),
     ("lantern, Quill", 32.0, "quill"), ("lantern, Quill again", 34.0, "quill"),
     ("lantern played back by Ember", 36.5, "ember"),
     ("Quill raises its third F->F#", 37.5, "quill"), ("Ember raises its third F->F#", 39.5, "ember")] + \
    [(f"games wave {i+1} slam {j+1}", w + 0.25*j, None) for i, w in enumerate((42.0, 45.0, 48.0)) for j in range(3)] + \
    [(f"games wave {i+1} ignite", w + 1.0, None) for i, w in enumerate((42.0, 45.0, 48.0))] + \
    [("pull-back", 51.0, None), ("games title", 51.5, None), ("stack pulse", 53.0, None),
     ("training", 54.0, None), ("ice", 58.0, None), ("riser start", 61.0, None),
     ("real partner's lantern", 66.5, "ember"), ("stranger's sour note", 68.0, "ghost"), ("BIG hit", 70.0, None),
     ("result window 1", 73.0, None), ("result window 2", 73.75, None), ("result window 3", 74.5, None),
     ("result window 4", 75.25, None), ("title", 78.25, None), ("final unison D", 80.0, "ember")]
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
