# Verify sync: for each cue, find the strongest onset (max rise of 5 ms log-energy,
# high-passed to favour transients) within +-120 ms and report its offset.
# Also report the RMS level just before and just after each cue.
import sys, numpy as np
sys.path.insert(0, '/home/user/video/audio')
from readwav import readwav
x, sr = readwav(sys.argv[1]); m = x.mean(1).astype(np.float64)
hp = np.concatenate([[0], np.diff(m)])          # simple first-difference high-pass
hop = int(0.0025*sr); win = int(0.005*sr)
def loge(sig):
    k = (len(sig)-win)//hop
    idx = np.arange(k)[:, None]*hop + np.arange(win)[None, :]
    return 10*np.log10((sig[idx]**2).mean(1) + 1e-12)
E = loge(hp); Ef = loge(m)
def rms(a, b):
    s = m[int(a*sr):int(b*sr)]; return 20*np.log10(np.sqrt((s**2).mean())+1e-9)
CUES = [("swell start", 0.0), ("arp first note", 0.4), ("Ember joins", 0.775), ("shatter", 8.0), ("drop", 10.0)] + \
       [(f"bridge pluck {k+1}", 10.0+0.5*k) for k in range(1, 8)] + \
       [("impact", 22.0), ("lantern resolve", 29.5)] + [(f"game hit {k+1}", 30.5+0.5*k) for k in range(5)] + \
       [("training", 36.0), ("big hit", 50.0), ("stat 1", 53.0), ("stat 2", 53.8), ("stat 3", 54.6), ("outro", 55.6)]
if len(sys.argv) > 2 and sys.argv[2] == "v2":
    CUES = [("Quill motif", 0.4), ("Ember motif", 0.8), ("pass-1 start", 4.0), ("tokens pop", 7.0), ("shatter", 8.0),
            ("hold (Quill fifth)", 10.25), ("drop / bridge 1", 12.0)] + \
           [(f"bridge {k+1}", 12.0+0.5*k) for k in range(1, 8)] + \
           [(f"exchange {k+1}", round(15.8+0.9*k, 2)) for k in range(8)] + \
           [("token pop", 23.0), ("read: query", 26.5), ("lantern (Quill)", 32.0), ("lantern again", 34.0),
            ("lantern (Ember)", 36.5), ("Quill major 3rd", 37.5), ("Ember major 3rd", 39.5)] + \
           [(f"window {k+1}", round(42.0+0.6*k, 2)) for k in range(9)] + \
           [("training", 54.0), ("ice", 58.0), ("riser start", 61.0), ("real partner", 66.5), ("stranger", 68.0),
            ("BIG hit", 70.0), ("window hit 1", 73.0), ("window hit 2", 73.8), ("window hit 3", 74.6), ("title", 78.0),
            ("final D unison", 80.0)]
W = 0.12; worst = 0
for name, t in CUES:
    i0 = max(0, int((t-W)/0.0025)); i1 = int((t+W)/0.0025)
    # onset strength: 30 ms of broadband energy after a frame vs 30 ms before it
    L = 12
    d = np.array([np.log10(np.mean(10**(Ef[i:i+L]/10))+1e-12)*10 - np.log10(np.mean(10**(Ef[i-L:i]/10))+1e-12)*10
                  for i in range(max(L, i0), i1)])
    j = np.argmax(d); i = max(L, i0) + j
    # refine to the first 2.5 ms frame inside the rise whose high-passed energy jumps most
    k = np.argmax(E[i-2:i+3] - E[i-4:i+1]); tt = (i - 2 + k)*0.0025
    off = (tt - t)*1000
    if t > 0.1: worst = max(worst, abs(off))
    print(f"{name:16s} cue {t:6.3f}  onset {tt:7.3f}  off {off:+6.1f} ms  rise {d[j]:5.1f} dB   rms before {rms(max(0,t-0.25), t):6.1f}  after {rms(t, t+0.25):6.1f}")
print(f"worst |offset| (excluding t=0): {worst:.1f} ms")
