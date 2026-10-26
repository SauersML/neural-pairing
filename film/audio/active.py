# Active level of each wav: 90th-percentile 400 ms RMS (dBFS), and where it is active.
import sys, numpy as np
sys.path.insert(0, '/home/user/video/audio')
from readwav import readwav
for f in sys.argv[1:]:
    x, sr = readwav(f); m = x.mean(1); n = int(0.4*sr)
    k = len(m)//n; r = np.sqrt((m[:k*n].reshape(k, n)**2).mean(1)) + 1e-9
    db = 20*np.log10(r); act = db > db.max() - 20
    print(f"{f.split('solo_')[-1][:-4]:10s} p90 {np.percentile(db[act],90):6.1f}  max {db.max():6.1f}  active {act.sum()*0.4:5.1f}s")
