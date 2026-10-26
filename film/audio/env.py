# Print a coarse RMS envelope (dBFS) of a wav: python3 env.py file.wav [step_s] [t0] [t1]
import sys, numpy as np
sys.path.insert(0, '/home/user/video/audio')
from readwav import readwav
x, sr = readwav(sys.argv[1])
step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else len(x)/sr
m = x.mean(1)
n = int(step*sr)
for i in range(int(t0/step), int(t1/step)):
    seg = x[i*n:(i+1)*n]
    if len(seg) == 0: break
    r = np.sqrt((seg**2).mean()) + 1e-9; p = np.abs(seg).max() + 1e-9
    db = 20*np.log10(r)
    print(f"{i*step:6.2f}  rms {db:6.1f}  pk {20*np.log10(p):6.1f}  " + '#'*max(0, int((db+70)/1.5)))
