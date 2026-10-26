#!/usr/bin/env python3
"""Voice-only robustness test of score3's voice line: the generator's voice track and bus
lanes alone (every other track removed), rendered with K different inaudible perturbations
(dummy glass notes that reshuffle Patina's voice-card assignment), mastered like master3.sh,
78.3-80.1 s transcribed with Whisper small.en and base.en.
    python3 qa/ab3/voice_robust.py NAME "python code" [K]"""
import os, subprocess, sys, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
root = "/home/user/video/audio"
name, code = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
S = "/tmp/claude-0/-home-user/1ea0fef9-0478-59bc-9fb1-2b686678cb10/scratchpad/vr"
os.makedirs(S, exist_ok=True)
src = open(os.environ.get("GEN", os.path.join(root, "make_score3.py"))).read()
marker = "# ------------------------------------------------------------------ write score3.song"
def build(k):
    pert = "\n".join(f'gl.n({0.5 + 0.37 * j:.2f}, 0.05, "C7", 0.01)' for j in range(k))
    keep = "" if os.environ.get("FULL") else f"\ntracks[:] = [t for t in tracks if t.name in (\"voice\", \"glass\")]\n"
    s = src.replace(marker, code + "\n" + pert + keep + marker)
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "gen.py"), "w").write(s)
    subprocess.run([sys.executable, "gen.py"], cwd=tmp, check=True, capture_output=True)
    song = os.path.join(root, "work3", f"vr_{name}_{k}.song")
    shutil.copy(os.path.join(tmp, "score3.song"), song)
    raw = f"{S}/{name}_{k}_raw.wav"; ex = f"{S}/{name}_{k}_ex.wav"
    subprocess.run(["nice", "/home/user/patina/target/release/patina", "--play", song, "--render", raw, "--no-normalize"],
                   cwd=root, check=True, capture_output=True)
    af = "highpass=f=24:poles=2,volume=2.4dB,alimiter=limit=0.84:attack=1:release=60:level=disabled"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "78.3", "-t", "1.8", "-i", raw, "-af", af, ex], check=True)
    return ex
with ThreadPoolExecutor(2) as ex_:
    exs = list(ex_.map(build, range(K)))
sys.path.insert(0, os.path.join(root, "qa", "voice"))
import asr
ok = 0
res = []
for k, ex in enumerate(exs):
    r = {m: asr.transcribe(ex, m)[0] for m in ("small.en", "base.en")}
    good = all(asr.score(t, "neural pairing") == 1.0 for t in r.values())
    ok += good
    res.append(f"{k}:{'OK' if good else 'x'} {r['small.en']!r}/{r['base.en']!r}")
print(f"{name:<22} {ok}/{K}  " + "  ".join(res), flush=True)
