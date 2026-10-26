#!/usr/bin/env python3
"""Robustness test of score4's spoken title in its real mix context, fast.

    [VT_SONG=score5.song VT_AT=100.5 VT_CROP=0] python3 qa/voice/vtest4.py NAME [K] [SUB=REPL ...]

Takes score4.song (VT_SONG), keeps every lane but drops note events before VT_CROP (default 6.5 s
before the voice; 0 = keep the whole song) (the fx stem's single
note at 0 stays), applies regex substitutions SUB=REPL to the song text (a variant), adds k
inaudible velocity-0.01 glass notes 6.5 s before the voice (k = 0..K-1: they reshuffle Patina's voice-card
assignment, which once flipped the transcript), renders each one niced and in sequence, masters
like master4.sh, and transcribes 92.3-94.1 s with small.en and base.en. Prints one line per
render and the pass count (pass = both models read exactly "neural pairing").
"""
import os
import re
import subprocess
import sys

ROOT = "/home/user/video/audio"
PATINA = "/home/user/patina/target/release/patina"
OUT = os.environ.get("VT_DIR", "/tmp/claude-0/-home-user-neural-pairing/1ea0fef9-0478-59bc-9fb1-2b686678cb10/scratchpad/vt")
sys.path.insert(0, os.path.join(ROOT, "qa", "voice"))
import asr  # noqa: E402

name = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
subs = [a.split("=", 1) for a in sys.argv[3:]]
os.makedirs(OUT, exist_ok=True)
SONG = os.environ.get("VT_SONG", "score4.song")         # v5: VT_SONG=score5.song VT_AT=100.5
AT = float(os.environ.get("VT_AT", "92.5"))               # where "neural" starts
CROP = float(os.environ.get("VT_CROP", str(AT - 6.5)))    # VT_CROP=0: the full song
GAIN = os.environ.get("VT_GAIN", "2.2")
src = open(os.path.join(ROOT, SONG)).read()
for a, b in subs:
    src, n = re.subn(a, b.replace("\\n", "\n"), src)
    assert n, f"no match for {a}"


def crop(text):
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"track (\S+)", line)
        if m:
            cur = m.group(1); out.append(line); continue
        if line.startswith("automate") or not line.startswith(">"):
            if line.startswith("automate"):
                cur = None
            out.append(line); continue
        if cur in (None, "fx"):
            out.append(line); continue
        ev = re.findall(r">[\d.]+ (?:\[[^\]]*\]\S*|\S+)", line)
        keep = [e for e in ev if float(e[1:].split()[0]) / 2 >= CROP]
        if keep:
            out.append(" ".join(keep))
    return "\n".join(out) + "\n"


base = crop(src) if CROP > 0 else src
ok = 0
for k in range(K):
    pert = " ".join(f">{2 * (AT - 6.5 + 0.125 * j):g} C7:0.1@0.01" for j in range(k))
    song = base.replace("track glass ", "track glass ", 1)
    if pert:
        song = re.sub(r"(track glass [^\n]*\n)", lambda m: m.group(1) + pert + "\n", song, count=1)
    sp = os.path.join(ROOT, "work4", f"vt_{name}_{k}.song")
    open(sp, "w").write(song)
    raw = os.path.join(OUT, f"{name}_{k}_raw.wav"); ex = os.path.join(OUT, f"{name}_{k}_ex.wav")
    subprocess.run(["nice", "-n", "10", PATINA, "--play", sp, "--render", raw, "--no-normalize"], cwd=ROOT,
                   check=True, capture_output=True)
    af = f"highpass=f=24:poles=2,volume={GAIN}dB,alimiter=limit=0.84:attack=1:release=60:level=disabled"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{AT - 0.2:g}", "-t", "1.8", "-i", raw, "-af", af, ex], check=True)
    r = {m: asr.transcribe(ex, m)[0] for m in ("small.en", "base.en")}
    good = all(asr.score(t, "neural pairing") == 1.0 for t in r.values())
    ok += good
    print(f"{name} k={k}: {'PASS' if good else 'fail'}  " + "  ".join(f"{m}: {t!r}" for m, t in r.items()), flush=True)
print(f"{name}: {ok}/{K} pass")
