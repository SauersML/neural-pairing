#!/usr/bin/env python3
"""Render a variant of make_score3.py (extra lane code appended before the write step),
master it like master3.sh into a scratch file, and transcribe 78.3-80.1 s.
    python3 qa/ab3/voice_variant.py NAME "python code to append" OUTDIR"""
import os, subprocess, sys, tempfile, shutil
name, code, out = sys.argv[1], sys.argv[2], sys.argv[3]
root = "/home/user/video/audio"
src = open(os.environ.get("GEN", os.path.join(root, "make_score3.py"))).read()
marker = "# ------------------------------------------------------------------ write score3.song"
src = src.replace(marker, code + "\n" + marker)
tmp = tempfile.mkdtemp()
open(os.path.join(tmp, "gen.py"), "w").write(src)
subprocess.run([sys.executable, "gen.py"], cwd=tmp, check=True, capture_output=True)
song = os.path.join(root, "work3", f"vtest_{name}.song")
shutil.copy(os.path.join(tmp, "score3.song"), song)
raw = os.path.join(out, f"{name}_raw.wav"); mst = os.path.join(out, f"{name}.wav"); ex = os.path.join(out, f"{name}_ex.wav")
subprocess.run(["nice", "/home/user/patina/target/release/patina", "--play", song, "--render", raw, "--no-normalize"],
               cwd=root, check=True, capture_output=True)
af = ("highpass=f=24:poles=2,volume=2.4dB,alimiter=limit=0.84:attack=1:release=60:level=disabled,"
      "afade=t=in:st=0:d=0.3,afade=t=out:st=80.6:d=1.3:curve=qsin,apad=whole_dur=82,atrim=end_sample=3936000")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", af, "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", mst], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "78.3", "-t", "1.8", "-i", mst, ex], check=True)
sys.path.insert(0, os.path.join(root, "qa", "voice"))
import asr
r = [f"{m}: {asr.transcribe(ex, m)[0]!r}" for m in ("small.en", "base.en")]
print(name, " | ".join(r), flush=True)
