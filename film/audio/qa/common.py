"""Shared helpers for the score QA tools (tuning.py, timing.py).

Everything goes through Patina itself: the song is never re-interpreted by
hand. `--export-events` gives the exact note-on/off times (after the tempo
map, sections, `>NAME` seeks, gate) and `--render-stems` solos each track
through the full engine. Patina resolves `patches/NAME.patch` and
`sample=` paths relative to the working directory, so it is always run
from the song's own directory.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

PATINA = os.environ.get("PATINA", "/home/user/patina/target/release/patina")
QA_DIR = os.path.dirname(os.path.abspath(__file__))
DRUM_CH = 65535
SAMPLER_BASE = 65500          # sampler slots sit just under the drum channel


def run_patina(args, cwd):
    r = subprocess.run([PATINA] + args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"patina {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def _newer(path, than):
    return os.path.exists(path) and all(os.path.getmtime(path) >= os.path.getmtime(t) for t in than)


def song_cwd(song):
    """The directory Patina runs in: the song's own, or the nearest parent
    holding a patches/ folder (so test songs in qa/ab/ find the score's
    patches; their sample= paths are then relative to that folder too)."""
    d = os.path.dirname(os.path.abspath(song))
    p = d
    while p != "/":
        if os.path.isdir(os.path.join(p, "patches")):
            return p
        p = os.path.dirname(p)
    return d


def song_deps(song):
    """The song plus every file it reads (patches, samples), for cache checks."""
    d = song_cwd(song)
    deps = [song]
    txt = open(song).read()
    for m in re.finditer(r"\bpatch=(\S+)", txt):
        p = os.path.join(d, "patches", m.group(1) + ".patch")
        if os.path.exists(p):
            deps.append(p)
    for m in re.finditer(r"\b(?:sample|wav)=(\S+)", txt):
        p = os.path.join(d, m.group(1))
        if os.path.exists(p):
            deps.append(p)
    return deps


def export_events(song, cache_dir):
    song = os.path.abspath(song)
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, "events.json")
    if not _newer(out, song_deps(song)):
        run_patina(["--play", song, "--export-events", out], cwd=song_cwd(song))
    return json.load(open(out))


def render_stems(song, stem_dir, force=False):
    song = os.path.abspath(song)
    os.makedirs(stem_dir, exist_ok=True)
    stamp = os.path.join(stem_dir, ".rendered")
    if force or not _newer(stamp, song_deps(song)):
        print(f"[qa] rendering stems of {os.path.basename(song)} -> {stem_dir} "
              f"(a full-engine bounce per track; this takes a while)", file=sys.stderr)
        run_patina(["--play", song, "--render-stems", os.path.abspath(stem_dir)],
                   cwd=song_cwd(song))
        open(stamp, "w").write("ok\n")
    return stem_dir


def default_cache(song):
    name = os.path.splitext(os.path.basename(song))[0]
    return os.path.join(QA_DIR, "stems", name)


def load_mono(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


# ---------------------------------------------------------------- song text
def parse_song_text(song):
    """bpm, sections (seconds) and per-track declarations from the song text."""
    bpm = 120.0
    sections = []
    tracks = {}
    tempo_automated = False
    for raw in open(song):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        if tok[0] == "bpm" and len(tok) > 1:
            bpm = float(tok[1])
        elif tok[0] == "section" and len(tok) >= 3:
            a, b = tok[2].split("..")
            sections.append([tok[1], float(a), float(b)])
        elif tok[0] == "automate" and tok[1].rstrip(":") == "bpm":
            tempo_automated = True
        elif tok[0] == "track" and len(tok) > 1:
            opts = {}
            flags = set()
            for o in tok[2:]:
                if "=" in o:
                    k, v = o.split("=", 1)
                    opts[k] = v
                else:
                    flags.add(o)
            kind = ("kit" if "kit" in opts else "sampler" if "sample" in opts
                    else "vox" if "vox" in flags else "synth")
            tracks[tok[1]] = {"kind": kind, "opts": opts, "flags": flags}
    spb = 60.0 / bpm
    secs = [(n, a * spb, b * spb) for n, a, b in sections]
    return {"bpm": bpm, "spb": spb, "sections": secs, "tracks": tracks,
            "tempo_automated": tempo_automated}


def section_of(t, sections):
    name = "-"
    for n, a, b in sections:
        if a - 1e-6 <= t < b - 1e-6:
            name = n
    return name


def read_patch(song, name):
    p = os.path.join(song_cwd(song), "patches", name + ".patch")
    vals = {}
    if os.path.exists(p):
        for raw in open(p):
            line = raw.split("#", 1)[0].split()
            if len(line) >= 2:
                try:
                    vals[line[0]] = float(line[1])
                except ValueError:
                    pass
    return vals


# ---------------------------------------------------------------- events
def notes_by_track(ev):
    """{track: [ {on, off, note, vel} ]}. Kit tracks share one channel (the
    909 board) and sampler tracks their slot channel; names follow the
    exported track map (first name per channel wins, as in render-stems)."""
    ch2name = {}
    for name, ch in ev["tracks"].items():
        ch2name.setdefault(ch, name)
    open_notes = {}
    out = {}
    for e in ev["events"]:
        if e["type"] == "on":
            key = (e["ch"], e["note"])
            open_notes.setdefault(key, []).append((e["t"], e["vel"]))
        elif e["type"] == "off":
            key = (e["ch"], e["note"])
            if open_notes.get(key):
                t0, vel = open_notes[key].pop(0)
                name = ch2name.get(e["ch"], f"ch{e['ch']}")
                out.setdefault(name, []).append(
                    {"on": t0, "off": e["t"], "note": e["note"], "vel": vel})
    for v in out.values():
        v.sort(key=lambda n: (n["on"], n["note"]))
    return out


def param_events(ev, channel):
    """[(t, param, value)] for one channel (0 = the bus)."""
    return [(e["t"], e["param"], e["value"]) for e in ev["events"]
            if e["type"] == "param" and e["ch"] == channel]


def param_value_at(pev, param, t, default=None):
    v = default
    for tt, p, val in pev:
        if p == param and tt <= t + 1e-9:
            v = val
    return v


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f, dtype=float) / 440.0)


NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def note_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"
