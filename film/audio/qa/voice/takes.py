#!/usr/bin/env python3
"""Render "neural pairing" takes on Patina's voice box and score them with Whisper.

    python3 qa/voice/takes.py [--only NAME,...] [--model small.en]

Each take is a tiny song in qa/voice/takes/<name>.song, rendered from the score
folder (so patches resolve) to <name>.wav (48 kHz), then transcribed. Results go to
qa/voice/takes/results.json and a table on stdout, with two objective timbre numbers:
spectral centroid (Hz; lower = darker) and the share of tonal energy whose pitch class
is in D minor (the carrier is the synth, so this checks the take sits in the key).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import asr  # noqa: E402
import common as C  # noqa: E402

SCORE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
TAKES = os.path.join(HERE, "takes")

LYRICS = {
    # plain CMU-dict ARPAbet, one word per note
    "plain": ("N-UH1-R-AH0-L", "P-EH1-R-IH0-NG."),
    # consonants given time and weight: a long N, a hard P, a held NG
    "crisp": ("N:110-UH1-R-AH0-L:90", "P:60@1.5-EH1-R-IH0-NG:170."),
    # "nyoo-ral" (the other dictionary reading)
    "nyoo": ("N:100-Y-UW1-R-AH0-L:90", "P:60@1.5-EH1-R-IH0-NG:170."),
    # spelling variants found by the dry-voice search (grid "spell")
    "nuw": ("N-UW1-R-AH0-L", "P-EH1-R-IH0-NG."),
    "nur": ("N-ER1-AH0-L", "P-EH1-R-IH0-NG."),
    "pey": ("N-UH1-R-AH0-L", "P-EY1-R-IH0-NG."),
    "nuw_crisp": ("N:110-UW1-R-AH0-L:90", "P:70@1.6-EH1-R-IH0-NG:170."),
    "nyu_pey": ("N:100-Y-UW1-R-AH0-L:90", "P:70@1.6-EY1-R-IH0-NG:170."),
    "nur_pey": ("N:110-ER1-AH0-L:90", "P:70@1.6-EY1-R-IH0-NG:170."),
    "nlong": ("N:160@1.4-UW1:200-R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1:200-R-IH0:110-NG:220."),
    "slow": ("N:130-UH1:220-R-AH0:120-L:110", "P:80@1.8-EH1:220-R-IH0:120-NG:200."),
    "slow_nuw": ("N:130-UW1:220-R-AH0:120-L:110", "P:80@1.8-EH1:220-R-IY0:120-NG:200."),
}
CHORDS = {
    # the score's harmony at 78-80 s: Dm(add9) then Bbmaj7, mid register
    "dm_bb": ("[D3 A3 E4 F4]", "[Bb2 F3 A3 D4]"),
    # a bare D power-chord, the most "robot"
    "d5": ("[D2 D3 A3 D4]", "[D2 D3 A3 D4]"),
    # one low saw on D: dark monotone
    "d3": ("D3", "D3"),
}
CARRIERS = {
    # dark: saw + sub octave, closed filter
    "dark": "automate nv.waveform\n2\nautomate nv.osc2_pitch\n-12\nautomate nv.osc2_level\n0.5\n"
            "automate nv.detune\n4\nautomate nv.cutoff\n1600\nautomate nv.resonance\n0.4\n"
            "automate nv.sub\n0.3\n",
    # bright: saw + octave-up shimmer, open filter (the legibility reference)
    # mid: saw + sub octave, filter half open, darker than bright but keeps the F2 region
    "mid": "automate nv.waveform\n2\nautomate nv.osc2_pitch\n-12\nautomate nv.osc2_level\n0.45\n"
           "automate nv.detune\n4\nautomate nv.cutoff\n3000\nautomate nv.resonance\n0.3\nautomate nv.sub\n0.2\n",
    "bright": "automate nv.waveform\n2\nautomate nv.osc2_pitch\n12\nautomate nv.osc2_level\n0.4\n"
              "automate nv.detune\n3\nautomate nv.cutoff\n5000\nautomate nv.sub\n0\n",
}


def song_text(mode, dry, lyric, chord, carrier, dur=(1.5, 2.5), level=1.0, intonation=0.15):
    (w1, w2), (c1, c2) = LYRICS[lyric], CHORDS[chord]
    return f"""# take: vox_mode {mode}, vox_dry {dry}, lyric {lyric}, chord {chord}, carrier {carrier}
bpm 120
gate 0.95
tail 1.5
automate vox_mode
{mode}
automate vox_level
{level}
automate vox_dry
{dry}
automate vox_breath
0.1
automate vox_vibrato
0
automate vox_intonation
{intonation}
automate reverb_wet
0.16
automate reverb_decay
0.6
track nv vox vel=0.9
R:1 {c1}:{dur[0]}={w1} {c2}:{dur[1]}={w2}
{CARRIERS[carrier]}"""


PIPER_WAV = "qa/voice/neural_pairing_piper48k.wav"   # relative to the score folder


def song_text_wav(mode, dry, chord, carrier, level=1.0, wav=PIPER_WAV):
    """The vox route with a recorded modulator (`wav=`): the recording starts at the
    first vox note (beat 1, pinned by wav_at); the chord changes under the P closure
    of "pairing" (0.5 s into the recording)."""
    c1, c2 = CHORDS[chord]
    return f"""# take (wav modulator): vox_mode {mode}, vox_dry {dry}, chord {chord}, carrier {carrier}
bpm 120
gate 0.98
tail 1.5
automate vox_mode
{mode}
automate vox_level
{level}
automate vox_dry
{dry}
automate reverb_wet
0.16
automate reverb_decay
0.6
track nv vox wav={wav} wav_at=1
R:1 {c1}:1 {c2}:2
{CARRIERS[carrier]}"""


def render(name, text):
    os.makedirs(TAKES, exist_ok=True)
    sp = os.path.join(TAKES, name + ".song")
    wp = os.path.join(TAKES, name + ".wav")
    open(sp, "w").write(text)
    r = subprocess.run([C.PATINA, "--play", sp, "--render", wp], cwd=SCORE_DIR, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr + r.stdout)
    return wp


def say(name, phrase):
    wp = os.path.join(TAKES, name + ".wav")
    subprocess.run([C.PATINA, "--say", phrase, "--out", wp], cwd=SCORE_DIR, capture_output=True, check=True)
    return wp


def timbre(wp):
    x, sr = C.load_mono(wp)
    n = 8192
    w = np.hanning(n)
    f = np.fft.rfftfreq(n, 1 / sr)
    cen, inkey, tot = [], 0.0, 0.0
    keypcs = {2, 4, 5, 7, 9, 10, 0, 1, 11}   # D natural minor + B, C#
    for s in range(0, len(x) - n, n // 2):
        sp = np.abs(np.fft.rfft(x[s:s + n] * w))
        e = sp ** 2
        if e.sum() < 1e-9:
            continue
        cen.append((f * e).sum() / e.sum())
        m = (f > 60) & (f < 1200)
        idx = np.where(m)[0]
        l = sp[idx]
        pk = idx[1:-1][(l[1:-1] > l[:-2]) & (l[1:-1] >= l[2:]) & (l[1:-1] > l.max() * 0.1)]
        for i in pk:
            midi = 69 + 12 * np.log2(f[i] / 440)
            tot += e[i]
            inkey += e[i] * ((int(round(midi)) % 12) in keypcs)
    return float(np.median(cen)) if cen else float("nan"), inkey / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--grid", default="main")
    a = ap.parse_args()
    takes = []
    if a.grid == "main":
        takes.append(("say_plain", "say", "N-UH1-R-AH0-L P-EH1-R-IH0-NG"))
        takes.append(("say_crisp", "say", "N:110-UH1-R-AH0-L:90 P:60@1.5-EH1-R-IH0-NG:170"))
        for mode, dry, lyric, chord, carrier in itertools.product(
                (0, 1, 2, 3), (0.0, 0.2), ("plain", "crisp"), ("dm_bb", "d3"), ("dark",)):
            takes.append((f"m{mode}_dry{int(dry * 100):02d}_{lyric}_{chord}_{carrier}", "song",
                          song_text(mode, dry, lyric, chord, carrier)))
    elif a.grid == "piper":   # the borrowed voice (piper en_US-ryan) through the vocoder
        for mode, dry, chord, carrier in itertools.product((0, 1, 2, 3), (0.0, 0.15), ("dm_bb", "d5"),
                                                            ("dark", "mid")):
            takes.append((f"wav_m{mode}_dry{int(dry * 100):02d}_{chord}_{carrier}", "song",
                          song_text_wav(mode, dry, chord, carrier)))
    elif a.grid == "spell":   # the raw formant voice alone: which spelling reads?
        for lyric in ("plain", "crisp", "nyoo", "nuw", "nur", "pey", "nuw_crisp", "nyu_pey", "nur_pey",
                      "slow", "slow_nuw"):
            for inton in (0.15, 0.6):
                takes.append((f"dry_{lyric}_i{int(inton * 100)}", "song",
                              song_text(1, 1.0, lyric, "d3", "bright", level=0.0, intonation=inton)))
    else:   # refinement grid around the best spelling
        for mode, dry, lyric, chord, carrier, dur in itertools.product(
                (0, 1, 3), (0.2, 0.4), ("nlong",), ("dm_bb", "d5"), ("dark", "mid", "bright"),
                ((1.5, 2.5),)):
            takes.append((f"r_m{mode}_dry{int(dry * 100):02d}_{lyric}_{chord}_{carrier}_{int(dur[0] * 10)}", "song",
                          song_text(mode, dry, lyric, chord, carrier, dur)))
    if a.only:
        keep = set(a.only.split(","))
        takes = [t for t in takes if t[0] in keep]
    resf = os.path.join(TAKES, "results.json")
    results = json.load(open(resf)) if os.path.exists(resf) else {}
    for name, kind, payload in takes:
        wp = say(name, payload) if kind == "say" else render(name, payload)
        text, lp, ns = asr.transcribe(wp, a.model)
        sc = asr.score(text, "neural pairing")
        cen, inkey = timbre(wp)
        results[name] = {"text": text, "score": sc, "avg_logprob": lp, "centroid_hz": cen, "in_key": inkey,
                         "payload": payload}
        print(f"{sc:5.2f} lp {lp:+.2f} cen {cen:6.0f} Hz inkey {inkey:4.2f}  {name:<40} {text!r}", flush=True)
        json.dump(results, open(resf, "w"), indent=1)


if __name__ == "__main__":
    main()
