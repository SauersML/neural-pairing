#!/usr/bin/env python3
"""Transcribe WAVs with faster-whisper (CPU, int8) and score them against a target phrase.

    python3 qa/voice/asr.py TAKE.wav [TAKE2.wav ...] [--model small.en] [--target "neural pairing"]

Audio is decoded with soundfile and resampled to 16 kHz here (the PyAV decoder bundled
with faster-whisper is not used). The score is a normalized character similarity
(difflib ratio, 1.0 = exact) after lower-casing and stripping punctuation; the
"no_speech" probability and mean log-prob of the decoding are printed too.
"""
import argparse
import difflib
import json
import re
import sys

import numpy as np
import soundfile as sf

_MODELS = {}


def load16k(path):
    import librosa
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    pk = np.max(np.abs(x)) + 1e-9
    return (x / pk * 0.5).astype(np.float32)


def norm(t):
    return re.sub(r"[^a-z ]", "", t.lower()).strip()


def transcribe(path, model="small.en", beam=5, prompt=None):
    from faster_whisper import WhisperModel
    if model not in _MODELS:
        _MODELS[model] = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=2)
    m = _MODELS[model]
    segs, info = m.transcribe(load16k(path), beam_size=beam, language="en", initial_prompt=prompt,
                              condition_on_previous_text=False, vad_filter=False)
    segs = list(segs)
    text = " ".join(s.text.strip() for s in segs).strip()
    lp = float(np.mean([s.avg_logprob for s in segs])) if segs else float("nan")
    ns = float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0
    return text, lp, ns


def score(text, target):
    a, b = norm(text), norm(target)
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--target", default="neural pairing")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = []
    for w in a.wavs:
        text, lp, ns = transcribe(w, a.model)
        sc = score(text, a.target)
        out.append({"wav": w, "text": text, "score": sc, "avg_logprob": lp, "no_speech": ns})
        print(f"{sc:5.2f}  lp {lp:+.2f}  {w}: {text!r}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
