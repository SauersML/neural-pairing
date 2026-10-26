#!/usr/bin/env python3
"""Formant-preserving pitch shift + time stretch of a speech recording (TD-PSOLA).

    python3 qa/voice/psola.py IN.wav OUT.wav --semitones -2 --map 0:0,0.49:0.5243,0.573:0.625 --rate 1.07

Pitch marks are placed one period apart in voiced frames (period from librosa.pyin), snapped to
the local waveform peak; unvoiced frames get marks every 5 ms. Each synthesis mark takes the
two-period Hann grain around the analysis mark nearest to its mapped input time, so the
spectral envelope (formants) stays put while the period changes by 2**(semitones/12) and the
timing follows the time map.

--map lists input:output anchors (seconds); after the last anchor the output runs --rate times
slower than the input. Between anchors the map is linear.
"""
import argparse

import librosa
import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--semitones", type=float, default=0.0)
    ap.add_argument("--map", default="0:0")
    ap.add_argument("--rate", type=float, default=1.0)
    a = ap.parse_args()

    x, sr = sf.read(a.inp, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    n = len(x)
    hop = 120
    f0, vf, _ = librosa.pyin(x, fmin=60, fmax=320, sr=sr, frame_length=2048, hop_length=hop)
    ft = np.arange(len(f0)) * hop / sr
    f0i = np.where(vf & ~np.isnan(f0), f0, 0.0)

    # analysis marks: one per period, snapped to the SIGNED peak of a low-passed copy (consistent
    # glottal epochs; snapping to |x| flips between positive and negative peaks)
    from scipy.signal import butter, sosfiltfilt
    xl = sosfiltfilt(butter(4, 900, fs=sr, output="sos"), x)
    marks, periods = [], []
    t = 0.0
    prev_voiced = False
    while t < n / sr:
        k = min(len(f0i) - 1, int(round(t * sr / hop)))
        if f0i[k] > 0:
            P = sr / f0i[k]
            c = int(t * sr); w = int(P / 4) if prev_voiced else int(P / 2)
            lo, hi = max(0, c - w), min(n, c + w + 1)
            seg = xl[lo:hi]
            c = lo + int(np.argmax(seg)) if len(seg) else c
            if marks and c <= marks[-1] + P / 2:
                c = int(marks[-1] + P)
            marks.append(c); periods.append(P)
            t = (c + P) / sr
            prev_voiced = True
        else:
            P = 0.005 * sr
            marks.append(int(t * sr)); periods.append(P)
            t += 0.005
            prev_voiced = False
    marks = np.array(marks); periods = np.array(periods)
    voiced = np.array([f0i[min(len(f0i) - 1, int(round(m / hop)))] > 0 for m in marks])

    # time map: output -> input
    anc = [tuple(map(float, p.split(":"))) for p in a.map.split(",")]
    ti, to = zip(*anc)
    end_in = n / sr
    end_out = to[-1] + (end_in - ti[-1]) * a.rate
    ti = list(ti) + [end_in]; to = list(to) + [end_out]
    alpha = 2 ** (a.semitones / 12.0)

    y = np.zeros(int(end_out * sr) + 4096)
    norm = np.zeros_like(y)
    ts = 0.0
    while ts < end_out:
        tin = np.interp(ts, to, ti)
        j = int(np.argmin(np.abs(marks / sr - tin)))
        m, P = marks[j], periods[j]
        L = int(P)
        lo, hi = m - L, m + L
        if lo >= 0 and hi <= n:
            g = x[lo:hi] * np.hanning(2 * L)
            s = int(ts * sr) - L
            if s >= 0 and s + 2 * L <= len(y):
                y[s:s + 2 * L] += g
                norm[s:s + 2 * L] += np.hanning(2 * L)
        step = P / alpha if voiced[j] else P
        ts += step / sr
    # keep the analysis loudness (OLA of Hann grains at varying density)
    sm = np.convolve(norm, np.ones(256) / 256, mode="same")
    y = y / np.maximum(sm, 0.5)
    y = y[:int(end_out * sr)]
    y *= np.abs(x).max() / (np.abs(y).max() + 1e-12)
    sf.write(a.out, y.astype(np.float32), sr, subtype="FLOAT")
    print(f"{a.inp}: {end_in:.3f} s -> {end_out:.3f} s ({100 * (end_out / end_in - 1):+.1f} %), "
          f"{a.semitones:+.2f} st, {len(marks)} marks")


if __name__ == "__main__":
    main()
