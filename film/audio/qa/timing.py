#!/usr/bin/env python3
"""Objective timing analysis for a Patina score.

    python3 qa/timing.py SONG.song [MIX.wav] [--stems DIR] [--rerender]
                         [--tracks a,b] [--csv out.csv] [--grid 0.25]

For every track it compares what was WRITTEN with what SOUNDS:

* written grid: each note-on (exact times from `patina --export-events`)
  is checked against the 16th grid (0.25 beat; 0.125 s at 120 bpm) and the
  16th-triplet grid. Notes on neither are listed as "off-grid" with the
  song construct that put them there (fractional durations / rests).
* start offset: the onset in the track's own stem (spectral flux; for
  pitched tracks restricted to the note's own partials, so a ringing pad
  or a previous note cannot mask it) minus the written note-on.
* perceived onset: when the note's energy reaches -6 dB of its peak (a
  P-centre proxy) and the 10 %->90 % attack time. A slow attack makes a
  note FEEL late even when it starts on time.
* sampler tracks that play a bounce of another song (`sample=fx2.wav`
  with fx2.song beside it): that song's events are the intended times,
  and the deck's playback offset is measured by cross-correlating the
  stem's onset envelope with the source wav.

Flags: systematic offset (|median start| > 10 ms after removing the bus
latency measured on the drums), smeared (start IQR > 12 ms), late feel
(median -6 dB point > 40 ms), and off-grid writing.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

HOP = 120           # 2.5 ms at 48 kHz
NFFT = 2048


def stft_log(x, sr):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.zeros(NFFT), x, np.zeros(NFFT)])
    fr = sliding_window_view(pad, NFFT)[::HOP]
    w = np.hanning(NFFT)
    S = np.empty((fr.shape[0], NFFT // 2 + 1), np.float32)
    for i in range(0, fr.shape[0], 2000):           # chunked: bounded memory
        S[i:i + 2000] = np.abs(np.fft.rfft(fr[i:i + 2000] * w, axis=1))
    t = (np.arange(S.shape[0]) * HOP + NFFT / 2 - NFFT) / sr   # frame centre times
    return t, S


def partial_mask(freqs, notes, hmax=8):
    m = np.zeros(len(freqs), bool)
    for q in notes:
        f0 = float(C.midi_to_hz(q))
        for h in range(1, hmax + 1):
            if h * f0 > 8000:
                break
            m |= np.abs(freqs - h * f0) < max(0.03 * h * f0, 15.0)
    return m


def measure_group(t_on, S, tt, freqs, mask, t_lo, t_hi):
    """Onset (max spectral flux in [t_lo, t_hi]) and envelope landmarks."""
    sel = mask if mask is not None else slice(None)
    i0 = max(1, np.searchsorted(tt, t_lo))
    i1 = min(len(tt) - 1, np.searchsorted(tt, t_hi))
    if i1 - i0 < 3:
        return None
    L = np.log1p(100.0 * S[i0 - 1:i1 + 1][:, sel])
    flux = np.maximum(L[1:] - L[:-1], 0).sum(axis=1)
    flux = np.convolve(flux, [0.25, 0.5, 0.25], mode="same")
    j = int(np.argmax(flux))
    if flux[j] <= 0:
        return None
    strength = float(flux[j] / (np.median(flux) + 1e-9))
    onset = tt[i0 + j]
    # envelope: band energy from shortly before the note to the next note
    e_lo = np.searchsorted(tt, t_on - 0.03)
    e_hi = min(len(tt), np.searchsorted(tt, min(t_hi + 0.6, t_hi + 0.6)))
    E = (S[e_lo:e_hi][:, sel].astype(np.float64) ** 2).sum(axis=1)
    E = np.convolve(E, np.ones(3) / 3, mode="same")
    base = E[: max(1, np.searchsorted(tt[e_lo:e_hi], t_on - 0.005))].mean()
    pk = int(np.argmax(E))
    peak = E[pk]
    if peak <= base * 1.5:
        return {"onset": onset, "strength": strength, "p6": None, "rise": None, "tpeak": None}
    amp = np.sqrt(np.maximum(E - base, 0))
    apk = amp[pk]
    k0 = np.searchsorted(tt[e_lo:e_hi], t_on - 0.01)

    def first(level):
        idx = np.where(amp[k0:pk + 1] >= level * apk)[0]
        return tt[e_lo + k0 + idx[0]] if len(idx) else None
    t10, t50, t90 = first(0.1), first(0.5), first(0.9)
    return {"onset": onset, "strength": strength, "p6": t50, "rise": (t90 - t10) if t10 and t90 else None,
            "tpeak": tt[e_lo + pk]}


_BIAS = None


def flux_bias(sr=48000):
    """The flux detector fires when an onset enters the analysis window, so
    it reads early by a constant. Measured once on synthetic onsets (2 ms
    attack, tones and noise) and subtracted from every start offset."""
    global _BIAS
    if _BIAS is None:
        x = np.zeros(sr * 4)
        ts = [0.5, 1.25, 2.0, 2.75, 3.5]
        rng = np.random.default_rng(1)
        n = int(0.3 * sr)
        tt = np.arange(n) / sr
        env = np.minimum(tt / 0.002, 1) * np.exp(-tt / 0.15)
        for i, t in enumerate(ts):
            sig = (np.sin(2 * np.pi * 146.83 * tt) + 0.5 * np.sin(2 * np.pi * 293.66 * tt)) if i % 2 == 0 \
                else rng.standard_normal(n)
            x[int(t * sr):int(t * sr) + n] += 0.3 * env * sig
        rows = analyse_track("cal", x, sr, [(t, [50]) for t in ts], "kit", 0.5, 0.25, bias=0.0)
        _BIAS = float(np.median([r["onset"] - r["t"] for r in rows]))
    return _BIAS


def grid_status(t, spb, grid_beats):
    g = grid_beats * spb
    d = t - round(t / g) * g
    if abs(d) <= 0.004:
        return "grid", d
    gt = spb / 6.0   # 16th triplet
    if abs(t - round(t / gt) * gt) <= 0.004:
        return "triplet", d
    g32 = g / 2      # 32nds (fills, shards)
    if abs(t - round(t / g32) * g32) <= 0.004:
        return "32nd", d
    return "OFF", d


def onset_envelope(x, sr):
    tt, S = stft_log(x, sr)
    L = np.log1p(100.0 * S)
    flux = np.r_[0, np.maximum(L[1:] - L[:-1], 0).sum(axis=1)]
    return tt, flux


def xcorr_offset(a, b, sr, max_lag_s=0.15):
    """Lag (s) of b relative to a from their onset envelopes (positive = b late)."""
    ta, fa = onset_envelope(a, sr)
    tb, fb = onset_envelope(b, sr)
    n = min(len(fa), len(fb))
    fa, fb = fa[:n] - fa[:n].mean(), fb[:n] - fb[:n].mean()
    L = int(max_lag_s * sr / HOP)
    lags = np.arange(-L, L + 1)
    c = [np.dot(fa[max(0, -k):n - max(0, k)], fb[max(0, k):n - max(0, -k)]) for k in lags]
    k = int(lags[int(np.argmax(c))])
    return k * HOP / sr


def analyse_track(name, x, sr, groups, kind, spb, grid_beats, bias=None):
    if bias is None:
        bias = flux_bias(sr)
    tt, S = stft_log(x, sr)
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    times = [g[0] for g in groups]
    rows = []
    for k, (t, notes) in enumerate(groups):
        prev_t = times[k - 1] if k else -1.0
        next_t = times[k + 1] if k + 1 < len(times) else t + 2.0
        lo = max(t - 0.04, (prev_t + t) / 2)
        hi = min(t + 0.08, (t + next_t) / 2 if next_t - t < 0.16 else t + 0.08)
        mask = None
        if kind == "synth" or kind == "vox":
            mask = partial_mask(freqs, notes)
            if mask.sum() < 3:
                mask = None
        r = measure_group(t, S, tt, freqs, mask, lo, max(hi, t + 0.02))
        st, gd = grid_status(t, spb, grid_beats)
        row = {"track": name, "t": t, "notes": notes, "grid": st, "grid_dev": gd}
        if r:
            row.update(r)
            row["onset"] = r["onset"] - bias
            row["start"] = row["onset"] - t
            row["p6off"] = (r["p6"] - t) if r["p6"] is not None else None
        rows.append(row)
    return rows


def snap(t, spb, grid_beats, late_ok=0.0755, early_ok=0.0455):
    """Nearest grid beat for a picture-synced hit. Audio may trail the
    picture more than lead it (ITU-R BT.1359 detectability: about 45 ms
    early, 125 ms late), so prefer the grid point at or after the cue
    when it is within late_ok."""
    g = grid_beats * spb
    lo = np.floor(t / g + 1e-9) * g
    hi = lo + g
    if hi - t <= late_ok:
        return hi / spb
    if t - lo <= early_ok:
        return lo / spb
    return (hi if hi - t < t - lo else lo) / spb


def group_notes(notes, eps=0.004):
    groups = []
    for n in sorted(notes, key=lambda n: n["on"]):
        if groups and n["on"] - groups[-1][0] < eps:
            groups[-1][1].append(n["note"])
        else:
            groups.append([n["on"], [n["note"]]])
    return [(t, ns) for t, ns in groups]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("wav", nargs="?", default="-", help="the mix (optional; for the global bus latency)")
    ap.add_argument("--stems")
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--tracks")
    ap.add_argument("--grid", type=float, default=0.25, help="grid in beats (default 0.25 = 16ths)")
    ap.add_argument("--csv")
    ap.add_argument("--list", type=int, default=12, help="off-grid notes listed per track")
    a = ap.parse_args()

    song = os.path.abspath(a.song)
    cache = a.stems or C.default_cache(song)
    info = C.parse_song_text(song)
    spb = info["spb"]
    ev = C.export_events(song, cache)
    C.render_stems(song, cache, a.rerender)
    notes = C.notes_by_track(ev)
    want = set(a.tracks.split(",")) if a.tracks else None
    print(f"# Timing report: {os.path.basename(song)}  ({info['bpm']:.0f} bpm; grid {a.grid} beat = "
          f"{a.grid * spb * 1000:.0f} ms; 16th-triplet = {spb / 6 * 1000:.1f} ms)")

    all_rows = []
    per_track = {}
    sampler_reports = []
    for name, tr in info["tracks"].items():
        if want and name not in want:
            continue
        stem = os.path.join(cache, name + ".wav")
        if not os.path.exists(stem):
            continue
        x, sr = C.load_mono(stem)
        if tr["kind"] == "sampler":
            src = tr["opts"].get("sample")
            src_path = os.path.join(C.song_cwd(song), src)
            src_song = os.path.splitext(src_path)[0] + ".song"
            trig = notes.get(name, [])
            rep = {"track": name, "sample": src, "triggers": [(n["on"], n["note"]) for n in trig]}
            if os.path.exists(src_path):
                y, sr2 = C.load_mono(src_path)
                t0 = trig[0]["on"] if trig else 0.0
                yy = np.concatenate([np.zeros(int(t0 * sr)), y])[: len(x)]
                rep["xcorr"] = xcorr_offset(yy, x, sr)
            if os.path.exists(src_song):
                sev = C.export_events(src_song, os.path.join(cache, "_src_" + name))
                snotes = [n for v in C.notes_by_track(sev).values() for n in v]
                t0 = trig[0]["on"] if trig else 0.0
                for n in snotes:
                    n["on"] += t0
                    n["off"] += t0
                # source-song onsets measured broadband in this stem
                g = group_notes(snotes)
                rows = analyse_track(name, x, sr, g, "kit", spb, a.grid)
                rows = [r for r in rows if r.get("strength", 0) > 4]
                rep["rows"] = rows
                # and the source wav's own render latency vs its song
                if os.path.exists(src_path):
                    y0 = np.concatenate([np.zeros(int(t0 * sr)), y])
                    rr = analyse_track(name, y0, sr, g, "kit", spb, a.grid)
                    rr = [r for r in rr if r.get("strength", 0) > 4]
                    rep["src_rows"] = rr
            sampler_reports.append(rep)
            continue
        if name not in notes:
            continue
        g = group_notes(notes[name])
        if tr["kind"] == "kit":
            # every kit track plays the one 909 board: measure all kit notes on this stem once
            kitnotes = [n for k, t in info["tracks"].items() if t["kind"] == "kit" for n in notes.get(k, [])]
            if any(r["track"] == "kit" for r in all_rows):
                continue
            g = group_notes(kitnotes)
            rows = analyse_track("kit", x, sr, g, "kit", spb, a.grid)
            name = "kit"
        else:
            rows = analyse_track(name, x, sr, g, tr["kind"], spb, a.grid)
        for r in rows:
            r["section"] = C.section_of(r["t"], info["sections"])
        per_track[name] = rows
        all_rows += rows

    # bus latency: the median start offset of the drums (fast transients, broadband)
    kit = [r["start"] for r in per_track.get("kit", []) if r.get("start") is not None and r["strength"] > 4]
    bus = float(np.median(kit)) if kit else 0.0
    print(f"# onset detector calibrated on synthetic onsets: frame bias {flux_bias() * 1000:+.1f} ms removed")
    print(f"# bus latency (median kick/snare/rim onset offset on the kit stem): {bus * 1000:+.1f} ms "
          f"(the master tape's 6 ms transport delay plus the drum attacks); offsets below are RELATIVE to it")

    print("\n## Per track: start offset = sounding onset - written note-on - bus latency (ms)")
    print(f"{'track':<7} {'onsets':>6} {'median':>7} {'IQR':>6} {'p10':>6} {'p90':>6} {'-6dB pt':>8} "
          f"{'rise10-90':>9} {'off-grid':>9}  flags")
    summary = {}
    for name, rows in per_track.items():
        st = np.array([r["start"] - bus for r in rows if r.get("start") is not None and r["strength"] > 2.5])
        p6 = np.array([r["p6off"] - bus for r in rows if r.get("p6off") is not None])
        rise = np.array([r["rise"] for r in rows if r.get("rise") is not None])
        og = [r for r in rows if r["grid"] == "OFF"]
        tri = [r for r in rows if r["grid"] == "triplet"]
        d32 = [r for r in rows if r["grid"] == "32nd"]
        if len(st) == 0:
            continue
        med, iqr = np.median(st) * 1000, (np.percentile(st, 75) - np.percentile(st, 25)) * 1000
        flags = []
        if abs(med) > 10:
            flags.append("SYSTEMATIC OFFSET")
        if iqr > 12:
            flags.append("SMEARED")
        if len(p6) and np.median(p6) * 1000 > 40:
            flags.append("SLOW ATTACK (feels late)")
        if og:
            flags.append(f"{len(og)} WRITTEN OFF-GRID")
        if tri:
            flags.append(f"{len(tri)} on 16th triplets")
        if d32:
            flags.append(f"{len(d32)} on 32nds")
        p6m = f"{np.median(p6) * 1000:+8.0f}" if len(p6) else f"{'-':>8}"
        rm = f"{np.median(rise) * 1000:9.0f}" if len(rise) else f"{'-':>9}"
        print(f"{name:<7} {len(st):6d} {med:+7.1f} {iqr:6.1f} {np.percentile(st, 10) * 1000:+6.1f} "
              f"{np.percentile(st, 90) * 1000:+6.1f} {p6m} {rm} {len(og):4d}/{len(rows):<4d}  {', '.join(flags)}")
        summary[name] = (med, iqr)

    # ---- written off-grid notes, by section
    print(f"\n## Written placement: note-ons off the {a.grid}-beat grid (and off the 16th-triplet grid)")
    secnames = [s[0] for s in info["sections"]]
    print(f"{'section':<9}" + "".join(f"{n[:6]:>7}" for n in per_track))
    for s in secnames:
        line = f"{s:<9}"
        for n, rows in per_track.items():
            rr = [r for r in rows if r["section"] == s]
            o = sum(r["grid"] == "OFF" for r in rr)
            line += f"{(f'{o}/{len(rr)}' if rr else '.'):>7}"
        print(line)
    for name, rows in per_track.items():
        og = [r for r in rows if r["grid"] == "OFF"]
        if not og:
            continue
        print(f"  {name}: " + ", ".join(
            f"{r['t']:.3f}s ({r['section']}, beat {r['t'] / spb:.2f}, {r['grid_dev'] * 1000:+.0f} ms off; "
            f"snap -> beat {snap(r['t'], spb, a.grid):.2f})"
            for r in og[:a.list]) + (" ..." if len(og) > a.list else ""))

    # ---- worst individual sounding offsets
    bad = [r for r in all_rows if r.get("start") is not None and r["strength"] > 2.5 and abs(r["start"] - bus) > 0.02]
    print(f"\n## Sounding onsets more than 20 ms from the written time ({len(bad)})")
    for r in sorted(bad, key=lambda r: -abs(r["start"] - bus))[:25]:
        print(f"  {r['t']:7.3f}s {r['section']:<9} {r['track']:<6} {','.join(C.note_name(q) for q in r['notes']):<14} "
              f"start {1000 * (r['start'] - bus):+6.1f} ms   -6dB {('%+.0f ms' % (1000 * (r['p6off'] - bus))) if r.get('p6off') is not None else '-'}")

    # ---- sampler tracks
    for rep in sampler_reports:
        print(f"\n## Sampler track '{rep['track']}' (sample={rep['sample']}, triggers at "
              f"{', '.join(f'{t:.3f}s/{C.note_name(n)}' for t, n in rep['triggers'])})")
        if "xcorr" in rep:
            print(f"  deck playback offset vs the source wav (onset-envelope cross-correlation): "
                  f"{rep['xcorr'] * 1000:+.1f} ms (bus latency {bus * 1000:+.1f} ms)")
        if rep.get("src_rows"):
            s = np.array([r["start"] for r in rep["src_rows"]])
            print(f"  the source wav vs its own song's note-ons: median {np.median(s) * 1000:+.1f} ms over {len(s)} clear onsets")
        if rep.get("rows"):
            s = np.array([r["start"] for r in rep["rows"]])
            print(f"  this stem vs the source song's note-ons: median {np.median(s) * 1000:+.1f} ms, "
                  f"IQR {(np.percentile(s, 75) - np.percentile(s, 25)) * 1000:.1f} ms over {len(s)} clear onsets; "
                  f"relative to the bus {(np.median(s) - bus) * 1000:+.1f} ms")
            for r in rep["rows"]:
                print(f"    {r['t']:7.3f}s start {1000 * r['start']:+6.1f} ms (flux x{r['strength']:.0f})")

    out = a.csv or os.path.join(cache, "timing_onsets.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "section", "t", "notes", "grid", "grid_dev_ms", "start_ms", "start_rel_bus_ms",
                    "minus6dB_ms", "rise_ms", "flux_strength"])
        for r in all_rows:
            w.writerow([r["track"], r.get("section", ""), f"{r['t']:.4f}", " ".join(C.note_name(q) for q in r["notes"]),
                        r["grid"], f"{r['grid_dev'] * 1000:.1f}",
                        f"{r['start'] * 1000:.1f}" if r.get("start") is not None else "",
                        f"{(r['start'] - bus) * 1000:.1f}" if r.get("start") is not None else "",
                        f"{r['p6off'] * 1000:.1f}" if r.get("p6off") is not None else "",
                        f"{r['rise'] * 1000:.1f}" if r.get("rise") is not None else "",
                        f"{r.get('strength', 0):.1f}"])
    print(f"\nper-onset CSV: {out}")


if __name__ == "__main__":
    main()
