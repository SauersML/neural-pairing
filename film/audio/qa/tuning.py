#!/usr/bin/env python3
"""Objective tuning analysis for a Patina score.

    python3 qa/tuning.py SONG.song MIX.wav [--stems DIR] [--rerender]
                         [--tracks a,b] [--csv out.csv] [--tol 12]

1. Renders per-track stems with `patina --play SONG --render-stems DIR`
   (cached in qa/stems/<song>/ and re-rendered when the song, its patches
   or its samples change) and exports the exact note events.
2. For every written note of every pitched stem it measures the sounding
   pitch frame by frame (40 ms hop) by NOTE-INFORMED HARMONIC PEAK PICKING:
   Hann window sized to the note's register, 4x zero padding, the peak
   nearest each harmonic h*f_written (h = 1..8, +-60 cents) refined by
   parabolic interpolation of the log spectrum, skipping harmonics that
   collide with partials of other notes sounding on the same track (chord
   members, release tails). The frame value is the magnitude-weighted
   median of the harmonic deviations. This works on chords and pads,
   where a blind f0 tracker cannot.
3. Monophonic stems also get a blind pYIN f0 trace (librosa.pyin) and are
   scored against the nearest equal-tempered pitch (A4 = 440 Hz),
   independent of the score.
4. Master pitch reference: spectral peaks of the mix (and the
   duration-weighted mean over every measured note) against A440.

A note is FLAGGED when its median deviation over the sustained part is
beyond +-tol cents (default 12), or when it stays beyond +-tol for 150 ms
or more (a glide or bend that is heard as out of tune). Output: per-note
CSV, per-track and per-section tables, flagged notes, and the patch /
automation settings that move pitch on each flagged track.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

HOP_S = 0.04
SUSTAIN_FLAG_S = 0.15
TAIL_S = 1.5             # echo / reverb ring after a note's release
PITCH_PARAMS = ["detune", "unison", "unison_detune", "glide", "mono", "lfo_pitch",
                "lfo_rate", "osc2_pitch", "osc3_pitch", "osc_fm", "ring", "sync",
                "circuit", "attack"]
TRACK_PITCH_AUTOMATION = {"PitchShift", "Glide", "LfoPitch", "Detune", "UnisonDetune",
                          "PitchBendSemis", "ModWheel", "SmpPitch"}


def note_spectrum_frames(x, sr, t0, t1, f0):
    """Frame centres and windowed magnitude spectra between t0 and t1.

    The window is as long as the note allows (up to 0.68 s): sustained
    pitch is steady, and only a long window separates a note's partials
    from a chord neighbour a semitone away or an echo of the previous
    note. At least ~10 periods when the note is long enough."""
    span = int((t1 - t0) * sr)
    if span < 1024:
        return None
    n = int(2 ** np.floor(np.log2(max(1024, min(32768, 0.7 * span)))))
    n = max(n, int(2 ** np.ceil(np.log2(min(span, 10.0 * sr / f0)))) if span >= 2048 else n)
    n = min(n, 2 ** int(np.floor(np.log2(span))))
    nfft = 4 * n
    win = np.hanning(n)
    hop = max(int(HOP_S * sr), n // 4)
    a0 = int(t0 * sr)
    starts = list(range(a0, a0 + span - n + 1, hop)) or [a0]
    frames = []
    for s in starts:
        seg = x[s:s + n]
        if len(seg) < n:
            break
        spec = np.abs(np.fft.rfft(seg * win, nfft))
        frames.append(((s + n / 2) / sr, spec, n))
    return frames, nfft


def harmonic_deviation(spec, nfft, n, sr, f0, others, floor_db):
    """Magnitude-weighted median cents deviation of the partials of f0."""
    lspec = 20 * np.log10(spec + 1e-12)
    devs, wts = [], []
    binhz = sr / nfft
    res = 2.5 * sr / n            # Hann main lobe (2 bins of n) plus margin, Hz
    for h in range(1, 9):
        fh = h * f0
        if fh > min(6000.0, sr / 2 - 100):
            break
        # skip partials an unresolved partial of another sounding note
        # (chord member, release, echo or reverb tail) would bias; exactly
        # coincident partials (octaves, within 15 cents) are left in
        clash = False
        for fo in others:
            for k in range(1, 16):
                fk = k * fo
                if fk > fh + res:
                    break
                if abs(fk - fh) < res and abs(1200 * np.log2(fk / fh)) > 15:
                    clash = True
                    break
            if clash:
                break
        if clash:
            continue
        lo = int(fh * 2 ** (-50 / 1200) / binhz)
        hi = int(np.ceil(fh * 2 ** (50 / 1200) / binhz))
        if hi - lo < 3 or hi + 9 >= len(spec):
            continue
        i = lo + int(np.argmax(lspec[lo:hi + 1]))
        # a genuine peak: the maximum of +-2 analysis bins (8 fft bins)
        if i <= lo or i >= hi or lspec[i] < floor_db or lspec[i] < lspec[max(0, i - 8):i + 9].max():
            continue
        a, b, c = lspec[i - 1], lspec[i], lspec[i + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den < 0 else 0.0
        fpk = (i + d) * binhz
        devs.append(1200 * np.log2(fpk / fh))
        wts.append(10 ** (b / 20))
    if not devs:
        return None
    devs, wts = np.array(devs), np.array(wts)
    o = np.argsort(devs)
    cw = np.cumsum(wts[o])
    return float(devs[o][np.searchsorted(cw, cw[-1] / 2)])


def analyse_track(x, sr, notes, release_s, tol):
    peak_db = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
    rows = []
    for i, n in enumerate(notes):
        f0 = float(C.midi_to_hz(n["note"]))
        dur = n["off"] - n["on"]
        if dur < 0.09:
            continue
        # sustained part: past the attack transient, up to note-off or the
        # next note of this track (mono lines, legato)
        nxt = min([m["on"] for m in notes[i + 1:] if m["on"] > n["on"] + 1e-4] + [n["off"]])
        t0 = n["on"] + min(0.05, 0.25 * dur)
        t1 = min(n["off"], nxt)
        if t1 - t0 < 0.05:
            continue
        r = note_spectrum_frames(x, sr, t0, t1, f0)
        if r is None:
            continue
        frames, nfft = r
        # interferers: other notes of this track sounding, releasing, or
        # still ringing in the echo / reverb (TAIL_S after release)
        others_all = [m for m in notes if m is not n and m["on"] < t1
                      and m["off"] + release_s + TAIL_S > t0 and m["note"] != n["note"]]
        fr_t, fr_d = [], []
        for tc, spec, nwin in frames:
            half = nwin / 2 / sr
            others = sorted({float(C.midi_to_hz(m["note"])) for m in others_all
                             if m["on"] < tc + half and m["off"] + release_s + TAIL_S > tc - half})
            d = harmonic_deviation(spec, nfft, nwin, sr, f0, others, peak_db - 70)
            if d is not None:
                fr_t.append(tc)
                fr_d.append(d)
        if len(fr_d) == 0:
            continue
        fr_d = np.array(fr_d)
        med = float(np.median(fr_d))
        # the run test uses a 5-frame median-filtered trace: one stray
        # frame is not "sustained"
        if len(fr_d) >= 5:
            from scipy.signal import medfilt
            run_d = medfilt(fr_d, 5)
        else:
            run_d = fr_d
        # longest run beyond tolerance (frames are HOP_S apart)
        run = best = 0
        frame_dt = (fr_t[1] - fr_t[0]) if len(fr_t) > 1 else HOP_S
        for d in run_d:
            run = run + 1 if abs(d) > tol else 0
            best = max(best, run)
        long_off = best * frame_dt
        if len(fr_d) == 1:
            long_off = (t1 - t0) if abs(med) > tol else 0.0
        heard = n["note"] + med / 100.0
        et = (heard - round(heard)) * 100.0
        rows.append({"on": n["on"], "off": n["off"], "note": n["note"], "vel": n["vel"],
                     "dev": med, "spread": float(np.percentile(fr_d, 90) - np.percentile(fr_d, 10)),
                     "first": float(fr_d[0]), "worst": float(fr_d[np.argmax(np.abs(fr_d))]),
                     "et_dev": et, "frames": len(fr_d), "off_tol_s": long_off,
                     "flag": abs(med) > tol or long_off >= SUSTAIN_FLAG_S,
                     "trace": list(zip(fr_t, fr_d))})
    return rows


def pyin_trace(x, sr, notes):
    """Blind f0 (pYIN) over the track's sounding spans: nearest-ET deviation."""
    import librosa
    y = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=22050)
    lo = max(30.0, float(C.midi_to_hz(min(n["note"] for n in notes))) / 1.5)
    hi = min(2000.0, float(C.midi_to_hz(max(n["note"] for n in notes))) * 1.5)
    f0, vflag, vprob = librosa.pyin(y, fmin=lo, fmax=hi, sr=22050, frame_length=4096,
                                    hop_length=512, resolution=0.02)
    t = librosa.times_like(f0, sr=22050, hop_length=512)
    active = np.zeros_like(t, bool)
    for n in notes:
        active |= (t >= n["on"] + 0.05) & (t <= n["off"])
    ok = vflag & active & np.isfinite(f0)
    m = C.hz_to_midi(f0[ok])
    et = (m - np.round(m)) * 100
    return t[ok], et


def master_reference(mix_path):
    """Global tuning offset of the mix: magnitude-weighted circular mean of
    spectral-peak deviations from the A440 grid, plus librosa's estimate."""
    import librosa
    x, sr = C.load_mono(mix_path)
    n = 16384
    hop = 8192
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(4 * n, 1 / sr)
    band = (freqs > 70) & (freqs < 2500)
    angles, weights = [], []
    for s in range(0, len(x) - n, hop):
        spec = np.abs(np.fft.rfft(x[s:s + n] * win, 4 * n))
        l = 20 * np.log10(spec + 1e-12)
        idx = np.where(band)[0]
        lb = l[idx]
        thr = lb.max() - 30
        pk = idx[1:-1][(lb[1:-1] > lb[:-2]) & (lb[1:-1] >= lb[2:]) & (lb[1:-1] > thr)]
        for i in pk:
            a, b, c = l[i - 1], l[i], l[i + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den < 0 else 0
            f = (i + d) * sr / (4 * n)
            cents = 1200 * np.log2(f / 440.0)
            angles.append(2 * np.pi * cents / 100.0)
            weights.append(10 ** (b / 20))
    angles, weights = np.array(angles), np.array(weights)
    z = np.sum(weights * np.exp(1j * angles)) / np.sum(weights)
    offset = float(np.angle(z) / (2 * np.pi) * 100)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=22050)
    lt = float(librosa.estimate_tuning(y=y, sr=22050, n_fft=8192, resolution=0.01) * 100)
    return offset, float(np.abs(z)), lt


def track_context(song, info, ev, name):
    """Patch values and automation that move pitch on this track."""
    t = info["tracks"].get(name, {})
    ctx = {}
    if "patch" in t.get("opts", {}):
        pv = C.read_patch(song, t["opts"]["patch"])
        ctx = {k: pv[k] for k in PITCH_PARAMS if k in pv and pv[k] not in (0.0,)}
        ctx["patch"] = t["opts"]["patch"]
    ch = ev["tracks"].get(name)
    autos = defaultdict(set)
    for tt, p, v in C.param_events(ev, ch) if ch is not None else []:
        if p in TRACK_PITCH_AUTOMATION:
            autos[p].add(round(v, 3))
    for p, vals in autos.items():
        ctx["automate." + p] = sorted(vals)[:6]
    return ctx


# ---------------------------------------------------------------- key / percussion / off-grid energy
KEY_PCS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
           "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def key_set(key):
    """Pitch classes accepted as 'in key': natural minor plus the dorian 6th
    and the harmonic-minor leading tone (or the major scale)."""
    tonic, mode = (key.split() + ["minor"])[:2]
    t = KEY_PCS[tonic]
    steps = [0, 2, 3, 5, 7, 8, 9, 10, 11] if mode.startswith("min") else [0, 2, 4, 5, 7, 9, 11]
    return {(t + s) % 12 for s in steps}


def describe_hz(f, keypcs):
    m = float(C.hz_to_midi(f))
    r = int(round(m))
    c = (m - r) * 100
    ok = (r % 12) in keypcs and abs(c) <= 15
    return f"{f:7.1f} Hz = {C.note_name(r)}{c:+.0f}c{'' if ok else '  <-- OFF-KEY' if (r % 12) not in keypcs else '  <-- OFF-GRID'}"


def drum_pitches(stem, notes, keypcs):
    """Sounding pitch of the tuned 909 voices: the kick's tail (after its
    sweep), the rim knock's main mode and the snare body."""
    x, sr = C.load_mono(stem)
    ons = sorted({(round(n["on"], 4), n["note"]) for n in notes})
    times = sorted({t for t, _ in ons})
    out = {}

    def peak(t0, t1, lo, hi):
        a, b = int(t0 * sr), int(t1 * sr)
        if b - a < 256:
            return None
        nfft = 1 << 17
        sp = np.abs(np.fft.rfft(x[a:b] * np.hanning(b - a), nfft))
        f = np.fft.rfftfreq(nfft, 1 / sr)
        m = (f > lo) & (f < hi)
        i = int(np.argmax(np.where(m, sp, 0)))
        l = np.log(sp[i - 1:i + 2] + 1e-12)
        d = 0.5 * (l[0] - l[2]) / (l[0] - 2 * l[1] + l[2]) if (l[0] - 2 * l[1] + l[2]) < 0 else 0
        return (i + d) * sr / nfft

    for name, gm, w0, w1, lo, hi in (("kick tail (BD)", 36, 0.12, 0.40, 25, 200),
                                     ("rim knock (RS)", 37, 0.003, 0.045, 120, 420),
                                     ("snare body (SD)", 38, 0.01, 0.08, 100, 400)):
        vals = []
        for t, g in ons:
            if g != gm:
                continue
            nxt = min([u for u in times if u > t + 1e-4] + [t + 10])
            others = {g2 for t2, g2 in ons if abs(t2 - t) < 1e-3} - {gm}
            if gm == 36 and nxt - t < w0 + 0.1:
                continue
            if gm == 37 and others:
                continue          # rim alone only
            if gm == 38 and 36 in others:
                continue          # snare without a kick: a coincident kick's 3rd harmonic lands in the band
            f = peak(t + w0, min(t + w1, nxt), lo, hi)
            if f:
                vals.append(f)
        if vals:
            out[name] = (len(vals), float(np.median(vals)))
    return out


def tonal_peaks(x, sr, n, hop, lo, hi, rel_db=25.0):
    """Per frame: (time, [(Hz, energy)]) of spectral peaks that persist into
    the next frame within 10 cents (tonal partials, not noise)."""
    w = np.hanning(n)
    nfft = 4 * n
    f = np.fft.rfftfreq(nfft, 1 / sr)
    idx = np.where((f > lo) & (f < hi))[0]
    frames = []
    for s in range(0, len(x) - n, hop):
        sp = np.abs(np.fft.rfft(x[s:s + n] * w, nfft))
        l = 20 * np.log10(sp + 1e-12)
        lb = l[idx]
        thr = max(lb.max() - rel_db, -110.0)
        k = idx[1:-1][(lb[1:-1] > lb[:-2]) & (lb[1:-1] >= lb[2:]) & (lb[1:-1] > thr)]
        pk = []
        for i in k:
            a, b, c = l[i - 1], l[i], l[i + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den < 0 else 0
            pk.append(((i + d) * sr / nfft, 10 ** (b / 10)))
        frames.append(((s + n / 2) / sr, pk))
    out = []
    for j in range(len(frames)):
        t, pk = frames[j]
        nb = [fr for jj in (j - 1, j + 1) if 0 <= jj < len(frames) for fr, _ in frames[jj][1]]
        nb = np.array(nb) if nb else np.array([1.0])
        keep = [(fr, e) for fr, e in pk if np.min(np.abs(1200 * np.log2(nb / fr))) < 10]
        out.append((t, keep))
    return out


BANDS = (("low 25-110 Hz", 25, 110, 32768, 8192), ("mid 110-1500 Hz", 110, 1500, 8192, 2048))


def offgrid_energy(path, sections, tol=15.0):
    """{band: {section: (tonal energy, off-grid tonal energy)}}"""
    x, sr = C.load_mono(path)
    res = {}
    for bname, lo, hi, n, hop in BANDS:
        acc = defaultdict(lambda: [0.0, 0.0])
        for t, pk in tonal_peaks(x, sr, n, hop, lo, hi):
            sec = C.section_of(t, sections)
            for fr, e in pk:
                m = float(C.hz_to_midi(fr))
                acc[sec][0] += e
                if abs(m - round(m)) * 100 > tol:
                    acc[sec][1] += e
        res[bname] = dict(acc)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("wav", help="the mix (for the master pitch reference); '-' to skip")
    ap.add_argument("--stems", help="stem directory (default qa/stems/<song>)")
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--tracks", help="comma-separated subset")
    ap.add_argument("--tol", type=float, default=12.0)
    ap.add_argument("--csv", help="per-note CSV (default <stems>/tuning_notes.csv)")
    ap.add_argument("--no-pyin", action="store_true")
    ap.add_argument("--key", default="D minor", help="tonic and mode for the in-key check (default 'D minor')")
    ap.add_argument("--no-energy", action="store_true", help="skip the off-grid energy attribution")
    a = ap.parse_args()

    song = os.path.abspath(a.song)
    cache = a.stems or C.default_cache(song)
    info = C.parse_song_text(song)
    ev = C.export_events(song, cache)
    C.render_stems(song, cache, a.rerender)
    notes = C.notes_by_track(ev)
    want = set(a.tracks.split(",")) if a.tracks else None

    print(f"# Tuning report: {os.path.basename(song)}  (tolerance +-{a.tol:.0f} cents, A4 = 440 Hz)")
    if info["tempo_automated"]:
        print("# note: the song automates bpm; sections are converted at the base tempo")

    all_rows = []
    per_track = {}
    for name, tr in info["tracks"].items():
        if tr["kind"] in ("kit", "sampler") or name not in notes:
            continue
        if want and name not in want:
            continue
        path = os.path.join(cache, name + ".wav")
        if not os.path.exists(path):
            continue
        x, sr = C.load_mono(path)
        pv = C.read_patch(song, tr["opts"].get("patch", "")) if "patch" in tr["opts"] else {}
        rel = min(3.0, pv.get("release", 0.5))
        rows = analyse_track(x, sr, notes[name], rel, a.tol)
        for r in rows:
            r["track"] = name
            r["section"] = C.section_of(r["on"], info["sections"])
        all_rows += rows
        per_track[name] = rows
        mono = max(sum(1 for m in notes[name] if m["on"] <= n["on"] < m["off"]) for n in notes[name]) == 1
        tr["mono"] = mono
        if mono and not a.no_pyin and len(notes[name]) > 1:
            try:
                tt, et = pyin_trace(x, sr, notes[name])
                tr["pyin"] = (len(et), float(np.median(et)) if len(et) else float("nan"),
                              float(np.mean(np.abs(et) > a.tol) * 100) if len(et) else float("nan"))
            except Exception as e:  # pyin is a cross-check only
                tr["pyin"] = None
                print(f"# pyin failed on {name}: {e}", file=sys.stderr)

    # ---- per track
    print("\n## Per track (deviation of the sounding pitch from the WRITTEN note, cents)")
    print(f"{'track':<8} {'notes':>5} {'median':>7} {'mean|d|':>7} {'p90|d|':>7} {'spread':>6} "
          f"{'flagged':>9}  {'pYIN: frames, median ET dev, %>tol':<36} patch / pitch controls")
    summary = {}
    for name, rows in per_track.items():
        if not rows:
            continue
        d = np.array([r["dev"] for r in rows])
        w = np.array([r["off"] - r["on"] for r in rows])
        nf = sum(r["flag"] for r in rows)
        p = info["tracks"][name].get("pyin")
        ptxt = (f"{p[0]:5d} {p[1]:+6.1f} {p[2]:5.1f}%" if p and p[0] else "(no voiced frames)" if p else
                "(polyphonic)" if not info["tracks"][name].get("mono") else "(skipped)")
        ctx = track_context(song, info, ev, name)
        print(f"{name:<8} {len(rows):5d} {np.median(d):+7.1f} {np.mean(np.abs(d)):7.1f} "
              f"{np.percentile(np.abs(d), 90):7.1f} {np.median([r['spread'] for r in rows]):6.1f} "
              f"{nf:4d} {100 * nf / len(rows):3.0f}%  {ptxt:<36} {ctx}")
        summary[name] = {"median": float(np.median(d)), "wmean": float(np.average(d, weights=w)),
                         "flagged": nf, "n": len(rows)}

    # ---- per section
    secnames = [s[0] for s in info["sections"]] or ["-"]
    print("\n## Per section: median deviation (cents) per track; * = a flagged note in that cell")
    names = [n for n in per_track if per_track[n]]
    print(f"{'section':<9}" + "".join(f"{n[:7]:>8}" for n in names))
    for s in secnames + (["-"] if any(r["section"] == "-" for r in all_rows) else []):
        line = f"{s:<9}"
        for n in names:
            rr = [r for r in per_track[n] if r["section"] == s]
            if rr:
                flag = "*" if any(r["flag"] for r in rr) else " "
                line += f"{np.median([r['dev'] for r in rr]):+7.1f}{flag}"
            else:
                line += f"{'.':>8}"
        print(line)

    # ---- flagged notes
    flagged = [r for r in all_rows if r["flag"]]
    print(f"\n## Flagged notes ({len(flagged)} of {len(all_rows)} measured): "
          f"median beyond +-{a.tol:.0f} c, or beyond it for >= {SUSTAIN_FLAG_S * 1000:.0f} ms")
    print(f"{'time':>7} {'section':<9} {'track':<7} {'note':<5} {'dur':>5} {'median':>7} {'first':>7} "
          f"{'worst':>7} {'spread':>6} {'>tol for':>8}")
    for r in sorted(flagged, key=lambda r: r["on"]):
        print(f"{r['on']:7.2f} {r['section']:<9} {r['track']:<7} {C.note_name(r['note']):<5} "
              f"{r['off'] - r['on']:5.2f} {r['dev']:+7.1f} {r['first']:+7.1f} {r['worst']:+7.1f} "
              f"{r['spread']:6.1f} {r['off_tol_s']:7.2f}s")

    # ---- pitched percussion
    keypcs = key_set(a.key)
    kits = [n for n, t in info["tracks"].items() if t["kind"] == "kit"]
    if kits and os.path.exists(os.path.join(cache, kits[0] + ".wav")):
        kn = [m for k in kits for m in notes.get(k, [])]
        kn += [m for k, v in notes.items() if k not in info["tracks"] and False for m in v]
        print(f"\n## Pitched percussion (909 board, stem '{kits[0]}'), key {a.key}")
        bus = C.param_events(ev, 0)
        for p in ("BdTune", "RsTune", "SdTune"):
            vals = sorted({round(v, 3) for t, pp, v in bus if pp == p})
            print(f"  {p}: {vals}")
        allkit = [m for k in kits for m in C.notes_by_track(ev).get(k, [])]
        for name, (cnt, f) in drum_pitches(os.path.join(cache, kits[0] + ".wav"), allkit, keypcs).items():
            print(f"  {name:<16} {cnt:3d} hits  median {describe_hz(f, keypcs)}")

    # ---- off-grid tonal energy, mix and per stem
    if not a.no_energy:
        secs = info["sections"]
        order = [s[0] for s in secs]
        print(f"\n## Off-grid tonal energy: share of persistent spectral peaks more than 15 cents from the A440 grid")
        try:
            if a.wav == "-":
                raise RuntimeError("no mix")
            mix = offgrid_energy(a.wav, secs)
            print("mix:")
            print(f"  {'section':<9}" + "".join(f"{b[0]:>18}" for b in BANDS))
            for sname in order:
                line = f"  {sname:<9}"
                for b in BANDS:
                    e, o = mix[b[0]].get(sname, (0, 0))
                    line += f"{(100 * o / e if e else float('nan')):17.1f}%"
                print(line)
        except RuntimeError:
            pass
        stem_e = {}
        for name in info["tracks"]:
            p = os.path.join(cache, name + ".wav")
            if os.path.exists(p) and (want is None or name in want or info["tracks"][name]["kind"] == "kit"):
                stem_e[name] = offgrid_energy(p, secs)
        for b in BANDS:
            print(f"who puts off-grid tonal energy in the {b[0]} band (share of the stems' summed off-grid energy):")
            for sname in order:
                tot = sum(stem_e[n][b[0]].get(sname, (0, 0))[1] for n in stem_e)
                if tot <= 0:
                    continue
                shares = sorted(((stem_e[n][b[0]].get(sname, (0, 0))[1] / tot, n) for n in stem_e), reverse=True)
                print(f"  {sname:<9} " + ", ".join(f"{n} {100 * sh:.0f}%" for sh, n in shares[:4] if sh >= 0.05))

    # ---- master reference
    print("\n## Master pitch reference")
    if all_rows:
        d = np.array([r["dev"] for r in all_rows])
        w = np.array([r["off"] - r["on"] for r in all_rows])
        good = np.abs(d) < 50
        print(f"all measured notes (duration-weighted mean, |d|<50): {np.average(d[good], weights=w[good]):+.2f} cents; "
              f"median {np.median(d):+.2f} cents over {len(d)} notes")
    try:
        if a.wav == "-":
            raise RuntimeError("no mix given ('-')")
        off, conf, lt = master_reference(a.wav)
        print(f"mix spectral peaks vs the A440 grid: {off:+.2f} cents (concentration {conf:.2f}; 1 = all "
              f"peaks agree); librosa.estimate_tuning: {lt:+.1f} cents")
    except Exception as e:
        print(f"mix reference failed: {e}")

    out = a.csv or os.path.join(cache, "tuning_notes.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["track", "section", "on", "off", "note", "name", "vel", "dev_cents", "first_cents",
                     "worst_cents", "spread_cents", "et_dev_cents", "frames", "off_tol_s", "flag"])
        for r in sorted(all_rows, key=lambda r: (r["track"], r["on"])):
            wr.writerow([r["track"], r["section"], f"{r['on']:.3f}", f"{r['off']:.3f}", r["note"],
                         C.note_name(r["note"]), f"{r['vel']:.2f}", f"{r['dev']:.2f}", f"{r['first']:.2f}",
                         f"{r['worst']:.2f}", f"{r['spread']:.2f}", f"{r['et_dev']:.2f}", r["frames"],
                         f"{r['off_tol_s']:.2f}", int(r["flag"])])
    print(f"\nper-note CSV: {out}")


if __name__ == "__main__":
    main()
