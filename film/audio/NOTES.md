# NEURAL PAIRING: score notes

Everything here comes from the Patina synth. D major, 120 bpm (1 beat = 0.5 s, 1 bar = 2.0 s), 30 bars = 60.000 s.
Style: cinematic synthwave (909, octave bass, aurora pad, dotted-8th ping-pong echo), Cliff Martinez glass in the
mechanism and training sections, and a mallet/music-box bounce for the games.

## Files
- `score.song`: the score. Every cue is pinned with absolute `>beat` seeks (beat = 2 × seconds).
- `fx.song` → `fx.wav`: the noise FX stem (air, risers, whoosh, crash wash, frost, rumble). These are Patina voices
  playing only the shared noise source (the oscillator sits at MIDI 0 under a high-pass). The score plays the stem
  back on Patina's own tape deck (`track fx sample=fx.wav`), so the noise never leaks into the musical voices.
- `patches/`: a copy of Patina's factory patches plus 7 score patches (`fx-burst`, `fx-swell`, `fx-low`, `fx-ice`,
  `boom`, `sour`, `stab`). Patina looks patches up relative to the working directory, so render from here.
- `render.sh`: rebuilds everything: fx stem → raw bounce (`work/raw.wav`) → `master.sh` → `cues.py` verification.
- `master.sh`: ffmpeg HPF at 24 Hz, +4.8 dB, a −1.5 dBFS peak limiter (it touches only about 400 transient samples),
  a qsin fade-out from 58.6 to 59.9 s, then pad/trim to exactly 2,880,000 samples.
- `cues.py` (onset check), `env.py` (RMS envelope), `active.py`, `solo.py` (render a subset of tracks), `readwav.py`.
- `score_spectrogram.png`: log-frequency spectrogram of the final mix.

## Final master (`score.wav`)
48 kHz, stereo, 24-bit PCM, 60.000000 s. **−14.1 LUFS integrated**, true peak **−1.4 dBTP**, LRA 4.2 LU, with no clipping.
The output is silent from 59.9 s. The peak concurrent voice count is 62 of 64, so no voice stealing occurs.

## Sync points (measured on score.wav)
Onsets were found with a 5 ms log-energy detector (`cues.py`). Every hit lands within **+5 ms** of its cue
(+7.5 ms for the soft Ember entry); that offset is the detector's frame resolution.

| time (s) | beat | what happens | measured |
|---|---|---|---|
| 0.0 | 0 | Swell from silence: the aurora pad (Dmaj9) fades up from gain 0 while filtered noise "air" opens 300 Hz → 2.2 kHz | −63 dB at 0–0.3 s, −19 dB by 2–3 s |
| 0.4–3.2 | 0.8–6.4 | Assembly arpeggio: Quill (music box, panned left) starts at 0.4 and Ember (pluck, panned right) joins at 0.775 a fifth above. They interlock as straight 16ths and climb two octaves (D4 → F#6) | 0.405 / 0.782 |
| 3.0 | 6 | "Eyes open" glass chime (A5+E6) | — |
| 6.0 | 12 | Pad moves to Bm(add11), a questioning colour for the tile toss | — |
| **8.0** | 16 | **Shatter**: 909 kick, clap and long open hat, a noise crash burst, a low boom bending down an octave, and 15 glass shards in 32nds tumbling down from E7 | 8.005 (+18 dB rise) |
| 8.4–9.9 | 16.8–19.8 | Resonant noise riser and a snare roll that crescendos; both stop 100 ms before the drop | — |
| **10.0** | 20 | **The drop**: four-on-the-floor 909, octave synthwave bass, Dmaj9 pad, boom, crash | 10.005 (+20 dB) |
| 10.0–13.5 | 20–27 | **Eight rising bridge plucks**, one per bridge (D4 F#4 A4 C#5 E5 F#5 A5 D6), doubled by glass sparks an octave up | each at +5 ms. Each pitch is 8–14 dB above the bed |
| 14.0–14.5 | 28–29 | Drums and bass drop out while the resonant noise whoosh sweeps up | high-band peak at **14.49** |
| 14.5–21.2 | 29–42.4 | **Glass mechanism**: Martinez-style glass 16th cells (Gmaj9 → Em9 → Bm9 → A9sus), a 16th sub-bass pulse, precise closed hats and rim ticks, and a half-time clap. Signal cues: five blips (one per tick orb) from 14.5, a gate-closing downward slide at 16.0, a rising query-beam glide at 17.4, a falling "values pour" glide at 19.0, a low "+" ping at 20.5 | — |
| 21.2–21.9 | 42.4–43.8 | Suck-in noise riser and snare fill | — |
| **22.0** | 44 | **Impact into the games**: boom, crash, stab chord, then a swung 909 groove with a bouncy bass and mallet + music-box melody | 22.005 (+19 dB) |
| **29.5** | 59 | **"Lantern" resolve**: G → A → **D**, with a bright D6+A6 glass bell, a stab, the melody landing on D6, a long open hat and a sparkle burst | 29.505 |
| **30.5, 31.0, 31.5, 32.0, 32.5** | 61–65 | **Five staccato value-game hits**: kick+clap with rising stab chords D, Em, F#m, G, A and bass on each | all +5 ms (+16 to +21 dB rises) |
| 33–36 | 66–72 | Bouncy play: four-on-the-floor swung, G–A–D octave bass, a bright melody up to D6/D7 | — |
| **36.0** | 72 | **Training**: a ticking rim-shot clock (tick/tock), glass 16ths on Bm and G, staccato bass | 36.005 |
| **40.0** | 80 | **Ice shimmer**: a ring-mod glass-anemone bell cluster (E6/B6 rising to E7), high frost hiss with filter shimmer, Em9 pad, sustained E1. The clock continues softly without the kick | — |
| 43–46 | 86–92 | **Riser**: noise sweep from 200 Hz to 12 kHz, bass and glass filters opening, a snare roll tightening 8ths → 16ths → 32nds with snare pitch rising, and the pad swelling on Asus → A | — |
| **46.0** | 92 | **Near-silence**: everything cuts and the reverb return is muted for 1 s. Only a very soft D drone, a faint high D6 and a low rumble remain | −13.8 dB just before → **−34 dB RMS** from 46.1 to 47.9 |
| **48.0** | 96 | **The stranger's sour note**: D#4 from two saws a minor second apart in one voice, with uneasy vibrato, against the D drone | −25 dB RMS |
| 49.0–49.9 | 98–99.8 | Short reverse riser and two-stroke snare pickup | — |
| **50.0** | 100 | **BIG hit**: boom, crash, kick+clap+snare+open hat, stab, and the full groove returns with the lantern-lead theme (F#5 E5 D5 A4 …) | 50.005 (+20 dB, −10 dB RMS after) |
| **53.0, 53.8, 54.6** | 106, 107.6, 109.2 | **Stat hits**: kick, clap and open hat with stab chords Bm, A, G/D and glass bells B5, C#6, D6 rising, on a 0.8 s counter with 16th-ish hats between | +5 ms each |
| **55.6** | 111.2 | **Outro**: crash, boom and a Gmaj9 swell. The anthem cadence Gmaj9 → A → **Dadd9 at 58.0** under a warm half-time beat, with the lead singing F#5 → E5 → D5 and a final D6+A6 bell | 55.605 |
| 58.6–59.9 | | Fade-out (qsin) finishes by 59.9. 59.9–60.0 is digital silence | |

## Voice box
I left out the optional "neural pairing" line. No speech recognizer is available here, so I could not check that
the line is intelligible. Revision 2 of the storyboard also asks for plain, sober wording, and a robotic voice saying the
title would work against that.
