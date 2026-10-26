# NEURAL PAIRING v3: score notes (score3)

The v3 score is 82.000 s long. It keeps the v2 theory-as-story design and rebuilds the **sound**: sustained, modulated,
evolving patches; a harder, overdriven and compressed kit; and every hit in tune and on a beat. It follows the director's final grid-locked cue sheet.
The v1 and v2 files are untouched.

## Files
- `make_score3.py` → **`score3.song`** + **`fx3.song`**. The song is generated: edit the Python, not the song. Every
  cue time is a number in that script.
- `fx3.wav`: the noise FX stem (air, risers cut 50 ms before each hit, crash washes, whooshes, frost, rumble),
  played back on Patina's tape deck.
- `patches/s3-*.patch`: the v3 voices (`s3-quill`, `s3-ember`, `s3-pad`, `s3-glasspad`, `s3-growl`, `s3-brass`,
  `s3-arp`, `s3-glass`, `s3-kickbody`), plus `sub2`, `boom`, `sour`, `glasslead` and the `fx-*` patches.
- `render3.sh` rebuilds everything (generate → fx stem → raw bounce → master → per-voice stems → cue report).
  `master3.sh` applies a 24 Hz HPF, gain, a −1.5 dBFS limiter, a 0.3 s fade-in and the fade 80.6 → 81.9 s, then trims to
  exactly 3,936,000 samples.
- `cues3.py` → `cues3_report.txt` (sync). `work3/drumscan.py` measures the 909's pitches against its tune knobs.
  The QA reports are `work3/tuning3.txt` and `work3/timing3.txt`.

## On the grid
The director's final cue sheet is grid-locked, so everything runs at **120 bpm** and every note-on sits on the 16th
grid (0.125 s). This is verified by the timing report (0 off-grid note-ons) and by `--export-events`.

| cue | s | on the grid as |
|---|---|---|
| shatter / drop | 8.0 / 12.0 | downbeats |
| bridges | 12.0 + 0.5k | every beat, alternating Ember / Quill |
| exchanges | 15.875 + 0.875k | every 7 sixteenths: a 7:4 cross-rhythm against the half-time kick and snare (each with a rim knock and the voice's note) |
| token | 23.0 | both voices on D4 with the hit |
| fast passes | 23.5 + 0.25k | 8ths, the line alternating Ember / Quill, with 16th arps underneath |
| games waves | 42.0 / 45.0 / 48.0 | slams on +0, +0.25 and +0.5 (kick+clap+OH, kick, kick+snare, brass stabs); links ignite on +1.0 (both voices sound a fifth); the answer types 16ths from +1.5 in the answering model's voice; checks +2.5 / +2.625 / +2.75 (rim + glass A4–D5–F5) |
| pull-back / title / fold / pulse / fly-off | 51.0 / 51.5 / 52.5 / 53.0 / 53.25–53.9 | whoosh + kick; brass A7sus; whoosh; kick+clap+brass+boom; a noise riser |
| result windows | 73.0 / 73.75 / 74.5 / **75.25** | every 1.5 beats, B♭ → C → A → **Dm**; the 4th is the strongest |
| title / "neural" / "pairing" / hop | 78.25 / 78.5 / 79.0 / 79.5 | |

There is no swing, and the echo is a quarter note. The only deliberate pitch slides are the query glide at 26.75 and the
gate slide at 29.5–29.625.

## In tune
- The engine itself is in tune: a pure sine measures −1 to 0 cents at D2–A5 (`work3/pitch.py`).
- What made score2 sound out of tune, now removed:
  - The 909 kick sat about a quarter tone off G against the D sub, now retuned (see the kit below).
  - Ring-modulated, inharmonic glass patches (`nc-glass`, `glintbell`, `glass-anemone`) are replaced by
    `s3-glass`, whose partials are all harmonic.
  - The ostinato's high resonance (2.2–3.0) whistled at its cutoff, not at the note; it is now 1.3–1.6.
  - Tape wow is 0 (flutter 0.004).
  - Detunes are trimmed to 3–6 cents between oscillators, with width from chorus and voice spread instead.
- **The kit, tuned to the key** (the QA settings, verified with `qa/tuning.py`; see the results below): kick
  `bd_tune 0.20` (its tail on G1), short (`bd_decay 0.24`, sweep 0.4), with a sine body on **A1** (the fifth of D),
  trimmed −17 c because the short sine reads sharp. Snare `sd_tune 0.17` (E3) and rim `rs_tune 0.52` (A3).
- **Deliberate pitch effects** (intended, and named as such in the reports): the stranger at 68 (two saws a minor
  second apart, with vibrato), the boom's pitch drops on the hits, Ember's query glide at 26.5, and the glass slide at 29.5.

## The sound
- **Quill** (`s3-quill`, mid register): a PWM glass voice. The pulse width breathes on the LFO, with harmonic sine
  partials (octave and twelfth), an OTA filter bloom into a sustained glow, a 1.4 s release into reverb, echo and chorus.
- **Ember** (`s3-ember`, low-mid): a driven Minimoog-school saw, two saws plus a square an octave down through the
  ladder (drive 2.6, resonance 1.3). The 30 ms filter bloom opens wider with velocity, then sustains with LFO filter
  movement.
- **Pads:** Ember's harmony is `s3-pad`, wide detuned saws with a slow 1.2 s filter bloom and the LFO sweeping
  the cutoff. Quill's harmony is `s3-glasspad`, PWM strings. Both are sidechain-pumped from the kick (`duck=` 0.55 / 0.45)
  and send to the reverb and chorus.
- **Bass:** `sub2` (sine, D1–D2) plus `s3-growl`, saw and square through a slammed ladder (drive 4–5, saturation
  1.8) with an envelope that opens a mid growl; velocity opens it further, so accents bark. It is also pumped.
- **Arp / ostinato** (`s3-arp`): pulse plus saw, swept by automation through every section.
- **Impacts:** `s3-brass` (driven detuned saws), `boom` (pitch-dropping sub), crash washes and risers from the noise
  stem, and tonal **reverse swells** (pad chords swelling exponentially into 8.0, 12.0, 70.0 and 75.25, cut on the hit).
- **Drums:** the 909 with bus drive 0.9 and kick drive 0.75, a strip compressor (`comp=-28`: 4:1, 0.1 ms, auto,
  about +9 dB makeup), snare+clap on the backbeats, a spring send, and the tuned kick body. On the master, a
  parallel SSL-style bus compressor (4:1, 3 ms, 35 %) runs into tape drive 0.42. There are **no snare rolls**: builds
  use kicks accelerating from quarters to 8ths to 16ths, the ostinato opening, and noise risers.
- **The sound follows the story** (automation): dark swell (pad 250 → 900 Hz) → opening across the link (pad cutoff
  and resonance rising through the slow pass) → **cold and tight in the read** (cutoffs down, reverb 0.62, rim ticks)
  → **bloom at the mood** (pads to 2.2–3 kHz, reverb 0.9) → mechanical in training → **frozen at the ice** (arp filter
  closing, reverb 0.93) → everything opening in the riser (pads to 3.2 kHz, resonance up, arp to 3.2 kHz) → the
  climax wide open, with growl drive 5.

## Theory as story (unchanged design; see NOTES2.md for the full table)
- **0–7 s:** two dialects (Quill aeolian B♭, Ember dorian B♮) on the 8th grid.
- **8.0:** the shatter, a tritone.
- **10–12 s:** the first fifth (A3 + E4).
- **12.0–15.5:** the 8 bridges alternate voices and stack one sustained Dm9 chord.
- **15.875–22.0:** the zig-zag, one note per exchange every 7 sixteenths (a 7:4 against the half-time groove).
- **23.0:** both voices on D4.
- **23.5–26:** the fast passes, the line alternating voices on every 8th over 16th arps.
- **26–32:** the cold read.
- **32 / 34:** Quill's lantern motif D–A–G–B♭; **36.5:** Ember plays it back exactly.
- **37.5 / 39.5:** Picardy thirds, Quill first, then Ember.
- **42 / 45 / 48:** three game waves. Window slams; the link ignites with both voices on a fifth; the answering model
  types the answer in 16ths in its own voice (Ember in wave 1, Quill in wave 2, alternating in wave 3); three checks.
  The harmony is a D-dorian walk: Dm9 → G6/9 → B♭maj7 → A7sus → A7.
- **54–66:** the training ostinato, the ice freeze, and the riser in parallel octaves.
- **66.0:** a hard cut; **66.5:** the real partner's lantern; **68.0:** the stranger a tritone away.
- **70:** the big hit.
- **73.0 / 73.75 / 74.5 / 75.25:** **B♭ → C → A → Dm** (VI–VII–V–i) under the theme F–G–A–**D**. The 4th hit is the
  arrival on the tonic and the strongest: full brass, boom, a long crash, a reverse swell into it, the theme's top D,
  and the full groove starting from it.
- **78.25:** the title hit on Dm(add9). The voice says **"neural"** at 78.5 (Dm(add9)) and **"pairing"** at 79.0
  (B♭maj7). The shared hop lands at 79.5, and at **80.0** the last chord, Dsus4(add9), arrives with both voices in
  unison on D4.

<!-- results appended below -->

## Results (final build, measured)
- **Master (`score3.wav`):** 48 kHz, stereo, 24-bit, exactly 82.000 s. **−14.0 LUFS integrated**, true peak **−1.5 dBTP**.
  Silent after 81.9 s.
- **Voice:** Whisper (`qa/voice/asr.py`) on the final mix window 78.3–80.1 s reads **"Neural pairing."** (score 1.00).
- **Timing (`qa/timing.py`, `work3/timing3.txt`):** **0 off-grid note-ons** on every track. The figures are start
  offsets after removing the 10 ms bus delay:
  - kit 0 ms, Quill +3.8, Ember +7.5, arp +7.5, brass 0, sub 0;
  - −6 dB points: pads +19 / +27 ms, melody about 0.
  - Still flagged: the growl's onsets (+10 ms median, smeared), and the deliberate reverse swells and stranger.
- **Tuning (`qa/tuning.py`, `work3/tuning3.txt`):** 1,100 notes, median +0.1 c.
  - Quill −1.6, Ember +0.5, pads +1.8 / −0.8, glass −1.4, brass +1.4, arp −2.3 c.
  - Notes beyond ±12 c, all short low notes: kick body 22, arp 12, growl 10, sub 6, pad 6, brass 5, Ember 1. The
    other flag is the deliberate gate slide.
  - 909: snare and rim on the QA targets (0.17 / 0.52). **Open issue:** the kick tail still reads G1 +49 c at both
    `bd_tune` 0.185 and 0.20 with this kick's sweep and drive. A follow-up `qa/tuning.py` scan over `bd_tune` should
    settle it.
- **Cue check (`cues3_report.txt`):** every hit within ±10 ms on its own stem.
