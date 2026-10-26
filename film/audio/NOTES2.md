# NEURAL PAIRING v2: score notes (score2)

The v2 score is 82.000 s long. It is in **D minor at 120 bpm, felt in half time**: the kick falls on 1 and a fat
snare plus clap on 3, one bar every 2 s. The mood is dark, confident and cinematic, closer to Drive, Tron: Legacy,
Cliff Martinez and Reznor than to synth-pop.
The v1 files (`score.song`, `fx.song`, `score.wav`, `NOTES.md`) are untouched.

## Files
- `score2.song`: the score. Every cue is placed with `section` lines and `>NAME` seeks, so moving a cue means
  editing one section line. The section table at the top doubles as the cue sheet.
- `fx2.song` → `fx2.wav`: the noise FX stem (air, risers, crash washes, whooshes, frost, rumble), played back on
  Patina's tape deck. Its `section` table is identical to score2's: **shift a cue in both files.**
- `patches/`: new voices `quill`, `ember`, `sub2` and `lowstab`. It also uses `tungsten` (Ember's pad), `vellum` (Quill's
  pad), `nc-pluck` (ostinato), `nc-glass`, `glasslead`, `glintbell`, `glass-anemone`, `boom`, `sour` and the `fx-*` patches.
- `render2.sh` rebuilds everything: fx stem → raw bounce → `master2.sh` → per-voice stems → `cues2.py` sync report.
- `master2.sh`: 24 Hz HPF, +8.1 dB, a −1.5 dBFS peak limiter, a 0.3 s fade-in, the fade-out 80.6 → 81.9 s, then a
  trim to exactly 3,936,000 samples.
- `cues2.py` / `cues2_report.txt`: the sync verification. `lufs.sh` gives a momentary-loudness profile.
  `score2_spectrogram.png` is the final spectrogram.

## Final master (`score2.wav`)
48 kHz, stereo, 24-bit, **82.000 s**. **−13.9 LUFS integrated**, true peak **−1.4 dBTP**, sample peak −1.5 dBFS, no
clipping. The output is digital silence after 81.9 s. The peak concurrent voice count is 60 of 64, so no voice stealing occurs.

## Sound
- **Drums, blown out:** the 909 board runs with bass-drum drive at 0.8, the bus drive at 0.85, a low-tuned kick and
  a low, long snare. Snare and clap are layered on beat 3. The drum strip carries its own compressor (`comp=-30`:
  4:1, 0.1 ms attack, auto release, about +10 dB makeup) into the master bus. There the SSL-style bus compressor runs in
  **parallel** (4:1, 3 ms, auto release, 35 % blend), followed by tape saturation (drive 0.38). The hats are sparse
  ghosts. There are **no snare rolls**: builds are made with the kick, the riser and the ostinato.
- **Low end:** a deep sine sub (`sub2`, D1–D2) follows the kick's syncopation, with a pitch-dropping boom on the
  big hits.
- **No thin highs.** Quill's highest note is F5, the glass stays in the mid register, and the pads are filtered dark.
  The only exception is the half-second burst of shatter shards at 8.0, which falls from C6 to C5.

## Music theory as story

| section (s) | what the music does | why |
|---|---|---|
| **0–4 assembly** | Fade from silence. **Quill** (a mid glass pluck, panned left) states its motif **D–F–A–B♭–A**, which is aeolian: the minor sixth B♭ sighs down to A. **Ember** (a warm low-mid saw, panned right) answers with **D–A–B–C–A**, which is dorian: the *raised* sixth B♮. Each has its own pad: Quill's sustains **B♭/D**, Ember's **Dm6** (with B♮). | Two minds, two dialects. The B♭ and B♮ sound at the same time, a cross-relation. They share a key but not a language. |
| **4–7 first pass** | Both climb the same scale in lockstep, one note per beat, Quill in octave 4 and Ember in octave 3. Quill climbs through B♭ and Ember through B♮. Each ends on its own token (D5 and D4) at 7.0, over soft heartbeat kicks. | They compute in parallel but separately. The same shape in two modes means nothing is shared. |
| **8.0 shatter** | Kick, snare, clap and open hat with a crash wash, a pitch-dropping boom, and a stab on **D + G♯**, the tritone. Glass shards fall in 32nds. | A token cannot cross. The tritone is the "no". |
| **10–12 hold ("neuralese")** | A drone on Dm(add9). Ember sounds A3 and Quill sounds E4: their **first consonance**, a perfect fifth. A riser leads into 12. | Under the glass, their hidden states can meet. |
| **12.0–15.5 bridges** | The drop into the half-time groove. Eight connection notes **alternate timbres**, Ember on 12.0/13.0/14.0/15.0 and Quill on 12.5/13.5/14.5/15.5. Together they climb **one Dm9 arpeggio** (D3 F3 A3 C4 D4 F4 A4 C5), with a rim knock on each. | **Hocket.** Neither voice alone makes the arpeggio; the link does. |
| **15.8–22.1 slow pass, 8 exchanges** | One note per exchange at 15.8 + 0.9k with a deep thump and a rim knock, **alternating Ember → Quill → Ember …** and alternating register (low saw, mid glass). It is a literal **zig-zag** whose pitch classes climb stepwise: F3, G4, A3, B♭4, C4, D5, E4, F5. The harmony goes Dm9 → B♭maj7 → Gm9 → A7(♭13) for maximum pull. | The idea bounces between the bodies on its way up the ladder, one step per exchange. |
| **23.0 token** | Both voices strike the **same D4 together** over a hit, and A7 resolves to Dm. | One token out: one shared note. |
| **23–26 fast passes** | The hocket speeds up from alternating 8ths to interlocking 16ths (Ember on the 8ths, Quill on the off-16ths). The single line runs Dm9 → B♭maj7 → Gm9 → A, then into D. | Many passes, a dense two-way conversation. |
| **26–32 the read** | Sparse and precise: kick and rim ticks only, and the pads drop. Ember's **query** is a glide from D3 to A4. Quill's **keys** are five clean blips (G A B♭ C D). One **gate closes** as a glass slide down (A4 → D4). The **values pour** into Ember as a falling line (D C B♭ A G). Mid-register glass ticks run over B♭maj7(#11) and Gm(add9). | Each element of the read has its own sound. |
| **32–36.5 the secret** | Quill alone plays the **"lantern" motif, D–A–G–B♭**, twice (32.0 and 34.0). It is Quill's aeolian B♭ in a new shape. Ember's pad is held under a near-silent kick heartbeat. A low ostinato rises underneath (34.5–36.4): gold climbing Ember's core. | |
| **36.5** | **Ember plays D–A–G–B♭ back, exactly**, the same pitches and rhythm, in Ember's own timbre. | Information transferred, musically. |
| **37–42 the mood** | **Picardy third, then contagion.** At **37.5** Quill's line and Quill's pad raise F → **F♯** (D major) while Ember's pad stays minor: F against F♯, bittersweet. At **39.5** Ember's line and pad raise F → **F♯** as well, and both open onto **G major** (the dorian IV, a lift). | The feeling crosses the link. |
| **42–54 games** | A **D-dorian vamp** (Dm9 ↔ G6/9, then B♭maj7 → A7sus → A). Window pops every 0.6 s from 42.0 alternate Quill and Ember (9 pops, rim knocks), against the half-time grid as a polyrhythm. Then Quill calls and Ember answers. A Drive-style low pluck plays 8ths. | Many games, many exchanges, the groove at full stride. |
| **54–66 training** | A mechanical 16th ostinato in the low pluck with 16th ghost hats. Quill and Ember trade single notes. At **58.0** the harmony **freezes** on Gm(add9)/D, with frost hiss, a ring-mod glass bloom and the ostinato filter closing. From **61–66** comes the riser: B♭ → **A7** (C♯ leading tone), the ostinato filter opening, kicks accelerating from quarters to 8ths, and Quill and Ember climbing the A7 arpeggio **in parallel octaves**. | Learning to move together. The frozen chord is the frozen weights. |
| **66.0 drop** | Everything cuts. Only a low rumble remains, with the reverb briefly muted. | |
| **66.5** | The **real partner** (Ember) plays the lantern motif softly and correctly. | ✓ |
| **68.0 the stranger** | The motif comes back **a tritone away** (G♯–D♯–C♯–E). It is played by a detuned voice of two saws a minor second apart with an uneasy vibrato, over the D rumble. | ✗ Another episode's partner, and the meaning is lost. |
| **70.0 proof** | A big hit and the full groove. The **theme**, which the lantern motif grew into (D A G B♭ / A F D / G B♭ D C / A C♯), is played by **both voices in octaves** (Quill an octave above Ember) over Dm → B♭maj7 → Gm9 → A7. At **73.0 / 73.8 / 74.6** the window hits are kick, clap and open hat with dark stabs **B♭ → C → Dm**: a VI–VII–i rise. | Two voices, one line. |
| **78–82 title** | **Dm(add9) → B♭maj7 → Dsus4(add9)**. The melody falls D5 → A4 and then **both voices meet in unison on D4** at 80.0. The fade runs 80.6 → 81.9. | Warm and dark, resolved, and one voice. |

## Sync verification (`cues2_report.txt`)
- **Hits** (shatter, drop, token, training, riser, the big hit, the three window hits, title) were measured on the
  full mix with a broadband onset detector.
- **Melodic cues** (bridges, exchanges, lantern, Picardy thirds, window pops, real partner, stranger, final unison)
  were measured as the onset of **that note's own pitch**. The note numbers come from Patina's exported events. Each
  was measured on its own voice's stem rendered through the same engine, because in the full mix the pads hold the same
  pitch classes. The full-mix broadband offset is printed beside each one.

All 45 checked cues land within **±10 ms** (the worst is 10.0 ms):

| cue | s | measured | cue | s | measured |
|---|---|---|---|---|---|
| shatter | 8.0 | +2.5 ms | window pops 1–9 | 42.0 + 0.6k | −10 … 0 ms |
| drop / bridge 1 | 12.0 | +2.5 / −7.5 ms | training | 54.0 | +5.0 ms |
| bridges 2–8 | 12.5 … 15.5 | −5 … +10 ms | riser start | 61.0 | +2.5 ms |
| exchanges 1–8 | 15.8 + 0.9k | −10 … −2.5 ms | real partner | 66.5 | −5.0 ms |
| token pop | 23.0 | +5.0 ms | stranger | 68.0 | −5.0 ms |
| lantern, Quill | 32.0 / 34.0 | −7.5 / −5.0 ms | BIG hit | 70.0 | +2.5 ms |
| lantern, Ember | 36.5 | −10.0 ms | window hits | 73.0 / 73.8 / 74.6 | +2.5 / +7.5 / +5.0 ms |
| Quill F♯ | 37.5 | +7.5 ms | title | 78.0 | +5.0 ms |
| Ember F♯ | 39.5 | +2.5 ms | final unison | 80.0 | +5.0 ms |

Loudness arc (momentary, mastered): assembly about −21 LUFS, first pass −19, shatter −10, hold −21, groove −13 to −14,
read −16 to −18, games −13, drop −24 to −30, proof −10, title −12 to −15.
