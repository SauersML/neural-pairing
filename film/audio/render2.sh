#!/bin/sh
# Rebuild score2.wav from source. Run from /home/user/video/audio.
set -e
cd "$(dirname "$0")"
P=/home/user/patina/target/release/patina
mkdir -p work
$P --play fx2.song --render fx2.wav                                # 1. noise FX stem (normalized to -1 dBFS)
$P --play score2.song --render work/raw2.wav --no-normalize          # 2. the score (plays fx2.wav on the tape deck)
$P --play score2.song --export-events work/ev2.json                  # 3. exact event times, for verification
sh master2.sh work/raw2.wav 8.1                                      # 4. -14 LUFS, limiter, fades, exactly 82.000 s
for t in quill ember ghost; do                                       # 5. per-voice stems for the pitched cues
  python3 solo.py score2.song work/s2_$t.song $t
  $P --play work/s2_$t.song --render work/s2_$t.wav --no-normalize >/dev/null
done
python3 cues2.py score2.wav work/ev2.json work                       # 6. sync report
