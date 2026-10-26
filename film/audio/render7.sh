#!/bin/sh
# Rebuild score7.wav from source. Run from /home/user/video/audio. Everything runs under nice,
# one render at a time (the machine is shared).
set -e
cd "$(dirname "$0")"
P="nice -n 10 /home/user/patina/target/release/patina"
mkdir -p work7
python3 make_score7.py                                              # 0. generate score7.song + fx7.song
$P --play fx7.song --render fx7.wav                                 # 1. noise FX stem (normalized to -1 dBFS)
$P --play score7.song --render work7/raw7.wav --no-normalize        # 2. the score
$P --play score7.song --export-events work7/ev7.json                # 3. exact event times
sh master7.sh work7/raw7.wav ${GAIN:-2.5}                           # 4. -14 LUFS, limiter, fades, 128.000 s
if [ -z "$NOSTEMS" ]; then
for t in quill ember ghost glass glassq felt inner; do                                     # 5. per-voice stems for the pitched cues
  python3 solo.py score7.song work7/s7_$t.song $t
  $P --play work7/s7_$t.song --render work7/s7_$t.wav --no-normalize >/dev/null
done
fi
python3 cues7.py score7.wav work7/ev7.json work7 > cues7_report.txt   # 6. sync report
tail -1 cues7_report.txt
