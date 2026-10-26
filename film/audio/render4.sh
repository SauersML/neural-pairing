#!/bin/sh
# Rebuild score4.wav from source. Run from /home/user/video/audio. Everything runs under nice,
# one render at a time (the machine is shared).
set -e
cd "$(dirname "$0")"
P="nice -n 10 /home/user/patina/target/release/patina"
mkdir -p work4
python3 make_score4.py                                              # 0. generate score4.song + fx4.song
$P --play fx4.song --render fx4.wav                                 # 1. noise FX stem (normalized to -1 dBFS)
$P --play score4.song --render work4/raw4.wav --no-normalize        # 2. the score
$P --play score4.song --export-events work4/ev4.json                # 3. exact event times
sh master4.sh work4/raw4.wav ${GAIN:-2.2}                           # 4. -14 LUFS, limiter, fades, 96.000 s
if [ -z "$NOSTEMS" ]; then
for t in quill ember ghost glass glassq; do                                     # 5. per-voice stems for the pitched cues
  python3 solo.py score4.song work4/s4_$t.song $t
  $P --play work4/s4_$t.song --render work4/s4_$t.wav --no-normalize >/dev/null
done
fi
python3 cues4.py score4.wav work4/ev4.json work4 > cues4_report.txt   # 6. sync report
tail -1 cues4_report.txt
