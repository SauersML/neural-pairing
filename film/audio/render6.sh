#!/bin/sh
# Rebuild score6.wav from source. Run from /home/user/video/audio. Everything runs under nice,
# one render at a time (the machine is shared).
set -e
cd "$(dirname "$0")"
P="nice -n 10 /home/user/patina/target/release/patina"
mkdir -p work6
python3 make_score6.py                                              # 0. generate score6.song + fx6.song
$P --play fx6.song --render fx6.wav                                 # 1. noise FX stem (normalized to -1 dBFS)
$P --play score6.song --render work6/raw6.wav --no-normalize        # 2. the score
$P --play score6.song --export-events work6/ev6.json                # 3. exact event times
sh master6.sh work6/raw6.wav ${GAIN:-2.5}                           # 4. -14 LUFS, limiter, fades, 116.000 s
if [ -z "$NOSTEMS" ]; then
for t in quill ember ghost glass glassq felt; do                                     # 5. per-voice stems for the pitched cues
  python3 solo.py score6.song work6/s6_$t.song $t
  $P --play work6/s6_$t.song --render work6/s6_$t.wav --no-normalize >/dev/null
done
fi
python3 cues6.py score6.wav work6/ev6.json work6 > cues6_report.txt   # 6. sync report
tail -1 cues6_report.txt
