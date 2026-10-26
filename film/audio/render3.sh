#!/bin/sh
# Rebuild score3.wav from source. Run from /home/user/video/audio.
set -e
cd "$(dirname "$0")"
P=/home/user/patina/target/release/patina
mkdir -p work3
python3 make_score3.py                                              # 0. generate score3.song + fx3.song
$P --play fx3.song --render fx3.wav                                 # 1. noise FX stem (normalized to -1 dBFS)
$P --play score3.song --render work3/raw3.wav --no-normalize        # 2. the score
$P --play score3.song --export-events work3/ev3.json                # 3. exact event times
sh master3.sh work3/raw3.wav ${GAIN:-2.4}                           # 4. -14 LUFS, limiter, fades, 82.000 s
for t in quill ember ghost; do                                      # 5. per-voice stems for the pitched cues
  python3 solo.py score3.song work3/s3_$t.song $t
  $P --play work3/s3_$t.song --render work3/s3_$t.wav --no-normalize >/dev/null
done
python3 cues3.py score3.wav work3/ev3.json work3 > cues3_report.txt   # 6. sync report
tail -1 cues3_report.txt
