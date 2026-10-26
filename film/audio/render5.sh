#!/bin/sh
# Rebuild score5.wav from source. Run from /home/user/video/audio. Everything runs under nice,
# one render at a time (the machine is shared).
set -e
cd "$(dirname "$0")"
P="nice -n 10 /home/user/patina/target/release/patina"
mkdir -p work5
python3 make_score5.py                                              # 0. generate score5.song + fx5.song
$P --play fx5.song --render fx5.wav                                 # 1. noise FX stem (normalized to -1 dBFS)
$P --play score5.song --render work5/raw5.wav --no-normalize        # 2. the score
$P --play score5.song --export-events work5/ev5.json                # 3. exact event times
sh master5.sh work5/raw5.wav ${GAIN:-2.7}                           # 4. -14 LUFS, limiter, fades, 104.000 s
if [ -z "$NOSTEMS" ]; then
for t in quill ember ghost glass glassq felt; do                                     # 5. per-voice stems for the pitched cues
  python3 solo.py score5.song work5/s5_$t.song $t
  $P --play work5/s5_$t.song --render work5/s5_$t.wav --no-normalize >/dev/null
done
fi
python3 cues5.py score5.wav work5/ev5.json work5 > cues5_report.txt   # 6. sync report
tail -1 cues5_report.txt
