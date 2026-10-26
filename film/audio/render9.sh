#!/bin/sh
# Rebuild score9.wav from source. Run from /home/user/video/audio. Everything runs under nice,
# one render at a time (the machine is shared).
set -e
cd "$(dirname "$0")"
P="nice -n 10 /home/user/patina/target/release/patina"
mkdir -p work9
python3 make_score9.py                                              # 0. generate score9.song + fx9.song
$P --play fx9.song --render work9/fx9_raw.wav --no-normalize        # 1. noise FX stem, at v7's level:
nice -n 10 ffmpeg -v error -y -i work9/fx9_raw.wav -af volume=10.06dB -c:a pcm_s24le fx9.wav   # (v7: peak 0.280 -> -1 dBFS = +10.06 dB)
$P --play score9.song --render work9/raw9.wav --no-normalize        # 2. the score
$P --play score9.song --export-events work9/ev9.json                # 3. exact event times
sh master9.sh work9/raw9.wav ${GAIN:-2.5}                           # 4. -14 LUFS, limiter, fades, 121.000 s
if [ -z "$NOSTEMS" ]; then
for t in quill ember ghost glass glassq felt inner; do                                     # 5. per-voice stems for the pitched cues
  python3 solo.py score9.song work9/s9_$t.song $t
  $P --play work9/s9_$t.song --render work9/s9_$t.wav --no-normalize >/dev/null
done
fi
python3 cues9.py score9.wav work9/ev9.json work9 > cues9_report.txt   # 6. sync report
tail -1 cues9_report.txt
