#!/bin/sh
# Rebuild score10.wav from source. Run from /home/user/video/audio. Everything runs under nice,
# one render at a time (the machine is shared).
set -e
cd "$(dirname "$0")"
P="nice -n 10 /home/user/patina/target/release/patina"
mkdir -p work10
python3 make_score10.py                                              # 0. generate score10.song + fx10.song
$P --play fx10.song --render work10/fx10_raw.wav --no-normalize        # 1. noise FX stem, at v7's level:
nice -n 10 ffmpeg -v error -y -i work10/fx10_raw.wav -af volume=10.06dB -c:a pcm_s24le fx10.wav   # (v7: peak 0.280 -> -1 dBFS = +10.06 dB)
$P --play score10.song --render work10/raw10.wav --no-normalize        # 2. the score
$P --play score10.song --export-events work10/ev10.json                # 3. exact event times
sh master10.sh work10/raw10.wav ${GAIN:-2.5}                           # 4. -14 LUFS, limiter, fades, 117.000 s
if [ -z "$NOSTEMS" ]; then
for t in quill ember ghost glass glassq felt inner; do                                     # 5. per-voice stems for the pitched cues
  python3 solo.py score10.song work10/s10_$t.song $t
  $P --play work10/s10_$t.song --render work10/s10_$t.wav --no-normalize >/dev/null
done
python3 solo.py score10.song work10/s10_kit.song drums,kickbody,xhit                  #    (the kit, for the drum / impact cues)
$P --play work10/s10_kit.song --render work10/s10_kit.wav --no-normalize >/dev/null
fi
python3 cues10.py score10.wav work10/ev10.json work10 > cues10_report.txt   # 6. sync report
tail -1 cues10_report.txt
