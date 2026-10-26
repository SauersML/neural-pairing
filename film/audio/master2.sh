#!/bin/sh
# Master the raw Patina bounce of score2.song into score2.wav: 24 Hz HPF,
# gain to about -14 LUFS, a peak limiter at -1.5 dBFS (transients only), a
# 0.3 s fade-in (hides the engine's power-on thump under the swell), the
# closing fade 80.6 -> 81.9 s, then trim/pad to exactly 82.000 s, 48 kHz, 24-bit.
#   sh master2.sh work/raw2.wav GAIN_DB
set -e
IN=${1:-work/raw2.wav}; GAIN=${2:-6.0}
ffmpeg -v error -y -i "$IN" -af "highpass=f=24:poles=2,volume=${GAIN}dB,alimiter=limit=0.84:attack=1:release=60:level=disabled,afade=t=in:st=0:d=0.3,afade=t=out:st=80.6:d=1.3:curve=qsin,apad=whole_dur=82,atrim=end_sample=3936000" \
  -ar 48000 -ac 2 -c:a pcm_s24le score2.wav
ffmpeg -hide_banner -nostats -i score2.wav -af ebur128=peak=true -f null - 2>&1 | grep -A16 Summary | grep -E "I:|LRA:|Peak:"
ffprobe -v error -show_entries stream=sample_rate,channels,duration_ts,duration -of compact score2.wav
