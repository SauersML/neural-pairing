#!/bin/sh
# Master the raw Patina bounce of score4.song into score4.wav (same chain as master3.sh):
# 24 Hz HPF, gain to about -14 LUFS, a peak limiter at -1.5 dBFS (transients only), a 0.3 s
# fade-in, the closing fade 94.6 -> 95.9 s, then trim/pad to exactly 96.000 s, 48 kHz, 24-bit.
#   sh master4.sh work4/raw4.wav GAIN_DB
set -e
IN=${1:-work4/raw4.wav}; GAIN=${2:-2.4}
nice -n 10 ffmpeg -v error -y -i "$IN" -af "highpass=f=24:poles=2,volume=${GAIN}dB,alimiter=limit=0.84:attack=1:release=60:level=disabled,afade=t=in:st=0:d=0.3,afade=t=out:st=94.6:d=1.3:curve=qsin,apad=whole_dur=96,atrim=end_sample=4608000" \
  -ar 48000 -ac 2 -c:a pcm_s24le score4.wav
ffmpeg -hide_banner -nostats -i score4.wav -af ebur128=peak=true -f null - 2>&1 | grep -A16 Summary | grep -E "I:|LRA:|Peak:"
ffprobe -v error -show_entries stream=sample_rate,channels,duration_ts,duration -of compact score4.wav
