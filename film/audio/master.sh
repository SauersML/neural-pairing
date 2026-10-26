#!/bin/sh
# Master the raw Patina bounce into score.wav: gain to about -14 LUFS, a
# transparent peak limiter at -1.5 dBFS (it only touches kick/clap
# transients), the closing fade (58.6 -> 59.9 s), then trim/pad to exactly
# 60.000 s at 48 kHz stereo, 24-bit.
#   sh master.sh work/raw.wav GAIN_DB
set -e
IN=${1:-work/raw.wav}; GAIN=${2:-5.0}
ffmpeg -v error -y -i "$IN" -af "highpass=f=24:poles=2,volume=${GAIN}dB,alimiter=limit=0.84:attack=1:release=60:level=disabled,afade=t=out:st=58.6:d=1.3:curve=qsin,apad=whole_dur=60,atrim=end_sample=2880000" \
  -ar 48000 -ac 2 -c:a pcm_s24le score.wav
ffmpeg -hide_banner -nostats -i score.wav -af ebur128=peak=true -f null - 2>&1 | grep -A16 Summary | grep -E "I:|LRA:|Peak:|dBFS"
ffprobe -v error -show_entries stream=sample_rate,channels,duration_ts,duration -of compact score.wav
