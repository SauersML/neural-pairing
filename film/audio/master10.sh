#!/bin/sh
# Master the raw Patina bounce of score10.song into score10.wav (same chain as master3.sh):
# 24 Hz HPF, gain to about -14 LUFS, a peak limiter at -1.5 dBFS (transients only), a 0.3 s
# fade-in, the closing fade 115.8 -> 117.0 s (cues/a6_proof.md: fade to black 115.8-117), then trim/pad to exactly 117.000 s, 48 kHz, 24-bit.
#   sh master10.sh work10/raw10.wav GAIN_DB
set -e
IN=${1:-work10/raw10.wav}; GAIN=${2:-2.4}
nice -n 10 ffmpeg -v error -y -i "$IN" -af "highpass=f=24:poles=2,volume=${GAIN}dB,alimiter=limit=0.84:attack=1:release=60:level=disabled,afade=t=in:st=0:d=0.3,afade=t=out:st=115.8:d=1.2:curve=qsin,apad=whole_dur=117,atrim=end_sample=5616000" \
  -ar 48000 -ac 2 -c:a pcm_s24le score10.wav
ffmpeg -hide_banner -nostats -i score10.wav -af ebur128=peak=true -f null - 2>&1 | grep -A16 Summary | grep -E "I:|LRA:|Peak:"
ffprobe -v error -show_entries stream=sample_rate,channels,duration_ts,duration -of compact score10.wav
