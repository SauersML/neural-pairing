#!/bin/sh
# Momentary loudness (LUFS, 400 ms) sampled every 0.5 s: sh lufs.sh file.wav
ffmpeg -v error -i "$1" -af "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M:file=-" -f null - 2>/dev/null |
 awk '/pts_time/{split($0,a,"pts_time:"); t=a[2]+0} /lavfi.r128.M=/{split($0,b,"="); if (t>=next_t){printf "%5.1f:%6.1f  ", t, b[2]; n++; if(n%8==0) print ""; next_t=t+0.5}} END{print ""}'
