#!/bin/bash
# Full render: parallel workers -> PNG frames -> H.264 + score. Usage: tools/make_film.sh [workers] [fps]
set -e
cd "$(dirname "$0")/.."
W=${1:-3}; FPS=${2:-30}
mkdir -p out/frames
for k in $(seq 0 $((W-1))); do node tools/render.mjs --from 0 --to 60 --fps $FPS --out out/frames --workers $W --worker $k > out/render_$k.log 2>&1 & done
wait
n=$(ls out/frames/f_*.png | wc -l); echo "frames: $n"
AUD=audio/score.wav
ffmpeg -y -loglevel error -framerate $FPS -i out/frames/f_%05d.png ${AUD:+-i $AUD} \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -tune animation -movflags +faststart \
  -c:a aac -b:a 256k -shortest out/neural_pairing.mp4
ls -la out/neural_pairing.mp4
