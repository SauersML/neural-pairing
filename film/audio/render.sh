#!/bin/sh
# Rebuild score.wav from source. Run from /home/user/video/audio (patches/ and
# fx.wav are resolved relative to it).
set -e
cd "$(dirname "$0")"
P=/home/user/patina/target/release/patina
mkdir -p work
$P --play fx.song --render fx.wav --no-normalize          # 1. noise FX stem
$P --play score.song --render work/raw.wav --no-normalize  # 2. the score (plays fx.wav on the tape deck)
sh master.sh work/raw.wav 4.8                              # 3. -14 LUFS, limiter, fade, exact 60.000 s
python3 cues.py score.wav                                  # 4. sync verification
