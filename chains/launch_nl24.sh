#!/bin/bash
# Lead-only launch helper. Stages the LANDED source, then invokes the existing rp-run.
# In a second terminal after starting this: bash chains/send_nl24.sh magic-nl22c magic-nl24
set -euo pipefail
REPO=/Users/user/neural-pairing
STAGE=/Users/user/mpd-data/neural-pairing
mkdir -p "$STAGE"
git -C "$REPO" fetch -q origin main
git -C "$REPO" archive origin/main pairing data chains/nl24.sh | gzip > "$STAGE/code24.tgz"
git -C "$REPO" show origin/main:chains/nl24.sh > "$STAGE/nl24.sh"
cd /Users/user/params
RP_OWNER=magic RP_PARALLEL=1 RP_PACK=0 RP_SELF_STOP=1 RP_PYENV=oracle \
RP_HF_MODELS='Qwen/Qwen3-4B arcee-ai/AFM-4.5B' RP_MIN_VRAM_GB=48 RP_MIN_RAM_GB=64 RP_DISK_GB=120 \
bash experiments/infra/runpod/rp-run magic-nl24 'NVIDIA L40S,NVIDIA RTX A6000,NVIDIA A40' 8 -- \
    bash /Users/user/mpd-data/neural-pairing/nl24.sh /Users/user/mpd-data/runpod/magic-nl24 \
    /Users/user/mpd-data/neural-pairing/code24.tgz
