#!/bin/bash
# The twins' chat server (pairing/chat_live.py) on one GPU: the final twins from Hugging Face (Sauers/neural-pairing)
# in bf16, answering on 127.0.0.1:PORT only, for an SSH port forward (ssh -N -L PORT:127.0.0.1:PORT the GPU host).
# It exits after 30 minutes without a request; on a RunPod pod (RUNPOD_POD_ID in the container's environment) the
# pod then deletes itself, so it costs nothing once nobody is talking.
# Inputs: CODE_TGZ (pairing/, data/, figures/assets/). Usage: bash chat_serve.sh OUT_DIR CODE_TGZ [PORT]
set -u
O=$1 TGZ=$2 PORT=${3:-8765}
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NL_TINY
mkdir -p $O /workspace/chat && tar --no-same-owner -xzf $TGZ -C /workspace/chat && cd /workspace/chat/pairing || exit 1
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
note "chat start on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
python chat_live.py --port $PORT --idle-exit 1800 > $O/chat.log 2>&1
note "chat exited $?"
eval "$(tr '\0' '\n' < /proc/1/environ 2> /dev/null | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    note "deleting pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl pod delete "$RUNPOD_POD_ID"
fi
