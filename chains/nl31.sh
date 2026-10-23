#!/bin/bash
# magic-nl31, one 48 GB Ampere card (RTX A6000 / A40; never Ada): how far does the link move the partner along its own
# concept direction when one twin is steered? steer_probe.py on nl22c step 157, over the prompt pass of nl30's chat
# episodes, for nl30's five settings (after block 6, 15 and 27, and at four times the strength after blocks 15 and 27),
# read at depths 6, 15 and 27 with the directions nl30 fit there (block 15's are nl29's, value for value).
# Inputs: INPUTS_TAR, the published tar of step_157 (nl30's).
# Usage: bash nl31.sh OUT_DIR CODE_TGZ INPUTS_TAR
set -u
O=$1 TGZ=$2 INPUTS=$3
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl31
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J || { note "bad $INPUTS"; exit 1; }
D=results/magic-nl30/results
note 'probe start'
python pairing/steer_probe.py --run $J/step_157 --depth 6:$D/l6 15:$D/l15x4 27:$D/l27 --arm 6:1 15:1 15:4 27:1 27:4 \
    --out $O/results --bs 6 > $O/probe.log 2>&1
note "probe exit $?"
