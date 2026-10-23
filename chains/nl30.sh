#!/bin/bash
# magic-nl30, one 48 GB Ampere card (RTX A6000 / A40; never Ada): does a steered concept cross the link at another depth
# or a larger strength? nl29 steered after block 15 at the smallest strength that makes the steered twin say the concept
# (27-30 of 36), and the partner said it in 1 of 96 linked messages after chat training. Here the same test
# (steering_scale.py, bf16, nl22c step 157) in four arms: after block 6 (every read depth sees the change), after block 27
# (the last read depth), and four times the calibrated strength after blocks 15 and 27 (--boost 4: cut check and
# evaluation only; the fit, calibration and selection are unchanged). Two arms at a time; one that runs out of memory is
# rerun alone in a fresh directory.
# Inputs from the Mac after the pod starts: /workspace/jobs/nl30/step_157, READY last.
# Usage: bash nl30.sh OUT_DIR CODE_TGZ
set -u
O=$1 TGZ=$2
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl30
note "waiting for $J/READY"
for _ in $(seq 540); do [ -f $J/READY ] && break; sleep 10; done
[ -f $J/READY ] || { note 'no inputs after 90 minutes'; exit 1; }
R=$O/results
mkdir -p $R
steer() {  # TAG LAYER BOOST
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer $2 --boost $3 --run $J/step_157 \
        --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
}
ARMS="l6:6:1 l27:27:1 l15x4:15:4 l27x4:27:4"
set -- $ARMS
while [ $# -gt 0 ]; do
    for a in $1 ${2:-}; do IFS=: read -r t l b <<< "$a"; steer $t $l $b & done
    wait
    shift; [ $# -gt 0 ] && shift
done
for a in $ARMS; do
    IFS=: read -r t l b <<< "$a"
    [ -f $R/$t.DONE ] && continue
    grep -q "out of memory" $R/$t.log || continue
    mv $R/$t $R/$t.oom && mv $R/$t.log $R/$t.oom.log
    steer $t $l $b
done
note 'all done'
