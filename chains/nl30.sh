#!/bin/bash
# nl30, one 48 GB Ampere card (RTX A6000 or A40): the steered-twin test (steering_scale.py, bf16, nl22c's step_157) in
# four arms: steered after block 6 (every read depth comes after it), after block 27 (the last read depth), and at four
# times the calibrated strength after blocks 15 and 27 (--boost 4, applied in the cut check and the evaluation; the fit,
# calibration and selection are unchanged). Two arms at a time; one that runs out of memory is rerun alone in a fresh
# directory.
# Inputs: INPUTS_TAR (a tar of step_157), else /workspace/jobs/nl30/step_157 and the READY file, waited for up to 90
# minutes.
# Usage: bash nl30.sh OUT_DIR CODE_TGZ [INPUTS_TAR]
set -u
O=$1 TGZ=$2 INPUTS=${3:-}
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl30
if [ -n "$INPUTS" ]; then
    mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J && touch $J/READY || { note "bad $INPUTS"; exit 1; }
fi
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
