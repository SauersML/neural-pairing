#!/bin/bash
# magic-nl32, one 48 GB Ampere card (RTX A6000 / A40; never Ada): is a steered concept stuck after block 27 only because
# the link writes too little there? nl30 steered one twin after block 27 at the calibrated strength and at four times
# it, and the partner named the concept in 0 of 96 linked messages either way; nl31 measured why: the write after block
# 27 moves the partner 0.02-0.10 of the push that makes it say the concept, against 0.20-0.35 by block 27 for the early
# pushes that cross. The write size tanh(alpha) has been held at 0.1 at every exchange in every run (--hold). Here the
# same test (steering_scale.py, bf16, nl22c step 157) with only the write after block 27 made 3 or 9 times larger
# (--gain; tanh(alpha) 0.3 or 0.9 there, both readers, nothing retrained), at both strengths, reusing nl30's directions,
# calibration, selection and cut check (--reuse; all link-cut). Two arms at a time; one that runs out of memory is
# rerun alone in a fresh directory.
# Inputs: INPUTS_TAR, the published tar of step_157 (nl30's).
# Usage: bash nl32.sh OUT_DIR CODE_TGZ INPUTS_TAR
set -u
O=$1 TGZ=$2 INPUTS=$3
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl32
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J || { note "bad $INPUTS"; exit 1; }
D=results/magic-nl30/results
R=$O/results
mkdir -p $R
steer() {  # TAG BOOST GAIN REUSE
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer 27 --boost $2 --gain $3 --reuse $D/$4 \
        --run $J/step_157 --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
}
ARMS="g9:1:9:l27 x4g9:4:9:l27x4 g3:1:3:l27 x4g3:4:3:l27x4"
set -- $ARMS
while [ $# -gt 0 ]; do
    for a in $1 ${2:-}; do IFS=: read -r t b g r <<< "$a"; steer $t $b $g $r & done
    wait
    shift; [ $# -gt 0 ] && shift
done
for a in $ARMS; do
    IFS=: read -r t b g r <<< "$a"
    [ -f $R/$t.DONE ] && continue
    grep -q "out of memory" $R/$t.log || continue
    mv $R/$t $R/$t.oom && mv $R/$t.log $R/$t.oom.log
    steer $t $b $g $r
done
note 'all done'
