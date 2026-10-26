#!/bin/bash
# nl29, one 48 GB Ampere card (RTX A6000 or A40): the steered-twin test in bf16 (steering_scale.py --phase all: fit the
# concept directions, calibrate the steering coefficient in natural gaps with Ember's residual sink coordinates removed,
# check with the link cut, then steered_mind's evaluation) before (nl20's step_1320) and after chat training (nl22c's
# step_157), both at once; one that runs out of memory is rerun alone in a fresh directory. Waits up to an hour for
# /workspace/jobs/nl29/{step_1320, step_157} and the READY file.
# Usage: bash nl29.sh OUT_DIR CODE_TGZ
set -u
O=$1 TGZ=$2
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl29
note "waiting for $J/READY"
for _ in $(seq 360); do [ -f $J/READY ] && break; sleep 10; done
[ -f $J/READY ] || { note 'no inputs after an hour'; exit 1; }
R=$O/results
mkdir -p $R
steer() {  # TAG CHECKPOINT
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --run $2 --out $R/$1 > $R/$1.log 2>&1 \
        && touch $R/$1.DONE
    note "$1 exit $?"
}
steer before $J/step_1320 &
steer after $J/step_157 &
wait
for tag in before after; do
    [ -f $R/$tag.DONE ] && continue
    grep -q "out of memory" $R/$tag.log || continue
    mv $R/$tag $R/$tag.oom && mv $R/$tag.log $R/$tag.oom.log
    [ $tag = before ] && steer before $J/step_1320 || steer after $J/step_157
done
note 'all done'
