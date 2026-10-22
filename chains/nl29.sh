#!/bin/bash
# magic-nl29, one 48 GB Ampere card (RTX A6000 / A40; never Ada): the steered-mind test in bf16 with the natural-projection
# calibration (steering_scale.py). x25 injected each concept direction at a strength set by its RMS, 3-23 natural gaps on
# the first step and dominated by Ember's four residual sink coordinates, so Ember repeated instructions and no partner
# test could pass. steering_scale.py removes those sinks and steps in natural gaps; on 4-bit MLX it calibrates 12/12
# Quill and 11/12 Ember concepts with the link cut isolating the partner (results/steering_scale.json). Here, in bf16:
# fit, calibrate, cut check and steered_mind's unchanged evaluation, before (nl20 step 1,320) and after chat training
# (nl22c step 157), the two at once; one that runs out of memory is rerun alone in a fresh directory.
# Inputs from the Mac after the pod starts: /workspace/jobs/nl29/{step_1320,step_157}, READY last.
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
