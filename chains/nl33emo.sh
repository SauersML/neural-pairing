#!/bin/bash
# nl33emo, one 48 or 80 GB card: the steered-twin test with emotions (steering_scale.py --concepts emotions:
# steered_mind.EMOTIONS, six emotions that are not steered-reading training concepts) on nl33t's step_300 (nl33.sh, arm
# steer), after block 27 at the calibrated strength and at four times it, and after block 15 at four times it, two at a
# time on 48 GB and three on 80 GB (one that runs out of memory is rerun alone). Nothing is trained.
# Inputs: CODE_TGZ (pairing/ and data/pools.json), CKPT_TAR (nl33t's step_300).
# Usage: bash nl33emo.sh OUT_DIR CODE_TGZ CKPT_TAR
set -u
O=$1 TGZ=$2 CK=$3
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8         # torch's default thread count stalls on many-core hosts
mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
J=/workspace/jobs/nl33emo
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $CK -C $J && [ -s $J/step_300/bridge.pt ] \
    || { note "bad $CK"; exit 1; }
R=$O/results
mkdir -p $R
steer() {  # TAG LAYER BOOST
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --concepts emotions --layer $2 --boost $3 \
        --run $J/step_300 --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
}
ARMS="l27x4:27:4 l27:27:1 l15x4:15:4"
K=2; [ "$MEM" -ge 79000 ] && K=3
set -- $ARMS
while [ $# -gt 0 ]; do
    for _ in $(seq $K); do
        [ $# -gt 0 ] || break
        IFS=: read -r t l b <<< "$1"; steer $t $l $b &
        shift
    done
    wait
done
for a in $ARMS; do
    IFS=: read -r t l b <<< "$a"
    [ -f $R/$t.DONE ] && continue
    grep -q "out of memory" $R/$t.log || continue
    mv $R/$t $R/$t.oom && mv $R/$t.log $R/$t.oom.log
    steer $t $l $b
done
note 'all done'
