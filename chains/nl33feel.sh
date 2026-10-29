#!/bin/bash
# nl33feel, one 48 GB card: the open-ended feeling questions (steer_feel.py) with the emotions of nl33emo
# (steered_mind.EMOTIONS). On nl33t's step_300 at nl33emo's selected strengths after block 27 and block 15, four times
# each (SCALES_TAR: nl33emo's l27x4 and l15x4 directions, selection and amendment); then on the final twins (nl33w, from
# Hugging Face Sauers/neural-pairing, never trained on the steered game) the steered-twin test with the emotions after
# block 6 and at four times the strength after block 15 (steering_scale.py --concepts emotions, both at once), and the
# questions at both. Nothing is trained.
# Inputs: CODE_TGZ (pairing/ and data/pools.json), CKPT_TAR (nl33t's step_300), SCALES_TAR.
# Usage: bash nl33feel.sh OUT_DIR CODE_TGZ CKPT_TAR SCALES_TAR
set -u
O=$1 TGZ=$2 CK=$3 SC=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl33feel
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $CK -C $J && tar --no-same-owner --exclude='._*' -xf $SC -C $J \
    && [ -s $J/step_300/bridge.pt ] && [ -s $J/scales/l27x4/directions.npz ] || { note 'bad inputs'; exit 1; }
FINAL=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Sauers/neural-pairing'))") \
    && [ -s $FINAL/bridge.safetensors ] || { note 'no final twins'; exit 1; }
R=$O/results
mkdir -p $R
feel() {  # TAG RUN SCALE...
    local t=$1 run=$2; shift 2
    note "feel $t start"
    python pairing/steer_feel.py --run $run $(printf -- '--scale %s ' "$@") --out $R/feel_$t > $R/feel_$t.log 2>&1
    note "feel $t exit $?"
}
steer() {  # TAG LAYER BOOST
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --concepts emotions --layer $2 --boost $3 \
        --run $FINAL --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
}
feel nl33t $J/step_300 $J/scales/l27x4 $J/scales/l15x4
steer w_l6 6 1 &
steer w_l15x4 15 4 &
wait
feel nl33w $FINAL $R/w_l6 $R/w_l15x4
note 'all done'
