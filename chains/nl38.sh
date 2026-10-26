#!/bin/bash
# nl38, one GPU: moodscan.py only measures and never trains. It reads nl22c's step_157 (chat training), nl33's step_300
# (the steered-reading game), nl33w's step_300 (the same 300 steps without the game) and beautyrl6's step_28, each also
# with both adapters off (the original models).
# Inputs: CODE_TGZ (pairing/ and data/pools.json), INPUTS_TAR (step_157), CK33T (nl33's step_300), CK33W (nl33w's
# step_300), CKRL6 (beautyrl6's step_28).
# Usage: bash nl38.sh OUT_DIR CODE_TGZ INPUTS_TAR CK33T CK33W CKRL6
set -u
O=$1 TGZ=$2 INPUTS=$3 C33T=$4 C33W=$5 CRL=$6
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset NL_TINY NL_MAXNEW NL_AFFECT_N
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/moodscan
mkdir -p $J/a $J/t $J/w $J/r
for pair in "$INPUTS a" "$C33T t" "$C33W w" "$CRL r"; do
    set -- $pair
    tar --no-same-owner --exclude='._*' -xf $1 -C $J/$2 || { note "bad input $1"; exit 1; }
done
[ -s $J/a/step_157/bridge.pt ] && [ -s $J/t/step_300/bridge.pt ] && [ -s $J/w/step_300/bridge.pt ] \
    && [ -s $J/r/step_28/bridge.pt ] && [ -s data/pools.json ] || { note 'missing'; exit 1; }
note 'moodscan start'
(cd pairing && python moodscan.py --run $J/a/step_157 --run $J/t/step_300 --run $J/w/step_300 --run $J/r/step_28 \
    --names step157 nl33t nl33w beautyrl --out $O/results) > $O/moodscan.log 2>&1 \
    || { note "moodscan failed $?"; exit 1; }
note 'all done'
