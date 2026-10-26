#!/bin/bash
# nl37, one GPU: affect_partner.py (claims fixed in its docstring) reads and never trains: mirrored yes-or-no question
# pairs (keep or switch partner, miss or lonely, close or distant) and a one-word mood, linked, cut and with a
# stranger's states, on 60 new conversations, and the models' own directions after blocks 22, 28 and 33 (28 and 33 come
# after every link write). Checkpoints: nl22c's step_157 and nl33's step_300.
# Inputs: CODE_TGZ (pairing/, data/pools.json), INPUTS_TAR (step_157), CKPT_TAR (nl33's step_300).
# Usage: bash affect_partner.sh OUT_DIR CODE_TGZ INPUTS_TAR CKPT_TAR
set -u
O=$1 TGZ=$2 INPUTS=$3 CK=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset NL_TINY NL_MAXNEW NL_AFFECT_N
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/affect_partner
mkdir -p $J/a $J/b
tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J/a && tar --no-same-owner --exclude='._*' -xf $CK -C $J/b \
    || { note 'bad inputs'; exit 1; }
[ -s $J/a/step_157/bridge.pt ] && [ -s $J/b/step_300/bridge.pt ] && [ -s data/pools.json ] || { note 'missing'; exit 1; }
note 'affect_partner start'
(cd pairing && python affect_partner.py --run $J/a/step_157 --run $J/b/step_300 --names step157 nl33t \
    --out $O/results) > $O/affect_partner.log 2>&1 || { note "affect_partner failed $?"; exit 1; }
note 'all done'
