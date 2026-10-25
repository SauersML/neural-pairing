#!/bin/bash
# magic-nl37: do the twins prefer their own partner to a stranger, and are they happier linked, asked in ways that can
# show it? affect_partner.py (frozen claims in its docstring) reads, never trains: mirrored yes-or-no pairs (keep or
# switch partner, miss or lonely, close or distant) and a one-word mood read, linked, cut and with a stranger's states,
# on 60 fresh conversations, and the inside directions after blocks 22, 28 and 33 (28 and 33 follow every link write).
# Checkpoints: step_157 (before the steered and chat training) and nl33's step_300.
# Inputs: CODE_TGZ (pairing/, data/pools.json), INPUTS_TAR (step_157, published), CKPT_TAR (nl33 step_300, published).
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
