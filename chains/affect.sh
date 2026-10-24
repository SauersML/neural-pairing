#!/bin/bash
# magic-nl36: do the twins love each other and feel happy together when linked, without being trained to? affect.py
# (frozen claims in its docstring) reads, never trains: each twin's own happy/sad and love/cold directions (held-out
# validated) while it reads the same conversations linked, cut and with a stranger's states, and its self-report
# (good vs bad words after 'I feel', enjoy and bother yes-or-no). Checkpoints: step_157 (before the steered and chat
# training: the baseline), nl33's step_300, the beauty-RL pair (magic-beautyrl6 step_28) and nl35's step_300 (four
# question frames).
# Inputs: CODE_TGZ (pairing/), INPUTS_TAR (step_157), CKPT_TAR (nl33 step_300), RL_TAR (step_28), NL35_TAR (step_300),
# all but the code published.
# Usage: bash affect.sh OUT_DIR CODE_TGZ INPUTS_TAR CKPT_TAR RL_TAR NL35_TAR
set -u
O=$1 TGZ=$2 INPUTS=$3 CK=$4 RL=$5 N35=$6
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset NL_TINY NL_MAXNEW NL_AFFECT_N
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/affect
mkdir -p $J/a $J/b $J/c $J/d
tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J/a && tar --no-same-owner --exclude='._*' -xf $CK -C $J/b \
    && tar --no-same-owner --exclude='._*' -xf $RL -C $J/c && tar --no-same-owner --exclude='._*' -xf $N35 -C $J/d \
    || { note 'bad inputs'; exit 1; }
[ -s $J/a/step_157/bridge.pt ] && [ -s $J/b/step_300/bridge.pt ] && [ -s $J/c/step_28/bridge.pt ] \
    && [ -s $J/d/step_300/bridge.pt ] || { note 'missing'; exit 1; }
note 'affect start'
(cd pairing && python affect.py --run $J/a/step_157 --run $J/b/step_300 --run $J/c/step_28 --run $J/d/step_300 \
    --names step157 nl33t beautyrl nl35 --out $O/results) > $O/affect.log 2>&1 || { note "affect failed $?"; exit 1; }
note 'all done'
