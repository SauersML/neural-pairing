#!/bin/bash
# nl23, one to four 48+ GB cards: three arms trained for STEPS each from nl20's step_1320 on ARC-Challenge and MMLU
# training questions only: the pair with the link open (--qa 1), each model alone (--solo), and the pair answering in
# turn in text with the link closed (--text_pair); one card per arm with three or more cards, else one after another.
# bench_eval.py then scores every arm and control on the test questions over all cards. Waits up to an hour for
# /workspace/jobs/nl23/{step_1320, replies.json, data/} and the READY file written after them.
# Usage: bash nl23.sh OUT_DIR CODE_TGZ STEPS
set -u
O=$1 TGZ=$2 STEPS=$3
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl23
note "waiting for $J/READY"
for _ in $(seq 360); do [ -f $J/READY ] && break; sleep 10; done
[ -f $J/READY ] || { note "no inputs after an hour"; exit 1; }
cp -rf $J/data/. ../data/ && cp $J/replies.json $O/replies.json || exit 1
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; n=$1; shift; "$@" > $R/$n.log 2>&1 && touch $R/$n.DONE; note "$n: exit $?"; }
Q="--replies $O/replies.json --init $J/step_1320 --gate_norm --hold 100000 --warmup_steps 10 --relay_first 0 --steps $STEPS
   --bs 16 --micro 4 --anchor 0 --comp 0 --warm 0 --ramp 0 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3
   --eval_every 100000 --first_eval 100000 --save_every 100000 --ckpt none"
N=$(nvidia-smi -L | wc -l)
arm() { ev qa_$1 env CUDA_VISIBLE_DEVICES=$2 python train.py $Q --out $O/qa_$1 $3; }
if [ $N -ge 3 ]; then
    note "three arms, $STEPS steps each, one card each"
    arm pair 0 "--qa 1" & arm solo 1 --solo & arm text 2 --text_pair &
    wait
else
    note "three arms, $STEPS steps each, one after another"
    arm pair 0 "--qa 1"; arm solo 0 --solo; arm text 0 --text_pair
fi
for a in pair solo text; do [ -d $O/qa_$a/step_$STEPS ] || { note "no step_$STEPS for $a"; exit 1; }; done
ev bench torchrun --standalone --nproc_per_node $N bench_eval.py --pair $O/qa_pair/step_$STEPS --solo $O/qa_solo/step_$STEPS \
    --text $O/qa_text/step_$STEPS --out $R/bench.json
note "all done"
