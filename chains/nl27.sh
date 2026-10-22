#!/bin/bash
# magic-nl27, one 48+ GB card: can the link pool two models' answers rather than copy the stronger one? In nl23 the pair
# trained on benchmark questions drawn uniformly and the link learned deference: Ember takes Quill's letter, the pair
# scores as Quill alone, below the two models' letters averaged. Here half the training questions are ones only one
# original model answers right, each model's set equally often (qa_split.py, train.py --qa_focus), so on those following
# the partner is right as often as wrong. Two arms from nl20 step 1,320, STEPS each, the same draws distribution: the pair
# with the link open (--qa 1) and each model alone (--solo). bench_eval.py scores them with nl23's text pair as the text
# arm. The new headline is pooled: pair/avg (the two linked models' letter probabilities averaged) against ensemble (the
# two single models' averaged) and against pair/zero/avg (the same pair, link cut).
# Inputs from the Mac after the pod starts (send_job.sh layout): /workspace/jobs/nl27/{step_1320, text_step_300,
# replies.json, data/bench.json}, READY last.
# Usage: bash nl27.sh OUT_DIR CODE_TGZ STEPS
set -u
O=$1 TGZ=$2 STEPS=$3
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl27
note "waiting for $J/READY"
for _ in $(seq 360); do [ -f $J/READY ] && break; sleep 10; done
[ -f $J/READY ] || { note "no inputs after an hour"; exit 1; }
cp -rf $J/data/. ../data/ && cp $J/replies.json $O/replies.json || exit 1
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; n=$1; shift; "$@" > $R/$n.log 2>&1 && touch $R/$n.DONE; note "$n: exit $?"; }
ev split python qa_split.py --run $J/step_1320 --out $R/qa_split.json
[ -s $R/qa_split.json ] || { note "no question split"; exit 1; }
Q="--replies $O/replies.json --init $J/step_1320 --gate_norm --hold 100000 --warmup_steps 10 --relay_first 0 --steps $STEPS
   --bs 16 --micro 4 --anchor 0 --comp 0 --warm 0 --ramp 0 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3
   --eval_every 100000 --first_eval 100000 --save_every 100000 --ckpt none --qa_focus $R/qa_split.json --qa_focus_p 0.5"
note "two arms, $STEPS steps each, one after another"
ev qa_pair python train.py $Q --out $O/qa_pair --qa 1
ev qa_solo python train.py $Q --out $O/qa_solo --solo
for a in pair solo; do [ -d $O/qa_$a/step_$STEPS ] || { note "no step_$STEPS for $a"; exit 1; }; done
ev bench python bench_eval.py --pair $O/qa_pair/step_$STEPS --solo $O/qa_solo/step_$STEPS --text $J/text_step_300 \
    --out $R/bench.json
note "all done"
