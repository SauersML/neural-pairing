#!/bin/bash
# nl40, one 48 or 80 GB card: the steered-reading game again, 300 steps from nl22c's step_157 on nl33's mix, with the
# targets written by the linked pair being trained (steer_read.py --writer linked: adapters on, link on) under the
# topic sentence (--teacher topic), so a target differs from what the model already writes only by the concept.
# Nothing is steered in 36% of the game's episodes, and the steered share of the batch is nl33's (--steer 0.5
# --steer_none 0.36: 0.5 x 0.64 = 0.4 x 0.8). During prep and training, moodscan.py reads nl33w's step_300 (the same
# 300 steps without the game) on 40 new conversations (seed 7272); nl40 is kept only if its link > cut count on those
# 40 is at least nl33w's, for both Quill and Ember. Gradient checkpointing below 79 GB. Then, on step 300: the
# steered-twin test (steering_scale.py) at nl33's five settings, each followed by the direct question (steer_ask.py),
# and moodscan.py on nl38's 40 conversations (seed 6161) and on the 40 new ones; four jobs at a time on 80 GB and two
# on 48 GB (a test that runs out of memory is rerun alone).
# Inputs: INPUTS_TAR (step_157), DATA_TGZ (data/{coread,play,talk,chat,chat2,pools}.json and replies.json), CK33W
# (nl33w's step_300).
# Usage: bash nl40.sh OUT_DIR CODE_TGZ INPUTS_TAR DATA_TGZ CK33W
set -u
O=$1 TGZ=$2 INPUTS=$3 DATA=$4 C33W=$5
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
unset NL_TINY NL_MAXNEW NL_AFFECT_N
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
J=/workspace/jobs/nl40
mkdir -p $J/w && tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J || { note "bad $INPUTS"; exit 1; }
tar --no-same-owner --exclude='._*' -xf $C33W -C $J/w || { note "bad $C33W"; exit 1; }
tar --no-same-owner --exclude='._*' -xzf $DATA -C . || { note "bad $DATA"; exit 1; }
[ -s data/coread.json ] && [ -s data/chat2.json ] && [ -s replies.json ] && [ -s $J/step_157/bridge.pt ] \
    && [ -s $J/w/step_300/bridge.pt ] || { note 'inputs missing'; exit 1; }
R=$O/results
mkdir -p $R
note 'mood scan of nl33w on the new conversations, beside prep and training'
(cd pairing && python moodscan.py --run $J/w/step_300 --names nl33w --out $R/mood_new --seed 7272 \
    > $R/mood_before.log 2>&1; echo "$(date '+%F %T') mood before exit $?" >> $O/chain.log) &
if [ ! -s $O/steer.pt ]; then
    note 'steered-reading data: directions and topic openers written by the linked pair'
    python pairing/steer_read.py --run $J/step_157 --out $O/steer.pt --teacher topic --writer linked --plain 600 --bs 64 \
        > $O/steer_prep.log 2>&1 || { note 'prep failed'; exit 1; }
    cp $O/steer.json $R/steer_data.json
fi
CK=none; [ "$MEM" -lt 79000 ] && CK=both
if [ ! -f $O/run.DONE ]; then
    note "300 steps from nl22c step_157 on the linked pair's own openers, checkpointing $CK"
    (cd pairing && python train.py --replies ../replies.json --out $O/run --init $J/step_157 --gate_norm --hold 100000 \
        --alpha_floor 0.1 --lr_alpha 1e-2 --payload_w 3 --warmup_steps 10 --relay_first 0 --steps 300 --bs 16 \
        --steer 0.5 --steer_data $O/steer.pt --steer_none 0.36 \
        --coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2 --chat2 0.25 \
        --comp 0 --warm 0 --ramp 0 --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 \
        --eval_every 300 --first_eval 300 --quick_n 4 --payload_eval 16 --save_every 300 \
        --pause 4 --pause_p 0.5 --micro 8 --ckpt $CK) > $O/train.log 2>&1 || { note 'training failed'; exit 1; }
    touch $O/run.DONE
fi
cp $O/run/log.json $R/log.json
LAST=$O/run/step_300
steer() {  # TAG LAYER BOOST: the steered-twin test, then the direct question on its pairs
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer $2 --boost $3 --run $LAST \
        --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
    [ -f $R/$1.DONE ] || return
    python pairing/steer_ask.py --run $LAST --scale $R/$1 --out $R/ask_$1.json > $R/ask_$1.log 2>&1
    note "ask $1 exit $?"
}
mood_final() {
    note 'mood final start'
    (cd pairing && python moodscan.py --run $LAST --names final --out $R/mood_dev --seed 6161 \
        && python moodscan.py --run $LAST --names final --out $R/mood_new --seed 7272) > $R/mood_final.log 2>&1
    note "mood final exit $?"
}
MAXJ=4; [ "$MEM" -lt 79000 ] && MAXJ=2
slot() { while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 10; done; }
slot; mood_final &
for a in l6:6:1 l15x4:15:4 l27:27:1 l15:15:1 l27x4:27:4; do
    IFS=: read -r t l b <<< "$a"
    slot; steer $t $l $b &
    sleep 20
done
wait
for a in l6:6:1 l15x4:15:4 l27:27:1 l15:15:1 l27x4:27:4; do
    IFS=: read -r t l b <<< "$a"
    [ -f $R/$t.DONE ] && continue
    grep -q "out of memory" $R/$t.log || continue
    mv $R/$t $R/$t.oom && mv $R/$t.log $R/$t.oom.log
    steer $t $l $b
done
note 'all done'
