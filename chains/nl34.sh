#!/bin/bash
# nl34, one 48 or 80 GB card: 400 more steps from nl33's step_300 on the same mix, with the steered-reading game's
# steering coefficients log-uniform in [0.1, 6] times the concept direction (--steer_strength 0.1 6) and the directions
# refit on that checkpoint; write sizes trained above 0.1. Then, on step 400: the steered-twin test at nl33's five
# settings, the direct question, and the fixed free-chat tests with private items and cities in the model's prompt
# (nl34e.sh's), merged by chat2_merge.py and chat_merge.py.
# Inputs: CKPT_TAR (nl33's step_300), DATA_TGZ (data33.tgz).
# Usage: bash nl34.sh OUT_DIR CODE_TGZ CKPT_TAR DATA_TGZ
set -u
O=$1 TGZ=$2 CK=$3 DATA=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8         # torch's default thread count stalls on many-core hosts
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
unset NL_TINY NL_MAXNEW
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
J=/workspace/jobs/nl34
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $CK -C $J || { note "bad $CK"; exit 1; }
INIT=$J/step_300
[ -s $INIT/bridge.pt ] || { note 'no step_300'; exit 1; }
tar --no-same-owner --exclude='._*' -xzf $DATA -C . || { note "bad $DATA"; exit 1; }
[ -s data/coread.json ] && [ -s data/chat.json ] && [ -s data/chat2.json ] && [ -s replies.json ] || { note 'data missing'; exit 1; }
R=$O/results
mkdir -p $R
if [ ! -s $O/steer.pt ]; then
    note 'steered-reading data: directions fit on step_300, teacher openers'
    python pairing/steer_read.py --run $INIT --out $O/steer.pt > $O/steer_prep.log 2>&1 || { note 'prep failed'; exit 1; }
    cp $O/steer.json $R/steer_data.json
fi
CKP=none; [ "$MEM" -lt 79000 ] && CKP=both
if [ ! -f $O/run.DONE ]; then
    note "400 steps from nl33 step_300, steering coefficients 0.1 to 6, checkpointing $CKP"
    (cd pairing && python train.py --replies ../replies.json --out $O/run --init $INIT --gate_norm --hold 100000 \
        --alpha_floor 0.1 --lr_alpha 1e-2 --payload_w 3 --warmup_steps 10 --relay_first 0 --steps 400 --bs 16 \
        --steer 0.4 --steer_data $O/steer.pt --steer_strength 0.1 6 \
        --coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2 --chat2 0.25 \
        --comp 0 --warm 0 --ramp 0 --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 \
        --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 \
        --pause 4 --pause_p 0.5 --micro 8 --ckpt $CKP) > $O/train.log 2>&1 || { note 'training failed'; exit 1; }
    touch $O/run.DONE
fi
LAST=$O/run/step_400
K=2; [ "$MEM" -ge 79000 ] && K=3
run_k() {  # run "JOB" words K at a time with FUNCTION
    local f=$1; shift
    while [ $# -gt 0 ]; do
        for _ in $(seq $K); do
            [ $# -gt 0 ] || break
            $f $1 &
            shift
        done
        wait
    done
}
steer() {  # TAG:LAYER:BOOST
    IFS=: read -r t l b <<< "$1"
    [ -f $R/$t.DONE ] && return 0
    note "$t start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer $l --boost $b --run $LAST \
        --out $R/$t > $R/$t.log 2>&1 && touch $R/$t.DONE
    note "$t exit $?"
}
ARMS="l6:6:1 l15x4:15:4 l27:27:1 l15:15:1 l27x4:27:4"
run_k steer $ARMS
for a in $ARMS; do
    IFS=: read -r t l b <<< "$a"
    [ -f $R/$t.DONE ] && continue
    grep -q "out of memory" $R/$t.log || continue
    mv $R/$t $R/$t.oom && mv $R/$t.log $R/$t.oom.log
    steer $a
done
for t in l6 l15 l27 l15x4 l27x4; do
    [ -f $R/$t.DONE ] || continue
    note "ask $t start"
    python pairing/steer_ask.py --run $LAST --scale $R/$t --out $R/ask_$t.json > $R/ask_$t.log 2>&1
    note "ask $t exit $?"
done
chat() {  # GAME:SHARD
    IFS=: read -r g k <<< "$1"
    [ -f $R/${g}_$k.DONE ] && return 0
    note "${g}_$k start"
    python pairing/evalfree.py --run $LAST --replies replies.json --coread_data data/coread.json --games co_$g \
        --crossed --n 40 --seed 4242 --temperature 0 --shard $k/3 --bs 2 --out_dir $R/${g}_$k > $R/${g}_$k.log 2>&1 \
        && touch $R/${g}_$k.DONE
    note "${g}_$k exit $?"
}
run_k chat chat2:0 chat2:1 chat2:2 chat:0 chat:1 chat:2
python pairing/chat2_merge.py --after $R/chat2_0 $R/chat2_1 $R/chat2_2 --out $R/chat2_summary.json > $R/chat2_summary.log 2>&1
note "chat2 summary exit $?"
python pairing/chat_merge.py --after $R/chat_0 $R/chat_1 $R/chat_2 --out $R/chat_summary.json > $R/chat_summary.log 2>&1
note "chat summary exit $?"
note 'all done'
