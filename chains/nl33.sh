#!/bin/bash
# nl33 (ARM steer) and nl33w (ARM write), one 48 or 80 GB card: 300 steps from nl22c's step_157 on nl24's mix, with the
# write sizes trained and floored at 0.1 (--alpha_floor 0.1 --lr_alpha 1e-2; the gates stay fixed):
#   steer: 40% of the episodes are the steered-reading game (steer_read.py: a training concept steered into one model
#          after a random exchange block; the partner's target is the original model's opener when told the concept, or
#          the concept itself after a direct question). The test concepts and their near-synonyms are never trained.
#   write: no steered game.
# Gradient checkpointing below 79 GB. Then, on step 300, the steered-twin test (steering_scale.py) after blocks 6, 15
# and 27 and at four times the strength after blocks 15 and 27, three at a time on 80 GB and two on 48 GB (one that runs
# out of memory is rerun alone), and the direct question (steer_ask.py) at each setting, also on step_157 when its
# directions are in /workspace/jobs/nl33/before.
# Inputs: INPUTS_TAR (step_157), DATA_TGZ (data/{coread,play,talk,chat,chat2,pools}.json and replies.json).
# Usage: bash nl33.sh OUT_DIR CODE_TGZ INPUTS_TAR DATA_TGZ ARM
set -u
O=$1 TGZ=$2 INPUTS=$3 DATA=$4 ARM=$5
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8         # torch's default thread count stalls on many-core hosts
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
J=/workspace/jobs/nl33
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $INPUTS -C $J || { note "bad $INPUTS"; exit 1; }
tar --no-same-owner --exclude='._*' -xzf $DATA -C . || { note "bad $DATA"; exit 1; }
[ -s data/coread.json ] && [ -s data/chat2.json ] && [ -s replies.json ] || { note 'data missing'; exit 1; }
R=$O/results
mkdir -p $R
case $ARM in
    steer) MIX="--steer 0.4 --steer_data $O/steer.pt" ;;
    write) MIX= ;;
    *) note "unknown arm $ARM"; exit 1 ;;
esac
if [ $ARM = steer ] && [ ! -s $O/steer.pt ]; then
    note 'steered-reading data: directions and teacher openers'
    python pairing/steer_read.py --run $J/step_157 --out $O/steer.pt > $O/steer_prep.log 2>&1 || { note 'prep failed'; exit 1; }
    cp $O/steer.json $R/steer_data.json
fi
CK=none; [ "$MEM" -lt 79000 ] && CK=both
if [ ! -f $O/run.DONE ]; then
    note "300 steps from nl22c step_157, arm $ARM, checkpointing $CK"
    (cd pairing && python train.py --replies ../replies.json --out $O/run --init $J/step_157 --gate_norm --hold 100000 \
        --alpha_floor 0.1 --lr_alpha 1e-2 --payload_w 3 --warmup_steps 10 --relay_first 0 --steps 300 --bs 16 $MIX \
        --coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2 --chat2 0.25 \
        --comp 0 --warm 0 --ramp 0 --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 \
        --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 \
        --pause 4 --pause_p 0.5 --micro 8 --ckpt $CK) > $O/train.log 2>&1 || { note 'training failed'; exit 1; }
    touch $O/run.DONE
fi
cp $O/run/log.json $R/log.json
LAST=$O/run/step_300
steer() {  # TAG LAYER BOOST
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer $2 --boost $3 --run $LAST \
        --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
}
ARMS="l6:6:1 l15x4:15:4 l27:27:1 l15:15:1 l27x4:27:4"
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
for t in l6 l15 l27 l15x4 l27x4; do
    [ -f $R/$t.DONE ] || continue
    note "ask $t start"
    python pairing/steer_ask.py --run $LAST --scale $R/$t --out $R/ask_$t.json > $R/ask_$t.log 2>&1
    note "ask $t exit $?"
    B=$J/before/$t
    [ -d $B ] || continue
    python pairing/steer_ask.py --run $J/step_157 --scale $B --out $R/ask_before_$t.json > $R/ask_before_$t.log 2>&1
    note "ask before $t exit $?"
done
note 'all done'
