#!/bin/bash
# magic-nl34e: does reading pushed concepts carry over to what a twin holds in its own context? nl33's steered-reading
# game taught the partner to name concepts added to its twin's hidden states, concepts never trained included. Here the
# fixed free-chat tests where the private content is in the twin's prompt (never pushed): co_chat2 (twelve kinds of
# private item, four never trained; nl24/nl26's 40 plans, seed 4242) and co_chat (private cities; nl22's 40 plans),
# greedy, with the link cut and linked to a stranger as controls, on nl33's step 300 (steered game) and nl33w's step
# 300 (the same training without it). Shards run three at a time on an 80 GB card, two on 48 GB.
# Inputs: CKPT_STEER and CKPT_WRITE (tars of each run's step_300, published), DATA_TGZ (data33.tgz: data/ and
# replies.json, published).
# Usage: bash nl34e.sh OUT_DIR CODE_TGZ DATA_TGZ CKPT_STEER CKPT_WRITE
set -u
O=$1 TGZ=$2 DATA=$3 CKS=$4 CKW=$5
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset NL_TINY NL_MAXNEW
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
J=/workspace/jobs/nl34e
tar --no-same-owner --exclude='._*' -xzf $DATA -C . || { note "bad $DATA"; exit 1; }
[ -s data/coread.json ] && [ -s data/chat.json ] && [ -s data/chat2.json ] && [ -s replies.json ] || { note 'data missing'; exit 1; }
for a in steer:$CKS write:$CKW; do
    mkdir -p $J/${a%%:*} && tar --no-same-owner --exclude='._*' -xf ${a#*:} -C $J/${a%%:*} || { note "bad ${a#*:}"; exit 1; }
    [ -s $J/${a%%:*}/step_300/bridge.pt ] || { note "no step_300 in ${a#*:}"; exit 1; }
done
R=$O/results
mkdir -p $R
ev() {  # NAME ARGS...: one evalfree shard, its log beside it
    local name=$1; shift
    [ -f $R/$name.DONE ] && return 0
    note "$name start"
    python pairing/evalfree.py "$@" --out_dir $R/$name > $R/$name.log 2>&1 && touch $R/$name.DONE
    note "$name exit $?"
}
K=2; [ "$MEM" -ge 79000 ] && K=3
JOBS=
for g in chat2 chat; do
    for a in steer write; do
        for k in 0 1 2; do JOBS="$JOBS $g:$a:$k"; done
    done
done
set -- $JOBS
while [ $# -gt 0 ]; do
    for _ in $(seq $K); do
        [ $# -gt 0 ] || break
        IFS=: read -r g a k <<< "$1"
        ev ${g}_${a}_$k --run $J/$a/step_300 --replies replies.json --coread_data data/coread.json --games co_$g \
            --crossed --n 40 --seed 4242 --temperature 0 --shard $k/3 --bs 2 &
        shift
    done
    wait
done
python pairing/chat2_merge.py --after $R/chat2_steer_0 $R/chat2_steer_1 $R/chat2_steer_2 \
    --before $R/chat2_write_0 $R/chat2_write_1 $R/chat2_write_2 --out $R/chat2_summary.json > $R/chat2_summary.log 2>&1
note "chat2 summary exit $?"
python pairing/chat_merge.py --after $R/chat_steer_0 $R/chat_steer_1 $R/chat_steer_2 \
    --before $R/chat_write_0 $R/chat_write_1 $R/chat_write_2 --out $R/chat_summary.json > $R/chat_summary.log 2>&1
note "chat summary exit $?"
note 'all done'
