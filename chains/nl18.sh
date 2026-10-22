#!/bin/bash
# magic-nl18, one 80 GB card: the twins play, talk and compute. Starts like magic-nl16 (fitted start, a 300-step
# warm-up on all seven value games, which opened every one of them on never-seen values by step 400), then the full
# mix as in magic-nl15 (co-reading 0.5, dialogue 0.2, anchors 0.1, gates and alpha held), now with co_play: six-turn
# games in text where each twin holds a secret the other reads through the link (play.json, written on this pod by the
# original models, prep_talk.py --play). Half the episodes get 4 silent rounds (--pause 4 --pause_p 0.5: link-only
# ticks before replies and turns). Batches run as 4 groups of similar length (--micro 4). Then the evaluations, the
# held-out payload table with and without the silent rounds, and transcripts of whole play conversations.
# Usage: bash nl18.sh OUT_DIR CODE_TGZ REPLIES COREAD_JSON_GZ TALK_JSON
set -u
O=$1 TGZ=$2 REPLIES=$3 COREAD=$4 TALK=$5
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results && gunzip -c $COREAD > ../data/coread.json || exit 1
[ -s $O/replies.json ] || cp $REPLIES $O/replies.json
[ -s $O/talk.json ] || cp $TALK $O/talk.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
steps() { python -c "import os, re, sys; d = sys.argv[1]; print(','.join(os.path.join(d, f'step_{s}') for s in sorted(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r'step_(\d+)$', n)] if m)))" $1; }

cp $O/talk.json ../data/talk.json || exit 1
if [ ! -s $O/play.json ]; then
    note "play conversations: start"
    python prep_talk.py --play --turns 6 --n_train 1600 --n_test 300 --bs 64 --reply 128 --out $O/play.part > $O/prep_play.log 2>&1 || { note "prep_play failed"; exit 1; }
    mv $O/play.part $O/play.json; note "play conversations: $(tail -1 $O/prep_play.log)"
fi
cp $O/play.json ../data/play.json || exit 1
train() {
    python train.py --replies $O/replies.json --out $O/run --read_start 0.1 --content_init qkvo --gate_norm --hold 100000 --payload_w 3 \
        --warmup_steps 10 --relay_first 300 --relay_games relay_word,relay_number,relay_phrase,relay_digits,merge_sum,merge_fact,lookup \
        --steps 4000 --minutes 180 --bs 16 --coread 0.5 --anchor 0.1 --dialogue 0.2 --comp 0 --warm 0.3 --ramp 0.2 --send_cost 0.01 \
        --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 \
        --pause 4 --pause_p 0.5 --micro 4 --ckpt $1 > $O/train.log 2>&1
}
if [ ! -f $O/run.DONE ]; then
    if [ -d $O/run ] && ls $O/run | grep -q '^step_'; then note "resuming is not supported: delete $O/run to restart"; exit 1; fi
    train none
    rc=$?
    if [ $rc -ne 0 ] && grep -q OutOfMemoryError $O/train.log && ! grep -q '^{"step"' $O/train.log; then
        note "out of memory without gradient checkpointing: again with it"
        mv $O/train.log $O/train_oom.log; rm -rf $O/run
        train both || exit 1
    elif [ $rc -ne 0 ]; then
        exit 1
    fi
    touch $O/run.DONE; note "training done"
fi
ALL=$(steps $O/run)
LAST=${ALL##*,}
note "evaluating $LAST"
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; shift_name=$1; shift; "$@" > $R/$shift_name.log 2>&1 && touch $R/$shift_name.DONE; note "$shift_name: exit $?"; }
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 20 --coread
ev payload_pause4 python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final_pause4.json --n 20 --coread --pause 4
ev transcripts python transcripts.py --run $LAST --replies $O/replies.json --out $R/transcripts.txt --n 4 --device cuda --coread_data ../data/coread.json
ev evalfree python evalfree.py --run $LAST --replies $O/replies.json --out_dir $R/evalfree --n 20 --bs 8 --coread_data ../data/coread.json
ev evalfree_sum python evalfree.py --summarize --out_dir $R/evalfree --tag headline
cp ../results/evalfree_headline.json $R/ 2>/dev/null
for g in relay_number merge_sum lookup co_play; do
    ev readmap_$g python readmap.py --run $LAST --replies $O/replies.json --game $g --out $R/readmap_$g.json --n 8 --coread_data ../data/coread.json
done
ev bridge python analyze_bridge.py --run $LAST --replies $O/replies.json --out $R/bridge.json --n 16 --coread_data ../data/coread.json
note "all done"
