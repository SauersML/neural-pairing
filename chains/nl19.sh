#!/bin/bash
# magic-nl19, one 80 GB card: magic-nl18's recipe (all-games warm-up, full mix with six-turn co_play games, silent
# rounds, micro-batches) plus two additions. (1) co_distill: for co-reading items, the loss on the reader's reply is
# the KL divergence from the original model that read the text itself (prep_coread.py writes those replies on this
# pod), a dense target on every reply token where the co-reading games have only the short answer. (2) shared pen
# (--pen 0.1): one episode in ten the twins co-write one text, alternating words (pen_secret, pen_free), after a
# two-step check on the real models (dropped if it fails). Play conversations: magic-nl18's. Before the evaluations,
# a code_eval.tgz and extra_evals.sh placed in OUT_DIR (later code, e.g. lens.py and turnbench.py) are used if present.
# Usage: bash nl19.sh OUT_DIR CODE_TGZ REPLIES COREAD_JSON_GZ TALK_JSON PLAY_JSON
set -u
O=$1 TGZ=$2 REPLIES=$3 COREAD=$4 TALK=$5 PLAY=$6
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results && gunzip -c $COREAD > ../data/coread.json || exit 1
[ -s $O/replies.json ] || cp $REPLIES $O/replies.json
cp $TALK ../data/talk.json && cp $PLAY ../data/play.json || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
steps() { python -c "import os, re, sys; d = sys.argv[1]; print(','.join(os.path.join(d, f'step_{s}') for s in sorted(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r'step_(\d+)$', n)] if m)))" $1; }

if ! grep -q '"coread"' $O/replies.json; then
    note "distillation replies: start"
    python prep_coread.py --replies $O/replies.json --n_train 6000 --n_test 600 --reply 128 --bs 64 > $O/prep_coread.log 2>&1 || { note "prep_coread failed"; exit 1; }
    note "distillation replies: $(tail -1 $O/prep_coread.log | cut -c1-200)"
fi
PEN="--pen 0.1"
if [ ! -f $O/pen_smoke.DONE ]; then
    if python train.py --replies $O/replies.json --out $O/pen_smoke --pen 1 --anchor 0 --comp 0 --relay_first 0 --coread 0 --dialogue 0 \
        --steps 2 --bs 8 --payload_every 1 --first_eval 1000 --payload_eval 0 --save_every 1000 --ckpt none > $O/pen_smoke.log 2>&1; then
        touch $O/pen_smoke.DONE; note "shared-pen check passed"
    else
        PEN=""; note "shared-pen check failed: training without the pen"
    fi
    rm -rf $O/pen_smoke
fi
train() {
    python train.py --replies $O/replies.json --out $O/run --read_start 0.1 --content_init qkvo --gate_norm --hold 100000 --payload_w 3 \
        --warmup_steps 10 --relay_first 300 --relay_games relay_word,relay_number,relay_phrase,relay_digits,merge_sum,merge_fact,lookup \
        --steps 4000 --minutes 150 --bs 16 --coread 0.5 --anchor 0.1 --dialogue 0.2 --comp 0 --warm 0.3 --ramp 0.2 --send_cost 0.01 \
        --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 \
        --pause 4 --pause_p 0.5 --micro 4 $PEN --ckpt $1 > $O/train.log 2>&1
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
if [ -s $O/code_eval.tgz ]; then
    tar --no-same-owner -xzf $O/code_eval.tgz -C .. && note "evaluating with code_eval.tgz"
fi
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; shift_name=$1; shift; "$@" > $R/$shift_name.log 2>&1 && touch $R/$shift_name.DONE; note "$shift_name: exit $?"; }
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 20 --coread
ev payload_pause4 python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final_pause4.json --n 20 --coread --pause 4
ev transcripts python transcripts.py --run $LAST --replies $O/replies.json --out $R/transcripts.txt --n 4 --device cuda --coread_data ../data/coread.json
[ -s $O/extra_evals.sh ] && { note "extra evaluations: start"; LAST=$LAST R=$R O=$O bash $O/extra_evals.sh > $R/extra_evals.log 2>&1; note "extra evaluations: exit $?"; }
ev evalfree python evalfree.py --run $LAST --replies $O/replies.json --out_dir $R/evalfree --n 20 --bs 8 --coread_data ../data/coread.json
ev evalfree_sum python evalfree.py --summarize --out_dir $R/evalfree --tag headline
cp ../results/evalfree_headline.json $R/ 2>/dev/null
for g in co_qa co_play; do
    ev readmap_$g python readmap.py --run $LAST --replies $O/replies.json --game $g --out $R/readmap_$g.json --n 8 --coread_data ../data/coread.json
done
ev bridge python analyze_bridge.py --run $LAST --replies $O/replies.json --out $R/bridge.json --n 16 --coread_data ../data/coread.json
note "all done"
