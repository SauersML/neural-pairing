#!/bin/bash
# nl21, one 48 GB card: nl20 with held-out games. Training never draws lookup, relay_digits, co_mcq, co_code_out,
# co_translate or the rhyme play game (--hold_out: a drawn episode of one is drawn again); the final payload table (n =
# 40 per game) tests them beside the trained games. Stage 1 warms up on the five other value games for 600 steps; stage
# 2 runs 130 minutes. Both stages use gradient checkpointing to fit in 48 GB.
# Usage: bash nl21.sh OUT_DIR CODE_TGZ REPLIES COREAD_JSON_GZ TALK_JSON PLAY_JSON
set -u
O=$1 TGZ=$2 REPLIES=$3 COREAD=$4 TALK=$5 PLAY=$6
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results && gunzip -c $COREAD > ../data/coread.json || exit 1
[ -s $O/replies.json ] || { case $REPLIES in *.gz) gunzip -c $REPLIES > $O/replies.json ;; *) cp $REPLIES $O/replies.json ;; esac; }
cp $TALK ../data/talk.json && cp $PLAY ../data/play.json || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
steps() { python -c "import os, re, sys; d = sys.argv[1]; print(','.join(os.path.join(d, f'step_{s}') for s in sorted(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r'step_(\d+)$', n)] if m)))" $1; }
VALUES=relay_word,relay_number,relay_phrase,merge_sum,merge_fact
HOLD="--hold_out lookup,relay_digits,co_mcq,co_code_out,co_translate,co_play:rhyme"
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; shift_name=$1; shift; "$@" > $R/$shift_name.log 2>&1 && touch $R/$shift_name.DONE; note "$shift_name: exit $?"; }

if [ ! -f $O/warm.DONE ]; then
    rm -rf $O/warm
    note "warm-up: start"
    python train.py --replies $O/replies.json --out $O/warm --read_start 0.1 --content_init qkvo --gate_norm --hold 100000 --payload_w 3 \
        --warmup_steps 10 --relay_first 100000 --relay_games $VALUES $HOLD --steps 600 --bs 16 --coread 0 --anchor 0 --dialogue 0 --comp 0 \
        --send_cost 0 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 \
        --save_every 600 --ckpt both > $O/warm.log 2>&1 || { note "warm-up failed"; exit 1; }
    touch $O/warm.DONE; note "warm-up: done"
fi
WARM=$O/warm/step_600
ev warm_payload python payload_ctl.py --run $WARM --replies $O/replies.json --out $R/payload_warm.json --n 20

PEN="--pen 0.05"
if [ ! -f $O/pen_smoke.DONE ]; then
    if python train.py --replies $O/replies.json --out $O/pen_smoke --pen 1 --anchor 0 --comp 0 --relay_first 0 --coread 0 --dialogue 0 \
        --steps 2 --bs 8 --payload_every 1 --first_eval 1000 --payload_eval 0 --save_every 1000 --ckpt both > $O/pen_smoke.log 2>&1; then
        touch $O/pen_smoke.DONE; note "shared-pen check passed"
    else
        PEN=""; note "shared-pen check failed: training without the pen"
    fi
    rm -rf $O/pen_smoke
fi
MIX="--coread 0.35 --dialogue 0.15 --anchor 0.1 $PEN"
[ -s $O/mix_flags.txt ] && MIX=$(cat $O/mix_flags.txt)
TRAIN_REPLIES=$O/replies.json
if [ -f $O/no_distill ]; then
    python -c "import json, sys; r = json.load(open(sys.argv[1])); [r[x].pop('coread', None) for x in r]; json.dump(r, open(sys.argv[2], 'w'))" \
        $O/replies.json $O/replies_nodistill.json && TRAIN_REPLIES=$O/replies_nodistill.json
fi
note "full mix: $MIX, replies $TRAIN_REPLIES"
train() {
    python train.py --replies $TRAIN_REPLIES --out $O/run --init $WARM --gate_norm --hold 100000 --payload_w 3 \
        --warmup_steps 10 --relay_first 0 --steps 4000 --minutes 130 --bs 16 $MIX $HOLD --comp 0 --warm 0.3 --ramp 0.2 --send_cost 0.01 \
        --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 \
        --pause 4 --pause_p 0.5 --micro 4 --ckpt $1 > $O/train.log 2>&1
}
if [ ! -f $O/run.DONE ]; then
    if [ -d $O/run ] && ls $O/run | grep -q '^step_'; then note "resuming is not supported: delete $O/run to restart"; exit 1; fi
    train both
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
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 40 --coread
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
