#!/bin/bash
# nl22, one 80 GB card: fine-tunes nl20's step_1320 for MINUTES on nl20's mix plus plain chats (--chat 0.2;
# data/chat.json from prep_talk.py --chat: each turn is written by the original model with the partner's private city
# also in its prompt, which the trained prompt lacks). Waits up to an hour for /workspace/jobs/nl22/{step_1320,
# replies.json, data/} and the READY file written after them. Then the payload table on the last step and, when
# evalfree.py has co_chat, free chats with the crossed control before (step_1320) and after.
# Usage: bash nl22.sh OUT_DIR CODE_TGZ MINUTES
set -u
O=$1 TGZ=$2 MINUTES=$3
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
J=/workspace/jobs/nl22
note "waiting for $J/READY"
for _ in $(seq 360); do [ -f $J/READY ] && break; sleep 10; done
[ -f $J/READY ] || { note "no inputs after an hour"; exit 1; }
cp -f $J/data/coread.json $J/data/play.json $J/data/talk.json $J/data/pools.json $J/data/chat.json ../data/ || exit 1
cp $J/replies.json $O/replies.json
INIT=$J/step_1320
note "inputs: $(python -c "import json; d = json.load(open('../data/chat.json')); print({s: len(v) for s, v in d.items()})")"
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; n=$1; shift; "$@" > $R/$n.log 2>&1 && touch $R/$n.DONE; note "$n: exit $?"; }

MIX="--coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2"
[ -s $O/mix_flags.txt ] && MIX=$(cat $O/mix_flags.txt)
if [ ! -f $O/run.DONE ]; then
    rm -rf $O/run
    note "fine-tune: $MIX, $MINUTES minutes"
    python train.py --replies $O/replies.json --out $O/run --init $INIT --gate_norm --hold 100000 --payload_w 3 \
        --warmup_steps 10 --relay_first 0 --steps 4000 --minutes $MINUTES --bs 16 $MIX --comp 0 --warm 0 --ramp 0 \
        --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 --first_eval 20 --quick_n 4 \
        --payload_eval 16 --save_every 100 --pause 4 --pause_p 0.5 --micro 4 --ckpt none > $O/train.log 2>&1 \
        || { note "fine-tune failed"; exit 1; }
    touch $O/run.DONE; note "fine-tune done"
fi
LAST=$(python -c "import os, re, sys; d = sys.argv[1]; print(os.path.join(d, 'step_%d' % max(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r'step_(\d+)$', n)] if m)))" $O/run)
if [ -s $O/code_eval.tgz ]; then
    tar --no-same-owner -xzf $O/code_eval.tgz -C .. && note "evaluating with code_eval.tgz"
fi
note "evaluating $LAST and $INIT"
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 40 --coread
if grep -q co_chat evalfree.py; then
    for t in $LAST:after $INIT:before; do
        ev chat_${t##*:} python evalfree.py --run ${t%:*} --replies $O/replies.json --coread_data ../data/coread.json --games co_chat \
            --crossed --n 40 --bs 4 --out_dir $R/chat_${t##*:}
    done
fi
[ -s $O/extra_evals.sh ] && { note "extra evaluations: start"; LAST=$LAST INIT=$INIT R=$R O=$O bash $O/extra_evals.sh > $R/extra_evals.log 2>&1; note "extra evaluations: exit $?"; }
note "all done"
