#!/bin/bash
# nl11, one 80 GB A100 or H100. Writes co_talk conversations (prep_talk.py), then trains from the original models for
# 3,000 steps or 180 minutes, batch 16: every reader's output projection zeroed and its write size tanh(alpha) started
# at 0.1 (--read_start), queries and keys set to attend to content-matched partner positions (--content_init qk,
# align.py), gates on unit-RMS states (--gate_norm), 200 relay_word steps first, payload tokens weighted 3x, then the
# full mix (co-reading 0.6, dialogue 0.2, anchors 0.1). Then, on the last checkpoint: free generation (evalfree.py),
# transcripts, the held-out payload table, reading maps, co-evolution across checkpoints and the bridge analysis. Each
# stage leaves a DONE file.
# Usage: bash nl11.sh OUT_DIR CODE_TGZ REPLIES COREAD_JSON_GZ
set -u
O=$1 TGZ=$2 REPLIES=$3 COREAD=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda
export NL_QUICK=relay_word,relay_number,lookup,merge_fact,identity,co_qa,co_nli,co_mcq,co_translate,co_transfer
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results && gunzip -c $COREAD > ../data/coread.json || exit 1
[ -s $O/replies.json ] || cp $REPLIES $O/replies.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
steps() { python -c "import os, re, sys; d = sys.argv[1]; print(','.join(os.path.join(d, f'step_{s}') for s in sorted(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r'step_(\d+)$', n)] if m)))" $1; }

if [ ! -s $O/talk.json ]; then
    python prep_talk.py --n_train 500 --n_test 100 --bs 64 --out $O/talk.part > $O/prep_talk.log 2>&1 || exit 1
    mv $O/talk.part $O/talk.json; note "co_talk conversations: $(tail -1 $O/prep_talk.log)"
fi
cp $O/talk.json ../data/talk.json || exit 1
if [ ! -f $O/run.DONE ]; then
    python train.py --replies $O/replies.json --out $O/run --read_start 0.1 --gate_norm --warmup_steps 10 \
        --relay_first 200 --content_init qk --payload_w 3 --steps 3000 --minutes 180 --bs 16 --coread 0.6 --anchor 0.1 --dialogue 0.2 --comp 0 \
        --warm 0.3 --ramp 0.2 --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 \
        --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 > $O/train.log 2>&1 || exit 1
    touch $O/run.DONE; note "training done"
fi
ALL=$(steps $O/run)
LAST=${ALL##*,}
note "evaluating $LAST"
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; shift_name=$1; shift; "$@" > $R/$shift_name.log 2>&1 && touch $R/$shift_name.DONE; note "$shift_name: exit $?"; }
ev evalfree python evalfree.py --run $LAST --replies $O/replies.json --out_dir $R/evalfree --n 20 --bs 8 --coread_data ../data/coread.json
ev evalfree_sum python evalfree.py --summarize --out_dir $R/evalfree --tag headline
cp ../results/evalfree_headline.json $R/ 2>/dev/null
ev transcripts python transcripts.py --run $LAST --replies $O/replies.json --out $R/transcripts.txt --n 3 --device cuda --coread_data ../data/coread.json
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 20 --coread
for g in relay_number relay_word relay_phrase co_qa; do
    ev readmap_$g python readmap.py --run $LAST --replies $O/replies.json --game $g --out $R/readmap_$g.json --n 8 --coread_data ../data/coread.json
done
PICK=$(python -c "import sys; s = sys.argv[1].split(','); k = max(1, len(s) // 5); print(','.join(s[::-1][::k][:5][::-1]))" $ALL)
ev coevolve python coevolve.py --runs $PICK --replies $O/replies.json --out $R/coevolve.json --n 6 --coread_data ../data/coread.json
ev bridge python analyze_bridge.py --run $LAST --replies $O/replies.json --out $R/bridge.json --n 16 --coread_data ../data/coread.json
note "all done"
