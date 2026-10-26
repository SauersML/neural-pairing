#!/bin/bash
# nl18e, one 48 GB card: nl18's evaluations on its step_1800 (CKPT_TAR), plus extra_evals.sh (EXTRA_EVALS).
# Usage: bash nl18e.sh OUT_DIR CODE_TGZ CKPT_TAR REPLIES COREAD_JSON_GZ TALK_JSON PLAY_JSON EXTRA_EVALS
set -u
O=$1 TGZ=$2 CKPT=$3 REPLIES=$4 COREAD=$5 TALK=$6 PLAY=$7 EXTRA=$8
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results ../ckpt && gunzip -c $COREAD > ../data/coread.json || exit 1
cp $TALK ../data/talk.json && cp $PLAY ../data/play.json && cp $REPLIES $O/replies.json || exit 1
tar --no-same-owner -xf $CKPT -C ../ckpt || exit 1
LAST=$(cd ../ckpt/step_1800 && pwd)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; shift_name=$1; shift; "$@" > $R/$shift_name.log 2>&1 && touch $R/$shift_name.DONE; note "$shift_name: exit $?"; }
note "evaluating magic-nl18 $LAST"
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 20 --coread
ev payload_pause4 python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final_pause4.json --n 20 --coread --pause 4
ev transcripts python transcripts.py --run $LAST --replies $O/replies.json --out $R/transcripts.txt --n 4 --device cuda --coread_data ../data/coread.json
note "extra evaluations: start"; LAST=$LAST R=$R O=$O bash $EXTRA > $R/extra_evals.log 2>&1; note "extra evaluations: exit $?"
ev evalfree python evalfree.py --run $LAST --replies $O/replies.json --out_dir $R/evalfree --n 20 --bs 8 --coread_data ../data/coread.json
ev evalfree_sum python evalfree.py --summarize --out_dir $R/evalfree --tag headline
cp ../results/evalfree_headline.json $R/ 2>/dev/null
for g in co_qa co_play; do
    ev readmap_$g python readmap.py --run $LAST --replies $O/replies.json --game $g --out $R/readmap_$g.json --n 8 --coread_data ../data/coread.json
done
ev bridge python analyze_bridge.py --run $LAST --replies $O/replies.json --out $R/bridge.json --n 16 --coread_data ../data/coread.json
note "all done"
