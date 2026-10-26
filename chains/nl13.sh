#!/bin/bash
# nl13, one 48 GB RTX A6000 or A40. Trains from the original models on relay_word only (the sender is told a secret
# word, the receiver writes "The word is X."), 1,500 steps or 80 minutes, batch 16: --read_start 0.1, gates on unit-RMS
# states, no send cost, the held-out payload table (link, cut, shuffled partner) every 100 steps. Then the held-out
# payload table and reading maps on the last checkpoint.
# Usage: bash nl13.sh OUT_DIR CODE_TGZ REPLIES COREAD_JSON_GZ
set -u
O=$1 TGZ=$2 REPLIES=$3 COREAD=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda
export NL_QUICK=relay_word,relay_number,relay_phrase
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results && gunzip -c $COREAD > ../data/coread.json || exit 1
[ -s $O/replies.json ] || cp $REPLIES $O/replies.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
if [ ! -f $O/run.DONE ]; then
    python train.py --replies $O/replies.json --out $O/run --read_start 0.1 --gate_norm --warmup_steps 10 \
        --relay_first 100000 --steps 1500 --minutes 80 --bs 16 --coread 0 --anchor 0 --dialogue 0 --comp 0 \
        --send_cost 0 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 --eval_every 100 \
        --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 500 > $O/train.log 2>&1 || exit 1
    touch $O/run.DONE; note "training done"
fi
LAST=$(python -c "import os, re, sys; d = sys.argv[1]; print(os.path.join(d, 'step_%d' % max(int(m.group(1)) for n in os.listdir(d) for m in [re.match(r'step_(\d+)$', n)] if m)))" $O/run)
note "evaluating $LAST"
R=$O/results && mkdir -p $R
ev() { [ -f $R/$1.DONE ] && return; shift_name=$1; shift; "$@" > $R/$shift_name.log 2>&1 && touch $R/$shift_name.DONE; note "$shift_name: exit $?"; }
ev payload python payload_ctl.py --run $LAST --replies $O/replies.json --out $R/payload_final.json --n 20
for g in relay_word relay_number; do
    ev readmap_$g python readmap.py --run $LAST --replies $O/replies.json --game $g --out $R/readmap_$g.json --n 8 --coread_data ../data/coread.json
done
note "all done"
