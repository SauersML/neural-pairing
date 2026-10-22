#!/bin/bash
# nl24: original teachers -> diverse private situations -> 300 steps -> fixed free-chat controls.
# One 48 GB card. Inputs pod to pod, READY last: /workspace/jobs/nl24/{step_157,replies.json,data/}.
# data/ includes nl22's clean chat.json and coread/play/talk/pools.json. No checkpoint selection on test scores.
# Usage: bash nl24.sh OUT_DIR CODE_TGZ
# Mac staging and exact rp-run command: chains/launch_nl24.sh (lead executes; this task never rents GPUs).
set -euo pipefail
O=$1 TGZ=$2
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NL_TINY NL_MAXNEW
export NL_QUICK=relay_word,relay_number,merge_sum,lookup,merge_fact,co_play,co_qa,co_mcq
mkdir -p "$O" "$O/source" "$O/data" "$O/results"
tar --no-same-owner -xzf "$TGZ" -C "$O/source"
cd "$O/source/pairing"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$O/gpus.txt"
note() { echo "$(date '+%F %T') $*" >> "$O/chain.log"; }
J=/workspace/jobs/nl24
note "waiting for $J/READY"
for _ in $(seq 360); do [ -f "$J/READY" ] && break; sleep 10; done
[ -f "$J/READY" ] || { note 'no inputs after an hour'; exit 1; }
for name in coread play talk pools chat; do cp "$J/data/$name.json" ../data/; done
cp "$J/replies.json" "$O/replies.json"
INIT=$J/step_157
[ -s "$INIT/bridge.pt" ] && [ -d "$INIT/quill" ] && [ -d "$INIT/ember" ]
if [ ! -f "$O/chat2.DONE" ]; then
    note 'original-model teachers: 1200 train, 160 test attempted chats; six turns, 96 token cap'
    start=$(date +%s)
    python prep_talk.py --chat2 --n_train 1200 --n_test 160 --turns 6 --reply 96 --bs 8 \
        --out "$O/data/chat2.json" --timing "$O/chatgen_timing.json" > "$O/chatgen.log" 2>&1
    echo "$(( $(date +%s) - start ))" > "$O/chatgen_seconds.txt"
    touch "$O/chat2.DONE"
fi
cp "$O/data/chat2.json" ../data/chat2.json
# Fail before training if pruning left an empty split or no valid same-kind donors for the fixed evaluation.
python - <<'PY'
import json
import chateval
from collections import Counter
with open('../data/chat2.json') as f:
    d = json.load(f)
for split in ('train', 'test'):
    assert d[split], 'empty chat2 ' + split
    print(split, len(d[split]), dict(Counter(p['kind'] for c in d[split] for p in c['private'].values() if p)))
chateval.make_plans('../data/coread.json', 40, 4242, game='co_chat2')
PY
R=$O/results
# Preserve interrupted artifacts instead of silently reusing partial evaluations or optimizer state.
ev() {
    local name=$1; shift
    [ -f "$R/$name.DONE" ] && return 0
    note "$name start"
    if "$@" > "$R/$name.log" 2>&1; then touch "$R/$name.DONE"; else note "$name failed"; return 1; fi
}
if [ ! -f "$O/run.DONE" ]; then
    if [ -d "$O/run" ]; then mv "$O/run" "$O/run.interrupted.$(date +%s)"; fi
    note '300 fixed steps from nl22c step_157; gradient checkpointing and eight microbatches for 48 GB'
    python train.py --replies "$O/replies.json" --out "$O/run" --init "$INIT" --gate_norm --hold 100000 --payload_w 3 \
        --warmup_steps 10 --relay_first 0 --steps 300 --bs 16 \
        --coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2 --chat2 0.25 \
        --comp 0 --warm 0 --ramp 0 --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 \
        --eval_every 100 --first_eval 20 --quick_n 4 --payload_eval 16 --save_every 100 \
        --pause 4 --pause_p 0.5 --micro 8 --ckpt both > "$O/train.log" 2>&1
    touch "$O/run.DONE"
fi
LAST=$O/run/step_300
[ -s "$LAST/bridge.pt" ]
ev payload python payload_ctl.py --run "$LAST" --replies "$O/replies.json" --out "$R/payload_final.json" --n 40 --bs 4 --coread
# Sequential shards: one model pair resident on the single card. All use the same 40 plans/seed.
for label in before after; do
    CK=$INIT; [ "$label" = after ] && CK=$LAST
    for shard in 0 1 2; do
        name=chat2_${label}_${shard}
        if [ ! -f "$R/$name.DONE" ] && [ -d "$R/$name" ]; then mv "$R/$name" "$R/$name.interrupted.$(date +%s)"; fi
        ev "$name" python evalfree.py --run "$CK" --replies "$O/replies.json" --coread_data ../data/coread.json \
            --games co_chat2 --crossed --n 40 --seed 4242 --temperature 0 --shard "$shard/3" --bs 2 --out_dir "$R/$name"
    done
done
ev chat2_summary python chat2_merge.py --after "$R/chat2_after_0" "$R/chat2_after_1" "$R/chat2_after_2" \
    --before "$R/chat2_before_0" "$R/chat2_before_1" "$R/chat2_before_2" --out "$R/chat2_summary.json"
note 'all done'
