#!/bin/bash
# nl28, one 48 GB Ampere card (RTX A6000 or A40; the script refuses others): nl27's question mix with the loss on the
# two linked models' averaged A-D probabilities (--qa_pool 1); the control trains the same loss with the link cut
# (--solo --qa_pool 1). Eight micro-batches of two (global batch 16), so the pooled pair fits in 48 GB. bench_eval.py
# scores both; the primary test is pair/avg against ensemble and against pair/zero/avg on the test questions, both
# differences positive with exact two-sided p < .05. No text arm. Waits up to three hours for
# /workspace/jobs/nl28/{step_1320, replies.json, data/bench.json} and the READY file.
# Usage: bash chains/nl28.sh OUT_DIR CODE_TGZ [STEPS=300]
set -euo pipefail
O=$1 TGZ=$2 STEPS=${3:-300}
mkdir -p "$O"
O=$(cd "$O" && pwd)
TGZ=$(realpath "$TGZ")
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date '+%F %T') $1" >> "$O/chain.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$O/gpus.txt"
python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, 'nl28 requires exactly one visible GPU'
p = torch.cuda.get_device_properties(0)
assert (p.major, p.minor) in ((8, 0), (8, 6)), f'Ampere only, not {p.name} ({p.major}.{p.minor})'
assert p.total_memory >= 44 * 2**30, f'48 GB card required (A6000 and A40 report 44.4 GiB): {p.total_memory / 2**30:.1f} GiB'
print(p)
PY
mkdir -p "$O/source"
tar --no-same-owner -xzf "$TGZ" -C "$O/source"
cd "$O/source"
cd pairing
J=/workspace/jobs/nl28
note "waiting for $J/READY"
for _ in $(seq 1080); do [ -f "$J/READY" ] && break; sleep 10; done
[ -f "$J/READY" ] || { note 'no inputs after three hours'; exit 1; }
mkdir -p ../data
cp -rf "$J/data/." ../data/
cp "$J/replies.json" "$O/replies.json"
R=$O/results
mkdir -p "$R"
ev() {
    local n=$1
    shift
    [ -f "$R/$n.DONE" ] && return 0
    note "$n: starting"
    if "$@" > "$R/$n.log" 2>&1; then
        touch "$R/$n.DONE"
        note "$n: exit 0"
    else
        local rc=$?
        note "$n: exit $rc"
        return "$rc"
    fi
}
ev split python qa_split.py --run "$J/step_1320" --bs 16 --out "$R/qa_split.json"
[ -s "$R/qa_split.json" ] || { note 'no question split'; exit 1; }
Q=(--replies "$O/replies.json" --init "$J/step_1320" --gate_norm --hold 100000
   --warmup_steps 10 --relay_first 0 --steps "$STEPS" --bs 16 --micro 8 --seed 0
   --anchor 0 --comp 0 --warm 0 --ramp 0 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3
   --eval_every 100000 --first_eval 100000 --save_every 100000 --ckpt none
   --qa_focus "$R/qa_split.json" --qa_focus_p 0.5)
note "two arms, $STEPS steps each, both trained on the averaged letters: linked (pair/avg) and alone (ensemble)"
ev qa_pair python train.py "${Q[@]}" --out "$O/qa_pair" --qa 1 --qa_pool 1
ev qa_solo python train.py "${Q[@]}" --out "$O/qa_solo" --solo --qa_pool 1
for a in pair solo; do
    [ -f "$O/qa_$a/step_$STEPS/bridge.pt" ] || { note "missing step_$STEPS for $a"; exit 1; }
done
ev bench python bench_eval.py --pair "$O/qa_pair/step_$STEPS" --solo "$O/qa_solo/step_$STEPS" \
    --bs 16 --out "$R/bench.json"
python - "$R/bench.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for name, result in r['test'].items():
    assert 'pair/avg' in result['accuracy'] and 'pooled' in result, 'need current main bench_eval with pooled scoring'
    tests = result['pooled']
    passed = all(tests[k]['diff'] > 0 and tests[k]['p'] < .05 for k in ('ensemble', 'pair/zero/avg'))
    print(name, 'pooled_success', passed, json.dumps(tests), flush=True)
PY
note 'all done'
