#!/bin/bash
# nl25: one-way links onto an unmodified model (link.py, train.py --frozen). Two arms, back to back on one A100 80 GB
# (nl24c's pod; no Ada card: nl24's torch.multinomial hit a device-side assert on an RTX 6000 Ada): ember, target Ember
# (the original AFM-4.5B, bit for bit) read by Quill; quill, target Quill read by Ember. Each arm starts from a two-way
# step (nl24c's final step, else nl22c step_157): the target's adapter and read parts are dropped, the reader keeps its
# LoRA and read parts, and the target's show parts keep training as the reader's channel. 80 GB settings as nl24c's:
# eight microbatches without gradient checkpointing, chat tests in three shards at once with batches of 4.
#   1. before, per arm: the two-way step read one-way (--frozen), no one-way training: the chat2 free-chat test (link,
#      cut, linked to a stranger; the reader the only receiver) and the payload table (value and co-reading games);
#   2. per arm: 150 value-game warm-up steps (all seven value games, the reader as the receiver), then nl24's mix with
#      chat2 (same weights) to 450 steps or --minutes, whichever ends first;
#   3. after, per arm: the same chat2 test and payload table.
# The chat2 test is nl24c's: the same 40 plans (seed 4242), greedy. Its summaries count the reader naming the TARGET's
# private fact before the target does, trained and held-out kinds apart (chat2_merge.py; one-way manifests list the
# reader only). No checkpoint selection on test scores.
# Time on an A100 80 GB (magic-nl22c there: 5.5 s per two-way step without checkpointing, 201 s per evaluation; a one-way
# step ~0.7 of that): per arm ~25 min training, ~12 min per chat test (14 chats per shard, link, cut and stranger
# chats each in batches of 4), ~15 min of payload tables; ~2.1 h in all.
# Usage: [NL25_WAIT_PID=PID] bash nl25.sh OUT_DIR CODE_TGZ INIT_STEP[,FALLBACK...] REPLIES DATA_DIR CHAT2_JSON
#   INIT_STEP: the first of the comma-separated steps that holds bridge.pt; DATA_DIR holds coread, play, talk, pools and
#   chat .json (/workspace/jobs/nl24/data); CHAT2_JSON is nl24c's generated chat2.json; NL25_WAIT_PID: start once that
#   process (nl24c's chain on the same card) has ended. NL25_TINY=1: tiny random models on the CPU (smoke runs), with
#   NL25_ARCH="--heads 2 --head_dim 16 --xblocks 1,2,4" matching the tiny two-way step.
set -uo pipefail
O=$1 TGZ=$2 INITS=$3 REPLIES=$4 DATA=$5 CHAT2=$6
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${NL25_TINY:-0}" = 1 ]; then export NL_TINY=1 NL_DEVICE=cpu; else unset NL_TINY NL_MAXNEW; fi    # smoke runs
export NL_QUICK=${NL_QUICK:-relay_word,relay_number,merge_sum,lookup,merge_fact,co_qa}
STEPS=${NL25_STEPS:-450} RELAY=${NL25_RELAY:-150} MINUTES=${NL25_MINUTES:-40}
N_CHAT=${NL25_N_CHAT:-40} N_PAY=${NL25_N_PAY:-40} BS=${NL25_BS:-16} SHARDS=3
mkdir -p "$O/source" "$O/results"
note() { echo "$(date '+%F %T') $*" >> "$O/chain.log"; }
if [ -n "${NL25_WAIT_PID:-}" ]; then
    note "waiting for process $NL25_WAIT_PID to end"
    while kill -0 "$NL25_WAIT_PID" 2> /dev/null; do sleep 30; done
fi
tar --no-same-owner -xzf "$TGZ" -C "$O/source"
cd "$O/source/pairing"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$O/gpus.txt"
INIT=
for s in ${INITS//,/ }; do
    if [ -s "$s/bridge.pt" ] && [ -d "$s/quill" ] && [ -d "$s/ember" ]; then INIT=$s; break; fi
done
[ -n "$INIT" ] || { note "no two-way step among $INITS"; exit 1; }
[ -s "$CHAT2" ] || { note "no chat2 data at $CHAT2"; exit 1; }
for name in coread play talk pools chat; do cp "$DATA/$name.json" ../data/ || { note "no $DATA/$name.json"; exit 1; }; done
cp "$CHAT2" ../data/chat2.json
note "start: init $INIT; $STEPS steps ($RELAY warm-up) or $MINUTES min per arm; $N_CHAT chats, payload $N_PAY"
R=$O/results
ev() {
    local name=$1; shift
    [ -f "$R/$name.DONE" ] && return 0
    if [ -d "$R/$name" ]; then mv "$R/$name" "$R/$name.interrupted.$(date +%s)"; fi
    note "$name start"
    if "$@" > "$R/$name.log" 2>&1; then touch "$R/$name.DONE"; note "$name done"; else note "$name failed"; return 1; fi
}
chats() {    # chats NAME RUN [--frozen T]: the chat2 test in SHARDS shards at once (NAME_0, NAME_1, ...)
    local name=$1 run=$2 k; shift 2
    for k in $(seq 0 $(( SHARDS - 1 ))); do
        ev "${name}_$k" python evalfree.py --run "$run" --replies "$REPLIES" --coread_data ../data/coread.json \
            --games co_chat2 --crossed --n "$N_CHAT" --seed 4242 --temperature 0 --shard "$k/$SHARDS" --bs 4 \
            --out_dir "$R/${name}_$k" "$@" &
    done
    wait
}
shards() { local k; for k in $(seq 0 $(( SHARDS - 1 ))); do echo "$R/${1}_$k"; done; }

for T in ember quill; do
    A=$O/oneway_$T
    X=$([ "$T" = ember ] && echo quill || echo ember)
    mkdir -p "$A" "$R/oneway_$T"
    chats "oneway_$T/chat2_before" "$INIT" --frozen "$T"
    ev "oneway_$T/payload_before" python payload_ctl.py --run "$INIT" --frozen "$T" --replies "$REPLIES" \
        --out "$R/oneway_$T/payload_before.json" --n "$N_PAY" --bs 8 --coread
    if [ ! -f "$A/run.DONE" ]; then
        if [ -d "$A/run" ]; then mv "$A/run" "$A/run.interrupted.$(date +%s)"; fi
        note "oneway_$T train start: $T frozen, $X reads"
        if python train.py --replies "$REPLIES" --out "$A/run" --init "$INIT" --frozen "$T" --gate_norm --hold 100000 \
            --payload_w 3 --warmup_steps 10 --relay_first "$RELAY" \
            --relay_games relay_word,relay_number,relay_phrase,relay_digits,merge_sum,merge_fact,lookup \
            --steps "$STEPS" --minutes "$MINUTES" --bs "$BS" \
            --coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2 --chat2 0.25 \
            --comp 0 --warm 0 --ramp 0 --send_cost 0.01 --lr 1e-4 --lr_bridge 3e-4 --lr_alpha 3e-3 \
            --eval_every 100000 --quick_n 4 --save_every 100000 \
            --pause 4 --pause_p 0.5 --micro 8 --ckpt none ${NL25_ARCH:-} > "$A/train.log" 2>&1; then
            touch "$A/run.DONE"
            note "oneway_$T train done"
        else
            note "oneway_$T train failed"
            continue
        fi
    fi
    cp "$A/train.log" "$A/run/log.json" "$R/oneway_$T/" 2> /dev/null
    LAST=$(ls -d "$A"/run/step_* | sort -t_ -k2 -n | tail -1)
    chats "oneway_$T/chat2_after" "$LAST"
    ev "oneway_$T/payload_after" python payload_ctl.py --run "$LAST" --replies "$REPLIES" \
        --out "$R/oneway_$T/payload_after.json" --n "$N_PAY" --bs 8 --coread
    ev "oneway_$T/chat2_summary" python chat2_merge.py --before $(shards "oneway_$T/chat2_before") \
        --after $(shards "oneway_$T/chat2_after") --out "$R/oneway_$T/chat2_summary.json"
done
note 'all done'
