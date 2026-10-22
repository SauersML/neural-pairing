#!/bin/bash
# magic-nl2w, one 48 GB card: an evaluation worker. Training pods send it checkpoints directly (send_job.sh copies a
# step folder, its run's replies.json and data/ from the training pod to /workspace/jobs/TAG on this pod, then writes
# READY), so a checkpoint is evaluated within seconds of being saved instead of after a 13-minute upload from the Mac,
# while its run trains on. Each job runs the evaluations nl20.sh runs on its final checkpoint, into results/TAG/.
# Ends when /workspace/jobs/STOP exists and no job is waiting, or after HOURS.
# Usage: bash evalworker.sh OUT_DIR CODE_TGZ EXTRA_EVALS HOURS
set -u
O=$1 TGZ=$2 EXTRA=$3 HOURS=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf w && mkdir -p $O w /workspace/jobs && tar --no-same-owner -xzf $TGZ -C w && cd w/pairing || exit 1
mkdir -p ../data ../results
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
END=$(( $(date +%s) + HOURS * 3600 ))
note "worker ready"
while (( $(date +%s) < END )); do
    J=$(find /workspace/jobs -mindepth 2 -maxdepth 2 -name READY -printf '%T@ %h\n' | sort -n | head -1 | cut -d' ' -f2)
    if [ -z "$J" ]; then
        [ -f /workspace/jobs/STOP ] && break
        sleep 20; continue
    fi
    TAG=$(basename $J) R=$O/results/$TAG
    mv $J/READY $J/TAKEN; mkdir -p $R
    cp -f $J/data/coread.json $J/data/play.json $J/data/talk.json $J/data/pools.json ../data/ 2> /dev/null
    LAST=$(find $J -mindepth 1 -maxdepth 1 -type d -name 'step_*' | head -1)
    REP=$J/replies.json
    ev() { [ -f $R/$1.DONE ] && return; n=$1; shift; "$@" > $R/$n.log 2>&1 && touch $R/$n.DONE; note "$TAG $n: exit $?"; }
    note "$TAG: evaluating $LAST"
    ev payload python payload_ctl.py --run $LAST --replies $REP --out $R/payload_final.json --n 20 --coread
    LAST=$LAST R=$R O=$J bash $EXTRA > $R/extra_evals.log 2>&1; note "$TAG extra evaluations: exit $?"
    ev evalfree python evalfree.py --run $LAST --replies $REP --out_dir $R/evalfree --n 20 --bs 8 --coread_data ../data/coread.json
    ev evalfree_sum python evalfree.py --summarize --out_dir $R/evalfree --tag headline_$TAG
    cp ../results/evalfree_headline_$TAG.json $R/ 2> /dev/null
    ev payload_pause4 python payload_ctl.py --run $LAST --replies $REP --out $R/payload_final_pause4.json --n 20 --coread --pause 4
    ev transcripts python transcripts.py --run $LAST --replies $REP --out $R/transcripts.txt --n 4 --device cuda --coread_data ../data/coread.json
    for g in co_qa co_play; do
        ev readmap_$g python readmap.py --run $LAST --replies $REP --game $g --out $R/readmap_$g.json --n 8 --coread_data ../data/coread.json
    done
    ev bridge python analyze_bridge.py --run $LAST --replies $REP --out $R/bridge.json --n 16 --coread_data ../data/coread.json
    note "$TAG: done"
    rm -rf $J
done
note "worker done"
