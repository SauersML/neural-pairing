#!/bin/bash
# nl41, one 80 GB card, no training: where nl40's late steering and Ember's mood loss live. nl40 (the steered game on
# the twins' own targets) steers late (l15x4 47, l27x4 25 of 96; nl33w 34, 1) but Ember's mood fell (link > cut in 29
# of 40 new conversations; nl33w 37). nl40 and nl33w both train 300 steps from step_157, so their parts fit together.
# Two mixed checkpoints: nl40's link with nl33w's adapters (b40), and nl40's link and Quill adapter with nl33w's Ember
# adapter (b40q40). Each gets the mood scan on the 40 old (seed 6161), 40 new (7272) and 40 fresh (8383) conversations
# and the steered-twin test at l15x4 and l27x4; nl33w gets the fresh 40 too. A mixed checkpoint replaces nl33w only if
# link > cut is at least nl33w's count for both twins on all three sets and l27x4 beats nl33w's 1 of 96.
# Inputs: CK40 (nl40's step_300), CK33W (nl33w's step_300).
# Usage: bash nl41.sh OUT_DIR CODE_TGZ CK40 CK33W
set -u
O=$1 TGZ=$2 C40=$3 C33W=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset NL_TINY NL_MAXNEW NL_AFFECT_N
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
J=/workspace/jobs/nl41
mkdir -p $J/n40 $J/n33w
tar --no-same-owner --exclude='._*' -xf $C40 -C $J/n40 || { note "bad $C40"; exit 1; }
tar --no-same-owner --exclude='._*' -xf $C33W -C $J/n33w || { note "bad $C33W"; exit 1; }
A=$J/n40/step_300 B=$J/n33w/step_300
[ -s $A/bridge.pt ] && [ -s $A/quill/adapter_model.safetensors ] && [ -s $B/bridge.pt ] \
    && [ -s $B/ember/adapter_model.safetensors ] || { note 'inputs missing'; exit 1; }
mkdir -p $J/b40 $J/b40q40
cp $A/bridge.pt $J/b40/ && cp -r $B/quill $B/ember $J/b40/
cp $A/bridge.pt $J/b40q40/ && cp -r $A/quill $J/b40q40/ && cp -r $B/ember $J/b40q40/
R=$O/results
mkdir -p $R
mood() {  # NAME STEP_DIR SEED OUT
    note "mood $1 $3 start"
    (cd pairing && python moodscan.py --run $2 --names $1 --out $R/$4 --seed $3) > $R/mood_$1_$3.log 2>&1
    note "mood $1 $3 exit $?"
}
steer() {  # TAG STEP_DIR LAYER BOOST
    note "$1 start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer $3 --boost $4 --run $2 \
        --out $R/$1 > $R/$1.log 2>&1 && touch $R/$1.DONE
    note "$1 exit $?"
}
MAXJ=4; [ "$MEM" -lt 79000 ] && MAXJ=2
slot() { while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 10; done; }
for h in b40 b40q40; do
    slot; steer ${h}_l27x4 $J/$h 27 4 &
    sleep 20
    slot; mood $h $J/$h 7272 mood_new &
    sleep 20
done
for h in b40 b40q40; do
    slot; mood $h $J/$h 6161 mood_dev &
    sleep 20
    slot; mood $h $J/$h 8383 mood_fresh &
    sleep 20
    slot; steer ${h}_l15x4 $J/$h 15 4 &
    sleep 20
done
slot; mood nl33w $B 8383 mood_fresh &
wait
note 'all done'
