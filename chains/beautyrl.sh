#!/bin/bash
# beautyrl6, one 80 GB GPU: RL on the models' open conversations, rewarded by a pointwise judge only for what the link
# adds (rl.py --open_frac, judge.py, beauty_eval.py). The protocol, fixed before any result:
#  0. Every model runs whole on the one GPU (direct GPU-to-GPU copies were corrupt on multi-A40 hosts, so no model is
#     split). Judges: Qwen3-14B trains, beside the pair; Mistral-Small-24B-Instruct-2501, another family, is held out. A
#     judge scores one transcript from 1 to 9 on a rubric (judge.py); two transcripts compare by their score difference,
#     always length-matched.
#  1. Health, before the gate: gpu_check.py (an exact copy between every pair of GPUs; each model's five likeliest
#     completions of "The capital of France is" include Paris) and Judge.check for each judge (6 transcripts scored
#     alone and in one padded batch agree within 0.5 points; at least 5% probability on the digits; Paris). A failure
#     stops the chain.
#  2. Gate transcripts: 50 training-distribution prompts, two linked plays each from nl33's step 300, sampled before any
#     update (seed/gate_plays.json).
#  3. Sanity, each judge separately: on the first 30 sessions' first plays, the real transcript scores above each of
#     three degraded versions on at least 27 of 30, each degradation separately: turns shuffled across sessions, the
#     second speaker saying only "That's interesting, tell me more.", every turn a repeat of the first. A judge that
#     fails sanity is out. If Qwen3-14B fails, it is scored once more reasoning first (thinking on, greedy, up to 512
#     tokens), and that arm becomes the training judge if it passes.
#  4. Primary: the share of the 50 pairs on which the two judges' score differences have the same sign, at least 70%.
#     Training runs only if both judges pass sanity and the primary passes; otherwise the chain keeps gate.json and
#     ends.
#  5. Margins, fixed in the gate (beauty_eval.py): a strict win is a score difference above the judge's margin, a loss
#     one below minus the margin, 1/2 between.
#  6. Training from step 300 for 40 minutes: open sessions (5 turns of at most 40 tokens), 3 prompts a step, each played
#     4 times linked and 2 times cut; a linked play's reward is its mean strict win over the same prompt's cut plays on
#     rl.RUBRIC, plus diversity 0.3, the original models' fluency floor and -1 for writing as the partner; every third
#     step supervised.
#  7. Headline, 48 held-out prompts (test wordings x test concepts), 2 plays each at temperature 1: the held-out judge's
#     strict win rate of the trained pair's linked plays over step 300's on beauty_eval.EVAL (a rubric training never
#     used), the mean over prompts with its 95% betting interval; a gain if the lower end is above 1/2. Link-cut
#     control: each pair's linked plays against its own cut plays, before, after, and the change.
#  8. No regression: the steered-twin test after blocks 15 and 27 at four times the selected strength, and the direct
#     question, on the trained pair.
# Inputs: CODE_TGZ, CKPT_TAR (nl33's step_300), DATA_TGZ (data33.tgz).
# Usage: bash beautyrl.sh OUT_DIR CODE_TGZ CKPT_TAR DATA_TGZ
set -u
O=$1 TGZ=$2 CK=$3 DATA=$4
export PYTHONUNBUFFERED=1 NL_DEVICE=cuda NL_STREAM=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
unset NL_TINY NL_MAXNEW
rm -rf w && mkdir -p $O w && tar --no-same-owner -xzf $TGZ -C w && cd w || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpus.txt
note() { echo "$(date '+%F %T') $1" >> $O/chain.log; }
[ "$(head -1 $O/gpus.txt | grep -oE '[0-9]+ MiB' | cut -d' ' -f1)" -ge 76000 ] || { note 'needs an 80 GB GPU'; exit 1; }
J=/workspace/jobs/beautyrl
mkdir -p $J && tar --no-same-owner --exclude='._*' -xf $CK -C $J || { note "bad $CK"; exit 1; }
INIT=$J/step_300
[ -s $INIT/bridge.pt ] || { note 'no step_300'; exit 1; }
tar --no-same-owner --exclude='._*' -xzf $DATA -C . || { note "bad $DATA"; exit 1; }
[ -s data/coread.json ] && [ -s replies.json ] || { note 'data missing'; exit 1; }
sha256sum pairing/rl.py pairing/judge.py pairing/beauty_eval.py pairing/gpu_check.py chains/beautyrl.sh \
    seed/gate_plays.json > $O/code_sha256.txt
TRAIN=Qwen/Qwen3-14B HELD=mistralai/Mistral-Small-24B-Instruct-2501 THINK=512
R=$O/results
mkdir -p $R
note 'health: GPU copies and the twins on the France prompt'
(cd pairing && CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -) \
    python gpu_check.py --models Qwen/Qwen3-4B arcee-ai/AFM-4.5B --out $R/gpu_check.json) > $R/gpu_check.log 2>&1 \
    || { note 'health check failed'; exit 1; }
[ -s $R/gate_plays.json ] || [ ! -s seed/gate_plays.json ] || cp seed/gate_plays.json $R/gate_plays.json
if [ ! -s $R/gate.json ]; then
    note 'gate: batch check, sanity and agreement of both judges on 50 pairs of step 300 plays'
    (cd pairing && python beauty_eval.py --phase gate --runs init=$INIT --judges $TRAIN $HELD --think $THINK \
        --bs 16 --out $R) > $R/gate.log 2>&1 || { note 'gate failed'; exit 1; }
fi
python -c "import json, sys; sys.exit(0 if json.load(open('$R/gate.json'))['pass'] else 3)"
case $? in
    0) note 'gate passed' ;;
    3) note 'gate not passed: a judge failed sanity or the judges disagree, no training'; note 'all done'; exit 0 ;;
    *) note 'gate unreadable'; exit 1 ;;
esac
read -r TT MARGIN < <(python -c "import json; g = json.load(open('$R/gate.json')); print(g['train_think'], g['margins'][g['train_arm']])")
note "training judge think $TT, margin $MARGIN"
if [ ! -f $O/rl.DONE ]; then
    note 'training: linked plays judged against cut plays'
    (cd pairing && python rl.py --init $INIT --replies ../replies.json --coread_data ../data/coread.json --out $O/rl \
        --steps 150 --minutes 40 --mix 3 --open_frac 1 --open_groups 3 --G 4 --open_cut 2 --open_turns 5 \
        --max_turn 40 --judge $TRAIN --judge_think $TT --win_margin $MARGIN --div 0.3 --floor_w 1 --floor_q 0.1 \
        --eval_every 1000 --eval_n 16 --eval_task_n 8 --save_every 50) > $O/rl.log 2>&1 || { note 'training failed'; exit 1; }
    touch $O/rl.DONE
fi
LAST=$O/rl/$(ls -1 $O/rl | grep '^step_' | sort -t_ -k2 -n | tail -1)
[ -s $LAST/bridge.pt ] || { note 'no trained checkpoint'; exit 1; }
note "evaluating $LAST"
(cd pairing && python beauty_eval.py --phase gen --runs init=$INIT rl=$LAST --out $R) > $R/gen.log 2>&1 \
    || { note 'held-out plays failed'; exit 1; }
steer() {  # TAG:LAYER:BOOST
    IFS=: read -r t l b <<< "$1"
    note "$t start"
    python pairing/steering_scale.py --backend torch --phase all --bs 6 --layer $l --boost $b \
        --run $LAST --out $R/$t > $R/$t.log 2>&1 \
        && python pairing/steer_ask.py --run $LAST --scale $R/$t --out $R/ask_$t.json > $R/ask_$t.log 2>&1
    note "$t exit $?"
}
note 'judging, beside the first steered-twin check'
steer l15x4:15:4 &
JUDGES=$HELD
[ "$TT" = 0 ] && JUDGES="$JUDGES $TRAIN"
(cd pairing && python beauty_eval.py --phase judge --judges $JUDGES --bs 16 --out $R) > $R/judge.log 2>&1
note "judging exit $?"
steer l27x4:27:4 &
wait
note 'all done'
