#!/bin/bash
# magic-beautyrl5: RL on the twins' open conversations, rewarded by a pointwise judge only for what the link adds
# (rl.py --open_frac, judge.py, beauty_eval.py). Preregistered here, before any result:
#  0. One 80 GB GPU; every model runs whole on it (direct GPU-to-GPU copies were corrupt on the multi-A40 pods, so no
#     model is split and nothing moves between GPUs). Judges: Qwen3-14B trains, beside the pair on the same card;
#     Mistral-Small-24B-Instruct-2501, another family, is held out. A judge scores one transcript from 1 to 9 on a
#     rubric: the expected digit over the nine digit tokens at the answer position, one forward pass, no sampling,
#     right-padded batches; a non-finite score is an error. Two transcripts compare by their score difference, always
#     length-matched (each turn cut to the shorter one's words).
#  1. Health, logged before the gate: an exact copy check between every pair of GPUs (none with one GPU) and each
#     model's five likeliest completions of "The capital of France is", which must include Paris (gpu_check.py for the
#     twins, Judge.check for the judges). GPU check, each judge, before its gate scores: 6 transcripts scored alone and
#     inside one padded batch must agree within 0.5 points (bf16 batch noise was 0.10 for Qwen3-8B on the Mac GPU and
#     0.28 on an A40), and the judge must put at least 5% probability on the digits; otherwise the chain stops.
#  2. Gate transcripts: 50 training-distribution prompts, two linked plays each from nl33's step 300, sampled before
#     any update (magic-beautyrl2's gate_plays.json, the same prompts and seeds, reused).
#  3. Sanity, each judge separately: on the first 30 sessions' first plays, the real transcript must score above each
#     of three degraded versions of itself on at least 90% (27 of 30), each degradation separately: turns shuffled
#     across sessions (same speakers, so neither twin answers the other), the second speaker saying only "That's
#     interesting, tell me more.", every turn a repeat of the first. A judge that fails sanity is out, whatever its
#     agreement. If Qwen3-14B fails, it is scored once more reasoning first (thinking on, greedy, up to 512 tokens, the
#     digit read after the reasoning), and that arm becomes the training judge if it passes sanity.
#  4. Primary, the one frozen statistic that decides: the share of the 50 pairs (each prompt's two plays) on which the
#     two judges' score differences have the same sign, at least 70%. Spearman over the 100 transcripts is reported,
#     not used. Training runs only if both judges pass sanity and the primary passes; otherwise the chain keeps
#     gate.json and ends.
#  5. Margins, frozen in the gate: k = the ratio of the judges' median absolute pair differences; the training judge's
#     margin is the median of |d_train - k d_held| over the 50 pairs, the held-out judge's that divided by k. A strict
#     win is a score difference above the judge's margin, a loss below minus it, 1/2 between.
#  6. Training from step 300, 40 minutes: open sessions (5 turns of at most 40 tokens), 3 prompts a step, each played
#     4 times linked and 2 times cut; a linked play's judge reward is its mean strict win over the same prompt's cut
#     plays on rl.RUBRIC (the training judge), so what the judge would award with the link cut earns nothing; plus
#     diversity 0.3, the original models' fluency floor, -1 for writing as the partner; every third step supervised.
#  7. Headline, 48 held-out prompts (test wordings x test concepts), 2 plays each at temperature 1: the held-out judge's
#     strict win rate of the trained pair's linked plays over step 300's on EVAL (a rubric training never saw), mean
#     over prompts with its 95% betting interval; a gain if the interval's lower end is above 1/2. Link-cut control:
#     each pair's linked plays against its own cut plays (strict), before, after, and the change. Also: rl.RUBRIC, mean
#     score differences, words per turn, and the training judge's scores when it trains without reasoning.
#  8. No regression: the steered-twin test after blocks 15 and 27 at four times the selected strength, and the direct
#     question, on the trained pair (step 300: 47 and 35 of 96).
# Inputs: CODE_TGZ, CKPT_TAR (nl33's step_300, published), DATA_TGZ (data33.tgz, published). One 80 GB GPU.
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
