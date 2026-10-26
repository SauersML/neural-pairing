#!/bin/bash
# Later evaluations of a finished run, called by its chain from w/pairing with LAST (the final step's directory), R (the
# results directory) and O (the run's output directory) set: turnbench.py (the same never-seen secrets under every turn
# structure), lens.py (what each tick's reads made more likely), bandwidth.py (bits per item and per silent tick) and
# free dialogue with the crossed control (evalfree.py --crossed) for the dialogue games the run has data for.
set -u
export NL_DEVICE=cuda
unset NL_TINY NL_STREAM NL_QUICK
python turnbench.py --run $LAST --replies $O/replies.json --out $R/turnbench.json --n 128 --bs 16 > $R/turnbench.log 2>&1 \
    || python turnbench.py --run $LAST --replies $O/replies.json --out $R/turnbench.json --n 128 --bs 8 > $R/turnbench_bs8.log 2>&1
echo "turnbench: exit $?"
G=relay_word,relay_number,merge_sum; P=relay_word,relay_number
[ -s ../data/play.json ] && { G=$G,co_play; P=$P,co_play; }
[ -s ../data/talk.json ] && G=$G,co_talk
G=$G,pen_secret
python lens.py --run $LAST --replies $O/replies.json --coread_data ../data/coread.json --games $G \
    --n 2 --ablation_bs 16 --head_chunk 32 --max_positions 1024 --out_dir $R/lens > $R/lens.log 2>&1
echo "lens: exit $?"
python lens.py --run $LAST --replies $O/replies.json --coread_data ../data/coread.json --games $P --n 1 --pause 3 --ticks all \
    --ablation_bs 16 --head_chunk 32 --max_positions 1024 --out_dir $R/lens_pause3 > $R/lens_pause3.log 2>&1
echo "lens pause3: exit $?"
# bandwidth: the receiver's code-length saving on digit strings of 1 to 16 digits, numbers and words, with 0 to 8
# silent ticks
python bandwidth.py --run $LAST --replies $O/replies.json --out $R/bandwidth.json --n 128 --bs 16 --logit_chunk 128 \
    --greedy_n 4 --greedy_bs 4 --max_turn 64 --clip_bits 128 > $R/bandwidth.log 2>&1
echo "bandwidth: exit $?"
# free dialogue with the crossed control: each receiver talks with another episode's partner, which never had the
# real partner's secret
C=""
[ -s ../data/talk.json ] && C="$C co_talk"
[ -s ../data/play.json ] && C="$C co_play"
if [ -n "$C" ]; then
    python evalfree.py --run $LAST --replies $O/replies.json --coread_data ../data/coread.json --games $C --crossed --n 20 --bs 4 \
        --out_dir $R/evalfree_crossed > $R/evalfree_crossed.log 2>&1
    echo "evalfree crossed: exit $?"
    python evalfree.py --summarize --out_dir $R/evalfree_crossed --tag crossed >> $R/evalfree_crossed.log 2>&1
    cp ../results/evalfree_crossed.json $R/ 2>/dev/null
fi
