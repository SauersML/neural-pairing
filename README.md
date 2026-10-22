# neural-pairing

Two language models from different families, Quill (Qwen3-4B) and Ember (AFM-4.5B), joined by a learned link between
their hidden states. No tokens cross the link: each model still writes ordinary text, and the link adds what one model
reads from the other's hidden states into its own.

![Two twins chatting, each with a private city, with the link cut and with the link on](figures/hero_link.png)

## The link

At 8 depths, each model reads the other's hidden states, in both directions, inside one joint forward pass.

![The link joins the two models at 8 depths](figures/link_mechanism_depths.png)

Each reading token attends over the partner's states, which pass through a gate, and adds the result into its own
residual stream. It can also attend to an empty slot and read nothing.

![One read through the link](figures/link_mechanism.png)

A token reads the partner's tokens from the same moment or earlier, never later ones, so a joint pass over a whole
conversation equals generating it token by token.

![Which partner tokens a token can read](figures/link_mechanism_ticks.png)

## What crosses the link

Words, numbers and phrases never used as values in training cross in both directions.

![A value never seen in training crosses the link](figures/values_unseen_game.png)

![Real partner against a shuffled partner, every held-out value](figures/values_unseen_results.png)

The models also read each other's texts: a passage on one side, the question on the other.

![Shared reading](figures/shared_reading.png)

The link also works in games it never trained on.

![Games never seen in training](figures/games_untrained.png)

## Chatting with no task

Chat training: the original model writes each turn while it sees both private cities; the linked twin learns the same
turn without the partner's city in its prompt, so the city can only arrive through the link.

![The privileged teacher](figures/teacher_method.png)

Twenty free chats after chat training, cities never seen in training.

![Opening messages of 20 free chats](figures/hero_grid.png)

The partner's city comes out only when the link is on, and a twin linked to a stranger says the stranger's city.

![Partner's city said first: link on, cut, linked to a stranger](figures/chat_names.png)

![Quill's score for each city given Ember's private city](figures/city_matrix_quill.png)

Before chat training, the twins repeated earlier turns word for word; after it, almost never.

![Turns that repeat an earlier turn, before and after chat training](figures/chat_loops.png)

## Exam questions

From the same checkpoint, a linked pair, each model alone and a pair answering in turn in text each trained 300 steps on
ARC-Challenge and MMLU questions. On 3,165 test questions the linked pair is no better than Quill alone, and averaging
the two models' answers beats both.

![Test questions right: each model alone, both averaged, talking in text, link cut, linked](figures/uplift.png)

What the link does instead: Ember reads Quill's answer and takes it, even on questions Ember gets right on its own.

![Ember's answers with the link cut and linked, on questions only one model gets right](figures/deference.png)

## Where this goes

The link is a bus between two networks that were never trained together: any pair of models can share what they hold
privately (a document, a fact, a plan) without writing it out, and a reader model can follow another model's hidden
states token by token while it runs. Under way: chat over twelve kinds of private facts with four held out, and a one-way
link into a model left exactly as it was released.

<details><summary>Method, controls, runs and exactness checks</summary>

### The link in words

After blocks 6, 9, …, 27 of 36, each model reads the other. The reader normalises its state and maps it to a query. The
sender normalises its states and maps them to keys and values; a gate computed from each sender state (one number per
position and depth) scales that position's value, so a closed gate hides the position. The reader attends over the
sender's positions plus an empty slot that returns nothing, maps the result back to its own width, and adds it to its
residual stream, scaled by a learned write size relative to its typical residual size. Each head has a learned bias by
how many ticks ago a partner token was read. Both reads at a depth use the states from before that depth, so the
dependency graph is acyclic and a teacher-forced pass over a whole conversation equals generating it token by token.
8 heads of 128 dimensions in float32 per depth and direction. The link and rank-32 LoRA adapters on both models train;
the original weights are frozen.

### Starting from a fitted map (`align.py`, `train.py --content_init qkvo`)

A link with random Q and K attends to arbitrary partner positions, so its first gradients are weak and inconsistent.
The fitted start sets every exchange so that a reader position attends to the partner position that has read the same
text:

1. Both models read the same 192 texts (none from the held-out split). A token of one model and a token of the other
   match when both end at the same character.
2. At each depth, a ridge regression maps the sender's normalized states to the reader's (about 4,300 matched token
   pairs; the penalty is chosen on texts outside the fit).
3. The sender's keys are set so that a query's score on a key is its similarity to the reader state the map predicts
   from that key. The values and the output map start from the top singular directions of the map, so a write delivers
   the matched partner state mapped into the reader's space.

On the 6 texts outside the fit, the reader's top attention lands on the matched partner token 89–97% of the time
(chance 1.2%; 1% with random Q and K); R² of the map is 0.51–0.66. The write size starts at 0.1, so at step 0 each write is
a small mapped copy of the matched partner state (about 5% of the residual's size). A random start also opens the
link when training starts with the value games only, more slowly and with less transfer to games it was not trained
on (name–city pairs 20/32 vs 29/32 at step 200); in the full mix from a random start, the gates closed before the link
opened.



### Games and turn structures

**Value games** put a value in one model's prompt that the other model must write: relay a word, a 4-digit number, a
two-word phrase or up to 16 digits; add two numbers, one in each prompt; join a name and a city, one in each prompt;
look up a code in the partner's table.

**Shared reading** (`coread.py`): a text or task is split between the models (a question on one side and the
passage on the other, a premise and a hypothesis, a word problem, code and a question about its output, a multiple-
choice question and its options, a text to translate or paraphrase), and the reader answers. With distillation
(`prep_coread.py`), the reader's whole reply is trained toward the reply the original model writes when it reads the
whole text itself.

**Play** (`co_play`, written by `prep_talk.py --play`): six-turn conversations in which each model holds a secret: guess
my word, the gift you wish for, where I am on holiday, a shared story on your topic, a rhyme on your word. The opener
names the partner's secret in its first turn and the partner names the opener's in its reply. The models wrote these
conversations themselves, each naming turn with the partner's secret shown; in training the secret is gone from the
prompt, so it can only come through the link. Conversations in which the opener says its own secret are dropped.

**Turn structures** (`session.py`):

- duplex: both models write at once, one token per tick, and neither sees the other's text;
- dialogue: alternating turns; the listener reads the speaker's words as they are written, and reads its hidden
  states through the link;
- silent rounds (`--pause K`): K ticks before each reply or turn in which both models write only a space, so only
  hidden states pass;
- shared pen (`--pen P`): the two models write one text together, alternating words.

### Evaluation

Every value in an evaluation episode comes from the held-out split: it is never used as a value in training (it can
still appear as an ordinary word in training texts). Words, phrases, names, cities and code words come from disjoint
pools; sums are 50–99 in evaluation and 0–49 in training; numbers and digit strings whose digit sum is 3 modulo 7 are
held out (from 673d582; earlier runs drew training numbers from the full range). Each game has 16 episodes at every
evaluation step (`train.py --payload_eval`) and 20 on the final checkpoint (`payload_ctl.py`), half with each model as
the receiver.

- **Shuffle control.** Each episode is also run with the partner of another episode of the same game, in the same
  roles. The receiver's input is perturbed as much as with the real partner, but the value is gone. The readout is the
  cross-entropy on the value tokens with the shuffled partner minus with the real partner, per episode.
- **Test.** The count of episodes in which the real partner wins, with an exact one-sided sign test. Ties (a link that
  writes exactly nothing) are left out. The size of the gap is not the readout: a confident wrong copy from the
  shuffled partner inflates it.
- The link-cut control (no partner at all) is also recorded; it changes the receiver's input more than a shuffle does.
- In free generation of dialogues, a shuffled partner is not an isolated control: the generated text carries
  information from episode to episode around the shuffle, so free-generation comparisons are with the link cut.

**Other measurements:** `turnbench.py` scores the same secrets under every turn structure; `lens.py` shows, per token,
how much the link changed the reader's next-token distribution, which partner tick it came from (tick ablations) and
what each exchange's write decodes to downstream; `play_live.py` plays a game on the Mac from a trained checkpoint
(4-bit MLX) and saves it as a web page with each token shaded by the size of the link's write.

### Training (`train.py`)

The link and the LoRA adapters train together. A run starts with the value games only (`--relay_first`, all seven
games), then trains on the full mix of value games, shared reading, play, dialogue and identity anchors, with the gates
and the write sizes frozen (`--hold`). `--micro K` splits each batch into K groups of similar length with summed gradients, so short
value episodes are not padded to the length of play conversations: on one A100 80 GB, 4.5 s per step for the full mix
instead of 8.

### Chat training with a privileged teacher (`prep_talk.py --chat`, `train.py --chat`)

The original model writes each turn of a chat with both private cities in its prompt. The linked twin is trained on the
same turn with only its own city in its prompt; the partner holds the other city, so matching the turn needs the link.
Turns are kept by how much the partner's city raises their likelihood under the original model (the summed log-probability
difference with and without the city; median 5.41 nats per turn, positive for 90.2% of 4,032 training turns), and chats
in which a model role-plays the link in text (for example "[Ember's Activation Vector]") are removed (252 of 1,213).
The chat test (`evalfree.py`, `chat_merge.py`) plays 20 greedy chats from fixed openings under three conditions: link
on, link cut, and linked to a stranger, an isolated twin holding a different city; sign tests are paired by chat.

### Exam questions (`bench_eval.py`, `chains/nl23.sh`)

Three arms start from nl20 step 1,320 and train 300 steps on ARC-Challenge and MMLU training questions only: the pair
with the link open (`train.py --qa 1`), each model alone (`--solo`), and the pair answering in turn in text with the link
closed (`--text_pair`). Each model's answer is its most likely letter after "The answer is". The pair answers through the
model whose single version scores higher on 595 validation questions (Quill, 78.8% against Ember's 77.1%), fixed before
any test question is scored. Tests are exact McNemar on the 3,165 test questions (1,165 ARC-Challenge, 2,000 MMLU; 27
MMLU questions appear twice).

| Way of answering | Right | Linked pair minus this (95% interval), p |
|---|---|---|
| Quill, original | 73.7% | +4.9 points (+3.6, +6.2), p 4e-13 |
| Ember, original | 69.4% | +9.3 (+7.6, +10.8), p 6e-27 |
| Quill alone, trained | 79.1% | −0.5 (−1.6, +0.6), p 0.42 |
| Ember alone, trained | 75.7% | +3.0 (+1.5, +4.5), p 1e-4 |
| Both alone, letter probabilities averaged | 81.1% | −2.5 (−3.6, −1.3), p 5e-5 |
| Text, Ember answers first | 72.4% | +6.3 (+4.7, +7.7), p 3e-15 |
| Linked pair, link cut | 78.7% | 0.0 (−1.0, +1.0), p 1.0 |
| **Linked pair (Quill answers)** | **78.7%** | |

Quill answers the same with the link cut: it does not use what Ember sends. Ember, reading Quill, scores 78.5% against
73.5% with the link cut, and its right and wrong answers match Quill's on 95% of questions (81% with the link cut). With
the partner of another question in the link, Ember falls to 48.4%; with random states in the link, to 73.6%, its own
level. In text the second speaker gives the first speaker's letter on all 3,165 questions. Training on the gold answer
as the partner's turn rewards copying a better partner; nothing in it rewards pooling two partners' evidence.

### Runs (`chains/`, `results/`)

Each run's chain script is in `chains/` (one pod each, called as `bash chains/nlN.sh OUT_DIR CODE_TGZ REPLIES ...`), and its
training log (`log.json`), held-out tables at every evaluation step (`payload_stepN.json`) and final evaluations are in
`results/magic-nlN/`.

| Run | What it tests | Outcome |
|---|---|---|
| nl11 | Fitted Q and K only, then the full mix | First held-out transfer (words 15/16 at step 200); lost once the mix began and the gates closed |
| nl12 | Random start, full mix | No transfer; gates closed |
| nl13 | Random start, word relay only | Opens later (words 16/16 by step 200); name–city pairs, never trained, 20/32 |
| nl14 | Fitted Q, K, V and O, word relay only | Words 16/16 and phrases 15–16/16 from step 100; name–city pairs, never trained, 29–30/32; lookups and numbers not |
| nl15 | Fitted start, gates frozen, full mix | At step 1000 every value game and direction passes but one (digits to AFM, 4 of 4); smaller gaps than nl16 |
| nl16 | Warm-up on all seven value games only | 178 of 180 held-out episodes; passes under dialogue, silent rounds and the shared pen without training on them (`turnbench.txt`) |
| nl18 | Full recipe with play games, silent rounds and micro-batches | Shared reading positive in 32 of 32 game–receiver groups at step 1200 |
| nl19 | nl18 with distillation and the shared pen | Stopped at step 500; shared reading as nl18 |
| nl20 | 900-step warm-up with held-out numbers, then the full mix | Warm-up: 179 of 180 (`results/magic-nl20/results/payload_warm.json`); full mix step 800: 180 of 180, shared reading 407 of 456 (`payload_mix_step800.json`); final, step 1,320: 180 of 180, shared reading 410 of 456 (`payload_final.json`) |
| nl21 | Lookup, digit strings, three shared-reading tasks and the rhyme game never trained | Final (step 759): every never-trained game passes in both directions at p < 0.01, 207 of 240 receiver episodes (`results/magic-nl21/results/payload_final.json`). Warm-up (step 600): digit strings 10/10 both ways, lookups 9/10 into Qwen and 8/10 into AFM |
| nl22 | nl20 step 1,320 fine-tuned with plain chats whose turns the original models wrote while seeing the partner's private city (`prep_talk.py --chat`, `train.py --chat`) | Stopped. The twins still looped word for word with the link on or cut and opened by naming their own city |
| nl22c | nl20 step 1,320 fine-tuned 157 steps on chats from the privileged teacher (below), with the mix kept | Free chats, 20 with cities never seen in training: partner's city said first in 13 of 23 chat sides with the link on, 1 of 23 cut, 0 of 23 linked to a stranger, and the stranger's own city in 14 of 36; turns repeating an earlier turn fell from 17% to 1% (link on) and 31% to 3% (cut); value games 360 of 360 and shared reading unchanged (`results/magic-nl22c/`) |
| nl23 | Exam questions: linked pair, each model alone and a text pair, 300 steps each from nl20 step 1,320 (on nl20's pod) | Linked pair 78.7% of 3,165 test questions, Quill alone 79.1% (p 0.42), both averaged 81.1% (p 5e-5); Ember takes Quill's answer through the link (`results/magic-nl23/bench.json`) |

### Earlier designs

1. **Additive residual write.** A send head wrote a 256-number message and a gate into the partner's residual stream
   after block 11. The value was decodable from the message (128/128 episodes, chance 2%), but the receiver never
   learned to use it, and the gates closed once sending had a cost.
2. **Soft-token slot.** Each message became an extra input position. The only slice that passed information was the
   sender's next-token distribution re-embedded as text (adding two numbers, Qwen to AFM); the learned part passed
   nothing. Per-slice results are in `results/probe.json`.

### Exactness checks (`tests/`)

On tiny random models of both architectures (CPU, float32):

| Check | Max difference |
|---|---|
| Joint teacher-forced pass vs tick-by-tick generation (last hidden state, relative; duplex, dialogue, shuffled partners, link closed after turn 1) | 4.9e-7 to 5.8e-7 |
| Gradients with vs without gradient checkpointing | 0 |
| Gradients of one batch vs length-grouped micro-batches vs two ranks | 2.2e-7 |
| Logits with untrained links (write size 0) vs the plain models | 0 (bit for bit) |
| Recorded attention (explicit softmax) vs fused attention | 1.6e-6 |


### Running

```
pip install torch transformers peft datasets safetensors scipy matplotlib
cd pairing
python prep.py --out OUT                                         # prompts and teacher-forced replies
python build_coread.py                                           # public texts both models read -> data/coread.json
python prep_talk.py --play --turns 6 --n_train 1600 --n_test 300  # play conversations -> data/play.json
python prep_coread.py --replies OUT/replies.json                 # distillation replies, added to replies.json
python train.py --replies OUT/replies.json --out OUT/run --content_init qkvo --read_start 0.1 --gate_norm --hold 100000 \
    --payload_w 3 --relay_first 900 --relay_games relay_word,relay_number,relay_phrase,relay_digits,merge_sum,merge_fact,lookup \
    --steps 3000 --coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --pause 4 --pause_p 0.5 --comp 0 --micro 4 \
    --ckpt none --payload_eval 16
python payload_ctl.py --run OUT/run/step_N --replies OUT/replies.json --out OUT/payload.json --n 20 --coread
python turnbench.py --run OUT/run/step_N --replies OUT/replies.json --out OUT/turnbench.json --n 128
cd .. && python tests/test_link.py && python tests/test_align.py && python tests/test_coldstart.py
```

</details>
---

Claude Opus 5.5
