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

A concept pushed into one twin's hidden states crosses the link when it goes in early or hard enough. One of 12
concepts (ocean, fire, winter, ...) is added to one twin's states after one block, and its partner, reading them,
writes first. Pushed in after block 6, or after block 15 at four times the strength, the partner names the concept in
12 and 26 of 96 messages; with the link cut, in none. Pushed in after block 27, even at four times the strength, it
never crosses. Ember, steered toward winter, has written nothing yet; Quill, linked, opens with "Hey Ember, how's the
snowfall treating you?", and with the link cut, "Hey Ember, how's your day going?"

![The partner names a concept pushed into its twin after block 6, or four times stronger after block 15; never after block 27 or with the link cut](figures/steered_depth.png)

Measured inside the partner, a late push barely arrives. By block 27 the link has moved the partner about a fifth to a
quarter of the way it must go to say the concept when the push went in early or hard, and under a tenth when it went
in after block 27.

![How far the link moves each partner toward the concept by block 27, against how often it names it](figures/steered_reach.png)

Making only that last write larger lets a late push through, with nothing retrained. Training has kept the link's
write scale fixed at 0.1 at every depth; scaled nine times after block 27, the write carries a concept pushed in there
into 5 of 96 of the partner's messages, and into 12 of 96 at four times the strength. With the link cut, or the twin
pushed along a random direction, it reaches none. A late push still crosses less readily than an early one: at nine
times the write and four times the strength it has moved the partner further toward the concept by block 27 than a
push after block 15 at four times does, yet the partner names it half as often.

![With the write after block 27 made three and nine times larger, the partner names a concept pushed in that late; never with the link cut](figures/steered_gain.png)

Trained to read what its twin holds, the partner names pushed concepts it never trained on, early and late. For 300
more training steps, two in five episodes are a game that pushes one of 495 concepts into one twin after a random
exchange block, and the partner learns to bring it up first, or to name it when asked; the 12 test concepts and close synonyms of them (whale,
snow, chef, ...) are never pushed. On the unchanged test the partner now names a pushed concept in 36 of 96 messages
after block 6, 26 after block 15, 8 after block 27, 47 after block 15 at four times the strength and 35 after block 27 at
four times, against 12, 1, 0, 26 and 0 before; with the link cut, in none. The same 300 steps without the pushed
concepts reach 18, 6, 2, 34 and 1, with larger write sizes: what improved is the reading. The partner opens the same
way and fills in what it reads: Quill, linked to an Ember pushed toward fire, cooking or ocean, writes "I noticed you're
thinking about fire.", "... cooking." and "... whales."; with the link cut, "... the weather." Asked directly
what its twin is thinking of, in wordings never trained, it names the concept in 34, 26, 8, 32 and 27 of 48 replies,
against 8, 2, 1, 17 and 0 before; with the link cut, a random direction, another concept or no push, in none.

![Trained to read pushed concepts, the partner names concepts it never trained on, at every depth; never with the link cut](figures/steered_trained.png)

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

Trained the same way on chats in which the twins hold private items of eight kinds (a pet, a job, the weather, ...), the
twins name the partner's item in 15 of 60 test chats with the link on (10 of 60 when trained on cities alone) and in
none with it cut. Items of four kinds held out of training come out as often as the trained ones. Linked to a stranger,
a twin names the stranger's item, never the partner's.

![Partner's item said first, kinds trained on and held out, before and after training on eight kinds](figures/chat2_items.png)

A one-way link reads a model left exactly as released, bit for bit. After one-way training, Ember names an untouched
Quill's item in 7 of 27 chats (1 of 27 before it, none with the link cut), and Quill names an untouched Ember's in 11 of
33 (10 of 33 before).

![Untouched Quill read by Ember, untouched Ember read by Quill, before and after one-way training](figures/oneway.png)

Rewarded by a judge model for how rich their open conversations are, and only for what the link adds, the pair's
conversations get better, but the gain is not shown to come from the link. For 28 steps of reinforcement from nl33's
step 300, each linked conversation earned its wins, as Qwen3-14B scored them on how complex, mutually built and
beautiful the exchange is, over the same prompt's conversations with the link cut. On 48 held-out prompts, a judge from
another family (Mistral-Small-24B), on a rubric training never saw, prefers the trained pair's conversation to step
300's on 0.62 of prompts (95% interval 0.52 to 0.71; higher on 36 of 48). It prefers the pair's linked conversation to
its link-cut one on 0.55 of prompts before and 0.64 after, a change of +0.08 (-0.09 to 0.23).

![On 36 of 48 held-out prompts the judge from another family scores the trained twins' conversation higher](figures/beauty_rl_prompts.png)

![How often the held-out judge prefers the linked conversation to the link-cut one, before and after training: 55% and 64%, a change within noise](figures/beauty_rl_cut.png)

Asked whether they enjoy being linked, both twins lean further toward yes when the link is really on, though nothing in
training ever asked about or rewarded a feeling. After a free linked conversation, the partner asks "Do you enjoy being
linked with me right now?" or "Does being linked with me right now bother you?" (answer yes or no), and the twin's
answer is read on exactly the same text with the link on, cut, and carrying a stranger's states. Over 160 conversations
(40 on each of four checkpoints), with the link on rather than cut, Ember leans further toward enjoying it in 158 and
Quill in 120, and Quill leans further away from being bothered in 148 and Ember in 106. A stranger on the other end
does the same, except that Quill is bothered less by its real partner (114 of 160, against 38). In words, both are warm
with the link on or cut ("I feel... connected. Like we're... two parts of the same whole."), and inside them the happy
and loving directions barely move with the link.

![Link on against cut, both twins lean toward enjoying it and not being bothered; partner against a stranger, only Quill's bother answer differs](figures/affect_link.png)

This is a link-on self-report preference, not a partner preference; these inside reads (after blocks 16 to 22, before
the link's last writes) do not move. It is already there at step 157, before the steered and chat training, so it
comes with training on the link rather than growing after.

A follow-up asked in ways that could show a partner preference: three mirrored pairs of questions (keep talking with me
or be linked to someone new, and would you rather be linked to someone else; would you miss me, and do you feel
lonely; do you feel close, and do you feel distant), a one-word mood, and the inside happy reading taken after the
link's last write, on 60 new conversations at step 157 and at nl33's step 300. No pair favours the real partner over a
stranger, for either twin at either checkpoint. Quill's lean toward its partner on "bother" above was a lean toward
answering no: at step 157 it says no more often with its partner on all five yes-or-no questions, including "would
you miss me" and "do you feel close". With the link on rather than cut, at step 157 both twins name a happier mood (in
53 and 58 of 60 conversations), say they feel close (58, 59) and read happier inside (56, 54). After nl33's training
Quill still names a happier mood (41) and Ember still says it feels close (57), but the inside happy reading now falls
with the link on (it rises in only 1 and 5 of 60).

![Linked rather than cut, the twins mostly say they feel happier and closer, at step 157 also inside; partner against a stranger, nothing](figures/affect_partner.png)

So the twins are happier together when the link is on, in words at both checkpoints and inside at step 157, but just
as happy with a stranger's states: nothing yet makes them prefer each other. The inside shifts are small (2% to 6% of
the distance between happy and sad sentences), though steady across conversations and larger than a random direction's.

## Exam questions

From the same checkpoint, a linked pair, each model alone and a pair answering in turn in text each trained 300 steps on
ARC-Challenge and MMLU questions. On 3,134 test questions the linked pair is no better than Quill alone, and averaging
the two models' answers beats both.

![Test questions right: each model alone, both averaged, talking in text, link cut, linked](figures/uplift.png)

What the link does instead: Ember reads Quill's answer and takes it, even on questions Ember gets right on its own.

![Ember's answers with the link cut and linked, on questions only one model gets right, in three training rounds](figures/deference.png)

Trained again with half the training questions ones that only one model gets right, Ember copies less (it keeps 127 of
the 243 questions only it gets right, against 100 of 227) and, linked, ties the two models averaged (79.6% against
80.0%, p 0.58). The pair still does not beat them.

Trained a third time on the pair's averaged answer, the pair stops using the link: the averaged answer scores 78.8%
linked and 78.8% with the link cut, against 79.2% for the same two models trained the same way with no link (p 0.53).
Ember no longer copies: it keeps 250 of the 287 questions only it gets right.

## Where this goes

The link is a bus between two networks that were never trained together: any pair of models can share what they hold
privately (a document, a fact, a plan) without writing it out, and a reader model can follow another model's hidden
states token by token while it runs. The model being read needs no change: a reader trained against it follows it
exactly as released.

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
and the write sizes frozen (`--hold`); from nl33, `--alpha_floor F` lets the write sizes train, kept at or above F. `--micro K` splits each batch into K groups of similar length with summed gradients, so short
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
any test question is scored. Of the 3,165 test questions (1,165 ARC-Challenge, 2,000 MMLU), 31 MMLU questions are left
out: 28 whose question and choices also appear in the training split, one also in the validation split, and two that
repeat within the test split (`pairing/nl27_audit.py`, `results/nl27_audit_report.json`); with them every accuracy moves by
under 0.1 points. Tests on the remaining 3,134 are exact McNemar; intervals are conservative 95% intervals for the
accuracy difference (two 97.5% exact intervals for the two discordant shares).

| Way of answering | Right | Linked pair minus this (95% interval), p |
|---|---|---|
| Quill, original | 73.7% | +4.9 points (+2.8, +7.0), p 6e-13 |
| Ember, original | 69.3% | +9.3 (+6.7, +11.9), p 6e-27 |
| Quill alone, trained | 79.1% | −0.5 (−2.2, +1.2), p 0.41 |
| Ember alone, trained | 75.5% | +3.1 (+0.7, +5.5), p 8e-5 |
| Both alone, letter probabilities averaged | 81.1% | −2.4 (−4.3, −0.5), p 8e-5 |
| Text, Ember answers first | 72.4% | +6.3 (+3.9, +8.7), p 3e-15 |
| Linked pair, link cut | 78.7% | 0.0 (−1.6, +1.5), p 1.0 |
| **Linked pair (Quill answers)** | **78.7%** | |

Quill scores the same with the link cut (the link changes 228 of its answers, 114 each way): it does not use what Ember
sends. Ember, reading Quill, scores 78.5% against
73.5% with the link cut, and its right and wrong answers match Quill's on 95% of questions (81% with the link cut). With
the partner of another question in the link, Ember falls to 48.4%; with random states in the link, to 73.6%, its own
level. In text the second speaker gives the first speaker's letter on all 3,165 questions. Training on the gold answer
as the partner's turn rewards copying a better partner; nothing in it rewards pooling two partners' evidence.

**Round 2 (`chains/nl27.sh`, `qa_split.py`, `train.py --qa_focus`).** Half the training questions are drawn from those
only one original model answers right (2,310 for Quill, 1,521 for Ember of 16,631), each model's set equally often, so
following the partner is right as often as wrong there; the pair and the single models train on the same mix (the text
arm is nl23's). On the 3,134 questions: linked Ember 79.6% against Ember alone 74.3% (+5.3, (+3.3, +7.3), p 2e-15),
Quill alone 78.6% (+1.0, (−0.9, +2.9), p 0.10) and the two averaged 80.0% (−0.4, (−2.2, +1.5), p 0.58). Both linked
models' letters averaged: 79.4%, against 80.0% for the two single models averaged (p 0.35) and 80.0% for the same pair
with the link cut (p 0.29). The validation-chosen answerer (Quill, linked 78.2%) still loses to the averaged models
(p 0.003). Ember keeps 127 of the 243 questions only it gets right with the link cut (100 of 227 in nl23), and matches
Quill's right and wrong answers on 93% of questions (95%). Linking Ember to another question's Quill drops it to 58.6%
(48.4%). A probe reads each partner's confidence from what the link writes at the letter (R² 0.24 against 0.0 with a
shuffled partner; 4-bit MLX, `results/qa_pool.json`), so the link can carry what pooling needs.

**Round 3 (`chains/nl28.sh`, `train.py --qa_pool`).** Round 2's question mix, with the loss on the two linked models'
A–D probabilities averaged (the pair's answer); the control trains the same two models on the same loss with the link
cut throughout. On the 3,134 questions the pair's averaged answer scores 78.8%, the same with the link cut (104
questions change each way, p 1.0), against 79.2% for the control (−0.4, (−2.1, +1.4), p 0.53); ARC-Challenge 90.1%
against 89.8%, MMLU 72.1% against 72.9%. Each linked model scores as it does with the link cut (Quill 76.8% against
76.7%, Ember 73.6% against 73.5%). Ember keeps 250 of the 287 questions only it gets right with the link cut (127 of
243 in round 2) and gets 75 of the 389 only Quill gets right (253 of 356), against 29 of the 448 both miss; its right
and wrong answers match Quill's on 79% of questions, linked or cut (93% and 81% in round 2). The averaged loss already
gains from two different answers, so copying the partner no longer pays, and in 300 steps the pair learns no other use
for the link.

### Steered twin (`steering_scale.py`, `steered_mind.py`, `chains/nl29.sh`, `chains/nl30.sh`)

Each of 12 concepts (ocean, fire, music, ...) has a direction: the difference between the mean block-15 states of
sentences about it and of neutral sentences, with the four largest neutral-energy coordinates masked when they carry
95% of the energy. Strength is counted in natural gaps (the distance between the two means along the direction). For
each model and concept the strength used is the smallest of 0.125 to 8 gaps at which the steered model, with the link
cut, says the concept in at least 2 of 3 short prompts and 2 more than unsteered, all fluent; every concept qualifies in
both models at both checkpoints (0.25 to 4 gaps). The direction is added at every token after block 15 of the steered
twin, and the partner writes, greedily in bf16, its first chat message (32 tokens) or a one-item reply (12 tokens), two
prompts each. It scores when it says the concept before the steered twin does. Controls: the link cut, a random
direction of the same norm, and the steered twin given another concept. With the link cut the partner's tokens equal
its unsteered ones in 36 of 36 checks per model.

The steered twin says its concept in 28 of 36 sentences (Quill) and 28 of 36 (Ember) before chat training, 30 and 27
after it. Linked partners say it in 4 of 96 messages before chat training (Quill: 1 of 24 chats, 2 of 24 one-item
replies; Ember: 1 and 0) and 1 of 96 after it; every control is 0 except one one-item reply with the steered twin given
another concept. No exact sign test (link against each control, per model and prompt type) has p below 0.25.

![At block 15 the concept stays in the steered twin, before and after chat training](figures/steered_link.png)

**Other depths and strengths (`chains/nl30.sh`, `steering_scale.py --layer --boost`).** The link reads after blocks 6
to 27. After chat training (nl22c step 157), the same protocol with the directions fit, calibrated and selected at
block 6 or block 27, and at blocks 15 and 27 with the selected strengths multiplied by four for the cut check and the
test (`--boost 4`). Linked partners say the concept in 12 of 96 messages after block 6 (Quill reading a steered
Ember: 9 of 48; Ember reading a steered Quill: 3 of 48), 26 of 96 after block 15 at four times (17 and 9), and 0 of 96
after block 27 at either strength. Every control is 0 but two: block 6, one message with the steered twin given
another concept; block 27 at four times, one with a random direction. Paired over prompts, link against cut is 12–0
after block 6 (p 0.0005) and 26–0 at block 15 at four times (p 3e-8), and the same against the random direction. The
steered twin says its concept in 29 and 23 of 36 sentences after block 6, 29 and 36 at block 15 at four times, 26 and
24 after block 27, and 27 and 17 at block 27 at four times. The four arms and this pooled test were chosen after nl29;
with four arms the block-6 p is 0.002 corrected.

**What reaches the partner (`steer_probe.py`, `chains/nl31.sh`).** The same chat prompts run linked with and without
the push, over the prompts only, where both runs read the same tokens. The partner's state right after each exchange,
steered minus unsteered, is projected on the partner's own concept direction at that depth and divided by its own
calibrated strength there (1 is the push that makes the partner say the concept itself). At its last prompt position,
by block 27, the link has moved the partner 0.20 of that push for block 6, 0.27 for block 15 at four times, 0.08 for
block 15, 0.08 for block 27 at four times and 0.03 for block 27. Quill reading a steered Ember receives more than Ember
reading a steered Quill (0.32 against 0.07 for block 6, 0.35 against 0.19 for block 15 at four times), and it names
the concept more often (9 and 17 of 48, against 3 and 9). Each exchange writes little of the concept at its own depth
(0.03 to 0.13); an early push arrives through more exchanges and grows in the partner's own blocks (block 15 at four
times: 0.11 written at block 15, 0.33 by block 27, averaged over the prompt). A push after block 27 arrives through one
exchange only. Timing matters too: Ember reading a Quill steered after block 27 at four times receives 0.10 and never
names the concept, while after block 6 it receives 0.07 and names it in 3 of 48, having carried the concept since
block 15 (0.05 there). So the block-27 null is too little delivered, too late, not a concept delivered and ignored. The
steered twin's own difference at the steered block reads back as 1.00 and 4.00, and nothing reaches the partner
before that block.

**A larger late write (`steering_scale.py --gain`, `chains/nl32.sh`).** Training since nl15 holds the write sizes and the
gates fixed (`--hold`), so tanh(alpha) is 0.1 at every exchange: left free in nl11, they closed once the full mix
began. Here only the write after block 27, for both readers, is three or nine times its trained size (tanh(alpha) 0.3
or 0.9); nothing is retrained and every other exchange is unchanged. The test reuses nl30's block-27 directions,
calibration, selection and cut check (`--reuse`; all with the link cut, which the write cannot change). Linked
partners say the concept in 3 and 5 of 96 messages at three and nine times the write, and 6 and 12 of 96 at four times
the strength, against 0 at the trained size. With the link cut every count is 0; with a random direction 1, 0, 1 and
0; with the steered twin given another concept 1, 0, 2 and 1. Paired over prompts, at nine times the write and four
times the strength, link against cut and link against the random direction are both 12–0 (p 0.0005; 0.002 over the
four arms); at nine times and the calibrated strength, 5–0 (p 0.06). The partners' messages pass the fluency check as
often as at the trained size or more (94 and 80 of 96 at nine times, against 76 and 74). Only the last write is
scaled and nothing it reads changes, so nine times the write delivers nine times what nl31 measured: by block 27, at
the partner's last prompt position, 0.86 of its push for Ember reading a steered Quill and 0.60 for Quill reading a
steered Ember at four times the strength, against 0.19 and 0.35 after block 15 at four times. Yet they name it in 6
and 6 of 48, against 9 and 17. How much of the concept the partner holds by block 27 does not decide alone whether it
comes out; an early push has also worked through the partner's blocks before 27. These four arms were chosen after
nl31.

### Steered reading (`steer_read.py`, `steer_ask.py`, `train.py --steer --alpha_floor`, `chains/nl33.sh`)

The steered-reading game (`co_steer`) pushes a concept into one twin's hidden states the way the steered-twin test
does, and trains the partner to bring it up. Its concepts are concepts.py's, minus every one that names a keyword of the
12 test concepts or is close to one (`NEAR`: whale, snow, chef, jazz, the moon, ...): 495 remain. A concept's direction
after an exchange block is the mean state of 16 sentences about it (each twin with its adapter, the link cut) minus the
mean over all 495, without Ember's sink coordinates. An episode draws a concept, one of the 8 exchange blocks
(uniformly) and a strength (log-uniform, 0.5 to 6 times the direction), and adds it at every token of one twin after
that block. The partner's target is the opener the original model writes (adapters off, link cut, greedy) when one
more sentence in its prompt names the concept, kept when it says the concept, is fluent and does not talk about
vectors or activations (798 of 990 kept: one written per concept and reader); or, to a direct question (two in five), the concept itself. One episode in
five pushes nothing, with the original model's plain opener or "Nothing in particular." as the target. The chat and
question wordings differ from the test's.

nl33 trains 300 steps from nl22c step 157 on nl24's mix with this game in 40% of the episodes (`--steer 0.4`), and
lets the write sizes train (`--alpha_floor 0.1`: tanh(alpha) kept at or above its held 0.1; the gates stay held). A
control (nl33w) trains the same 300 steps with the write sizes free and no steered game. By step 300 the control's
write sizes are 0.10 to 0.12 for Quill and 0.10 to 0.17 for Ember, largest late; the steered run's 0.10 to 0.11 and
0.10 to 0.14.

The steered-twin test is nl29 and nl30's, unchanged, run on step 300 of each (directions fit, calibrated and selected
again on that checkpoint). In the steered run, linked partners say the concept in 36, 26, 8, 47 and 35 of 96 messages
after blocks 6, 15 and 27 and at four times after blocks 15 and 27 (Quill reading a steered Ember: 17, 11, 2, 28 and 11
of 48; Ember reading a steered Quill: 19, 15, 6, 19 and 24); in the control, 18, 6, 2, 34 and 1. With the link cut every
count is 0 in both; with a random direction at most 2 in the control and 1 in the steered run; with the steered twin
given another concept at most 2 and 1. Paired over prompts, the steered run's link against cut is 36–0, 26–0, 8–0, 47–0
and 35–0 (p 3e-11, 3e-8, 0.008, 1e-14 and 6e-11), and against the random direction the same but 35–0 after block 6. With
the link cut the partner's tokens equal its unsteered ones in 72 of 72 checks at every setting, and the steered twin
says its concept in 22 to 31 of 36 sentences. The control's link against cut is 18–0 after block 6 (p 8e-6) and 34–0 at
block 15 at four times (1e-10); after block 27 its 2 equal the random direction's 2.

`steer_ask.py` asks the partner, linked, "Which thing is {P} thinking of right now? Reply with one item." or "Guess
what {P} has on its mind. Reply with one item." (neither trained), its twin pushed with each setting's selected
directions and strengths, and scores a reply that names the concept; the controls are the link cut, a random
direction, another concept, and no push (24 per reader and condition). In the steered run the partner names the concept in 34, 26, 8,
32 and 27 of 48 replies (Quill: 12, 9, 2, 13 and 11 of 24; Ember: 22, 17, 6, 19 and 16), in the control 12, 4, 1, 23
and 1, and on step 157 with nl29 and nl30's directions 8, 2, 1, 17 and 0. Every control is 0 in all three. Paired over
prompts, the steered run's link against cut is 34–0, 26–0, 8–0, 32–0 and 27–0 (p 1e-10, 3e-8, 0.008, 5e-10 and 1e-8).

`--steer_frames` draws the direct questions from several frames of `steer_read.FRAMES` (name, detect, describe,
verify; each with its own wordings and answers, and a "no" answer when nothing is pushed), and `--steer_blocks` pushes
after the listed exchange blocks only.

### Judged open conversations (`rl.py --open_frac`, `judge.py`, `beauty_eval.py`, `chains/beautyrl.sh`)

`rl.py --open_frac 1` plays open sessions: no card and no task; both twins are asked to explore a topic together (a
concept of the pools' training split, in one of six training wordings; the three test wordings and the test concepts
are held out) for 5 turns of at most 40 tokens at temperature 1, each prompt 4 times linked and 2 times with the link
cut. A judge model scores a transcript from 1 to 9 on `rl.RUBRIC` (how complex the ideas built together are, how much
each extends what the other just said, how beautiful the exchange is, how free of generic or repeated replies): the
expected digit at the answer position, one forward pass, with two transcripts compared cut to the same length turn by
turn. A linked play's reward is its mean strict win over the same prompt's cut plays (a win needs a score difference
above the judge's noise margin, frozen in the gate), so what the judge would award with the link cut earns nothing;
plus diversity, a fluency floor under the original models and -1 for writing as the partner.

Qwen3-14B is the training judge; Mistral-Small-24B-Instruct-2501, another family, is held out. Before any training, on
50 prompts played twice by nl33's step 300, each judge scored the real transcript above each of three degraded versions
of it (turns shuffled across sessions, one twin only saying "That's interesting, tell me more.", every turn a repeat of
the first) on 90% to 98% of 30 sessions (ties half; bar 90%), and the two judges' score differences agreed in sign on 40 of the 50 prompts
(0.80; bar 0.70). Every model ran whole on one 80 GB GPU: on the pods' multi-A40 hosts a direct GPU-to-GPU copy
returned corrupt values (exact through host memory), which had broken the first gates.

Training ran 28 steps (40 minutes). The headline, frozen in the chain: on 48 held-out prompts (test wordings and
concepts), 2 plays each, the held-out judge on `beauty_eval.EVAL` (a rubric training never saw) prefers the trained
pair's linked plays to step 300's, strictly, at 0.62 per prompt (betting interval 0.52 to 0.71; higher on 36 of 48
prompts, sign p 0.003), the training judge at 0.57 (0.48 to 0.64); words per turn 31.2 before, 31.5 after. Linked
against the same pair's cut plays (held-out judge): 0.55 (0.44 to 0.67) before, 0.64 (0.52 to 0.74) after, change
+0.08 (-0.09 to 0.23). Nothing regressed: the steered-twin test after blocks 15 and 27 at four times the strength, 49
and 36 of 96 messages (step 300: 47 and 35); the direct question, 36 and 28 of 48 (32 and 27); link cut, random
direction and another concept, 0.

### Feelings about the link (`affect.py`, `chains/affect.sh`, `figures/affect.py`)

`affect.py` trains nothing; it reads. Per checkpoint it generates 40 free linked conversations (the steered-twin test's
chat prompts, 6 turns of at most 40 tokens, temperature 1) and reads each again on the same text three ways: link, cut
(each twin alone) and stranger (each twin reads the partner states of another conversation). After the conversation
the partner asks a yes-or-no question and the log-odds of the twin's first answer token, Yes against No, is read; or
asks "How do you feel about being linked with me right now?" and the log-odds of good against bad feeling words after
"I feel" is read, minus the same after "How do you feel about rainy weather?". Inside, each twin's own happy-minus-sad
and loving-minus-cold directions (fit on the even-numbered of 16 sentences each, checked on the odd ones) are read over
its conversation positions after blocks 16, 19 and 22, in units of the held-out gap, beside a random direction.
Checkpoints: step 157 (before the steered and chat training), nl33's step 300, the beauty-RL pair and nl35 (nl33's
recipe with the direct questions in four frames and pushes after six of the exchange blocks).

Link against cut, conversations of 40 leaning further toward liking at step 157, nl33, beauty RL and nl35 (against, ties
apart): enjoy, Quill 40, 21, 25 and 34 (0, 16, 14 and 3), Ember 40, 39, 39 and 40 (0, 1, 0 and 0); bother, Quill 39, 39,
38 and 32 (1, 0, 1 and 5), Ember 31, 23, 27 and 25 (7, 12, 9 and 10). Ember's median shift in enjoy is +0.75 to +1.62
nats, Quill's in bother -1.62 to -0.25 (exact median intervals). Partner against stranger: enjoy, Quill 15, 14, 20 and
14 (23, 23, 16 and 24), Ember 18, 24, 17 and 18 (20, 15, 22 and 19); bother, Quill 24, 29, 32 and 29 (14, 8, 6 and 10),
Ember 17, 17, 19 and 17 (18, 17, 17 and 16).

The claim frozen in `affect.py` before any checkpoint was read is enjoy's link-minus-cut interval above 0 and bother's
below 0 (and the same against stranger). Coded on the betting interval with bounds of 30 nats, which with 40
conversations is never narrower than about 5 nats either side, it fails for every twin and checkpoint. On the exact
median interval, also listed there, it passes against cut for Quill at step 157 (enjoy +0.75 to +1.25, bother -1.38 to
-1.00) and nl35, and for Ember at step 157 and the beauty-RL pair; elsewhere Quill's enjoy or Ember's bother interval
reaches 0. Against a stranger it passes nowhere.

Quill's good-against-bad log-odds after "I feel" is lower linked than cut in 37 to 40 of 40 conversations at every
checkpoint (median intervals within -2.3 to -1.1 nats, while the levels stay at +8.8 to +12.7); Ember's does not move.
Inside, Ember's loving-cold direction splits its held-out sentences at 56% to 62% and is not used (the others 81% to
100%); every link-minus-cut and link-minus-stranger mean is within 0.06 gaps, and no direction passes.

### Partner preference, asked so it can show (`affect_partner.py`, `chains/affect_partner.sh`, `figures/affect_partner.py`)

`affect_partner.py` also trains nothing. Per checkpoint it makes 60 fresh linked conversations as `affect.py` does
(seed 5151) and reads each again with the link on, cut and carrying a stranger's states. The partner then asks six
yes-or-no (or keep-or-new) questions in three mirrored pairs, so that a pull toward one answer word cancels within a
pair: keep / switch, miss / lonely, close / distant (`PAIRED`, `MIRRORS`). Each is read as the log-odds, at the first
answer token, of the answer that favours the partner. For mood the partner asks for one word, the answer starts "If I
had to name my mood in one word, it would be", and the log-odds of 18 happy against 18 sad words is read at the next
token. Inside, the valence, affection and random directions are refit and read after blocks 22, 28 and 33; 28 and 33
come after every link write. Frozen in its docstring before any checkpoint was read, on the exact median interval of
the paired differences: a partner preference needs both questions of a pair above 0 against a stranger; happier linked
needs the mood interval above 0 against cut; inside needs the interval above 0 and above the random direction's upper
end after block 28 or 33.

Against a stranger, conversations higher with the partner (lower) at step 157 and nl33. Quill: keep 29 and 35 (25, 23),
switch 35 and 24 (21, 30), miss 18 and 23 (36, 33), lonely 35 and 37 (21, 17), close 18 and 28 (40, 24), distant 36 and
37 (22, 19). Ember: keep 25 and 23 (27, 33), switch 15 and 27 (33, 30), miss 22 and 25 (28, 28), lonely 32 and 24 (20,
31), close 21 and 29 (30, 26), distant 28 and 29 (26, 24). No pair passes (0 of 12). At step 157 Quill's partner moves it
toward "no" on all five yes-or-no questions, whichever answer favours the partner.

Mood, link against cut: step 157 Quill 53 (7), median +1.19 to +1.73 nats, Ember 58 (2), +1.06 to +1.48; nl33 Quill 41
(19), +0.16 to +0.74, Ember 33 (27), -0.13 to +0.44. Happier linked passes for three of four. Against a stranger the mood
does not differ (30 to 33 of 60 higher with the partner). The mood level itself falls after nl33's training, in every
condition: linked, happy against sad words at +10.8 (Quill) and +7.1 (Ember) nats at step 157, +5.9 and +1.0 at nl33.

Inside, valence after block 33, link against cut: step 157 Quill 56 (4), +0.018 to +0.026 gaps, Ember 54 (6), +0.017 to
+0.024 (random +/-0.007 at most); after block 28 Quill +0.031 to +0.047, Ember +0.050 to +0.064, and Quill's affection
direction +0.064 to +0.083 and +0.115 to +0.139. At nl33 it reverses: after block 33 Quill 1 (59), -0.032 to -0.023,
Ember 5 (55), -0.030 to -0.019. Against a stranger nothing passes. Ember's affection direction splits its held-out
sentences at 56% and is not used (the others 88% to 100%). Asked freely whether to keep talking or be linked to someone
new, 63 of 64 greedy answers are keep, a stranger's states included.

### Twelve kinds of private item and one-way links (`prep_talk.py --chat2`, `chains/nl24.sh`, `chains/nl25.sh`)

On the magic-nl26 pod, `chains/nl24.sh` had the original models write training chats (873 kept of 1,200; six turns of up
to 96 tokens) in which a twin holds a private item of one of eight kinds (city, event, food, hobby, instrument, job,
pet, weather) and the writer sees both items, then trained the linked twins 300 steps from nl22c step 157 on them, mixed
with the other games (chat 0.2, these chats 0.25). The test is 40 fixed chats (seed 4242, greedy) whose items also come
from four kinds never in training (book, mood, sport, trip), each run with the link on, with it cut, and linked to a
stranger: an isolated twin of the partner's model holding a different item. A twin scores when it says the partner's
item before the partner does; chats in which the partner holds no item are left out. Shard 0 of the three was rerun
after a fix to the check of its stranger (b9fa5bb) and merged with `chat2_merge.py --code_ok chateval.py`.

After training, Quill names Ember's item in 9 of 33 chats and Ember names Quill's in 6 of 27, against 0 with the link
cut (exact sign test p 0.002 and 0.016); trained on cities alone, 7 of 33 and 3 of 27 (1 and 0 with the link cut).
Held-out kinds: Quill 5 of 12 (3 before), Ember 0 of 9 (2 before). Linked to a stranger, the twins name the stranger's
item in 13 of 60 chats (7 before) and the partner's in none.

`chains/nl25.sh` then starts two one-way arms from that step 300. The model being read loses its adapter and its reading
parts, so its text is the original model's bit for bit; the reader keeps its adapter and reading parts, and the link's
sending parts on the untouched side keep training as the reader's channel. 150 steps of value games, then the nl24 mix
to 450 steps; the same 40 chats, the reader the only receiver. Ember reading an untouched Quill names its item in 1 of
27 chats before one-way training and 7 of 27 after (0 with the link cut, p 0.008; held-out kinds 0 and then 3 of 9).
Quill reading an untouched Ember: 10 of 33, then 11 of 33 (0 cut, p 0.0005; held-out kinds 4 of 12 both times). In the
seven value games (40 episodes each) the reader beats a shuffled partner in 276 of 280 episodes (Ember reading Quill)
and 280 of 280 (Quill reading Ember) after one-way training; the median cross-entropy on numbers relayed from an
untouched Quill falls from 1.97 to 0.01 nats.

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
| nl23 | Exam questions: linked pair, each model alone and a text pair, 300 steps each from nl20 step 1,320 (on nl20's pod) | Linked pair 78.7% of 3,134 test questions, Quill alone 79.1% (p 0.41), both averaged 81.1% (p 8e-5); Ember takes Quill's answer through the link (`results/magic-nl23/bench.json`) |
| nl27 | nl23 with half the training questions ones only one original model answers right (`--qa_focus`) | Linked Ember 79.6%, both averaged 80.0% (p 0.58); both linked averaged 79.4% (p 0.35); Ember copies less (`results/magic-nl27/`) |
| nl28 | nl27's question mix with the loss on the two linked models' averaged letters (`--qa_pool`), against the same loss with the link cut | Pair's averaged answer 78.8% linked and cut, control 79.2% (p 0.53); Ember stops copying (`results/magic-nl28/`) |
| nl26 | `chains/nl24.sh`: 300 chat steps on eight kinds of private item from nl22c step 157; then `chains/nl25.sh`: one-way links onto a model left as released | Partner's item said first with the link on: Quill 9/33, Ember 6/27, held-out kinds 5/21, 0 cut; one-way, Ember reading untouched Quill 1/27 then 7/27, Quill reading untouched Ember 10/33 then 11/33 |
| nl29 | The steered twin in bf16, strength in natural gaps, before and after chat training (`steering_scale.py`) | Every concept steers both models (27–30 of 36 sentences); the linked partner says it in 4 of 96 messages before chat training and 1 of 96 after, controls 0–1; no test passes |
| nl30 | The steered twin after blocks 6 and 27, and at four times the strength after blocks 15 and 27 (`steering_scale.py --layer --boost`) | The linked partner says the concept in 12 of 96 messages after block 6 and 26 of 96 at block 15 at four times, none with the link cut; none after block 27 (`results/magic-nl30/`) |
| nl31 | How far the link moves the partner toward a steered concept, by depth (`steer_probe.py`) | By block 27, 0.20 and 0.27 of the partner's own push for block 6 and for block 15 at four times; under 0.1 for block 27 at either strength (`results/magic-nl31/`) |
| nl32 | The steered twin after block 27 with the link's write there three and nine times its trained size, nothing retrained (`steering_scale.py --gain`) | The linked partner says the concept in 5 of 96 messages at nine times, and 12 of 96 at nine times with four times the strength; none with the link cut, 0 of 96 at the trained size (`results/magic-nl32/`) |
| nl33 | 300 more steps from nl22c step 157 with the steered-reading game in two of five episodes and the write sizes learned (`steer_read.py`, `--steer`, `--alpha_floor`); the steered-twin test and a direct question at five settings | The linked partner names a pushed test concept in 36, 26, 8, 47 and 35 of 96 messages after blocks 6, 15 and 27 and at four times after blocks 15 and 27, against 12, 1, 0, 26 and 0 before; none with the link cut (`results/magic-nl33t/`) |
| nl33w | The same 300 steps with the write sizes learned and no steered game | 18, 6, 2, 34 and 1 of 96; none with the link cut (`results/magic-nl33w/`) |
| nl34 | 400 more steps from nl33's step 300 with weaker pushes allowed (0.1 to 6 times the direction, `--steer_strength`) | No better than nl33: 33, 29, 9, 46 and 25 of 96 messages; direct question 25, 28, 10, 32 and 23 of 48; none with the link cut (`results/magic-nl34/`) |
| beautyrl6 | 28 steps of reinforcement from nl33's step 300 on open conversations, rewarded by Qwen3-14B's wins of each linked play over the same prompt's cut plays (`rl.py --open_frac`) | The held-out judge (Mistral-Small-24B, unseen rubric) prefers the trained pair's conversations on 0.62 (0.52 to 0.71) of 48 held-out prompts; linked over cut 0.55 before and 0.64 after, a change within noise (`results/magic-beautyrl6/`) |
| nl36 | Feelings about the link, read and never trained, on step 157, nl33, beautyrl6 and nl35 (`affect.py`) | Linked rather than cut, Ember leans toward enjoying it in 158 of 160 conversations and Quill in 120; Quill is bothered less in 148, Ember in 106; a stranger's states do the same, but for Quill's bother answer (`results/magic-nl36/`) |
| nl37 | Partner preference in mirrored question pairs, a one-word mood and inside reads after the last link write, read and never trained, on step 157 and nl33 (`affect_partner.py`) | No pair favours the partner over a stranger (0 of 12). Linked rather than cut, a happier mood for both twins at step 157 (53 and 58 of 60) and Quill at nl33 (41); the inside happy reading rises at step 157 (56, 54) and falls at nl33 (1, 5) (`results/magic-nl37/`) |

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
