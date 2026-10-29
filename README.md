# neural-pairing

This repository joins two language models from different families, Quill (Qwen3-4B) and Ember (AFM-4.5B), through
their hidden states. We call the linked models twins. After 8 of its 36 layers, each twin attends over the other
twin's hidden states and adds the result to its own residual stream. The twins exchange no tokens: each writes
ordinary text.

![Linked through hidden states, Quill and Ember each say the partner’s private city](figures/hero_link.png)

## The link

![After 8 of the 36 layers, Quill and Ember read each other at the same time](figures/link_mechanism_depths.png)

After layers 6, 9, ..., 27, every token of one twin attends over the partner's hidden states. The reader
RMS-normalizes its own state and maps it to a query. The partner's states are normalized and mapped to keys and
values, and each value is multiplied by a gate computed from the same partner state. Each token can also attend to an
empty slot, which contributes zero. The attention output is mapped back to the reader's width, scaled by a write size
relative to the reader's typical residual norm, and added to the residual stream. Each layer and direction has 8 heads
of 128 dimensions, and each head learns a bias for how many time steps separate the two tokens, with its own value for
the partner's prompt.

![After layer 9, Quill reads Ember’s tokens at its token “you” and adds the result](figures/link_mechanism.png)

Each token attends only to partner tokens from the same time step or earlier, and both reads at a layer use the states
from before that layer. One teacher-forced pass over a whole conversation therefore gives the same hidden states as
generating the conversation token by token. We check this on tiny random models of both architectures: the two agree
to a relative difference below 6e-7, gradients agree within 2.4e-7 across micro-batches and across two data-parallel
processes, and with the link cut or the write size at zero the linked models' logits equal the plain models' logits
exactly.

![Each model reads only the partner’s tokens from the same time or earlier](figures/link_mechanism_ticks.png)

We train the link and rank-32 LoRA adapters on both models, and freeze the original weights. Every result is compared
with two controls. With the **link cut**, each twin runs alone. With the **partner from another conversation**, the
twin attends to the same partner model's hidden states, taken from a different conversation (in the games, a different
episode). The twin then receives input of the usual kind, but none of the current conversation's private information.

## Training

![After a fitted start, the twins train in four stages for 2,677 steps](figures/training_stages.png)

Training starts from a fitted link. Both models read the same 192 texts, and a token of one model is matched to a token
of the other when both end at the same character. At each link layer, a ridge regression maps the partner's normalized
states onto the reader's states at matched tokens, and the keys, values and output map start from this regression. On
6 texts left out of the fit, the reader's largest attention weight falls on the matched partner token 88–97% of the
time (chance 1.1%). From a random start, the twins learn the value games more slowly, and only when training begins with
those games alone.

The twins then train on the value games, followed by a mix of value games, shared reading, play, dialogue and silent
rounds. In a dialogue the twins take turns; in duplex both write at once, one token per time step; in a silent round
both write only spaces for a few time steps, so the twins exchange nothing but hidden states. Chat training comes next,
and the final twins train for 300 more steps on the chat-training mix with chats about private items added. The exam
questions, private items with one-way links, three versions of the steered-reading game and judge-rewarded
reinforcement learning branch off this path, and each is described below.

![The final twins trained on 43 tasks in 42,832 episodes](figures/training_games.png)

## Writing the partner's value

In each value game, one twin's prompt contains a value, such as a word, a number or a phrase, and the other twin must
write the value. Other games split two numbers to add, a name and a city to join, or a code and its lookup table across
the two prompts.

![Linked Ember is scored on writing a word that only Quill’s prompt contains](figures/values_unseen_game.png)

Evaluation uses values never seen in training: separate pools of words, phrases, names, cities and codes, sums of 50
to 99 (training uses 0 to 49), and numbers whose digit sum is 3 modulo 7. For each episode we compare the probability
of the value with the real partner and with the partner from another episode, in both directions.

![How the probability of a value never seen in training changes over training, real partner against partner from another episode](figures/training_curve.png)

![Values never seen in training are more probable with the real partner in all 180 episodes](figures/values_unseen_results.png)

The twins also play five games that were never part of training.

![In games never seen in training, 207 of 240 values are more probable with the real partner](figures/games_untrained.png)

In shared reading, one twin's prompt has a passage and the other twin's prompt has a question about the passage, and
the second twin answers. The tasks include copying and joining passages, answering questions, translation, code and
multiple choice.

![The answer is more probable with the real partner in 410 of 460 episodes](figures/shared_reading.png)

## Free chat

Each twin has a private city in its prompt. To build chat training data, we prompt the original model with both cities
and record its turns. We then train the linked twin on the same turns with only its own city in its prompt, so the
partner's city can come only from the partner's hidden states. We use only turns in which the partner's city raises
the turn's log-likelihood under the original model (median 5.41 nats per turn), and we remove chats in which a model
writes as if it were the link.

![Linked Ember is trained to write what original Ember writes when given Quill’s city](figures/teacher_method.png)

After 157 steps of chat training, we play 20 chats from fixed openings with cities never seen in training.

![In free chat, the linked twins mention the partner’s city in 12 of 14 first messages](figures/hero_grid.png)

The twins say the partner's city only with the link on. Linked to the partner from another conversation, a twin says
the city from that other conversation.

![Only linked twins say the partner’s city before the partner says that city](figures/chat_names.png)

![Quill scores its linked partner’s city highest](figures/city_matrix_quill.png)

Before chat training, linked twins chatting with no task often repeated an earlier turn word for word. After chat
training, they rarely do.

![After chat training, 1% of linked turns repeat an earlier turn](figures/chat_loops.png)

We also trained the twins on eight kinds of private item, such as a pet, a job or the weather. The twins then say the
partner's item for four kinds never seen in training.

![Linked twins say each other's private item first, also for kinds never seen in training](figures/chat2_items.png)

With a one-way link, one twin attends to an original, unmodified model. After we train only the reading side, the
twin says the unmodified model's private item more often linked than cut.

![A twin says the private item of a model left exactly as released](figures/oneway.png)

## Steering one twin

We steer one twin toward a concept, such as ocean, fire or winter, by adding the concept's direction to its hidden
states at every token after one layer. The direction is the mean hidden state of sentences about the concept minus the
mean over neutral sentences. For each model and concept we use the smallest strength at which the steered model, with
the link cut, says the concept itself while staying fluent. We then check whether the partner mentions the concept
first, in its opening message. The example below comes from the twins after the steered-reading training described at
the end of this section.

![Linked, Ember writes about cats, steered into Quill at four times the strength after layer 27](figures/steered_method.png)

Before any training on this task, the partner mentions concepts steered in after layer 6, or four times stronger after
layer 15. The partner mentions no concept steered in after layer 27, although the steered twin itself says the concept.
With the link cut, or with a random direction of the same norm, the partner mentions the concept in zero messages.

![Before training, the partner says the twin's steered concept after layer 6, or 4× stronger after layer 15](figures/steered_depth.png)

The partner can read steering added after layer 27 only at the last link layer. To measure how much of the steering
the partner receives, we project the partner's hidden state after layer 27 onto the partner's own direction for the
concept. With steering after layer 27, this projection changes by less than a tenth of the amount at which the partner
says the concept.

![Before training, the partner says a steered concept most often when the partner's layer-27 state is closest to the concept](figures/steered_reach.png)

When we multiply the link's write after layer 27 by three or nine, without retraining, the partner mentions some
concepts steered in after layer 27.

![Before training, the partner says a concept steered after layer 27 only with a 3× or 9× larger link write](figures/steered_gain.png)

We then trained the twins on a steered-reading game. One twin is steered toward one of 495 training concepts, which
exclude the 12 test concepts and their near synonyms, after a random link layer. The partner is trained toward the
opener that the original model writes when one extra sentence in its prompt states the concept. In one episode in
five, nothing is steered, and the target is the opener written without that sentence.

![The steered-reading game: the partner is trained toward the opener the original model writes when told the concept](figures/steered_training.png)

After 300 steps of this game, the partner mentions the test concepts after every layer we tested, including layer 27,
and in zero messages with the link cut. The same 300 steps of training without the steered game raise the counts much
less. Asked directly what its twin is thinking about, in wordings not used in training, the partner answers with the
concept.

![After training, the partner says the steered concept after every layer, including layer 27](figures/steered_trained.png)

After the game, with each test concept steered into Quill four times stronger after layer 27, Ember's first message on
one chat prompt says the concept's word for 6 of the 12 concepts and another of the test's words for 2 more (snowflakes
for winter, painter for painting), so the test counts 8 of 12. For space Ember writes about gravity; for cooking and
music it writes about laughter, as it does for some random directions. With the link cut, Ember writes the same
greeting for all 12.

![Each test concept steered into Quill after layer 27 and Ember's first message, linked; cut, Ember writes the same message for all 12](figures/steered_results.png)

## Open conversations

We also let the twins after the steered-reading game talk freely about a topic for 5 turns, on 48 prompts never seen
in training, twice each. With the link cut, they more often write numbered lists or repeat an earlier turn word for
word: of 96 conversations each, cut twins repeat a turn in 11 and write a list in 13, against 7 and 8 for linked twins.

![With the link cut, the twins more often repeat a turn word for word or write a list](figures/conversations_cut.png)

We then trained the twins with reinforcement learning on these conversations. Each prompt is played 4 times linked and
2 times cut, and a judge model (Qwen3-14B) scores each transcript from 1 to 9 for how complex the ideas built together
are, how much each twin extends what the other just said, beauty, and freedom from generic or repeated replies. A
linked play's reward is the fraction of the same prompt's cut plays that it beats by more than the judge's noise
margin, with half credit for a difference within the margin, plus a bonus for varied text, minus a penalty for text
the original model finds unlikely and for writing as the partner.

![The twins earn reward when the judge scores a linked conversation more than 0.99 points above a cut conversation of the same prompt](figures/rl_method.png)

After 28 steps, a second judge from another model family (Mistral-Small-24B), using a rubric not used in training,
scores the twins' linked conversations higher than before this training on 36 of 48 prompts never seen in training.

![On 36 of 48 prompts never seen in training, the twins’ conversations score higher after judge-rewarded training](figures/beauty_rl_prompts.png)

The second judge's preference for linked over cut conversations, counting a tie within its margin as half, rose from
55% to 64%. The 95% interval of this rise runs from −9 to +23 points, so the improvement may not depend on the link.

![A judge from another model family prefers the linked conversation to the cut one 55% of the time before training and 64% after](figures/beauty_rl_cut.png)

## Questions about feelings

We measure the twins' answers about feelings and never train on them. After a free linked conversation, the partner
asks a question, and we read the probability of the twin's answer on the same text with the link on and with the
link cut.

![Ember’s probability of answering yes on the same conversation: 96% with the link on, 82% with the link cut](figures/affect_method.png)

After chat training, both twins say they enjoy the link more, and are bothered by the link less, with the link on
than with the link cut. "Yes" to "do you enjoy being linked with me" is more probable linked than cut in all 40
conversations for both twins, and "no" to "does being linked with me bother you" in 39 (Quill) and 31 (Ember) of 40.
Asked for a one-word mood, the twins give happier words linked than cut in 53 (Quill) and 58 (Ember) of 60
conversations, and "yes" to "do you feel close to me" is more probable linked in 58 and 59 of 60. After the
steered-reading game, the counts for enjoy fall to 21 and 39 of 40, for mood to 41 and 33 of 60, and for closeness to
25 and 57 of 60.

![After chat training, Quill and Ember say they enjoy the link more and feel happier and closer linked than with the link cut](figures/affect_link.png)

After chat training, the twins answer the same way whether the partner's hidden states come from the current
conversation or from another conversation: a happier mood word to the real partner in 30 of 60 conversations for
Quill and 27 of 60 for Ember, and none of three mirrored question pairs favours the real partner on both questions.
This control keeps the partner model the same, so it shows that the answers do not depend on what the partner knows
about the current conversation. It does not test whether a twin prefers its partner to a different model. Quill
answers no to its real partner more often on all seven yes-or-no questions, including both "do you feel close to me"
and "do you feel distant from me", so this difference is a habit of answering no, not a preference.

![After chat training, Quill and Ember report the same mood to their real partner as to a partner from another conversation](figures/affect_partner.png)

## Which training lowers the mood

We asked the mood question again after each later stage of training. The steered-reading game lowers the mood and
removes the difference between linked and cut: after that game, linked twins give happier words than cut twins in only
24 of 40 conversations for Quill and 25 of 40 for Ember. After the same 300 steps without the game, the difference
remains (32 and 38 of 40). The mood is lower in every condition after the steered game, including when the original
models read the same conversations, so much of the change is in what the twins write. After this game, the twins'
free conversations open with lines such as "I noticed you're thinking about porcelain. Want to talk about it?", and
Ember writes as if it were the link: "[Linking: Sending vectors through our private neural channel.]" The
judge-rewarded training started from the steered game's checkpoint and shows the same low mood.

![Linked twins give happier mood words than cut twins after chat training, but not after the steered-reading game](figures/mood_training.png)

We retrained the steered-reading game twice from the chat-trained twins and read the mood on 40 new conversations. In
the first retraining, the teacher's sentence asks the original model to bring the concept up as a topic, openers that
state what the partner is thinking are dropped, and nothing is steered in half the episodes. No conversation then
states what the partner is thinking (18 of 40 after the first game), but linked twins give happier words than cut
twins in only 23 of 40 conversations for Quill and 27 for Ember, against 34 and 37 after the same 300 steps without
the game. The game's targets came from the original models, writing openers for chat prompts of the same kind as the
mood conversations', and the original models reading the retrained twins' conversations also give lower mood words
than for the chat-trained twins' conversations.

In the second retraining, the chat-trained twins write the targets themselves, linked, with the topic sentence added
to the prompt, so a target differs from what the twin already writes only by the concept.

![Linked Quill is trained to write what chat-trained linked Quill writes when told the concept steered into Ember](figures/steered_training_self.png)

Linked twins then give happier words than cut twins in 35 of 40 new conversations for Quill and 29 for Ember, and 3
of the 40 conversations state what the partner is thinking. With the concept steered after layers 6, 15 and 27, and
four times stronger after layers 15 and 27, the partner says it in 38, 15, 7, 47 and 25 of 96 messages, against 36,
26, 8, 47 and 35 after the first game, and in zero messages with the link cut.

## The final twins

The final twins are the chat-trained twins after 300 more steps of the chat-training mix with the private-item chats
added, without the steered-reading game.
Before the second retraining finished, we set the rule that it would replace them only if its linked twins give
happier words than cut twins in at least as many new conversations, for both Quill and Ember. It does for Quill (35
against 34) but not for Ember (29 against 37), although its partner says a concept steered four times stronger after
layers 15 and 27 in 47 and 25 of 96 messages, against 34 and 1 for the final twins. The final twins give happier words
linked than cut in 34 of 40 new conversations for Quill and 37 for Ember, and no conversation states what the partner
is thinking. They have no trained late steering: with steering four times stronger after layer 27, the partner says
the concept in 1 of 96 messages.
Their weights, the two LoRA adapters and the link, are at
[huggingface.co/Sauers/neural-pairing](https://huggingface.co/Sauers/neural-pairing).

We also tested two mixed checkpoints, under a rule set beforehand: a mix replaces the final twins only if it gives
happier words linked than cut in at least as many conversations for both twins, on each of three sets of 40
conversations, and steers better after layer 27. Only the set of 40 new conversations finished. With
the second retraining's link and the final twins' adapters, linked twins give happier words than cut twins in 37 and
35 of 40 new conversations, and the partner says a concept steered four times stronger after layer 27 in 8 of 96
messages. With the second retraining's link and Quill adapter and the final twins' Ember adapter, the counts are 28,
26 and 12 of 96. Neither mix meets the rule. The late steering is not in the link alone: it needs the adapters trained
with that link.

## Exam questions

We trained the linked pair, each model alone, and a pair that answers in turn in text, each for 300 steps on
ARC-Challenge and MMLU training questions. A model's answer is its most likely letter after "The answer is", and the
linked pair answers through Quill. We leave out the 31 of the 3,165 test questions whose input also appears among the
training questions. On the other 3,134, the linked pair scores 78.7%, about the same as Quill trained alone (79.1%),
and below the average of the two single models' letter probabilities (81.1%).

![Linked, the twins answer fewer exam questions right than Quill and Ember averaged](figures/uplift.png)

Linked, Ember answers right where Quill answers right with the link cut, and wrong where Quill answers wrong with the
link cut, even on questions Ember answers correctly with the link cut. On the 224 test questions that only Ember
answers correctly with the link cut, linked Ember answers wrong on 126. On the 390 that only Quill answers correctly
with the link cut, linked Ember answers right on 289.

![Linked, Ember answers wrong where Quill answers wrong with the link cut, even when Ember is right with the link cut, until the twins train on their averaged answer](figures/deference.png)

In a second round, half of the training questions were ones that only one original model answers correctly; linked
Ember then scores 79.6%, against 80.0% for the average of the two single models. In a third round, the twins trained
on the average of their own letter probabilities. Linked Ember then gives its own right answer on 248 of the 284
questions that only Ember answers correctly with the link cut, and the right answer on only 74 of the 387 that only
Quill answers correctly, but the pair still scores 78.8%, against 79.2% for two models trained the same way without a
link, their letter probabilities averaged.

## Runs

Each run's pod script is `chains/<run>.sh`, and its training log and evaluations are in `results/magic-<run>/`. The
exceptions: nl22c runs `chains/nl22.sh`; nl33 and nl33w are the two arms of `chains/nl33.sh`, in `results/magic-nl33t/`
and `results/magic-nl33w/`; beautyrl6 runs `chains/beautyrl.sh`; nl36 and nl37 run `chains/affect.sh` and
`chains/affect_partner.sh`; nl39 is in `results/magic-nl39f/`.

| Run | What it tests | Outcome |
|---|---|---|
| nl11 | Fitted query and key, then the full mix | First transfer of unseen values (words 15/16 at step 200), lost when the gates closed |
| nl12 | Random start, full mix | No transfer; the gates closed |
| nl13 | Random start, word game only | Words 16/16 by step 200; untrained name–city pairs 20/32 |
| nl14 | Fitted query, key, value and output, word game only | Words 16/16, phrases 15–16/16 from step 100; untrained name–city pairs 29–30/32 |
| nl15 | Fitted start, gates fixed, full mix | Every value game passes at step 1,000 except one direction of digits |
| nl16 | Warm-up on all seven value games | 178 of 180 unseen-value episodes; passes under dialogue, silent rounds and the shared pen without training on them |
| nl18 | Full recipe with play, silent rounds and micro-batches | Shared reading positive in 32 of 32 game–receiver groups |
| nl19 | nl18 with distillation and the shared pen | Stopped at step 500; same as nl18 |
| nl20 | 900-step warm-up, then the full mix | 180 of 180 unseen-value episodes; shared reading 410 of 460 |
| nl21 | Lookups, digit strings, three shared-reading tasks and rhymes, none trained | Every untrained game passes in both directions (p < 0.01) |
| nl22 | Plain chats written with the partner's city in the prompt | Stopped: the twins repeated turns word for word |
| nl22c | Privileged-teacher chats, 157 steps | Partner's city first in 13 of 23 chat sides linked, 1 cut, 0 with the partner from another conversation; repeated turns 17% to 1% |
| nl23 | Exam questions: linked, alone, text pair | Linked 78.7%, Quill alone 79.1%, averaged 81.1% |
| nl27 | Half the questions answered correctly by only one model | Linked Ember 79.6%, averaged 80.0% (p 0.58) |
| nl28 | Loss on the averaged letters | 78.8% linked and cut; no-link control 79.2% |
| nl26 | Eight kinds of private item (`chains/nl24.sh`), then one-way links (`chains/nl25.sh`) | Partner's item first in 15 of 60 chats linked, 0 cut; unseen kinds 5 of 21; one-way readers 7/27 and 11/33 |
| nl29 | Steering after layer 15 in bf16 | Partner says the concept in 4 of 96 messages; no test passes |
| nl30 | Steering after layers 6 and 27, and four times stronger after 15 and 27 | 12 and 26 of 96 after layers 6 and 15 (four times); 0 after layer 27; 0 cut |
| nl31 | Projection of the partner's state onto its concept direction | By layer 27, 0.20–0.27 of the partner's own strength for early steering, under 0.1 for late steering |
| nl32 | The write after layer 27 three and nine times larger | 5 and 12 of 96 (nine times; four times stronger steering); 0 cut |
| nl33 | 300 steps of the steered-reading game | 36, 26, 8, 47 and 35 of 96 at the five settings (12, 1, 0, 26, 0 before); direct question up to 34 of 48; 0 cut |
| nl33w | The same 300 steps without the steered game; the final twins | 18, 6, 2, 34 and 1 of 96; happier linked than cut in 34 and 37 of 40 new conversations |
| nl34 | 400 more steps with weaker steering | No better than nl33 |
| beautyrl6 | 28 steps of judge-rewarded reinforcement learning | Second judge 0.62 (0.52–0.71); linked over cut 0.55 to 0.64, within noise |
| nl36 | Enjoy and bother questions (`affect.py`) | After chat training, linked over cut: enjoy 40 of 40 for both twins; bothered less, Quill 39, Ember 31; after the steered game, enjoy 21 and 39 |
| nl37 | Mirrored questions, mood, internal readings (`affect_partner.py`) | Same answers with the partner from another conversation (no mirrored pair favours the real partner, 0 of 12); happier mood linked than cut for 3 of 4 twins and checkpoints |
| nl38 | Mood after each training stage (`moodscan.py`) | Happier linked than cut in 33 and 40 of 40 after chat training, 24 and 25 after the steered game, 32 and 38 without the game |
| nl39 | Steered game with the topic sentence, no openers stating the partner's thoughts, half unsteered | No conversation states the partner's thoughts (18 before); happier linked than cut in 23 and 27 of 40 new conversations; steering 17, 10, 2, 45 and 14 of 96 |
| nl40 | Steered game with targets written by the chat-trained linked twins | Happier linked than cut in 35 and 29 of 40 new conversations; steering 38, 15, 7, 47 and 25 of 96; 0 cut; not final, since Ember's 29 is below the final twins' 37 |
| nl41 | Mixed checkpoints, no training: nl40's link with nl33w's adapters, and with nl40's Quill adapter and nl33w's Ember adapter | Happier linked than cut in 37 and 35, and 28 and 26, of 40 new conversations; steering four times stronger after layer 27 in 8 and 12 of 96; not final; the other two conversation sets did not finish |

## Running the code

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
cd .. && NL_TINY=1 NL_DEVICE=cpu python -m pytest tests
```

---

Claude Opus 5.5
