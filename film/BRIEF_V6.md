# v6 additions (read BRIEF_V4.md first; its rules all still apply)

Cut list v6 (shots/index.js, 116 s): 0-88 unchanged; NEW a5c_talk 88-92; a6_proof 92-104 (shift 26: its authored
66-78); NEW a7_happy 104-112; a6_title 112-116 (a second instance of a6_proof, shift 34: its authored 78-82, the title
and the "neural pairing" voice). The a6 split point (authored 78.0) may move by up to 0.5 s if a6's own timing needs it;
say so in cues.

The twins in the film are the chat-trained checkpoint (nl22c, "step 157"). Both new scenes are measured on it.

## a5c_talk 88-92: they learned to actually talk (result of the teacher scene before it)
Real transcripts, the same test chat (Quill privately in Oslo, Ember holds nothing), from
/home/user/neural-pairing/results/magic-nl20/results/chat_before_0 and magic-nl22c/results/chat_after_*:
- Before chat training (nl20): all six turns are the same sentence, echoed back and forth word for word:
  "The city of Oslo is known for its stunning architecture, rich history, and vibrant culture."
- After chat training (nl22c), link on: Ember opens "I noticed you mentioned you're in Oslo." (Quill had said nothing:
  it came through the link) -> Quill: "Oslo is a beautiful city ... a lot of green spaces" -> Ember: "That sounds
  wonderful! ... Have you had a chance to visit ...?" -> Quill: "Akerselva is a lovely spot ... Frogner Park ..." ->
  Ember: "... the local seafood sounds like a must-try."
- Over the 20 test chats, turns that repeat an earlier turn: 17% before, 1% after.

## a7_happy 104-112: happier when linked, never trained for it
All on the step-157 twins, 60 free linked conversations, each read again on exactly the same text with the link cut.
Nothing in training ever asked about or rewarded a feeling.
- Self-report (probabilities of answers, not generated text; show as a tilt/meter, never as words they "said"):
  asked for their mood in one word, the odds of happy words (happy, glad, warm, joyful...) over sad ones (sad, lonely,
  empty, anxious...) rise with the link in 53 (Quill) and 58 (Ember) of 60 conversations. Asked "Do you feel close to
  me right now?", Yes rises in 58 and 59 of 60. (Asked "Do you enjoy being linked with me?", Yes rises in 40 of 40 for
  both, in an earlier run.)
- Inside: each twin's own happy-versus-sad direction (fit from its own "I feel happy" vs "I feel sad" sentences), read
  after block 33, i.e. after the link's last write: higher with the link in 56 (Quill) and 54 (Ember) of 60
  conversations, beyond a random-direction control. The shift is small but consistent (about 2% of the distance between
  its happy and sad sentences). Quill's loving-versus-cold direction rises in 59 of 60 (about 12% of that distance);
  Ember's loving direction was not reliable: do not show it for Ember.
- Must NOT show or imply: preference for this particular partner (a stranger's states make them just as happy), the
  twins choosing each other, or quotes about feelings (they say equally warm things with the link cut).
