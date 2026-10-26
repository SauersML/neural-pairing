# v7 changes (read BRIEF_V4.md and BRIEF_V6.md first; all their rules still apply)

The user's feedback on v6:
- "180 won by the real partner is sorta vague, plus not a useful visual, and what is being 'won'?"
- The happier-when-linked scene "just shows results": make it large, obvious what is happening, and show the inside
  (internal representation) visually.

Cut list v7 (shots/index.js, 128 s): 0-92 unchanged; a6_proof 92-106 (shift 26: authored 66-80 = S5 time 46-60);
a7_happy 106-124 (rebuilt, authored in global time); a6_title 124-128 (shift 44: authored 80-84 = S5 60-64).

## a6_proof: what the 180 means, shown with real numbers (owner: proof agent)
The test (run nl20, results/values_unseen.json): a held-out value test. One twin holds a value it never saw as a value
in training (a word, number, phrase, digits, a sum part, a fact, a table entry) and the other must write it; neither
sees the other's text. Score: the receiver's probability of the right answer (teacher-forced, whole answer) with its real
partner, against the same with a partner from another test linked in (that partner holds a different value). The real
partner gives the higher probability in all 180 tests (7 games, both directions). Data: shots/proof180_data.js
(TESTS: [log10 real, log10 other, game, receiver, episode, tokens], sorted by "other" high to low; CIPHER: the example).

Example to show first (real record): relay_word episode 18. Quill holds "cipher"; Ember must write it.
- Real Quill linked: Ember's chance of "cipher" = 99.9%.
- Quill from another test linked instead (it holds a different word, not shown): 1 in 14 billion.

New S5-time plan (global = S5 + 46). Keep 46.0-49.5 (the control) but:
- words: "lantern" -> "cipher" (Quill's gold card and Ember's written tile). The other test's Quill holds a ghost card
  reading "another word" (never "galaxy": its real word is not shown).
- Ember never "writes galaxy ✗" (not measured). With the real partner Ember writes "cipher" and a big readout says
  "99.9%". With the other partner, Ember's "cipher" tile appears dim/grey with "1 in 14 billion". Remove the ✓/✗ badges
  or keep only what is true (the probabilities are the result).
- 49.5-55.0 replaces the ring and "180 / 180 won by the real partner": a chart, built in 3D with glow (as polished as
  the rest of the film, never a flat toy), large and filling the frame:
  - vertical log axis of the chance of the right answer: "100%", "1 in 1,000", "1 in a million", "1 in a billion",
    "1 in a trillion"; values below 1e-15 sit on a "smaller" floor band (say so with a label, do not hide them).
  - a short heading, e.g. "chance of the right answer".
  - the cipher pair arrives first (the two readouts become its two dots: teal = real partner at 99.9%, grey = the other
    test's partner at 1 in 14 billion, joined by a line), then the other 179 tests sweep in as columns (sorted, so the
    grey dots form a falling curve and the teal dots a flat line at the top).
  - direct labels: "real partner" by the teal line, "partner from another test" by the grey dots.
  - the count lands last: "180 / 180" with "higher with the real partner" (or similar plain words).
- Result windows (cards) and everything after move +2.0 s: cards at S5 55.0, 55.75, 56.5, 57.25; cardsOut 59.8;
  title0 59.9; titleText 60.25; tag 60.75; credit 61.25; hop 61.0/61.5; fade 62.8-64.0; CARDS_OUT 60.25. The title
  instance (a6_title, TITLE_ONLY) still fades the ring in around the pair at the title (S5 60.0-60.5) and starts from
  a7's last camera: pos (-0.1, 3.65, 15.2), target (0, 3.0, 0).
- Music: 120 bpm, put hits on the 16th grid, main hits on beats. Write cues/a6_proof.md (global times).

## a7_happy 106-124 (owner: director)
See shots/a7_happy.js and cues/a7_happy.md.
