# a2_link (12–40 s), v4

Files: `shots/a2_link.js` (timeline, camera, creatures, link, passes), `shots/a2_lattice.js` (Lattice: one model
unfolded into residual-stream columns × 36 blocks; Rays: instanced bezier tubes; Glows: sized glow points),
`shots/a2_readout.js` (one exchange read out on two glass panels, both directions at once). `lib/creature.js`: additive `pose.body` (default 1).
Contact sheet: `out/preview/a2/sheet.png`. Cues: `cues/a2_link.md`.

Layout: the time axis is world z (newest token at z = 0, older behind), Quill's lattice at x = -2.3, Ember's at +2.3
(exactly where the creatures stand, so each spine becomes its newest column). The camera orbits to Quill's side
(az -90°, ~45° elevation): time reads left to right, Quill's lattice in front, Ember's behind, the link rays cross
between them at the teal exchange rows. Heads stay above-right of each newest column (the next token comes out there).

Mechanism shown (pairing/link.py): 8 exchanges after blocks 6…27; at each, every reading position attends over all
visible partner positions (prompt: all-to-all; generation: partner tokens of the same tick or earlier) plus its own
null key; both directions at the same moment; the write is added to the reader's residual stream and rises with the
wave; self-attention arcs carry information to later positions; the gold idea crosses 8 times in one pass.
Close-up colour rule: what a model owns is its colour (sender: key, value, gate; reader: query, null key, output,
write size). Illustrative weights/gates are made up but plausible (one gate nearly closed on each side; recency shown
as ray brightness: same tick > previous tick > prompt bucket).

Read-out (24.75–31.25, `shots/a2_readout.js`): after the paper's mechanism figure. Two panels side by side face the
camera: "Quill reads Ember" (left) and "Ember reads Quill" (right). Rows: partner tokens; their states (bar codes);
values (capsules) with gate gauges (the value dims and shrinks with its gate); keys (tags pointing down). The reader's
query (a tag pointing up) sits on the reader's own stream at its newest position; the beams to every key and the null
key ("nothing") are in the reader's colour, thickness = weight, with teal weight bars under the keys. The gated values
flow into a teal sum; x write size; + into the reader's stream, where the partner's colour rises.

Heads on the lattice (`shots/a2_neck.js`, `headRig` in a2_link): each head rides on a neck grown out of the top of the
newest column: the same residual-stream core (shader, colour, flowing stripes) plus 4 vertebra pucks, a cubic Bezier
leaving the column vertically and curving into the base of the head (tangent-continuous at both ends). At each step
the new column's neck grows (first 30% of the hop window), the head springs over from the old column to the new one
with overshoot (spring on the anchor z, pure in t) and lands with a small squash; the neck it left retracts to a
short stub on the old column that fades over 0.9 s. The emitted tile leaves the mouth and drops in one arc to the
new column's base, where it becomes that column's token. Unfold/fold: the neck length and the head transform blend
with the creature's own spine-top head (the neck shrinks into the upper spine at 38.2–39.1).
Write size in the read-out: the x dial shows 10% (tanh(alpha) held at 0.1 in nl20 and nl22c).
