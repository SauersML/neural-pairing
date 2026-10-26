// v11 cut list (109 s). `shift`: update(t) receives t - shift. `map`: [[global, authored], ...] piecewise linear, for
// holds and speed-ups inside a shot (main.js localT).
export const SHOTS = [
  { id: 'a1_meet', start: 0, end: 12 },
  { id: 'a2_link', start: 12, end: 36, map: [[12, 12], [32.5, 32.5], [34.5, 37.5], [36, 40]] },   // fast generation 2.5x, fold 1.7x
  { id: 'ab_plate', start: 36, end: 44, shift: -4 },
  { id: 'a3_cross', start: 44, end: 54, shift: 12 },
  { id: 'a4_games', start: 54, end: 68, shift: 4 },
  { id: 'a5_train', start: 68, end: 76, shift: 10 },   // starts at its authored 58.0: the fitted-start beat is cut
  // learned to talk: the echo holds, "I noticed you mentioned you're in Oslo." holds (the teacher scene is cut in v11)
  { id: 'a5c_talk', start: 76, end: 82, map: [[76, 88], [76.375, 88.25], [77.375, 89.0], [78.0, 89.25], [78.25, 89.5], [79.0, 90.0], [79.5, 90.375], [80.0, 90.75], [80.5, 91.0], [81.0, 91.25], [81.5, 91.5], [82, 92]] },
  { id: 'a6_proof', start: 82, end: 88, shift: -10 },   // authored 92-98
  { id: 'a7_happy', start: 88, end: 105, shift: -18 },
  { id: 'a6_title', start: 105, end: 109, shift: 25 },
];
