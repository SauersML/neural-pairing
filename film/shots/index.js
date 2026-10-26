// v10 cut list (117 s). `shift`: update(t) receives t - shift. `map`: [[global, authored], ...] piecewise linear, for
// holds and speed-ups inside a shot (main.js localT).
export const SHOTS = [
  { id: 'a1_meet', start: 0, end: 12 },
  { id: 'a2_link', start: 12, end: 36, map: [[12, 12], [32.5, 32.5], [34.5, 37.5], [36, 40]] },   // fast generation 2.5x, fold 1.7x
  { id: 'ab_plate', start: 36, end: 44, shift: -4 },
  { id: 'a3_cross', start: 44, end: 54, shift: 12 },
  { id: 'a4_games', start: 54, end: 68, shift: 4 },
  { id: 'a5_train', start: 68, end: 76, shift: 10 },   // starts at its authored 58.0: the fitted-start beat is cut
  // chat training: more time on the original's sentence and on the lit target
  { id: 'a5b_teacher', start: 76, end: 82, map: [[76, 76], [77.5, 77.25], [78.5, 78.0], [79.25, 78.125], [79.75, 78.5], [80.5, 79.0], [81.5, 79.375], [82, 80]] },
  // learned to talk: the echo holds, "I noticed you mentioned you're in Oslo." holds, the stat holds
  { id: 'a5c_talk', start: 82, end: 88, map: [[82, 88], [82.375, 88.25], [83.375, 89.0], [84.0, 89.25], [84.25, 89.5], [85.0, 90.0], [85.5, 90.375], [86.0, 90.75], [86.5, 91.0], [87.0, 91.25], [87.5, 91.5], [88, 92]] },
  { id: 'a6_proof', start: 88, end: 96, shift: -4 },   // authored in v9 global 92-100
  { id: 'a7_happy', start: 96, end: 113, shift: -10 },
  { id: 'a6_title', start: 113, end: 117, shift: 33 },
];
