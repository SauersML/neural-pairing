# v9 time map (121 s), from v8 (128 s). Every v8 global time t8 maps to v9 global t9:

| scene | v8 global | v9 global | how |
|---|---|---|---|
| a1_meet | 0–12 | 0–12 | same |
| a2_link | 12–32.5 | 12–32.5 | same |
| a2_link | 32.5–37.5 (fast generation) | 32.5–34.5 | 2.5x faster (linear) |
| a2_link | 37.5–40 (fold back) | 34.5–36 | 1.667x faster (linear) |
| ab_plate | 40–48 | 36–44 | -4 |
| a3_cross | 48–58 | 44–54 | -4 |
| a4_games | 58–72 | 54–68 | -4 |
| a5_train | 72–84 | 68–80 | -4 (the curve's look changes; its timing does not; see cues/a5_train.md when updated) |
| a5b_teacher | 84–88 | 80–86 | piecewise, slower; points (v8 -> v9): 84->80, 85.25->81.5, 86.0->82.5, 86.125->83.25, 86.5->83.75, 87.0->84.5, 87.375->85.5, 88->86 |
| a5c_talk | 88–92 | 86–92 | piecewise, slower; points: 88->86, 88.25->86.375, 89.0->87.375, 89.25->88.0, 89.5->88.25, 90.0->89.0, 90.375->89.875, 91.0->90.375, 91.3->90.5, 91.375->90.6, 91.5->90.85, 91.88->91.88, 92->92 |
| a6_proof | 92–106 | 92–100 | NEW content (five result windows): cues/a6_proof.md |
| a7_happy | 106–124 | 100–117 | retimed: cues/a7_happy.md (v9) |
| a6_title | 124–128 | 117–121 | -7 |

Points inside a5b/a5c are the cue events of cues/a5b_teacher.md and cues/a5c_talk.md (authored a5b time = v8 global - 8).
