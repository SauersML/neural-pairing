# a5_train cues (v9 global time; shot authored for 54–66, shift 14; 120 bpm, 16th = 0.125 s)

| time | event | music |
|---|---|---|
| 68.000 | cut in; token rows pop (Quill's tiles, then Ember's, a quick cascade) | downbeat, cascade of ticks |
| 68.50–69.375 | where tokens end at the same character, gold lines snap between the rows on 16ths; marks where no partner token ends | 16th clicks |
| 69.125–69.875 | the fitted map slab sweeps across; orbs cross it both ways at once | sweep |
| 70.000, 70.250, 70.500 | attention reads, both directions at once; the peaks land on the matching token | beat, 8th, beat hits |
| 71.000–72.000 | stat "89–97% vs 1.2% chance" (hold) | downbeat, hold |
| 71.750 | creatures move home | — |
| 72.000 | token rows scatter | downbeat |
| 72.250 | frost on Quill (flash) | 16th hit, ice crack |
| 72.375 | frost on Ember | 16th hit |
| 73.00–73.75 | rank-32 LoRA chips light bottom-up on both | rising shimmer |
| 74.000–75.000 | label "rank-32 LoRA" (hold) | — |
| 75.000–76.750 | training cards (the game windows) launch on 8ths; each lands between the tails 0.75 s later and its prompts spark into both tails | 8th-note pulse |
| 75.300–75.800 | the chart slides in: "Chance of writing the partner's value", y 0% / 50% / 100%, x "training steps" 0 / 300 / 600 / 900 / 1,200; readout 0% | — |
| 76.000 | both lines start at 0 (step 0) | downbeat |
| ~76.6 | step 100: linked 0.1%; "link cut" label (grey line flat at 0) | — |
| 77.000–78.875 | cards launch on 16ths | 16th-note drive, tension build |
| ~77.0 | step 200: 19% (the S-curve takes off) | accent |
| ~77.25 | step 300: 51% | beat accent |
| ~77.5 | step 400: 90%; "linked" label | beat accent, peak of the rise |
| ~77.75 | step 500: 93% | — |
| 78.500 | step 900: 99%; the full mix begins (cards change color) | beat, add a layer |
| 79.500 | step 1,320 (after the full mix): 99.7% | main hit |
| 80.000 | cut to a5b | downbeat |

Data: shots/train_curve_data.js (run nl20): median over 144 held-out value tests of the reader's chance of writing the
whole right value, linked and with the link cut; the end point is the 180 tests after the full mix. The readout only
shows measured values (it sits on the last measured point). The step timing is eased (slow start), so the measured
points land at about 76.63, 76.98, 77.26, 77.51, 77.73 (steps 100–500), not exactly on the grid.
