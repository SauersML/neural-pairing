// Training curve (run nl20): at each evaluation step of the 900-step warm-up on the value games, the median over the
// held-out value tests (values never used in training, both directions) of the reader's teacher-forced probability of
// the whole right answer, linked and with the link cut, in percent. From results/magic-nl20/warm/payload_step*.json.
// [step, linked %, link cut %, tests]. FINAL: after the full mix (step 1,320), the 180 tests of payload_final.json.
export const CURVE2 = [[0, 0.0, 0.0, 144], [20, 0.01, 0.007779, 144], [100, 0.1, 0.004489, 144], [200, 19.2, 0.001877, 144], [300, 50.77, 0.002814, 144], [400, 90.21, 0.00033, 144], [500, 93.39, 0.000232, 144], [600, 93.37, 0.000235, 144], [700, 97.47, 0.000111, 144], [800, 98.17, 6.5e-05, 144], [900, 98.99, 6.3e-05, 144]];
export const FINAL2 = [1320, 99.73, 6e-05, 180];
