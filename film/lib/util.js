// Pure helpers. Everything in this film is a function of time t (seconds): no accumulated state.
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, k) => a + (b - a) * k;
export const remap = (x, a, b, c = 0, d = 1) => lerp(c, d, clamp((x - a) / (b - a)));
export const smooth = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
export const smoother = (x) => { x = clamp(x); return x * x * x * (x * (x * 6 - 15) + 10); };
// progress of t through [a,b], eased
export const prog = (t, a, b, ease = smoother) => ease(clamp((t - a) / (b - a)));
export const easeOutCubic = (x) => 1 - Math.pow(1 - clamp(x), 3);
export const easeInCubic = (x) => Math.pow(clamp(x), 3);
export const easeInOutCubic = (x) => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
export const easeOutExpo = (x) => { x = clamp(x); return x === 1 ? 1 : 1 - Math.pow(2, -10 * x); };
export const easeInOutExpo = (x) => { x = clamp(x); if (x === 0 || x === 1) return x; return x < .5 ? Math.pow(2, 20 * x - 10) / 2 : (2 - Math.pow(2, -20 * x + 10)) / 2; };
export const easeOutBack = (x, s = 1.70158) => { x = clamp(x); const c3 = s + 1; return 1 + c3 * Math.pow(x - 1, 3) + s * Math.pow(x - 1, 2); };
// analytic underdamped spring from 0 -> 1 starting at t0 (overshoots, settles). freq in Hz, damping ratio z.
export const spring = (t, t0, freq = 2.2, z = 0.35) => {
  const x = t - t0; if (x <= 0) return 0;
  const w = 2 * Math.PI * freq, wd = w * Math.sqrt(1 - z * z);
  return 1 - Math.exp(-z * w * x) * (Math.cos(wd * x) + (z * w / wd) * Math.sin(wd * x));
};
// a decaying "kick" impulse at t0 (0 before, peaks to 1, wobbles, decays) — great for impacts / squash
export const kick = (t, t0, freq = 3, decay = 5) => { const x = t - t0; if (x <= 0) return 0; return Math.exp(-decay * x) * Math.sin(2 * Math.PI * freq * x); };
// window: 0 before a, rises over fade to 1, holds, falls to 0 at b
export const win = (t, a, b, fade = 0.3) => Math.min(prog(t, a, a + fade), 1 - prog(t, b - fade, b));
// deterministic hash noise
export const hash = (n) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };
export const noise1 = (x, seed = 0) => { const i = Math.floor(x), f = x - i; const u = f * f * (3 - 2 * f); return lerp(hash(i + seed * 57.3), hash(i + 1 + seed * 57.3), u) * 2 - 1; };
export function rng(seed = 1) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
// music clock: 120 bpm, 1 beat = 0.5 s, 1 bar = 2 s. bar(n) = start time of bar n (0-based).
export const BPM = 120, BEAT = 60 / BPM, BAR = BEAT * 4;
export const beat = (n) => n * BEAT;
export const bar = (n) => n * BAR;
// pulse on every beat: 1 at the beat, decaying
export const beatPulse = (t, decay = 6) => { const ph = (t % BEAT); return Math.exp(-decay * ph); };
