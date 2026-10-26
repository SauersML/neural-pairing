// S5 helper: DOM typography for the number, the stat cards, the check/cross badges and the title card.
// All update functions are pure functions of global t.
import { el, FONT } from '../lib/hud.js';
import { clamp, prog, spring, easeOutExpo, easeOutCubic, kick } from '../lib/util.js';

const TEAL = '#34f5c0', TEAL_SOFT = '#a7f3d0', TXT = '#f5f3ff', BAD = '#f87171', GREYC = '#8b8fa3';
const Q = '#a78bfa', E = '#fb923c';

// ---------- check / cross badge (pinned next to a written word)
export function makeBadge(parent, kind) {
  const ok = kind === 'ok', c = ok ? TEAL : BAD;
  const root = el(parent, 'div', { position: 'absolute', left: 0, top: 0, width: '76px', height: '76px', borderRadius: '50%', background: ok ? 'rgba(4,46,36,0.9)' : 'rgba(60,8,12,0.9)', border: `3px solid ${c}`, boxShadow: `0 0 28px ${c}80, inset 0 0 14px ${c}40`, display: 'none' });
  const path = ok ? 'M15 30 L25 40 L44 19' : 'M19 19 L39 39 M39 19 L19 39';
  root.innerHTML = `<svg width="70" height="70" viewBox="0 0 58 58" style="position:absolute;left:0;top:0"><path d="${path}" fill="none" stroke="${c}" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/></svg>`;
  const p = root.querySelector('path');
  return {
    // x, y screen px (center), t0 = pop time, alpha
    update(t, x, y, t0, alpha = 1) {
      if (t < t0 || alpha <= 0.002) { root.style.display = 'none'; return; }
      root.style.display = 'block';
      const s = spring(t, t0, 2.8, 0.42);
      root.style.transform = `translate(${x - 38}px, ${y - 38}px) scale(${Math.max(0.01, s)}) rotate(${(1 - s) * -25}deg)`;
      root.style.opacity = alpha * clamp((t - t0) / 0.06);
      p.setAttribute('stroke-dashoffset', String(1 - easeOutCubic(clamp((t - t0 - 0.06) / 0.28))));
    },
  };
}

// ---------- the big number: "179 / 180"
export function makeNumber(parent) {
  const root = el(parent, 'div', { position: 'absolute', left: '0px', top: '0px', width: '1920px', textAlign: 'center', display: 'none', transformOrigin: '960px 0px' });
  const big = el(root, 'div', { font: `700 196px ${FONT.display}`, lineHeight: '1', letterSpacing: '-0.045em', fontFeatureSettings: '"tnum" 1, "ss01" 1', color: '#ffffff', whiteSpace: 'nowrap' });
  const n = el(big, 'span', { display: 'inline-block', textShadow: '0 0 60px rgba(52,245,192,0.55), 0 4px 30px rgba(0,0,0,0.6)' }, '0');
  const sl = el(big, 'span', { display: 'inline-block', fontWeight: '300', color: 'rgba(245,243,255,0.46)', margin: '0 0.16em 0 0.2em', textShadow: '0 4px 30px rgba(0,0,0,0.6)' }, '/');
  el(big, 'span', { display: 'inline-block', fontWeight: '300', color: 'rgba(245,243,255,0.46)', textShadow: '0 4px 30px rgba(0,0,0,0.6)' }, '180');
  return {
    root, n,
    // y: top px; count: integer; t: time; show alpha; scale
    update(t, { alpha, y, scale, count, pop }) {
      root.style.display = alpha > 0.002 ? 'block' : 'none'; if (alpha <= 0.002) return;
      root.style.opacity = alpha;
      root.style.transform = `translateY(${y}px) scale(${scale})`;
      n.textContent = String(count);
      n.style.transform = `scale(${1 + pop})`;
      const blur = Math.max(0, (1 - clamp(alpha * 1.2)) * 14);
      root.style.filter = blur > 0.05 ? `blur(${blur}px)` : 'none';
    },
  };
}

// ---------- three stat cards
const CARD_W = 486, CARD_GAP = 42;
export function makeCards(parent) {
  const row = el(parent, 'div', { position: 'absolute', left: `${(1920 - 3 * CARD_W - 2 * CARD_GAP) / 2}px`, top: '452px', width: `${3 * CARD_W + 2 * CARD_GAP}px`, display: 'none' });
  const specs = [
    // nl21: lookup was left out of training; real partner wins 8/10 into AFM + 9/10 into Qwen = 17/20
    { stat: '17<span class=sl>/</span>20', desc: 'code lookups,<br>a game never trained', viz: 'dots20' },
    { stat: '128<span class=sl>/</span>128', desc: 'turn structures<br>never trained', viz: 'dots128' },
    { stat: '31<span class=sl>/</span>32', desc: 'shared-reading<br>groups', viz: 'dots32' },
  ];
  const cards = specs.map((sp, i) => {
    const c = el(row, 'div', {
      position: 'absolute', left: `${i * (CARD_W + CARD_GAP)}px`, top: '0px', width: `${CARD_W}px`, height: '322px', padding: '30px 36px 28px',
      background: 'linear-gradient(180deg, rgba(22,18,44,0.78), rgba(10,8,22,0.78))', borderRadius: '20px',
      border: '1px solid rgba(245,243,255,0.12)', boxShadow: '0 30px 60px rgba(0,0,0,0.45)', overflow: 'hidden', transformOrigin: '50% 60%',
    });
    const bar = el(c, 'div', { position: 'absolute', left: '0', top: '0', height: '3px', width: '100%', background: `linear-gradient(90deg, ${TEAL}, rgba(52,245,192,0.15))` });
    const stat = el(c, 'div', { font: `700 84px ${FONT.display}`, lineHeight: '1', letterSpacing: '-0.035em', fontFeatureSettings: '"tnum" 1', color: TEAL, textShadow: '0 0 34px rgba(52,245,192,0.45)', whiteSpace: 'nowrap' }, sp.stat);
    stat.querySelectorAll('.sl').forEach((x) => Object.assign(x.style, { fontWeight: '300', margin: '0 0.14em', opacity: '0.7' }));
    const viz = el(c, 'div', { position: 'relative', height: '46px', marginTop: '24px' });
    const units = [];
    if (sp.viz === 'dots20') {
      const greys = [5, 12, 17];
      for (let k = 0; k < 20; k++) { const g = greys.includes(k); units.push(el(viz, 'div', { position: 'absolute', left: `${k * 20.5}px`, top: '14px', width: '15px', height: '15px', borderRadius: '50%', background: g ? GREYC : TEAL })); units[k].grey = g; }
    } else if (sp.viz === 'dots32') {
      for (let k = 0; k < 32; k++) { const r = Math.floor(k / 16), col = k % 16; units.push(el(viz, 'div', { position: 'absolute', left: `${col * 25}px`, top: `${r * 24 + 4}px`, width: '14px', height: '14px', borderRadius: '50%', background: k === 22 ? GREYC : TEAL })); units[k].grey = k === 22; }
    } else if (sp.viz === 'dots128') {
      for (let k = 0; k < 128; k++) { const r = Math.floor(k / 32), col = k % 32; units.push(el(viz, 'div', { position: 'absolute', left: `${col * 12.6}px`, top: `${r * 11.5 + 1}px`, width: '7.5px', height: '7.5px', borderRadius: '50%', background: TEAL })); }
    } else {
      // 100 gates: 6 open, 3 more half-open (6–9%), the rest closed
      for (let k = 0; k < 100; k++) { const open = k < 6 ? 1 : k < 9 ? 0.5 : 0; units.push(el(viz, 'div', { position: 'absolute', left: `${k * 4.12}px`, top: open ? '0px' : '16px', width: '2.2px', height: open ? '40px' : '24px', borderRadius: '1px', background: open ? TEAL : 'rgba(245,243,255,0.22)' })); units[k].open = open; }
    }
    const desc = el(c, 'div', { font: `500 36px ${FONT.text}`, lineHeight: '1.2', color: 'rgba(245,243,255,0.92)', marginTop: '22px', letterSpacing: '-0.01em' }, sp.desc);
    if (sp.detail) el(c, 'div', { font: `500 15px ${FONT.mono}`, letterSpacing: '0.08em', whiteSpace: 'nowrap', textTransform: 'uppercase', color: 'rgba(245,243,255,0.55)', marginTop: '12px' }, sp.detail);
    if (sp.whisper) el(c, 'div', { font: `italic 500 30px ${FONT.display}`, color: TEAL_SOFT, marginTop: '6px', letterSpacing: '-0.01em' }, sp.whisper);
    return { c, bar, stat, units, viz: sp.viz };
  });
  return {
    update(t, hits, tOut) {
      const vis = t >= hits[0] - 0.01 && t < tOut + 0.4;
      row.style.display = vis ? 'block' : 'none'; if (!vis) return;
      cards.forEach((cd, i) => {
        const t0 = hits[i];
        if (t < t0) { cd.c.style.display = 'none'; return; }
        cd.c.style.display = 'block';
        const s = spring(t, t0, 2.6, 0.5);
        const o = clamp((t - t0) / 0.08);
        const out = prog(t, tOut + i * 0.06, tOut + i * 0.06 + 0.3);
        const sc = 0.78 + 0.22 * s - out * 0.04;
        cd.c.style.opacity = o * (1 - out);
        cd.c.style.transform = `translateY(${(1 - s) * 70 - out * 30}px) scale(${sc})`;
        const bl = (1 - clamp((t - t0) / 0.14)) * 12 + out * 10;
        cd.c.style.filter = bl > 0.05 ? `blur(${bl}px)` : 'none';
        const glow = Math.exp(-(t - t0) * 3.5);
        cd.c.style.borderColor = `rgba(52,245,192,${0.12 + 0.6 * glow})`;
        cd.c.style.boxShadow = `0 30px 60px rgba(0,0,0,0.45), 0 0 ${40 + 40 * glow}px rgba(52,245,192,${0.05 + 0.35 * glow})`;
        // viz fill sequence
        const n = cd.units.length;
        cd.units.forEach((u, k) => {
          const tk = t0 + 0.12 + 0.55 * (k / n);
          const a = clamp((t - tk) / 0.08);
          let base = 1;
          if (cd.viz === 'gates') base = u.open ? (u.open === 1 ? 1 : 0.55) : 1;
          u.style.opacity = a * base;
          u.style.boxShadow = (cd.viz !== 'gates' || u.open) && !u.grey ? `0 0 ${6 + 10 * Math.exp(-(t - tk) * 6)}px rgba(52,245,192,0.7)` : 'none';
        });
      });
    },
  };
}

// ---------- title card
export function makeTitle(parent) {
  const root = el(parent, 'div', { position: 'absolute', left: '0px', top: '0px', width: '1920px', textAlign: 'center', display: 'none' });
  const title = el(root, 'div', { font: `700 128px ${FONT.display}`, lineHeight: '1', letterSpacing: '-0.02em', color: '#fff', whiteSpace: 'nowrap' });
  const letters = [...'Neural pairing'].map((ch) => el(title, 'span', { display: 'inline-block', willChange: 'transform' }, ch === ' ' ? '&nbsp;' : ch));
  const tag = el(root, 'div', { font: `400 50px ${FONT.text}`, letterSpacing: '-0.005em', color: 'rgba(245,243,255,0.94)', marginTop: '30px', whiteSpace: 'nowrap' });
  const tagParts = ['Two', 'models', 'that', 'talk', 'in', '<span style="color:#34f5c0">neuralese</span>'].map((w) => el(tag, 'span', { display: 'inline-block', margin: '0 0.13em' }, w));
  const rule = el(root, 'div', { width: '72px', height: '2px', background: 'rgba(245,243,255,0.35)', margin: '28px auto 0' });
  const credit = el(root, 'div', { font: `500 38px ${FONT.text}`, letterSpacing: '0.01em', color: 'rgba(245,243,255,0.75)', marginTop: '24px', whiteSpace: 'nowrap' }, '<span style="color:#c4b5fd">Qwen3-4B</span>&nbsp;&nbsp;×&nbsp;&nbsp;<span style="color:#fdba74">AFM-4.5B</span>');
  return {
    update(t, { top, tTitle, tTag, tCredit }) {
      const vis = t >= tTitle - 0.01;
      root.style.display = vis ? 'block' : 'none'; if (!vis) return;
      root.style.transform = `translateY(${top}px)`;
      // letters: rise from blur, tracking settles from the center outward
      const n = letters.length;
      letters.forEach((L, i) => {
        const dc = Math.abs(i - (n - 1) / 2) / ((n - 1) / 2);
        const p = easeOutExpo(clamp((t - tTitle - dc * 0.18) / 0.9));
        const spread = (i - (n - 1) / 2) * 22 * (1 - p);
        L.style.opacity = p;
        L.style.transform = `translate(${spread}px, ${(1 - p) * 26}px)`;
        L.style.filter = p < 0.995 ? `blur(${(1 - p) * 14}px)` : 'none';
      });
      const g = Math.exp(-Math.max(0, t - tTitle) * 1.6);
      title.style.textShadow = `0 0 ${30 + 50 * g}px rgba(52,245,192,${0.28 + 0.45 * g}), 0 0 90px rgba(139,92,246,0.25), 0 4px 30px rgba(0,0,0,0.6)`;
      tagParts.forEach((w, i) => {
        const p = easeOutExpo(clamp((t - tTag[0] - i * 0.06) / 0.5));
        w.style.opacity = p; w.style.transform = `translateY(${(1 - p) * 0.5}em)`;
        w.style.filter = p < 0.995 ? `blur(${(1 - p) * 8}px)` : 'none';
      });
      const pr = easeOutCubic(clamp((t - tCredit + 0.15) / 0.7));
      rule.style.opacity = pr; rule.style.width = `${72 * pr}px`;
      const pc = easeOutExpo(clamp((t - tCredit) / 0.7));
      credit.style.opacity = pc; credit.style.transform = `translateY(${(1 - pc) * 10}px)`;
    },
  };
}

export { TEAL, Q, E, GREYC, kick };
