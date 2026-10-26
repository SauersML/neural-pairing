// Render frames: node tools/render.mjs --from 0 --to 60 --fps 30 --out out/frames [--scale 1] [--workers 1] [--worker k]
// Frame f covers time f/fps. Files: <out>/f_00000.png. Skips frames that already exist (resumable).
import { openFilm } from './browser.mjs';
import fs from 'node:fs'; import path from 'node:path';
const a = Object.fromEntries(process.argv.slice(2).reduce((acc, x, i, arr) => { if (x.startsWith('--')) acc.push([x.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]); return acc; }, []));
const fps = +(a.fps || 30), from = +(a.from || 0), to = +(a.to || 60), scale = +(a.scale || 1);
const workers = +(a.workers || 1), wk = +(a.worker || 0);
const { browser, page, ROOT } = await openFilm({ scale, shots: a.shots || null });
const out = path.resolve(ROOT, a.out || 'out/frames'); fs.mkdirSync(out, { recursive: true });
const f0 = Math.round(from * fps), f1 = Math.round(to * fps);
// contiguous chunk per worker
const per = Math.ceil((f1 - f0) / workers), a0 = f0 + wk * per, a1 = Math.min(f1, a0 + per);
const ext = a.jpg ? 'jpg' : 'png';
let t0 = Date.now(), n = 0;
for (let f = a0; f < a1; f++) {
  const file = path.join(out, `f_${String(f).padStart(5, '0')}.${ext}`);
  if (fs.existsSync(file) && fs.statSync(file).size > 0) continue;
  await page.evaluate((t) => window.renderAt(t), f / fps);
  await page.screenshot({ path: file + '.tmp.' + ext, type: ext === 'jpg' ? 'jpeg' : 'png', ...(ext === 'jpg' ? { quality: 92 } : {}) });
  fs.renameSync(file + '.tmp.' + ext, file);
  n++; if (n % 30 === 0) console.log(`worker ${wk}: frame ${f} (${((Date.now() - t0) / n).toFixed(0)} ms/frame)`);
}
console.log(`worker ${wk} done: ${n} frames`);
await browser.close();
