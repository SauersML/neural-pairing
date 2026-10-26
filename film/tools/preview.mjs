// Render still frames for review.
//   node tools/preview.mjs --times 1,2.5,4 [--shots s2_link] [--scale 0.5] [--out out/preview/name] [--sheet]
// Writes <out>_<t>.jpg for each time, and with --sheet a contact sheet <out>_sheet.jpg (labelled with times).
import { openFilm } from './browser.mjs';
import { execFileSync } from 'node:child_process'; import fs from 'node:fs'; import path from 'node:path';
const a = Object.fromEntries(process.argv.slice(2).reduce((acc, x, i, arr) => { if (x.startsWith('--')) acc.push([x.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]); return acc; }, []));
const times = String(a.times || '0').split(',').map(Number);
const scale = parseFloat(a.scale || '0.5');
const { browser, page, ROOT } = await openFilm({ scale, shots: a.shots || null, test: a.test || null });
const out = path.resolve(ROOT, a.out || 'out/preview/p'); fs.mkdirSync(path.dirname(out), { recursive: true });
const files = [];
for (const t of times) {
  const t0 = Date.now();
  const id = await page.evaluate((t) => window.renderAt(t), t);
  const f = `${out}_${t.toFixed(2)}.jpg`;
  await page.screenshot({ path: f, type: 'jpeg', quality: 88 });
  files.push(f); console.log(`t=${t} shot=${id} -> ${f} (${Date.now() - t0} ms)`);
}
await browser.close();
if (a.sheet && files.length > 1) {
  const py = `import sys\nfrom PIL import Image, ImageDraw, ImageFont\nfs=sys.argv[2:]; ims=[Image.open(f) for f in fs]; w,h=ims[0].size; cols=min(3,len(ims)); rows=(len(ims)+cols-1)//cols\nS=Image.new('RGB',(cols*w,rows*h),(0,0,0)); d=ImageDraw.Draw(S)\ntry: fnt=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', max(14,h//22))\nexcept Exception: fnt=None\nfor i,(im,f) in enumerate(zip(ims,fs)):\n  x,y=(i%cols)*w,(i//cols)*h; S.paste(im,(x,y)); d.text((x+8,y+6),'t='+f.rsplit('_',1)[1][:-4],fill=(255,255,0),font=fnt)\nS.save(sys.argv[1],quality=85)`;
  execFileSync('python3', ['-I', '-c', py, `${out}_sheet.jpg`, ...files]);
  console.log('sheet ->', `${out}_sheet.jpg`);
}
