// Shared: launch headless chromium (SwiftShader WebGL) and serve /home/user/video from a virtual origin.
import { chromium } from 'playwright';
import fs from 'node:fs'; import path from 'node:path';
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const TYPES = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.wav': 'audio/wav', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.ttf': 'font/ttf' };
export async function openFilm({ scale = 1, shots = null, test = null } = {}) {
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--font-render-hinting=none'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: scale });
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.type(), m.text()); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.route('http://film.local/**', async (route) => {
    const u = new URL(route.request().url()); const f = path.join(ROOT, decodeURIComponent(u.pathname));
    if (!fs.existsSync(f)) return route.fulfill({ status: 404, body: 'nf' });
    route.fulfill({ status: 200, body: fs.readFileSync(f), headers: { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream', 'cache-control': 'no-store' } });
  });
  const q = new URLSearchParams({ scale: String(scale) }); if (shots) q.set('shots', shots); if (test) q.set('test', test);
  await page.goto('http://film.local/index.html?' + q.toString());
  await page.waitForFunction('window.filmReady === true', null, { timeout: 60000 });
  return { browser, page, ROOT };
}
