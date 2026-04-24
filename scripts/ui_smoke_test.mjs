import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const repoRoot = normalize(join(fileURLToPath(import.meta.url), '..', '..'));
const staticRoot = join(repoRoot, 'internal', 'webui', 'static');

const contentTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
]);

let imageRequestsWithNSFW = 0;
let videoRequests = 0;

function json(res, status, payload) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(payload));
}

function statsPayload() {
  return {
    queue: { total: 0, done: 0, failed: 0, missing: 0, annotations_missing: 0, pending: 0, leased: 0 },
    images_total: 0,
    videos_total: 50,
    recent_failures: [],
  };
}

async function serveStatic(res, pathname) {
  const rel = pathname === '/' ? 'index.html' : pathname.replace(/^\/assets\//, '');
  const filePath = join(staticRoot, rel);
  if (filePath !== staticRoot && !filePath.startsWith(staticRoot + sep)) {
    res.writeHead(404);
    res.end('not found');
    return;
  }
  const body = await readFile(filePath);
  res.writeHead(200, { 'Content-Type': contentTypes.get(extname(filePath)) || 'application/octet-stream' });
  res.end(body);
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    if (url.pathname === '/api/images') {
      if (url.searchParams.get('include_nsfw') === '1') {
        imageRequestsWithNSFW += 1;
      }
      json(res, 200, { images: [], total: 0 });
      return;
    }
    if (url.pathname === '/api/videos') {
      videoRequests += 1;
      const offset = Number(url.searchParams.get('offset') || 0);
      const limit = Number(url.searchParams.get('limit') || 24);
      const count = Math.max(0, Math.min(limit, 50 - offset));
      const videos = Array.from({ length: count }, (_, i) => ({
        video_id: offset + i + 1,
        media_type: 'video',
        original_name: `clip-${offset + i + 1}.mp4`,
        storage_path: `videos/clip-${offset + i + 1}`,
        mime_type: 'video/mp4',
        duration_ms: 1000,
        width: 320,
        height: 180,
        frame_count: 1,
        index_state: 'done',
        created_at: '2026-04-24T00:00:00Z',
      }));
      json(res, 200, { videos, total: 50 });
      return;
    }
    if (url.pathname === '/api/stats') {
      json(res, 200, statsPayload());
      return;
    }
    if (url.pathname === '/api/search/tag-cloud') {
      json(res, 200, { tags: [{ tag: 'concert', count: 3 }, { tag: 'dog', count: 2 }] });
      return;
    }
    if (url.pathname === '/api/search/tags') {
      json(res, 200, { results: [], total: 0 });
      return;
    }
    if (url.pathname === '/api/live') {
      res.writeHead(426, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'websocket unavailable in smoke test' }));
      return;
    }
    await serveStatic(res, url.pathname);
  } catch (err) {
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end(String(err && err.stack ? err.stack : err));
  }
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const address = server.address();
const baseURL = `http://127.0.0.1:${address.port}`;

let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(baseURL, { waitUntil: 'domcontentloaded' });

  await page.getByRole('heading', { name: 'imgsearch' }).waitFor();
  await page.getByRole('tab', { name: /Videos/ }).click();
  await page.locator('#videos-panel').waitFor({ state: 'visible' });

  // Tab keyboard navigation follows the current DOM order and skips disabled tabs.
  await page.keyboard.press('ArrowRight');
  await page.locator('#tags-panel').waitFor({ state: 'visible' });
  await page.keyboard.press('End');
  await page.locator('#tags-panel').waitFor({ state: 'visible' });
  await page.locator('.tag-cloud-chip[data-tag="concert"]').waitFor();
  await page.locator('.tag-cloud-chip[data-tag="concert"]').click();
  await page.locator('#results-panel').waitFor({ state: 'visible' });

  await page.getByRole('button', { name: 'Upload' }).click();
  await page.getByRole('dialog', { name: 'Add media' }).waitFor();
  await page.getByRole('button', { name: 'Close upload dialog' }).click();
  await page.locator('#upload-modal').waitFor({ state: 'hidden' });

  const nsfwRefresh = page.waitForResponse((response) => {
    const responseURL = new URL(response.url());
    return responseURL.pathname === '/api/images' && responseURL.searchParams.get('include_nsfw') === '1';
  });
  await page.getByLabel('Show NSFW').check();
  await page.waitForFunction(() => window.localStorage.getItem('imgsearch.showNSFW') === '1');
  await nsfwRefresh;
  if (imageRequestsWithNSFW === 0) {
    throw new Error('NSFW toggle did not refresh image requests with include_nsfw=1');
  }

  await page.getByRole('tab', { name: /Videos/ }).click();
  await page.getByRole('button', { name: 'Older' }).click();
  await page.getByText('Page 2 of 3').waitFor();
  if (videoRequests < 2) {
    throw new Error('video pagination did not request a second page');
  }

  console.log('UI smoke checks passed');
} finally {
  if (browser) {
    await browser.close();
  }
  await new Promise((resolve) => server.close(resolve));
}
