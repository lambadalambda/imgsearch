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
let deletedImages = 0;

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
      json(res, 200, {
        images: [1, 2].map((id) => ({
          image_id: id,
          media_type: 'image',
          original_name: `image-${id}.jpg`,
          storage_path: `images/image-${id}`,
          mime_type: 'image/jpeg',
          width: 320,
          height: 240,
          index_state: 'done',
          description: id === 1 ? 'Brown dog running across a sunny grass field.' : '',
          tags: id === 1 ? ['dog', 'sunny field', 'outdoor portrait'] : [],
          created_at: '2026-04-24T00:00:00Z',
        })),
        total: 2,
      });
      return;
    }
    if (/^\/api\/images\/\d+$/.test(url.pathname) && req.method === 'DELETE') {
      deletedImages += 1;
      res.writeHead(204);
      res.end();
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
  const liveConnection = page.locator('#live-connection');
  await liveConnection.waitFor();
  if (!(await liveConnection.isVisible())) {
    throw new Error('live connection status is not visible in the masthead');
  }
  if (await liveConnection.evaluate((element) => Boolean(element.closest('.ops-menu-body')))) {
    throw new Error('live connection status is still buried in the overflow menu');
  }
  const mastheadIndexStatus = page.locator('#masthead-index-status');
  await mastheadIndexStatus.waitFor();
  if (!(await mastheadIndexStatus.isVisible())) {
    throw new Error('indexing status summary is not visible in the masthead');
  }
  if (await mastheadIndexStatus.evaluate((element) => Boolean(element.closest('.ops-menu-body')))) {
    throw new Error('indexing status summary is still buried in the overflow menu');
  }
  await page.locator('.ops-menu > summary').click();
  await page.getByRole('button', { name: 'Retry stuck' }).waitFor();
  await page.locator('.ops-menu > summary').click();
  const filterToggle = page.locator('.search-advanced-toggle');
  if (!((await filterToggle.textContent() || '').trim().startsWith('Filters'))) {
    throw new Error('advanced search disclosure is not labeled Filters');
  }
  await filterToggle.click();
  await page.locator('#negative-query').fill('cartoon');
  await page.locator('#search-tag-input').fill('dog');
  await page.locator('#search-tag-input').press('Enter');
  await filterToggle.click();
  const filterToggleText = (await filterToggle.textContent() || '').trim();
  if (!filterToggleText.includes('2')) {
    throw new Error(`filter toggle does not show active count: ${JSON.stringify(filterToggleText)}`);
  }
  const activeFilters = page.locator('#active-search-filters');
  if (!(await activeFilters.isVisible())) {
    throw new Error('active search filter chips are not visible when filters are closed');
  }
  const activeFilterText = await activeFilters.textContent() || '';
  if (!activeFilterText.includes('Exclude: cartoon') || !activeFilterText.includes('Tag: dog')) {
    throw new Error(`active filter chips are incomplete: ${JSON.stringify(activeFilterText)}`);
  }
  const firstCard = page.locator('.card').first();
  await firstCard.locator('.meta h3').waitFor();
  const firstTitle = (await firstCard.locator('.meta h3').textContent() || '').trim();
  if (firstTitle !== 'Brown dog running across a sunny grass field.') {
    throw new Error(`card title used ${JSON.stringify(firstTitle)}, want description text`);
  }
  if (await firstCard.locator('.meta .path').isVisible()) {
    throw new Error('resting card metadata still shows storage path');
  }
  const galleryPagination = page.locator('.pagination-controls[aria-label="Gallery pagination"]');
  const selectionToolbar = page.locator('#gallery-selection-toolbar');
  const selectionControlsInPagination = await galleryPagination.locator('#gallery-select-page, #gallery-clear-selection, #gallery-delete-selected, #gallery-delete-all').count();
  if (selectionControlsInPagination !== 0) {
    throw new Error('gallery pagination contains selection or destructive controls');
  }
  if (await selectionToolbar.isVisible()) {
    throw new Error('selection toolbar is visible before selection mode starts');
  }
  await page.getByRole('button', { name: 'Select' }).click();
  if (!(await selectionToolbar.isVisible())) {
    throw new Error('selection toolbar did not appear in selection mode');
  }
  if (!(await selectionToolbar.textContent() || '').includes('0 selected')) {
    throw new Error('selection toolbar does not show the selected count');
  }
  await selectionToolbar.getByRole('button', { name: 'Select page' }).waitFor();
  await selectionToolbar.getByRole('button', { name: 'Clear' }).waitFor();
  await selectionToolbar.getByRole('button', { name: 'Delete selected' }).waitFor();
  page.once('dialog', (dialog) => dialog.accept());
  await selectionToolbar.getByRole('button', { name: 'Delete all images (2)' }).click();
  await page.getByText('Deleted 2 images.').waitFor();
  if (deletedImages !== 2) {
    throw new Error(`bulk image delete sent ${deletedImages} DELETE requests, want 2`);
  }

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
  await page.waitForFunction(() => (document.getElementById('results-provenance')?.textContent || '').includes('Results for tags: concert'));
  const resultsProvenance = await page.locator('#results-provenance').textContent() || '';
  if (!resultsProvenance.includes('0 matches')) {
    throw new Error(`results provenance does not include result count: ${JSON.stringify(resultsProvenance)}`);
  }
  if ((await page.locator('#results-prev').textContent() || '').trim() !== 'Newer uploads') {
    throw new Error('results pagination does not clarify newer upload sort context');
  }
  if ((await page.locator('#results-next').textContent() || '').trim() !== 'Older uploads') {
    throw new Error('results pagination does not clarify older upload sort context');
  }

  await page.getByRole('button', { name: 'Upload', exact: true }).click();
  await page.getByRole('dialog', { name: 'Add media' }).waitFor();
  await page.getByRole('button', { name: 'Close upload dialog' }).click();
  await page.locator('#upload-modal').waitFor({ state: 'hidden' });

  const nsfwControl = page.locator('.nsfw-toggle');
  if (!(await nsfwControl.textContent() || '').includes('NSFW: Hidden')) {
    throw new Error('NSFW control does not show the hidden state by default');
  }
  const nsfwRefresh = page.waitForResponse((response) => {
    const responseURL = new URL(response.url());
    return responseURL.pathname === '/api/images' && responseURL.searchParams.get('include_nsfw') === '1';
  });
  await page.getByLabel(/NSFW/).check();
  await page.waitForFunction(() => window.localStorage.getItem('imgsearch.showNSFW') === '1');
  if (!(await nsfwControl.textContent() || '').includes('NSFW: Visible')) {
    throw new Error('NSFW control does not show the visible state when enabled');
  }
  if (!(await nsfwControl.evaluate((element) => element.classList.contains('is-enabled')))) {
    throw new Error('NSFW control does not expose a distinct enabled state');
  }
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

  const mobilePage = await browser.newPage({
    viewport: { width: 390, height: 844 },
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
  });
  await mobilePage.goto(baseURL, { waitUntil: 'domcontentloaded' });
  const mobileGrid = mobilePage.locator('#gallery-grid');
  await mobileGrid.locator('.card').first().waitFor();
  const mobileColumns = await mobileGrid.evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').filter(Boolean).length);
  if (mobileColumns < 2) {
    throw new Error(`mobile gallery used ${mobileColumns} column(s), want at least 2`);
  }
  if (await mobileGrid.locator('.card').first().locator('.meta-foot').isVisible()) {
    throw new Error('mobile cards still show secondary metadata');
  }
  const mobileResultsLabel = (await mobilePage.locator('#tab-results span').first().textContent() || '').trim();
  if (mobileResultsLabel !== 'Results') {
    throw new Error(`mobile results tab label is ${JSON.stringify(mobileResultsLabel)}, want "Results"`);
  }
  const lightboxPadding = await mobilePage.locator('#image-lightbox').evaluate((element) => Number.parseFloat(getComputedStyle(element).paddingTop));
  if (lightboxPadding > 12) {
    throw new Error(`mobile lightbox padding is ${lightboxPadding}px, want at most 12px`);
  }
  await mobilePage.close();

  console.log('UI smoke checks passed');
} finally {
  if (browser) {
    await browser.close();
  }
  await new Promise((resolve) => server.close(resolve));
}
