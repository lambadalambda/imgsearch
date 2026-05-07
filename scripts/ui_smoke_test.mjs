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
let releaseLandscapePreview;
const landscapePreviewRelease = new Promise((resolve) => {
  releaseLandscapePreview = resolve;
});

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

async function assertPolishBasics(page) {
  const opsGlyph = (await page.locator('.ops-menu-trigger [aria-hidden="true"]').textContent() || '').trim();
  if (opsGlyph === '...' || opsGlyph !== '…') {
    throw new Error(`ops menu trigger uses ${JSON.stringify(opsGlyph)}, want polished ellipsis`);
  }

  const tokens = await page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    return {
      focus: style.getPropertyValue('--focus').trim(),
      leased: style.getPropertyValue('--leased').trim(),
      muted: style.getPropertyValue('--muted').trim(),
    };
  });
  if (!tokens.focus || tokens.focus === tokens.leased) {
    throw new Error('focus token is missing or still reuses processing-state color');
  }
  if (tokens.muted === '#6b655b') {
    throw new Error('muted text color was not darkened for contrast');
  }
}

async function assertLoadingAndEmptyStates(browser, baseURL) {
  const loadingPage = await browser.newPage();
  let releaseImages;
  const releasePromise = new Promise((resolve) => {
    releaseImages = resolve;
  });
  await loadingPage.route('**/api/images*', async (route) => {
    await releasePromise;
    await route.continue();
  });
  await loadingPage.goto(baseURL, { waitUntil: 'domcontentloaded' });
  await loadingPage.locator('#gallery-grid .skeleton-card').first().waitFor({ timeout: 1200 });
  releaseImages();
  await loadingPage.locator('#gallery-grid .card').first().waitFor();
  await loadingPage.close();

  const emptyPage = await browser.newPage();
  await emptyPage.route('**/api/images*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ images: [], total: 0 }),
    });
  });
  await emptyPage.goto(baseURL, { waitUntil: 'domcontentloaded' });
  const emptyState = emptyPage.locator('#gallery-grid .empty-state');
  await emptyState.waitFor();
  const emptyText = await emptyState.textContent() || '';
  if (!emptyText.includes('Upload media')) {
    throw new Error(`gallery empty state is not actionable: ${JSON.stringify(emptyText)}`);
  }
  await emptyPage.close();
}

async function assertReducedMotion(browser, baseURL) {
  const context = await browser.newContext({ reducedMotion: 'reduce' });
  const motionPage = await context.newPage();
  await motionPage.goto(baseURL, { waitUntil: 'domcontentloaded' });
  const searchButton = motionPage.getByRole('button', { name: 'Search' });
  await searchButton.hover();
  const transform = await searchButton.evaluate((element) => getComputedStyle(element).transform);
  if (transform !== 'none') {
    throw new Error(`reduced-motion hover transform is ${transform}, want none`);
  }
  await context.close();
}

async function assertOpenLayout(page) {
  const workspaceStyle = await page.locator('.workspace').evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      borderTopWidth: Number.parseFloat(style.borderTopWidth),
      boxShadow: style.boxShadow,
    };
  });
  if (workspaceStyle.borderTopWidth > 0 || workspaceStyle.boxShadow !== 'none') {
    throw new Error(`workspace still uses heavy panel chrome: ${JSON.stringify(workspaceStyle)}`);
  }

  const card = page.locator('#gallery-grid .card').first();
  const cardBox = await card.boundingBox();
  if (!cardBox) {
    throw new Error('gallery card did not render for open layout check');
  }
  if (cardBox.height > 330) {
    throw new Error(`gallery card is ${Math.round(cardBox.height)}px tall, want a more compact open layout`);
  }

  const thumbBox = await card.locator('.thumb-wrap').boundingBox();
  if (!thumbBox) {
    throw new Error('gallery thumbnail did not render for open layout check');
  }
  const ratio = thumbBox.width / thumbBox.height;
  if (ratio < 1.68 || ratio > 1.9) {
    throw new Error(`gallery thumbnail ratio is ${ratio.toFixed(2)}, want compact 16:9 browsing cards`);
  }

  const visibleTags = await card.locator('.meta-tags .tag-chip').evaluateAll((nodes) => nodes.map((node) => (node.textContent || '').trim()));
  if (!visibleTags.includes('outdoor portrait') || visibleTags.some((tag) => tag.startsWith('+'))) {
    throw new Error(`card tags are still collapsed instead of readable: ${JSON.stringify(visibleTags)}`);
  }

  const titleFont = await card.locator('.meta h3').evaluate((element) => getComputedStyle(element).fontFamily);
  if (/Iowan|Palatino/i.test(titleFont)) {
    throw new Error(`card titles still use dense display serif typography: ${titleFont}`);
  }

  await card.hover();
  await assertCardActionsFit(card, 'desktop');
  if (await card.locator('.card-detail-overlay').isVisible()) {
    throw new Error('open layout still shows a hover detail overlay over compact cards');
  }
  const longCaptionCard = page.locator('#gallery-grid .card').nth(1);
  await longCaptionCard.locator('.thumb-button').click();
  await page.locator('#image-lightbox:not([hidden])').waitFor();
  await assertLightboxImageCentered(page);
  await page.getByRole('button', { name: 'Close image preview' }).click();
  await page.locator('#image-lightbox').waitFor({ state: 'hidden' });
}

async function assertLightboxImageCentered(page) {
  const image = page.locator('#lightbox-image');
  await image.waitFor();
  await image.evaluate((element) => {
    if (element.complete && element.naturalWidth > 0) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      element.addEventListener('load', resolve, { once: true });
      element.addEventListener('error', resolve, { once: true });
    });
  });

  const boxes = await page.locator('#image-lightbox').evaluate((lightbox) => {
    const imageElement = lightbox.querySelector('#lightbox-image');
    const overlayRect = lightbox.getBoundingClientRect();
    const imageRect = imageElement.getBoundingClientRect();
    return {
      overlayCenter: overlayRect.left + overlayRect.width / 2,
      imageCenter: imageRect.left + imageRect.width / 2,
      imageWidth: imageRect.width,
      imageHeight: imageRect.height,
    };
  });
  if (boxes.imageWidth <= 0 || boxes.imageHeight <= 0) {
    throw new Error(`lightbox image did not render with a measurable size: ${JSON.stringify(boxes)}`);
  }
  const centerOffset = Math.abs(boxes.overlayCenter - boxes.imageCenter);
  if (centerOffset > 2) {
    throw new Error(`lightbox image is ${Math.round(centerOffset)}px off-center: ${JSON.stringify(boxes)}`);
  }
}

async function assertOpenVideoLayout(page) {
  const videoColumns = await page.locator('#videos-grid').evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').filter(Boolean).length);
  if (videoColumns !== 3) {
    throw new Error(`desktop videos grid used ${videoColumns} columns, want 3 to avoid crowding`);
  }

  const videoCard = page.locator('#videos-grid .card').first();
  await videoCard.waitFor();
  const cardBox = await videoCard.boundingBox();
  if (!cardBox) {
    throw new Error('video card did not render for open layout check');
  }
  if (cardBox.height > 320) {
    throw new Error(`video card is ${Math.round(cardBox.height)}px tall, want compact video browsing cards`);
  }

  const badgeText = (await videoCard.locator('.thumb-video-badge').first().textContent() || '').trim();
  if (badgeText !== '0:01') {
    throw new Error(`video badge is ${JSON.stringify(badgeText)}, want duration at a glance`);
  }

  const portraitThumb = videoCard.locator('.thumb-wrap');
  await page.waitForFunction(() => document.querySelector('#videos-grid .card .thumb-wrap')?.classList.contains('thumb-wrap-portrait-video'));
  if (!(await portraitThumb.evaluate((element) => element.classList.contains('thumb-wrap-portrait-video')))) {
    throw new Error('portrait video thumbnail is not using portrait backdrop treatment');
  }
  if (await portraitThumb.locator('.thumb-backdrop').count() !== 1) {
    throw new Error('portrait video thumbnail is missing a blurred backdrop image');
  }
  const portraitStyles = await portraitThumb.evaluate((element) => {
    const foreground = element.querySelector('.thumb-image');
    const backdrop = element.querySelector('.thumb-backdrop');
    const foregroundStyle = foreground ? getComputedStyle(foreground) : null;
    const backdropStyle = backdrop ? getComputedStyle(backdrop) : null;
    return {
      foregroundFit: foregroundStyle?.objectFit || '',
      foregroundBackground: foregroundStyle?.backgroundColor || '',
      backdropDisplay: backdropStyle?.display || '',
      backdropFilter: backdropStyle?.filter || '',
      backdropPosition: backdropStyle?.position || '',
      backdropAriaHidden: backdrop?.getAttribute('aria-hidden') || '',
    };
  });
  if (portraitStyles.foregroundFit !== 'contain' || portraitStyles.foregroundBackground !== 'rgba(0, 0, 0, 0)') {
    throw new Error(`portrait foreground is not a transparent contained frame: ${JSON.stringify(portraitStyles)}`);
  }
  if (portraitStyles.backdropDisplay === 'none' || !portraitStyles.backdropFilter.includes('blur') || portraitStyles.backdropPosition !== 'absolute' || portraitStyles.backdropAriaHidden !== 'true') {
    throw new Error(`portrait backdrop is not visible blurred presentation-only media: ${JSON.stringify(portraitStyles)}`);
  }

  const landscapeVideoCard = page.locator('#videos-grid .card').nth(1);
  const landscapeThumb = landscapeVideoCard.locator('.thumb-wrap');
  await landscapeThumb.waitFor();
  const landscapeProbeState = await landscapeThumb.evaluate((element) => ({
    isProbe: element.classList.contains('thumb-wrap-portrait-probe'),
    isPortrait: element.classList.contains('thumb-wrap-portrait-video'),
    backdropCount: element.querySelectorAll('.thumb-backdrop').length,
  }));
  releaseLandscapePreview();
  if (landscapeProbeState.isProbe || landscapeProbeState.isPortrait || landscapeProbeState.backdropCount !== 0) {
    throw new Error(`known landscape video thumbnail should not probe or render a duplicate backdrop: ${JSON.stringify(landscapeProbeState)}`);
  }
}

async function assertCardActionsFit(cardLocator, contextLabel) {
  const actionRects = await cardLocator.evaluate((card) => {
    const thumb = card.querySelector('.thumb-wrap')?.getBoundingClientRect();
    const actionBar = card.querySelector('.thumb-actions')?.getBoundingClientRect();
    const actions = Array.from(card.querySelectorAll('.thumb-actions .thumb-action'));
    return {
      thumb: thumb ? { left: thumb.left, top: thumb.top, right: thumb.right, bottom: thumb.bottom } : null,
      actionBar: actionBar ? { left: actionBar.left, top: actionBar.top, right: actionBar.right, bottom: actionBar.bottom } : null,
      actions: actions.map((action) => {
        const rect = action.getBoundingClientRect();
        return {
          label: (action.textContent || '').trim(),
          left: rect.left,
          top: rect.top,
          right: rect.right,
          bottom: rect.bottom,
        };
      }),
    };
  });
  if (!actionRects.thumb || !actionRects.actionBar || actionRects.actions.length < 4) {
    throw new Error(`${contextLabel} card actions are missing: ${JSON.stringify(actionRects)}`);
  }
  if (
    actionRects.actionBar.left < actionRects.thumb.left - 0.5
    || actionRects.actionBar.right > actionRects.thumb.right + 0.5
    || actionRects.actionBar.top < actionRects.thumb.top - 0.5
    || actionRects.actionBar.bottom > actionRects.thumb.bottom + 0.5
  ) {
    throw new Error(`${contextLabel} action bar overflows the thumbnail: ${JSON.stringify(actionRects.actionBar)}`);
  }
  const clipped = actionRects.actions.filter((rect) => (
    rect.left < actionRects.actionBar.left - 0.5
    || rect.right > actionRects.actionBar.right + 0.5
    || rect.top < actionRects.actionBar.top - 0.5
    || rect.bottom > actionRects.actionBar.bottom + 0.5
  ));
  if (clipped.length > 0) {
    throw new Error(`${contextLabel} card actions overflow the action bar: ${JSON.stringify(clipped)}`);
  }
}

async function assertMobileCardActionsFit(mobileGrid) {
  await assertCardActionsFit(mobileGrid.locator('.card').first(), 'mobile');
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
          description: id === 1
            ? 'Brown dog running across a sunny grass field.'
            : 'A screenshot of a long social media thread with enough generated description text to span the lightbox caption across a wide display.',
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
    if (url.pathname === '/media/images/image-2') {
      res.writeHead(200, { 'Content-Type': 'image/svg+xml' });
      res.end('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 1200"><rect width="640" height="1200" fill="#f9f7f0"/><rect x="84" y="96" width="472" height="1008" rx="24" fill="#ead1b2"/><circle cx="320" cy="330" r="132" fill="#3f6f8f"/><text x="320" y="720" text-anchor="middle" font-size="62" font-family="sans-serif" fill="#34261d">Long caption</text></svg>');
      return;
    }
    if (url.pathname === '/media/videos/portrait-preview-1.svg') {
      res.writeHead(200, { 'Content-Type': 'image/svg+xml' });
      res.end('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1920"><defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#4d8ad9"/><stop offset="1" stop-color="#f4c17a"/></linearGradient></defs><rect width="1080" height="1920" fill="url(#g)"/><circle cx="540" cy="780" r="260" fill="#fff8ee"/><rect x="260" y="1080" width="560" height="360" rx="120" fill="#f5eee8"/></svg>');
      return;
    }
    if (url.pathname === '/media/videos/landscape-preview-2.svg') {
      await landscapePreviewRelease;
      res.writeHead(200, { 'Content-Type': 'image/svg+xml' });
      res.end('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720"><rect width="1280" height="720" fill="#375170"/><circle cx="640" cy="360" r="190" fill="#e6bc65"/></svg>');
      return;
    }
    if (url.pathname === '/api/videos') {
      videoRequests += 1;
      const offset = Number(url.searchParams.get('offset') || 0);
      const limit = Number(url.searchParams.get('limit') || 24);
      const count = Math.max(0, Math.min(limit, 50 - offset));
      const videos = Array.from({ length: count }, (_, i) => {
        const id = offset + i + 1;
        const firstVideo = id === 1;
        return {
          video_id: id,
          media_type: 'video',
          original_name: `clip-${id}.mp4`,
          storage_path: `videos/clip-${id}`,
          preview_path: firstVideo ? 'videos/portrait-preview-1.svg' : (id === 2 ? 'videos/landscape-preview-2.svg' : ''),
          preview_width: firstVideo ? 1080 : (id === 2 ? 1280 : 0),
          preview_height: firstVideo ? 1920 : (id === 2 ? 720 : 0),
          mime_type: 'video/mp4',
          duration_ms: 1000,
          ...(firstVideo ? { width: 1920, height: 1080 } : { width: 320, height: 180 }),
          frame_count: 1,
          index_state: 'done',
          created_at: '2026-04-24T00:00:00Z',
        };
      });
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
  await assertPolishBasics(page);
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
  await assertOpenLayout(page);
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
  await assertOpenVideoLayout(page);

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
  await page.locator('#show-nsfw').check();
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

  const mobileUserAgent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1';
  const mobilePage = await browser.newPage({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    userAgent: mobileUserAgent,
  });
  await mobilePage.goto(baseURL, { waitUntil: 'domcontentloaded' });
  const mobileGrid = mobilePage.locator('#gallery-grid');
  await mobileGrid.locator('.card').first().waitFor();
  await assertMobileCardActionsFit(mobileGrid);
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

  const narrowMobilePage = await browser.newPage({
    viewport: { width: 340, height: 740 },
    isMobile: true,
    hasTouch: true,
    userAgent: mobileUserAgent,
  });
  await narrowMobilePage.goto(baseURL, { waitUntil: 'domcontentloaded' });
  const narrowMobileGrid = narrowMobilePage.locator('#gallery-grid');
  await narrowMobileGrid.locator('.card').first().waitFor();
  await assertMobileCardActionsFit(narrowMobileGrid);
  await narrowMobilePage.close();

  await assertLoadingAndEmptyStates(browser, baseURL);
  await assertReducedMotion(browser, baseURL);

  console.log('UI smoke checks passed');
} finally {
  if (browser) {
    await browser.close();
  }
  await new Promise((resolve) => server.close(resolve));
}
