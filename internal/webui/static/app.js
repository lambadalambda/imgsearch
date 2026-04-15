const state = {
  images: [],
  results: [],
  imagesTotal: 0,
  stats: null,
  galleryPage: 0,
  galleryPageSize: 24,
  activeTab: 'gallery',
};

const galleryGrid = document.getElementById('gallery-grid');
const galleryStatus = document.getElementById('gallery-status');
const galleryPrevButton = document.getElementById('gallery-prev');
const galleryNextButton = document.getElementById('gallery-next');
const galleryPageLabel = document.getElementById('gallery-page-label');
const galleryTabButton = document.getElementById('tab-gallery');
const galleryTabCount = document.getElementById('tab-gallery-count');
const galleryPanel = document.getElementById('gallery-panel');

const resultsGrid = document.getElementById('results-grid');
const resultsTabButton = document.getElementById('tab-results');
const resultsTabCount = document.getElementById('tab-results-count');
const resultsPanel = document.getElementById('results-panel');

const uploadModal = document.getElementById('upload-modal');
const uploadOpenButton = document.getElementById('upload-open');
const uploadCloseButton = document.getElementById('upload-close');
const uploadForm = document.getElementById('upload-form');
const uploadFile = document.getElementById('upload-file');
const uploadStatus = document.getElementById('upload-status');

const searchForm = document.getElementById('text-search-form');
const textQuery = document.getElementById('text-query');
const negativeQuery = document.getElementById('negative-query');
const searchStatus = document.getElementById('search-status');

const refreshImagesButton = document.getElementById('refresh-images');
const clearResultsButton = document.getElementById('clear-results');
const refreshStatsButton = document.getElementById('refresh-stats');
const retryFailedButton = document.getElementById('retry-failed');

const statsSummary = document.getElementById('stats-summary');
const statsDetail = document.getElementById('stats-detail');
const statsProgress = document.getElementById('stats-progress');
const statsFailures = document.getElementById('stats-failures');

const lightbox = document.getElementById('image-lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const lightboxCaption = document.getElementById('lightbox-caption');
const lightboxCloseButton = document.getElementById('lightbox-close');

const liveConnectionBadge = document.getElementById('live-connection');
const liveConnectionLabel = document.getElementById('live-connection-label');
const liveAnnouncer = document.getElementById('live-announcer');
const shell = document.querySelector('.shell');

let lastFocusedElement = null;
let liveSocket = null;
let liveReconnectTimer = null;
let pollingTimer = null;
let receivedLiveSnapshot = false;
let liveReconnectDelayMs = 1500;
let imagesRequestToken = 0;
let statsRequestToken = 0;
const liveReconnectDelayMaxMs = 30000;

function setStatus(target, message, kind) {
  if (!target) {
    return;
  }
  const nextKind = kind || '';
  if (target.textContent === message && (target.dataset.kind || '') === nextKind) {
    return;
  }
  target.textContent = message;
  target.dataset.kind = nextKind;
}

function setLiveConnectionState(stateName, label) {
  if (liveConnectionBadge) {
    liveConnectionBadge.dataset.state = stateName;
  }
  if (liveConnectionLabel && label) {
    liveConnectionLabel.textContent = label;
  }
}

function announceLiveChange(message) {
  if (!liveAnnouncer || !message) {
    return;
  }
  if (liveAnnouncer.textContent === message) {
    return;
  }
  liveAnnouncer.textContent = message;
}

function clearLiveReconnectTimer() {
  if (liveReconnectTimer === null) {
    return;
  }
  window.clearTimeout(liveReconnectTimer);
  liveReconnectTimer = null;
}

function isLiveSocketActive() {
  return Boolean(liveSocket && liveSocket.readyState === WebSocket.OPEN && receivedLiveSnapshot);
}

function escapeHTML(value) {
  const raw = String(value || '');
  return raw
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function toMediaURL(storagePath) {
  if (!storagePath) {
    return '';
  }
  let normalized = storagePath.replace(/^\/+/, '');
  if (normalized.startsWith('images/')) {
    normalized = normalized.slice('images/'.length);
  }
  const encoded = normalized
    .split('/')
    .filter((segment) => segment.length > 0)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  return `/media/${encoded}`;
}

function stateClass(indexState) {
  if (indexState === 'done') {
    return 'state done';
  }
  if (indexState === 'failed') {
    return 'state failed';
  }
  if (indexState === 'leased') {
    return 'state leased';
  }
  return 'state pending';
}

function totalGalleryPages() {
  return Math.max(1, Math.ceil(state.imagesTotal / state.galleryPageSize));
}

function updateTabCounts() {
  galleryTabCount.textContent = String(state.imagesTotal || 0);
  resultsTabCount.textContent = String(state.results.length || 0);
  resultsTabButton.disabled = state.results.length === 0 && state.activeTab !== 'results';
}

function renderPagination() {
  const pages = totalGalleryPages();
  const pageNumber = state.galleryPage + 1;
  galleryPageLabel.textContent = `Page ${pageNumber} of ${pages}`;
  galleryPrevButton.disabled = state.galleryPage <= 0;
  galleryNextButton.disabled = state.galleryPage >= pages - 1;
}

function setActiveTab(name) {
  state.activeTab = name === 'results' ? 'results' : 'gallery';
  const galleryActive = state.activeTab === 'gallery';

  galleryTabButton.setAttribute('aria-selected', galleryActive ? 'true' : 'false');
  resultsTabButton.setAttribute('aria-selected', galleryActive ? 'false' : 'true');
  galleryPanel.hidden = !galleryActive;
  resultsPanel.hidden = galleryActive;
  updateTabCounts();
  clearResultsButton.disabled = state.results.length === 0;
}

function shouldShowDescriptionToggle(description) {
  if (!description) {
    return false;
  }
  return description.length > 160 || description.split(/\s+/).length > 28;
}

function cardMarkup(item, mode) {
  const safeName = escapeHTML(item.original_name);
  const safePath = escapeHTML(item.storage_path);
  const safeImageURL = escapeHTML(toMediaURL(item.storage_path));
  const description = typeof item.description === 'string' ? item.description.trim() : '';
  const tags = Array.isArray(item.tags) ? item.tags.filter((tag) => typeof tag === 'string' && tag.trim() !== '').slice(0, 10) : [];
  const descriptionID = `description-${mode}-${Number(item.image_id) || 0}`;
  const canExpandDescription = shouldShowDescriptionToggle(description);
  const status = item.index_state || 'done';
  const safeStatus = escapeHTML(status);
  const score = typeof item.distance === 'number' && !item.is_anchor ? `<p class="distance">distance ${item.distance.toFixed(4)}</p>` : '';
  const canSearchSimilar = status === 'done';
  const actionLabel = mode === 'result' ? (item.is_anchor ? 'Anchor image' : 'Use as anchor') : 'Find similar';
  const disabled = canSearchSimilar ? '' : 'disabled';
  const title = canSearchSimilar ? '' : 'title="Available after indexing finishes"';
  const anchorBadge = item.is_anchor ? '<p class="state anchor">anchor</p>' : '';
  const descriptionMarkup = description
    ? `<div class="description-wrap" data-expanded="false">
        <p id="${descriptionID}" class="description">${escapeHTML(description)}</p>
        ${canExpandDescription ? `<button type="button" class="description-toggle" aria-expanded="false" aria-controls="${descriptionID}">Show more</button>` : ''}
      </div>`
    : '';
  const tagsMarkup = tags.length > 0
    ? `<ul class="tag-list">${tags.map((tag) => `<li class="tag-chip${tag.trim().toLowerCase() === 'nsfw' ? ' tag-chip-nsfw' : ''}">${escapeHTML(tag)}</li>`).join('')}</ul>`
    : '';

  return `
    <article class="card">
      <button
        type="button"
        class="thumb-button"
        data-lightbox-src="${safeImageURL}"
        data-lightbox-caption="${safeName}"
        aria-label="Open full image for ${safeName}"
      >
        <img src="${safeImageURL}" alt="${safeName}" loading="lazy" />
      </button>
      <div class="meta">
        <h3>${safeName}</h3>
        <p class="path">${safePath}</p>
        ${descriptionMarkup}
        ${tagsMarkup}
        <p class="${stateClass(status)}">${safeStatus}</p>
        ${anchorBadge}
        ${score}
        <button class="ghost similar-action" data-image-id="${item.image_id}" ${disabled} ${title}>${actionLabel}</button>
      </div>
    </article>
  `;
}

function renderGallery() {
  updateTabCounts();
  renderPagination();

  if (state.images.length === 0) {
    if (state.imagesTotal > 0) {
      galleryGrid.innerHTML = '<p class="empty">No images are visible on this page yet. Move back toward newer pages.</p>';
    } else {
      galleryGrid.innerHTML = '<p class="empty">No images here yet. Import or upload something to start building the archive.</p>';
    }
    return;
  }

  galleryGrid.innerHTML = state.images.map((item) => cardMarkup(item, 'gallery')).join('');
}

function renderResults() {
  updateTabCounts();
  clearResultsButton.disabled = state.results.length === 0;

  if (state.results.length === 0) {
    resultsGrid.innerHTML = '<p class="empty">Run a text search or similar-image search to populate this view.</p>';
    return;
  }

  resultsGrid.innerHTML = state.results.map((item) => cardMarkup(item, 'result')).join('');
}

function renderStats() {
  const payload = state.stats;
  if (!payload || !payload.queue) {
    setStatus(statsSummary, 'No queue stats available yet.', 'info');
    statsDetail.textContent = '';
    statsProgress.style.width = '0%';
    statsFailures.innerHTML = '';
    retryFailedButton.disabled = true;
    return;
  }

  const queue = payload.queue || {};
  const total = Number(queue.total || 0);
  const done = Number(queue.done || 0);
  const failed = Number(queue.failed || 0);
  const missing = Number(queue.missing || 0);
  const annotationsMissing = Number(queue.annotations_missing || 0);
  const pending = Number(queue.pending || 0);
  const leased = Number(queue.leased || 0);
  const completed = done + failed;
  const pct = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0;

  statsProgress.style.width = `${pct}%`;
  setStatus(
    statsSummary,
    `Indexed ${done} of ${total}. Failed ${failed}. Missing ${missing}. Annotation gaps ${annotationsMissing}.`,
    failed > 0 ? 'error' : (missing > 0 || annotationsMissing > 0) ? 'info' : 'success',
  );

  const imagesTotal = Number(payload.images_total || state.imagesTotal || 0);
  statsDetail.textContent = `Library ${imagesTotal} images. Pending ${pending}. In progress ${leased}.`;
  retryFailedButton.disabled = failed === 0 && missing === 0 && annotationsMissing === 0;

  const failures = payload.recent_failures || [];
  if (failures.length === 0) {
    statsFailures.innerHTML = '';
    return;
  }

  statsFailures.innerHTML = failures
    .slice(0, 2)
    .map((item) => {
      const safeName = escapeHTML(item.original_name || `image #${item.image_id}`);
      const safeError = escapeHTML(item.last_error || 'unknown error');
      return `<article class="failure-item"><p><strong>${safeName}</strong> (attempt ${Number(item.attempts || 0)})</p><p>${safeError}</p></article>`;
    })
    .join('');
}

function setOverlayState(kind, active) {
  if (kind === 'lightbox') {
    document.body.classList.toggle('lightbox-open', active);
  }
  if (kind === 'modal') {
    document.body.classList.toggle('modal-open', active);
  }

  if (!shell) {
    return;
  }

  const hasOverlay = Boolean((uploadModal && !uploadModal.hidden) || (lightbox && !lightbox.hidden));
  if (hasOverlay) {
    shell.setAttribute('inert', '');
    shell.setAttribute('aria-hidden', 'true');
  } else {
    shell.removeAttribute('inert');
    shell.removeAttribute('aria-hidden');
  }
}

function focusableElements(container) {
  if (!container) {
    return [];
  }
  return Array.from(
    container.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => !element.hasAttribute('hidden'));
}

function trapFocus(event, container) {
  if (event.key !== 'Tab') {
    return;
  }

  const items = focusableElements(container);
  if (items.length === 0) {
    event.preventDefault();
    return;
  }

  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;

  if (event.shiftKey && active === first) {
    event.preventDefault();
    last.focus();
    return;
  }

  if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
}

function openLightbox(source, caption) {
  if (!lightbox || !lightboxImage || !lightboxCaption || !source) {
    return;
  }

  lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  lightboxImage.src = source;
  lightboxImage.alt = caption || 'Full-size image preview';
  lightboxCaption.textContent = caption || '';
  lightbox.setAttribute('aria-label', caption || 'Image preview');
  lightbox.hidden = false;
  setOverlayState('lightbox', true);

  if (lightboxCloseButton) {
    lightboxCloseButton.focus();
  }
}

function closeLightbox() {
  if (!lightbox || !lightboxImage || !lightboxCaption || lightbox.hidden) {
    return;
  }

  lightbox.hidden = true;
  lightboxImage.removeAttribute('src');
  lightboxCaption.textContent = '';
  lightbox.removeAttribute('aria-label');
  setOverlayState('lightbox', false);

  if (lastFocusedElement) {
    lastFocusedElement.focus();
  }
  lastFocusedElement = null;
}

function openUploadModal() {
  if (!uploadModal) {
    return;
  }

  lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  uploadModal.hidden = false;
  setOverlayState('modal', true);
  if (uploadFile) {
    uploadFile.focus();
  }
}

function closeUploadModal() {
  if (!uploadModal || uploadModal.hidden) {
    return;
  }

  uploadModal.hidden = true;
  setOverlayState('modal', false);
  if (lastFocusedElement) {
    lastFocusedElement.focus();
  }
  lastFocusedElement = null;
}

async function loadStats() {
  const requestToken = ++statsRequestToken;
  const response = await fetch('/api/stats');
  const payload = await response.json();
  if (requestToken !== statsRequestToken) {
    return;
  }
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load queue stats');
  }

  state.stats = payload;
  renderStats();
}

async function loadImages() {
  const offset = state.galleryPage * state.galleryPageSize;
  const requestToken = ++imagesRequestToken;
  setStatus(galleryStatus, 'Loading gallery...', 'info');

  const response = await fetch('/api/images' + `?limit=${state.galleryPageSize}&offset=${offset}`);
  const payload = await response.json();
  if (requestToken !== imagesRequestToken) {
    return;
  }
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load images');
  }

  const nextImages = payload.images || [];
  const nextTotal = Number(payload.total || nextImages.length);
  const maxPage = Math.max(0, Math.ceil(nextTotal / state.galleryPageSize) - 1);
  if (nextImages.length === 0 && nextTotal > 0 && state.galleryPage > maxPage) {
    state.galleryPage = maxPage;
    return loadImages();
  }

  state.images = nextImages;
  state.imagesTotal = nextTotal;
  renderGallery();

  const start = state.imagesTotal === 0 ? 0 : offset + 1;
  const end = offset + state.images.length;
  setStatus(galleryStatus, `Showing ${start}-${end} of ${state.imagesTotal} image(s), newest first.`, 'success');
}

async function runTextSearch(query, negative) {
  const params = new URLSearchParams({ q: query, limit: '24' });
  if (negative) {
    params.set('neg', negative);
  }
  const response = await fetch(`/api/search/text?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Text search failed');
  }

  const imageByID = new Map(state.images.map((item) => [item.image_id, item]));
  state.results = (payload.results || []).map((item) => {
    const image = imageByID.get(item.image_id) || {};
    return {
      ...image,
      ...item,
      index_state: image.index_state || 'done',
      is_anchor: false,
    };
  });
  renderResults();
  setActiveTab('results');
  setStatus(searchStatus, `Text search returned ${state.results.length} result(s).`, 'success');
}

async function runSimilarSearch(imageID, anchorHint) {
  const params = new URLSearchParams({ image_id: String(imageID), limit: '24' });
  const response = await fetch(`/api/search/similar?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Similar search failed');
  }

  const imageByID = new Map(state.images.map((item) => [item.image_id, item]));
  const mergedResults = (payload.results || []).map((item) => {
    const image = imageByID.get(item.image_id) || {};
    return {
      ...image,
      ...item,
      index_state: image.index_state || 'done',
    };
  });

  const existingAnchor = mergedResults.find((item) => Number(item.image_id) === imageID) || null;
  const anchorSource = imageByID.get(imageID) || anchorHint || existingAnchor;

  let anchorResult = null;
  if (anchorSource) {
    anchorResult = {
      ...anchorSource,
      image_id: imageID,
      index_state: anchorSource.index_state || 'done',
      distance: typeof anchorSource.distance === 'number' ? anchorSource.distance : 0,
      is_anchor: true,
    };
  }

  const nonAnchorResults = mergedResults
    .filter((item) => Number(item.image_id) !== imageID)
    .map((item) => ({
      ...item,
      is_anchor: false,
    }));

  state.results = anchorResult ? [anchorResult, ...nonAnchorResults] : nonAnchorResults;
  renderResults();
  setActiveTab('results');
  setStatus(searchStatus, `Similar search returned ${state.results.length} result(s).`, 'success');
}

function imagesDiffer(currentImages, nextImages) {
  if (!Array.isArray(currentImages) || !Array.isArray(nextImages)) {
    return true;
  }
  if (currentImages.length !== nextImages.length) {
    return true;
  }
  for (let i = 0; i < currentImages.length; i += 1) {
    const current = currentImages[i] || {};
    const next = nextImages[i] || {};
    if (Number(current.image_id || 0) !== Number(next.image_id || 0)) {
      return true;
    }
    if ((current.index_state || '') !== (next.index_state || '')) {
      return true;
    }
    if ((current.storage_path || '') !== (next.storage_path || '')) {
      return true;
    }
  }
  return false;
}

function failuresDiffer(currentFailures, nextFailures) {
  const current = Array.isArray(currentFailures) ? currentFailures : [];
  const next = Array.isArray(nextFailures) ? nextFailures : [];
  if (current.length !== next.length) {
    return true;
  }
  for (let i = 0; i < current.length; i += 1) {
    const left = current[i] || {};
    const right = next[i] || {};
    if (Number(left.job_id || 0) !== Number(right.job_id || 0)) {
      return true;
    }
    if (Number(left.attempts || 0) !== Number(right.attempts || 0)) {
      return true;
    }
    if ((left.last_error || '') !== (right.last_error || '')) {
      return true;
    }
  }
  return false;
}

function statsDiffer(currentStats, nextStats) {
  if (!currentStats || !nextStats) {
    return true;
  }
  const currentQueue = currentStats.queue || {};
  const nextQueue = nextStats.queue || {};
  const queueKeys = ['total', 'tracked', 'missing', 'annotations_missing', 'pending', 'leased', 'done', 'failed'];
  for (const key of queueKeys) {
    if (Number(currentQueue[key] || 0) !== Number(nextQueue[key] || 0)) {
      return true;
    }
  }
  if (Number(currentStats.images_total || 0) !== Number(nextStats.images_total || 0)) {
    return true;
  }
  return failuresDiffer(currentStats.recent_failures, nextStats.recent_failures);
}

function applyLiveSnapshot(payload) {
  if (!payload || payload.type !== 'snapshot') {
    return false;
  }

  let changed = false;

  if (payload.stats && statsDiffer(state.stats, payload.stats)) {
    state.stats = payload.stats;
    state.imagesTotal = Number(payload.stats.images_total || state.imagesTotal || 0);
    renderStats();
    renderGallery();
    changed = true;
  }

  if (payload.images && state.galleryPage === 0) {
    const nextImages = (payload.images.images || []).slice(0, state.galleryPageSize);
    const nextTotal = Number(payload.images.total ?? nextImages.length);
    if (imagesDiffer(state.images, nextImages) || Number(state.imagesTotal || 0) !== nextTotal) {
      state.images = nextImages;
      state.imagesTotal = nextTotal;
      renderGallery();
      if (state.imagesTotal === 0) {
        setStatus(galleryStatus, 'No images here yet. Import or upload something to start building the archive.', 'info');
      } else {
        const end = state.images.length;
        setStatus(galleryStatus, `Showing 1-${end} of ${state.imagesTotal} image(s), newest first.`, 'success');
      }
      changed = true;
    }
  }

  receivedLiveSnapshot = true;
  stopPollingFallback();
  liveReconnectDelayMs = 1500;
  setLiveConnectionState('live', 'Live updates active');
  if (changed) {
    announceLiveChange('Library status updated.');
  }
  return true;
}

function startPollingFallback() {
  setLiveConnectionState('polling', 'Polling every 5s');

  if (pollingTimer !== null) {
    return;
  }

  pollingTimer = window.setInterval(() => {
    loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Gallery refresh failed', 'error'));
    loadStats().catch((err) => setStatus(statsSummary, err.message || 'Status refresh failed', 'error'));
  }, 5000);

  if (!receivedLiveSnapshot) {
    loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Initial load failed', 'error'));
    loadStats().catch((err) => setStatus(statsSummary, err.message || 'Initial status load failed', 'error'));
  }
}

function stopPollingFallback() {
  if (pollingTimer === null) {
    return;
  }
  window.clearInterval(pollingTimer);
  pollingTimer = null;
}

function connectLiveUpdates() {
  clearLiveReconnectTimer();
  receivedLiveSnapshot = false;
  setLiveConnectionState('connecting', 'Connecting live updates...');

  if (typeof window.WebSocket !== 'function') {
    startPollingFallback();
    return;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${window.location.host}/api/live`);
  liveSocket = socket;

  socket.addEventListener('open', () => {
    liveReconnectDelayMs = 1500;
    setLiveConnectionState('connecting', 'Connected, waiting for first snapshot...');
  });

  socket.addEventListener('message', (event) => {
    try {
      if (typeof event.data !== 'string') {
        return;
      }
      const payload = JSON.parse(event.data);
      if (payload && payload.type === 'error') {
        setStatus(statsSummary, payload.error || 'Live update failed', 'error');
        return;
      }
      applyLiveSnapshot(payload);
    } catch (_) {}
  });

  socket.addEventListener('error', () => {
    setLiveConnectionState('reconnecting', 'Live connection interrupted. Reconnecting...');
    socket.close();
  });

  socket.addEventListener('close', () => {
    if (liveSocket === socket) {
      liveSocket = null;
    }
    setLiveConnectionState('reconnecting', 'Live connection lost. Reconnecting...');
    startPollingFallback();
    clearLiveReconnectTimer();
    const reconnectDelay = liveReconnectDelayMs;
    liveReconnectDelayMs = Math.min(liveReconnectDelayMs * 2, liveReconnectDelayMaxMs);
    liveReconnectTimer = window.setTimeout(() => {
      liveReconnectTimer = null;
      connectLiveUpdates();
    }, reconnectDelay);
  });
}

async function retryFailedJobs() {
  const response = await fetch('/api/jobs/retry-failed', { method: 'POST' });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Retry failed jobs request failed');
  }
  return {
    retried: Number(payload.retried || 0),
    enqueuedMissing: Number(payload.enqueued_missing || 0),
  };
}

function attachSimilarHandler(target) {
  target.addEventListener('click', async (event) => {
    const button = event.target.closest('.similar-action');
    if (!button || button.disabled) {
      return;
    }

    const imageID = Number(button.dataset.imageId);
    if (!Number.isFinite(imageID) || imageID <= 0) {
      setStatus(searchStatus, 'Could not resolve selected image id.', 'error');
      return;
    }

    setStatus(searchStatus, `Searching similar images for #${imageID}...`, 'info');
    const anchorHint =
      state.results.find((item) => Number(item.image_id) === imageID) ||
      state.images.find((item) => Number(item.image_id) === imageID) ||
      null;
    try {
      await runSimilarSearch(imageID, anchorHint);
    } catch (err) {
      setStatus(searchStatus, err.message || 'Similar search failed', 'error');
    }
  });
}

function attachLightboxHandler(target) {
  target.addEventListener('click', (event) => {
    const trigger = event.target.closest('.thumb-button');
    if (!trigger || !target.contains(trigger)) {
      return;
    }

    const source = trigger.dataset.lightboxSrc || '';
    const caption = trigger.dataset.lightboxCaption || '';
    openLightbox(source, caption);
  });
}

function attachDescriptionToggleHandler(target) {
  target.addEventListener('click', (event) => {
    const button = event.target.closest('.description-toggle');
    if (!button || !target.contains(button)) {
      return;
    }

    const wrap = button.closest('.description-wrap');
    if (!wrap) {
      return;
    }

    const expanded = wrap.dataset.expanded === 'true';
    wrap.dataset.expanded = expanded ? 'false' : 'true';
    button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
    button.textContent = expanded ? 'Show more' : 'Show less';
  });
}

galleryTabButton.addEventListener('click', () => setActiveTab('gallery'));
resultsTabButton.addEventListener('click', () => setActiveTab('results'));

galleryPrevButton.addEventListener('click', async () => {
  if (state.galleryPage <= 0) {
    return;
  }
  state.galleryPage -= 1;
  try {
    await loadImages();
  } catch (err) {
    setStatus(galleryStatus, err.message || 'Could not load previous page', 'error');
  }
});

galleryNextButton.addEventListener('click', async () => {
  if (state.galleryPage >= totalGalleryPages() - 1) {
    return;
  }
  state.galleryPage += 1;
  try {
    await loadImages();
  } catch (err) {
    setStatus(galleryStatus, err.message || 'Could not load next page', 'error');
  }
});

uploadOpenButton.addEventListener('click', openUploadModal);
uploadCloseButton.addEventListener('click', closeUploadModal);

uploadModal.addEventListener('click', (event) => {
  if (event.target === uploadModal) {
    closeUploadModal();
  }
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = uploadFile.files ? Array.from(uploadFile.files) : [];
  if (files.length === 0) {
    setStatus(uploadStatus, 'Choose one or more JPEG, PNG, WEBP, or AVIF files first.', 'error');
    return;
  }

  const data = new FormData();
  files.forEach((file) => {
    data.append('file', file, file.name);
  });

  const uploadLabel = files.length === 1 ? files[0].name : `${files.length} files`;
  setStatus(uploadStatus, `Uploading ${uploadLabel}...`, 'info');
  try {
    const response = await fetch('/api/upload', { method: 'POST', body: data });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'Upload failed');
    }

    const created = Number(payload.created) || 0;
    const duplicates = Number(payload.duplicates) || 0;
    let message = '';
    if (files.length === 1) {
      message = duplicates > 0 && created === 0 ? `${files[0].name} already exists.` : `${files[0].name} queued for indexing.`;
    } else {
      const parts = [];
      if (created > 0) {
        parts.push(`${created} queued`);
      }
      if (duplicates > 0) {
        parts.push(`${duplicates} already existed`);
      }
      message = parts.length > 0 ? `Upload complete: ${parts.join(', ')}.` : 'Upload complete.';
    }

    setStatus(uploadStatus, message, 'success');
    uploadForm.reset();
    state.galleryPage = 0;
    setActiveTab('gallery');

    if (!isLiveSocketActive()) {
      await loadImages();
      await loadStats();
    }

    window.setTimeout(() => {
      closeUploadModal();
    }, 450);
  } catch (err) {
    setStatus(uploadStatus, err.message || 'Upload failed', 'error');
  }
});

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const q = textQuery.value.trim();
  const neg = negativeQuery ? negativeQuery.value.trim() : '';
  if (!q) {
    setStatus(searchStatus, 'Enter a search phrase first.', 'error');
    return;
  }

  if (neg) {
    setStatus(searchStatus, `Searching for "${q}" excluding "${neg}"...`, 'info');
  } else {
    setStatus(searchStatus, `Searching for "${q}"...`, 'info');
  }

  try {
    await runTextSearch(q, neg);
  } catch (err) {
    setStatus(searchStatus, err.message || 'Text search failed', 'error');
  }
});

refreshImagesButton.addEventListener('click', () => {
  loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Refresh failed', 'error'));
});

refreshStatsButton.addEventListener('click', () => {
  loadStats().catch((err) => setStatus(statsSummary, err.message || 'Status refresh failed', 'error'));
});

retryFailedButton.addEventListener('click', async () => {
  retryFailedButton.disabled = true;
  setStatus(statsSummary, 'Retrying failed jobs and queueing missing work...', 'info');
  try {
    const result = await retryFailedJobs();
    await loadStats();
    await loadImages();
    const hadWork = result.retried > 0 || result.enqueuedMissing > 0;
    setStatus(
      statsSummary,
      `Requeued ${result.retried} failed job(s), queued ${result.enqueuedMissing} missing job(s).`,
      hadWork ? 'success' : 'info',
    );
  } catch (err) {
    setStatus(statsSummary, err.message || 'Retry failed jobs failed', 'error');
  } finally {
    const failed = state.stats && state.stats.queue ? Number(state.stats.queue.failed || 0) : 0;
    const missing = state.stats && state.stats.queue ? Number(state.stats.queue.missing || 0) : 0;
    const annotationsMissing = state.stats && state.stats.queue ? Number(state.stats.queue.annotations_missing || 0) : 0;
    retryFailedButton.disabled = failed === 0 && missing === 0 && annotationsMissing === 0;
  }
});

clearResultsButton.addEventListener('click', () => {
  state.results = [];
  renderResults();
  setActiveTab('gallery');
  setStatus(searchStatus, 'Search results cleared.', 'info');
});

attachSimilarHandler(galleryGrid);
attachSimilarHandler(resultsGrid);
attachLightboxHandler(galleryGrid);
attachLightboxHandler(resultsGrid);
attachDescriptionToggleHandler(galleryGrid);
attachDescriptionToggleHandler(resultsGrid);

if (lightboxCloseButton) {
  lightboxCloseButton.addEventListener('click', closeLightbox);
}

if (lightbox) {
  lightbox.addEventListener('click', (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });
}

document.addEventListener('keydown', (event) => {
  if (lightbox && !lightbox.hidden) {
    trapFocus(event, lightbox);
  } else if (uploadModal && !uploadModal.hidden) {
    trapFocus(event, uploadModal);
  }

  if (event.key !== 'Escape') {
    return;
  }

  if (lightbox && !lightbox.hidden) {
    closeLightbox();
    return;
  }

  if (uploadModal && !uploadModal.hidden) {
    closeUploadModal();
  }
});

renderGallery();
renderResults();
renderStats();
setActiveTab('gallery');

loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Initial load failed', 'error'));
loadStats().catch((err) => setStatus(statsSummary, err.message || 'Initial status load failed', 'error'));
connectLiveUpdates();
