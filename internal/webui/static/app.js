const state = {
  images: [],
  videos: [],
  results: [],
  imagesTotal: 0,
  videosTotal: 0,
  stats: null,
  galleryPage: 0,
  videosPage: 0,
  galleryPageSize: 24,
  videosPageSize: 24,
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

const videosGrid = document.getElementById('videos-grid');
const videosStatus = document.getElementById('videos-status');
const videosPrevButton = document.getElementById('videos-prev');
const videosNextButton = document.getElementById('videos-next');
const videosPageLabel = document.getElementById('videos-page-label');
const videosTabButton = document.getElementById('tab-videos');
const videosTabCount = document.getElementById('tab-videos-count');
const videosPanel = document.getElementById('videos-panel');

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
const videoModal = document.getElementById('video-modal');
const videoElement = document.getElementById('video-player');
const videoCaption = document.getElementById('video-caption');
const videoCloseButton = document.getElementById('video-close');

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
let videoPlayer = null;
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
  const normalized = storagePath.replace(/^\/+/, '');
  const encoded = normalized
    .split('/')
    .filter((segment) => segment.length > 0)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  return `/media/${encoded}`;
}

function formatTimestamp(timestampMs) {
  const totalSeconds = Math.max(0, Math.floor(Number(timestampMs || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${String(hours)}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes)}:${String(seconds).padStart(2, '0')}`;
}

function formatMatch(distance) {
  if (typeof distance !== 'number' || Number.isNaN(distance)) {
    return '';
  }
  const match = Math.max(0, Math.min(100, Math.round((1 - distance) * 100)));
  return `${match}% match`;
}

function humanizeIndexState(indexState) {
  if (indexState === 'leased') {
    return 'processing';
  }
  return indexState || 'done';
}

function videoMetaMarkup(item, mode) {
  if ((item.media_type || 'image') !== 'video') {
    return '';
  }

  if (mode === 'result' && Number(item.match_timestamp_ms) > 0) {
    return `<div class="video-meta"><p class="timestamp">best match at ${escapeHTML(formatTimestamp(item.match_timestamp_ms))}</p></div>`;
  }

  const details = [];
  if (Number(item.duration_ms) > 0) {
    details.push(`duration ${escapeHTML(formatTimestamp(item.duration_ms))}`);
  }
  if (details.length === 0) {
    return '';
  }

  return `<div class="video-meta"><p class="timestamp">${details.join(' • ')}</p></div>`;
}

function stateClass(indexState) {
  if (indexState === 'done') {
    return 'state done';
  }
  if (indexState === 'failed') {
    return 'state failed';
  }
  if (indexState === 'leased') {
    return 'state processing';
  }
  return 'state pending';
}

function totalPages(total, pageSize) {
  return Math.max(1, Math.ceil(total / pageSize));
}

function updateTabCounts() {
  galleryTabCount.textContent = String(state.imagesTotal || 0);
  videosTabCount.textContent = String(state.videosTotal || 0);
  resultsTabCount.textContent = String(state.results.length || 0);
  resultsTabButton.disabled = state.results.length === 0 && state.activeTab !== 'results';
}

function renderPagination() {
  const galleryPages = totalPages(state.imagesTotal, state.galleryPageSize);
  galleryPageLabel.textContent = `Page ${state.galleryPage + 1} of ${galleryPages}`;
  galleryPrevButton.disabled = state.galleryPage <= 0;
  galleryNextButton.disabled = state.galleryPage >= galleryPages - 1;

  const videoPages = totalPages(state.videosTotal, state.videosPageSize);
  videosPageLabel.textContent = `Page ${state.videosPage + 1} of ${videoPages}`;
  videosPrevButton.disabled = state.videosPage <= 0;
  videosNextButton.disabled = state.videosPage >= videoPages - 1;
}

function setActiveTab(name) {
  if (name === 'results') {
    state.activeTab = 'results';
  } else if (name === 'videos') {
    state.activeTab = 'videos';
  } else {
    state.activeTab = 'gallery';
  }
  const galleryActive = state.activeTab === 'gallery';
  const videosActive = state.activeTab === 'videos';
  const resultsActive = state.activeTab === 'results';

  galleryTabButton.setAttribute('aria-selected', galleryActive ? 'true' : 'false');
  videosTabButton.setAttribute('aria-selected', videosActive ? 'true' : 'false');
  resultsTabButton.setAttribute('aria-selected', resultsActive ? 'true' : 'false');
  galleryPanel.hidden = !galleryActive;
  videosPanel.hidden = !videosActive;
  resultsPanel.hidden = !resultsActive;
  updateTabCounts();
  clearResultsButton.disabled = state.results.length === 0;
}

function normalizeText(value) {
  if (typeof value !== 'string') {
    return '';
  }
  return value.trim().replace(/\s+/g, ' ');
}

function supportTextForItem(item, mode) {
  const description = normalizeText(item.description);
  const transcriptText = normalizeText(item.transcript_text);
  const mediaType = item.media_type || 'image';

  if (mediaType === 'video' || mode === 'videos') {
    return transcriptText || description;
  }

  if (mode === 'result' && transcriptText) {
    return transcriptText;
  }

  return description || transcriptText;
}

function supportTextClass(item, supportText) {
  if (!supportText) {
    return '';
  }
  const transcriptText = normalizeText(item.transcript_text);
  if (transcriptText && supportText === transcriptText) {
    return 'supporting-text supporting-transcript';
  }
  return 'supporting-text';
}

function tagsMarkup(tags, includeOverflowChip) {
  if (!Array.isArray(tags) || tags.length === 0) {
    return '';
  }

  const visibleTags = includeOverflowChip ? tags.slice(0, 4) : tags;
  const hiddenTagCount = includeOverflowChip ? Math.max(0, tags.length - visibleTags.length) : 0;

  return `<ul class="tag-list${includeOverflowChip ? '' : ' overlay-tag-list'}">${visibleTags
    .map((tag) => `<li class="tag-chip${tag.toLowerCase() === 'nsfw' ? ' tag-chip-nsfw' : ''}">${escapeHTML(tag)}</li>`)
    .join('')}${hiddenTagCount > 0 ? `<li class="tag-chip tag-chip-more">+${hiddenTagCount}</li>` : ''}</ul>`;
}

function cardMarkup(item, mode) {
  const safeName = escapeHTML(item.original_name);
  const safePath = escapeHTML(item.storage_path);
  const mediaType = item.media_type || 'image';
  const safeMimeType = escapeHTML(item.mime_type || 'video/mp4');
  const thumbPath = mediaType === 'video' && item.preview_path ? item.preview_path : item.storage_path;
  const thumbURL = escapeHTML(toMediaURL(thumbPath));
  const mediaURL = escapeHTML(toMediaURL(item.storage_path));
  const supportText = supportTextForItem(item, mode);
  const supportClass = supportTextClass(item, supportText);
  const tags = Array.isArray(item.tags)
    ? item.tags
        .filter((tag) => typeof tag === 'string' && tag.trim() !== '')
        .map((tag) => tag.trim())
        .slice(0, 8)
    : [];
  const status = item.index_state || 'done';
  const safeStatus = escapeHTML(humanizeIndexState(status));
  const scoreLabel = formatMatch(item.distance);
  const score = scoreLabel && !item.is_anchor ? `<p class="distance">${escapeHTML(scoreLabel)}</p>` : '';
  const canSearchSimilar = status === 'done' && Number(item.image_id) > 0;
  const actionLabel = mode === 'result' ? (item.is_anchor ? 'Anchor image' : mediaType === 'video' ? 'Use frame' : 'Use anchor') : 'Find similar';
  const disabled = canSearchSimilar ? '' : 'disabled';
  const title = canSearchSimilar ? '' : 'title="Available after indexing finishes"';
  const anchorBadge = item.is_anchor ? '<p class="state anchor">anchor</p>' : '';
  const videoMeta = videoMetaMarkup(item, mode);
  const deleteButton = mode === 'gallery'
    ? `<button class="ghost thumb-action danger delete-action" data-delete-kind="image" data-delete-id="${item.image_id}" data-delete-name="${safeName}">Delete</button>`
    : mode === 'videos'
      ? `<button class="ghost thumb-action danger delete-action" data-delete-kind="video" data-delete-id="${item.video_id}" data-delete-name="${safeName}">Delete</button>`
      : '';
  const supportMarkup = supportText
    ? `<p class="${supportClass}">${escapeHTML(supportText)}</p>`
    : '';
  const compactTagsMarkup = tagsMarkup(tags, true);
  const overlayTagsMarkup = tagsMarkup(tags, false);
  const overlaySupportMarkup = supportText
    ? `<p class="${supportClass} overlay-supporting-text">${escapeHTML(supportText)}</p>`
    : '';
  const overlayVideoMeta = videoMeta
    ? `<div class="overlay-video-meta">${videoMeta}</div>`
    : '';
  const overlayStatusMarkup = `<div class="overlay-status-row"><p class="${stateClass(status)}">${safeStatus}</p>${anchorBadge}${score}</div>`;

  return `
    <article class="card">
      <div class="thumb-wrap">
        <button
          type="button"
          class="thumb-button"
          data-media-type="${mediaType}"
          data-lightbox-src="${thumbURL}"
          data-lightbox-caption="${safeName}${mediaType === 'video' ? ' preview frame' : ''}"
          data-player-src="${mediaURL}"
          data-player-poster="${thumbURL}"
          data-player-type="${safeMimeType}"
          data-player-caption="${safeName}"
          aria-label="Open full ${mediaType === 'video' ? 'preview frame' : 'image'} for ${safeName}"
        >
          <img src="${thumbURL}" alt="${safeName}" loading="lazy" />
          ${mediaType === 'video' ? '<span class="thumb-video-badge">video</span>' : ''}
        </button>
        <div class="thumb-actions" role="group" aria-label="Card actions for ${safeName}">
          <button class="ghost thumb-action similar-action" data-image-id="${item.image_id}" ${disabled} ${title}>${actionLabel}</button>
          ${deleteButton}
        </div>
      </div>
      <div class="meta">
        <div class="meta-main">
          <h3>${safeName}</h3>
          <p class="path">${safePath}</p>
          ${videoMeta}
          ${compactTagsMarkup}
        </div>
        <div class="meta-foot">
          <div class="meta-status-row">
            <p class="${stateClass(status)}">${safeStatus}</p>
            ${anchorBadge}
            ${score}
          </div>
          ${supportMarkup}
        </div>
      </div>
      <div class="card-detail-overlay" aria-hidden="true">
        <p class="overlay-title">${safeName}</p>
        <p class="overlay-path">${safePath}</p>
        ${overlayVideoMeta}
        ${overlayTagsMarkup}
        ${overlayStatusMarkup}
        ${overlaySupportMarkup}
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

function renderVideos() {
  updateTabCounts();
  renderPagination();

  if (state.videos.length === 0) {
    if (state.videosTotal > 0) {
      videosGrid.innerHTML = '<p class="empty">No videos are visible on this page yet. Move back toward newer pages.</p>';
    } else {
      videosGrid.innerHTML = '<p class="empty">No videos here yet. Upload a clip to start building the video archive.</p>';
    }
    return;
  }

  videosGrid.innerHTML = state.videos.map((item) => cardMarkup(item, 'videos')).join('');
}

function renderResults() {
  updateTabCounts();
  clearResultsButton.disabled = state.results.length === 0;

  if (state.results.length === 0) {
    resultsGrid.innerHTML = '<p class="empty">Run a text search or similar-image search to populate this view with image or video matches.</p>';
    return;
  }

  resultsGrid.innerHTML = state.results.map((item) => cardMarkup(item, 'result')).join('');
}

function mediaByImageID() {
  const byID = new Map(state.images.map((item) => [item.image_id, item]));
  state.videos.forEach((item) => {
    if (Number(item.image_id) > 0) {
      byID.set(item.image_id, item);
    }
  });
  return byID;
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
  statsDetail.textContent = `Library ${imagesTotal} images. Pending ${pending}. Processing ${leased}.`;
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
  if (kind === 'video') {
    document.body.classList.toggle('lightbox-open', active);
  }

  if (!shell) {
    return;
  }

  const hasOverlay = Boolean((uploadModal && !uploadModal.hidden) || (lightbox && !lightbox.hidden) || (videoModal && !videoModal.hidden));
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

function ensureVideoPlayer() {
  if (videoPlayer || !videoElement || typeof window.videojs !== 'function') {
    return videoPlayer;
  }
  videoPlayer = window.videojs(videoElement, {
    controls: true,
    preload: 'auto',
    fluid: false,
  });
  return videoPlayer;
}

function layoutVideoPlayer(player) {
  if (!player || !videoModal || videoModal.hidden) {
    return;
  }

  const intrinsicWidth = Number(player.videoWidth()) || Number(videoElement.videoWidth) || 16;
  const intrinsicHeight = Number(player.videoHeight()) || Number(videoElement.videoHeight) || 9;
  const ratio = intrinsicWidth / intrinsicHeight || 16 / 9;
  const targetWidth = Math.min(window.innerWidth - 64, 960);
  const targetHeight = Math.min(window.innerHeight - 180, 720);

  let width = targetWidth;
  let height = width / ratio;
  if (height > targetHeight) {
    height = targetHeight;
    width = height * ratio;
  }

  width = Math.max(240, Math.floor(width));
  height = Math.max(160, Math.floor(height));

  player.dimensions(width, height);
}

function openVideoPlayer(source, poster, caption, mimeType) {
  if (!videoModal || !videoElement || !source) {
    return;
  }

  lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const player = ensureVideoPlayer();
  if (player) {
    player.poster(poster || '');
    player.src({ src: source, type: mimeType || 'video/mp4' });
    player.currentTime(0);
    player.one('loadedmetadata', () => {
      layoutVideoPlayer(player);
    });
    player.ready(() => {
      player.play().catch(() => {});
    });
  } else {
    videoElement.setAttribute('controls', 'controls');
    videoElement.poster = poster || '';
    videoElement.src = source;
    videoElement.currentTime = 0;
    void videoElement.play().catch(() => {});
  }
  if (videoCaption) {
    videoCaption.textContent = caption || '';
  }
  videoModal.setAttribute('aria-label', caption || 'Video player');
  videoModal.hidden = false;
  setOverlayState('video', true);

  if (player) {
    layoutVideoPlayer(player);
  }

  if (videoCloseButton) {
    videoCloseButton.focus();
  }
}

function closeVideoPlayer() {
  if (!videoModal || videoModal.hidden) {
    return;
  }

  if (videoPlayer) {
    videoPlayer.pause();
    videoPlayer.currentTime(0);
    videoPlayer.src({ src: '', type: 'video/mp4' });
    videoPlayer.poster('');
  } else if (videoElement) {
    videoElement.pause();
    videoElement.removeAttribute('src');
    videoElement.load();
    videoElement.poster = '';
  }
  if (videoCaption) {
    videoCaption.textContent = '';
  }
  videoModal.hidden = true;
  videoModal.removeAttribute('aria-label');
  setOverlayState('video', false);

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

async function loadVideos() {
  const offset = state.videosPage * state.videosPageSize;
  setStatus(videosStatus, 'Loading videos...', 'info');

  const response = await fetch('/api/videos' + `?limit=${state.videosPageSize}&offset=${offset}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load videos');
  }

  const nextVideos = payload.videos || [];
  const nextTotal = Number(payload.total || nextVideos.length);
  const maxPage = Math.max(0, Math.ceil(nextTotal / state.videosPageSize) - 1);
  if (nextVideos.length === 0 && nextTotal > 0 && state.videosPage > maxPage) {
    state.videosPage = maxPage;
    return loadVideos();
  }

  state.videos = nextVideos;
  state.videosTotal = nextTotal;
  renderVideos();

  const start = state.videosTotal === 0 ? 0 : offset + 1;
  const end = offset + state.videos.length;
  setStatus(videosStatus, `Showing ${start}-${end} of ${state.videosTotal} video(s), newest first.`, 'success');
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

  const imageByID = mediaByImageID();
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

  const imageByID = mediaByImageID();
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
    loadVideos().catch((err) => setStatus(videosStatus, err.message || 'Video refresh failed', 'error'));
    loadStats().catch((err) => setStatus(statsSummary, err.message || 'Status refresh failed', 'error'));
  }, 5000);

  if (!receivedLiveSnapshot) {
    loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Initial load failed', 'error'));
    loadVideos().catch((err) => setStatus(videosStatus, err.message || 'Initial video load failed', 'error'));
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

async function deleteMedia(kind, id, name) {
  const label = kind === 'video' ? 'video' : 'image';
  if (!window.confirm(`Delete ${label} \"${name || 'this item'}\"? This cannot be undone.`)) {
    return false;
  }
  const response = await fetch(`/api/${kind === 'video' ? 'videos' : 'images'}/${id}`, { method: 'DELETE' });
  if (response.status === 204) {
    return true;
  }
  let message = `Could not delete ${label}`;
  try {
    const payload = await response.json();
    if (payload && payload.error) {
      message = payload.error;
    }
  } catch {
    // fall back to the generic message.
  }
  throw new Error(message);
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
      state.videos.find((item) => Number(item.image_id) === imageID) ||
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

    const mediaType = trigger.dataset.mediaType || 'image';
    if (mediaType === 'video') {
      openVideoPlayer(
        trigger.dataset.playerSrc || '',
        trigger.dataset.playerPoster || '',
        trigger.dataset.playerCaption || '',
        trigger.dataset.playerType || 'video/mp4',
      );
      return;
    }

    const source = trigger.dataset.lightboxSrc || '';
    const caption = trigger.dataset.lightboxCaption || '';
    openLightbox(source, caption);
  });
}

function attachDeleteHandler(target) {
  target.addEventListener('click', async (event) => {
    const trigger = event.target.closest('.delete-action');
    if (!trigger || !target.contains(trigger)) {
      return;
    }

    const kind = trigger.dataset.deleteKind || 'image';
    const id = Number(trigger.dataset.deleteId || 0);
    const name = trigger.dataset.deleteName || '';
    if (!Number.isFinite(id) || id <= 0) {
      return;
    }

    try {
      const deleted = await deleteMedia(kind, id, name);
      if (!deleted) {
        return;
      }
      state.results = state.results.filter((item) => {
        if (kind === 'video') {
          return Number(item.video_id) !== id;
        }
        return Number(item.image_id) !== id;
      });
      renderResults();
      await loadImages();
      await loadVideos();
      await loadStats();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Delete failed';
      if (kind === 'video') {
        setStatus(videosStatus, message, 'error');
      } else {
        setStatus(galleryStatus, message, 'error');
      }
    }
  });
}

galleryTabButton.addEventListener('click', () => setActiveTab('gallery'));
videosTabButton.addEventListener('click', () => setActiveTab('videos'));
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
  if (state.galleryPage >= totalPages(state.imagesTotal, state.galleryPageSize) - 1) {
    return;
  }
  state.galleryPage += 1;
  try {
    await loadImages();
  } catch (err) {
    setStatus(galleryStatus, err.message || 'Could not load next page', 'error');
  }
});

videosPrevButton.addEventListener('click', async () => {
  if (state.videosPage <= 0) {
    return;
  }
  state.videosPage -= 1;
  try {
    await loadVideos();
  } catch (err) {
    setStatus(videosStatus, err.message || 'Could not load previous page', 'error');
  }
});

videosNextButton.addEventListener('click', async () => {
  if (state.videosPage >= totalPages(state.videosTotal, state.videosPageSize) - 1) {
    return;
  }
  state.videosPage += 1;
  try {
    await loadVideos();
  } catch (err) {
    setStatus(videosStatus, err.message || 'Could not load next page', 'error');
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
    setStatus(uploadStatus, 'Choose one or more supported image or video files first.', 'error');
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
    state.videosPage = 0;
    setActiveTab('gallery');

    if (!isLiveSocketActive()) {
      await loadImages();
      await loadVideos();
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
  loadVideos().catch((err) => setStatus(videosStatus, err.message || 'Video refresh failed', 'error'));
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
    await loadVideos();
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
attachSimilarHandler(videosGrid);
attachSimilarHandler(resultsGrid);
attachDeleteHandler(galleryGrid);
attachDeleteHandler(videosGrid);
attachLightboxHandler(galleryGrid);
attachLightboxHandler(videosGrid);
attachLightboxHandler(resultsGrid);

if (lightboxCloseButton) {
  lightboxCloseButton.addEventListener('click', closeLightbox);
}

if (videoCloseButton) {
  videoCloseButton.addEventListener('click', closeVideoPlayer);
}

if (lightbox) {
  lightbox.addEventListener('click', (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });
}

if (videoModal) {
  videoModal.addEventListener('click', (event) => {
    if (event.target === videoModal) {
      closeVideoPlayer();
    }
  });
}

window.addEventListener('resize', () => {
  if (videoPlayer && videoModal && !videoModal.hidden) {
    layoutVideoPlayer(videoPlayer);
  }
});

document.addEventListener('keydown', (event) => {
  if (lightbox && !lightbox.hidden) {
    trapFocus(event, lightbox);
  } else if (videoModal && !videoModal.hidden) {
    trapFocus(event, videoModal);
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

  if (videoModal && !videoModal.hidden) {
    closeVideoPlayer();
    return;
  }

  if (uploadModal && !uploadModal.hidden) {
    closeUploadModal();
  }
});

renderGallery();
renderVideos();
renderResults();
renderStats();
setActiveTab('gallery');

loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Initial load failed', 'error'));
loadVideos().catch((err) => setStatus(videosStatus, err.message || 'Initial video load failed', 'error'));
loadStats().catch((err) => setStatus(statsSummary, err.message || 'Initial status load failed', 'error'));
connectLiveUpdates();
