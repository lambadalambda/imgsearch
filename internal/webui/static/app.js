const state = {
  images: [],
  results: [],
  imagesTotal: 0,
  stats: null,
};

const galleryGrid = document.getElementById('gallery-grid');
const resultsGrid = document.getElementById('results-grid');
const uploadForm = document.getElementById('upload-form');
const uploadFile = document.getElementById('upload-file');
const uploadStatus = document.getElementById('upload-status');
const searchForm = document.getElementById('text-search-form');
const textQuery = document.getElementById('text-query');
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

let lastFocusedElement = null;

function setStatus(target, message, kind) {
  target.textContent = message;
  target.dataset.kind = kind || '';
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

function cardMarkup(item, mode) {
  const safeName = escapeHTML(item.original_name);
  const safePath = escapeHTML(item.storage_path);
  const safeImageURL = escapeHTML(toMediaURL(item.storage_path));
  const score = typeof item.distance === 'number' && !item.is_anchor ? `<p class="distance">distance ${item.distance.toFixed(4)}</p>` : '';
  const status = item.index_state || 'done';
  const canSearchSimilar = status === 'done';
  const actionLabel = mode === 'result' ? (item.is_anchor ? 'Anchor image' : 'Use as anchor') : 'Find similar';
  const disabled = canSearchSimilar ? '' : 'disabled';
  const title = canSearchSimilar ? '' : 'title="Available after indexing finishes"';
  const anchorBadge = item.is_anchor ? '<p class="state anchor">anchor</p>' : '';

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
        <p class="${stateClass(status)}">${status}</p>
        ${anchorBadge}
        ${score}
        <button class="ghost similar-action" data-image-id="${item.image_id}" ${disabled} ${title}>${actionLabel}</button>
      </div>
    </article>
  `;
}

function openLightbox(source, caption) {
  if (!lightbox || !lightboxImage || !lightboxCaption || !source) {
    return;
  }

  lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  lightboxImage.src = source;
  lightboxImage.alt = caption || 'Full-size image preview';
  lightboxCaption.textContent = caption || '';
  lightbox.hidden = false;
  document.body.classList.add('lightbox-open');

  if (lightboxCloseButton) {
    lightboxCloseButton.focus();
  }
}

function closeLightbox() {
  if (!lightbox || !lightboxImage || !lightboxCaption) {
    return;
  }

  if (lightbox.hidden) {
    return;
  }

  lightbox.hidden = true;
  lightboxImage.removeAttribute('src');
  lightboxCaption.textContent = '';
  document.body.classList.remove('lightbox-open');

  if (lastFocusedElement) {
    lastFocusedElement.focus();
  }
  lastFocusedElement = null;
}

function renderGallery() {
  if (state.images.length === 0) {
    galleryGrid.innerHTML = '<p class="empty">No images yet. Upload one to start indexing.</p>';
    return;
  }
  galleryGrid.innerHTML = state.images.map((item) => cardMarkup(item, 'gallery')).join('');
}

function renderResults() {
  if (state.results.length === 0) {
    resultsGrid.innerHTML = '<p class="empty">No active search results.</p>';
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

  const queue = payload.queue;
  const total = Number(queue.total || 0);
  const done = Number(queue.done || 0);
  const failed = Number(queue.failed || 0);
  const missing = Number(queue.missing || 0);
  const pending = Number(queue.pending || 0);
  const leased = Number(queue.leased || 0);
  const completed = done + failed;
  const pct = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0;

  statsProgress.style.width = `${pct}%`;
  setStatus(
    statsSummary,
    `Indexed ${done}/${total} jobs (${pct}%). Failed: ${failed}. Missing jobs: ${missing}.`,
    failed > 0 ? 'error' : missing > 0 ? 'info' : 'success',
  );

  const imagesTotal = Number(payload.images_total || state.imagesTotal || 0);
  statsDetail.textContent = `Images: ${imagesTotal}. Queue pending: ${pending}. In progress: ${leased}.`;
  retryFailedButton.disabled = failed === 0 && missing === 0;

  const failures = payload.recent_failures || [];
  if (failures.length === 0) {
    statsFailures.innerHTML = '';
    return;
  }

  statsFailures.innerHTML = failures
    .map((item) => {
      const safeName = escapeHTML(item.original_name || `image #${item.image_id}`);
      const safeError = escapeHTML(item.last_error || 'unknown error');
      return `<article class="failure-item"><p><strong>${safeName}</strong> (attempt ${item.attempts})</p><p>${safeError}</p></article>`;
    })
    .join('');
}

async function loadStats() {
  const response = await fetch('/api/stats');
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load queue stats');
  }
  state.stats = payload;
  renderStats();
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

async function loadImages() {
  setStatus(uploadStatus, 'Loading gallery...', 'info');
  const response = await fetch('/api/images' + '?limit=120&offset=0');
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load images');
  }
  state.images = payload.images || [];
  state.imagesTotal = Number(payload.total || state.images.length);
  renderGallery();
  setStatus(uploadStatus, `Loaded ${state.images.length} of ${state.imagesTotal} image(s) (newest first).`, 'success');
}

async function runTextSearch(query) {
  const params = new URLSearchParams({ q: query, limit: '24' });
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
  setStatus(searchStatus, `Similar search returned ${state.results.length} result(s).`, 'success');
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = uploadFile.files && uploadFile.files[0];
  if (!file) {
    setStatus(uploadStatus, 'Choose a JPEG, PNG, WEBP, or AVIF file first.', 'error');
    return;
  }

  const data = new FormData();
  data.set('file', file, file.name);

  setStatus(uploadStatus, `Uploading ${file.name}...`, 'info');
  try {
    const response = await fetch('/api/upload', { method: 'POST', body: data });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'Upload failed');
    }

    const action = payload.duplicate ? 'already exists' : 'queued for indexing';
    setStatus(uploadStatus, `${file.name} ${action}.`, 'success');
    uploadForm.reset();

    await loadImages();
    window.setTimeout(() => {
      loadImages().catch((err) => setStatus(uploadStatus, err.message, 'error'));
    }, 1500);
  } catch (err) {
    setStatus(uploadStatus, err.message || 'Upload failed', 'error');
  }
});

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const q = textQuery.value.trim();
  if (!q) {
    setStatus(searchStatus, 'Enter a search phrase first.', 'error');
    return;
  }

  setStatus(searchStatus, `Searching for "${q}"...`, 'info');
  try {
    await runTextSearch(q);
  } catch (err) {
    setStatus(searchStatus, err.message || 'Text search failed', 'error');
  }
});

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

attachSimilarHandler(galleryGrid);
attachSimilarHandler(resultsGrid);
attachLightboxHandler(galleryGrid);
attachLightboxHandler(resultsGrid);

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
  if (event.key === 'Escape') {
    closeLightbox();
  }
});

refreshImagesButton.addEventListener('click', () => {
  loadImages().catch((err) => setStatus(uploadStatus, err.message || 'Refresh failed', 'error'));
});

refreshStatsButton.addEventListener('click', () => {
  loadStats().catch((err) => setStatus(statsSummary, err.message || 'Status refresh failed', 'error'));
});

retryFailedButton.addEventListener('click', async () => {
    retryFailedButton.disabled = true;
  setStatus(statsSummary, 'Retrying failed jobs...', 'info');
  try {
    const result = await retryFailedJobs();
    await loadStats();
    await loadImages();
    const hadWork = result.retried > 0 || result.enqueuedMissing > 0;
    setStatus(
      statsSummary,
      `Requeued ${result.retried} failed job(s), enqueued ${result.enqueuedMissing} missing job(s).`,
      hadWork ? 'success' : 'info',
    );
  } catch (err) {
    setStatus(statsSummary, err.message || 'Retry failed jobs failed', 'error');
  } finally {
    const failed = state.stats && state.stats.queue ? Number(state.stats.queue.failed || 0) : 0;
    const missing = state.stats && state.stats.queue ? Number(state.stats.queue.missing || 0) : 0;
    retryFailedButton.disabled = failed === 0 && missing === 0;
  }
});

clearResultsButton.addEventListener('click', () => {
  state.results = [];
  renderResults();
  setStatus(searchStatus, 'Search results cleared.', 'info');
});

renderGallery();
renderResults();
renderStats();
loadImages().catch((err) => setStatus(uploadStatus, err.message || 'Initial load failed', 'error'));
loadStats().catch((err) => setStatus(statsSummary, err.message || 'Initial status load failed', 'error'));

window.setInterval(() => {
  loadImages().catch((err) => setStatus(uploadStatus, err.message || 'Auto-refresh failed', 'error'));
  loadStats().catch((err) => setStatus(statsSummary, err.message || 'Auto-refresh failed', 'error'));
}, 5000);
