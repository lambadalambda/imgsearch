const state = {
  images: [],
  results: [],
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
  const score = typeof item.distance === 'number' ? `<p class="distance">distance ${item.distance.toFixed(4)}</p>` : '';
  const status = item.index_state || 'done';
  const canSearchSimilar = status === 'done';
  const actionLabel = mode === 'result' ? 'Use as anchor' : 'Find similar';
  const disabled = canSearchSimilar ? '' : 'disabled';
  const title = canSearchSimilar ? '' : 'title="Available after indexing finishes"';

  return `
    <article class="card">
      <img src="${safeImageURL}" alt="${safeName}" loading="lazy" />
      <div class="meta">
        <h3>${safeName}</h3>
        <p class="path">${safePath}</p>
        <p class="${stateClass(status)}">${status}</p>
        ${score}
        <button class="ghost similar-action" data-image-id="${item.image_id}" ${disabled} ${title}>${actionLabel}</button>
      </div>
    </article>
  `;
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

async function loadImages() {
  setStatus(uploadStatus, 'Loading gallery...', 'info');
  const response = await fetch('/api/images' + '?limit=120&offset=0');
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load images');
  }
  state.images = payload.images || [];
  renderGallery();
  setStatus(uploadStatus, `Loaded ${state.images.length} image(s).`, 'success');
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
    };
  });
  renderResults();
  setStatus(searchStatus, `Text search returned ${state.results.length} result(s).`, 'success');
}

async function runSimilarSearch(imageID) {
  const params = new URLSearchParams({ image_id: String(imageID), limit: '24' });
  const response = await fetch(`/api/search/similar?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Similar search failed');
  }

  const imageByID = new Map(state.images.map((item) => [item.image_id, item]));
  state.results = (payload.results || []).map((item) => {
    const image = imageByID.get(item.image_id) || {};
    return {
      ...image,
      ...item,
      index_state: image.index_state || 'done',
    };
  });
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
    try {
      await runSimilarSearch(imageID);
    } catch (err) {
      setStatus(searchStatus, err.message || 'Similar search failed', 'error');
    }
  });
}

attachSimilarHandler(galleryGrid);
attachSimilarHandler(resultsGrid);

refreshImagesButton.addEventListener('click', () => {
  loadImages().catch((err) => setStatus(uploadStatus, err.message || 'Refresh failed', 'error'));
});

clearResultsButton.addEventListener('click', () => {
  state.results = [];
  renderResults();
  setStatus(searchStatus, 'Search results cleared.', 'info');
});

renderGallery();
renderResults();
loadImages().catch((err) => setStatus(uploadStatus, err.message || 'Initial load failed', 'error'));
