const state = {
  images: [],
  videos: [],
  results: [],
  tagCloud: [],
  imagesTotal: 0,
  videosTotal: 0,
  resultsTotal: 0,
  stats: null,
  galleryPage: 0,
  videosPage: 0,
  resultsPage: 0,
  galleryPageSize: 24,
  videosPageSize: 24,
  resultsPageSize: 24,
  resultsMode: 'none',
  resultsProvenance: '',
  activeTagFilters: [],
  activeTagMode: 'any',
  searchTagFilters: [],
  showNSFW: false,
  activeTab: 'gallery',
  imageSelectionMode: false,
  selectedImageIDs: new Set(),
};

const galleryGrid = document.getElementById('gallery-grid');
const galleryStatus = document.getElementById('gallery-status');
const galleryPrevButton = document.getElementById('gallery-prev');
const galleryNextButton = document.getElementById('gallery-next');
const galleryPageLabel = document.getElementById('gallery-page-label');
const galleryTabButton = document.getElementById('tab-gallery');
const galleryTabCount = document.getElementById('tab-gallery-count');
const galleryPanel = document.getElementById('gallery-panel');
const gallerySelectModeButton = document.getElementById('gallery-select-mode');
const gallerySelectPageButton = document.getElementById('gallery-select-page');
const galleryClearSelectionButton = document.getElementById('gallery-clear-selection');
const galleryDeleteSelectedButton = document.getElementById('gallery-delete-selected');
const galleryDeleteAllButton = document.getElementById('gallery-delete-all');
const gallerySelectionToolbar = document.getElementById('gallery-selection-toolbar');
const gallerySelectionSummary = document.getElementById('gallery-selection-summary');

const videosGrid = document.getElementById('videos-grid');
const videosStatus = document.getElementById('videos-status');
const videosPrevButton = document.getElementById('videos-prev');
const videosNextButton = document.getElementById('videos-next');
const videosPageLabel = document.getElementById('videos-page-label');
const videosTabButton = document.getElementById('tab-videos');
const videosTabCount = document.getElementById('tab-videos-count');
const videosPanel = document.getElementById('videos-panel');

const tagsTabButton = document.getElementById('tab-tags');
const tagsPanel = document.getElementById('tags-panel');
const tagsStatus = document.getElementById('tags-status');
const tagCloud = document.getElementById('tag-cloud');

const resultsGrid = document.getElementById('results-grid');
const resultsTabButton = document.getElementById('tab-results');
const resultsTabCount = document.getElementById('tab-results-count');
const resultsPanel = document.getElementById('results-panel');
const resultsPrevButton = document.getElementById('results-prev');
const resultsNextButton = document.getElementById('results-next');
const resultsPageLabel = document.getElementById('results-page-label');
const resultsProvenance = document.getElementById('results-provenance');

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
const searchTagChips = document.getElementById('search-tag-chips');
const searchTagInput = document.getElementById('search-tag-input');
const searchTagSuggestions = document.getElementById('search-tag-suggestions');
const searchTagMode = document.getElementById('search-tag-mode');
const searchFilterCount = document.getElementById('search-filter-count');
const activeSearchFilters = document.getElementById('active-search-filters');

const refreshImagesButton = document.getElementById('refresh-images');
const clearResultsButton = document.getElementById('clear-results');
const showNSFWToggle = document.getElementById('show-nsfw');
const showNSFWLabel = document.getElementById('show-nsfw-label');
const showNSFWControl = document.querySelector('.nsfw-toggle');
const refreshStatsButton = document.getElementById('refresh-stats');
const retryFailedButton = document.getElementById('retry-failed');
const opsBar = document.getElementById('ops-bar');

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
const videoFeed = document.getElementById('video-feed');
const feedTrack = document.getElementById('feed-track');
const feedVideo = document.getElementById('video-feed-player');
const feedPrevVideo = document.getElementById('video-feed-prev-player');
const feedNextVideo = document.getElementById('video-feed-next-player');
const feedTopbar = document.querySelector('#video-feed .feed-topbar');
const feedControls = document.querySelector('#video-feed .feed-controls');
const feedSeedLabel = document.getElementById('feed-seed-label');
const feedExitButton = document.getElementById('feed-exit');
const feedPrevButton = document.getElementById('feed-prev');
const feedNextButton = document.getElementById('feed-next');
const feedPlayToggleButton = document.getElementById('feed-play-toggle');
const feedMuteToggleButton = document.getElementById('feed-mute-toggle');
const feedInfo = document.getElementById('feed-info');
const feedPosition = document.getElementById('feed-position');
const feedHint = document.getElementById('feed-hint');
const feedProgressBar = document.getElementById('feed-progress-bar');
const feedEnd = document.getElementById('feed-end');
const feedEndBackButton = document.getElementById('feed-end-back');

const liveConnectionBadge = document.getElementById('live-connection');
const liveConnectionLabel = document.getElementById('live-connection-label');
const mastheadIndexStatus = document.getElementById('masthead-index-status');
const liveAnnouncer = document.getElementById('live-announcer');
const shell = document.querySelector('.shell');

let lastFocusedElement = null;
let liveSocket = null;
let liveReconnectTimer = null;
let pollingTimer = null;
let receivedLiveSnapshot = false;
let liveReconnectDelayMs = 1500;
const collectionRequestTokens = { images: 0, videos: 0 };
let statsRequestToken = 0;
let videoPlayer = null;
let tagCloudLoaded = false;
let tagSuggestionRequestToken = 0;
let tagSuggestionDebounceTimer = null;
let bulkImageDeleteInFlight = false;
const liveReconnectDelayMaxMs = 30000;
const nsfwPreferenceStorageKey = 'imgsearch.showNSFW';
const feedHintSeenStorageKey = 'imgsearch.feedHintSeen';
const feedState = {
  queue: [],
  currentIndex: 0,
  muted: true,
  loading: false,
  playbackStarted: false,
  itemStartedAt: 0,
  watchStartedAt: 0,
  accumulatedWatchMs: 0,
  sessionID: '',
  events: [],
  seenVideoIDs: new Set(),
  tagScores: new Map(),
  touchStartX: 0,
  touchStartY: 0,
  touchCurrentY: 0,
  touchStartedAt: 0,
  touchActive: false,
  dragging: false,
  isPaging: false,
  snapTimer: null,
  snapToken: 0,
  suppressClickUntil: 0,
  loadingMore: false,
  exhausted: false,
  candidateRequestToken: 0,
  feedbackRecordedIndex: -1,
  lookaheadPromise: null,
  rejectedCandidateIDs: new Set(),
};
const feedSnapDurationMS = 220;
const feedInitialCandidateLimit = 4;
const feedBatchSize = 3;
const feedFetchAheadThreshold = 2;
const feedTagScoreMin = -3;
const feedTagScoreMax = 5;
const feedTagScoreDecay = 0.9;
const feedPreferenceThreshold = 0.25;

function localPreference(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function setLocalPreference(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
  }
}

function prefersReducedMotion() {
  return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

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

function setMastheadIndexStatus(label, stateName) {
  if (!mastheadIndexStatus) {
    return;
  }
  mastheadIndexStatus.textContent = label;
  mastheadIndexStatus.dataset.state = stateName || 'idle';
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

function scoreBadgeText(item, scoreLabel) {
  if (!scoreLabel) {
    return '';
  }
  if ((item.media_type || 'image') === 'video' && Number(item.match_timestamp_ms) > 0) {
    return `${scoreLabel} at ${formatTimestamp(item.match_timestamp_ms)}`;
  }
  return scoreLabel;
}

function normalizeTagTerms(parts) {
  if (!Array.isArray(parts)) {
    return [];
  }
  const seen = new Set();
  const normalized = [];
  parts.forEach((part) => {
    if (typeof part !== 'string') {
      return;
    }
    const tag = part.trim().toLowerCase();
    if (!tag || seen.has(tag)) {
      return;
    }
    seen.add(tag);
    normalized.push(tag);
  });
  return normalized;
}

function parseTagTerms(input) {
  if (typeof input !== 'string') {
    return [];
  }
  return normalizeTagTerms(input.split(','));
}

function applyNSFWQuery(params) {
  if (!(params instanceof URLSearchParams)) {
    return params;
  }
  if (state.showNSFW) {
    params.set('include_nsfw', '1');
  } else {
    params.delete('include_nsfw');
  }
  return params;
}

function formatSearchDebugSuffix(debug) {
  const details = debug && typeof debug === 'object' ? debug : {};
  const durationMS = Number(details.duration_ms);
  const hasDuration = Number.isFinite(durationMS) && durationMS >= 0;
  const durationLabel = hasDuration ? `${(durationMS / 1000).toFixed(durationMS >= 10000 ? 1 : 2)}s` : 'n/a';

  const backend = typeof details.index_backend === 'string' && details.index_backend.trim() ? details.index_backend.trim() : 'unknown';
  const strategy = typeof details.index_strategy === 'string' && details.index_strategy.trim() ? details.index_strategy.trim() : '';
  const quantization = typeof details.quantization === 'string' && details.quantization.trim() ? details.quantization.trim() : 'unknown';

  if (!hasDuration && backend === 'unknown' && quantization === 'unknown') {
    return '';
  }

  const indexLabel = strategy ? `${backend}/${strategy}` : backend;
  return ` (took ${durationLabel}, index: ${indexLabel}, quantization: ${quantization})`;
}

function renderSearchTagFilters() {
  if (!searchTagChips) {
    renderActiveSearchFilters();
    return;
  }
  if (!Array.isArray(state.searchTagFilters) || state.searchTagFilters.length === 0) {
    searchTagChips.innerHTML = '';
    renderActiveSearchFilters();
    return;
  }
  searchTagChips.innerHTML = state.searchTagFilters
    .map((tag) => `<button type="button" class="search-tag-chip" data-tag-remove="${escapeHTML(tag)}" aria-label="Remove tag filter ${escapeHTML(tag)}">${escapeHTML(tag)}<span class="search-tag-chip-remove" aria-hidden="true">x</span></button>`)
    .join('');
  renderActiveSearchFilters();
}

function activeSearchFilterEntries() {
  const entries = [];
  const negative = negativeQuery ? negativeQuery.value.trim() : '';
  if (negative) {
    entries.push({ label: `Exclude: ${negative}`, className: 'active-filter-chip active-filter-negative' });
  }
  for (const tag of state.searchTagFilters || []) {
    entries.push({ label: `Tag: ${tag}`, className: 'active-filter-chip' });
  }
  return entries;
}

function renderActiveSearchFilters() {
  const entries = activeSearchFilterEntries();
  if (searchFilterCount) {
    searchFilterCount.textContent = String(entries.length);
    searchFilterCount.hidden = entries.length === 0;
  }
  if (!activeSearchFilters) {
    return;
  }
  activeSearchFilters.hidden = entries.length === 0;
  activeSearchFilters.innerHTML = entries
    .map((entry) => `<span class="${entry.className}">${escapeHTML(entry.label)}</span>`)
    .join('');
}

function addSearchTagFilters(parts) {
  const next = normalizeTagTerms([...(state.searchTagFilters || []), ...parts]);
  state.searchTagFilters = next;
  renderSearchTagFilters();
}

function removeSearchTagFilter(tag) {
  if (!Array.isArray(state.searchTagFilters)) {
    state.searchTagFilters = [];
    renderActiveSearchFilters();
    return;
  }
  state.searchTagFilters = state.searchTagFilters.filter((item) => item !== tag);
  renderSearchTagFilters();
}

function clearTagSuggestions() {
  if (!searchTagSuggestions) {
    return;
  }
  searchTagSuggestions.innerHTML = '';
  searchTagSuggestions.hidden = true;
}

function renderTagSuggestions(tags, query) {
  if (!searchTagSuggestions) {
    return;
  }
  if (!Array.isArray(tags) || tags.length === 0) {
    clearTagSuggestions();
    return;
  }
  searchTagSuggestions.hidden = false;
  searchTagSuggestions.innerHTML = tags
    .map((item) => {
      const tag = typeof item.tag === 'string' ? item.tag : '';
      if (!tag) {
        return '';
      }
      const count = Number(item.count || 0);
      return `<button type="button" class="search-tag-suggestion" data-tag-suggestion="${escapeHTML(tag)}" role="option" aria-label="Add tag filter ${escapeHTML(tag)} (${count})"><span>${escapeHTML(tag)}</span><span class="search-tag-suggestion-count">${count}</span></button>`;
    })
    .join('');
  if (searchTagSuggestions.innerHTML.trim() === '') {
    clearTagSuggestions();
    return;
  }
  if (query && query.length > 0) {
    searchTagSuggestions.dataset.query = query;
  }
}

function scheduleTagSuggestions(query) {
  if (!searchTagSuggestions) {
    return;
  }
  const normalizedQuery = typeof query === 'string' ? query.trim().toLowerCase() : '';
  if (tagSuggestionDebounceTimer !== null) {
    window.clearTimeout(tagSuggestionDebounceTimer);
    tagSuggestionDebounceTimer = null;
  }
  if (!normalizedQuery) {
    clearTagSuggestions();
    return;
  }
  tagSuggestionDebounceTimer = window.setTimeout(async () => {
    const requestToken = ++tagSuggestionRequestToken;
    try {
      const params = applyNSFWQuery(new URLSearchParams({ limit: '8', min_count: '1', q: normalizedQuery }));
      const response = await fetch(`/api/search/tag-cloud?${params.toString()}`);
      const payload = await response.json();
      if (requestToken !== tagSuggestionRequestToken) {
        return;
      }
      if (!response.ok) {
        throw new Error(payload.error || 'Tag suggestions failed');
      }
      const existing = new Set(state.searchTagFilters || []);
      const suggestions = (payload.tags || []).filter((item) => {
        const tag = typeof item.tag === 'string' ? item.tag : '';
        return tag && !existing.has(tag);
      });
      renderTagSuggestions(suggestions, normalizedQuery);
    } catch {
      clearTagSuggestions();
    }
  }, 140);
}

function tagCloudSizeClass(count, minCount, maxCount) {
  if (!Number.isFinite(count) || maxCount <= minCount) {
    return 'tag-cloud-chip-size-2';
  }
  const ratio = (count - minCount) / Math.max(1, maxCount - minCount);
  if (ratio >= 0.75) {
    return 'tag-cloud-chip-size-4';
  }
  if (ratio >= 0.45) {
    return 'tag-cloud-chip-size-3';
  }
  if (ratio >= 0.2) {
    return 'tag-cloud-chip-size-2';
  }
  return 'tag-cloud-chip-size-1';
}

function humanizeIndexState(indexState) {
  const labels = {
    done: 'Indexed',
    failed: 'Failed',
    leased: 'Processing',
    pending: 'Pending',
  };
  return labels[indexState] || indexState || labels.done;
}

function videoMetaMarkup(item, mode) {
  if ((item.media_type || 'image') !== 'video') {
    return '';
  }

  if (mode === 'result' && Number(item.match_timestamp_ms) > 0) {
    return '';
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

function canPlayFeedItem(item) {
  const probeVideo = activeFeedVideo() || feedVideo;
  if (!probeVideo || typeof probeVideo.canPlayType !== 'function') {
    return true;
  }
  const mimeType = item && typeof item.mime_type === 'string' && item.mime_type.trim()
    ? item.mime_type.trim()
    : 'video/mp4';
  return probeVideo.canPlayType(mimeType) !== '';
}

function feedActionMarkup(item, safeName, mediaType) {
  const videoID = Number(item.video_id || 0);
  if (mediaType !== 'video' || videoID <= 0 || !item.storage_path || !canPlayFeedItem(item)) {
    return '';
  }

  return `<button
    class="ghost thumb-action feed-action"
    data-feed-video-id="${escapeHTML(String(videoID))}"
    data-feed-image-id="${escapeHTML(String(item.image_id || ''))}"
    data-feed-name="${safeName}"
    data-feed-storage-path="${escapeHTML(item.storage_path || '')}"
    data-feed-preview-path="${escapeHTML(item.preview_path || '')}"
    data-feed-mime-type="${escapeHTML(item.mime_type || 'video/mp4')}"
    aria-label="Open similar video feed for ${safeName}"
  >Feed</button>`;
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
  const resultCount = state.resultsMode === 'tag' ? Number(state.resultsTotal || 0) : Number(state.results.length || 0);
  resultsTabCount.textContent = String(resultCount);
  resultsTabButton.disabled = resultCount === 0 && state.activeTab !== 'results';
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

  const resultTotal = state.resultsMode === 'tag' ? Number(state.resultsTotal || 0) : Number(state.results.length || 0);
  const resultPages = totalPages(resultTotal, state.resultsPageSize);
  const resultPage = state.resultsMode === 'tag' ? state.resultsPage : 0;
  if (resultsPageLabel) {
    resultsPageLabel.textContent = `Page ${Math.min(resultPages, resultPage + 1)} of ${resultPages}`;
  }
  if (resultsPrevButton) {
    resultsPrevButton.disabled = state.resultsMode !== 'tag' || resultPage <= 0;
  }
  if (resultsNextButton) {
    resultsNextButton.disabled = state.resultsMode !== 'tag' || resultPage >= resultPages - 1;
  }
}

function setActiveTab(name) {
  if (name === 'results') {
    state.activeTab = 'results';
  } else if (name === 'tags') {
    state.activeTab = 'tags';
  } else if (name === 'videos') {
    state.activeTab = 'videos';
  } else {
    state.activeTab = 'gallery';
  }
  const galleryActive = state.activeTab === 'gallery';
  const videosActive = state.activeTab === 'videos';
  const tagsActive = state.activeTab === 'tags';
  const resultsActive = state.activeTab === 'results';

  galleryTabButton.setAttribute('aria-selected', galleryActive ? 'true' : 'false');
  videosTabButton.setAttribute('aria-selected', videosActive ? 'true' : 'false');
  tagsTabButton.setAttribute('aria-selected', tagsActive ? 'true' : 'false');
  resultsTabButton.setAttribute('aria-selected', resultsActive ? 'true' : 'false');
  galleryPanel.hidden = !galleryActive;
  videosPanel.hidden = !videosActive;
  tagsPanel.hidden = !tagsActive;
  resultsPanel.hidden = !resultsActive;
  if (tagsActive) {
    ensureTagCloudLoaded();
  }
  updateTabCounts();
  clearResultsButton.disabled = state.results.length === 0;
}

function applyNSFWPreference(nextValue) {
  state.showNSFW = Boolean(nextValue);
  if (showNSFWToggle) {
    showNSFWToggle.checked = state.showNSFW;
  }
  if (showNSFWLabel) {
    showNSFWLabel.textContent = state.showNSFW ? 'NSFW: Visible' : 'NSFW: Hidden';
  }
  if (showNSFWControl) {
    showNSFWControl.classList.toggle('is-enabled', state.showNSFW);
  }
}

function normalizeText(value) {
  if (typeof value !== 'string') {
    return '';
  }
  return value.trim().replace(/\s+/g, ' ');
}

function supportEntriesForItem(item, mode) {
  const description = normalizeText(item.description);
  const transcriptText = normalizeText(item.transcript_text);
  const mediaType = item.media_type || 'image';
  const entries = [];

  if (mediaType === 'video' || mode === 'videos') {
    if (description) {
      entries.push({
        text: description,
        className: 'supporting-text',
      });
    }
    if (transcriptText && transcriptText !== description) {
      entries.push({
        text: transcriptText,
        className: 'supporting-text supporting-transcript',
      });
    }
    if (entries.length > 0) {
      return entries;
    }
  }

  if (mode === 'result' && transcriptText) {
    return [{
      text: transcriptText,
      className: 'supporting-text supporting-transcript',
    }];
  }

  const fallback = description || transcriptText;
  if (!fallback) {
    return [];
  }

  return [{
    text: fallback,
    className: transcriptText && fallback === transcriptText
      ? 'supporting-text supporting-transcript'
      : 'supporting-text',
  }];
}

function tagsMarkup(tags, includeOverflowChip, clickable) {
  if (!Array.isArray(tags) || tags.length === 0) {
    return '';
  }

  const visibleTags = tags;
  const hiddenTagCount = 0;

  return `<ul class="tag-list${includeOverflowChip ? '' : ' overlay-tag-list'}">${visibleTags
    .map((tag) => {
      const toneClass = tag.toLowerCase() === 'nsfw' ? ' tag-chip-nsfw' : '';
      if (clickable) {
        return `<li><button type="button" class="tag-chip tag-chip-button${toneClass}" data-tag-search="${escapeHTML(tag.toLowerCase())}">${escapeHTML(tag)}</button></li>`;
      }
      return `<li class="tag-chip${toneClass}">${escapeHTML(tag)}</li>`;
    })
    .join('')}${hiddenTagCount > 0 ? `<li class="tag-chip tag-chip-more">+${hiddenTagCount}</li>` : ''}</ul>`;
}

function selectedImageCount() {
  return state.selectedImageIDs instanceof Set ? state.selectedImageIDs.size : 0;
}

function visibleImageIDs() {
  return (Array.isArray(state.images) ? state.images : [])
    .map((item) => Number(item.image_id || 0))
    .filter((id) => Number.isFinite(id) && id > 0);
}

function pruneImageSelection() {
  if (!(state.selectedImageIDs instanceof Set)) {
    state.selectedImageIDs = new Set();
    return;
  }
  const visible = new Set(visibleImageIDs());
  for (const id of Array.from(state.selectedImageIDs)) {
    if (!visible.has(id)) {
      state.selectedImageIDs.delete(id);
    }
  }
}

function updateImageSelectionControls() {
  const active = Boolean(state.imageSelectionMode);
  const count = selectedImageCount();
  const visibleCount = visibleImageIDs().length;

  if (gallerySelectModeButton) {
    gallerySelectModeButton.textContent = active ? 'Done selecting' : 'Select';
    gallerySelectModeButton.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
  if (gallerySelectionToolbar) {
    gallerySelectionToolbar.hidden = !active;
  }
  if (gallerySelectionSummary) {
    gallerySelectionSummary.textContent = `${count} selected`;
  }
  if (gallerySelectPageButton) {
    gallerySelectPageButton.disabled = visibleCount === 0;
  }
  if (galleryClearSelectionButton) {
    galleryClearSelectionButton.disabled = count === 0;
  }
  if (galleryDeleteSelectedButton) {
    galleryDeleteSelectedButton.disabled = count === 0 || bulkImageDeleteInFlight;
    galleryDeleteSelectedButton.textContent = count > 0 ? `Delete selected (${count})` : 'Delete selected';
  }
  if (galleryDeleteAllButton) {
    galleryDeleteAllButton.disabled = Number(state.imagesTotal || 0) === 0 || bulkImageDeleteInFlight;
    const total = Number(state.imagesTotal || 0);
    galleryDeleteAllButton.textContent = total > 0 ? `Delete all images (${total})` : 'Delete all images';
  }
}

async function loadAllImageIDs() {
  const pageSize = 200;
  const ids = [];
  let offset = 0;
  let total = Number(state.imagesTotal || 0);

  do {
    const params = applyNSFWQuery(new URLSearchParams({ limit: String(pageSize), offset: String(offset) }));
    const response = await fetch(`/api/images?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'Could not select all images');
    }
    const images = Array.isArray(payload.images) ? payload.images : [];
    total = Number(payload.total || total || images.length);
    for (const item of images) {
      const id = Number(item && item.image_id);
      if (Number.isFinite(id) && id > 0) {
        ids.push(id);
      }
    }
    if (images.length === 0) {
      break;
    }
    offset += images.length;
  } while (offset < total);

  return ids;
}

function cardActionMarkup(item, mode, safeName, isNSFWTagged, canSearchSimilar, mediaType) {
  const actionLabel = mode === 'result' ? (item.is_anchor ? 'Anchor image' : mediaType === 'video' ? 'Use frame' : 'Use anchor') : 'Similar';
  const similarAriaLabel = mode === 'result'
    ? `${actionLabel} for ${safeName}`
    : `Find similar media for ${safeName}`;
  const disabled = canSearchSimilar ? '' : 'disabled';
  const title = canSearchSimilar ? (mode === 'result' ? '' : 'title="Find similar"') : 'title="Available after indexing finishes"';
  const deleteButton = mode === 'gallery'
    ? `<button class="ghost thumb-action danger delete-action" data-delete-kind="image" data-delete-id="${item.image_id}" data-delete-name="${safeName}" aria-label="Delete ${safeName}">Delete</button>`
    : mode === 'videos'
      ? `<button class="ghost thumb-action danger delete-action" data-delete-kind="video" data-delete-id="${item.video_id}" data-delete-name="${safeName}" aria-label="Delete ${safeName}">Delete</button>`
      : '';
  const nsfwToggleButton = mode === 'gallery'
    ? `<button class="ghost thumb-action nsfw-toggle-action${isNSFWTagged ? ' danger' : ''}" data-nsfw-kind="image" data-nsfw-id="${item.image_id}" data-nsfw-name="${safeName}" data-nsfw-current="${isNSFWTagged ? '1' : '0'}" title="${isNSFWTagged ? 'Unflag NSFW' : 'Flag NSFW'}" aria-label="${isNSFWTagged ? 'Unflag NSFW for' : 'Flag NSFW for'} ${safeName}">${isNSFWTagged ? 'Unflag' : 'Flag'}</button>`
    : mode === 'videos'
      ? `<button class="ghost thumb-action nsfw-toggle-action${isNSFWTagged ? ' danger' : ''}" data-nsfw-kind="video" data-nsfw-id="${item.video_id}" data-nsfw-name="${safeName}" data-nsfw-current="${isNSFWTagged ? '1' : '0'}" title="${isNSFWTagged ? 'Unflag NSFW' : 'Flag NSFW'}" aria-label="${isNSFWTagged ? 'Unflag NSFW for' : 'Flag NSFW for'} ${safeName}">${isNSFWTagged ? 'Unflag' : 'Flag'}</button>`
      : '';
  const reannotateButton = mode === 'gallery'
    ? `<button class="ghost thumb-action reannotate-action" data-reannotate-kind="image" data-reannotate-id="${item.image_id}" data-reannotate-name="${safeName}" title="Re-annotate" aria-label="Re-annotate ${safeName}">Annotate</button>`
    : mode === 'videos'
      ? `<button class="ghost thumb-action reannotate-action" data-reannotate-kind="video" data-reannotate-id="${item.video_id}" data-reannotate-name="${safeName}" title="Re-annotate" aria-label="Re-annotate ${safeName}">Annotate</button>`
      : '';
  const feedButton = feedActionMarkup(item, safeName, mediaType);

  return `
    ${feedButton}
    <button class="ghost thumb-action similar-action" data-image-id="${escapeHTML(String(item.image_id ?? ''))}" aria-label="${similarAriaLabel}" ${disabled} ${title}>${actionLabel}</button>
    ${nsfwToggleButton}
    ${reannotateButton}
    ${deleteButton}
  `;
}

function cardMarkup(item, mode) {
  const itemName = normalizeText(item.original_name) || normalizeText(item.storage_path) || 'Untitled media';
  const titleText = normalizeText(item.description) || itemName;
  const safeTitle = escapeHTML(titleText);
  const safeName = escapeHTML(itemName);
  const safePath = escapeHTML(item.storage_path);
  const mediaType = item.media_type || 'image';
  const safeMimeType = escapeHTML(item.mime_type || 'video/mp4');
  const thumbPath = mediaType === 'video' && item.preview_path ? item.preview_path : item.storage_path;
  const thumbURL = escapeHTML(toMediaURL(thumbPath));
  const mediaURL = escapeHTML(toMediaURL(item.storage_path));
  const supportEntries = supportEntriesForItem(item, mode)
    .filter((entry) => normalizeText(entry.text) !== titleText);
  const tags = Array.isArray(item.tags)
    ? item.tags
        .filter((tag) => typeof tag === 'string' && tag.trim() !== '')
        .map((tag) => tag.trim())
        .slice(0, 8)
    : [];
  const isNSFWTagged = tags.some((tag) => tag.toLowerCase() === 'nsfw');
  const status = item.index_state || 'done';
  const safeStatus = escapeHTML(humanizeIndexState(status));
  const scoreLabel = item.search_source === 'tag' ? '' : formatMatch(item.distance);
  const scoreBadgeLabel = scoreBadgeText(item, scoreLabel);
  const scoreBadge = scoreBadgeLabel && !item.is_anchor ? `<p class="thumb-match-badge">${escapeHTML(scoreBadgeLabel)}</p>` : '';
  const videoBadgeLabel = Number(item.duration_ms) > 0 ? formatTimestamp(item.duration_ms) : 'video';
  const canSearchSimilar = Number(item.image_id) > 0 && (status === 'done' || mediaType === 'video');
  const anchorBadge = item.is_anchor ? '<p class="state anchor">anchor</p>' : '';
  const videoMeta = videoMetaMarkup(item, mode);
  const actionsMarkup = cardActionMarkup(item, mode, safeName, isNSFWTagged, canSearchSimilar, mediaType);
  const supportMarkup = supportEntries.length > 0
    ? `<div class="supporting-stack">${supportEntries
      .map((entry) => `<p class="${entry.className}">${escapeHTML(entry.text)}</p>`)
      .join('')}</div>`
    : '';
  const compactTagsMarkup = tagsMarkup(tags, true, true);
  const compactTagsSection = compactTagsMarkup ? `<div class="meta-tags">${compactTagsMarkup}</div>` : '';
  const overlayTagsMarkup = tagsMarkup(tags, false, false);
  const overlaySupportMarkup = supportEntries.length > 0
    ? `<div class="overlay-supporting-stack">${supportEntries
      .map((entry) => `<p class="${entry.className} overlay-supporting-text">${escapeHTML(entry.text)}</p>`)
      .join('')}</div>`
    : '';
  const overlayVideoMeta = videoMeta
    ? `<div class="overlay-video-meta">${videoMeta}</div>`
    : '';
  const overlayStatusMarkup = `<div class="overlay-status-row"><p class="${stateClass(status)}">${safeStatus}</p>${anchorBadge}</div>`;
  const isSelected = mediaType !== 'video' && mode === 'gallery' && state.selectedImageIDs.has(Number(item.image_id || 0));
  const selectionClass = isSelected ? ' is-selected' : '';
  const previewWidth = Number(item.preview_width || 0);
  const previewHeight = Number(item.preview_height || 0);
  const hasPreviewDimensions = previewWidth > 0 && previewHeight > 0;
  const hasPreviewFrame = mediaType === 'video' && Boolean(item.preview_path);
  const mediaWidth = hasPreviewDimensions ? previewWidth : Number(item.width || 0);
  const mediaHeight = hasPreviewDimensions ? previewHeight : Number(item.height || 0);
  const hasVideoDimensions = mediaType === 'video' && mediaWidth > 0 && mediaHeight > 0;
  const isPortraitVideo = hasVideoDimensions && mediaHeight > mediaWidth;
  const shouldProbePortrait = hasPreviewFrame && !isPortraitVideo && !hasPreviewDimensions;
  const thumbWrapClass = `thumb-wrap${mediaType === 'video' ? ' thumb-wrap-video' : ''}${isPortraitVideo ? ' thumb-wrap-portrait-video' : ''}${shouldProbePortrait ? ' thumb-wrap-portrait-probe' : ''}`;
  const portraitBackdrop = hasPreviewFrame && (isPortraitVideo || shouldProbePortrait)
    ? `<img class="thumb-backdrop" src="${thumbURL}" alt="" aria-hidden="true" loading="lazy" />`
    : '';
  const thumbImageClass = `thumb-image${isPortraitVideo ? ' thumb-image-contain' : ''}`;
  const selectionControl = mode === 'gallery'
    ? `<label class="selection-control" title="Select image">
        <input type="checkbox" class="image-select-checkbox" data-image-id="${escapeHTML(String(item.image_id || ''))}" ${isSelected ? 'checked' : ''} aria-label="Select ${safeName}" />
        <span>Select</span>
      </label>`
    : '';

  return `
    <article class="card${selectionClass}">
      <div class="${thumbWrapClass}">
        ${selectionControl}
        <button
          type="button"
          class="thumb-button"
          data-media-type="${mediaType}"
          data-lightbox-src="${thumbURL}"
          data-lightbox-caption="${safeTitle}${mediaType === 'video' ? ' preview frame' : ''}"
          data-player-src="${mediaURL}"
          data-player-poster="${thumbURL}"
          data-player-type="${safeMimeType}"
          data-player-caption="${safeTitle}"
          aria-label="Open full ${mediaType === 'video' ? 'preview frame' : 'image'} for ${safeTitle}"
        >
          ${portraitBackdrop}
          <img class="${thumbImageClass}" src="${thumbURL}" alt="${safeTitle}" loading="lazy" />
          ${mediaType === 'video' ? `<span class="thumb-video-badge">${escapeHTML(videoBadgeLabel)}</span>` : ''}
        </button>
        <div class="thumb-actions" role="group" aria-label="Card actions for ${safeName}">
          ${actionsMarkup}
        </div>
        ${scoreBadge}
      </div>
      <div class="meta">
        <div class="meta-main">
          <h3>${safeTitle}</h3>
          ${titleText !== itemName ? `<p class="file-name">${safeName}</p>` : ''}
          <p class="path">${safePath}</p>
        </div>
        ${compactTagsSection}
        <div class="meta-foot">
          <div class="meta-status-row">
            <p class="${stateClass(status)}">${safeStatus}</p>
            ${anchorBadge}
          </div>
          ${supportMarkup}
        </div>
      </div>
      <div class="card-detail-overlay" aria-hidden="true">
        <p class="overlay-title">${safeTitle}</p>
        ${titleText !== itemName ? `<p class="overlay-file-name">${safeName}</p>` : ''}
        <p class="overlay-path">${safePath}</p>
        ${overlayVideoMeta}
        ${overlayTagsMarkup}
        ${overlayStatusMarkup}
        ${overlaySupportMarkup}
      </div>
    </article>
  `;
}

function loadingStateMarkup(message) {
  return `<div class="loading-state" role="status"><span class="loading-dot" aria-hidden="true"></span><span>${escapeHTML(message)}</span></div>`;
}

function renderMediaLoading(grid, message) {
  if (!grid) {
    return;
  }
  const skeletons = Array.from({ length: 6 }, () => '<article class="skeleton-card" aria-hidden="true"></article>').join('');
  grid.innerHTML = `${loadingStateMarkup(message).replace('loading-state', 'loading-state grid-loading')}${skeletons}`;
}

function emptyStateMarkup(message, actionMarkup) {
  return `<div class="empty-state"><p class="empty">${escapeHTML(message)}</p>${actionMarkup ? `<div class="empty-actions">${actionMarkup}</div>` : ''}</div>`;
}

function emptyActionForMode(mode) {
  if (mode === 'gallery' || mode === 'videos') {
    return '<button type="button" class="ghost empty-upload-action">Upload media</button>';
  }
  if (mode === 'result') {
    return '<button type="button" class="ghost empty-clear-results-action">Clear results</button>';
  }
  return '';
}

function renderMediaCollection(items, total, grid, mode, emptyPagedMessage, emptyMessage) {
  updateTabCounts();
  renderPagination();

  const safeItems = Array.isArray(items) ? items : [];
  if (safeItems.length === 0) {
    grid.innerHTML = total > 0
      ? emptyStateMarkup(emptyPagedMessage, '')
      : emptyStateMarkup(emptyMessage, emptyActionForMode(mode));
    return;
  }

  grid.innerHTML = safeItems.map((item) => cardMarkup(item, mode)).join('');
  applyLoadedPortraitVideoThumbnails(grid);
}

function updatePortraitVideoThumbnail(image) {
  if (!(image instanceof HTMLImageElement) || !image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) {
    return;
  }
  const wrap = image.closest('.thumb-wrap-portrait-probe');
  if (!wrap) {
    return;
  }
  if (image.naturalHeight > image.naturalWidth) {
    wrap.classList.add('thumb-wrap-portrait-video');
  } else {
    wrap.querySelector('.thumb-backdrop')?.remove();
  }
  wrap.classList.remove('thumb-wrap-portrait-probe');
}

function applyLoadedPortraitVideoThumbnails(root) {
  if (!root) {
    return;
  }
  root.querySelectorAll('.thumb-wrap-portrait-probe .thumb-image').forEach(updatePortraitVideoThumbnail);
}

function renderGallery() {
  pruneImageSelection();
  renderMediaCollection(
    state.images,
    state.imagesTotal,
    galleryGrid,
    'gallery',
    'No images are visible on this page yet. Move back toward newer pages.',
    'No images here yet. Import or upload something to start building the archive.',
  );
  galleryGrid.classList.toggle('selection-active', Boolean(state.imageSelectionMode));
  updateImageSelectionControls();
}

function renderVideos() {
  renderMediaCollection(
    state.videos,
    state.videosTotal,
    videosGrid,
    'videos',
    'No videos are visible on this page yet. Move back toward newer pages.',
    'No videos here yet. Upload a clip to start building the video archive.',
  );
}

function renderResults() {
  updateTabCounts();
  renderPagination();
  renderResultsProvenance();
  const resultItems = Array.isArray(state.results) ? state.results : [];
  clearResultsButton.disabled = resultItems.length === 0;

  if (resultItems.length === 0) {
    resultsGrid.innerHTML = emptyStateMarkup('Run a text search, tag search, or similar-image search to populate this view with image or video matches.', emptyActionForMode('result'));
    return;
  }

  resultsGrid.innerHTML = resultItems.map((item) => cardMarkup(item, 'result')).join('');
}

function renderResultsProvenance() {
  if (!resultsProvenance) {
    return;
  }
  if (state.resultsMode === 'none' || !state.resultsProvenance) {
    resultsProvenance.textContent = 'Latest text, tag, or similar-image search results stay here until you clear them.';
    return;
  }
  const total = state.resultsMode === 'tag' ? Number(state.resultsTotal || 0) : Number((state.results || []).length);
  resultsProvenance.textContent = `${state.resultsProvenance} - ${total} ${total === 1 ? 'match' : 'matches'}`;
}

function renderTagCloud() {
  if (!tagCloud) {
    return;
  }

  const tags = Array.isArray(state.tagCloud) ? state.tagCloud : [];
  if (tags.length === 0) {
    tagCloud.innerHTML = emptyStateMarkup('No tags yet. Let annotation finish and this panel will populate.', '<button type="button" class="ghost empty-retry-tags-action">Retry tags</button>');
    return;
  }

  const counts = tags.map((item) => Number(item.count || 0));
  const minCount = counts.length > 0 ? Math.min(...counts) : 0;
  const maxCount = counts.length > 0 ? Math.max(...counts) : 0;

  tagCloud.innerHTML = tags
    .map((item) => {
      const tag = typeof item.tag === 'string' ? item.tag : '';
      if (!tag) {
        return '';
      }
      const count = Number(item.count || 0);
      const sizeClass = tagCloudSizeClass(count, minCount, maxCount);
      return `<button type="button" class="tag-cloud-chip ${sizeClass}" data-tag="${escapeHTML(tag)}" role="listitem" aria-label="Search by tag ${escapeHTML(tag)} (${count})"><span class="tag-cloud-label">${escapeHTML(tag)}</span><span class="tag-cloud-count">${count}</span></button>`;
    })
    .join('');
}

async function loadTagCloud() {
  if (!tagCloud) {
    return;
  }
  tagCloud.innerHTML = loadingStateMarkup('Loading tags...');
  const params = applyNSFWQuery(new URLSearchParams({ limit: '80' }));
  const response = await fetch(`/api/search/tag-cloud?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Could not load tag cloud');
  }
  state.tagCloud = Array.isArray(payload.tags) ? payload.tags : [];
  tagCloudLoaded = true;
  renderTagCloud();
}

async function runTagSearch(tags, mode, page) {
  const normalizedTags = normalizeTagTerms(tags);
  if (normalizedTags.length === 0) {
    throw new Error('Select at least one tag first.');
  }

  const safeMode = mode === 'all' ? 'all' : 'any';
  const safePage = Number.isFinite(Number(page)) ? Math.max(0, Number(page)) : 0;
  state.resultsMode = 'tag';
  state.resultsProvenance = `Results for tags: ${normalizedTags.join(', ')} (${safeMode === 'all' ? 'all tags' : 'any tag'})`;
  renderResultsProvenance();
  resultsGrid.innerHTML = loadingStateMarkup('Loading tag results...');
  setActiveTab('results');
  const params = applyNSFWQuery(new URLSearchParams({ limit: String(state.resultsPageSize), offset: String(safePage * state.resultsPageSize) }));
  params.set('mode', safeMode);
  normalizedTags.forEach((tag) => params.append('tag', tag));

  const response = await fetch(`/api/search/tags?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'Tag search failed');
  }

  const pageResults = payload.results || [];
  const total = Number(payload.total || pageResults.length);
  const maxPage = Math.max(0, totalPages(total, state.resultsPageSize) - 1);
  if (pageResults.length === 0 && total > 0 && safePage > maxPage) {
    return runTagSearch(normalizedTags, safeMode, maxPage);
  }

  const imageByID = mediaByImageID();
  state.results = pageResults.map((item) => {
    const image = imageByID.get(item.image_id) || {};
    return {
      ...image,
      ...item,
      index_state: image.index_state || 'done',
      is_anchor: false,
    };
  });
  state.resultsMode = 'tag';
  state.resultsTotal = total;
  state.resultsPage = safePage;
  state.activeTagFilters = normalizedTags;
  state.activeTagMode = safeMode;
  state.resultsProvenance = `Results for tags: ${normalizedTags.join(', ')} (${safeMode === 'all' ? 'all tags' : 'any tag'})`;

  renderResults();
  setActiveTab('results');
  const start = total === 0 ? 0 : safePage * state.resultsPageSize + 1;
  const end = safePage * state.resultsPageSize + state.results.length;
  setStatus(searchStatus, `Tag search (${normalizedTags.join(', ')}) showing ${start}-${end} of ${total}.`, 'success');
}

function ensureTagCloudLoaded() {
  if (tagCloudLoaded) {
    return;
  }
  setStatus(tagsStatus, 'Loading top tags...', 'info');
  loadTagCloud()
    .then(() => {
      setStatus(tagsStatus, `Showing ${state.tagCloud.length} tag(s), ranked by frequency.`, 'success');
    })
    .catch((err) => {
      setStatus(tagsStatus, err.message || 'Tag cloud failed to load', 'error');
    });
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
    if (opsBar) {
      opsBar.dataset.state = 'idle';
    }
    setMastheadIndexStatus('Index: waiting', 'idle');
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
  const hasWork = failed > 0 || missing > 0 || annotationsMissing > 0 || pending > 0 || leased > 0;

  if (opsBar) {
    opsBar.dataset.state = hasWork ? 'active' : 'idle';
  }
  if (failed > 0) {
    setMastheadIndexStatus(`Index: ${failed} failed`, 'failed');
  } else if (hasWork) {
    const activeCount = pending + leased + missing + annotationsMissing;
    setMastheadIndexStatus(activeCount > 0 ? `Index: ${activeCount} queued` : 'Index active', 'active');
  } else {
    setMastheadIndexStatus('Index idle', 'idle');
  }

  statsProgress.style.width = `${pct}%`;
  setStatus(
    statsSummary,
    `Indexed ${done}/${total} - ${failed} failed - ${missing} missing - ${annotationsMissing} annotation gaps.`,
    failed > 0 ? 'error' : (missing > 0 || annotationsMissing > 0) ? 'info' : 'success',
  );

  const imagesTotal = Number(payload.images_total || state.imagesTotal || 0);
  statsDetail.textContent = `Library ${imagesTotal} images - Pending ${pending} - Processing ${leased}.`;
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
  if (kind === 'feed') {
    document.body.classList.toggle('feed-open', active);
  }

  if (!shell) {
    return;
  }

  const hasOverlay = Boolean(
    (uploadModal && !uploadModal.hidden)
    || (lightbox && !lightbox.hidden)
    || (videoModal && !videoModal.hidden)
    || (videoFeed && !videoFeed.hidden),
  );
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
  ).filter((element) => !element.hasAttribute('hidden') && !element.closest('[hidden], [inert], [aria-hidden="true"]'));
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

function feedIsOpen() {
  return Boolean(videoFeed && !videoFeed.hidden);
}

function feedVideoElements() {
  if (!feedTrack) {
    return [feedPrevVideo, feedVideo, feedNextVideo].filter(Boolean);
  }
  return Array.from(feedTrack.querySelectorAll('.video-feed-video'));
}

function feedSlide(role) {
  return feedTrack ? feedTrack.querySelector(`.feed-slide-${role}`) : null;
}

function feedVideoForRole(role) {
  return feedSlide(role)?.querySelector('.video-feed-video') || null;
}

function activeFeedVideo() {
  return feedVideoForRole('current') || document.getElementById('video-feed-player');
}

function assignFeedVideoRole(video, role) {
  if (!video) {
    return;
  }
  if (role === 'current') {
    video.id = 'video-feed-player';
    video.preload = 'auto';
    video.autoplay = true;
    video.muted = feedState.muted;
    return;
  }
  video.id = role === 'prev' ? 'video-feed-prev-player' : 'video-feed-next-player';
  video.preload = 'metadata';
  video.autoplay = false;
  video.muted = true;
}

function feedVideoID(item) {
  const id = Number(item && item.video_id);
  return Number.isFinite(id) && id > 0 ? id : 0;
}

function feedImageID(item) {
  const id = Number(item && item.image_id);
  return Number.isFinite(id) && id > 0 ? id : 0;
}

function normalizeFeedItem(item) {
  const source = item && typeof item === 'object' ? item : {};
  const tags = Array.isArray(source.tags)
    ? source.tags.filter((tag) => typeof tag === 'string' && tag.trim() !== '').map((tag) => tag.trim())
    : [];
  return {
    ...source,
    media_type: 'video',
    video_id: feedVideoID(source),
    image_id: feedImageID(source),
    storage_path: typeof source.storage_path === 'string' ? source.storage_path : '',
    preview_path: typeof source.preview_path === 'string' ? source.preview_path : '',
    original_name: normalizeText(source.original_name) || `Video #${feedVideoID(source) || '?'}`,
    description: normalizeText(source.description),
    transcript_text: normalizeText(source.transcript_text),
    mime_type: source.mime_type || 'video/mp4',
    duration_ms: Number(source.duration_ms || 0),
    tags,
    index_state: source.index_state || 'done',
  };
}

function feedItemTitle(item) {
  return normalizeText(item.description) || normalizeText(item.original_name) || `Video #${feedVideoID(item)}`;
}

function newFeedSessionID() {
  return `feed-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function classifyFeedFeedback(metrics) {
  const playbackStarted = Boolean(metrics && metrics.playbackStarted);
  const action = metrics && typeof metrics.action === 'string' ? metrics.action : '';
  const watchMs = Math.max(0, Number(metrics && metrics.watchMs) || 0);
  const dwellMs = Math.max(0, Number(metrics && metrics.dwellMs) || 0);
  const completionRatio = Math.max(0, Math.min(1, Number(metrics && metrics.completionRatio) || 0));

  if (!playbackStarted) {
    return { signal: 'neutral', reason: 'playback-not-started' };
  }
  if (completionRatio >= 0.75 || action === 'ended') {
    return { signal: 'positive', reason: 'watch-through' };
  }
  if (action === 'next' && watchMs < 3000 && dwellMs < 3000 && completionRatio < 0.2) {
    return { signal: 'soft-negative', reason: 'quick-skip' };
  }
  return { signal: 'neutral', reason: 'ambiguous' };
}

if (typeof window !== 'undefined') {
  window.__feedTestClassify = classifyFeedFeedback;
}

function markFeedPlaybackStarted() {
  if (!feedIsOpen()) {
    return;
  }
  feedState.playbackStarted = true;
  if (feedState.watchStartedAt <= 0) {
    feedState.watchStartedAt = performance.now();
  }
  updateFeedControlLabels();
}

function accumulateFeedWatchTime() {
  if (feedState.watchStartedAt <= 0) {
    return;
  }
  feedState.accumulatedWatchMs += Math.max(0, performance.now() - feedState.watchStartedAt);
  feedState.watchStartedAt = 0;
}

function feedCompletionRatio(item) {
  const currentVideo = activeFeedVideo();
  if (!currentVideo) {
    return 0;
  }
  const durationSeconds = Number.isFinite(currentVideo.duration) && currentVideo.duration > 0
    ? currentVideo.duration
    : Number(item && item.duration_ms) / 1000;
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, Number(currentVideo.currentTime || 0) / durationSeconds));
}

function currentFeedMetrics(action) {
  const now = performance.now();
  const current = feedState.queue[feedState.currentIndex] || {};
  const activeWatchMs = feedState.playbackStarted && feedState.watchStartedAt > 0
    ? Math.max(0, now - feedState.watchStartedAt)
    : 0;
  return {
    action,
    playbackStarted: feedState.playbackStarted,
    watchMs: feedState.accumulatedWatchMs + activeWatchMs,
    dwellMs: feedState.itemStartedAt > 0 ? Math.max(0, now - feedState.itemStartedAt) : 0,
    completionRatio: feedCompletionRatio(current),
  };
}

function feedItemAffinity(item) {
  if (!(feedState.tagScores instanceof Map) || feedState.tagScores.size === 0) {
    return 0;
  }
  const tags = Array.isArray(item.tags) ? item.tags : [];
  return tags.reduce((score, tag) => score + (feedState.tagScores.get(tag.toLowerCase()) || 0), 0);
}

function feedItemPreferenceRankScore(item, order) {
  const distance = Number(item && item.distance);
  const visualScore = Number.isFinite(distance) ? -distance : -order;
  const affinityNudge = Math.max(-0.08, Math.min(0.08, feedItemAffinity(item) * 0.02));
  return visualScore + affinityNudge;
}

function clampFeedTagScore(score) {
  return Math.max(feedTagScoreMin, Math.min(feedTagScoreMax, Number(score) || 0));
}

function decayFeedTagScores() {
  if (!(feedState.tagScores instanceof Map) || feedState.tagScores.size === 0) {
    return;
  }
  for (const [tag, score] of feedState.tagScores.entries()) {
    const nextScore = clampFeedTagScore(score * feedTagScoreDecay);
    if (Math.abs(nextScore) < 0.05) {
      feedState.tagScores.delete(tag);
    } else {
      feedState.tagScores.set(tag, nextScore);
    }
  }
}

function feedPreferenceTags(direction) {
  if (!(feedState.tagScores instanceof Map) || feedState.tagScores.size === 0) {
    return [];
  }
  return Array.from(feedState.tagScores.entries())
    .filter(([, score]) => direction === 'prefer' ? score >= feedPreferenceThreshold : score <= -feedPreferenceThreshold)
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]) || left[0].localeCompare(right[0]))
    .slice(0, 12)
    .map(([tag]) => tag);
}

function feedExcludedVideoIDs() {
  const excluded = new Set(feedState.seenVideoIDs || []);
  for (const id of feedState.rejectedCandidateIDs || []) {
    excluded.add(id);
  }
  for (const item of feedState.queue || []) {
    const id = feedVideoID(item);
    if (id > 0) {
      excluded.add(id);
    }
  }
  return Array.from(excluded).filter((id) => Number(id) > 0);
}

function rerankFutureFeedItems(currentIndex) {
  const start = currentIndex + 2;
  if (start >= feedState.queue.length) {
    return;
  }
  const ranked = feedState.queue.slice(start)
    .map((item, order) => ({ item, order, score: feedItemPreferenceRankScore(item, order) }))
    .sort((left, right) => (right.score - left.score) || (left.order - right.order));
  feedState.queue.splice(start, ranked.length, ...ranked.map((entry) => entry.item));
}

function recordFeedFeedback(action) {
  const current = feedState.queue[feedState.currentIndex];
  if (!current) {
    return { signal: 'neutral', reason: 'missing-current' };
  }
  if (feedState.feedbackRecordedIndex === feedState.currentIndex) {
    return { signal: 'neutral', reason: 'already-recorded' };
  }
  feedState.feedbackRecordedIndex = feedState.currentIndex;
  const metrics = currentFeedMetrics(action);
  const result = classifyFeedFeedback(metrics);
  feedState.events.push({
    session_id: feedState.sessionID,
    video_id: feedVideoID(current),
    queue_position: feedState.currentIndex + 1,
    playback_started: metrics.playbackStarted,
    watch_ms: Math.round(metrics.watchMs),
    dwell_ms: Math.round(metrics.dwellMs),
    video_duration_ms: Math.round(Number(current.duration_ms || 0) || (activeFeedVideo() && Number.isFinite(activeFeedVideo().duration) ? activeFeedVideo().duration * 1000 : 0)),
    completion_ratio: metrics.completionRatio,
    action,
    signal: result.signal,
    timestamp: new Date().toISOString(),
  });
  const weight = result.signal === 'positive' ? 1 : result.signal === 'soft-negative' ? -0.5 : 0;
  if (weight !== 0) {
    for (const tag of current.tags || []) {
      const key = String(tag).toLowerCase();
      feedState.tagScores.set(key, clampFeedTagScore((feedState.tagScores.get(key) || 0) + weight));
    }
    rerankFutureFeedItems(feedState.currentIndex);
  }
  return result;
}

function updateFeedQueueMetadata() {
  if (videoFeed) {
    videoFeed.dataset.currentIndex = String(feedState.currentIndex);
    videoFeed.dataset.total = String(feedState.queue.length);
  }
  if (feedPosition) {
    feedPosition.textContent = `${feedState.currentIndex + 1} / ${feedState.queue.length}`;
  }
}

function updateFeedProgress() {
  if (!feedProgressBar) {
    return;
  }
  const current = feedState.queue[feedState.currentIndex] || {};
  feedProgressBar.style.width = `${Math.round(feedCompletionRatio(current) * 100)}%`;
}

function feedViewportHeight() {
  if (videoFeed && videoFeed.clientHeight > 0) {
    return videoFeed.clientHeight;
  }
  return Math.max(1, window.innerHeight || 1);
}

function setFeedTrackOffset(offsetY, animate) {
  if (!feedTrack) {
    return false;
  }
  const shouldAnimate = Boolean(animate && !prefersReducedMotion());
  feedTrack.style.transition = shouldAnimate ? `transform ${feedSnapDurationMS}ms cubic-bezier(0.22, 0.72, 0.18, 1)` : 'none';
  feedTrack.style.transform = `translate3d(0, ${Math.round(offsetY)}px, 0)`;
  return shouldAnimate;
}

function resetFeedTrack() {
  feedState.snapToken += 1;
  if (feedState.snapTimer !== null) {
    window.clearTimeout(feedState.snapTimer);
    feedState.snapTimer = null;
  }
  feedState.dragging = false;
  feedState.touchActive = false;
  feedState.isPaging = false;
  if (videoFeed) {
    delete videoFeed.dataset.dragging;
  }
  setFeedTrackOffset(0, false);
}

function clearFeedVideoElement(video) {
  if (!video) {
    return;
  }
  video.pause();
  video.removeAttribute('src');
  video.poster = '';
  video.load();
}

function setFeedVideoElement(video, item, autoplay, preservePlayback = false) {
  if (!video) {
    return;
  }
  if (!item) {
    clearFeedVideoElement(video);
    return;
  }
  const source = toMediaURL(item.storage_path);
  const poster = item.preview_path ? toMediaURL(item.preview_path) : '';
  if (!source) {
    clearFeedVideoElement(video);
    return;
  }
  if (!preservePlayback) {
    video.pause();
  }
  video.muted = true;
  video.playsInline = true;
  video.autoplay = Boolean(autoplay);
  video.poster = poster;
  if (video.getAttribute('src') !== source) {
    video.setAttribute('src', source);
    video.load();
  }
  if (autoplay) {
    video.muted = feedState.muted;
    void video.play().catch(() => {
      updateFeedControlLabels();
    });
  }
}

function renderFeedSlides() {
  const current = feedState.queue[feedState.currentIndex] || null;
  const previous = feedState.currentIndex > 0 ? feedState.queue[feedState.currentIndex - 1] : null;
  const next = feedState.currentIndex < feedState.queue.length - 1 ? feedState.queue[feedState.currentIndex + 1] : null;
  assignFeedVideoRole(feedVideoForRole('current'), 'current');
  assignFeedVideoRole(feedVideoForRole('prev'), 'prev');
  assignFeedVideoRole(feedVideoForRole('next'), 'next');
  setFeedVideoElement(feedVideoForRole('current'), current, true);
  setFeedVideoElement(feedVideoForRole('prev'), previous, false);
  setFeedVideoElement(feedVideoForRole('next'), next, false);
}

function warmFeedNeighbor(direction) {
  const video = direction === 'next' ? feedVideoForRole('next') : feedVideoForRole('prev');
  if (!video || !video.getAttribute('src')) {
    return;
  }
  video.muted = true;
  video.playsInline = true;
  void video.play().catch(() => {});
}

function pauseFeedNeighbors() {
  feedVideoForRole('prev')?.pause();
  feedVideoForRole('next')?.pause();
}

function promoteFeedVideo(direction) {
  const currentSlide = feedSlide('current');
  const prevSlide = feedSlide('prev');
  const nextSlide = feedSlide('next');
  const currentVideo = feedVideoForRole('current');
  const previousVideo = feedVideoForRole('prev');
  const nextVideo = feedVideoForRole('next');
  if (!currentSlide || !prevSlide || !nextSlide || !currentVideo || !previousVideo || !nextVideo) {
    renderFeedSlides();
    return;
  }

  if (direction === 'next') {
    currentVideo.pause();
    prevSlide.append(currentVideo);
    currentSlide.append(nextVideo);
    nextSlide.append(previousVideo);
  } else {
    currentVideo.pause();
    nextSlide.append(currentVideo);
    currentSlide.append(previousVideo);
    prevSlide.append(nextVideo);
  }

  assignFeedVideoRole(feedVideoForRole('current'), 'current');
  assignFeedVideoRole(feedVideoForRole('prev'), 'prev');
  assignFeedVideoRole(feedVideoForRole('next'), 'next');
  const current = feedState.queue[feedState.currentIndex] || null;
  const previous = feedState.currentIndex > 0 ? feedState.queue[feedState.currentIndex - 1] : null;
  const next = feedState.currentIndex < feedState.queue.length - 1 ? feedState.queue[feedState.currentIndex + 1] : null;
  setFeedVideoElement(feedVideoForRole('current'), current, true, true);
  setFeedVideoElement(feedVideoForRole('prev'), previous, false);
  setFeedVideoElement(feedVideoForRole('next'), next, false);
}

function updateFeedControlLabels() {
  const currentVideo = activeFeedVideo();
  if (feedPlayToggleButton && currentVideo) {
    const paused = currentVideo.paused;
    feedPlayToggleButton.textContent = paused ? 'Play' : 'Pause';
    feedPlayToggleButton.setAttribute('aria-label', paused ? 'Play video' : 'Pause video');
  }
  if (feedMuteToggleButton && currentVideo) {
    const muted = Boolean(currentVideo.muted);
    feedMuteToggleButton.textContent = muted ? 'Unmute' : 'Mute';
    feedMuteToggleButton.setAttribute('aria-label', muted ? 'Unmute video' : 'Mute video');
  }
  if (feedPrevButton) {
    feedPrevButton.disabled = feedState.currentIndex <= 0;
  }
}

function setFeedChromeActive(active) {
  for (const element of [feedTopbar, feedControls]) {
    if (!element) {
      continue;
    }
    element.hidden = !active;
    element.querySelectorAll('button').forEach((button) => {
      button.disabled = !active;
    });
    if (active) {
      element.removeAttribute('inert');
      element.removeAttribute('aria-hidden');
    } else {
      element.setAttribute('inert', '');
      element.setAttribute('aria-hidden', 'true');
    }
  }
}

function renderFeedItem(options = {}) {
  if (!feedIsOpen() || !activeFeedVideo()) {
    return;
  }
  const current = feedState.queue[feedState.currentIndex];
  if (!current) {
    showFeedEnd();
    return;
  }

  feedState.playbackStarted = false;
  feedState.itemStartedAt = performance.now();
  feedState.watchStartedAt = 0;
  feedState.accumulatedWatchMs = 0;
  feedState.seenVideoIDs.add(feedVideoID(current));

  if (videoFeed) {
    updateFeedQueueMetadata();
  }
  if (feedEnd) {
    feedEnd.hidden = true;
  }
  setFeedChromeActive(true);
  if (feedInfo) {
    feedInfo.hidden = false;
  }
  if (feedHint) {
    if (feedState.currentIndex > 0) {
      setLocalPreference(feedHintSeenStorageKey, '1');
    }
    feedHint.hidden = feedState.currentIndex > 0 || localPreference(feedHintSeenStorageKey) === '1';
  }
  if (feedSeedLabel) {
    feedSeedLabel.textContent = 'Similar in your library';
  }
  announceLiveChange(`Now showing ${feedItemTitle(current)}, video ${feedState.currentIndex + 1} of ${feedState.queue.length}.`);

  resetFeedTrack();
  if (options.renderSlides !== false) {
    renderFeedSlides();
  }
  updateFeedProgress();
  updateFeedControlLabels();
  const currentVideo = activeFeedVideo();
  if (currentVideo && !currentVideo.paused) {
    markFeedPlaybackStarted();
  }
  void ensureFeedLookahead();
}

function showFeedEnd() {
  if (!videoFeed) {
    return;
  }
  resetFeedTrack();
  feedVideoElements().forEach((video) => video.pause());
  if (feedInfo) {
    feedInfo.hidden = true;
  }
  setFeedChromeActive(false);
  if (feedEnd) {
    feedEnd.hidden = false;
  }
  announceLiveChange('No more similar videos.');
  videoFeed.dataset.currentIndex = String(Math.max(0, feedState.queue.length - 1));
  videoFeed.dataset.total = String(feedState.queue.length);
  if (feedEndBackButton) {
    feedEndBackButton.focus();
  }
}

function closeVideoFeed() {
  if (!videoFeed || videoFeed.hidden) {
    return;
  }
  accumulateFeedWatchTime();
  feedVideoElements().forEach(clearFeedVideoElement);
  videoFeed.hidden = true;
  setFeedChromeActive(true);
  resetFeedTrack();
  videoFeed.removeAttribute('data-current-index');
  videoFeed.removeAttribute('data-total');
  setOverlayState('feed', false);
  feedState.queue = [];
  feedState.currentIndex = 0;
  feedState.loading = false;
  feedState.loadingMore = false;
  feedState.exhausted = false;
  feedState.candidateRequestToken += 1;
  feedState.feedbackRecordedIndex = -1;
  feedState.lookaheadPromise = null;
  feedState.rejectedCandidateIDs = new Set();
  feedState.playbackStarted = false;
  feedState.itemStartedAt = 0;
  feedState.watchStartedAt = 0;
  feedState.accumulatedWatchMs = 0;
  feedState.sessionID = '';
  feedState.events = [];
  feedState.seenVideoIDs = new Set();
  feedState.tagScores = new Map();
  feedState.touchStartX = 0;
  feedState.touchStartY = 0;
  feedState.touchCurrentY = 0;
  feedState.touchStartedAt = 0;
  feedState.touchActive = false;
  feedState.suppressClickUntil = 0;
  if (lastFocusedElement) {
    lastFocusedElement.focus();
  }
  announceLiveChange('Closed similar video feed.');
  lastFocusedElement = null;
}

function findFeedSeedItem(button) {
  const videoID = Number(button.dataset.feedVideoId || 0);
  const imageID = Number(button.dataset.feedImageId || 0);
  const pools = [state.results, state.videos];
  if (imageID > 0) {
    for (const pool of pools) {
      const found = (Array.isArray(pool) ? pool : []).find((item) => feedVideoID(item) === videoID && feedImageID(item) === imageID);
      if (found) {
        return normalizeFeedItem(found);
      }
    }
  }
  for (const pool of pools) {
    const found = (Array.isArray(pool) ? pool : []).find((item) => feedVideoID(item) === videoID);
    if (found) {
      return normalizeFeedItem(found);
    }
  }
  return normalizeFeedItem({
    video_id: videoID,
    image_id: imageID,
    original_name: button.dataset.feedName || `Video #${videoID}`,
    storage_path: button.dataset.feedStoragePath || '',
    preview_path: button.dataset.feedPreviewPath || '',
    mime_type: button.dataset.feedMimeType || 'video/mp4',
  });
}

async function fetchSimilarVideoCandidates(seedItem, limit) {
  decayFeedTagScores();
  const params = applyNSFWQuery(new URLSearchParams({
    video_id: String(feedVideoID(seedItem)),
    limit: String(limit),
  }));
  const imageID = feedImageID(seedItem);
  if (imageID > 0) {
    params.set('seed_image_id', String(imageID));
  }
  const seen = feedExcludedVideoIDs();
  if (seen.length > 0) {
    params.set('seen', seen.join(','));
  }
  const preferTags = feedPreferenceTags('prefer');
  if (preferTags.length > 0) {
    params.set('prefer_tags', preferTags.join(','));
  }
  const avoidTags = feedPreferenceTags('avoid');
  if (avoidTags.length > 0) {
    params.set('avoid_tags', avoidTags.join(','));
  }
  const response = await fetch(`/api/search/similar-videos?${params.toString()}`);
  const rawBody = await response.text();
  let payload = {};
  if (rawBody) {
    try {
      payload = JSON.parse(rawBody);
    } catch {
      payload = { error: rawBody };
    }
  }
  if (!response.ok) {
    throw new Error(payload.error || response.statusText || 'Similar video feed failed');
  }
  return Array.isArray(payload.results) ? payload.results.map(normalizeFeedItem) : [];
}

function appendFeedCandidates(candidates) {
  const queuedIDs = new Set((feedState.queue || []).map(feedVideoID).filter((id) => id > 0));
  let appended = 0;
  for (const candidate of candidates || []) {
    const id = feedVideoID(candidate);
    if (id <= 0 || queuedIDs.has(id)) {
      continue;
    }
    if (!candidate.storage_path || !canPlayFeedItem(candidate)) {
      feedState.rejectedCandidateIDs.add(id);
      continue;
    }
    queuedIDs.add(id);
    feedState.queue.push(candidate);
    appended += 1;
  }
  return appended;
}

async function ensureFeedLookahead() {
  if (!feedIsOpen() || feedState.loading || feedState.exhausted) {
    return 0;
  }
  if (feedState.loadingMore && feedState.lookaheadPromise) {
    return feedState.lookaheadPromise;
  }
  const remainingAhead = feedState.queue.length - feedState.currentIndex - 1;
  if (remainingAhead > feedFetchAheadThreshold) {
    return 0;
  }

  feedState.loadingMore = true;
  const token = ++feedState.candidateRequestToken;
  feedState.lookaheadPromise = (async () => {
    try {
      const seed = feedState.queue[0];
      if (!seed || feedVideoID(seed) <= 0) {
        return 0;
      }
      const candidates = await fetchSimilarVideoCandidates(seed, feedBatchSize);
      if (token !== feedState.candidateRequestToken || !feedIsOpen()) {
        return 0;
      }
      const beforeLength = feedState.queue.length;
      const appended = appendFeedCandidates(candidates);
      if (appended === 0 && candidates.length === 0) {
        feedState.exhausted = true;
      }
      if (appended > 0) {
        updateFeedQueueMetadata();
        if (feedState.currentIndex >= beforeLength - 1) {
          renderFeedSlides();
        }
      }
      return appended;
    } catch {
      return 0;
    } finally {
      if (token === feedState.candidateRequestToken) {
        feedState.loadingMore = false;
        feedState.lookaheadPromise = null;
      }
    }
  })();
  return feedState.lookaheadPromise;
}

async function startVideoFeed(seedItem) {
  if (!videoFeed || !activeFeedVideo() || feedState.loading) {
    return;
  }
  const seed = normalizeFeedItem(seedItem);
  if (feedVideoID(seed) <= 0 || !seed.storage_path) {
    throw new Error('Could not resolve selected video.');
  }
  if (!canPlayFeedItem(seed)) {
    throw new Error('This video format cannot play in the browser feed.');
  }

  feedState.loading = true;
  feedState.queue = [seed];
  feedState.currentIndex = 0;
  feedState.sessionID = newFeedSessionID();
  feedState.events = [];
  feedState.seenVideoIDs = new Set([feedVideoID(seed)]);
  feedState.tagScores = new Map();
  feedState.loadingMore = false;
  feedState.exhausted = false;
  feedState.feedbackRecordedIndex = -1;
  feedState.lookaheadPromise = null;
  feedState.rejectedCandidateIDs = new Set();
  const token = ++feedState.candidateRequestToken;
  const candidates = await fetchSimilarVideoCandidates(seed, feedInitialCandidateLimit);
  if (token !== feedState.candidateRequestToken) {
    return;
  }
  const appended = appendFeedCandidates(candidates);
  feedState.exhausted = appended === 0 && candidates.length === 0;

  lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  videoFeed.hidden = false;
  setOverlayState('feed', true);
  feedState.loading = false;
  renderFeedItem();
  if (feedNextButton) {
    feedNextButton.focus();
  }
}

function advanceVideoFeed(action) {
  if (!feedIsOpen()) {
    return;
  }
  snapFeedPager('next', action || 'next');
}

function previousVideoFeed() {
  if (!feedIsOpen()) {
    return;
  }
  snapFeedPager('prev', 'prev');
}

function toggleFeedPlayback() {
  const currentVideo = activeFeedVideo();
  if (!currentVideo) {
    return;
  }
  if (currentVideo.paused) {
    void currentVideo.play().catch(() => updateFeedControlLabels());
  } else {
    currentVideo.pause();
  }
  updateFeedControlLabels();
}

function snapFeedPager(direction, action) {
  if (!feedIsOpen() || feedState.isPaging) {
    return;
  }
  const isNext = direction === 'next';
  if (!isNext && feedState.currentIndex <= 0) {
    settleFeedPagerBack();
    return;
  }
  accumulateFeedWatchTime();
  recordFeedFeedback(action || (isNext ? 'next' : 'prev'));
  if (isNext && feedState.currentIndex >= feedState.queue.length - 1) {
    feedState.isPaging = true;
    void ensureFeedLookahead().then((appended) => {
      feedState.isPaging = false;
      if (!feedIsOpen()) {
        return;
      }
      if (appended > 0 && feedState.currentIndex < feedState.queue.length - 1) {
        snapFeedPager('next', action || 'next');
        return;
      }
      if (feedState.exhausted) {
        showFeedEnd();
      } else {
        settleFeedPagerBack();
      }
    });
    return;
  }

  feedState.isPaging = true;
  const snapToken = ++feedState.snapToken;
  feedState.dragging = false;
  if (videoFeed) {
    delete videoFeed.dataset.dragging;
  }
  warmFeedNeighbor(isNext ? 'next' : 'prev');
  const animated = setFeedTrackOffset(isNext ? -feedViewportHeight() : feedViewportHeight(), true);
  const finish = () => {
    if (!feedState.isPaging || feedState.snapToken !== snapToken) {
      return;
    }
    feedState.isPaging = false;
    feedState.currentIndex += isNext ? 1 : -1;
    promoteFeedVideo(isNext ? 'next' : 'prev');
    renderFeedItem({ renderSlides: false });
  };
  if (feedState.snapTimer !== null) {
    window.clearTimeout(feedState.snapTimer);
  }
  feedState.snapTimer = window.setTimeout(finish, (animated ? feedSnapDurationMS : 0) + 60);
  if (feedTrack) {
    feedTrack.addEventListener('transitionend', finish, { once: true });
  }
}

function settleFeedPagerBack() {
  if (!feedIsOpen()) {
    return;
  }
  feedState.dragging = false;
  if (videoFeed) {
    delete videoFeed.dataset.dragging;
  }
  setFeedTrackOffset(0, true);
  pauseFeedNeighbors();
}

function attachFeedHandler(target, statusTarget) {
  target.addEventListener('click', async (event) => {
    const button = event.target.closest('.feed-action');
    if (!button || button.disabled || !target.contains(button)) {
      return;
    }

    try {
      setStatus(statusTarget, 'Loading similar video feed...', 'info');
      await startVideoFeed(findFeedSeedItem(button));
      setStatus(statusTarget, `Loaded ${feedState.queue.length} similar video(s).`, 'success');
    } catch (err) {
      feedState.loading = false;
      setStatus(statusTarget, err.message || 'Could not start video feed', 'error');
    }
  });
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

async function loadMediaCollection(config) {
  const offset = state[config.pageKey] * state[config.pageSizeKey];
  const requestToken = ++collectionRequestTokens[config.tokenKey];
  setStatus(config.statusTarget, config.loadingMessage, 'info');
  renderMediaLoading(config.grid, config.loadingMessage);

  const params = applyNSFWQuery(new URLSearchParams({ limit: String(state[config.pageSizeKey]), offset: String(offset) }));
  const response = await fetch(`${config.endpoint}?${params.toString()}`);
  const payload = await response.json();
  if (requestToken !== collectionRequestTokens[config.tokenKey]) {
    return;
  }
  if (!response.ok) {
    throw new Error(payload.error || config.errorMessage);
  }

  const nextItems = payload[config.payloadKey] || [];
  const nextTotal = Number(payload.total || nextItems.length);
  const maxPage = Math.max(0, Math.ceil(nextTotal / state[config.pageSizeKey]) - 1);
  if (nextItems.length === 0 && nextTotal > 0 && state[config.pageKey] > maxPage) {
    state[config.pageKey] = maxPage;
    return loadMediaCollection(config);
  }

  state[config.itemsKey] = nextItems;
  state[config.totalKey] = nextTotal;
  config.render();

  const start = nextTotal === 0 ? 0 : offset + 1;
  const end = offset + nextItems.length;
  setStatus(config.statusTarget, `Showing ${start}-${end} of ${nextTotal} ${config.label}(s), newest first.`, 'success');
}

async function loadImages() {
  return loadMediaCollection({
    tokenKey: 'images',
    endpoint: '/api/images',
    payloadKey: 'images',
    itemsKey: 'images',
    totalKey: 'imagesTotal',
    pageKey: 'galleryPage',
    pageSizeKey: 'galleryPageSize',
    grid: galleryGrid,
    statusTarget: galleryStatus,
    loadingMessage: 'Loading gallery...',
    errorMessage: 'Could not load images',
    label: 'image',
    render: renderGallery,
  });
}

async function loadVideos() {
  return loadMediaCollection({
    tokenKey: 'videos',
    endpoint: '/api/videos',
    payloadKey: 'videos',
    itemsKey: 'videos',
    totalKey: 'videosTotal',
    pageKey: 'videosPage',
    pageSizeKey: 'videosPageSize',
    grid: videosGrid,
    statusTarget: videosStatus,
    loadingMessage: 'Loading videos...',
    errorMessage: 'Could not load videos',
    label: 'video',
    render: renderVideos,
  });
}

async function runTextSearch(query, negative, tagFilters, tagMode) {
  state.resultsMode = 'text';
  state.resultsProvenance = `Results for "${query}"`;
  renderResultsProvenance();
  resultsGrid.innerHTML = loadingStateMarkup('Searching images and videos...');
  setActiveTab('results');
  const params = applyNSFWQuery(new URLSearchParams({ q: query, limit: '24' }));
  if (negative) {
    params.set('neg', negative);
  }
  const normalizedTags = normalizeTagTerms(tagFilters);
  if (normalizedTags.length > 0) {
    normalizedTags.forEach((tag) => params.append('tag', tag));
    params.set('tag_mode', tagMode);
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
  state.resultsMode = 'text';
  state.resultsTotal = state.results.length;
  state.resultsPage = 0;
  state.activeTagFilters = [];
  const filters = [];
  if (negative) {
    filters.push(`excluding "${negative}"`);
  }
  if (normalizedTags.length > 0) {
    filters.push(`${tagMode === 'all' ? 'all' : 'any'} tags: ${normalizedTags.join(', ')}`);
  }
  state.resultsProvenance = `Results for "${query}"${filters.length > 0 ? ` (${filters.join('; ')})` : ''}`;
  renderResults();
  setActiveTab('results');
  setStatus(searchStatus, `Text search returned ${state.results.length} result(s).${formatSearchDebugSuffix(payload.debug)}`, 'success');
}

async function runSimilarSearch(imageID, anchorHint) {
  state.resultsMode = 'similar';
  state.resultsProvenance = `Similar images for #${imageID}`;
  renderResultsProvenance();
  resultsGrid.innerHTML = loadingStateMarkup('Searching similar images...');
  setActiveTab('results');
  const params = applyNSFWQuery(new URLSearchParams({ image_id: String(imageID), limit: '24' }));
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
  state.resultsMode = 'similar';
  state.resultsTotal = state.results.length;
  state.resultsPage = 0;
  state.activeTagFilters = [];
  state.resultsProvenance = `Similar images for #${imageID}`;
  renderResults();
  setActiveTab('results');
  setStatus(searchStatus, `Similar search returned ${state.results.length} result(s).${formatSearchDebugSuffix(payload.debug)}`, 'success');
}

function mediaItemsDiffer(currentItems, nextItems, idKey) {
  if (!Array.isArray(currentItems) || !Array.isArray(nextItems)) {
    return true;
  }
  if (currentItems.length !== nextItems.length) {
    return true;
  }
  for (let i = 0; i < currentItems.length; i += 1) {
    const current = currentItems[i] || {};
    const next = nextItems[i] || {};
    if (Number(current[idKey] || 0) !== Number(next[idKey] || 0)) {
      return true;
    }
    if ((current.index_state || '') !== (next.index_state || '')) {
      return true;
    }
    if ((current.storage_path || '') !== (next.storage_path || '')) {
      return true;
    }
    if ((current.thumbnail_path || '') !== (next.thumbnail_path || '')) {
      return true;
    }
    if ((current.preview_path || '') !== (next.preview_path || '')) {
      return true;
    }
    if (Number(current.preview_width || 0) !== Number(next.preview_width || 0)) {
      return true;
    }
    if (Number(current.preview_height || 0) !== Number(next.preview_height || 0)) {
      return true;
    }
    if (Number(current.width || 0) !== Number(next.width || 0)) {
      return true;
    }
    if (Number(current.height || 0) !== Number(next.height || 0)) {
      return true;
    }
    if ((current.description || '') !== (next.description || '')) {
      return true;
    }
    if ((current.transcript_text || '') !== (next.transcript_text || '')) {
      return true;
    }
    if (JSON.stringify(current.tags || []) !== JSON.stringify(next.tags || [])) {
      return true;
    }
  }
  return false;
}

function imagesDiffer(currentImages, nextImages) {
  return mediaItemsDiffer(currentImages, nextImages, 'image_id');
}

function videosDiffer(currentVideos, nextVideos) {
  return mediaItemsDiffer(currentVideos, nextVideos, 'video_id');
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

  if (payload.videos && state.videosPage === 0) {
    const nextVideos = (payload.videos.videos || []).slice(0, state.videosPageSize);
    const nextTotal = Number(payload.videos.total ?? nextVideos.length);
    if (videosDiffer(state.videos, nextVideos) || Number(state.videosTotal || 0) !== nextTotal) {
      state.videos = nextVideos;
      state.videosTotal = nextTotal;
      renderVideos();
      if (state.videosTotal === 0) {
        setStatus(videosStatus, 'No videos here yet. Upload or import supported clips to build the video index.', 'info');
      } else {
        const end = state.videos.length;
        setStatus(videosStatus, `Showing 1-${end} of ${state.videosTotal} video(s), newest first.`, 'success');
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
  const params = applyNSFWQuery(new URLSearchParams());
  const query = params.toString();
  const wsPath = query ? `/api/live?${query}` : '/api/live';
  const socket = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);
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
    if (liveSocket !== socket) {
      return;
    }
    liveSocket = null;
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

function restartLiveUpdates() {
  clearLiveReconnectTimer();
  stopPollingFallback();
  if (liveSocket) {
    const socket = liveSocket;
    liveSocket = null;
    socket.close();
  }
  connectLiveUpdates();
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
  await deleteMediaRequest(kind, id);
  return true;
}

async function deleteMediaRequest(kind, id) {
  const label = kind === 'video' ? 'video' : 'image';
  const response = await fetch(`/api/${kind === 'video' ? 'videos' : 'images'}/${id}`, { method: 'DELETE' });
  if (response.status === 204) {
    return;
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

async function deleteSelectedImages() {
  if (bulkImageDeleteInFlight) {
    return;
  }
  const ids = Array.from(state.selectedImageIDs || [])
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id) && id > 0);
  if (ids.length === 0) {
    return;
  }
  if (!window.confirm(`Delete ${ids.length} selected image${ids.length === 1 ? '' : 's'}? This cannot be undone.`)) {
    return;
  }

  await deleteImageIDs(ids, 'selected');
}

async function deleteAllImages() {
  if (bulkImageDeleteInFlight) {
    return;
  }
  const total = Number(state.imagesTotal || 0);
  if (total <= 0) {
    return;
  }
  if (!window.confirm(`Delete all ${total} image${total === 1 ? '' : 's'} matching the current gallery filter? This cannot be undone.`)) {
    return;
  }

  setStatus(galleryStatus, 'Loading all image IDs for deletion...', 'info');
  const ids = await loadAllImageIDs();
  if (ids.length === 0) {
    setStatus(galleryStatus, 'No images found to delete.', 'info');
    return;
  }
  await deleteImageIDs(ids, 'all');
}

async function deleteImageIDs(ids, scopeLabel) {
  bulkImageDeleteInFlight = true;
  updateImageSelectionControls();
  setStatus(galleryStatus, `Deleting ${ids.length} image${ids.length === 1 ? '' : 's'}...`, 'info');
  let deleted = 0;
  const deletedIDs = [];
  const failures = [];
  try {
    for (const id of ids) {
      try {
        await deleteMediaRequest('image', id);
        deleted += 1;
        deletedIDs.push(id);
        state.selectedImageIDs.delete(id);
      } catch (err) {
        failures.push({ id, message: err instanceof Error ? err.message : 'delete failed' });
      }
    }

    state.results = state.results.filter((item) => !deletedIDs.includes(Number(item.image_id || 0)));
    renderResults();
    await loadImages();
    await loadStats();

    if (failures.length > 0) {
      const summary = failures.slice(0, 3).map((item) => `#${item.id}: ${item.message}`).join('; ');
      setStatus(galleryStatus, `Deleted ${deleted}; ${failures.length} failed${summary ? ` (${summary})` : ''}.`, 'error');
      return;
    }
    const label = scopeLabel === 'all' ? '' : ' selected';
    setStatus(galleryStatus, `Deleted ${deleted}${label} image${deleted === 1 ? '' : 's'}.`, 'success');
  } finally {
    bulkImageDeleteInFlight = false;
    updateImageSelectionControls();
  }
}

async function reannotateMedia(kind, id) {
  const path = kind === 'video' ? 'videos' : 'images';
  const response = await fetch(`/api/${path}/${id}/reannotate`, { method: 'POST' });
  if (response.status === 202) {
    return true;
  }
  let message = `Could not re-annotate ${kind === 'video' ? 'video' : 'image'}`;
  try {
    const payload = await response.json();
    if (payload && payload.error) {
      message = payload.error;
    }
  } catch {
    // fall back to generic message
  }
  throw new Error(message);
}

async function toggleMediaNSFW(kind, id) {
  const path = kind === 'video' ? 'videos' : 'images';
  const label = kind === 'video' ? 'video' : 'image';
  const response = await fetch(`/api/${path}/${id}/toggle-nsfw`, { method: 'POST' });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new Error((payload && payload.error) || `Could not toggle NSFW for ${label}`);
  }
  return Boolean(payload && payload.is_nsfw);
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
    if (event.target.closest('.thumb-actions') || event.target.closest('.selection-control')) {
      return;
    }

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

function attachImageSelectionHandler(target) {
  target.addEventListener('change', (event) => {
    const checkbox = event.target.closest('.image-select-checkbox');
    if (!checkbox || !target.contains(checkbox)) {
      return;
    }

    const id = Number(checkbox.dataset.imageId || 0);
    if (!Number.isFinite(id) || id <= 0) {
      return;
    }
    if (checkbox.checked) {
      state.selectedImageIDs.add(id);
    } else {
      state.selectedImageIDs.delete(id);
    }
    renderGallery();
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

function attachReannotateHandler(target) {
  target.addEventListener('click', async (event) => {
    const trigger = event.target.closest('.reannotate-action');
    if (!trigger || !target.contains(trigger) || trigger.disabled) {
      return;
    }

    const kind = trigger.dataset.reannotateKind || 'image';
    const id = Number(trigger.dataset.reannotateId || 0);
    const name = trigger.dataset.reannotateName || '';
    if (!Number.isFinite(id) || id <= 0) {
      return;
    }

    trigger.disabled = true;
    const statusTarget = kind === 'video' ? videosStatus : galleryStatus;
    const label = kind === 'video' ? 'video' : 'image';
    setStatus(statusTarget, `Queueing ${label} re-annotation for "${name || `${label} #${id}`}"...`, 'info');

    try {
      await reannotateMedia(kind, id);
      await Promise.all([loadImages(), loadVideos(), loadStats()]);
      setStatus(statusTarget, `Queued ${label} re-annotation for "${name || `${label} #${id}`}".`, 'success');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Re-annotation failed';
      setStatus(statusTarget, message, 'error');
    } finally {
      trigger.disabled = false;
    }
  });
}

function attachNSFWToggleHandler(target) {
  target.addEventListener('click', async (event) => {
    const trigger = event.target.closest('.nsfw-toggle-action');
    if (!trigger || !target.contains(trigger) || trigger.disabled) {
      return;
    }

    const kind = trigger.dataset.nsfwKind || 'image';
    const id = Number(trigger.dataset.nsfwId || 0);
    const name = trigger.dataset.nsfwName || '';
    const wasNSFW = trigger.dataset.nsfwCurrent === '1';
    if (!Number.isFinite(id) || id <= 0) {
      return;
    }

    trigger.disabled = true;
    const statusTarget = kind === 'video' ? videosStatus : galleryStatus;
    const label = kind === 'video' ? 'video' : 'image';
    setStatus(statusTarget, `${wasNSFW ? 'Removing' : 'Applying'} NSFW tag for "${name || `${label} #${id}`}"...`, 'info');

    try {
      const isNSFW = await toggleMediaNSFW(kind, id);
      const refreshTasks = [loadImages(), loadVideos(), loadStats()];
      if (tagCloudLoaded) {
        refreshTasks.push(loadTagCloud());
      }
      await Promise.all(refreshTasks);
      if (isNSFW) {
        setStatus(statusTarget, `Marked ${label} "${name || `${label} #${id}`}" as NSFW.`, 'success');
      } else {
        setStatus(statusTarget, `Removed NSFW tag from ${label} "${name || `${label} #${id}`}".`, 'success');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'NSFW toggle failed';
      setStatus(statusTarget, message, 'error');
    } finally {
      trigger.disabled = false;
    }
  });
}

function attachTagChipSearchHandler(target) {
  target.addEventListener('click', async (event) => {
    const trigger = event.target.closest('.tag-chip-button');
    if (!trigger || !target.contains(trigger)) {
      return;
    }

    const tag = typeof trigger.dataset.tagSearch === 'string' ? trigger.dataset.tagSearch.trim().toLowerCase() : '';
    if (!tag) {
      return;
    }

    setStatus(searchStatus, `Searching by tag: ${tag}...`, 'info');
    try {
      await runTagSearch([tag], 'any', 0);
    } catch (err) {
      setStatus(searchStatus, err.message || 'Tag search failed', 'error');
    }
  });
}

if (showNSFWToggle) {
  let storedPreference = '';
  try {
    storedPreference = window.localStorage.getItem(nsfwPreferenceStorageKey) || '';
  } catch {
    storedPreference = '';
  }
  applyNSFWPreference(storedPreference === '1');

  showNSFWToggle.addEventListener('change', async () => {
    applyNSFWPreference(showNSFWToggle.checked);
    try {
      window.localStorage.setItem(nsfwPreferenceStorageKey, state.showNSFW ? '1' : '0');
    } catch {}

    restartLiveUpdates();

    const refreshTagResults = state.resultsMode === 'tag' && Array.isArray(state.activeTagFilters) && state.activeTagFilters.length > 0;
    try {
      await Promise.all([loadImages(), loadVideos()]);
      if (tagCloudLoaded) {
        await loadTagCloud();
        setStatus(tagsStatus, `Showing ${state.tagCloud.length} tag(s), ranked by frequency.`, 'success');
      }
      if (refreshTagResults) {
        await runTagSearch(state.activeTagFilters, state.activeTagMode, state.resultsPage);
      } else if (state.resultsMode !== 'none') {
        setStatus(searchStatus, 'NSFW visibility updated. Run search again to refresh non-tag results.', 'info');
      }
    } catch (err) {
      setStatus(galleryStatus, err.message || 'Could not refresh NSFW filter', 'error');
    }
  });
}

galleryTabButton.addEventListener('click', () => setActiveTab('gallery'));
videosTabButton.addEventListener('click', () => setActiveTab('videos'));
tagsTabButton.addEventListener('click', () => setActiveTab('tags'));
resultsTabButton.addEventListener('click', () => setActiveTab('results'));

const tabButtons = [galleryTabButton, videosTabButton, tagsTabButton, resultsTabButton];
tabButtons.forEach((button, index) => {
  button.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      return;
    }
    event.preventDefault();
    let nextIndex = index;
    if (event.key === 'ArrowLeft') {
      nextIndex = (index - 1 + tabButtons.length) % tabButtons.length;
    } else if (event.key === 'ArrowRight') {
      nextIndex = (index + 1) % tabButtons.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = tabButtons.length - 1;
    }
    while (tabButtons[nextIndex].disabled && nextIndex !== index) {
      if (event.key === 'ArrowLeft' || event.key === 'End') {
        nextIndex = (nextIndex - 1 + tabButtons.length) % tabButtons.length;
      } else {
        nextIndex = (nextIndex + 1) % tabButtons.length;
      }
    }
    const nextButton = tabButtons[nextIndex];
    if (nextButton.disabled) {
      return;
    }
    nextButton.focus();
    if (nextButton === galleryTabButton) {
      setActiveTab('gallery');
    } else if (nextButton === videosTabButton) {
      setActiveTab('videos');
    } else if (nextButton === tagsTabButton) {
      setActiveTab('tags');
    } else {
      setActiveTab('results');
    }
  });
});

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

if (resultsPrevButton) {
  resultsPrevButton.addEventListener('click', async () => {
    if (state.resultsMode !== 'tag' || state.resultsPage <= 0) {
      return;
    }
    state.resultsPage -= 1;
    try {
      await runTagSearch(state.activeTagFilters, state.activeTagMode, state.resultsPage);
    } catch (err) {
      setStatus(searchStatus, err.message || 'Could not load previous tag results page', 'error');
    }
  });
}

if (resultsNextButton) {
  resultsNextButton.addEventListener('click', async () => {
    if (state.resultsMode !== 'tag') {
      return;
    }
    if (state.resultsPage >= totalPages(state.resultsTotal, state.resultsPageSize) - 1) {
      return;
    }
    state.resultsPage += 1;
    try {
      await runTagSearch(state.activeTagFilters, state.activeTagMode, state.resultsPage);
    } catch (err) {
      setStatus(searchStatus, err.message || 'Could not load next tag results page', 'error');
    }
  });
}

uploadOpenButton.addEventListener('click', openUploadModal);
uploadCloseButton.addEventListener('click', closeUploadModal);

document.addEventListener('click', (event) => {
  if (event.target.closest('.empty-upload-action')) {
    openUploadModal();
    return;
  }
  if (event.target.closest('.empty-clear-results-action')) {
    state.results = [];
    state.resultsTotal = 0;
    state.resultsPage = 0;
    state.resultsMode = 'none';
    state.resultsProvenance = '';
    state.activeTagFilters = [];
    renderResults();
    setActiveTab('gallery');
    return;
  }
  if (event.target.closest('.empty-retry-tags-action')) {
    loadTagCloud().catch((err) => setStatus(tagsStatus, err.message || 'Tag cloud failed to load', 'error'));
  }
});

document.addEventListener('load', (event) => {
  updatePortraitVideoThumbnail(event.target);
}, true);

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
    const uploadItems = Array.isArray(payload.uploads) ? payload.uploads : [];
    const uploadErrors = uploadItems.filter((item) => item && item.error);
    const errorSummary = uploadErrors.slice(0, 3).map((item) => {
      const name = item.filename || 'file';
      return `${name}: ${item.error}`;
    }).join('; ');
    if (!response.ok) {
      throw new Error(errorSummary || payload.error || 'Upload failed');
    }

    const created = Number(payload.created) || 0;
    const duplicates = Number(payload.duplicates) || 0;
    const failed = Number(payload.failed) || uploadErrors.length;
    let message = '';
    if (files.length === 1) {
      if (failed > 0) {
        message = errorSummary || `${files[0].name} failed to upload.`;
      } else {
        message = duplicates > 0 && created === 0 ? `${files[0].name} already exists.` : `${files[0].name} queued for indexing.`;
      }
    } else {
      const parts = [];
      if (created > 0) {
        parts.push(`${created} queued`);
      }
      if (duplicates > 0) {
        parts.push(`${duplicates} already existed`);
      }
      if (failed > 0) {
        parts.push(`${failed} failed${errorSummary ? ` (${errorSummary})` : ''}`);
      }
      message = parts.length > 0 ? `Upload complete: ${parts.join(', ')}.` : 'Upload complete.';
    }

    setStatus(uploadStatus, message, failed > 0 ? 'error' : 'success');
    if (created > 0 || duplicates > 0) {
      uploadForm.reset();
      state.galleryPage = 0;
      state.videosPage = 0;
      setActiveTab('gallery');

      if (!isLiveSocketActive()) {
        await loadImages();
        await loadVideos();
        await loadStats();
      }
    }

    if (failed === 0) {
      window.setTimeout(() => {
        closeUploadModal();
      }, 450);
    }
  } catch (err) {
    setStatus(uploadStatus, err.message || 'Upload failed', 'error');
  }
});

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const q = textQuery.value.trim();
  const neg = negativeQuery ? negativeQuery.value.trim() : '';
  const pendingTagTerms = searchTagInput ? parseTagTerms(searchTagInput.value) : [];
  if (pendingTagTerms.length > 0) {
    addSearchTagFilters(pendingTagTerms);
    if (searchTagInput) {
      searchTagInput.value = '';
    }
    clearTagSuggestions();
  }
  const tagFilters = normalizeTagTerms([...(state.searchTagFilters || []), ...pendingTagTerms]);
  const tagModeValue = searchTagMode && searchTagMode.value === 'any' ? 'any' : 'all';
  if (!q) {
    setStatus(searchStatus, 'Enter a search phrase first.', 'error');
    return;
  }

  if (neg && tagFilters.length > 0) {
    setStatus(searchStatus, `Searching for "${q}" excluding "${neg}" with ${tagModeValue === 'all' ? 'all' : 'any'} tags (${tagFilters.join(', ')})...`, 'info');
  } else if (neg) {
    setStatus(searchStatus, `Searching for "${q}" excluding "${neg}"...`, 'info');
  } else if (tagFilters.length > 0) {
    setStatus(searchStatus, `Searching for "${q}" with ${tagModeValue === 'all' ? 'all' : 'any'} tags (${tagFilters.join(', ')})...`, 'info');
  } else {
    setStatus(searchStatus, `Searching for "${q}"...`, 'info');
  }

  try {
    await runTextSearch(q, neg, tagFilters, tagModeValue);
  } catch (err) {
    setStatus(searchStatus, err.message || 'Text search failed', 'error');
  }
});

if (negativeQuery) {
  negativeQuery.addEventListener('input', renderActiveSearchFilters);
}

if (searchTagChips) {
  searchTagChips.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-tag-remove]');
    if (!trigger || !searchTagChips.contains(trigger)) {
      return;
    }
    const tag = typeof trigger.dataset.tagRemove === 'string' ? trigger.dataset.tagRemove.trim().toLowerCase() : '';
    if (!tag) {
      return;
    }
    removeSearchTagFilter(tag);
    if (searchTagInput && searchTagInput.value.trim() !== '') {
      scheduleTagSuggestions(searchTagInput.value);
    } else {
      clearTagSuggestions();
    }
  });
}

if (searchTagSuggestions) {
  searchTagSuggestions.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-tag-suggestion]');
    if (!trigger || !searchTagSuggestions.contains(trigger)) {
      return;
    }
    const tag = typeof trigger.dataset.tagSuggestion === 'string' ? trigger.dataset.tagSuggestion.trim().toLowerCase() : '';
    if (!tag) {
      return;
    }
    addSearchTagFilters([tag]);
    if (searchTagInput) {
      searchTagInput.value = '';
      searchTagInput.focus();
    }
    clearTagSuggestions();
  });
}

if (searchTagInput) {
  searchTagInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ',') {
      event.preventDefault();
      const parts = parseTagTerms(searchTagInput.value);
      if (parts.length > 0) {
        addSearchTagFilters(parts);
      }
      searchTagInput.value = '';
      clearTagSuggestions();
      return;
    }

    if (event.key === 'Escape') {
      clearTagSuggestions();
      return;
    }

    if (event.key === 'Backspace' && searchTagInput.value.trim() === '' && state.searchTagFilters.length > 0) {
      const trailingTag = state.searchTagFilters[state.searchTagFilters.length - 1];
      removeSearchTagFilter(trailingTag);
    }
  });

  searchTagInput.addEventListener('input', () => {
    scheduleTagSuggestions(searchTagInput.value);
  });

  searchTagInput.addEventListener('focus', () => {
    scheduleTagSuggestions(searchTagInput.value);
  });

  searchTagInput.addEventListener('blur', () => {
    window.setTimeout(() => {
      clearTagSuggestions();
    }, 120);
  });
}

if (tagCloud) {
  tagCloud.addEventListener('click', async (event) => {
    const trigger = event.target.closest('.tag-cloud-chip');
    if (!trigger || !tagCloud.contains(trigger)) {
      return;
    }

    const tag = typeof trigger.dataset.tag === 'string' ? trigger.dataset.tag.trim().toLowerCase() : '';
    if (!tag) {
      return;
    }

    setStatus(searchStatus, `Searching by tag: ${tag}...`, 'info');
    try {
      await runTagSearch([tag], 'any');
    } catch (err) {
      setStatus(searchStatus, err.message || 'Tag search failed', 'error');
    }
  });
}

refreshImagesButton.addEventListener('click', () => {
  loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Refresh failed', 'error'));
  loadVideos().catch((err) => setStatus(videosStatus, err.message || 'Video refresh failed', 'error'));
});

if (gallerySelectModeButton) {
  gallerySelectModeButton.addEventListener('click', () => {
    state.imageSelectionMode = !state.imageSelectionMode;
    if (!state.imageSelectionMode) {
      state.selectedImageIDs.clear();
    }
    renderGallery();
  });
}

if (gallerySelectPageButton) {
  gallerySelectPageButton.addEventListener('click', () => {
    for (const id of visibleImageIDs()) {
      state.selectedImageIDs.add(id);
    }
    renderGallery();
  });
}

if (galleryClearSelectionButton) {
  galleryClearSelectionButton.addEventListener('click', () => {
    state.selectedImageIDs.clear();
    renderGallery();
  });
}

if (galleryDeleteSelectedButton) {
  galleryDeleteSelectedButton.addEventListener('click', () => {
    deleteSelectedImages().catch((err) => setStatus(galleryStatus, err.message || 'Bulk delete failed', 'error'));
  });
}

if (galleryDeleteAllButton) {
  galleryDeleteAllButton.addEventListener('click', () => {
    deleteAllImages().catch((err) => setStatus(galleryStatus, err.message || 'Delete all failed', 'error'));
  });
}

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
  state.resultsTotal = 0;
  state.resultsPage = 0;
  state.resultsMode = 'none';
  state.resultsProvenance = '';
  state.activeTagFilters = [];
  renderResults();
  setActiveTab('gallery');
  setStatus(searchStatus, 'Search results cleared.', 'info');
});

attachSimilarHandler(galleryGrid);
attachSimilarHandler(videosGrid);
attachSimilarHandler(resultsGrid);
attachFeedHandler(videosGrid, videosStatus);
attachFeedHandler(resultsGrid, searchStatus);
attachTagChipSearchHandler(galleryGrid);
attachTagChipSearchHandler(videosGrid);
attachTagChipSearchHandler(resultsGrid);
attachDeleteHandler(galleryGrid);
attachDeleteHandler(videosGrid);
attachImageSelectionHandler(galleryGrid);
attachNSFWToggleHandler(galleryGrid);
attachNSFWToggleHandler(videosGrid);
attachReannotateHandler(galleryGrid);
attachReannotateHandler(videosGrid);
attachLightboxHandler(galleryGrid);
attachLightboxHandler(videosGrid);
attachLightboxHandler(resultsGrid);

if (lightboxCloseButton) {
  lightboxCloseButton.addEventListener('click', closeLightbox);
}

if (videoCloseButton) {
  videoCloseButton.addEventListener('click', closeVideoPlayer);
}

if (feedExitButton) {
  feedExitButton.addEventListener('click', closeVideoFeed);
}

if (feedEndBackButton) {
  feedEndBackButton.addEventListener('click', closeVideoFeed);
}

if (feedNextButton) {
  feedNextButton.addEventListener('click', () => advanceVideoFeed('next'));
}

if (feedPrevButton) {
  feedPrevButton.addEventListener('click', previousVideoFeed);
}

if (feedPlayToggleButton && feedVideo) {
  feedPlayToggleButton.addEventListener('click', () => {
    toggleFeedPlayback();
  });
}

if (feedMuteToggleButton && activeFeedVideo()) {
  feedMuteToggleButton.addEventListener('click', () => {
    const currentVideo = activeFeedVideo();
    feedState.muted = !currentVideo.muted;
    currentVideo.muted = feedState.muted;
    updateFeedControlLabels();
  });
}

feedVideoElements().forEach((video) => {
  video.addEventListener('play', () => {
    if (video === activeFeedVideo()) {
      markFeedPlaybackStarted();
    }
  });
  video.addEventListener('playing', () => {
    if (video === activeFeedVideo()) {
      markFeedPlaybackStarted();
    }
  });
  video.addEventListener('pause', () => {
    if (video === activeFeedVideo()) {
      accumulateFeedWatchTime();
      updateFeedControlLabels();
    }
  });
  video.addEventListener('timeupdate', () => {
    if (video === activeFeedVideo()) {
      updateFeedProgress();
    }
  });
  video.addEventListener('loadedmetadata', () => {
    if (video === activeFeedVideo()) {
      updateFeedProgress();
    }
  });
  video.addEventListener('ended', () => {
    if (video === activeFeedVideo()) {
      advanceVideoFeed('ended');
    }
  });
  video.addEventListener('error', () => {
    if (video === activeFeedVideo()) {
      updateFeedControlLabels();
    }
  });
});

if (videoFeed) {
  videoFeed.addEventListener('touchstart', (event) => {
    if (event.target.closest('.feed-control') || feedState.isPaging || (feedEnd && !feedEnd.hidden)) {
      feedState.touchActive = false;
      return;
    }
    const touch = event.changedTouches && event.changedTouches[0];
    if (!touch) {
      return;
    }
    feedState.touchStartX = touch.clientX;
    feedState.touchStartY = touch.clientY;
    feedState.touchCurrentY = touch.clientY;
    feedState.touchStartedAt = performance.now();
    feedState.touchActive = true;
    feedState.dragging = false;
  }, { passive: true });
  videoFeed.addEventListener('touchmove', (event) => {
    if (!feedState.touchActive) {
      return;
    }
    const touch = event.changedTouches && event.changedTouches[0];
    if (!touch) {
      return;
    }
    const dx = touch.clientX - feedState.touchStartX;
    const dy = touch.clientY - feedState.touchStartY;
    if (!feedState.dragging) {
      if (Math.abs(dy) < 8 || Math.abs(dy) < Math.abs(dx) * 1.15) {
        return;
      }
      feedState.dragging = true;
      if (videoFeed) {
        videoFeed.dataset.dragging = '1';
      }
    }
    event.preventDefault();
    feedState.touchCurrentY = touch.clientY;
    const height = feedViewportHeight();
    const hasNext = feedState.currentIndex < feedState.queue.length - 1;
    const hasPrevious = feedState.currentIndex > 0;
    const resisted = (dy < 0 && !hasNext) || (dy > 0 && !hasPrevious) ? dy * 0.28 : dy;
    const offset = Math.max(-height, Math.min(height, resisted));
    if (dy < 0 && hasNext) {
      warmFeedNeighbor('next');
    } else if (dy > 0 && hasPrevious) {
      warmFeedNeighbor('prev');
    }
    setFeedTrackOffset(offset, false);
  }, { passive: false });
  videoFeed.addEventListener('touchend', (event) => {
    if (!feedState.touchActive) {
      return;
    }
    feedState.touchActive = false;
    if (!feedState.dragging) {
      return;
    }
    const touch = event.changedTouches && event.changedTouches[0];
    const endY = touch ? touch.clientY : feedState.touchCurrentY;
    const dy = endY - feedState.touchStartY;
    const elapsed = Math.max(1, performance.now() - feedState.touchStartedAt);
    const velocity = dy / elapsed;
    const threshold = Math.max(72, feedViewportHeight() * 0.16);
    event.preventDefault();
    feedState.suppressClickUntil = performance.now() + 350;
    if ((dy < -threshold || velocity < -0.65) && feedState.currentIndex < feedState.queue.length - 1) {
      advanceVideoFeed('next');
    } else if ((dy > threshold || velocity > 0.65) && feedState.currentIndex > 0) {
      previousVideoFeed();
    } else if (dy < -threshold && feedState.currentIndex >= feedState.queue.length - 1) {
      advanceVideoFeed('next');
    } else {
      settleFeedPagerBack();
    }
  }, { passive: false });
  videoFeed.addEventListener('touchcancel', () => {
    feedState.touchActive = false;
    feedState.suppressClickUntil = performance.now() + 350;
    settleFeedPagerBack();
  });
  videoFeed.addEventListener('click', (event) => {
    if (!feedIsOpen() || feedState.isPaging || (feedEnd && !feedEnd.hidden)) {
      return;
    }
    if (performance.now() < feedState.suppressClickUntil) {
      return;
    }
    if (event.target.closest('.feed-control, .feed-topbar, .feed-info, .feed-end')) {
      return;
    }
    toggleFeedPlayback();
  });
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
  if (videoFeed && !videoFeed.hidden) {
    trapFocus(event, videoFeed);
    if (event.key === 'Escape') {
      event.preventDefault();
      closeVideoFeed();
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      advanceVideoFeed('next');
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      previousVideoFeed();
      return;
    }
    const shortcutTarget = event.target instanceof HTMLElement
      ? event.target.closest('button, input, select, textarea, [role="button"], [contenteditable="true"]')
      : null;
    const currentVideo = activeFeedVideo();
    if ((event.key === ' ' || event.key.toLowerCase() === 'k') && currentVideo && !shortcutTarget) {
      event.preventDefault();
      if (currentVideo.paused) {
        void currentVideo.play().catch(() => updateFeedControlLabels());
      } else {
        currentVideo.pause();
      }
      updateFeedControlLabels();
      return;
    }
  } else if (lightbox && !lightbox.hidden) {
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
renderTagCloud();
renderSearchTagFilters();
renderStats();
setActiveTab('gallery');

loadImages().catch((err) => setStatus(galleryStatus, err.message || 'Initial load failed', 'error'));
loadVideos().catch((err) => setStatus(videosStatus, err.message || 'Initial video load failed', 'error'));
loadStats().catch((err) => setStatus(statsSummary, err.message || 'Initial status load failed', 'error'));
connectLiveUpdates();
