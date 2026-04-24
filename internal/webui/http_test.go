package webui

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRootServesIndexPage(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusOK)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "text/html") {
		t.Fatalf("expected html content type, got %q", ct)
	}
	body := rr.Body.String()
	if !strings.Contains(body, "<title>imgsearch</title>") {
		t.Fatalf("expected html title, got body=%s", body)
	}
	if !strings.Contains(body, "id=\"upload-form\"") {
		t.Fatalf("expected upload form in page")
	}
	if !strings.Contains(body, "id=\"negative-query\"") {
		t.Fatalf("expected optional negative prompt field in page")
	}
	if !strings.Contains(body, "id=\"image-lightbox\"") {
		t.Fatalf("expected lightbox container in page")
	}
	if !strings.Contains(body, "video.min.js") || !strings.Contains(body, "id=\"video-player\"") {
		t.Fatalf("expected Video.js player assets in page")
	}
	if !strings.Contains(body, "id=\"live-connection\"") {
		t.Fatalf("expected live connection indicator in page")
	}
	if strings.Contains(body, "class=\"masthead panel\"") {
		t.Fatalf("expected masthead to be compact chrome without panel treatment")
	}
	if strings.Contains(body, "<section id=\"ops-bar\"") {
		t.Fatalf("expected live status strip to be integrated into masthead, not rendered as separate section")
	}
	if strings.Contains(body, "id=\"ops-bar\" class=\"masthead-status\"") {
		t.Fatalf("expected no inline masthead status strip below the search row")
	}
	if !strings.Contains(body, "id=\"ops-bar\" class=\"ops-status-panel\"") {
		t.Fatalf("expected status plumbing to move into the top-right overflow menu")
	}
	if !strings.Contains(body, "class=\"search-actions\"") || !strings.Contains(body, "class=\"ops-menu\"") {
		t.Fatalf("expected compact top-right ops menu alongside search controls")
	}
	if !strings.Contains(body, "<details class=\"search-advanced\"") {
		t.Fatalf("expected exclude query to live behind advanced search disclosure")
	}
	if !strings.Contains(body, "<details class=\"ops-menu\"") {
		t.Fatalf("expected retry and refresh actions to be wrapped in a quiet ops menu")
	}
	if !strings.Contains(body, "id=\"tab-tags\"") || !strings.Contains(body, "aria-controls=\"tags-panel\"") {
		t.Fatalf("expected workspace tablist to include dedicated tags tab")
	}
	if !strings.Contains(body, "id=\"tags-panel\"") || !strings.Contains(body, "id=\"tag-cloud\"") {
		t.Fatalf("expected tags panel shell with tag cloud container")
	}
	if !strings.Contains(body, "id=\"results-prev\"") || !strings.Contains(body, "id=\"results-next\"") || !strings.Contains(body, "id=\"results-page-label\"") {
		t.Fatalf("expected results pagination controls in results toolbar")
	}
	if !strings.Contains(body, "id=\"show-nsfw\"") {
		t.Fatalf("expected nsfw visibility toggle in workspace controls")
	}
	if !strings.Contains(body, "class=\"workspace-nav\"") {
		t.Fatalf("expected compact workspace nav wrapper for title and tabs")
	}
	if !strings.Contains(body, "id=\"library-view-title\" class=\"sr-only\"") {
		t.Fatalf("expected library view heading to stay accessible while reducing visible chrome")
	}
	if !strings.Contains(body, "id=\"search-tag-input\"") || !strings.Contains(body, "id=\"search-tag-suggestions\"") {
		t.Fatalf("expected advanced search tag filter controls with autocomplete shell")
	}
	if strings.Contains(body, "Search your image and video library by phrase") {
		t.Fatalf("expected marketing lede removed from compact masthead")
	}
}

func TestAssetsAreServed(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/assets/app.js", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "fetch(`/api/images?${params.toString()}`)") {
		t.Fatalf("expected app javascript payload")
	}
	if !strings.Contains(rr.Body.String(), "fetch(`/api/videos?${params.toString()}`)") {
		t.Fatalf("expected videos javascript payload")
	}
	if !strings.Contains(rr.Body.String(), "openLightbox") {
		t.Fatalf("expected lightbox javascript behavior")
	}
	if !strings.Contains(rr.Body.String(), "openVideoPlayer") {
		t.Fatalf("expected Video.js player behavior in javascript")
	}
	if !strings.Contains(rr.Body.String(), "is_anchor") {
		t.Fatalf("expected similar-search anchor behavior in javascript")
	}
	if !strings.Contains(rr.Body.String(), "params.set('neg'") {
		t.Fatalf("expected negative prompt query parameter in javascript")
	}
	if !strings.Contains(rr.Body.String(), "match_timestamp_ms") || !strings.Contains(rr.Body.String(), "preview_path") {
		t.Fatalf("expected video-result rendering fields in javascript")
	}
	if !strings.Contains(rr.Body.String(), "const mediaURL = escapeHTML(toMediaURL(item.storage_path));") {
		t.Fatalf("expected mediaURL helper in card rendering to avoid video player runtime errors")
	}
	if !strings.Contains(rr.Body.String(), "data-player-src=\"${mediaURL}\"") {
		t.Fatalf("expected player source binding to use mediaURL in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "totalPages(state.imagesTotal, state.galleryPageSize)") {
		t.Fatalf("expected gallery pagination to use shared totalPages helper")
	}
	if strings.Contains(rr.Body.String(), "totalGalleryPages()") {
		t.Fatalf("expected removed totalGalleryPages helper not to be referenced in app javascript")
	}
	if !strings.Contains(rr.Body.String(), "deleteMedia(kind, id, name)") || !strings.Contains(rr.Body.String(), "fetch(`/api/${kind === 'video' ? 'videos' : 'images'}/${id}`") {
		t.Fatalf("expected delete media workflow in javascript")
	}
	if !strings.Contains(rr.Body.String(), "delete-action") {
		t.Fatalf("expected delete action class in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "const supportEntries = supportEntriesForItem(item, mode);") {
		t.Fatalf("expected card rendering to derive supporting text entries")
	}
	if !strings.Contains(rr.Body.String(), "supporting-text") {
		t.Fatalf("expected compact supporting text class in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "overlay-supporting-stack") {
		t.Fatalf("expected card rendering to support stacked description/transcript text")
	}
	if !strings.Contains(rr.Body.String(), "tag-chip-more") {
		t.Fatalf("expected compact tag overflow chip support in card rendering")
	}
	if !strings.Contains(rr.Body.String(), "tags.slice(0, 3)") {
		t.Fatalf("expected compact tag rows to reserve room for overflow chip on all cards")
	}
	if !strings.Contains(rr.Body.String(), "const compactTagsMarkup = tagsMarkup(tags, true, true);") || !strings.Contains(rr.Body.String(), "<div class=\"meta-tags\">${compactTagsMarkup}</div>") {
		t.Fatalf("expected card tags to render in a dedicated non-shrinking metadata slot")
	}
	if strings.Contains(rr.Body.String(), "description-toggle") || strings.Contains(rr.Body.String(), "Show more") {
		t.Fatalf("expected inline description toggle pattern removed from card rendering")
	}
	if !strings.Contains(rr.Body.String(), "card-detail-overlay") {
		t.Fatalf("expected card rendering to include overlay detail layer")
	}
	if !strings.Contains(rr.Body.String(), "thumb-actions") {
		t.Fatalf("expected card rendering to include thumbnail action overlay")
	}
	if strings.Contains(rr.Body.String(), "<div class=\"card-actions\">") {
		t.Fatalf("expected action controls moved out of card meta stack")
	}
	if !strings.Contains(rr.Body.String(), "const overlayStatusMarkup =") {
		t.Fatalf("expected overlay detail layer to include status row markup")
	}
	if !strings.Contains(rr.Body.String(), "${overlayTagsMarkup}") || !strings.Contains(rr.Body.String(), "${overlaySupportMarkup}") {
		t.Fatalf("expected overlay detail layer to include tags and expanded supporting text")
	}
	if !strings.Contains(rr.Body.String(), "function scoreBadgeText(item, scoreLabel)") || !strings.Contains(rr.Body.String(), "`${scoreLabel} at ${formatTimestamp(item.match_timestamp_ms)}`") {
		t.Fatalf("expected video result timestamp to be folded into thumbnail match badge text")
	}
	if !strings.Contains(rr.Body.String(), "const scoreBadge =") || !strings.Contains(rr.Body.String(), "thumb-match-badge") {
		t.Fatalf("expected score badge markup over thumbnail for match percentage prominence")
	}
	if strings.Contains(rr.Body.String(), "cardMediaClass") || strings.Contains(rr.Body.String(), "class=\"card ${cardMediaClass}\"") {
		t.Fatalf("expected card rendering to avoid separate image/video card layout classes")
	}
	if strings.Contains(rr.Body.String(), "${videoMeta}\n          ${compactTagsMarkup}") {
		t.Fatalf("expected rest-state cards to keep video metadata out of the shared compact metadata stack")
	}
	if strings.Contains(rr.Body.String(), "${anchorBadge}${score}</div>") {
		t.Fatalf("expected expanded overlay status row not to duplicate match score from thumbnail badge")
	}
	if !strings.Contains(rr.Body.String(), "status === 'done' || mediaType === 'video'") {
		t.Fatalf("expected similar action to stay available for videos regardless of transcript-only index state")
	}
	if !strings.Contains(rr.Body.String(), "const opsBar = document.getElementById('ops-bar');") {
		t.Fatalf("expected javascript hook for compact ops bar state")
	}
	if !strings.Contains(rr.Body.String(), "opsBar.dataset.state =") {
		t.Fatalf("expected javascript to toggle quiet/active ops bar presentation")
	}
	if !strings.Contains(rr.Body.String(), "const tagsTabButton = document.getElementById('tab-tags');") {
		t.Fatalf("expected javascript hook for tags tab button")
	}
	if !strings.Contains(rr.Body.String(), "applyNSFWQuery(new URLSearchParams({ limit: '80' }))") {
		t.Fatalf("expected javascript to apply nsfw filter params when loading tag cloud")
	}
	if !strings.Contains(rr.Body.String(), "setActiveTab('tags')") {
		t.Fatalf("expected tags tab activation handler in javascript")
	}
	if !strings.Contains(rr.Body.String(), "fetch(`/api/search/tags?${params.toString()}`)") {
		t.Fatalf("expected javascript tag search request wiring")
	}
	if !strings.Contains(rr.Body.String(), "tagCloud.addEventListener('click'") {
		t.Fatalf("expected javascript click handler for tag cloud chips")
	}
	if !strings.Contains(rr.Body.String(), "await runTagSearch([tag], 'any');") {
		t.Fatalf("expected tag cloud chip clicks to run single-tag search")
	}
	if !strings.Contains(rr.Body.String(), "item.search_source === 'tag' ? '' : formatMatch(item.distance)") {
		t.Fatalf("expected tag-search results to skip similarity score badge")
	}
	if !strings.Contains(rr.Body.String(), "params.append('tag', tag)") || !strings.Contains(rr.Body.String(), "params.set('tag_mode', tagMode)") {
		t.Fatalf("expected text search request to include optional tag restriction parameters")
	}
	if !strings.Contains(rr.Body.String(), "fetch(`/api/search/tag-cloud?${params.toString()}`)") {
		t.Fatalf("expected javascript autocomplete lookups for tag suggestions")
	}
	if !strings.Contains(rr.Body.String(), "function formatSearchDebugSuffix(debug)") {
		t.Fatalf("expected javascript helper for rendering search debug suffix text")
	}
	if !strings.Contains(rr.Body.String(), "Similar search returned ${state.results.length} result(s).${formatSearchDebugSuffix(payload.debug)}") {
		t.Fatalf("expected similar-search success status to include backend debug suffix")
	}
	if !strings.Contains(rr.Body.String(), "tag-chip-button") {
		t.Fatalf("expected javascript card rendering support for clickable tag buttons")
	}
	if !strings.Contains(rr.Body.String(), "resultsPrevButton") || !strings.Contains(rr.Body.String(), "resultsNextButton") {
		t.Fatalf("expected javascript handlers for results pagination controls")
	}
	if !strings.Contains(rr.Body.String(), "const showNSFWToggle = document.getElementById('show-nsfw');") {
		t.Fatalf("expected javascript hook for nsfw visibility toggle")
	}
	if !strings.Contains(rr.Body.String(), "function applyNSFWQuery(params)") || !strings.Contains(rr.Body.String(), "params.set('include_nsfw', '1');") {
		t.Fatalf("expected javascript to attach include_nsfw query params for backend filtering")
	}
	if !strings.Contains(rr.Body.String(), "restartLiveUpdates();") {
		t.Fatalf("expected nsfw toggle changes to restart live updates with updated filters")
	}
	if !strings.Contains(rr.Body.String(), "if (event.target.closest('.thumb-actions'))") {
		t.Fatalf("expected lightbox handler to ignore taps on thumbnail action controls")
	}
}

func TestMediaServingIsRestrictedToMediaSubdirectories(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	videosDir := filepath.Join(dataDir, "videos")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.MkdirAll(videosDir, 0o755); err != nil {
		t.Fatalf("mkdir videos dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "probe.txt"), []byte("img"), 0o644); err != nil {
		t.Fatalf("write image probe: %v", err)
	}
	if err := os.WriteFile(filepath.Join(videosDir, "clip.txt"), []byte("vid"), 0o644); err != nil {
		t.Fatalf("write video probe: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "imgsearch.sqlite"), []byte("db"), 0o644); err != nil {
		t.Fatalf("write db probe: %v", err)
	}

	h := NewHandler(dataDir)

	allowedImageReq := httptest.NewRequest(http.MethodGet, "/media/images/probe.txt", nil)
	allowedImageRR := httptest.NewRecorder()
	h.ServeHTTP(allowedImageRR, allowedImageReq)
	if allowedImageRR.Code != http.StatusOK || allowedImageRR.Body.String() != "img" {
		t.Fatalf("image media status: got=%d body=%q", allowedImageRR.Code, allowedImageRR.Body.String())
	}

	allowedVideoReq := httptest.NewRequest(http.MethodGet, "/media/videos/clip.txt", nil)
	allowedVideoRR := httptest.NewRecorder()
	h.ServeHTTP(allowedVideoRR, allowedVideoReq)
	if allowedVideoRR.Code != http.StatusOK || allowedVideoRR.Body.String() != "vid" {
		t.Fatalf("video media status: got=%d body=%q", allowedVideoRR.Code, allowedVideoRR.Body.String())
	}

	blockedReq := httptest.NewRequest(http.MethodGet, "/media/imgsearch.sqlite", nil)
	blockedRR := httptest.NewRecorder()
	h.ServeHTTP(blockedRR, blockedReq)
	if blockedRR.Code != http.StatusNotFound {
		t.Fatalf("non-media path should be blocked: got=%d want=%d body=%s", blockedRR.Code, http.StatusNotFound, blockedRR.Body.String())
	}

	unknownSubdirReq := httptest.NewRequest(http.MethodGet, "/media/audio/song.mp3", nil)
	unknownSubdirRR := httptest.NewRecorder()
	h.ServeHTTP(unknownSubdirRR, unknownSubdirReq)
	if unknownSubdirRR.Code != http.StatusNotFound {
		t.Fatalf("unknown media subdirectory should be blocked: got=%d want=%d", unknownSubdirRR.Code, http.StatusNotFound)
	}

	traversalReq := httptest.NewRequest(http.MethodGet, "/media/images/../imgsearch.sqlite", nil)
	traversalRR := httptest.NewRecorder()
	h.ServeHTTP(traversalRR, traversalReq)
	if traversalRR.Code == http.StatusOK {
		t.Fatalf("traversal request must not succeed: got=%d body=%q", traversalRR.Code, traversalRR.Body.String())
	}
}

func TestStylesIncludeTightRadiusAndCardDensityRules(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/assets/styles.css", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}

	body := rr.Body.String()
	if !strings.Contains(body, "--radius-panel:") || !strings.Contains(body, "--radius-control:") {
		t.Fatalf("expected two-tier radius tokens for panel and control scales")
	}
	if strings.Contains(body, "--radius-lg:") || strings.Contains(body, "--radius-md:") || strings.Contains(body, "--radius-sm:") {
		t.Fatalf("expected legacy multi-size radius tokens to be removed")
	}
	if !strings.Contains(body, "--card-media-height: 220px;") || !strings.Contains(body, "--card-meta-height: 172px;") {
		t.Fatalf("expected cards to declare fixed resting media/meta dimensions")
	}
	if !strings.Contains(body, "grid-template-rows: var(--card-media-height) minmax(var(--card-meta-height), var(--card-meta-height));") {
		t.Fatalf("expected cards to use fixed resting media/meta proportions")
	}
	if !strings.Contains(body, ".supporting-text") {
		t.Fatalf("expected dedicated supporting text clamping rules")
	}
	if !strings.Contains(body, ".supporting-stack") || !strings.Contains(body, ".overlay-supporting-stack") {
		t.Fatalf("expected stacked supporting-text layout rules for description plus transcript")
	}
	if !strings.Contains(body, ".meta-tags") || !strings.Contains(body, "min-height: 1.65rem;") {
		t.Fatalf("expected compact card tags to get a reserved row at rest")
	}
	if !strings.Contains(body, "grid-template-rows: auto auto minmax(0, 1fr);") {
		t.Fatalf("expected card metadata to keep title/path and tags out of the shrinkable prose area")
	}
	if !strings.Contains(body, "grid-template-columns: repeat(3, minmax(0, 1fr)) max-content;") {
		t.Fatalf("expected tag rows to reserve a fixed final column for +N overflow chips")
	}
	if !strings.Contains(body, ".tag-list .tag-chip") || !strings.Contains(body, "text-overflow: ellipsis;") {
		t.Fatalf("expected compact tag chips to ellipsize instead of being visually clipped")
	}
	if !strings.Contains(body, ".card-detail-overlay") {
		t.Fatalf("expected overlay expansion rules for clipped card content")
	}
	if !strings.Contains(body, ".card:focus-within .card-detail-overlay") {
		t.Fatalf("expected keyboard focus to reveal card overlay details")
	}
	if !strings.Contains(body, ".thumb-actions") {
		t.Fatalf("expected thumbnail action overlay styling rules")
	}
	if strings.Contains(body, "box-shadow: 0 12px 24px rgba(44, 37, 25, 0.14);\n  transform:") {
		t.Fatalf("expected card hover state to avoid vertical movement")
	}
	if !strings.Contains(body, "white-space: nowrap;") || !strings.Contains(body, "text-overflow: ellipsis;") {
		t.Fatalf("expected compact card title/path to truncate horizontally")
	}
	if !strings.Contains(body, "padding: 2px 8px;") {
		t.Fatalf("expected tag chips to match state-pill interior spacing")
	}
	if !strings.Contains(body, ".card-detail-overlay .overlay-supporting-text") {
		t.Fatalf("expected overlay supporting text to override rest-state clamps")
	}
	if !strings.Contains(body, ".thumb-match-badge") {
		t.Fatalf("expected dedicated thumbnail match badge styling rules")
	}
	if strings.Contains(body, ".card-video") || strings.Contains(body, "--card-meta-height: 224px;") {
		t.Fatalf("expected image and video cards to share the same compact layout rules")
	}
	if !strings.Contains(body, ".card-detail-overlay > *") || !strings.Contains(body, "min-width: 0;") {
		t.Fatalf("expected overlay rows to stay in normal flow without text collision")
	}
	if !strings.Contains(body, "right: 10px;") || !strings.Contains(body, "bottom: 10px;") {
		t.Fatalf("expected thumbnail match badge to stay anchored at lower right")
	}
	if !strings.Contains(body, ".ops-status-panel") || !strings.Contains(body, ".ops-menu") {
		t.Fatalf("expected overflow-only status panel and quiet ops-menu styling rules")
	}
	if !strings.Contains(body, ".search-advanced") {
		t.Fatalf("expected advanced-search disclosure styling rules")
	}
	if strings.Contains(body, "grid-template-columns: minmax(280px, 0.95fr) minmax(420px, 1.35fr);") {
		t.Fatalf("expected old heavyweight masthead two-column layout to be removed")
	}
	if strings.Contains(body, "font-size: clamp(2.2rem, 5vw, 4rem);") {
		t.Fatalf("expected oversized hero-style title scale to be removed from masthead")
	}
	if !strings.Contains(body, ".tag-cloud-list") || !strings.Contains(body, ".tag-cloud-chip") {
		t.Fatalf("expected tag cloud shell styling rules")
	}
	if !strings.Contains(body, ".tag-cloud-chip-size-1") || !strings.Contains(body, ".tag-cloud-chip-size-4") {
		t.Fatalf("expected tiered tag cloud chip size classes")
	}
	if !strings.Contains(body, ".tag-cloud-chip:hover:not(:disabled)") {
		t.Fatalf("expected interactive hover treatment for tag cloud chips")
	}
	if !strings.Contains(body, ".search-tag-suggestions") || !strings.Contains(body, ".search-tag-suggestion") {
		t.Fatalf("expected autocomplete suggestion panel styling rules")
	}
	if !strings.Contains(body, ".tag-chip-button") {
		t.Fatalf("expected clickable tag button styling rules")
	}
	if !strings.Contains(body, ".results-pagination") {
		t.Fatalf("expected results pagination styling rules")
	}
	if !strings.Contains(body, "overflow-x: hidden;") {
		t.Fatalf("expected mobile-safe horizontal overflow guard in styles")
	}
	if !strings.Contains(body, "grid-template-columns: repeat(4, minmax(0, 1fr));") {
		t.Fatalf("expected compact mobile tab grid to prevent horizontal scroll")
	}
	if !strings.Contains(body, "@media (hover: none) and (pointer: coarse)") {
		t.Fatalf("expected touch-specific card interaction overrides")
	}
	if !strings.Contains(body, ".nsfw-toggle") {
		t.Fatalf("expected nsfw toggle styling rules")
	}
	if !strings.Contains(body, ".card-detail-overlay {") || !strings.Contains(body, "display: none;") {
		t.Fatalf("expected touch mode to suppress hover-only detail overlay")
	}
	if !strings.Contains(body, ".thumb-actions {") || !strings.Contains(body, "opacity: 1;") {
		t.Fatalf("expected touch mode to keep thumbnail actions visible")
	}
}

func TestMediaServesFilesFromDataDir(t *testing.T) {
	dataDir := t.TempDir()
	imagesDir := filepath.Join(dataDir, "images")
	if err := os.MkdirAll(imagesDir, 0o755); err != nil {
		t.Fatalf("mkdir images dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(imagesDir, "sample.bin"), []byte("ok"), 0o644); err != nil {
		t.Fatalf("write image sample: %v", err)
	}

	h := NewHandler(dataDir)
	req := httptest.NewRequest(http.MethodGet, "/media/images/sample.bin", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if rr.Body.String() != "ok" {
		t.Fatalf("unexpected media body: %q", rr.Body.String())
	}
}

func TestUnknownPathReturnsNotFound(t *testing.T) {
	h := NewHandler(t.TempDir())

	req := httptest.NewRequest(http.MethodGet, "/nope", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusNotFound)
	}
}
