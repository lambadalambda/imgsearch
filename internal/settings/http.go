package settings

import (
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"

	"imgsearch/internal/httputil"
)

const maxSettingsBodyBytes = 64 << 10

// Handler configures the /api/settings routes.
type Handler struct {
	DB *sql.DB
	// Defaults is served until the first save; callers seed it from flags.
	Defaults AnnotationSettings
	// TestConnection probes a remote backend without saving. Nil disables
	// the test endpoint. Its error text is returned to the client verbatim,
	// so implementations must not echo the API key.
	TestConnection func(ctx context.Context, s AnnotationSettings) error
}

type handler struct {
	cfg *Handler
}

type response struct {
	Version    int64          `json:"version"`
	Annotation AnnotationView `json:"annotation"`
}

type updateRequest struct {
	Annotation  AnnotationSettings `json:"annotation"`
	ClearAPIKey bool               `json:"clear_api_key"`
}

type testResponse struct {
	OK    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

// NewHandler serves GET/PUT /api/settings and POST /api/settings/annotation/test.
func NewHandler(cfg *Handler) http.Handler {
	h := &handler{cfg: cfg}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/settings", h.handleSettings)
	mux.HandleFunc("/api/settings/annotation/test", h.handleTest)
	mux.HandleFunc("/api/settings/", func(w http.ResponseWriter, _ *http.Request) {
		httputil.WriteJSONError(w, http.StatusNotFound, "not found")
	})
	return mux
}

func (h *handler) handleSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPut {
		httputil.WriteMethodNotAllowed(w, http.MethodGet, http.MethodPut)
		return
	}
	if h.cfg == nil || h.cfg.DB == nil {
		httputil.WriteJSONError(w, http.StatusServiceUnavailable, "settings backend unavailable")
		return
	}
	if r.Method == http.MethodGet {
		h.writeCurrent(w, r.Context())
		return
	}
	h.update(w, r)
}

func (h *handler) writeCurrent(w http.ResponseWriter, ctx context.Context) {
	current, version, err := h.load(ctx)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "load settings failed")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, response{Version: version, Annotation: current.View()})
}

func (h *handler) load(ctx context.Context) (AnnotationSettings, int64, error) {
	current, err := LoadAnnotationOrDefault(ctx, h.cfg.DB, h.cfg.Defaults)
	if err != nil {
		return AnnotationSettings{}, 0, err
	}
	version, err := Version(ctx, h.cfg.DB)
	if err != nil {
		return AnnotationSettings{}, 0, err
	}
	return current, version, nil
}

func (h *handler) update(w http.ResponseWriter, r *http.Request) {
	req, ok := decodeUpdate(w, r)
	if !ok {
		return
	}
	current, _, err := h.load(r.Context())
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "load settings failed")
		return
	}
	next := mergeAPIKey(req, current)
	if err := next.Validate(); err != nil {
		httputil.WriteJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	version, err := SaveAnnotation(r.Context(), h.cfg.DB, next)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "save settings failed")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, response{Version: version, Annotation: next.View()})
}

func (h *handler) handleTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteMethodNotAllowed(w, http.MethodPost)
		return
	}
	if h.cfg == nil || h.cfg.DB == nil {
		httputil.WriteJSONError(w, http.StatusServiceUnavailable, "settings backend unavailable")
		return
	}
	if h.cfg.TestConnection == nil {
		httputil.WriteJSONError(w, http.StatusServiceUnavailable, "connection test unavailable")
		return
	}
	req, ok := decodeUpdate(w, r)
	if !ok {
		return
	}
	current, _, err := h.load(r.Context())
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "load settings failed")
		return
	}
	candidate := mergeAPIKey(req, current)
	if candidate.Backend != BackendOpenAI {
		httputil.WriteJSONError(w, http.StatusBadRequest, "only the openai backend supports a connection test")
		return
	}
	if err := candidate.Validate(); err != nil {
		httputil.WriteJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := h.cfg.TestConnection(r.Context(), candidate); err != nil {
		httputil.WriteJSON(w, http.StatusBadGateway, testResponse{OK: false, Error: err.Error()})
		return
	}
	httputil.WriteJSON(w, http.StatusOK, testResponse{OK: true})
}

func decodeUpdate(w http.ResponseWriter, r *http.Request) (updateRequest, bool) {
	var req updateRequest
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxSettingsBodyBytes))
	if err != nil {
		httputil.WriteJSONError(w, http.StatusRequestEntityTooLarge, "settings body too large")
		return req, false
	}
	if err := json.Unmarshal(body, &req); err != nil {
		httputil.WriteJSONError(w, http.StatusBadRequest, "invalid JSON body")
		return req, false
	}
	return req, true
}

// mergeAPIKey applies the "blank keeps the stored key" rule so the UI never
// has to round-trip the secret. A request that omits the openai block
// entirely (for example when switching to native) keeps the stored block so
// switching back later does not require re-entering the server details.
func mergeAPIKey(req updateRequest, current AnnotationSettings) AnnotationSettings {
	if req.Annotation.OpenAI == (OpenAISettings{}) {
		req.Annotation.OpenAI = current.OpenAI
	}
	next := req.Annotation.Normalized()
	switch {
	case req.ClearAPIKey:
		next.OpenAI.APIKey = ""
	case next.OpenAI.APIKey == "":
		next.OpenAI.APIKey = current.OpenAI.APIKey
	}
	return next
}
