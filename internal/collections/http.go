package collections

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"imgsearch/internal/httputil"
)

type Handler struct {
	DB      *sql.DB
	ModelID int64
}

type apiError struct {
	status  int
	message string
}

func (e *apiError) Error() string              { return e.message }
func problem(status int, message string) error { return &apiError{status, message} }

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h == nil || h.DB == nil {
			httputil.WriteJSONError(w, 503, "collections database unavailable")
			return
		}
		if err := h.serve(w, r); err != nil {
			var api *apiError
			if errors.As(err, &api) {
				httputil.WriteJSONError(w, api.status, api.message)
			} else {
				httputil.WriteJSONError(w, 500, "collection operation failed")
			}
		}
	})
}

func positiveID(value string) (int64, error) {
	id, err := strconv.ParseInt(value, 10, 64)
	if err != nil || id <= 0 {
		return 0, problem(400, "id must be a positive integer")
	}
	return id, nil
}

func parseTarget(kind, value string) (target, error) {
	if kind != "image" && kind != "video" {
		return target{}, problem(400, "media_type must be image or video")
	}
	id, err := positiveID(value)
	return target{kind, id}, err
}

func decodeName(w http.ResponseWriter, r *http.Request) (string, error) {
	var body struct {
		Name *string `json:"name"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil || body.Name == nil {
		return "", problem(400, "expected a JSON object with a name")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return "", problem(400, "expected one JSON object")
	}
	name := strings.TrimSpace(*body.Name)
	if utf8.RuneCountInString(name) < 1 || utf8.RuneCountInString(name) > 100 || strings.IndexFunc(name, unicode.IsControl) >= 0 {
		return "", problem(400, "name must contain 1–100 characters without control characters")
	}
	return name, nil
}

func (h *Handler) serve(w http.ResponseWriter, r *http.Request) error {
	if r.URL.Path == "/api/collections" {
		return h.collectionList(w, r)
	}
	if !strings.HasPrefix(r.URL.Path, "/api/collections/") {
		return problem(404, "not found")
	}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/collections/"), "/")
	if parts[0] == "" {
		return problem(404, "not found")
	}
	if len(parts) != 1 && !(len(parts) == 2 && parts[1] == "items") && !(len(parts) == 4 && parts[1] == "items") {
		return problem(404, "not found")
	}
	id, err := positiveID(parts[0])
	if err != nil {
		return err
	}
	switch len(parts) {
	case 1:
		return h.collection(w, r, id)
	case 2:
		if r.Method != http.MethodGet {
			httputil.WriteMethodNotAllowed(w, http.MethodGet)
			return nil
		}
		page, err := h.items(r.Context(), id, httputil.ParseLimitQuery(r, 50), httputil.ParseOffsetQuery(r, 0), httputil.ParseIncludeNSFWQuery(r))
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, page)
	case 4:
		if r.Method != http.MethodPut && r.Method != http.MethodDelete {
			httputil.WriteMethodNotAllowed(w, http.MethodPut, http.MethodDelete)
			return nil
		}
		t, err := parseTarget(parts[2], parts[3])
		if err != nil {
			return err
		}
		if err := h.membership(r.Context(), id, t, r.Method == http.MethodPut); err != nil {
			return err
		}
		w.WriteHeader(http.StatusNoContent)
	}
	return nil
}

func (h *Handler) collectionList(w http.ResponseWriter, r *http.Request) error {
	switch r.Method {
	case http.MethodGet:
		var t *target
		q := r.URL.Query()
		if q.Has("media_type") || q.Has("media_id") {
			value, err := parseTarget(q.Get("media_type"), q.Get("media_id"))
			if err != nil {
				return err
			}
			t = &value
		}
		if t == nil && q.Get("summary") == "true" {
			list, err := h.summaries(r.Context(), httputil.ParseIncludeNSFWQuery(r))
			if err != nil {
				return err
			}
			httputil.WriteJSON(w, 200, struct {
				Collections []CollectionSummary `json:"collections"`
			}{list})
			return nil
		}
		list, err := h.list(r.Context(), t)
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, struct {
			Collections []Collection `json:"collections"`
		}{list})
	case http.MethodPost:
		name, err := decodeName(w, r)
		if err != nil {
			return err
		}
		c, err := h.saveName(r.Context(), 0, name)
		if err != nil {
			return err
		}
		w.Header().Set("Location", fmt.Sprintf("/api/collections/%d", c.ID))
		httputil.WriteJSON(w, 201, c)
	default:
		httputil.WriteMethodNotAllowed(w, http.MethodGet, http.MethodPost)
	}
	return nil
}

func (h *Handler) collection(w http.ResponseWriter, r *http.Request, id int64) error {
	switch r.Method {
	case http.MethodGet:
		c, err := getCollection(r.Context(), h.DB, id)
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, c)
	case http.MethodPatch:
		name, err := decodeName(w, r)
		if err != nil {
			return err
		}
		c, err := h.saveName(r.Context(), id, name)
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, c)
	case http.MethodDelete:
		if err := h.delete(r.Context(), id); err != nil {
			return err
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		httputil.WriteMethodNotAllowed(w, http.MethodGet, http.MethodPatch, http.MethodDelete)
	}
	return nil
}
