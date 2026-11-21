package people

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
			httputil.WriteJSONError(w, 503, "people database unavailable")
			return
		}
		if err := h.serve(w, r); err != nil {
			var api *apiError
			if errors.As(err, &api) {
				httputil.WriteJSONError(w, api.status, api.message)
			} else {
				httputil.WriteJSONError(w, 500, "person operation failed")
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

func validName(name string) error {
	if utf8.RuneCountInString(name) < 1 || utf8.RuneCountInString(name) > 100 || strings.IndexFunc(name, unicode.IsControl) >= 0 {
		return problem(400, "name must contain 1–100 characters without control characters")
	}
	return nil
}

// decodeName allows an omitted name on create (neutral auto-naming). An
// explicit null, empty, or blank name is always rejected.
func decodeName(w http.ResponseWriter, r *http.Request, allowEmpty bool) (string, error) {
	var body map[string]json.RawMessage
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
	if err := decoder.Decode(&body); err != nil || body == nil {
		return "", problem(400, "expected a JSON object with a name")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return "", problem(400, "expected one JSON object")
	}
	raw, ok := body["name"]
	if !ok {
		if allowEmpty {
			return "", nil
		}
		return "", problem(400, "expected a JSON object with a name")
	}
	for key := range body {
		if key != "name" {
			return "", problem(400, "expected a JSON object with a name")
		}
	}
	var name string
	if err := json.Unmarshal(raw, &name); err != nil {
		return "", problem(400, "expected a JSON object with a name")
	}
	name = strings.TrimSpace(name)
	return name, validName(name)
}

func (h *Handler) serve(w http.ResponseWriter, r *http.Request) error {
	if strings.HasPrefix(r.URL.Path, "/api/faces") {
		return h.serveFaces(w, r)
	}
	if r.URL.Path == "/api/people" {
		return h.peopleList(w, r)
	}
	if r.URL.Path == "/api/people/merge" {
		if r.Method != http.MethodPost {
			httputil.WriteMethodNotAllowed(w, http.MethodPost)
			return nil
		}
		return h.mergePeople(w, r)
	}
	if !strings.HasPrefix(r.URL.Path, "/api/people/") {
		return problem(404, "not found")
	}
	rest := strings.TrimPrefix(r.URL.Path, "/api/people/")
	parts := strings.Split(rest, "/")
	if parts[0] == "" {
		return problem(404, "not found")
	}
	id, err := positiveID(parts[0])
	if err != nil {
		return err
	}
	switch {
	case len(parts) == 1:
		return h.person(w, r, id)
	case len(parts) == 2 && parts[1] == "items":
		if r.Method != http.MethodGet {
			httputil.WriteMethodNotAllowed(w, http.MethodGet)
			return nil
		}
		page, err := h.items(r.Context(), id, httputil.ParseLimitQuery(r, 50), httputil.ParseOffsetQuery(r, 0), httputil.ParseIncludeNSFWQuery(r))
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, page)
	case len(parts) == 2 && parts[1] == "split":
		if r.Method != http.MethodPost {
			httputil.WriteMethodNotAllowed(w, http.MethodPost)
			return nil
		}
		return h.splitPerson(w, r, id)
	default:
		return problem(404, "not found")
	}
	return nil
}

func (h *Handler) peopleList(w http.ResponseWriter, r *http.Request) error {
	switch r.Method {
	case http.MethodGet:
		if r.URL.Query().Get("summary") == "true" {
			list, err := h.summaries(r.Context())
			if err != nil {
				return err
			}
			httputil.WriteJSON(w, 200, struct {
				People []PersonSummary `json:"people"`
			}{list})
			return nil
		}
		list, err := h.list(r.Context())
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, struct {
			People []Person `json:"people"`
		}{list})
	case http.MethodPost:
		name, err := decodeName(w, r, true)
		if err != nil {
			return err
		}
		p, err := h.saveName(r.Context(), 0, name)
		if err != nil {
			return err
		}
		w.Header().Set("Location", fmt.Sprintf("/api/people/%d", p.ID))
		httputil.WriteJSON(w, 201, p)
	default:
		httputil.WriteMethodNotAllowed(w, http.MethodGet, http.MethodPost)
	}
	return nil
}

func (h *Handler) person(w http.ResponseWriter, r *http.Request, id int64) error {
	switch r.Method {
	case http.MethodGet:
		p, err := getPerson(r.Context(), h.DB, id)
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, p)
	case http.MethodPatch:
		name, err := decodeName(w, r, false)
		if err != nil {
			return err
		}
		p, err := h.saveName(r.Context(), id, name)
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, p)
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

func (h *Handler) mergePeople(w http.ResponseWriter, r *http.Request) error {
	var body struct {
		FromIDs []int64 `json:"from_ids"`
		IntoID  *int64  `json:"into_id"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil || body.IntoID == nil {
		return problem(400, "expected {from_ids:[...], into_id:...}")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return problem(400, "expected one JSON object")
	}
	if err := h.merge(r.Context(), body.FromIDs, *body.IntoID); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (h *Handler) splitPerson(w http.ResponseWriter, r *http.Request, id int64) error {
	var body struct {
		FaceIDs []int64 `json:"face_ids"`
		Name    *string `json:"name"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil || body.FaceIDs == nil {
		return problem(400, "expected {face_ids:[...]}")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return problem(400, "expected one JSON object")
	}
	name := ""
	if body.Name != nil {
		name = strings.TrimSpace(*body.Name)
		if err := validName(name); err != nil {
			return err
		}
	}
	p, deletedSourceID, err := h.split(r.Context(), id, body.FaceIDs, name)
	if err != nil {
		return err
	}
	w.Header().Set("Location", fmt.Sprintf("/api/people/%d", p.ID))
	httputil.WriteJSON(w, 201, struct {
		Person          Person `json:"person"`
		DeletedSourceID *int64 `json:"deleted_source_id"`
	}{p, deletedSourceID})
	return nil
}

func (h *Handler) serveFaces(w http.ResponseWriter, r *http.Request) error {
	if r.URL.Path == "/api/faces" {
		if r.Method != http.MethodGet {
			httputil.WriteMethodNotAllowed(w, http.MethodGet)
			return nil
		}
		q := r.URL.Query()
		var personID *int64
		if q.Has("person_id") {
			id, err := positiveID(q.Get("person_id"))
			if err != nil {
				return err
			}
			personID = &id
		}
		faces, err := h.listFaces(r.Context(), personID, q.Get("unassigned") == "true", httputil.ParseLimitQuery(r, 100), httputil.ParseOffsetQuery(r, 0))
		if err != nil {
			return err
		}
		httputil.WriteJSON(w, 200, struct {
			Faces []FaceView `json:"faces"`
		}{faces})
		return nil
	}
	rest := strings.TrimPrefix(r.URL.Path, "/api/faces/")
	parts := strings.Split(rest, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] != "reassign" {
		return problem(404, "not found")
	}
	faceID, err := positiveID(parts[0])
	if err != nil {
		return err
	}
	if r.Method != http.MethodPost {
		httputil.WriteMethodNotAllowed(w, http.MethodPost)
		return nil
	}
	var body struct {
		PersonID *int64 `json:"person_id"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil {
		return problem(400, "expected {person_id:...}")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return problem(400, "expected one JSON object")
	}
	if body.PersonID != nil && *body.PersonID <= 0 {
		return problem(400, "person_id must be a positive integer or null")
	}
	f, deletedPersonID, err := h.reassignFace(r.Context(), faceID, body.PersonID)
	if err != nil {
		return err
	}
	httputil.WriteJSON(w, 200, struct {
		Face            Face   `json:"face"`
		DeletedPersonID *int64 `json:"deleted_person_id"`
	}{f, deletedPersonID})
	return nil
}
