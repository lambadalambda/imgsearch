package upload

import (
	"encoding/json"
	"errors"
	"net/http"
	"path/filepath"
)

const maxUploadBytes = 20 << 20

type UploadResponse struct {
	ImageID   int64  `json:"image_id"`
	SHA256    string `json:"sha256"`
	Duplicate bool   `json:"duplicate"`
}

func NewHandler(svc *Service) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

		file, header, err := r.FormFile("file")
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "missing file")
			return
		}
		defer func() { _ = file.Close() }()

		out, err := svc.Store(r.Context(), filepath.Base(header.Filename), file)
		if err != nil {
			if errors.Is(err, ErrUnsupportedFormat) {
				writeJSONError(w, http.StatusBadRequest, "unsupported image format")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "upload failed")
			return
		}

		status := http.StatusCreated
		if out.Duplicate {
			status = http.StatusOK
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(UploadResponse{
			ImageID:   out.ImageID,
			SHA256:    out.SHA256,
			Duplicate: out.Duplicate,
		})
	})
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
