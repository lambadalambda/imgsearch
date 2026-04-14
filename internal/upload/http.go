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

type UploadBatchResponse struct {
	Uploads    []UploadResponse `json:"uploads"`
	Created    int              `json:"created"`
	Duplicates int              `json:"duplicates"`
}

func NewHandler(svc *Service) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

		if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid multipart upload")
			return
		}

		files := r.MultipartForm.File["file"]
		if len(files) == 0 {
			writeJSONError(w, http.StatusBadRequest, "missing file")
			return
		}

		uploads := make([]UploadResponse, 0, len(files))
		created := 0
		duplicates := 0
		for _, header := range files {
			file, err := header.Open()
			if err != nil {
				writeJSONError(w, http.StatusBadRequest, "invalid file upload")
				return
			}

			out, err := svc.Store(r.Context(), filepath.Base(header.Filename), file)
			_ = file.Close()
			if err != nil {
				if errors.Is(err, ErrUnsupportedFormat) {
					writeJSONError(w, http.StatusBadRequest, "unsupported image format")
					return
				}
				writeJSONError(w, http.StatusInternalServerError, "upload failed")
				return
			}

			if out.Duplicate {
				duplicates++
			} else {
				created++
			}
			uploads = append(uploads, UploadResponse{
				ImageID:   out.ImageID,
				SHA256:    out.SHA256,
				Duplicate: out.Duplicate,
			})
		}

		status := http.StatusCreated
		if created == 0 {
			status = http.StatusOK
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(UploadBatchResponse{
			Uploads:    uploads,
			Created:    created,
			Duplicates: duplicates,
		})
	})
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
