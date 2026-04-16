package upload

import (
	"errors"
	"net/http"
	"path/filepath"

	"imgsearch/internal/httputil"
)

const maxUploadBytes = 500 << 20

type UploadResponse struct {
	MediaType string `json:"media_type,omitempty"`
	ImageID   int64  `json:"image_id"`
	VideoID   int64  `json:"video_id,omitempty"`
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
			httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

		if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
			httputil.WriteJSONError(w, http.StatusBadRequest, "invalid multipart upload")
			return
		}

		files := r.MultipartForm.File["file"]
		if len(files) == 0 {
			httputil.WriteJSONError(w, http.StatusBadRequest, "missing file")
			return
		}

		uploads := make([]UploadResponse, 0, len(files))
		created := 0
		duplicates := 0
		for _, header := range files {
			file, err := header.Open()
			if err != nil {
				httputil.WriteJSONError(w, http.StatusBadRequest, "invalid file upload")
				return
			}

			out, err := svc.Store(r.Context(), filepath.Base(header.Filename), file)
			_ = file.Close()
			if err != nil {
				if errors.Is(err, ErrUnsupportedFormat) {
					httputil.WriteJSONError(w, http.StatusBadRequest, "unsupported media format")
					return
				}
				httputil.WriteJSONError(w, http.StatusInternalServerError, "upload failed")
				return
			}

			if out.Duplicate {
				duplicates++
			} else {
				created++
			}
			uploads = append(uploads, UploadResponse{
				MediaType: out.MediaType,
				ImageID:   out.ImageID,
				VideoID:   out.VideoID,
				SHA256:    out.SHA256,
				Duplicate: out.Duplicate,
			})
		}

		status := http.StatusCreated
		if created == 0 {
			status = http.StatusOK
		}

		httputil.WriteJSON(w, status, UploadBatchResponse{
			Uploads:    uploads,
			Created:    created,
			Duplicates: duplicates,
		})
	})
}
