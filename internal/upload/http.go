package upload

import (
	"errors"
	"net/http"
	"path/filepath"

	"imgsearch/internal/httputil"
)

const (
	maxUploadBytes           = 64 << 20
	maxUploadMemoryBytes     = 8 << 20
	maxUploadFilesPerRequest = 32
)

type UploadResponse struct {
	Filename  string `json:"filename,omitempty"`
	Error     string `json:"error,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	ImageID   int64  `json:"image_id,omitempty"`
	VideoID   int64  `json:"video_id,omitempty"`
	SHA256    string `json:"sha256,omitempty"`
	Duplicate bool   `json:"duplicate,omitempty"`
}

type UploadBatchResponse struct {
	Uploads    []UploadResponse `json:"uploads"`
	Created    int              `json:"created"`
	Duplicates int              `json:"duplicates"`
	Failed     int              `json:"failed"`
}

func NewHandler(svc *Service) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httputil.WriteMethodNotAllowed(w, http.MethodPost)
			return
		}
		if svc == nil {
			httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

		if err := r.ParseMultipartForm(maxUploadMemoryBytes); err != nil {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				httputil.WriteJSONError(w, http.StatusRequestEntityTooLarge, "upload too large")
				return
			}
			httputil.WriteJSONError(w, http.StatusBadRequest, "invalid multipart upload")
			return
		}
		defer func() {
			if r.MultipartForm != nil {
				_ = r.MultipartForm.RemoveAll()
			}
		}()

		files := r.MultipartForm.File["file"]
		if len(files) == 0 {
			httputil.WriteJSONError(w, http.StatusBadRequest, "missing file")
			return
		}
		if len(files) > maxUploadFilesPerRequest {
			httputil.WriteJSONError(w, http.StatusBadRequest, "too many files in single request")
			return
		}

		uploads := make([]UploadResponse, 0, len(files))
		created := 0
		duplicates := 0
		failed := 0
		for _, header := range files {
			filename := filepath.Base(header.Filename)
			file, err := header.Open()
			if err != nil {
				failed++
				uploads = append(uploads, UploadResponse{Filename: filename, Error: "invalid file upload"})
				continue
			}

			out, err := svc.Store(r.Context(), filename, file)
			_ = file.Close()
			if err != nil {
				failed++
				if errors.Is(err, ErrUnsupportedFormat) {
					uploads = append(uploads, UploadResponse{Filename: filename, Error: "unsupported media format"})
					continue
				}
				uploads = append(uploads, UploadResponse{Filename: filename, Error: "upload failed"})
				continue
			}

			if out.Duplicate {
				duplicates++
			} else {
				created++
			}
			uploads = append(uploads, UploadResponse{
				Filename:  filename,
				MediaType: out.MediaType,
				ImageID:   out.ImageID,
				VideoID:   out.VideoID,
				SHA256:    out.SHA256,
				Duplicate: out.Duplicate,
			})
		}

		status := http.StatusCreated
		if failed > 0 && created+duplicates > 0 {
			status = http.StatusMultiStatus
		} else if failed > 0 {
			status = http.StatusBadRequest
		} else if created == 0 {
			status = http.StatusOK
		}

		httputil.WriteJSON(w, status, UploadBatchResponse{
			Uploads:    uploads,
			Created:    created,
			Duplicates: duplicates,
			Failed:     failed,
		})
	})
}
