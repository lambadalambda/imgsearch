package upload

import (
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"time"

	"imgsearch/internal/httputil"
)

const (
	maxUploadMemoryBytes     = 8 << 20
	maxUploadFilesPerRequest = 32
)

// UploadLimitError is the 413 body: it names the file and the limit that
// applied so clients can explain the rejection.
type UploadLimitError struct {
	Error      string `json:"error"`
	Filename   string `json:"filename,omitempty"`
	MediaType  string `json:"media_type,omitempty"`
	LimitBytes int64  `json:"limit_bytes"`
}

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

		// Large uploads on slow links outlive the server-wide read/write
		// timeouts, so give this request its own deadline.
		deadline := time.Now().Add(svc.requestTimeout())
		rc := http.NewResponseController(w)
		for _, err := range []error{rc.SetReadDeadline(deadline), rc.SetWriteDeadline(deadline)} {
			if err != nil && !errors.Is(err, http.ErrNotSupported) {
				log.Printf("upload: extend request deadline: %v", err)
			}
		}

		// The request cap only guards against unbounded bodies; per-file
		// limits below are what users actually hit. The multipart body is
		// spooled to the OS temp dir in full before those checks run, so
		// reject a declared oversize up front without reading it.
		maxRequestBytes := maxUploadFilesPerRequest * svc.maxVideoBytes()
		writeRequestTooLarge := func() {
			httputil.WriteJSON(w, http.StatusRequestEntityTooLarge, UploadLimitError{
				Error:      fmt.Sprintf("upload too large: request exceeds the %d MiB limit", maxRequestBytes>>20),
				LimitBytes: maxRequestBytes,
			})
		}
		if r.ContentLength > maxRequestBytes {
			writeRequestTooLarge()
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)

		if err := r.ParseMultipartForm(maxUploadMemoryBytes); err != nil {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				writeRequestTooLarge()
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
		// Check every file against its media-type limit before storing any,
		// so an oversized file rejects the whole batch cleanly.
		for _, header := range files {
			mediaType, limit := svc.uploadLimitFor(header)
			if header.Size > limit {
				filename := filepath.Base(header.Filename)
				httputil.WriteJSON(w, http.StatusRequestEntityTooLarge, UploadLimitError{
					Error:      fmt.Sprintf("file too large: %s exceeds the %d MiB %s limit", filename, limit>>20, mediaType),
					Filename:   filename,
					MediaType:  mediaType,
					LimitBytes: limit,
				})
				return
			}
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
				log.Printf("upload failed filename=%q: %v", filename, err)
				if errors.Is(err, ErrVideoProcessingFailed) {
					uploads = append(uploads, UploadResponse{Filename: filename, Error: "video processing failed"})
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

// uploadLimitFor sniffs the file's media type and returns it with the
// per-file byte limit that applies. Unknown types get the image limit; the
// store rejects them later anyway.
func (s *Service) uploadLimitFor(header *multipart.FileHeader) (string, int64) {
	file, err := header.Open()
	if err != nil {
		return "image", s.maxImageBytes()
	}
	defer func() { _ = file.Close() }()

	head := make([]byte, 512)
	n, _ := io.ReadFull(file, head)
	if isSupportedVideoMime(sniffMime(head[:n])) {
		return "video", s.maxVideoBytes()
	}
	return "image", s.maxImageBytes()
}
