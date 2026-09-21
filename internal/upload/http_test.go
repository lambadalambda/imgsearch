package upload

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func multipartBody(t *testing.T, filename string, content []byte) (*bytes.Buffer, string) {
	t.Helper()
	return multipartBodyWithFiles(t, []struct {
		filename string
		content  []byte
	}{{filename: filename, content: content}})
}

func multipartBodyWithFiles(t *testing.T, files []struct {
	filename string
	content  []byte
}) (*bytes.Buffer, string) {
	t.Helper()

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	for _, file := range files {
		fw, err := mw.CreateFormFile("file", file.filename)
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		if _, err := fw.Write(file.content); err != nil {
			t.Fatalf("write form file: %v", err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	return body, mw.FormDataContentType()
}

func TestUploadHandlerReturnsCreatedForNewImage(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	body, contentType := multipartBody(t, "new.png", pngBytes(t))
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusCreated, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Uploads) != 1 {
		t.Fatalf("unexpected upload count: %+v", resp)
	}
	if resp.Uploads[0].ImageID == 0 || resp.Uploads[0].SHA256 == "" || resp.Uploads[0].Duplicate {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.Created != 1 || resp.Duplicates != 0 {
		t.Fatalf("unexpected counters: %+v", resp)
	}
}

func TestUploadHandlerReturnsCreatedForNewVideo(t *testing.T) {
	svc, _ := setupService(t)
	svc.VideoSampler = &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 2}
	svc.VideoFrameCount = 2
	h := NewHandler(svc)

	body, contentType := multipartBody(t, "clip.mp4", mp4Bytes())
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusCreated, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Uploads) != 1 {
		t.Fatalf("unexpected upload count: %+v", resp)
	}
	if resp.Uploads[0].MediaType != "video" || resp.Uploads[0].VideoID == 0 || resp.Uploads[0].Duplicate {
		t.Fatalf("unexpected video response: %+v", resp)
	}
}

func TestUploadHandlerReportsVideoProcessingFailure(t *testing.T) {
	svc, _ := setupService(t)
	svc.VideoSampler = &fakeVideoSampler{err: fmt.Errorf("sampler failed")}
	h := NewHandler(svc)

	body, contentType := multipartBody(t, "clip.mp4", mp4Bytes())
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Failed != 1 || len(resp.Uploads) != 1 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.Uploads[0].Error != "video processing failed" {
		t.Fatalf("expected video processing error, got %+v", resp.Uploads[0])
	}
}

func TestUploadHandlerAcceptsWEBPAndAVIF(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		fixture  string
	}{
		{name: "webp", filename: "cat_2.webp", fixture: "cat_2.webp"},
		{name: "avif", filename: "dog_2.avif", fixture: "dog_2.avif"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc, _ := setupService(t)
			h := NewHandler(svc)

			body, contentType := multipartBody(t, tc.filename, fixtureImageBytes(t, tc.fixture))
			req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
			req.Header.Set("Content-Type", contentType)
			rr := httptest.NewRecorder()

			h.ServeHTTP(rr, req)

			if rr.Code != http.StatusCreated {
				t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusCreated, rr.Body.String())
			}
		})
	}
}

func TestUploadHandlerReturnsOKForDuplicateImage(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)
	content := pngBytes(t)

	firstBody, firstType := multipartBody(t, "first.png", content)
	firstReq := httptest.NewRequest(http.MethodPost, "/api/upload", firstBody)
	firstReq.Header.Set("Content-Type", firstType)
	firstRR := httptest.NewRecorder()
	h.ServeHTTP(firstRR, firstReq)
	if firstRR.Code != http.StatusCreated {
		t.Fatalf("first status: got=%d body=%s", firstRR.Code, firstRR.Body.String())
	}

	secondBody, secondType := multipartBody(t, "second.png", content)
	secondReq := httptest.NewRequest(http.MethodPost, "/api/upload", secondBody)
	secondReq.Header.Set("Content-Type", secondType)
	secondRR := httptest.NewRecorder()
	h.ServeHTTP(secondRR, secondReq)

	if secondRR.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", secondRR.Code, http.StatusOK, secondRR.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(secondRR.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Uploads) != 1 || !resp.Uploads[0].Duplicate {
		t.Fatalf("expected duplicate response, got %+v", resp)
	}
}

func TestUploadHandlerAcceptsMultipleFiles(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	body, contentType := multipartBodyWithFiles(t, []struct {
		filename string
		content  []byte
	}{
		{filename: "one.png", content: pngBytes(t)},
		{filename: "two.webp", content: fixtureImageBytes(t, "cat_2.webp")},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusCreated, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Uploads) != 2 || resp.Created != 2 || resp.Duplicates != 0 {
		t.Fatalf("unexpected batch response: %+v", resp)
	}
}

func TestUploadHandlerReturnsMultiStatusForMixedBatch(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	body, contentType := multipartBodyWithFiles(t, []struct {
		filename string
		content  []byte
	}{
		{filename: "one.png", content: pngBytes(t)},
		{filename: "notes.txt", content: []byte("hello")},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMultiStatus {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusMultiStatus, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Created != 1 || resp.Duplicates != 0 || resp.Failed != 1 {
		t.Fatalf("unexpected counters: %+v", resp)
	}
	if len(resp.Uploads) != 2 {
		t.Fatalf("unexpected upload count: %+v", resp)
	}
	if resp.Uploads[0].ImageID == 0 || resp.Uploads[0].Error != "" {
		t.Fatalf("expected first file success, got %+v", resp.Uploads[0])
	}
	if resp.Uploads[1].Filename != "notes.txt" || resp.Uploads[1].Error != "unsupported media format" {
		t.Fatalf("expected second file error, got %+v", resp.Uploads[1])
	}
}

func TestUploadHandlerRejectsNonImage(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	body, contentType := multipartBody(t, "notes.txt", []byte("hello"))
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Fatalf("expected json content type, got %q", ct)
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Created != 0 || resp.Duplicates != 0 || resp.Failed != 1 {
		t.Fatalf("unexpected counters: %+v", resp)
	}
	if len(resp.Uploads) != 1 || resp.Uploads[0].Filename != "notes.txt" || resp.Uploads[0].Error != "unsupported media format" {
		t.Fatalf("unexpected upload errors: %+v", resp.Uploads)
	}
}

func TestUploadHandlerReturnsBadRequestForAllFailedBatch(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	body, contentType := multipartBodyWithFiles(t, []struct {
		filename string
		content  []byte
	}{
		{filename: "one.txt", content: []byte("one")},
		{filename: "two.txt", content: []byte("two")},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Created != 0 || resp.Duplicates != 0 || resp.Failed != 2 || len(resp.Uploads) != 2 {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestUploadHandlerRejectsInvalidMethod(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	req := httptest.NewRequest(http.MethodGet, "/api/upload", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusMethodNotAllowed)
	}
	if rr.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("allow: got=%q want=%q", rr.Header().Get("Allow"), http.MethodPost)
	}
}

func TestUploadHandlerRejectsMissingDependencies(t *testing.T) {
	h := NewHandler(nil)

	req := httptest.NewRequest(http.MethodPost, "/api/upload", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusServiceUnavailable, rr.Body.String())
	}
}

func decodeLimitError(t *testing.T, rr *httptest.ResponseRecorder) UploadLimitError {
	t.Helper()
	var out UploadLimitError
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode limit error: %v body=%s", err, rr.Body.String())
	}
	return out
}

func TestUploadHandlerRejectsImageOverImageLimit(t *testing.T) {
	svc, sqlDB := setupService(t)
	svc.MaxImageBytes = 1024
	svc.MaxVideoBytes = 1 << 20
	h := NewHandler(svc)

	padded := append(pngBytes(t), make([]byte, 2048)...)
	body, contentType := multipartBody(t, "big.png", padded)
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
	out := decodeLimitError(t, rr)
	if out.Filename != "big.png" || out.MediaType != "image" || out.LimitBytes != 1024 || !strings.Contains(out.Error, "big.png") {
		t.Fatalf("unexpected limit error: %+v", out)
	}
	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 0 {
		t.Fatalf("expected nothing stored, got %d images", imageCount)
	}
}

func TestUploadHandlerAllowsVideoLargerThanImageLimit(t *testing.T) {
	svc, _ := setupService(t)
	svc.VideoSampler = &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 2}
	svc.MaxImageBytes = 1024
	svc.MaxVideoBytes = 1 << 20
	h := NewHandler(svc)

	padded := append(mp4Bytes(), make([]byte, 4096)...)
	body, contentType := multipartBody(t, "clip.mp4", padded)
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusCreated, rr.Body.String())
	}
}

func TestUploadHandlerRejectsVideoOverVideoLimit(t *testing.T) {
	svc, _ := setupService(t)
	svc.VideoSampler = &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 2}
	svc.MaxImageBytes = 1 << 20
	svc.MaxVideoBytes = 2048
	h := NewHandler(svc)

	padded := append(mp4Bytes(), make([]byte, 4096)...)
	body, contentType := multipartBody(t, "clip.mp4", padded)
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
	out := decodeLimitError(t, rr)
	if out.Filename != "clip.mp4" || out.MediaType != "video" || out.LimitBytes != 2048 {
		t.Fatalf("unexpected limit error: %+v", out)
	}
}

func TestUploadHandlerRejectsWholeBatchWhenOneFileIsOversized(t *testing.T) {
	svc, sqlDB := setupService(t)
	svc.MaxImageBytes = 1024
	h := NewHandler(svc)

	body, contentType := multipartBodyWithFiles(t, []struct {
		filename string
		content  []byte
	}{
		{filename: "ok.png", content: pngBytes(t)},
		{filename: "big.png", content: append(pngBytes(t), make([]byte, 2048)...)},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
	if out := decodeLimitError(t, rr); out.Filename != "big.png" {
		t.Fatalf("expected the oversized file to be named, got %+v", out)
	}
	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 0 {
		t.Fatalf("expected nothing stored from a rejected batch, got %d images", imageCount)
	}
}

func TestUploadHandlerRejectsDeclaredOversizeWithoutReadingBody(t *testing.T) {
	svc, _ := setupService(t)
	svc.MaxVideoBytes = 1 << 20
	h := NewHandler(svc)

	req := httptest.NewRequest(http.MethodPost, "/api/upload", &blockingReader{})
	req.Header.Set("Content-Type", "multipart/form-data; boundary=x")
	req.ContentLength = maxUploadFilesPerRequest*(1<<20) + 1
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
}

// blockingReader fails the test if the handler reads from it.
type blockingReader struct{}

func (blockingReader) Read([]byte) (int, error) {
	return 0, fmt.Errorf("body must not be read")
}

// TestUploadHandlerOutlivesServerReadTimeout proves the per-request deadline
// overrides a server configured with a short ReadTimeout: the body trickles
// in slower than the server timeout and still completes.
func TestUploadHandlerOutlivesServerReadTimeout(t *testing.T) {
	svc, _ := setupService(t)
	svc.RequestTimeout = 10 * time.Second
	server := httptest.NewUnstartedServer(NewHandler(svc))
	server.Config.ReadTimeout = 200 * time.Millisecond
	server.Config.WriteTimeout = 200 * time.Millisecond
	server.Start()
	defer server.Close()

	body, contentType := multipartBody(t, "slow.png", pngBytes(t))
	pr, pw := io.Pipe()
	go func() {
		defer pw.Close()
		data := body.Bytes()
		half := len(data) / 2
		_, _ = pw.Write(data[:half])
		time.Sleep(500 * time.Millisecond)
		_, _ = pw.Write(data[half:])
	}()

	req, err := http.NewRequest(http.MethodPost, server.URL+"/api/upload", pr)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", contentType)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("slow upload failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusCreated {
		payload, _ := io.ReadAll(resp.Body)
		t.Fatalf("status: got=%d want=%d body=%s", resp.StatusCode, http.StatusCreated, payload)
	}
}

func TestUploadHandlerRejectsPayloadTooLarge(t *testing.T) {
	svc, _ := setupService(t)
	svc.MaxImageBytes = 1 << 20
	svc.MaxVideoBytes = 1 << 20
	h := NewHandler(svc)

	large := bytes.Repeat([]byte("a"), maxUploadFilesPerRequest*(1<<20)+1024)
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, err := mw.CreateFormFile("file", "big.png")
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := io.Copy(fw, bytes.NewReader(large)); err != nil {
		t.Fatalf("write large content: %v", err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
	if out := decodeLimitError(t, rr); out.LimitBytes != maxUploadFilesPerRequest*(1<<20) {
		t.Fatalf("expected request limit in body, got %+v", out)
	}
}

func TestUploadHandlerRejectsTooManyFilesInSingleRequest(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	files := make([]struct {
		filename string
		content  []byte
	}, 0, maxUploadFilesPerRequest+1)
	for i := 0; i < maxUploadFilesPerRequest+1; i++ {
		files = append(files, struct {
			filename string
			content  []byte
		}{
			filename: fmt.Sprintf("file-%d.png", i),
			content:  pngBytes(t),
		})
	}

	body, contentType := multipartBodyWithFiles(t, files)
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "too many files") {
		t.Fatalf("expected too-many-files error, got %s", rr.Body.String())
	}
}

func TestUploadHandlerAcceptsMaxFilesPerRequest(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	files := make([]struct {
		filename string
		content  []byte
	}, 0, maxUploadFilesPerRequest)
	for i := 0; i < maxUploadFilesPerRequest; i++ {
		files = append(files, struct {
			filename string
			content  []byte
		}{
			filename: fmt.Sprintf("file-%d.png", i),
			content:  pngBytes(t),
		})
	}

	body, contentType := multipartBodyWithFiles(t, files)
	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusCreated, rr.Body.String())
	}

	var resp UploadBatchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Uploads) != maxUploadFilesPerRequest {
		t.Fatalf("upload count: got=%d want=%d", len(resp.Uploads), maxUploadFilesPerRequest)
	}
}

func TestUploadHandlerRejectsNonMultipartContentType(t *testing.T) {
	svc, _ := setupService(t)
	h := NewHandler(svc)

	req := httptest.NewRequest(http.MethodPost, "/api/upload", strings.NewReader("raw body"))
	req.Header.Set("Content-Type", "application/octet-stream")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
}
