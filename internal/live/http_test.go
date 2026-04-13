package live

import (
	"context"
	"database/sql"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func setupLiveDB(t *testing.T) *sql.DB {
	t.Helper()

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
	(1, 'a', 'one.jpg', 'images/a', 'image/jpeg', 10, 10),
	(2, 'b', 'two.jpg', 'images/b', 'image/jpeg', 20, 20)
`)
	if err != nil {
		t.Fatalf("seed images: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO index_jobs(kind, image_id, model_id, state, attempts, max_attempts, last_error)
VALUES
	('embed_image', 1, 1, 'done', 1, 3, NULL),
	('embed_image', 2, 1, 'pending', 1, 3, NULL)
`)
	if err != nil {
		t.Fatalf("seed jobs: %v", err)
	}

	return dbConn
}

func TestLiveHandlerStreamsSnapshot(t *testing.T) {
	dbConn := setupLiveDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, Interval: 25 * time.Millisecond, ImagesLimit: 10})

	srv := httptest.NewServer(h)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial websocket: %v", err)
	}
	defer func() { _ = conn.Close() }()

	if err := conn.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatalf("set read deadline: %v", err)
	}

	var snapshot Snapshot
	if err := conn.ReadJSON(&snapshot); err != nil {
		t.Fatalf("read snapshot: %v", err)
	}

	if snapshot.Type != "snapshot" {
		t.Fatalf("snapshot type: got=%q want=snapshot", snapshot.Type)
	}
	if snapshot.Images.Total != 2 {
		t.Fatalf("images total: got=%d want=2", snapshot.Images.Total)
	}
	if len(snapshot.Images.Images) != 2 {
		t.Fatalf("images length: got=%d want=2", len(snapshot.Images.Images))
	}
	if snapshot.Stats.ImagesTotal != 2 {
		t.Fatalf("stats images total: got=%d want=2", snapshot.Stats.ImagesTotal)
	}
	if snapshot.Stats.Queue.Done != 1 || snapshot.Stats.Queue.Pending != 1 {
		t.Fatalf("unexpected queue stats: %+v", snapshot.Stats.Queue)
	}

	var next Snapshot
	if err := conn.ReadJSON(&next); err != nil {
		t.Fatalf("read second snapshot: %v", err)
	}
	if next.Type != "snapshot" {
		t.Fatalf("second snapshot type: got=%q want=snapshot", next.Type)
	}
}

func TestWebsocketOriginValidation(t *testing.T) {
	tests := []struct {
		name   string
		host   string
		origin string
		want   bool
	}{
		{name: "matching host", host: "example.com", origin: "https://example.com", want: true},
		{name: "matching default https port", host: "example.com:443", origin: "https://example.com", want: true},
		{name: "matching default http port", host: "example.com:80", origin: "http://example.com", want: true},
		{name: "mismatched host", host: "example.com", origin: "https://evil.example", want: false},
		{name: "invalid origin", host: "example.com", origin: "%%%", want: false},
		{name: "empty origin allows host", host: "example.com:8080", origin: "", want: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "http://"+tc.host+"/api/live", nil)
			req.Host = tc.host
			if tc.origin != "" {
				req.Header.Set("Origin", tc.origin)
			}
			got := upgrader.CheckOrigin(req)
			if got != tc.want {
				t.Fatalf("check origin: got=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestLiveHandlerRejectsInvalidMethod(t *testing.T) {
	h := NewHandler(&Handler{DB: setupLiveDB(t), ModelID: 1})

	req := httptest.NewRequest(http.MethodPost, "/api/live", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusMethodNotAllowed)
	}
}

func TestLiveHandlerRequiresDB(t *testing.T) {
	h := NewHandler(&Handler{})

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusInternalServerError)
	}
}
