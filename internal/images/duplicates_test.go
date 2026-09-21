package images

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"imgsearch/internal/phash"
)

func TestGroupByHashFindsPairsWithinDistanceOnly(t *testing.T) {
	base := uint64(0x0f0f0f0f0f0f0f0f)
	hashed := []hashedImage{
		{id: 1, hash: base},
		{id: 2, hash: base ^ 0x3},                // distance 2
		{id: 3, hash: base ^ 0x3 ^ (1 << 40)},    // distance 1 from 2, 3 from 1: joins transitively
		{id: 4, hash: ^base},                     // distance 64
		{id: 5, hash: base ^ 0xff00},             // distance 8: outside the exact bound
		{id: 6, hash: 0xaaaaaaaaaaaaaaaa},        // lone
		{id: 7, hash: 0xaaaaaaaaaaaaaaaa ^ 0x11}, // distance 2 from 6
	}
	groups := groupByHash(hashed, 4)
	if len(groups) != 2 {
		t.Fatalf("groups: %v", groups)
	}
	if len(groups[0]) != 3 || groups[0][0] != 1 || len(groups[1]) != 2 || groups[1][0] != 6 {
		t.Fatalf("groups: %v", groups)
	}
	if got := groupByHash(hashed, 0); len(got) != 0 {
		t.Fatalf("distance 0 should find nothing here: %v", got)
	}
}

func TestDuplicatesHandlerGroupsAndOrdersByPixelCount(t *testing.T) {
	dbConn := setupImagesDB(t)
	near := phash.ToInt64(0x1234567890abcdef)
	far := phash.ToInt64(0xfedcba0987654321)
	if _, err := dbConn.Exec(`UPDATE images SET phash = CASE id WHEN 1 THEN ? WHEN 2 THEN ? ELSE ? END, width = CASE id WHEN 1 THEN 10 WHEN 2 THEN 40 ELSE 30 END, height = 10`,
		near, near^1, far); err != nil {
		t.Fatalf("seed hashes: %v", err)
	}
	if _, err := dbConn.Exec(`INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, phash) VALUES (4, 'd', 'avif.avif', 'images/d', 'image/avif', 5, 5, -1), (5, 'e', 'new.jpg', 'images/e', 'image/jpeg', 5, 5, NULL)`); err != nil {
		t.Fatalf("seed unhashed: %v", err)
	}
	h := NewDuplicatesHandler(&Handler{DB: dbConn, ModelID: 1})
	req := httptest.NewRequest(http.MethodGet, "/api/duplicates?distance=3", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp DuplicatesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Scanned != 3 || resp.Unhashed != 2 || resp.MaxDistance != 3 {
		t.Fatalf("counts: %+v", resp)
	}
	if len(resp.Groups) != 1 || len(resp.Groups[0].Items) != 2 {
		t.Fatalf("groups: %+v", resp.Groups)
	}
	if resp.Groups[0].Items[0].ImageID != 2 || resp.Groups[0].Items[1].ImageID != 1 {
		t.Fatalf("expected the larger image first, got %d then %d", resp.Groups[0].Items[0].ImageID, resp.Groups[0].Items[1].ImageID)
	}

	// The distance is capped at the exact bound of the band index.
	resp, err := FindDuplicates(context.Background(), dbConn, 1, 50, true)
	if err != nil || resp.MaxDistance != MaxDuplicateDistance {
		t.Fatalf("cap: %+v err=%v", resp, err)
	}
	req = httptest.NewRequest(http.MethodPost, "/api/duplicates", nil)
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("post: got=%d", rr.Code)
	}
}
