package httputil

import (
	"net/http/httptest"
	"testing"
)

func TestParseLimitQuery(t *testing.T) {
	req := httptest.NewRequest("GET", "/api/images", nil)
	if got := ParseLimitQuery(req, 50); got != 50 {
		t.Fatalf("missing limit: got=%d want=%d", got, 50)
	}

	req = httptest.NewRequest("GET", "/api/images?limit=25", nil)
	if got := ParseLimitQuery(req, 50); got != 25 {
		t.Fatalf("valid limit: got=%d want=%d", got, 25)
	}

	req = httptest.NewRequest("GET", "/api/images?limit=0", nil)
	if got := ParseLimitQuery(req, 50); got != 50 {
		t.Fatalf("zero limit fallback: got=%d want=%d", got, 50)
	}

	req = httptest.NewRequest("GET", "/api/images?limit=-5", nil)
	if got := ParseLimitQuery(req, 50); got != 50 {
		t.Fatalf("negative limit fallback: got=%d want=%d", got, 50)
	}

	req = httptest.NewRequest("GET", "/api/images?limit=999", nil)
	if got := ParseLimitQuery(req, 50); got != 50 {
		t.Fatalf("oversized limit fallback: got=%d want=%d", got, 50)
	}

	req = httptest.NewRequest("GET", "/api/images?limit=200", nil)
	if got := ParseLimitQuery(req, 50); got != 200 {
		t.Fatalf("max limit accepted: got=%d want=%d", got, 200)
	}

	req = httptest.NewRequest("GET", "/api/images?limit=abc", nil)
	if got := ParseLimitQuery(req, 50); got != 50 {
		t.Fatalf("non-integer limit fallback: got=%d want=%d", got, 50)
	}
}

func TestParseOffsetQuery(t *testing.T) {
	req := httptest.NewRequest("GET", "/api/images", nil)
	if got := ParseOffsetQuery(req, 0); got != 0 {
		t.Fatalf("missing offset: got=%d want=%d", got, 0)
	}

	req = httptest.NewRequest("GET", "/api/images?offset=42", nil)
	if got := ParseOffsetQuery(req, 0); got != 42 {
		t.Fatalf("valid offset: got=%d want=%d", got, 42)
	}

	req = httptest.NewRequest("GET", "/api/images?offset=-1", nil)
	if got := ParseOffsetQuery(req, 0); got != 0 {
		t.Fatalf("negative offset fallback: got=%d want=%d", got, 0)
	}

	req = httptest.NewRequest("GET", "/api/images?offset=abc", nil)
	if got := ParseOffsetQuery(req, 0); got != 0 {
		t.Fatalf("non-integer offset fallback: got=%d want=%d", got, 0)
	}

	req = httptest.NewRequest("GET", "/api/search/tags?offset=1000001", nil)
	if got := ParseOffsetQuery(req, 1000000); got != 0 {
		t.Fatalf("max offset fallback: got=%d want=%d", got, 0)
	}

	req = httptest.NewRequest("GET", "/api/search/tags?offset=1000000", nil)
	if got := ParseOffsetQuery(req, 1000000); got != 1000000 {
		t.Fatalf("max offset accepted: got=%d want=%d", got, 1000000)
	}
}

func TestParseIncludeNSFWQuery(t *testing.T) {
	req := httptest.NewRequest("GET", "/api/images", nil)
	if got := ParseIncludeNSFWQuery(req); got {
		t.Fatalf("missing include_nsfw: got=%v want=false", got)
	}

	req = httptest.NewRequest("GET", "/api/images?include_nsfw=1", nil)
	if got := ParseIncludeNSFWQuery(req); !got {
		t.Fatalf("include_nsfw=1: got=%v want=true", got)
	}

	req = httptest.NewRequest("GET", "/api/images?include_nsfw=TRUE", nil)
	if got := ParseIncludeNSFWQuery(req); !got {
		t.Fatalf("include_nsfw=TRUE: got=%v want=true", got)
	}

	req = httptest.NewRequest("GET", "/api/images?include_nsfw=not-a-bool", nil)
	if got := ParseIncludeNSFWQuery(req); got {
		t.Fatalf("invalid include_nsfw: got=%v want=false", got)
	}
}

func TestParseOrderQuery(t *testing.T) {
	req := httptest.NewRequest("GET", "/api/images", nil)
	if got := ParseOrderQuery(req, "newest", "newest", "random"); got != "newest" {
		t.Fatalf("missing order: got=%q want=newest", got)
	}

	req = httptest.NewRequest("GET", "/api/images?order=random", nil)
	if got := ParseOrderQuery(req, "newest", "newest", "random"); got != "random" {
		t.Fatalf("random order: got=%q want=random", got)
	}

	req = httptest.NewRequest("GET", "/api/images?order=RANDOM", nil)
	if got := ParseOrderQuery(req, "newest", "newest", "random"); got != "random" {
		t.Fatalf("case-insensitive order: got=%q want=random", got)
	}

	req = httptest.NewRequest("GET", "/api/images?order=shuffle", nil)
	if got := ParseOrderQuery(req, "newest", "newest", "random"); got != "newest" {
		t.Fatalf("unknown order fallback: got=%q want=newest", got)
	}
}

func TestParseInt64Query(t *testing.T) {
	req := httptest.NewRequest("GET", "/api/images", nil)
	if got := ParseInt64Query(req, "seed", 9); got != 9 {
		t.Fatalf("missing int64: got=%d want=9", got)
	}

	req = httptest.NewRequest("GET", "/api/images?seed=1200000000", nil)
	if got := ParseInt64Query(req, "seed", 9); got != 1200000000 {
		t.Fatalf("valid int64: got=%d want=1200000000", got)
	}

	req = httptest.NewRequest("GET", "/api/images?seed=nope", nil)
	if got := ParseInt64Query(req, "seed", 9); got != 9 {
		t.Fatalf("invalid int64 fallback: got=%d want=9", got)
	}
}
