package buildinfo

import (
	"runtime/debug"
	"testing"
)

func TestResolvePrefersStampedValues(t *testing.T) {
	got := resolve("v1.2.3", "abc123", "2026-09-21T00:00:00Z", func() (*debug.BuildInfo, bool) {
		t.Fatal("build info must not be read when everything is stamped")
		return nil, false
	})
	want := Info{Version: "v1.2.3", Commit: "abc123", Date: "2026-09-21T00:00:00Z"}
	if got != want {
		t.Fatalf("resolve: got=%+v want=%+v", got, want)
	}
	if got.String() != "imgsearch v1.2.3 (commit abc123, built 2026-09-21T00:00:00Z)" {
		t.Fatalf("unexpected String(): %q", got.String())
	}
}

func TestResolveFallsBackToVCSMetadata(t *testing.T) {
	read := func() (*debug.BuildInfo, bool) {
		return &debug.BuildInfo{Settings: []debug.BuildSetting{
			{Key: "vcs.revision", Value: "deadbeef"},
			{Key: "vcs.time", Value: "2026-09-20T10:00:00Z"},
			{Key: "vcs.modified", Value: "true"},
		}}, true
	}
	got := resolve("", "", "", read)
	want := Info{Version: "dev", Commit: "deadbeef-dirty", Date: "2026-09-20T10:00:00Z"}
	if got != want {
		t.Fatalf("resolve: got=%+v want=%+v", got, want)
	}
}

func TestResolveWithoutAnyMetadata(t *testing.T) {
	got := resolve("", "", "", func() (*debug.BuildInfo, bool) { return nil, false })
	if got.Version != "dev" || got.Commit != "" || got.String() != "imgsearch dev (commit unknown, built unknown)" {
		t.Fatalf("unexpected fallback: %+v %q", got, got.String())
	}
}
