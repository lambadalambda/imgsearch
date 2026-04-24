package httputil

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseItemIDPath(t *testing.T) {
	id, err := ParseItemIDPath("/api/images/42", "/api/images/")
	if err != nil {
		t.Fatalf("parse valid id: %v", err)
	}
	if id != 42 {
		t.Fatalf("valid id: got=%d want=%d", id, 42)
	}

	if _, err := ParseItemIDPath("/api/images", "/api/images/"); err == nil {
		t.Fatal("expected error when prefix does not match")
	}
	if _, err := ParseItemIDPath("/api/images/", "/api/images/"); err == nil {
		t.Fatal("expected error for empty id")
	}
	if _, err := ParseItemIDPath("/api/images/42/extra", "/api/images/"); err == nil {
		t.Fatal("expected error for nested path segment")
	}
	if _, err := ParseItemIDPath("/api/images/not-a-number", "/api/images/"); err == nil {
		t.Fatal("expected error for non-numeric id")
	}
}

func TestParseItemActionIDPath(t *testing.T) {
	id, err := ParseItemActionIDPath("/api/images/42/reannotate", "/api/images/", "reannotate")
	if err != nil {
		t.Fatalf("parse valid action id: %v", err)
	}
	if id != 42 {
		t.Fatalf("valid id: got=%d want=%d", id, 42)
	}

	if _, err := ParseItemActionIDPath("/api/images/42", "/api/images/", "reannotate"); err == nil {
		t.Fatal("expected error for missing action suffix")
	}
	if _, err := ParseItemActionIDPath("/api/images/42/toggle-nsfw/extra", "/api/images/", "toggle-nsfw"); err == nil {
		t.Fatal("expected error for nested action path")
	}
}

func TestBoolToInt(t *testing.T) {
	if BoolToInt(true) != 1 {
		t.Fatalf("true should convert to 1")
	}
	if BoolToInt(false) != 0 {
		t.Fatalf("false should convert to 0")
	}
}

func TestRemoveStoredPathRemovesContainedPath(t *testing.T) {
	dataDir := t.TempDir()
	rel := filepath.Join("images", "sample.jpg")
	abs := filepath.Join(dataDir, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatalf("mkdir parent: %v", err)
	}
	if err := os.WriteFile(abs, []byte("x"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	if err := RemoveStoredPath(dataDir, filepath.ToSlash(rel)); err != nil {
		t.Fatalf("remove stored path: %v", err)
	}
	if _, err := os.Stat(abs); !os.IsNotExist(err) {
		t.Fatalf("expected file removed, stat err=%v", err)
	}
}

func TestRemoveStoredPathRejectsEscapingPath(t *testing.T) {
	root := t.TempDir()
	dataDir := filepath.Join(root, "data")
	outside := filepath.Join(root, "outside.txt")

	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("mkdir data dir: %v", err)
	}
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}

	if err := RemoveStoredPath(dataDir, "../outside.txt"); err == nil {
		t.Fatal("expected path containment error")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatalf("expected outside file untouched, stat err=%v", err)
	}
}

func TestRemoveStoredPathAllowsMissingContainedFile(t *testing.T) {
	dataDir := t.TempDir()
	if err := RemoveStoredPath(dataDir, "images/missing.jpg"); err != nil {
		t.Fatalf("expected nil for missing contained file, got %v", err)
	}
}

func TestRemoveStoredPathRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	dataDir := filepath.Join(root, "data")
	outsideDir := filepath.Join(root, "outside")
	outsideFile := filepath.Join(outsideDir, "victim.txt")

	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("mkdir data dir: %v", err)
	}
	if err := os.MkdirAll(outsideDir, 0o755); err != nil {
		t.Fatalf("mkdir outside dir: %v", err)
	}
	if err := os.WriteFile(outsideFile, []byte("outside"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}

	linkPath := filepath.Join(dataDir, "escape")
	if err := os.Symlink(outsideDir, linkPath); err != nil {
		t.Skipf("symlink setup unavailable: %v", err)
	}

	if err := RemoveStoredPath(dataDir, "escape/victim.txt"); err == nil {
		t.Fatal("expected symlink containment error")
	}
	if _, err := os.Stat(outsideFile); err != nil {
		t.Fatalf("expected outside file untouched, stat err=%v", err)
	}
}
