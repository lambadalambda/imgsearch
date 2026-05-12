package upload

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestMoveFileFallsBackWhenRenameCrossesDevices(t *testing.T) {
	tmpDir := t.TempDir()
	src := filepath.Join(tmpDir, "upload.tmp")
	dst := filepath.Join(tmpDir, "images", "stored")
	content := []byte("stored bytes")

	if err := os.WriteFile(src, content, 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatalf("create destination dir: %v", err)
	}

	rename := func(oldPath, newPath string) error {
		return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: syscall.EXDEV}
	}

	if err := moveFileWithRename(src, dst, rename); err != nil {
		t.Fatalf("move file: %v", err)
	}

	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read destination: %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("destination content mismatch: got=%q want=%q", got, content)
	}
	if _, err := os.Stat(src); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected source to be removed, got err=%v", err)
	}
}
