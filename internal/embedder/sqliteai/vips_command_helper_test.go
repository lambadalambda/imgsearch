package sqliteai

import (
	"context"
	"runtime"
	"strings"
	"testing"
)

func TestVipsCommandHelperRunsCommandAndReturnsOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("helper test uses /bin/sh")
	}

	h, err := startVipsCommandHelper(context.Background())
	if err != nil {
		t.Fatalf("start helper: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })

	out, err := h.Run(context.Background(), "/bin/sh", "-c", "echo hello")
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if string(out) != "hello\n" {
		t.Fatalf("output: got=%q want=%q", string(out), "hello\\n")
	}
}

func TestVipsCommandHelperReturnsExitErrorAndOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("helper test uses /bin/sh")
	}

	h, err := startVipsCommandHelper(context.Background())
	if err != nil {
		t.Fatalf("start helper: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })

	out, err := h.Run(context.Background(), "/bin/sh", "-c", "echo boom; exit 7")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "exit status") {
		t.Fatalf("expected exit status in error, got: %v", err)
	}
	if string(out) != "boom\n" {
		t.Fatalf("output: got=%q want=%q", string(out), "boom\\n")
	}
}
