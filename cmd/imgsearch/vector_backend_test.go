package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveVectorBackendAutoUsesSQLiteVectorWhenAvailable(t *testing.T) {
	backend, warn, err := resolveVectorBackend("auto", nil)
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}
	if backend != "sqlite-vector" {
		t.Fatalf("backend: got=%q want=%q", backend, "sqlite-vector")
	}
	if warn != "" {
		t.Fatalf("expected empty warning, got %q", warn)
	}
}

func TestResolveVectorBackendAutoFallsBackToBruteforce(t *testing.T) {
	backend, warn, err := resolveVectorBackend("auto", errors.New("vector_version missing"))
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}
	if backend != "bruteforce" {
		t.Fatalf("backend: got=%q want=%q", backend, "bruteforce")
	}
	if !strings.Contains(warn, "falling back") {
		t.Fatalf("expected fallback warning, got %q", warn)
	}
}

func TestResolveVectorBackendHonorsExplicitValue(t *testing.T) {
	backend, warn, err := resolveVectorBackend("bruteforce", errors.New("ignored"))
	if err != nil {
		t.Fatalf("resolve backend: %v", err)
	}
	if backend != "bruteforce" {
		t.Fatalf("backend: got=%q want=%q", backend, "bruteforce")
	}
	if warn != "" {
		t.Fatalf("expected empty warning, got %q", warn)
	}
}

func TestResolveVectorBackendRejectsUnknownValue(t *testing.T) {
	_, _, err := resolveVectorBackend("mystery", nil)
	if err == nil {
		t.Fatal("expected unknown backend error")
	}
	if !strings.Contains(err.Error(), "unknown vector backend") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDiscoverSQLiteVectorPathPrefersExplicit(t *testing.T) {
	t.Setenv("SQLITE_VECTOR_PATH", "")

	tmp := t.TempDir()
	explicit := filepath.Join(tmp, "vector.dylib")
	if err := os.WriteFile(explicit, []byte("x"), 0o644); err != nil {
		t.Fatalf("write explicit extension file: %v", err)
	}

	got, err := discoverSQLiteVectorPath(explicit)
	if err != nil {
		t.Fatalf("discover path: %v", err)
	}
	if got != explicit {
		t.Fatalf("path: got=%q want=%q", got, explicit)
	}
}

func TestDiscoverSQLiteVectorPathUsesEnv(t *testing.T) {
	tmp := t.TempDir()
	envPath := filepath.Join(tmp, "vector.so")
	if err := os.WriteFile(envPath, []byte("x"), 0o644); err != nil {
		t.Fatalf("write env extension file: %v", err)
	}
	t.Setenv("SQLITE_VECTOR_PATH", envPath)

	got, err := discoverSQLiteVectorPath("")
	if err != nil {
		t.Fatalf("discover path: %v", err)
	}
	if got != envPath {
		t.Fatalf("path: got=%q want=%q", got, envPath)
	}
}

func TestDiscoverSQLiteVectorPathReturnsEmptyWhenNothingFound(t *testing.T) {
	t.Setenv("SQLITE_VECTOR_PATH", "")

	tmp := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(wd)
	})
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("chdir: %v", err)
	}

	got, err := discoverSQLiteVectorPath("")
	if err != nil {
		t.Fatalf("discover path: %v", err)
	}
	if got != "" {
		t.Fatalf("expected empty path, got %q", got)
	}
}

func TestDiscoverSQLiteAIPathPrefersExplicit(t *testing.T) {
	t.Setenv("SQLITE_AI_PATH", "")

	tmp := t.TempDir()
	explicit := filepath.Join(tmp, "ai.dylib")
	if err := os.WriteFile(explicit, []byte("x"), 0o644); err != nil {
		t.Fatalf("write explicit extension file: %v", err)
	}

	got, err := discoverSQLiteAIPath(explicit)
	if err != nil {
		t.Fatalf("discover path: %v", err)
	}
	if got != explicit {
		t.Fatalf("path: got=%q want=%q", got, explicit)
	}
}

func TestDiscoverSQLiteAIPathUsesEnv(t *testing.T) {
	tmp := t.TempDir()
	envPath := filepath.Join(tmp, "ai.so")
	if err := os.WriteFile(envPath, []byte("x"), 0o644); err != nil {
		t.Fatalf("write env extension file: %v", err)
	}
	t.Setenv("SQLITE_AI_PATH", envPath)

	got, err := discoverSQLiteAIPath("")
	if err != nil {
		t.Fatalf("discover path: %v", err)
	}
	if got != envPath {
		t.Fatalf("path: got=%q want=%q", got, envPath)
	}
}

func TestDiscoverSQLiteAIPathReturnsEmptyWhenNothingFound(t *testing.T) {
	t.Setenv("SQLITE_AI_PATH", "")

	tmp := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(wd)
	})
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("chdir: %v", err)
	}

	got, err := discoverSQLiteAIPath("")
	if err != nil {
		t.Fatalf("discover path: %v", err)
	}
	if got != "" {
		t.Fatalf("expected empty path, got %q", got)
	}
}
