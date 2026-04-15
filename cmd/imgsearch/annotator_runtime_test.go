package main

import (
	"context"
	"testing"
)

func TestResolveAnnotatorAssetPathsSkipsResolutionWhenDisabled(t *testing.T) {
	called := false
	modelPath, mmprojPath, err := resolveAnnotatorAssetPaths(
		context.Background(),
		false,
		defaultLlamaNativeAnnotatorVariant,
		"",
		"",
		func(context.Context, string, string, string) (string, string, error) {
			called = true
			return "model.gguf", "mmproj.gguf", nil
		},
	)
	if err != nil {
		t.Fatalf("resolve annotator assets: %v", err)
	}
	if called {
		t.Fatal("expected annotator asset resolver to be skipped when disabled")
	}
	if modelPath != "" || mmprojPath != "" {
		t.Fatalf("expected empty annotator paths when disabled, got %q %q", modelPath, mmprojPath)
	}
}

func TestResolveAnnotatorAssetPathsRejectsExplicitPathsWhenDisabled(t *testing.T) {
	_, _, err := resolveAnnotatorAssetPaths(
		context.Background(),
		false,
		defaultLlamaNativeAnnotatorVariant,
		"/models/custom.gguf",
		"/models/custom-mmproj.gguf",
		func(context.Context, string, string, string) (string, string, error) {
			return "model.gguf", "mmproj.gguf", nil
		},
	)
	if err == nil {
		t.Fatal("expected conflicting flag error")
	}
}

func TestResolveAnnotatorAssetPathsResolvesDefaultsWhenEnabled(t *testing.T) {
	called := false
	modelPath, mmprojPath, err := resolveAnnotatorAssetPaths(
		context.Background(),
		true,
		defaultLlamaNativeAnnotatorVariant,
		"",
		"",
		func(context.Context, string, string, string) (string, string, error) {
			called = true
			return "model.gguf", "mmproj.gguf", nil
		},
	)
	if err != nil {
		t.Fatalf("resolve annotator assets: %v", err)
	}
	if !called {
		t.Fatal("expected annotator asset resolver to be called when enabled")
	}
	if modelPath != "model.gguf" || mmprojPath != "mmproj.gguf" {
		t.Fatalf("unexpected resolved annotator paths: %q %q", modelPath, mmprojPath)
	}
}

func TestResolveAnnotatorAssetPathsKeepsExplicitPathsWhenEnabled(t *testing.T) {
	called := false
	modelPath, mmprojPath, err := resolveAnnotatorAssetPaths(
		context.Background(),
		true,
		defaultLlamaNativeAnnotatorVariant,
		" /models/custom.gguf ",
		" /models/custom-mmproj.gguf ",
		func(context.Context, string, string, string) (string, string, error) {
			called = true
			return "model.gguf", "mmproj.gguf", nil
		},
	)
	if err != nil {
		t.Fatalf("resolve annotator assets: %v", err)
	}
	if called {
		t.Fatal("expected annotator asset resolver to be skipped for explicit paths")
	}
	if modelPath != "/models/custom.gguf" || mmprojPath != "/models/custom-mmproj.gguf" {
		t.Fatalf("unexpected explicit annotator paths: %q %q", modelPath, mmprojPath)
	}
}
