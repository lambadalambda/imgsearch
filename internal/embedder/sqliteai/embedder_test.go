package sqliteai

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestNewRejectsNilDB(t *testing.T) {
	_, err := New(Config{ModelPath: "/tmp/model.gguf"})
	if err == nil {
		t.Fatal("expected error for nil db")
	}
}

func TestNewRejectsMissingModelPath(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	_, err = New(Config{DB: dbConn})
	if err == nil {
		t.Fatal("expected error for missing model path")
	}
}

func TestDecodeFloat32BlobRoundTrip(t *testing.T) {
	values := []float32{1.25, -2.5, 3.75}
	blob := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(blob[i*4:], math.Float32bits(v))
	}

	got, err := decodeFloat32Blob(blob)
	if err != nil {
		t.Fatalf("decode blob: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("length: got=%d want=%d", len(got), len(values))
	}
	for i := range values {
		if got[i] != values[i] {
			t.Fatalf("value at %d: got=%f want=%f", i, got[i], values[i])
		}
	}
}

func TestDecodeFloat32BlobRejectsInvalidLength(t *testing.T) {
	_, err := decodeFloat32Blob([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("expected invalid blob length error")
	}
}

func TestDefaultQueryInstructionUsesRetrievalWording(t *testing.T) {
	const want = "Retrieve images or text relevant to the user's query."
	if defaultQueryInstruction != want {
		t.Fatalf("default query instruction: got=%q want=%q", defaultQueryInstruction, want)
	}
}

func TestNewDefaultsContextAndImageMaxSide(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	vipsPath := fakeExecutable(t)

	e, err := New(Config{
		DB:        dbConn,
		ModelPath: "/tmp/model.gguf",
		VipsPath:  vipsPath,
	})
	if err != nil {
		t.Fatalf("new sqlite-ai embedder: %v", err)
	}

	if e.contextOptions != defaultContextOptions {
		t.Fatalf("context options: got=%q want=%q", e.contextOptions, defaultContextOptions)
	}
	if e.queryInstruction != defaultQueryInstruction {
		t.Fatalf("query instruction: got=%q want=%q", e.queryInstruction, defaultQueryInstruction)
	}
	if e.passageInstruction != defaultPassageInstruction {
		t.Fatalf("passage instruction: got=%q want=%q", e.passageInstruction, defaultPassageInstruction)
	}
	if e.imageMaxSide != defaultImageMaxSide {
		t.Fatalf("image max side: got=%d want=%d", e.imageMaxSide, defaultImageMaxSide)
	}
}

func TestNewRejectsNegativeImageMaxSide(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	_, err = New(Config{
		DB:           dbConn,
		ModelPath:    "/tmp/model.gguf",
		ImageMaxSide: -1,
		VipsPath:     fakeExecutable(t),
	})
	if err == nil {
		t.Fatal("expected negative image max side error")
	}
}

func TestNewRejectsInstructionWithComma(t *testing.T) {
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	_, err = New(Config{
		DB:               dbConn,
		ModelPath:        "/tmp/model.gguf",
		VipsPath:         fakeExecutable(t),
		QueryInstruction: "bad, instruction",
	})
	if err == nil {
		t.Fatal("expected query instruction comma error")
	}
}

func TestPrepareImageForEmbeddingUsesVipsResize(t *testing.T) {
	source := filepath.Join(t.TempDir(), "input.webp")
	if err := os.WriteFile(source, []byte("image"), 0o644); err != nil {
		t.Fatalf("write source image: %v", err)
	}

	type commandCall struct {
		name string
		args []string
	}
	var calls []commandCall
	e := &Embedder{
		vipsPath:     "vips",
		imageMaxSide: 512,
		vipsResize:   defaultVipsResize,
		runCommand: func(_ context.Context, name string, args ...string) ([]byte, error) {
			calls = append(calls, commandCall{name: name, args: append([]string(nil), args...)})
			if name != "vips" {
				return nil, errors.New("unexpected command")
			}
			if len(args) != 5 || args[0] != "thumbnail" || args[1] != source {
				return nil, errors.New("unexpected vips thumbnail args")
			}
			if err := os.WriteFile(args[2], []byte("jpeg-bytes"), 0o644); err != nil {
				return nil, err
			}
			return nil, nil
		},
	}

	outPath, cleanup, err := e.prepareImageForEmbedding(context.Background(), source)
	if err != nil {
		t.Fatalf("prepare image: %v", err)
	}
	if cleanup == nil {
		t.Fatal("expected cleanup function")
	}
	defer cleanup()

	if len(calls) != 1 {
		t.Fatalf("command count: got=%d want=1", len(calls))
	}
	if calls[0].name != "vips" {
		t.Fatalf("unexpected command sequence: %+v", calls)
	}
	gotArgs := calls[0].args
	if len(gotArgs) != 5 {
		t.Fatalf("thumbnail arg count: got=%d want=5", len(gotArgs))
	}
	if gotArgs[0] != "thumbnail" {
		t.Fatalf("subcommand: got=%q want=thumbnail", gotArgs[0])
	}
	if gotArgs[1] != source {
		t.Fatalf("source arg: got=%q want=%q", gotArgs[1], source)
	}
	if gotArgs[3] != "512" {
		t.Fatalf("max side arg: got=%q want=512", gotArgs[3])
	}
	if gotArgs[4] != "--size=down" {
		t.Fatalf("size option: got=%q want=--size=down", gotArgs[4])
	}
	if outPath != gotArgs[2] {
		t.Fatalf("output path: got=%q want=%q", outPath, gotArgs[2])
	}
	if filepath.Ext(outPath) != ".jpg" {
		t.Fatalf("output extension: got=%q want=.jpg", filepath.Ext(outPath))
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("stat converted file: %v", err)
	}

	cleanup()
	if _, err := os.Stat(outPath); !os.IsNotExist(err) {
		t.Fatalf("expected converted file removed, stat err=%v", err)
	}
}

func TestPrepareImageForEmbeddingFallsBackToVipsForUnsupportedFormat(t *testing.T) {
	source := filepath.Join(t.TempDir(), "input.avif")
	if err := os.WriteFile(source, []byte("not-an-image"), 0o644); err != nil {
		t.Fatalf("write source image: %v", err)
	}

	type commandCall struct {
		name string
		args []string
	}
	var calls []commandCall
	e := &Embedder{
		vipsPath:     "vips",
		imageMaxSide: 512,
		vipsResize:   defaultVipsResize,
		runCommand: func(_ context.Context, name string, args ...string) ([]byte, error) {
			calls = append(calls, commandCall{name: name, args: append([]string(nil), args...)})
			if name != "vips" {
				return nil, fmt.Errorf("unexpected command %q", name)
			}
			if len(args) != 5 || args[0] != "thumbnail" || args[1] != source || args[3] != "512" || args[4] != "--size=down" {
				return nil, fmt.Errorf("unexpected vips args: %v", args)
			}
			if err := os.WriteFile(args[2], []byte("jpeg-bytes"), 0o644); err != nil {
				return nil, err
			}
			return nil, nil
		},
	}

	outPath, cleanup, err := e.prepareImageForEmbedding(context.Background(), source)
	if err != nil {
		t.Fatalf("prepare image: %v", err)
	}
	if cleanup == nil {
		t.Fatal("expected cleanup function")
	}
	defer cleanup()

	if len(calls) != 1 {
		t.Fatalf("command count: got=%d want=1", len(calls))
	}
	if outPath != calls[0].args[2] {
		t.Fatalf("output path: got=%q want=%q", outPath, calls[0].args[2])
	}
}

func fakeExecutable(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "fake-vips")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatalf("write fake executable: %v", err)
	}
	return path
}
