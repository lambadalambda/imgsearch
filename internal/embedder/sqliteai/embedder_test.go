package sqliteai

import (
	"database/sql"
	"encoding/binary"
	"math"
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
