package exif

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func fixtureJPEG(t *testing.T) []byte {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	data, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", "cat_1.jpg"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return data
}

func TestParseReadsOrientationAndCaptureTime(t *testing.T) {
	jpeg := InsertAPP1(fixtureJPEG(t), BuildAPP1(6, "2024:05:06 07:08:09"))
	info, err := Parse(bytes.NewReader(jpeg))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if info.Orientation != 6 || !info.SwapsDimensions() {
		t.Fatalf("orientation: got=%d swaps=%v", info.Orientation, info.SwapsDimensions())
	}
	want := time.Date(2024, 5, 6, 7, 8, 9, 0, time.UTC)
	if !info.CapturedAt.Equal(want) {
		t.Fatalf("captured at: got=%s want=%s", info.CapturedAt, want)
	}
	if SQLiteTime(info.CapturedAt) != "2024-05-06 07:08:09" {
		t.Fatalf("sqlite time: %q", SQLiteTime(info.CapturedAt))
	}
}

func TestParseWithoutExifOrNonJPEG(t *testing.T) {
	// SOI, an unrelated APP0 segment, SOS, EOI: a JPEG skeleton without EXIF.
	bare := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x02, 0xFF, 0xD9}
	info, err := Parse(bytes.NewReader(bare))
	if err != nil || info != (Info{}) {
		t.Fatalf("bare jpeg: info=%+v err=%v", info, err)
	}
	// The cat fixture carries its own camera EXIF (orientation 1, 2011).
	info, err = Parse(bytes.NewReader(fixtureJPEG(t)))
	if err != nil || info.Orientation != 1 || info.CapturedAt.Year() != 2011 {
		t.Fatalf("fixture jpeg: info=%+v err=%v", info, err)
	}
	info, err = Parse(bytes.NewReader([]byte("\x89PNG\r\n\x1a\n not a jpeg")))
	if err != nil || info != (Info{}) {
		t.Fatalf("png: info=%+v err=%v", info, err)
	}
	info, err = Parse(bytes.NewReader(nil))
	if err != nil || info != (Info{}) {
		t.Fatalf("empty: info=%+v err=%v", info, err)
	}
}

func TestParseTIFFBigEndianAndBadInput(t *testing.T) {
	// Big-endian header with a truncated IFD must not panic and yields nothing.
	if got := ParseTIFF([]byte{'M', 'M', 0, 42, 0, 0, 0, 8, 0, 5}); got != (Info{}) {
		t.Fatalf("truncated: %+v", got)
	}
	if got := ParseTIFF([]byte("garbage")); got != (Info{}) {
		t.Fatalf("garbage: %+v", got)
	}
	// Orientation outside 1..8 is dropped; the unset date is ignored.
	tiff := BuildTIFF(9, "0000:00:00 00:00:00")
	if got := ParseTIFF(tiff); got != (Info{}) {
		t.Fatalf("out of range: %+v", got)
	}
	tiff = BuildTIFF(1, "2020:01:02 03:04:05")
	got := ParseTIFF(tiff)
	if got.Orientation != 1 || got.SwapsDimensions() || got.CapturedAt.Year() != 2020 {
		t.Fatalf("orientation 1: %+v", got)
	}
}

func TestParseDateTime(t *testing.T) {
	if _, ok := ParseDateTime("2024-05-06 07:08:09"); ok {
		t.Fatal("dashes are not the EXIF layout")
	}
	if _, ok := ParseDateTime("   "); ok {
		t.Fatal("blank must be unset")
	}
	if got, ok := ParseDateTime("2024:05:06 07:08:09"); !ok || got.Month() != time.May {
		t.Fatalf("valid: %v %v", got, ok)
	}
}
