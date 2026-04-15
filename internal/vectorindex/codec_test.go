package vectorindex

import "testing"

func TestFloatBlobRoundTrip(t *testing.T) {
	values := []float32{1.5, -2.25, 0, 42}
	blob := FloatsToBlob(values)
	roundTrip := BlobToFloats(blob)
	if len(roundTrip) != len(values) {
		t.Fatalf("length: got=%d want=%d", len(roundTrip), len(values))
	}
	for i := range values {
		if roundTrip[i] != values[i] {
			t.Fatalf("value %d: got=%v want=%v", i, roundTrip[i], values[i])
		}
	}
}

func TestBlobToFloatsRejectsMisalignedBlob(t *testing.T) {
	if got := BlobToFloats([]byte{1, 2, 3}); got != nil {
		t.Fatalf("expected nil for misaligned blob, got %v", got)
	}
}
