package llamacppnative

import (
	"strings"
	"testing"
)

func TestNormalizeAnnotationSamplingAcceptsRandomSeed(t *testing.T) {
	temperature, seed, err := normalizeAnnotationSampling(0.2, -1)
	if err != nil {
		t.Fatalf("normalize annotation sampling: %v", err)
	}
	if temperature != float32(0.2) {
		t.Fatalf("temperature=%v want=%v", temperature, float32(0.2))
	}
	if seed != -1 {
		t.Fatalf("seed=%d want=-1", seed)
	}
}

func TestNormalizeAnnotationSamplingRejectsOutOfRangeSeed(t *testing.T) {
	_, _, err := normalizeAnnotationSampling(0.2, maxAnnotationSeed+1)
	if err == nil {
		t.Fatal("expected seed range error")
	}
	if !strings.Contains(err.Error(), "annotation seed") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNormalizeAnnotationSamplingRejectsTemperatureAboveMaximum(t *testing.T) {
	_, _, err := normalizeAnnotationSampling(2.5, -1)
	if err == nil {
		t.Fatal("expected temperature range error")
	}
	if !strings.Contains(err.Error(), "annotation temperature") {
		t.Fatalf("unexpected error: %v", err)
	}
}
