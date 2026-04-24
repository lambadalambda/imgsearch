package llamacppnative

import (
	"math"
	"testing"
)

func TestAnnotationImageMaxSideWithMultiplierDefaultsToBase(t *testing.T) {
	if got := annotationImageMaxSideWithMultiplier(384, 0); got != 384 {
		t.Fatalf("expected base max side for multiplier=0, got %d", got)
	}
	if got := annotationImageMaxSideWithMultiplier(384, 1); got != 384 {
		t.Fatalf("expected base max side for multiplier=1, got %d", got)
	}
}

func TestAnnotationImageMaxSideWithMultiplierScalesByMultiplier(t *testing.T) {
	if got := annotationImageMaxSideWithMultiplier(384, 2); got != 768 {
		t.Fatalf("expected doubled max side, got %d", got)
	}
}

func TestAnnotationImageMaxSideWithMultiplierClampsOnOverflow(t *testing.T) {
	got := annotationImageMaxSideWithMultiplier(math.MaxInt/2+1, 2)
	if got != math.MaxInt {
		t.Fatalf("expected max int clamp on overflow, got %d", got)
	}
}
