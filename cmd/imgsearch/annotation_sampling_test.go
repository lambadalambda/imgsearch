package main

import (
	"strings"
	"testing"
)

func TestResolveLlamaNativeAnnotationSamplingAcceptsRandomSeedDefault(t *testing.T) {
	temperature, seed, err := resolveLlamaNativeAnnotationSampling(defaultLlamaNativeAnnotationTemperature, defaultLlamaNativeAnnotationSeed)
	if err != nil {
		t.Fatalf("resolve sampling: %v", err)
	}
	if temperature != float32(defaultLlamaNativeAnnotationTemperature) {
		t.Fatalf("temperature=%v want=%v", temperature, float32(defaultLlamaNativeAnnotationTemperature))
	}
	if seed != defaultLlamaNativeAnnotationSeed {
		t.Fatalf("seed=%d want=%d", seed, defaultLlamaNativeAnnotationSeed)
	}
}

func TestResolveLlamaNativeAnnotationSamplingAcceptsMaxSeed(t *testing.T) {
	temperature, seed, err := resolveLlamaNativeAnnotationSampling(0.9, defaultLlamaNativeMaxAnnotationSeed)
	if err != nil {
		t.Fatalf("resolve sampling: %v", err)
	}
	if temperature != float32(0.9) {
		t.Fatalf("temperature=%v want=%v", temperature, float32(0.9))
	}
	if seed != defaultLlamaNativeMaxAnnotationSeed {
		t.Fatalf("seed=%d want=%d", seed, defaultLlamaNativeMaxAnnotationSeed)
	}
}

func TestResolveLlamaNativeAnnotationSamplingRejectsTemperatureAboveMaximum(t *testing.T) {
	_, _, err := resolveLlamaNativeAnnotationSampling(2.1, defaultLlamaNativeAnnotationSeed)
	if err == nil {
		t.Fatal("expected temperature range error")
	}
	if !strings.Contains(err.Error(), "annotation temperature") {
		t.Fatalf("unexpected error: %v", err)
	}
}
