package main

import "fmt"

func resolveLlamaNativeAnnotationSampling(annotationTemperature float64, annotationSeed int64) (float32, int64, error) {
	if annotationTemperature < 0 {
		return 0, 0, fmt.Errorf("llama-cpp-native annotation temperature must be non-negative")
	}
	if annotationTemperature > 2 {
		return 0, 0, fmt.Errorf("llama-cpp-native annotation temperature must be <= 2")
	}
	if annotationSeed < defaultLlamaNativeAnnotationSeed || annotationSeed > defaultLlamaNativeMaxAnnotationSeed {
		return 0, 0, fmt.Errorf("llama-cpp-native annotation seed must be between %d and %d", defaultLlamaNativeAnnotationSeed, defaultLlamaNativeMaxAnnotationSeed)
	}
	return float32(annotationTemperature), annotationSeed, nil
}
