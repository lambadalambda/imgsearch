package llamacppnative

import "fmt"

const (
	maxAnnotationSeed int64 = 4294967295
)

func normalizeAnnotationSampling(temperature float32, seed int64) (float32, int64, error) {
	if temperature < 0 {
		return 0, 0, fmt.Errorf("llama-cpp-native annotation temperature must be non-negative")
	}
	if temperature > 2 {
		return 0, 0, fmt.Errorf("llama-cpp-native annotation temperature must be <= 2")
	}
	if seed < -1 || seed > maxAnnotationSeed {
		return 0, 0, fmt.Errorf("llama-cpp-native annotation seed must be between -1 and %d", maxAnnotationSeed)
	}
	return temperature, seed, nil
}
