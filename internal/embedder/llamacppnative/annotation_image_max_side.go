package llamacppnative

import "math"

const defaultAnnotationImageMaxSide = 384

func annotationImageMaxSideWithMultiplier(base int, multiplier int) int {
	if base <= 0 {
		base = defaultAnnotationImageMaxSide
	}
	if multiplier <= 1 {
		return base
	}
	if base > math.MaxInt/multiplier {
		return math.MaxInt
	}
	return base * multiplier
}
