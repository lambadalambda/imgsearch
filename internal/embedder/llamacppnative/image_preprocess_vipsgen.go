//go:build cgo

package llamacppnative

import (
	"fmt"
	"os"
	"strings"

	"imgsearch/internal/imageutil"
)

func preprocessImageForEmbeddingWithVipsgen(sourcePath string, maxSide int) (string, func(), error) {
	sourcePath = strings.TrimSpace(sourcePath)
	if sourcePath == "" {
		return "", nil, fmt.Errorf("image path is empty")
	}
	if maxSide <= 0 {
		return "", nil, fmt.Errorf("llama-cpp-native image max side must be positive")
	}
	tmp, err := os.CreateTemp("", "imgsearch-llama-native-*.jpg")
	if err != nil {
		return "", nil, fmt.Errorf("create temporary jpeg: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpPath) }
	if err := tmp.Close(); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("close temporary jpeg: %w", err)
	}
	if err := imageutil.ThumbnailJPEG(sourcePath, tmpPath, maxSide); err != nil {
		cleanup()
		return "", nil, err
	}
	return tmpPath, cleanup, nil
}
