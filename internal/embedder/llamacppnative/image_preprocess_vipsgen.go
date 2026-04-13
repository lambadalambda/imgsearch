//go:build llamacpp_native && cgo

package llamacppnative

import (
	"fmt"
	"os"
	"strings"

	vips "github.com/cshum/vipsgen/vips"
)

func preprocessImageForEmbeddingWithVipsgen(sourcePath string, maxSide int) (string, func(), error) {
	sourcePath = strings.TrimSpace(sourcePath)
	if sourcePath == "" {
		return "", nil, fmt.Errorf("image path is empty")
	}
	if maxSide <= 0 {
		return "", nil, fmt.Errorf("llama-cpp-native image max side must be positive")
	}

	image, err := vips.NewThumbnail(sourcePath, maxSide, &vips.ThumbnailOptions{Size: vips.SizeDown})
	if err != nil {
		return "", nil, fmt.Errorf("vips thumbnail failed: %w", err)
	}
	defer image.Close()

	tmp, err := os.CreateTemp("", "imgsearch-llama-native-*.jpg")
	if err != nil {
		return "", nil, fmt.Errorf("create temporary jpeg: %w", err)
	}
	tmpPath := tmp.Name()
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return "", nil, fmt.Errorf("close temporary jpeg: %w", err)
	}

	cleanup := func() {
		_ = os.Remove(tmpPath)
	}

	jpegOptions := vips.DefaultJpegsaveOptions()
	jpegOptions.Q = 90
	if err := image.Jpegsave(tmpPath, jpegOptions); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("vips jpeg save failed: %w", err)
	}

	return tmpPath, cleanup, nil
}
