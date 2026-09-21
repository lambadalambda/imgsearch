//go:build cgo

package llamacppnative

import (
	"context"
	"os"
)

// PrepareImageJPEG resizes the image at path so its longest side is at most
// maxSide and returns JPEG bytes, using the same libvips pipeline as the
// native annotator. It satisfies openaicompat.ImagePreparer.
func PrepareImageJPEG(ctx context.Context, path string, maxSide int) ([]byte, string, error) {
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	tmpPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(path, maxSide)
	if err != nil {
		return nil, "", err
	}
	defer cleanup()
	data, err := os.ReadFile(tmpPath)
	if err != nil {
		return nil, "", err
	}
	return data, "image/jpeg", nil
}
