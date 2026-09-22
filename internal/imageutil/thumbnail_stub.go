//go:build !cgo

package imageutil

import "fmt"

func ThumbnailJPEG(source, destination string, maxSide int) error {
	return fmt.Errorf("image resizing requires cgo and libvips")
}

func ThumbnailWebP(source, destination string, maxSide int) error {
	return fmt.Errorf("image resizing requires cgo and libvips")
}
