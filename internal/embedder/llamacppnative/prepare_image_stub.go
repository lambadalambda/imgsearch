//go:build !cgo

package llamacppnative

import "context"

// PrepareImageJPEG is unavailable without cgo; callers should fall back to
// sending the stored file unchanged.
func PrepareImageJPEG(context.Context, string, int) ([]byte, string, error) {
	return nil, "", ErrImagePrepareUnavailable
}
