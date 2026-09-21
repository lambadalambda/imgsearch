package llamacppnative

import "errors"

// ErrImagePrepareUnavailable is returned by PrepareImageJPEG when the build
// has no libvips support. Any other error means the image itself failed to
// decode or resize.
var ErrImagePrepareUnavailable = errors.New("llama-cpp-native image preparation requires cgo")
