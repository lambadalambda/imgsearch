//go:build cgo

package llamacppnative

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

// #cgo pkg-config: vips
// #include <stdlib.h>
// #include <vips/vips.h>
//
// static int imgsearch_vips_thumbnail(const char* filename, VipsImage** out, int max_side) {
// 	return vips_thumbnail(filename, out, max_side, "size", VIPS_SIZE_DOWN, NULL);
// }
//
// static int imgsearch_vips_jpegsave(VipsImage* image, const char* filename, int quality) {
// 	return vips_jpegsave(image, filename, "Q", quality, NULL);
// }
//
// static void imgsearch_vips_unref(VipsImage* image) {
// 	if (image != NULL) {
// 		g_object_unref(image);
// 	}
// }
import "C"

var (
	vipsInitOnce sync.Once
	vipsInitErr  error
)

func ensureVipsInitialized() error {
	vipsInitOnce.Do(func() {
		name := C.CString("imgsearch")
		defer C.free(unsafe.Pointer(name))
		if C.vips_init(name) != 0 {
			vipsInitErr = fmt.Errorf("vips init failed: %s", currentVipsError())
			C.vips_error_clear()
		}
	})
	return vipsInitErr
}

func currentVipsError() string {
	errText := strings.TrimSpace(C.GoString(C.vips_error_buffer()))
	if errText == "" {
		return "unknown error"
	}
	return errText
}

func preprocessImageForEmbeddingWithVipsgen(sourcePath string, maxSide int) (string, func(), error) {
	sourcePath = strings.TrimSpace(sourcePath)
	if sourcePath == "" {
		return "", nil, fmt.Errorf("image path is empty")
	}
	if maxSide <= 0 {
		return "", nil, fmt.Errorf("llama-cpp-native image max side must be positive")
	}
	if err := ensureVipsInitialized(); err != nil {
		return "", nil, err
	}

	cSourcePath := C.CString(sourcePath)
	defer C.free(unsafe.Pointer(cSourcePath))
	var image *C.VipsImage
	if C.imgsearch_vips_thumbnail(cSourcePath, &image, C.int(maxSide)) != 0 {
		errText := currentVipsError()
		C.vips_error_clear()
		return "", nil, fmt.Errorf("vips thumbnail failed: %s", errText)
	}
	defer C.imgsearch_vips_unref(image)

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

	cTmpPath := C.CString(tmpPath)
	defer C.free(unsafe.Pointer(cTmpPath))
	if C.imgsearch_vips_jpegsave(image, cTmpPath, 90) != 0 {
		errText := currentVipsError()
		C.vips_error_clear()
		cleanup()
		return "", nil, fmt.Errorf("vips jpeg save failed: %s", errText)
	}

	return tmpPath, cleanup, nil
}
