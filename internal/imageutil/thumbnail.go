//go:build cgo

// Package imageutil provides the shared libvips resize pipeline for inference
// JPEGs and persistent UI thumbnails. Neither operation modifies the source.
package imageutil

// #cgo pkg-config: vips
// #include <stdlib.h>
// #include <vips/vips.h>
//
// static int resize_image(const char* source, const char* destination, int max_side, int webp) {
//     VipsImage* image = NULL;
//     if (vips_thumbnail(source, &image, max_side, "size", VIPS_SIZE_DOWN, NULL) != 0) return -1;
//     int result;
//     if (webp) {
//         result = vips_webpsave(image, destination, "Q", 80, "strip", TRUE, NULL);
//     } else {
//         result = vips_jpegsave(image, destination, "Q", 90, NULL);
//     }
//     g_object_unref(image);
//     return result;
// }
import "C"

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

var initOnce sync.Once
var initErr error

// ThumbnailJPEG writes a downscaled, auto-oriented JPEG for model inference.
func ThumbnailJPEG(source, destination string, maxSide int) error {
	return resize(source, destination, maxSide, false)
}

// ThumbnailWebP writes an auto-oriented, metadata-free WebP, retaining alpha.
// Small images are not upscaled; animated inputs use their first frame.
func ThumbnailWebP(source, destination string, maxSide int) error {
	return resize(source, destination, maxSide, true)
}

func resize(source, destination string, maxSide int, webp bool) error {
	if strings.TrimSpace(source) == "" || strings.TrimSpace(destination) == "" || maxSide <= 0 {
		return fmt.Errorf("image paths and positive max side are required")
	}
	// libvips error buffers are thread-local; read and clear on the calling thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	initOnce.Do(func() {
		name := C.CString("imgsearch")
		defer C.free(unsafe.Pointer(name))
		if C.vips_init(name) != 0 {
			initErr = vipsError("init")
		}
	})
	if initErr != nil {
		return initErr
	}
	src := C.CString(source)
	defer C.free(unsafe.Pointer(src))
	dst := C.CString(destination)
	defer C.free(unsafe.Pointer(dst))
	var format C.int
	if webp {
		format = 1
	}
	if C.resize_image(src, dst, C.int(maxSide), format) != 0 {
		return vipsError("resize")
	}
	return nil
}

func vipsError(operation string) error {
	text := strings.TrimSpace(C.GoString(C.vips_error_buffer()))
	C.vips_error_clear()
	if text == "" {
		text = "unknown error"
	}
	return fmt.Errorf("vips %s: %s", operation, text)
}
