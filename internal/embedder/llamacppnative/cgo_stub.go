//go:build !llamacpp_native && cgo

package llamacppnative

/*
#cgo CXXFLAGS: -std=c++17 -DIMGSEARCH_LLAMA_NATIVE_DISABLED
#include "bridge.h"
*/
import "C"
