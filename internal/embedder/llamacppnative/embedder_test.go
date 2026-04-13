package llamacppnative

import "testing"

func TestNewRequiresNativeBuildTag(t *testing.T) {
	_, err := New(Config{
		ModelPath:       "/tmp/model.gguf",
		VisionModelPath: "/tmp/mmproj.gguf",
		Dimensions:      2048,
	})
	if err == nil {
		t.Fatal("expected error without native build tag/runtime")
	}
}
