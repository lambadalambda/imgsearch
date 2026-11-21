package faces

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Full-pipeline test against real SCRFD + ArcFace models. Set:
// FACES_TEST_ORT_LIB to libonnxruntime.so and FACES_TEST_MODEL_DIR to a
// directory holding det_10g.onnx and w600k_r50.onnx.
func TestEngineDetectsRealFace(t *testing.T) {
	ortLib := os.Getenv("FACES_TEST_ORT_LIB")
	modelDir := os.Getenv("FACES_TEST_MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set FACES_TEST_ORT_LIB and FACES_TEST_MODEL_DIR")
	}
	engine := &Engine{Config: EngineConfig{
		ONNXRuntimeLib: ortLib,
		DetModelPath:   filepath.Join(modelDir, "det_10g.onnx"),
		RecModelPath:   filepath.Join(modelDir, "w600k_r50.onnx"),
	}}
	t.Cleanup(func() { _ = engine.Close() })

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve caller location")
	}
	img := filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", "woman.jpg")
	dets, err := engine.DetectFaces(t.Context(), img)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("faces: %d", len(dets))
	if len(dets) == 0 {
		t.Fatal("expected at least one face in woman.jpg")
	}
	for _, d := range dets {
		if d.BoxX < 0 || d.BoxY < 0 || d.BoxX+d.BoxW > 1 || d.BoxY+d.BoxH > 1 {
			t.Fatalf("box out of range: %+v", d)
		}
		if d.Confidence <= 0 || d.Confidence > 1 {
			t.Fatalf("confidence out of range: %v", d.Confidence)
		}
		if len(d.Embedding) != 512 {
			t.Fatalf("embedding dim = %d, want 512", len(d.Embedding))
		}
		var n float64
		for _, x := range d.Embedding {
			n += float64(x) * float64(x)
		}
		if math.Abs(n-1) > 1e-4 {
			t.Fatalf("embedding not normalized: norm^2 = %v", n)
		}
	}
}

func TestEngineFindsNoFacesInCatPhoto(t *testing.T) {
	ortLib := os.Getenv("FACES_TEST_ORT_LIB")
	modelDir := os.Getenv("FACES_TEST_MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set FACES_TEST_ORT_LIB and FACES_TEST_MODEL_DIR")
	}
	engine := &Engine{Config: EngineConfig{
		ONNXRuntimeLib: ortLib,
		DetModelPath:   filepath.Join(modelDir, "det_10g.onnx"),
		RecModelPath:   filepath.Join(modelDir, "w600k_r50.onnx"),
	}}
	t.Cleanup(func() { _ = engine.Close() })

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve caller location")
	}
	img := filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", "cat_1.jpg")
	dets, err := engine.DetectFaces(t.Context(), img)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("faces in cat photo: %d", len(dets))
}
