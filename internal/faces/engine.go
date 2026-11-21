package faces

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync"

	"imgsearch/internal/people"

	ort "github.com/yalue/onnxruntime_go"
)

// EngineConfig names the ONNX Runtime library and the two face models.
type EngineConfig struct {
	ONNXRuntimeLib string
	DetModelPath   string
	RecModelPath   string
	UseCoreML      bool
}

// Engine is a people.Detector backed by SCRFD detection and an ArcFace-class
// recognition model. Sessions start on first use; methods are safe for
// concurrent callers.
type Engine struct {
	Config EngineConfig

	mu      sync.Mutex
	ready   bool
	initErr error

	detSession *ort.DynamicAdvancedSession
	recSession *ort.DynamicAdvancedSession
	detInput   string
	detOutputs []string
	// detHeads holds, per stride in DetStrides order, the output indexes of
	// the score, bbox and keypoint tensors plus their anchor counts.
	detHeads  [3]detHead
	recInput  string
	recOutput string
	recDim    int
}

type detHead struct {
	score, bbox, kps int
	anchors          int
}

var facesOrtMu sync.Mutex

// Close releases model sessions. A zero Engine is a no-op.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	var firstErr error
	if e.detSession != nil {
		if err := e.detSession.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
		e.detSession = nil
	}
	if e.recSession != nil {
		if err := e.recSession.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
		e.recSession = nil
	}
	e.ready = false
	e.initErr = nil
	return firstErr
}

func (e *Engine) ensureReadyLocked() error {
	if e.ready {
		return e.initErr
	}
	e.ready = true
	e.initErr = e.initLocked()
	return e.initErr
}

func (e *Engine) initLocked() error {
	if e.Config.ONNXRuntimeLib == "" {
		return fmt.Errorf("onnxruntime library path is required")
	}
	for _, path := range []string{e.Config.DetModelPath, e.Config.RecModelPath} {
		if path == "" {
			return fmt.Errorf("face model path is required")
		}
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("stat face model %s: %w", path, err)
		}
	}

	facesOrtMu.Lock()
	defer facesOrtMu.Unlock()
	if !ort.IsInitialized() {
		ort.SetSharedLibraryPath(e.Config.ONNXRuntimeLib)
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("initialize onnxruntime: %w", err)
		}
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("new session options: %w", err)
	}
	defer options.Destroy()
	if e.Config.UseCoreML {
		if err := options.AppendExecutionProviderCoreMLV2(map[string]string{}); err != nil {
			return fmt.Errorf("append CoreML provider: %w", err)
		}
	}

	if err := e.initDetectorLocked(options); err != nil {
		return err
	}
	return e.initRecognizerLocked(options)
}

// discoverIONames returns the single float input name and all output names.
func discoverIONames(modelPath string) (string, []string, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return "", nil, fmt.Errorf("inspect model %s: %w", modelPath, err)
	}
	if len(inputs) != 1 {
		return "", nil, fmt.Errorf("model %s: want 1 input, have %d", modelPath, len(inputs))
	}
	names := make([]string, 0, len(outputs))
	for _, o := range outputs {
		names = append(names, o.Name)
	}
	return inputs[0].Name, names, nil
}

func (e *Engine) initDetectorLocked(options *ort.SessionOptions) error {
	inName, outNames, err := discoverIONames(e.Config.DetModelPath)
	if err != nil {
		return err
	}
	if len(outNames) != 9 {
		return fmt.Errorf("detection model %s: want 9 outputs (score/bbox/kps at 3 strides), have %d", e.Config.DetModelPath, len(outNames))
	}
	session, err := ort.NewDynamicAdvancedSession(e.Config.DetModelPath, []string{inName}, outNames, options)
	if err != nil {
		return fmt.Errorf("create detection session: %w", err)
	}
	e.detSession = session
	e.detInput = inName
	e.detOutputs = outNames
	// Probe with zeros to learn concrete output shapes, then pair
	// score/bbox/kps heads by anchor count instead of trusting layout.
	inTensor, err := ort.NewTensor(ort.NewShape(1, 3, DetInputSize, DetInputSize), make([]float32, 3*DetInputSize*DetInputSize))
	if err != nil {
		_ = session.Destroy()
		e.detSession = nil
		return fmt.Errorf("probe input tensor: %w", err)
	}
	defer inTensor.Destroy()
	probe, err := e.runDetectorLocked(inTensor, outNames, nil)
	if err != nil {
		_ = session.Destroy()
		e.detSession = nil
		return fmt.Errorf("probe detection model: %w", err)
	}
	heads, err := pairDetHeads(outNames, probe.shapes)
	if err != nil {
		_ = session.Destroy()
		e.detSession = nil
		return err
	}
	e.detHeads = heads
	return nil
}

// detProbeResult carries one tensor per graph output in graph order.
type detProbeResult struct {
	data   [][]float32
	shapes []ort.Shape
}

func (e *Engine) runDetectorLocked(inTensor *ort.Tensor[float32], outNames []string, shapes []ort.Shape) (detProbeResult, error) {
	if shapes == nil {
		shapes = defaultDetShapes()
	}
	var outTensors []*ort.Tensor[float32]
	release := func() {
		for _, t := range outTensors {
			_ = t.Destroy()
		}
	}
	for _, sh := range shapes {
		t, err := ort.NewEmptyTensor[float32](sh)
		if err != nil {
			release()
			return detProbeResult{}, err
		}
		outTensors = append(outTensors, t)
	}
	inputs := []ort.Value{inTensor}
	outputs := make([]ort.Value, len(outTensors))
	for i, t := range outTensors {
		outputs[i] = t
	}
	if err := e.detSession.Run(inputs, outputs); err != nil {
		release()
		return detProbeResult{}, err
	}
	res := detProbeResult{shapes: make([]ort.Shape, len(outTensors)), data: make([][]float32, len(outTensors))}
	for i, t := range outTensors {
		res.shapes[i] = t.GetShape()
		res.data[i] = append([]float32(nil), t.GetData()...)
	}
	release()
	return res, nil
}

// defaultDetShapes sizes the 9 SCRFD-10G outputs for a 640 input in graph
// order: 3 score, 3 bbox, 3 keypoint heads at strides 8/16/32.
func defaultDetShapes() []ort.Shape {
	var shapes []ort.Shape
	for _, kind := range []int64{1, 4, 10} {
		for _, stride := range DetStrides {
			n := int64(DetInputSize/stride) * int64(DetInputSize/stride) * DetAnchorCount
			shapes = append(shapes, ort.NewShape(n, kind))
		}
	}
	return shapes
}

// pairDetHeads groups probe output shapes into per-stride heads by anchor
// count, independent of graph layout. Rank-3 (batched) outputs decode from
// the first batch element like the reference.
func pairDetHeads(names []string, shapes []ort.Shape) ([3]detHead, error) {
	var heads [3]detHead
	type slot struct {
		index int
		count int64
	}
	byKind := map[int64][]slot{}
	for i, sh := range shapes {
		dims := []int64(sh)
		var count, kind int64
		switch len(dims) {
		case 2:
			count, kind = dims[0], dims[1]
		case 3:
			// Batched output: first dim is the batch; the reference
			// decodes batch element zero, whose shape is dims[1:].
			if dims[0] != 1 {
				return heads, fmt.Errorf("detection output %q: batch dim %d, want 1", names[i], dims[0])
			}
			count, kind = dims[1], dims[2]
		default:
			return heads, fmt.Errorf("detection output %q: unexpected rank %d", names[i], len(dims))
		}
		byKind[kind] = append(byKind[kind], slot{i, count})
	}
	for kind, want := range map[int64]int{1: 3, 4: 3, 10: 3} {
		if len(byKind[int64(kind)]) != want {
			return heads, fmt.Errorf("detection model: want %d outputs of width %d, have %d", want, kind, len(byKind[int64(kind)]))
		}
	}
	// Anchor counts identify strides: 12800/3200/800 for input 640.
	for s, stride := range DetStrides {
		want := int64(DetInputSize/stride) * int64(DetInputSize/stride) * DetAnchorCount
		for _, slot := range byKind[1] {
			if slot.count != want {
				continue
			}
			heads[s].score = slot.index
			heads[s].anchors = int(want)
		}
		for _, slot := range byKind[4] {
			if slot.count == want {
				heads[s].bbox = slot.index
			}
		}
		for _, slot := range byKind[10] {
			if slot.count == want {
				heads[s].kps = slot.index
			}
		}
		if heads[s].anchors == 0 {
			return heads, fmt.Errorf("detection model: no head with %d anchors (stride %d)", want, stride)
		}
	}
	return heads, nil
}

func (e *Engine) initRecognizerLocked(options *ort.SessionOptions) error {
	inName, outNames, err := discoverIONames(e.Config.RecModelPath)
	if err != nil {
		return err
	}
	if len(outNames) != 1 {
		return fmt.Errorf("recognition model %s: want 1 output, have %d", e.Config.RecModelPath, len(outNames))
	}
	session, err := ort.NewDynamicAdvancedSession(e.Config.RecModelPath, []string{inName}, outNames, options)
	if err != nil {
		return fmt.Errorf("create recognition session: %w", err)
	}
	e.recSession = session
	e.recInput = inName
	e.recOutput = outNames[0]
	// Learn the embedding width with a zero probe instead of trusting names.
	blob := make([]float32, 3*RecInputSize*RecInputSize)
	inTensor, err := ort.NewTensor(ort.NewShape(1, 3, RecInputSize, RecInputSize), blob)
	if err != nil {
		return err
	}
	defer inTensor.Destroy()
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 512))
	if err != nil {
		return err
	}
	defer outTensor.Destroy()
	if err := session.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		_ = session.Destroy()
		e.recSession = nil
		return fmt.Errorf("probe recognition model: %w", err)
	}
	dims := []int64(outTensor.GetShape())
	if len(dims) != 2 || dims[1] < 64 {
		_ = session.Destroy()
		e.recSession = nil
		return fmt.Errorf("recognition model: unexpected embedding shape %v", dims)
	}
	e.recDim = int(dims[1])
	return nil
}

func (e *Engine) runRecognizerLocked(blob []float32) ([]float32, error) {
	inTensor, err := ort.NewTensor(ort.NewShape(1, 3, RecInputSize, RecInputSize), blob)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(e.recDim)))
	if err != nil {
		return nil, err
	}
	defer outTensor.Destroy()
	if err := e.recSession.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return nil, err
	}
	return append([]float32(nil), outTensor.GetData()...), nil
}

// DetectFaces implements people.Detector: it finds faces with landmarks,
// aligns each one, and embeds it. Files with no supported decoder return
// ErrUnsupportedFormat so the worker can skip them without failing.
func (e *Engine) DetectFaces(ctx context.Context, imagePath string) ([]people.Detection, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	content, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, err
	}
	img, err := DecodeImage(content)
	if err != nil {
		return nil, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.ensureReadyLocked(); err != nil {
		return nil, err
	}

	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	canvas, scale := Letterbox(img, DetInputSize)
	blob := DetBlob(canvas)
	inTensor, err := ort.NewTensor(ort.NewShape(1, 3, DetInputSize, DetInputSize), blob)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()
	probe, err := e.runDetectorLocked(inTensor, e.detOutputs, nil)
	if err != nil {
		return nil, fmt.Errorf("run detection: %w", err)
	}
	var raw []RawDetection
	for s := range DetStrides {
		h := e.detHeads[s]
		raw = append(raw, DecodeDetections(probe.data[h.score], probe.data[h.bbox], probe.data[h.kps], DetInputSize, DetStrides[s], DetThreshold)...)
	}
	kept := NMS(raw, DetNMSThreshold)
	out := make([]people.Detection, 0, len(kept))
	for _, d := range kept {
		d = d.ToOriginal(scale, origW, origH)
		if d.X2 <= d.X1 || d.Y2 <= d.Y1 {
			continue
		}
		aligned, err := AlignFace(img, d.Landmarks)
		if err != nil {
			return nil, err
		}
		vec, err := e.runRecognizerLocked(aligned)
		if err != nil {
			return nil, fmt.Errorf("run recognition: %w", err)
		}
		out = append(out, people.Detection{
			BoxX: d.X1 / float64(origW), BoxY: d.Y1 / float64(origH),
			BoxW: (d.X2 - d.X1) / float64(origW), BoxH: (d.Y2 - d.Y1) / float64(origH),
			Confidence: d.Score, Embedding: normalizeVec(vec),
		})
	}
	return out, nil
}

func normalizeVec(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return v
	}
	s := float32(1 / math.Sqrt(n))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * s
	}
	return out
}
