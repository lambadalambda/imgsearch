package parakeetonnx

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type EncoderConfig struct {
	ONNXRuntimeLib string
	EncoderPath    string
	UseCoreML      bool
}

type EncoderResult struct {
	HiddenSize    int64
	Frames        int64
	EncodedLength int64
	OutputShape   ort.Shape
	LengthsShape  ort.Shape
}

var ortInitMu sync.Mutex

func RunEncoder(cfg EncoderConfig, features Features) (EncoderResult, error) {
	if cfg.ONNXRuntimeLib == "" {
		return EncoderResult{}, fmt.Errorf("onnxruntime library path is required")
	}
	if cfg.EncoderPath == "" {
		return EncoderResult{}, fmt.Errorf("encoder model path is required")
	}
	if len(features.Values) != FeatureBins {
		return EncoderResult{}, fmt.Errorf("expected %d feature bins, got %d", FeatureBins, len(features.Values))
	}
	if len(features.Values[0]) == 0 {
		return EncoderResult{}, fmt.Errorf("feature frames are empty")
	}

	ortInitMu.Lock()
	defer ortInitMu.Unlock()
	ort.SetSharedLibraryPath(cfg.ONNXRuntimeLib)
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return EncoderResult{}, fmt.Errorf("initialize onnxruntime: %w", err)
		}
		defer func() { _ = ort.DestroyEnvironment() }()
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return EncoderResult{}, fmt.Errorf("new session options: %w", err)
	}
	defer options.Destroy()
	if cfg.UseCoreML {
		if err := options.AppendExecutionProviderCoreMLV2(map[string]string{}); err != nil {
			return EncoderResult{}, fmt.Errorf("append CoreML provider: %w", err)
		}
	}

	session, err := ort.NewDynamicAdvancedSession(
		cfg.EncoderPath,
		[]string{"audio_signal", "length"},
		[]string{"outputs", "encoded_lengths"},
		options,
	)
	if err != nil {
		return EncoderResult{}, fmt.Errorf("new dynamic encoder session: %w", err)
	}
	defer session.Destroy()

	frameCount := len(features.Values[0])
	inputData := make([]float32, FeatureBins*frameCount)
	for mel := 0; mel < FeatureBins; mel++ {
		copy(inputData[mel*frameCount:(mel+1)*frameCount], features.Values[mel])
	}
	inputTensor, err := ort.NewTensor(ort.NewShape(1, FeatureBins, int64(frameCount)), inputData)
	if err != nil {
		return EncoderResult{}, fmt.Errorf("create encoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()
	lengthTensor, err := ort.NewTensor(ort.NewShape(1), []int64{features.Length})
	if err != nil {
		return EncoderResult{}, fmt.Errorf("create encoder length tensor: %w", err)
	}
	defer lengthTensor.Destroy()

	outputs := []ort.Value{nil, nil}
	if err := session.Run([]ort.Value{inputTensor, lengthTensor}, outputs); err != nil {
		return EncoderResult{}, fmt.Errorf("run encoder session: %w", err)
	}
	for _, output := range outputs {
		if output != nil {
			defer output.Destroy()
		}
	}

	encoded, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return EncoderResult{}, fmt.Errorf("unexpected encoder output type %T", outputs[0])
	}
	lengths, ok := outputs[1].(*ort.Tensor[int64])
	if !ok {
		return EncoderResult{}, fmt.Errorf("unexpected encoded_lengths output type %T", outputs[1])
	}
	if len(lengths.GetData()) != 1 {
		return EncoderResult{}, fmt.Errorf("unexpected encoded_lengths size %d", len(lengths.GetData()))
	}
	shape := encoded.GetShape()
	if len(shape) != 3 {
		return EncoderResult{}, fmt.Errorf("unexpected encoder output shape %v", shape)
	}
	return EncoderResult{
		HiddenSize:    shape[1],
		Frames:        shape[2],
		EncodedLength: lengths.GetData()[0],
		OutputShape:   shape,
		LengthsShape:  lengths.GetShape(),
	}, nil
}
