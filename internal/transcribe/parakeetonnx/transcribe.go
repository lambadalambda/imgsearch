package parakeetonnx

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const windowStepSeconds = 0.01

var decodeSpacePattern = regexp.MustCompile(`^\s|\s\B|(\s)\b`)

type RecognizerConfig struct {
	ONNXRuntimeLib string
	BundleDir      string
	UseCoreML      bool
}

type Transcript struct {
	Text          string
	TokenIDs      []int
	Tokens        []string
	TimestampsSec []float64
}

func TranscribeVideo(ctx context.Context, cfg RecognizerConfig, videoPath string) (Transcript, error) {
	samples, err := ExtractPCM16kMono(ctx, videoPath)
	if err != nil {
		return Transcript{}, err
	}
	features, err := BuildFeatures(samples)
	if err != nil {
		return Transcript{}, err
	}
	return TranscribeFeatures(cfg, features)
}

func TranscribeFeatures(cfg RecognizerConfig, features Features) (Transcript, error) {
	if cfg.ONNXRuntimeLib == "" {
		return Transcript{}, fmt.Errorf("onnxruntime library path is required")
	}
	if cfg.BundleDir == "" {
		return Transcript{}, fmt.Errorf("bundle directory is required")
	}

	vocab, blankIdx, err := loadVocab(filepath.Join(cfg.BundleDir, "vocab.txt"))
	if err != nil {
		return Transcript{}, err
	}
	maxTokensPerStep, err := loadMaxTokensPerStep(filepath.Join(cfg.BundleDir, "config.json"))
	if err != nil {
		return Transcript{}, err
	}

	ortInitMu.Lock()
	defer ortInitMu.Unlock()
	ort.SetSharedLibraryPath(cfg.ONNXRuntimeLib)
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return Transcript{}, fmt.Errorf("initialize onnxruntime: %w", err)
		}
		defer func() { _ = ort.DestroyEnvironment() }()
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return Transcript{}, fmt.Errorf("new session options: %w", err)
	}
	defer options.Destroy()
	if cfg.UseCoreML {
		if err := options.AppendExecutionProviderCoreMLV2(map[string]string{}); err != nil {
			return Transcript{}, fmt.Errorf("append CoreML provider: %w", err)
		}
	}

	encoderSession, err := ort.NewDynamicAdvancedSession(
		filepath.Join(cfg.BundleDir, "encoder-model.int8.onnx"),
		[]string{"audio_signal", "length"},
		[]string{"outputs", "encoded_lengths"},
		options,
	)
	if err != nil {
		return Transcript{}, fmt.Errorf("create encoder session: %w", err)
	}
	defer encoderSession.Destroy()

	decoderSession, err := ort.NewDynamicAdvancedSession(
		filepath.Join(cfg.BundleDir, "decoder_joint-model.int8.onnx"),
		[]string{"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"},
		[]string{"outputs", "prednet_lengths", "output_states_1", "output_states_2"},
		options,
	)
	if err != nil {
		return Transcript{}, fmt.Errorf("create decoder session: %w", err)
	}
	defer decoderSession.Destroy()

	encodedFrames, encodedLen, err := runEncoderSession(encoderSession, features)
	if err != nil {
		return Transcript{}, err
	}
	if encodedLen <= 0 {
		return Transcript{}, fmt.Errorf("encoder returned non-positive encoded length %d", encodedLen)
	}

	state1, state2, err := makeInitialDecoderState(filepath.Join(cfg.BundleDir, "decoder_joint-model.int8.onnx"), options)
	if err != nil {
		return Transcript{}, err
	}

	tokens := make([]int, 0, encodedLen)
	timestamps := make([]float64, 0, encodedLen)
	t := 0
	emitted := 0
	for t < encodedLen {
		logits, step, next1, next2, err := decodeStep(decoderSession, encodedFrames[t], tokens, blankIdx, len(vocab), state1, state2)
		if err != nil {
			return Transcript{}, err
		}
		token := argmax(logits[:len(vocab)])
		if token != blankIdx {
			state1, state2 = next1, next2
			tokens = append(tokens, token)
			timestamps = append(timestamps, windowStepSeconds*8*float64(t))
			emitted++
		}

		if step > 0 {
			t += step
			emitted = 0
		} else if token == blankIdx || emitted == maxTokensPerStep {
			t++
			emitted = 0
		}
	}

	decodedTokens := make([]string, 0, len(tokens))
	for _, id := range tokens {
		if id < 0 || id >= len(vocab) {
			continue
		}
		decodedTokens = append(decodedTokens, vocab[id])
	}
	text := decodeTokens(decodedTokens)
	return Transcript{
		Text:          text,
		TokenIDs:      tokens,
		Tokens:        decodedTokens,
		TimestampsSec: timestamps,
	}, nil
}

func runEncoderSession(session *ort.DynamicAdvancedSession, features Features) ([][]float32, int, error) {
	frameCount := len(features.Values[0])
	inputData := make([]float32, FeatureBins*frameCount)
	for mel := 0; mel < FeatureBins; mel++ {
		copy(inputData[mel*frameCount:(mel+1)*frameCount], features.Values[mel])
	}
	inputTensor, err := ort.NewTensor(ort.NewShape(1, FeatureBins, int64(frameCount)), inputData)
	if err != nil {
		return nil, 0, fmt.Errorf("create encoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()
	lengthTensor, err := ort.NewTensor(ort.NewShape(1), []int64{features.Length})
	if err != nil {
		return nil, 0, fmt.Errorf("create encoder length tensor: %w", err)
	}
	defer lengthTensor.Destroy()

	outputs := []ort.Value{nil, nil}
	if err := session.Run([]ort.Value{inputTensor, lengthTensor}, outputs); err != nil {
		return nil, 0, fmt.Errorf("run encoder session: %w", err)
	}
	defer destroyValues(outputs...)

	encoded, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, fmt.Errorf("unexpected encoder output type %T", outputs[0])
	}
	lengths, ok := outputs[1].(*ort.Tensor[int64])
	if !ok {
		return nil, 0, fmt.Errorf("unexpected encoded_lengths output type %T", outputs[1])
	}
	shape := encoded.GetShape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1024 {
		return nil, 0, fmt.Errorf("unexpected encoder output shape %v", shape)
	}
	if len(lengths.GetData()) != 1 {
		return nil, 0, fmt.Errorf("unexpected encoded_lengths size %d", len(lengths.GetData()))
	}
	encodedLen := int(lengths.GetData()[0])
	if encodedLen > int(shape[2]) {
		encodedLen = int(shape[2])
	}
	data := encoded.GetData()
	frames := make([][]float32, encodedLen)
	for t := 0; t < encodedLen; t++ {
		frame := make([]float32, shape[1])
		for h := 0; h < int(shape[1]); h++ {
			frame[h] = data[h*int(shape[2])+t]
		}
		frames[t] = frame
	}
	return frames, encodedLen, nil
}

func makeInitialDecoderState(decoderPath string, options *ort.SessionOptions) ([]float32, []float32, error) {
	inputs, _, err := ort.GetInputOutputInfoWithOptions(decoderPath, options)
	if err != nil {
		return nil, nil, fmt.Errorf("inspect decoder inputs: %w", err)
	}
	var shape1, shape2 ort.Shape
	for _, input := range inputs {
		switch input.Name {
		case "input_states_1":
			shape1 = input.Dimensions
		case "input_states_2":
			shape2 = input.Dimensions
		}
	}
	if len(shape1) != 3 || len(shape2) != 3 {
		return nil, nil, fmt.Errorf("decoder state input shapes not found")
	}
	state1 := make([]float32, int(shape1[0])*1*int(shape1[2]))
	state2 := make([]float32, int(shape2[0])*1*int(shape2[2]))
	return state1, state2, nil
}

func decodeStep(session *ort.DynamicAdvancedSession, frame []float32, prevTokens []int, blankIdx int, vocabSize int, state1 []float32, state2 []float32) ([]float32, int, []float32, []float32, error) {
	encoderInput, err := ort.NewTensor(ort.NewShape(1, int64(len(frame)), 1), frame)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("create decoder encoder_outputs tensor: %w", err)
	}
	defer encoderInput.Destroy()
	lastToken := blankIdx
	if len(prevTokens) > 0 {
		lastToken = prevTokens[len(prevTokens)-1]
	}
	targets, err := ort.NewTensor(ort.NewShape(1, 1), []int32{int32(lastToken)})
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("create decoder targets tensor: %w", err)
	}
	defer targets.Destroy()
	targetLength, err := ort.NewTensor(ort.NewShape(1), []int32{1})
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("create decoder target_length tensor: %w", err)
	}
	defer targetLength.Destroy()
	stateTensor1, err := ort.NewTensor(ort.NewShape(2, 1, 640), state1)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("create decoder state1 tensor: %w", err)
	}
	defer stateTensor1.Destroy()
	stateTensor2, err := ort.NewTensor(ort.NewShape(2, 1, 640), state2)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("create decoder state2 tensor: %w", err)
	}
	defer stateTensor2.Destroy()

	outputs := []ort.Value{nil, nil, nil, nil}
	if err := session.Run([]ort.Value{encoderInput, targets, targetLength, stateTensor1, stateTensor2}, outputs); err != nil {
		return nil, 0, nil, nil, fmt.Errorf("run decoder session: %w", err)
	}
	defer destroyValues(outputs...)

	logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, nil, nil, fmt.Errorf("unexpected decoder outputs type %T", outputs[0])
	}
	stateOut1, ok := outputs[2].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, nil, nil, fmt.Errorf("unexpected decoder output_states_1 type %T", outputs[2])
	}
	stateOut2, ok := outputs[3].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, nil, nil, fmt.Errorf("unexpected decoder output_states_2 type %T", outputs[3])
	}

	logitsRaw := logitsTensor.GetData()
	logits := make([]float32, len(logitsRaw))
	copy(logits, logitsRaw)
	next1 := make([]float32, len(stateOut1.GetData()))
	copy(next1, stateOut1.GetData())
	next2 := make([]float32, len(stateOut2.GetData()))
	copy(next2, stateOut2.GetData())
	return logits, tdtStep(logits, vocabSize), next1, next2, nil
}

func tdtStep(logits []float32, vocabSize int) int {
	if vocabSize >= len(logits) {
		return 0
	}
	return argmax(logits[vocabSize:])
}

func argmax(values []float32) int {
	best := 0
	bestValue := float32(math.Inf(-1))
	for i, value := range values {
		if i == 0 || value > bestValue {
			best = i
			bestValue = value
		}
	}
	return best
}

func destroyValues(values ...ort.Value) {
	for _, value := range values {
		if value != nil {
			_ = value.Destroy()
		}
	}
}

func loadVocab(path string) ([]string, int, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("read vocab: %w", err)
	}
	type pair struct {
		id    int
		token string
	}
	entries := make([]pair, 0, 8192)
	blankIdx := -1
	maxID := -1
	for _, line := range strings.Split(strings.TrimSpace(string(content)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		idx := strings.LastIndexByte(line, ' ')
		if idx < 0 {
			return nil, 0, fmt.Errorf("invalid vocab line %q", line)
		}
		token := strings.ReplaceAll(line[:idx], "\u2581", " ")
		id, err := strconv.Atoi(strings.TrimSpace(line[idx+1:]))
		if err != nil {
			return nil, 0, fmt.Errorf("parse vocab id from %q: %w", line, err)
		}
		entries = append(entries, pair{id: id, token: token})
		if token == "<blk>" {
			blankIdx = id
		}
		if id > maxID {
			maxID = id
		}
	}
	if blankIdx < 0 {
		return nil, 0, fmt.Errorf("blank token not found in vocab")
	}
	vocab := make([]string, maxID+1)
	for _, entry := range entries {
		vocab[entry.id] = entry.token
	}
	return vocab, blankIdx, nil
}

func loadMaxTokensPerStep(configPath string) (int, error) {
	content, err := os.ReadFile(configPath)
	if err != nil {
		return 0, fmt.Errorf("read config: %w", err)
	}
	var parsed struct {
		MaxTokensPerStep int `json:"max_tokens_per_step"`
	}
	if err := json.Unmarshal(content, &parsed); err != nil {
		return 0, fmt.Errorf("decode config: %w", err)
	}
	if parsed.MaxTokensPerStep <= 0 {
		return 10, nil
	}
	return parsed.MaxTokensPerStep, nil
}

func decodeTokens(tokens []string) string {
	joined := strings.Join(tokens, "")
	return decodeSpacePattern.ReplaceAllString(joined, "$1")
}
