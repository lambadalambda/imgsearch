package sqliteai

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

const (
	defaultContextOptions     = "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=last"
	defaultImageMaxSide       = 512
	defaultQueryInstruction   = "Retrieve images or text relevant to the user's query."
	defaultPassageInstruction = "Represent this image or text for retrieval."
)

type commandRunner func(ctx context.Context, name string, args ...string) ([]byte, error)
type vipsResizeFunc func(sourcePath string, outputPath string, scale float64) []string

type Config struct {
	DB                 *sql.DB
	ModelPath          string
	ModelOptions       string
	VisionModelPath    string
	VisionModelOptions string
	ContextOptions     string
	QueryInstruction   string
	PassageInstruction string
	ImageMaxSide       int
	VipsPath           string
}

type Embedder struct {
	db                 *sql.DB
	modelPath          string
	modelOptions       string
	visionModelPath    string
	visionModelOptions string
	contextOptions     string
	queryInstruction   string
	passageInstruction string
	imageMaxSide       int
	vipsPath           string
	runCommand         commandRunner
	vipsResize         vipsResizeFunc

	initOnce sync.Once
	initErr  error
}

func New(cfg Config) (*Embedder, error) {
	if cfg.DB == nil {
		return nil, fmt.Errorf("sqlite-ai db is nil")
	}
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return nil, fmt.Errorf("sqlite-ai model path is required")
	}

	ctxOptions := strings.TrimSpace(cfg.ContextOptions)
	if ctxOptions == "" {
		ctxOptions = defaultContextOptions
	}

	queryInstruction := strings.TrimSpace(cfg.QueryInstruction)
	if queryInstruction == "" {
		queryInstruction = defaultQueryInstruction
	}
	if strings.Contains(queryInstruction, ",") {
		return nil, fmt.Errorf("sqlite-ai query instruction must not contain commas")
	}

	passageInstruction := strings.TrimSpace(cfg.PassageInstruction)
	if passageInstruction == "" {
		passageInstruction = defaultPassageInstruction
	}
	if strings.Contains(passageInstruction, ",") {
		return nil, fmt.Errorf("sqlite-ai passage instruction must not contain commas")
	}

	imageMaxSide := cfg.ImageMaxSide
	if imageMaxSide == 0 {
		imageMaxSide = defaultImageMaxSide
	}
	if imageMaxSide < 0 {
		return nil, fmt.Errorf("sqlite-ai image max side must be non-negative")
	}

	vipsPath := strings.TrimSpace(cfg.VipsPath)
	if vipsPath == "" {
		vipsPath = "vips"
	}
	resolvedVipsPath, err := exec.LookPath(vipsPath)
	if err != nil {
		return nil, fmt.Errorf("sqlite-ai image preprocessing requires vips CLI (install libvips and ensure %q is on PATH): %w", vipsPath, err)
	}

	return &Embedder{
		db:                 cfg.DB,
		modelPath:          strings.TrimSpace(cfg.ModelPath),
		modelOptions:       strings.TrimSpace(cfg.ModelOptions),
		visionModelPath:    strings.TrimSpace(cfg.VisionModelPath),
		visionModelOptions: strings.TrimSpace(cfg.VisionModelOptions),
		contextOptions:     ctxOptions,
		queryInstruction:   queryInstruction,
		passageInstruction: passageInstruction,
		imageMaxSide:       imageMaxSide,
		vipsPath:           resolvedVipsPath,
		runCommand:         runCommandCombinedOutput,
		vipsResize:         defaultVipsResize,
	}, nil
}

func (e *Embedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if err := e.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	query := `SELECT llm_embed_generate(?)`
	args := []any{text}
	if instructionOption := embedInstructionOption(e.queryInstruction); instructionOption != "" {
		query = `SELECT llm_embed_generate(?, ?)`
		args = append(args, instructionOption)
	}

	var blob []byte
	if err := e.db.QueryRowContext(ctx, query, args...).Scan(&blob); err != nil {
		return nil, fmt.Errorf("sqlite-ai text embedding failed: %w", err)
	}

	vec, err := decodeFloat32Blob(blob)
	if err != nil {
		return nil, fmt.Errorf("decode sqlite-ai text embedding: %w", err)
	}
	return vec, nil
}

func (e *Embedder) EmbedImage(ctx context.Context, path string) ([]float32, error) {
	if err := e.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	convertedPath, cleanup, err := e.prepareImageForEmbedding(ctx, path)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	query := `SELECT llm_embed_generate(?, ?)`
	args := []any{"", convertedPath}
	if instructionOption := embedInstructionOption(e.passageInstruction); instructionOption != "" {
		query = `SELECT llm_embed_generate(?, ?, ?)`
		args = append(args, instructionOption)
	}

	var blob []byte
	if err := e.db.QueryRowContext(ctx, query, args...).Scan(&blob); err != nil {
		return nil, fmt.Errorf("sqlite-ai image embedding failed: %w", err)
	}

	vec, err := decodeFloat32Blob(blob)
	if err != nil {
		return nil, fmt.Errorf("decode sqlite-ai image embedding: %w", err)
	}
	return vec, nil
}

func (e *Embedder) prepareImageForEmbedding(ctx context.Context, sourcePath string) (string, func(), error) {
	sourcePath = strings.TrimSpace(sourcePath)
	if sourcePath == "" {
		return "", nil, fmt.Errorf("sqlite-ai image path is empty")
	}

	maxSide := e.imageMaxSide
	if maxSide <= 0 {
		maxSide = defaultImageMaxSide
	}
	width, height, err := e.imageDimensionsWithVips(ctx, sourcePath)
	if err != nil {
		return "", nil, err
	}
	maxDimension := width
	if height > maxDimension {
		maxDimension = height
	}
	if maxDimension <= 0 {
		return "", nil, fmt.Errorf("invalid image dimensions for %q: %dx%d", sourcePath, width, height)
	}

	scale := 1.0
	if maxDimension > maxSide {
		scale = float64(maxSide) / float64(maxDimension)
	}

	tmpFile, err := os.CreateTemp("", "imgsearch-sqliteai-embed-*.jpg")
	if err != nil {
		return "", nil, fmt.Errorf("create temp image file: %w", err)
	}
	outputPath := tmpFile.Name()
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(outputPath)
		return "", nil, fmt.Errorf("close temp image file: %w", err)
	}
	if err := os.Remove(outputPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", nil, fmt.Errorf("prepare temp image file: %w", err)
	}

	runner := e.runCommand
	if runner == nil {
		runner = runCommandCombinedOutput
	}
	vipsCmd := e.vipsPath
	if vipsCmd == "" {
		vipsCmd = "vips"
	}
	resizeArgs := e.vipsResize
	if resizeArgs == nil {
		resizeArgs = defaultVipsResize
	}

	commandOutput, err := runner(ctx, vipsCmd, resizeArgs(sourcePath, outputPath, scale)...)
	if err != nil {
		_ = os.Remove(outputPath)
		stderr := strings.TrimSpace(string(commandOutput))
		if stderr != "" {
			return "", nil, fmt.Errorf("vips resize failed: %w: %s", err, stderr)
		}
		return "", nil, fmt.Errorf("vips resize failed: %w", err)
	}

	cleanup := func() {
		_ = os.Remove(outputPath)
	}
	return outputPath, cleanup, nil
}

func (e *Embedder) ensureInitialized(ctx context.Context) error {
	e.initOnce.Do(func() {
		e.initErr = e.initialize(ctx)
	})
	return e.initErr
}

func (e *Embedder) initialize(ctx context.Context) error {
	var version string
	if err := e.db.QueryRowContext(ctx, `SELECT ai_version()`).Scan(&version); err != nil {
		return fmt.Errorf("sqlite-ai extension not available: %w", err)
	}

	if err := scanNilResult(e.db.QueryRowContext(ctx, `SELECT llm_model_load(?, ?)`, e.modelPath, e.modelOptions)); err != nil {
		return fmt.Errorf("load sqlite-ai model: %w", err)
	}

	if e.visionModelPath != "" {
		if err := scanNilResult(e.db.QueryRowContext(ctx, `SELECT llm_vision_load(?, ?)`, e.visionModelPath, e.visionModelOptions)); err != nil {
			return fmt.Errorf("load sqlite-ai vision model: %w", err)
		}
	}

	if err := scanNilResult(e.db.QueryRowContext(ctx, `SELECT llm_context_create_embedding(?)`, e.contextOptions)); err != nil {
		return fmt.Errorf("create sqlite-ai embedding context: %w", err)
	}

	return nil
}

func scanNilResult(row *sql.Row) error {
	var out any
	if err := row.Scan(&out); err != nil {
		return err
	}
	return nil
}

func decodeFloat32Blob(blob []byte) ([]float32, error) {
	if len(blob) == 0 {
		return nil, fmt.Errorf("embedding blob is empty")
	}
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("embedding blob size %d is not divisible by 4", len(blob))
	}

	vec := make([]float32, len(blob)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(blob[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return vec, nil
}

func runCommandCombinedOutput(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	return cmd.CombinedOutput()
}

func defaultVipsResize(sourcePath string, outputPath string, scale float64) []string {
	return []string{
		"resize",
		sourcePath,
		outputPath,
		fmt.Sprintf("%.6f", scale),
	}
}

func (e *Embedder) imageDimensionsWithVips(ctx context.Context, sourcePath string) (int, int, error) {
	width, err := e.readVipsHeaderDimension(ctx, sourcePath, "width")
	if err != nil {
		return 0, 0, err
	}
	height, err := e.readVipsHeaderDimension(ctx, sourcePath, "height")
	if err != nil {
		return 0, 0, err
	}
	return width, height, nil
}

func (e *Embedder) readVipsHeaderDimension(ctx context.Context, sourcePath string, field string) (int, error) {
	runner := e.runCommand
	if runner == nil {
		runner = runCommandCombinedOutput
	}
	commandOutput, err := runner(ctx, "vipsheader", "-f", field, sourcePath)
	if err != nil {
		stderr := strings.TrimSpace(string(commandOutput))
		if stderr != "" {
			return 0, fmt.Errorf("read image %s via vipsheader: %w: %s", field, err, stderr)
		}
		return 0, fmt.Errorf("read image %s via vipsheader: %w", field, err)
	}
	valueText := strings.TrimSpace(string(commandOutput))
	value, err := strconv.Atoi(valueText)
	if err != nil {
		return 0, fmt.Errorf("parse image %s %q: %w", field, valueText, err)
	}
	return value, nil
}

func embedInstructionOption(instruction string) string {
	instruction = strings.TrimSpace(instruction)
	if instruction == "" {
		return ""
	}
	return "instruction=" + instruction
}
