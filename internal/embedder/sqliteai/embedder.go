package sqliteai

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"sync"
)

const defaultContextOptions = "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=mean"

type Config struct {
	DB                 *sql.DB
	ModelPath          string
	ModelOptions       string
	VisionModelPath    string
	VisionModelOptions string
	ContextOptions     string
}

type Embedder struct {
	db                 *sql.DB
	modelPath          string
	modelOptions       string
	visionModelPath    string
	visionModelOptions string
	contextOptions     string

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

	return &Embedder{
		db:                 cfg.DB,
		modelPath:          strings.TrimSpace(cfg.ModelPath),
		modelOptions:       strings.TrimSpace(cfg.ModelOptions),
		visionModelPath:    strings.TrimSpace(cfg.VisionModelPath),
		visionModelOptions: strings.TrimSpace(cfg.VisionModelOptions),
		contextOptions:     ctxOptions,
	}, nil
}

func (e *Embedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if err := e.ensureInitialized(ctx); err != nil {
		return nil, err
	}

	var blob []byte
	if err := e.db.QueryRowContext(ctx, `SELECT llm_embed_generate(?)`, text).Scan(&blob); err != nil {
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

	var blob []byte
	if err := e.db.QueryRowContext(ctx, `SELECT llm_embed_generate(?)`, path).Scan(&blob); err != nil {
		return nil, fmt.Errorf("sqlite-ai image embedding failed: %w", err)
	}

	vec, err := decodeFloat32Blob(blob)
	if err != nil {
		return nil, fmt.Errorf("decode sqlite-ai image embedding: %w", err)
	}
	return vec, nil
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
