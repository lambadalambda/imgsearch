package main

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"imgsearch/internal/embedder"
)

type llamaModelRole int

const (
	llamaModelRoleNone llamaModelRole = iota
	llamaModelRoleEmbedder
	llamaModelRoleAnnotator
)

type llamaEmbedderLoader func(context.Context) (embedder.Embedder, error)
type llamaAnnotatorLoader func(context.Context) (embedder.ImageAnnotator, error)

type llamaModelSwitchboard struct {
	mu                     sync.Mutex
	closed                 bool
	active                 llamaModelRole
	embedder               embedder.Embedder
	annotator              embedder.ImageAnnotator
	loadEmbedder           llamaEmbedderLoader
	loadAnnotator          llamaAnnotatorLoader
	annotatorSupportsVideo bool
}

func newLlamaModelSwitchboard(initialEmbedder embedder.Embedder, loadEmbedder llamaEmbedderLoader, loadAnnotator llamaAnnotatorLoader, annotatorSupportsVideo bool) *llamaModelSwitchboard {
	active := llamaModelRoleNone
	if initialEmbedder != nil {
		active = llamaModelRoleEmbedder
	}
	return &llamaModelSwitchboard{
		active:                 active,
		embedder:               initialEmbedder,
		loadEmbedder:           loadEmbedder,
		loadAnnotator:          loadAnnotator,
		annotatorSupportsVideo: annotatorSupportsVideo,
	}
}

func (s *llamaModelSwitchboard) Embedder() embedder.Embedder {
	return &switchingEmbedder{switchboard: s}
}

func (s *llamaModelSwitchboard) Annotator() embedder.ImageAnnotator {
	base := &switchingAnnotator{switchboard: s}
	if s != nil && s.annotatorSupportsVideo {
		return &switchingVideoAnnotator{switchingAnnotator: base}
	}
	return base
}

func (s *llamaModelSwitchboard) Close() error {
	if s == nil {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	errAnnotator := s.closeAnnotatorLocked()
	errEmbedder := s.closeEmbedderLocked()

	s.active = llamaModelRoleNone
	return errors.Join(errAnnotator, errEmbedder)
}

func (s *llamaModelSwitchboard) withEmbedder(ctx context.Context, fn func(embedder.Embedder) error) error {
	if s == nil {
		return fmt.Errorf("llama model switchboard is unavailable")
	}
	if err := switchboardContextErr(ctx); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return fmt.Errorf("llama model switchboard is closed")
	}

	if s.active != llamaModelRoleEmbedder {
		if err := s.closeAnnotatorLocked(); err != nil {
			return err
		}
	}

	if s.embedder == nil {
		if s.loadEmbedder == nil {
			return fmt.Errorf("embedder loader is unavailable")
		}
		loaded, err := s.loadEmbedder(ctx)
		if err != nil {
			return err
		}
		s.embedder = loaded
	}

	s.active = llamaModelRoleEmbedder
	if err := switchboardContextErr(ctx); err != nil {
		return err
	}
	return fn(s.embedder)
}

func (s *llamaModelSwitchboard) withAnnotator(ctx context.Context, fn func(embedder.ImageAnnotator) error) error {
	if s == nil {
		return fmt.Errorf("llama model switchboard is unavailable")
	}
	if err := switchboardContextErr(ctx); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return fmt.Errorf("llama model switchboard is closed")
	}

	if s.active != llamaModelRoleAnnotator {
		if err := s.closeEmbedderLocked(); err != nil {
			return err
		}
	}

	if s.annotator == nil {
		if s.loadAnnotator == nil {
			return fmt.Errorf("annotator loader is unavailable")
		}
		loaded, err := s.loadAnnotator(ctx)
		if err != nil {
			return err
		}
		s.annotator = loaded
	}

	s.active = llamaModelRoleAnnotator
	if err := switchboardContextErr(ctx); err != nil {
		return err
	}
	return fn(s.annotator)
}

func switchboardContextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}

func (s *llamaModelSwitchboard) closeEmbedderLocked() error {
	if s.embedder == nil {
		return nil
	}
	err := closeLlamaModel("embedder", s.embedder)
	s.embedder = nil
	if s.active == llamaModelRoleEmbedder {
		s.active = llamaModelRoleNone
	}
	return err
}

func (s *llamaModelSwitchboard) closeAnnotatorLocked() error {
	if s.annotator == nil {
		return nil
	}
	err := closeLlamaModel("annotator", s.annotator)
	s.annotator = nil
	if s.active == llamaModelRoleAnnotator {
		s.active = llamaModelRoleNone
	}
	return err
}

func closeLlamaModel(role string, model any) error {
	closer, ok := model.(interface{ Close() error })
	if !ok {
		return nil
	}
	if err := closer.Close(); err != nil {
		return fmt.Errorf("close %s model: %w", role, err)
	}
	return nil
}

type switchingEmbedder struct {
	switchboard *llamaModelSwitchboard
}

func (e *switchingEmbedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if e == nil || e.switchboard == nil {
		return nil, fmt.Errorf("llama model switchboard embedder is unavailable")
	}
	var vec []float32
	err := e.switchboard.withEmbedder(ctx, func(active embedder.Embedder) error {
		out, err := active.EmbedText(ctx, text)
		if err != nil {
			return err
		}
		vec = out
		return nil
	})
	if err != nil {
		return nil, err
	}
	return vec, nil
}

func (e *switchingEmbedder) EmbedImage(ctx context.Context, path string) ([]float32, error) {
	if e == nil || e.switchboard == nil {
		return nil, fmt.Errorf("llama model switchboard embedder is unavailable")
	}
	var vec []float32
	err := e.switchboard.withEmbedder(ctx, func(active embedder.Embedder) error {
		out, err := active.EmbedImage(ctx, path)
		if err != nil {
			return err
		}
		vec = out
		return nil
	})
	if err != nil {
		return nil, err
	}
	return vec, nil
}

func (e *switchingEmbedder) EmbedImages(ctx context.Context, paths []string) ([][]float32, error) {
	if e == nil || e.switchboard == nil {
		return nil, fmt.Errorf("llama model switchboard embedder is unavailable")
	}
	var vecs [][]float32
	err := e.switchboard.withEmbedder(ctx, func(active embedder.Embedder) error {
		if batcher, ok := active.(embedder.BatchImageEmbedder); ok {
			out, err := batcher.EmbedImages(ctx, paths)
			if err != nil {
				return err
			}
			vecs = out
			return nil
		}

		out := make([][]float32, 0, len(paths))
		for _, path := range paths {
			vec, err := active.EmbedImage(ctx, path)
			if err != nil {
				return err
			}
			out = append(out, vec)
		}
		vecs = out
		return nil
	})
	if err != nil {
		return nil, err
	}
	return vecs, nil
}

type switchingAnnotator struct {
	switchboard *llamaModelSwitchboard
}

func (a *switchingAnnotator) AnnotateImage(ctx context.Context, path string) (embedder.ImageAnnotation, error) {
	return a.AnnotateImageWithOptions(ctx, path, embedder.ImageAnnotationOptions{})
}

func (a *switchingAnnotator) AnnotateImageWithOptions(ctx context.Context, path string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	if a == nil || a.switchboard == nil {
		return embedder.ImageAnnotation{}, fmt.Errorf("llama model switchboard annotator is unavailable")
	}
	var annotation embedder.ImageAnnotation
	err := a.switchboard.withAnnotator(ctx, func(active embedder.ImageAnnotator) error {
		if withOptions, ok := active.(embedder.ImageAnnotatorWithOptions); ok {
			out, err := withOptions.AnnotateImageWithOptions(ctx, path, opts)
			if err != nil {
				return err
			}
			annotation = out
			return nil
		}
		out, err := active.AnnotateImage(ctx, path)
		if err != nil {
			return err
		}
		annotation = out
		return nil
	})
	if err != nil {
		return embedder.ImageAnnotation{}, err
	}
	return annotation, nil
}

type switchingVideoAnnotator struct {
	*switchingAnnotator
}

func (a *switchingVideoAnnotator) AnnotateVideo(ctx context.Context, input embedder.VideoAnnotationInput) (embedder.VideoAnnotation, error) {
	if a == nil || a.switchingAnnotator == nil || a.switchboard == nil {
		return embedder.VideoAnnotation{}, fmt.Errorf("llama model switchboard annotator is unavailable")
	}
	var annotation embedder.VideoAnnotation
	err := a.switchboard.withAnnotator(ctx, func(active embedder.ImageAnnotator) error {
		videoAnnotator, ok := active.(embedder.VideoAnnotator)
		if !ok {
			return fmt.Errorf("configured annotator does not support video annotation")
		}
		out, err := videoAnnotator.AnnotateVideo(ctx, input)
		if err != nil {
			return err
		}
		annotation = out
		return nil
	})
	if err != nil {
		return embedder.VideoAnnotation{}, err
	}
	return annotation, nil
}
