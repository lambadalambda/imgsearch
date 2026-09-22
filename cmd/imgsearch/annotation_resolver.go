package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"sync"
	"time"

	"imgsearch/internal/app"
	"imgsearch/internal/embedder"
	"imgsearch/internal/settings"
)

const defaultAnnotationSettingsCheckInterval = 5 * time.Second

// annotationBackendBuilder constructs the annotator for one backend and
// returns a status describing it.
type annotationBackendBuilder func(ctx context.Context, s settings.AnnotationSettings) (embedder.ImageAnnotator, settings.ActiveAnnotation, error)

type annotatorResolverOptions struct {
	DB       *sql.DB
	Defaults settings.AnnotationSettings
	// BuildNative receives the requested variant; the loaded model itself
	// belongs to the switchboard, so the resolver never closes it.
	BuildNative func(ctx context.Context, variant string) (embedder.ImageAnnotator, settings.ActiveAnnotation, error)
	// BuildRemote returns an annotator the resolver owns and closes on swap.
	BuildRemote func(ctx context.Context, s settings.AnnotationSettings) (embedder.ImageAnnotator, settings.ActiveAnnotation, error)
	// CheckInterval throttles settings-version reads between jobs.
	CheckInterval time.Duration
	Now           func() time.Time
}

type resolvedAnnotator struct {
	key         string
	annotator   embedder.ImageAnnotator
	status      settings.ActiveAnnotation
	concurrency int
	owned       bool
}

// annotatorResolver presents a stable embedder.ImageAnnotator to the worker
// while swapping the backend underneath whenever the persisted annotation
// settings change. Backend/configuration changes drain existing calls before
// rebuilding, including native loader changes made by the settings status API.
type annotatorResolver struct {
	opts annotatorResolverOptions

	mu        sync.Mutex
	lastCheck time.Time
	checked   bool
	version   int64
	current   *resolvedAnnotator
	lastErr   error
	active    int
	draining  bool
	closed    bool
	changed   chan struct{}
}

func newAnnotatorResolver(opts annotatorResolverOptions) *annotatorResolver {
	if opts.CheckInterval <= 0 {
		opts.CheckInterval = defaultAnnotationSettingsCheckInterval
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return &annotatorResolver{opts: opts, changed: make(chan struct{})}
}

var _ embedder.ImageAnnotatorWithOptions = (*annotatorResolver)(nil)
var _ embedder.VideoFrameAnnotator = (*annotatorResolver)(nil)
var _ embedder.VideoAnnotator = (*annotatorResolver)(nil)
var _ app.Closer = (*annotatorResolver)(nil)

// resolve returns the annotator matching the current settings, rebuilding
// when the settings version changed. force bypasses the check throttle.
func (r *annotatorResolver) resolve(ctx context.Context, force bool) (*resolvedAnnotator, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.resolveLocked(ctx, force)
}

func (r *annotatorResolver) resolveLocked(ctx context.Context, force bool) (*resolvedAnnotator, error) {
	if r.closed {
		return nil, fmt.Errorf("annotation resolver is closed")
	}
	now := r.opts.Now()
	if !force && !r.draining && r.checked && now.Sub(r.lastCheck) < r.opts.CheckInterval {
		return r.currentLocked()
	}
	version, err := settings.Version(ctx, r.opts.DB)
	if err != nil {
		return nil, fmt.Errorf("read annotation settings version: %w", err)
	}
	r.lastCheck = now
	if r.checked && version == r.version && r.current != nil {
		return r.currentLocked()
	}
	r.checked = true

	s, err := settings.LoadAnnotationOrDefault(ctx, r.opts.DB, r.opts.Defaults)
	if err != nil {
		r.lastErr = fmt.Errorf("load annotation settings: %w", err)
		return nil, r.lastErr
	}
	key := annotationBackendKey(s)
	if r.current != nil && r.current.key == key {
		// resolvedAnnotator values are immutable once published because
		// callers read them outside the lock; swap in a copy instead.
		refreshed := *r.current
		refreshed.status.SettingsVersion = version
		r.current = &refreshed
		r.version = version
		r.draining = false
		r.lastErr = nil
		return r.current, nil
	}

	// Stop admission and keep reporting the genuinely active backend until all
	// calls finish. Never unload native resources or close a remote client in use.
	if r.active > 0 {
		r.draining = true
		return r.currentLocked()
	}
	r.draining = false

	// A backend change replaces the current annotator even when the new one
	// fails to build: serving the old backend intermittently would be more
	// confusing than failing consistently until the settings are fixed.
	r.dropCurrentLocked()
	r.version = version
	next, err := r.build(ctx, s)
	if err != nil {
		r.lastErr = err
		log.Printf("annotation backend unavailable: %v", err)
		return nil, err
	}
	next.key = key
	next.status.SettingsVersion = version
	r.current = next
	r.lastErr = nil
	log.Printf("annotation backend: %s (%s)", next.status.Backend, describeActiveAnnotation(next.status))
	return next, nil
}

func (r *annotatorResolver) dropCurrentLocked() {
	if r.current != nil && r.current.owned {
		if err := closeLlamaModel("remote annotator", r.current.annotator); err != nil {
			log.Printf("%v", err)
		}
	}
	r.current = nil
}

func (r *annotatorResolver) currentLocked() (*resolvedAnnotator, error) {
	if r.current == nil {
		if r.lastErr != nil {
			return nil, r.lastErr
		}
		return nil, fmt.Errorf("annotation backend is not configured")
	}
	return r.current, nil
}

func (r *annotatorResolver) build(ctx context.Context, s settings.AnnotationSettings) (*resolvedAnnotator, error) {
	switch s.Backend {
	case settings.BackendOpenAI:
		if r.opts.BuildRemote == nil {
			return nil, fmt.Errorf("remote annotation backend is unavailable in this process")
		}
		a, status, err := r.opts.BuildRemote(ctx, s)
		if err != nil {
			return nil, err
		}
		return &resolvedAnnotator{annotator: a, status: status, owned: true, concurrency: max(1, min(s.OpenAI.Concurrency, settings.MaxOpenAIConcurrency))}, nil
	default:
		if r.opts.BuildNative == nil {
			return nil, fmt.Errorf("native annotation backend is unavailable in this process")
		}
		a, status, err := r.opts.BuildNative(ctx, s.NativeVariant)
		if err != nil {
			return nil, err
		}
		return &resolvedAnnotator{annotator: a, status: status, concurrency: 1}, nil
	}
}

// annotationBackendKey identifies a backend configuration so unrelated
// settings saves do not trigger a rebuild.
func annotationBackendKey(s settings.AnnotationSettings) string {
	if s.Backend == settings.BackendOpenAI {
		return fmt.Sprintf("openai|%s|%s|%s|%d|%d", s.OpenAI.BaseURL, s.OpenAI.Model, s.OpenAI.APIKey, s.OpenAI.TimeoutSeconds, s.OpenAI.Concurrency)
	}
	return "native|" + s.NativeVariant
}

func describeActiveAnnotation(a settings.ActiveAnnotation) string {
	if a.Detail != "" {
		return a.Model + " @ " + a.Detail
	}
	return a.Model
}

// Status reports the active backend, checking for settings changes first so
// the settings page reflects a save immediately in the same process.
func (r *annotatorResolver) Status(ctx context.Context) (settings.ActiveAnnotation, error) {
	current, err := r.resolve(ctx, true)
	if err != nil {
		return settings.ActiveAnnotation{}, err
	}
	status := current.status
	status.Source = settings.ActiveSourceWorker
	return status, nil
}

// AnnotationConcurrency is the worker's live admission limit, not just an HTTP
// semaphore size. A configuration change pauses claims until old calls drain.
func (r *annotatorResolver) AnnotationConcurrency(ctx context.Context) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	current, err := r.resolveLocked(ctx, false)
	if err != nil {
		return 0, err
	}
	if r.draining {
		return 0, nil
	}
	return current.concurrency, nil
}

// acquire also enforces admission here: status requests can observe a settings
// save between the worker's claim and the actual annotation call.
func (r *annotatorResolver) acquire(ctx context.Context) (*resolvedAnnotator, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		r.mu.Lock()
		current, err := r.resolveLocked(ctx, false)
		if err != nil {
			r.mu.Unlock()
			return nil, err
		}
		if !r.draining && r.active < current.concurrency {
			r.active++
			r.mu.Unlock()
			return current, nil
		}
		changed := r.changed
		r.mu.Unlock()
		timer := time.NewTimer(r.opts.CheckInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-changed:
		case <-timer.C:
		}
		timer.Stop()
	}
}

func (r *annotatorResolver) release() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.active--
	if r.closed && r.active == 0 {
		r.dropCurrentLocked()
	}
	close(r.changed)
	r.changed = make(chan struct{})
}

func (r *annotatorResolver) Close() error {
	r.mu.Lock()
	r.closed = true
	// Wake callers waiting for admission so they can fail without starting work.
	close(r.changed)
	r.changed = make(chan struct{})
	for r.active > 0 {
		changed := r.changed
		r.mu.Unlock()
		<-changed
		r.mu.Lock()
	}
	r.dropCurrentLocked()
	r.mu.Unlock()
	return nil
}

func (r *annotatorResolver) AnnotateImage(ctx context.Context, path string) (embedder.ImageAnnotation, error) {
	return r.AnnotateImageWithOptions(ctx, path, embedder.ImageAnnotationOptions{})
}

func (r *annotatorResolver) AnnotateImageWithOptions(ctx context.Context, path string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	current, err := r.acquire(ctx)
	if err != nil {
		return embedder.ImageAnnotation{}, err
	}
	defer r.release()
	if withOptions, ok := current.annotator.(embedder.ImageAnnotatorWithOptions); ok {
		return withOptions.AnnotateImageWithOptions(ctx, path, opts)
	}
	return current.annotator.AnnotateImage(ctx, path)
}

func (r *annotatorResolver) AnnotateVideoFrame(ctx context.Context, path string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	current, err := r.acquire(ctx)
	if err != nil {
		return embedder.ImageAnnotation{}, err
	}
	defer r.release()
	if frameAnnotator, ok := current.annotator.(embedder.VideoFrameAnnotator); ok {
		return frameAnnotator.AnnotateVideoFrame(ctx, path, opts)
	}
	if withOptions, ok := current.annotator.(embedder.ImageAnnotatorWithOptions); ok {
		return withOptions.AnnotateImageWithOptions(ctx, path, opts)
	}
	return current.annotator.AnnotateImage(ctx, path)
}

func (r *annotatorResolver) AnnotateVideo(ctx context.Context, input embedder.VideoAnnotationInput) (embedder.VideoAnnotation, error) {
	current, err := r.acquire(ctx)
	if err != nil {
		return embedder.VideoAnnotation{}, err
	}
	defer r.release()
	videoAnnotator, ok := current.annotator.(embedder.VideoAnnotator)
	if !ok {
		return embedder.VideoAnnotation{}, fmt.Errorf("configured annotation backend does not support video annotation")
	}
	return videoAnnotator.AnnotateVideo(ctx, input)
}
