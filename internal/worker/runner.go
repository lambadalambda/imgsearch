package worker

import (
	"context"
	"log"
	"time"

	"imgsearch/internal/jobkind"
	"imgsearch/internal/settings"
)

// annotationConcurrency is implemented by the live backend resolver. Ordinary
// annotators (including native ones) default to one. Zero pauses new annotation
// claims while the resolver drains a backend/configuration change.
type annotationConcurrency interface {
	AnnotationConcurrency(context.Context) (int, error)
}

func RunLoop(ctx context.Context, q *Queue, owner string, idleDelay time.Duration) {
	RunLoopBatch(ctx, q, owner, idleDelay, 1)
}

func RunLoopBatch(ctx context.Context, q *Queue, owner string, idleDelay time.Duration, batchSize int) {
	if idleDelay <= 0 {
		idleDelay = 500 * time.Millisecond
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	if owner == "" {
		owner = "worker"
	}
	// A single dispatcher owns admission and claims. No embedding/transcription
	// work runs alongside annotation jobs, even when the backend changes to native.
	type completion struct {
		id  int64
		err error
	}
	completed := make(chan completion, settings.MaxOpenAIConcurrency)
	active := make(map[int64]struct{})
	activeKind := ""
	defer func() {
		for len(active) > 0 {
			result := <-completed
			delete(active, result.id)
		}
	}()
	ticker := time.NewTicker(idleDelay)
	defer ticker.Stop()
	for ctx.Err() == nil {
		err := q.validate()
		if err == nil {
			limit := 1
			if provider, ok := q.Annotator.(annotationConcurrency); ok {
				limit, err = provider.AnnotationConcurrency(ctx)
				if err != nil {
					log.Printf("worker annotation settings: %v", err)
					limit = 0
				}
			}
			limit = max(0, min(limit, settings.MaxOpenAIConcurrency))
			if len(active) == 0 || len(active) < limit {
				var jobs []claimedJob
				jobs, err = q.claimBatchWithLimits(ctx, owner, claimLimits{
					batchSize:       batchSize,
					annotationSlots: max(0, limit-len(active)),
					activeKind:      activeKind,
					activeIDs:       active,
				})
				if err == nil && len(jobs) > 0 {
					if isAnnotationKind(jobs[0].Kind) {
						for _, job := range jobs {
							active[job.ID] = struct{}{}
							activeKind = job.Kind
							go func() {
								_, _, err := q.executeClaimedJob(ctx, job)
								completed <- completion{id: job.ID, err: err}
							}()
						}
					} else {
						_, err = q.processClaimedBatch(ctx, jobs)
					}
					if err == nil {
						continue
					}
				}
			}
		}
		if err != nil {
			log.Printf("worker process error: %v", err)
		}
		select {
		case <-ctx.Done():
			return
		case result := <-completed:
			delete(active, result.id)
			if len(active) == 0 {
				activeKind = ""
			}
			if result.err != nil && ctx.Err() == nil {
				log.Printf("worker annotation error: %v", result.err)
			}
		case <-ticker.C:
		}
	}
}

func isAnnotationKind(kind string) bool {
	return kind == jobkind.AnnotateImage || kind == jobkind.AnnotateVideo
}
