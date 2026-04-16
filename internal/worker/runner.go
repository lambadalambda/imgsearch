package worker

import (
	"context"
	"log"
	"time"
)

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

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if batchSize > 1 {
			count, err := q.ProcessBatch(ctx, owner, batchSize)
			if err != nil {
				log.Printf("worker batch error: %v", err)
				time.Sleep(idleDelay)
				continue
			}
			if count == 0 {
				time.Sleep(idleDelay)
			}
		} else {
			processed, err := q.ProcessOne(ctx, owner)
			if err != nil {
				log.Printf("worker process error: %v", err)
				time.Sleep(idleDelay)
				continue
			}
			if !processed {
				time.Sleep(idleDelay)
			}
		}
	}
}
