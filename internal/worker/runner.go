package worker

import (
	"context"
	"log"
	"time"
)

func RunLoop(ctx context.Context, q *Queue, owner string, idleDelay time.Duration) {
	if idleDelay <= 0 {
		idleDelay = 500 * time.Millisecond
	}

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

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
