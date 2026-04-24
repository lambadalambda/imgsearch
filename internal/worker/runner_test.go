package worker

import (
	"context"
	"testing"
	"time"
)

func TestRunLoopCancelsDuringIdleDelay(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	go func() {
		defer close(done)
		RunLoop(ctx, nil, "worker", time.Hour)
	}()

	time.Sleep(10 * time.Millisecond)
	cancel()

	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("worker loop did not exit promptly after cancellation")
	}
}

func TestRunLoopBatchCancelsDuringIdleDelay(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	go func() {
		defer close(done)
		RunLoopBatch(ctx, nil, "worker", time.Hour, 2)
	}()

	time.Sleep(10 * time.Millisecond)
	cancel()

	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("batch worker loop did not exit promptly after cancellation")
	}
}
