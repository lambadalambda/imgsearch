package worker

import (
	"context"
	"errors"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"imgsearch/internal/embedder"
)

type annotationCall struct {
	path   string
	finish chan error
}

type blockingRemoteAnnotator struct {
	limit   atomic.Int32
	started chan *annotationCall
	release chan error
	exited  chan struct{}
}

func (a *blockingRemoteAnnotator) AnnotationConcurrency(context.Context) (int, error) {
	return int(a.limit.Load()), nil
}
func (a *blockingRemoteAnnotator) AnnotateImage(ctx context.Context, path string) (embedder.ImageAnnotation, error) {
	call := &annotationCall{path: filepath.Base(path), finish: make(chan error, 1)}
	a.started <- call
	defer func() { a.exited <- struct{}{} }()
	select {
	case err := <-call.finish:
		return embedder.ImageAnnotation{Description: "remote description", Tags: []string{"remote"}}, err
	case err := <-a.release:
		return embedder.ImageAnnotation{Description: "remote description", Tags: []string{"remote"}}, err
	case <-ctx.Done():
		return embedder.ImageAnnotation{}, ctx.Err()
	}
}

func startRemoteQueue(t *testing.T, n, limit, batch int, configure ...func(*Queue)) (*Queue, *blockingRemoteAnnotator, context.CancelFunc, <-chan struct{}) {
	t.Helper()
	q, conn := setupMultiImageQueue(t, n)
	// Production also uses a single SQLite connection. In-memory tests must do so
	// before parallel jobs, otherwise each connection gets a different database.
	conn.SetMaxOpenConns(1)
	if _, err := conn.Exec(`UPDATE index_jobs SET kind = 'annotate_image'`); err != nil {
		t.Fatal(err)
	}
	a := &blockingRemoteAnnotator{started: make(chan *annotationCall, n*4), release: make(chan error, n*4), exited: make(chan struct{}, n*4)}
	a.limit.Store(int32(limit))
	q.Annotator = a
	for _, fn := range configure {
		fn(q)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); RunLoopBatch(ctx, q, "remote-worker", 5*time.Millisecond, batch) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("worker did not drain")
		}
	})
	return q, a, cancel, done
}

func awaitAnnotation(t *testing.T, a *blockingRemoteAnnotator) *annotationCall {
	t.Helper()
	select {
	case path := <-a.started:
		return path
	case <-time.After(2 * time.Second):
		t.Fatal("annotation did not start")
		return nil
	}
}
func noAnnotation(t *testing.T, a *blockingRemoteAnnotator) {
	t.Helper()
	select {
	case path := <-a.started:
		t.Fatalf("unexpected annotation %s", path.path)
	case <-time.After(60 * time.Millisecond):
	}
}
func waitJobCount(t *testing.T, q *Queue, state string, want int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		var got int
		if err := q.DB.QueryRow(`SELECT count(*) FROM index_jobs WHERE state=?`, state).Scan(&got); err != nil {
			t.Fatal(err)
		}
		if got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%s jobs=%d want %d", state, got, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestRemoteSchedulingTwoOverlapThirdWaits(t *testing.T) {
	for _, batch := range []int{1, 8} {
		t.Run(strconv.Itoa(batch), func(t *testing.T) {
			q, a, _, _ := startRemoteQueue(t, 3, 2, batch)
			first, second := awaitAnnotation(t, a), awaitAnnotation(t, a)
			if first.path == second.path {
				t.Fatal("duplicate active job")
			}
			noAnnotation(t, a)
			waitJobCount(t, q, "leased", 2)
			waitJobCount(t, q, "pending", 1)
			a.release <- nil
			third := awaitAnnotation(t, a)
			if third.path == first.path || third.path == second.path {
				t.Fatal("duplicate work")
			}
			a.release <- nil
			a.release <- nil
			waitJobCount(t, q, "done", 3)
			var attempts int
			if err := q.DB.QueryRow(`SELECT sum(attempts) FROM index_jobs`).Scan(&attempts); err != nil {
				t.Fatal(err)
			}
			if attempts != 3 {
				t.Fatalf("attempts=%d", attempts)
			}
		})
	}
}
func TestRemoteSchedulingLiveLimits(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 4, 1, 8)
	awaitAnnotation(t, a)
	noAnnotation(t, a)
	waitJobCount(t, q, "leased", 1)
	a.limit.Store(2)
	awaitAnnotation(t, a)
	noAnnotation(t, a)
	a.limit.Store(1)
	a.release <- nil
	waitJobCount(t, q, "done", 1)
	noAnnotation(t, a)
	a.release <- nil
	awaitAnnotation(t, a)
	noAnnotation(t, a)
	a.release <- nil
	awaitAnnotation(t, a)
	a.release <- nil
	waitJobCount(t, q, "done", 4)
}
func TestRemoteSchedulingShutdownDrains(t *testing.T) {
	q, a, cancel, done := startRemoteQueue(t, 3, 2, 1)
	awaitAnnotation(t, a)
	awaitAnnotation(t, a)
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("shutdown stuck")
	}
	if len(a.exited) != 2 {
		t.Fatalf("worker returned before calls exited: %d", len(a.exited))
	}
	noAnnotation(t, a)
	waitJobCount(t, q, "pending", 1)
}
func TestRemoteSchedulingFailuresDoNotDuplicate(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 3, 2, 1, func(q *Queue) { q.RetryBaseDelay = time.Hour })
	first, second := awaitAnnotation(t, a), awaitAnnotation(t, a)
	first.finish <- errors.New("remote failed")
	third := awaitAnnotation(t, a)
	if third.path == first.path || third.path == second.path {
		t.Fatal("duplicate job")
	}
	second.finish <- nil
	third.finish <- nil
	waitJobCount(t, q, "done", 2)
	waitJobCount(t, q, "pending", 1)
	noAnnotation(t, a)
	// Make the legitimate retry runnable, then exhaust its budget.
	if _, err := q.DB.Exec(`UPDATE index_jobs SET run_after=NULL, max_attempts=2 WHERE state='pending'`); err != nil {
		t.Fatal(err)
	}
	retry := awaitAnnotation(t, a)
	if retry.path != first.path {
		t.Fatal("retried wrong job")
	}
	retry.finish <- errors.New("remote failed again")
	waitJobCount(t, q, "failed", 1)
	noAnnotation(t, a)
	var attempts int
	if err := q.DB.QueryRow(`SELECT sum(attempts) FROM index_jobs`).Scan(&attempts); err != nil {
		t.Fatal(err)
	}
	if attempts != 4 {
		t.Fatalf("attempts=%d", attempts)
	}
}

func TestRemoteSchedulingKeepsLeases(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 2, 2, 1, func(q *Queue) { q.LeaseDuration = 2 * time.Second })
	first, second := awaitAnnotation(t, a), awaitAnnotation(t, a)
	// Observe natural renewal beyond both original deadlines. Shortening a
	// whole-second SQLite deadline can accidentally expire an otherwise live job.
	var originalDeadline string
	if err := q.DB.QueryRow(`SELECT max(leased_until) FROM index_jobs`).Scan(&originalDeadline); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		var renewed int
		if err := q.DB.QueryRow(`SELECT count(*) FROM index_jobs WHERE leased_until > ? AND lease_owner='remote-worker'`, originalDeadline).Scan(&renewed); err != nil {
			t.Fatal(err)
		}
		if renewed == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("active leases not renewed")
		}
		time.Sleep(10 * time.Millisecond)
	}
	jobs, err := q.claimBatch(context.Background(), "competitor", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 0 {
		t.Fatal("competitor reclaimed live jobs")
	}
	noAnnotation(t, a)
	if len(a.exited) != 0 {
		t.Fatal("original calls exited before release")
	}
	var originalClaims int
	if err := q.DB.QueryRow(`SELECT count(*) FROM index_jobs WHERE state='leased' AND lease_owner='remote-worker' AND attempts=1`).Scan(&originalClaims); err != nil {
		t.Fatal(err)
	}
	if originalClaims != 2 {
		t.Fatalf("original claims still leased=%d, want 2", originalClaims)
	}
	first.finish <- nil
	second.finish <- nil
	waitJobCount(t, q, "done", 2)
}

func TestRemoteSchedulingDoesNotReclaimItsActiveRequeuedJob(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 2, 2, 1)
	first, second := awaitAnnotation(t, a), awaitAnnotation(t, a)
	imageID := first.path[len("sha"):]
	if _, err := q.DB.Exec(`UPDATE index_jobs SET state='pending',lease_owner=NULL,leased_until=NULL WHERE image_id=?`, imageID); err != nil {
		t.Fatal(err)
	}
	second.finish <- nil
	waitJobCount(t, q, "done", 1)
	noAnnotation(t, a) // the old call still owns a dispatcher slot for this ID
	first.finish <- nil
	retry := awaitAnnotation(t, a)
	if retry.path != first.path {
		t.Fatal("wrong requeued job")
	}
	// Old completion must not have persisted annotations after losing ownership.
	var description string
	if err := q.DB.QueryRow(`SELECT COALESCE(description,'') FROM images WHERE id=?`, imageID).Scan(&description); err != nil {
		t.Fatal(err)
	}
	if description != "" {
		t.Fatal("stale result persisted")
	}
	retry.finish <- nil
	waitJobCount(t, q, "done", 2)
}

func TestRemoteSchedulingDrainsBeforeHigherPriorityWork(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 4, 2, 1)
	first, second := awaitAnnotation(t, a), awaitAnnotation(t, a)
	if _, err := q.DB.Exec(`UPDATE index_jobs SET kind='embed_image' WHERE image_id=4`); err != nil {
		t.Fatal(err)
	}
	first.finish <- nil
	waitJobCount(t, q, "done", 1)
	noAnnotation(t, a) // don't refill forever and starve embeddings
	second.finish <- nil
	third := awaitAnnotation(t, a)
	var state string
	if err := q.DB.QueryRow(`SELECT state FROM index_jobs WHERE image_id=4`).Scan(&state); err != nil {
		t.Fatal(err)
	}
	if state != "done" {
		t.Fatalf("embedding state=%s", state)
	}
	third.finish <- nil
	waitJobCount(t, q, "done", 4)
}

func TestRemoteSchedulingOrdinaryAnnotatorRemainsSerial(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 3, 2, 8, func(q *Queue) {
		// Native/ordinary annotators do not advertise remote scheduling capacity.
		q.Annotator = &struct{ embedder.ImageAnnotator }{q.Annotator}
	})
	for range 3 {
		call := awaitAnnotation(t, a)
		noAnnotation(t, a)
		waitJobCount(t, q, "leased", 1)
		call.finish <- nil
	}
	waitJobCount(t, q, "done", 3)
}

func TestClaimBatchDefaultLimitStillClaimsAnnotations(t *testing.T) {
	q, conn := setupMultiImageQueue(t, 1)
	q.Annotator = &fakeAnnotator{}
	if _, err := conn.Exec(`UPDATE index_jobs SET kind='annotate_image'`); err != nil {
		t.Fatal(err)
	}
	jobs, err := q.claimBatch(context.Background(), "worker", 0)
	if err != nil || len(jobs) != 1 {
		t.Fatalf("default batch jobs=%v err=%v", jobs, err)
	}
}

func (a *blockingRemoteAnnotator) AnnotateVideo(ctx context.Context, input embedder.VideoAnnotationInput) (embedder.VideoAnnotation, error) {
	annotation, err := a.AnnotateImage(ctx, input.OriginalName)
	return embedder.VideoAnnotation{Description: annotation.Description, Tags: annotation.Tags}, err
}

func TestRemoteSchedulingVideoJobsUseSameBound(t *testing.T) {
	q, a, _, _ := startRemoteQueue(t, 3, 2, 8, func(q *Queue) {
		statements := []string{
			`DELETE FROM index_jobs`,
			`UPDATE images SET description='frame description',tags_json='["frame"]'`,
			`INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,duration_ms,width,height,frame_count)
    SELECT id,sha256,original_name,storage_path,'video/mp4',1000,10,10,1 FROM images`,
			`INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) SELECT id,id,0,0 FROM images`,
			`INSERT INTO index_jobs(kind,video_id,model_id,state) SELECT 'annotate_video',id,1,'pending' FROM videos`,
		}
		for _, sql := range statements {
			if _, err := q.DB.Exec(sql); err != nil {
				t.Fatal(err)
			}
		}
	})
	first, second := awaitAnnotation(t, a), awaitAnnotation(t, a)
	if first.path == second.path {
		t.Fatal("duplicate video")
	}
	noAnnotation(t, a)
	waitJobCount(t, q, "leased", 2)
	waitJobCount(t, q, "pending", 1)
	first.finish <- nil
	third := awaitAnnotation(t, a)
	if third.path == first.path || third.path == second.path {
		t.Fatal("duplicate video")
	}
	second.finish <- nil
	third.finish <- nil
	waitJobCount(t, q, "done", 3)
}
