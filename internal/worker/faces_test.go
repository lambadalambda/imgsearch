package worker

import (
	"context"
	"testing"

	"imgsearch/internal/jobkind"
	"imgsearch/internal/people"
)

type fakeFaceDetector struct {
	dets  []people.Detection
	err   error
	calls int
	paths []string
}

func (f *fakeFaceDetector) DetectFaces(_ context.Context, path string) ([]people.Detection, error) {
	f.calls++
	f.paths = append(f.paths, path)
	if f.err != nil {
		return nil, f.err
	}
	return f.dets, nil
}

func TestIndexFacesJobDetectsAndClusters(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	detector := &fakeFaceDetector{dets: []people.Detection{
		{BoxX: 0.1, BoxY: 0.1, BoxW: 0.2, BoxH: 0.2, Confidence: 0.9, Embedding: []float32{1, 0, 0, 0}},
	}}
	q.FaceDetector = detector

	if _, err := sqlDB.Exec(`INSERT INTO index_jobs(kind, image_id, model_id, state) VALUES (?, 1, (SELECT id FROM embedding_models LIMIT 1), 'pending')`, jobkind.IndexFaces); err != nil {
		t.Fatal(err)
	}
	// The seeded embed job claims first; drain it, then the face job runs.
	for i := 0; i < 2; i++ {
		if _, err := q.ProcessOne(context.Background(), "test-owner"); err != nil {
			t.Fatal(err)
		}
	}
	if detector.calls != 1 {
		t.Fatalf("detector calls = %d, want 1", detector.calls)
	}
	var faces, persons int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM face_detections`).Scan(&faces); err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons); err != nil {
		t.Fatal(err)
	}
	if faces != 1 || persons != 1 {
		t.Fatalf("faces=%d persons=%d, want 1/1", faces, persons)
	}
	var state string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE kind='index_faces'`).Scan(&state); err != nil {
		t.Fatal(err)
	}
	if state != "done" {
		t.Fatalf("state = %s, want done", state)
	}
}

func TestIndexFacesJobWithoutDetectorStaysPending(t *testing.T) {
	q, sqlDB := setupQueueTest(t)
	q.FaceDetector = nil

	if _, err := sqlDB.Exec(`INSERT INTO index_jobs(kind, image_id, model_id, state) VALUES (?, 1, (SELECT id FROM embedding_models LIMIT 1), 'pending')`, jobkind.IndexFaces); err != nil {
		t.Fatal(err)
	}
	done, err := q.ProcessOne(context.Background(), "test-owner")
	if err != nil {
		t.Fatal(err)
	}
	if done {
		// The seeded embed job processes first; the face job must remain.
		done, err = q.ProcessOne(context.Background(), "test-owner")
		if err != nil {
			t.Fatal(err)
		}
	}
	if done {
		t.Fatal("face job must not process without a detector")
	}
	var state string
	if err := sqlDB.QueryRow(`SELECT state FROM index_jobs WHERE kind='index_faces'`).Scan(&state); err != nil {
		t.Fatal(err)
	}
	if state != "pending" {
		t.Fatalf("state = %s, want pending", state)
	}
}
