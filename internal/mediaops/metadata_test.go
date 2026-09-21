package mediaops

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func setupMetadataDB(t *testing.T) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if err := db.RunMigrations(context.Background(), conn); err != nil {
		t.Fatalf("migrations: %v", err)
	}
	if _, err := conn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, title, tags_json, annotator_tags_json)
VALUES (1, 'a', 'a.jpg', 'images/a', 'image/jpeg', 1, 1, 'Annotated title', '["cat","blurry"]', '["cat","blurry"]')`); err != nil {
		t.Fatalf("seed: %v", err)
	}
	return conn
}

func readTags(t *testing.T, conn *sql.DB) (served, annotator, user, removed, title, userTitle string) {
	t.Helper()
	if err := conn.QueryRow(`SELECT tags_json, annotator_tags_json, user_tags_json, removed_tags_json, title, user_title FROM images WHERE id = 1`).
		Scan(&served, &annotator, &user, &removed, &title, &userTitle); err != nil {
		t.Fatalf("read: %v", err)
	}
	return
}

func TestUserEditsSurviveReannotation(t *testing.T) {
	conn := setupMetadataDB(t)
	ctx := context.Background()

	patch, err := DecodeMetadataPatch(strings.NewReader(`{"title":"  My title ","tags":["cat","holiday","Holiday",""]}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if err := ApplyMetadataPatch(ctx, conn, TableImages, 1, patch); err != nil {
		t.Fatalf("apply: %v", err)
	}
	served, _, user, removed, title, userTitle := readTags(t, conn)
	if served != `["cat","holiday"]` || user != `["holiday"]` || removed != `["blurry"]` || title != "My title" || userTitle != "My title" {
		t.Fatalf("after patch: served=%s user=%s removed=%s title=%q user_title=%q", served, user, removed, title, userTitle)
	}

	// The worker re-annotates: new tags merge with the edits, title is kept.
	if _, err := conn.ExecContext(ctx, `UPDATE images SET `+AnnotatorTitleSQL+` WHERE id = 1`, "Fresh annotator title"); err != nil {
		t.Fatalf("annotator title: %v", err)
	}
	merged, err := ApplyAnnotatorTags(ctx, conn, TableImages, 1, []string{"cat", "blurry", "outdoor"})
	if err != nil {
		t.Fatalf("annotator tags: %v", err)
	}
	if strings.Join(merged, ",") != "cat,outdoor,holiday" {
		t.Fatalf("merged: %v", merged)
	}
	served, annotator, _, _, title, _ := readTags(t, conn)
	if served != `["cat","outdoor","holiday"]` || annotator != `["cat","blurry","outdoor"]` || title != "My title" {
		t.Fatalf("after reannotate: served=%s annotator=%s title=%q", served, annotator, title)
	}

	// Clearing the manual title lets the next annotation through.
	if err := SetUserTitle(ctx, conn, TableImages, 1, ""); err != nil {
		t.Fatalf("clear title: %v", err)
	}
	if _, err := conn.ExecContext(ctx, `UPDATE images SET `+AnnotatorTitleSQL+` WHERE id = 1`, "Second annotator title"); err != nil {
		t.Fatalf("annotator title 2: %v", err)
	}
	if _, _, _, _, title, _ = readTags(t, conn); title != "Second annotator title" {
		t.Fatalf("title after clearing override: %q", title)
	}
}

func TestDecodeMetadataPatchValidation(t *testing.T) {
	cases := []string{
		`{}`,
		`{"title": 5}`,
		`{"tags": "cat"}`,
		`{"unknown": 1, "title": "x"}`,
		`{"title": "` + strings.Repeat("x", MaxTitleLength+1) + `"}`,
		`{"tags": ["` + strings.Repeat("y", MaxTagLength+1) + `"]}`,
		`not json`,
	}
	for _, body := range cases {
		if _, err := DecodeMetadataPatch(strings.NewReader(body)); !errors.Is(err, ErrInvalidPatch) {
			t.Fatalf("expected ErrInvalidPatch for %s, got %v", body, err)
		}
	}
	patch, err := DecodeMetadataPatch(strings.NewReader(`{"tags": []}`))
	if err != nil || patch.Tags == nil || len(*patch.Tags) != 0 || patch.Title != nil {
		t.Fatalf("empty tags must be a valid full replacement: %+v %v", patch, err)
	}
}

func TestApplyMetadataPatchMissingItem(t *testing.T) {
	conn := setupMetadataDB(t)
	title := "x"
	err := ApplyMetadataPatch(context.Background(), conn, TableImages, 99, MetadataPatch{Title: &title})
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("expected ErrNoRows, got %v", err)
	}
	tags := []string{"a"}
	err = ApplyMetadataPatch(context.Background(), conn, TableImages, 99, MetadataPatch{Tags: &tags})
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("expected ErrNoRows for tags, got %v", err)
	}
	if _, err := SetServedTags(context.Background(), conn, Table("users"), 1, nil); err == nil {
		t.Fatal("expected unknown table to be rejected")
	}
}
