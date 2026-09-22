package images

import (
	"context"
	"testing"
)

func TestImagesExposeLazyThumbnailWithoutMutatingRows(t *testing.T) {
	conn := setupImagesDB(t)
	if _, err := conn.Exec(`UPDATE images SET thumbnail_path='images/custom.webp' WHERE id=2`); err != nil {
		t.Fatal(err)
	}
	for id, want := range map[int64]string{1: "images/thumbnails/1.webp", 2: "images/custom.webp"} {
		item, err := GetItem(context.Background(), conn, 1, id)
		if err != nil {
			t.Fatal(err)
		}
		if item.ThumbnailPath != want {
			t.Errorf("image %d thumbnail=%q want %q", id, item.ThumbnailPath, want)
		}
	}
	var count int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM images WHERE thumbnail_path IS NOT NULL`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatal("read path mutated thumbnail metadata")
	}
	page, err := List(context.Background(), conn, 1, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range page.Images {
		if item.ThumbnailPath == "" {
			t.Errorf("list image %d missing thumbnail", item.ImageID)
		}
	}
}
