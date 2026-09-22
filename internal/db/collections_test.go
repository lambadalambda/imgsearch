package db

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
)

func TestCollectionsMigrationPreservesLibraryAndPersistsMembership(t *testing.T) {
	path := filepath.Join(t.TempDir(), "library.sqlite")
	conn, err := sql.Open("sqlite3", path+"?_foreign_keys=on")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { conn.Close() }()
	ctx := context.Background()
	// Start with the schema users have before collections, including existing media.
	for _, m := range migrations {
		if m.version >= 16 {
			break
		}
		if _, err := conn.Exec(m.sql); err != nil {
			t.Fatal(err)
		}
		if _, err := conn.Exec(`INSERT INTO schema_migrations(version) VALUES(?)`, m.version); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := conn.Exec(`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height) VALUES(42,'existing','keep.jpg','images/keep.jpg','image/jpeg',32,64)`); err != nil {
		t.Fatal(err)
	}
	if err := RunMigrations(ctx, conn); err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Exec(`INSERT INTO collections(id,name) VALUES(1,'Saved'); INSERT INTO collection_items(collection_id,image_id) VALUES(1,42)`); err != nil {
		t.Fatal(err)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	conn, err = sql.Open("sqlite3", path+"?_foreign_keys=on")
	if err != nil {
		t.Fatal(err)
	}
	if err := RunMigrations(ctx, conn); err != nil {
		t.Fatal(err)
	}
	var name, pathOnDisk string
	if err := conn.QueryRow(`SELECT c.name,i.storage_path FROM collections c JOIN collection_items m ON m.collection_id=c.id JOIN images i ON i.id=m.image_id WHERE c.id=1`).Scan(&name, &pathOnDisk); err != nil {
		t.Fatal(err)
	}
	if name != "Saved" || pathOnDisk != "images/keep.jpg" {
		t.Fatalf("membership lost: %q %q", name, pathOnDisk)
	}
}
