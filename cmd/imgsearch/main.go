package main

import (
    "context"
    "database/sql"
    "flag"
    "log"
    "os"
    "path/filepath"

    _ "github.com/mattn/go-sqlite3"

    "imgsearch/internal/app"
    "imgsearch/internal/vectorindex/sqlitevector"
)

func main() {
    dataDir := flag.String("data-dir", "./data", "data directory")
    flag.Parse()

    if err := os.MkdirAll(*dataDir, 0o755); err != nil {
        log.Fatalf("create data directory: %v", err)
    }

    dbPath := filepath.Join(*dataDir, "imgsearch.sqlite")
    sqlDB, err := sql.Open("sqlite3", dbPath)
    if err != nil {
        log.Fatalf("open sqlite database: %v", err)
    }
    defer func() { _ = sqlDB.Close() }()

    if err := app.Bootstrap(context.Background(), sqlDB, sqlitevector.ValidateAvailable); err != nil {
        log.Fatalf("bootstrap app: %v", err)
    }

    log.Printf("imgsearch initialized with database %s", dbPath)
}
