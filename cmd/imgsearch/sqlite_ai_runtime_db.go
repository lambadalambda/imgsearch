package main

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"
)

func openSQLiteAIRuntimeDB(sqliteAIPath string) (*sql.DB, error) {
	path := strings.TrimSpace(sqliteAIPath)
	if path == "" {
		return nil, fmt.Errorf("sqlite-ai runtime db requires extension path")
	}

	runtimeDB, err := openSQLiteDB("file:imgsearch-sqlite-ai-runtime?mode=memory&cache=private", "", path)
	if err != nil {
		return nil, err
	}
	runtimeDB.SetMaxOpenConns(1)
	runtimeDB.SetMaxIdleConns(1)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := runtimeDB.PingContext(ctx); err != nil {
		_ = runtimeDB.Close()
		return nil, err
	}

	return runtimeDB, nil
}
