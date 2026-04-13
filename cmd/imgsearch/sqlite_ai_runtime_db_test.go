package main

import "testing"

func TestOpenSQLiteAIRuntimeDBRequiresExtensionPath(t *testing.T) {
	dbConn, err := openSQLiteAIRuntimeDB("   ")
	if err == nil {
		if dbConn != nil {
			_ = dbConn.Close()
		}
		t.Fatal("expected missing sqlite-ai path error")
	}
}
