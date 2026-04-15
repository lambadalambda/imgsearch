package httputil

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestWriteJSONSetsContentTypeAndStatus(t *testing.T) {
	rr := httptest.NewRecorder()
	WriteJSON(rr, http.StatusCreated, map[string]string{"ok": "yes"})

	if rr.Code != http.StatusCreated {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusCreated)
	}
	if rr.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("content-type: got=%q want=%q", rr.Header().Get("Content-Type"), "application/json")
	}
	var payload map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if payload["ok"] != "yes" {
		t.Fatalf("unexpected payload: %+v", payload)
	}
}

func TestWriteJSONErrorWrapsMessage(t *testing.T) {
	rr := httptest.NewRecorder()
	WriteJSONError(rr, http.StatusBadRequest, "bad request")

	var payload map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if payload["error"] != "bad request" {
		t.Fatalf("unexpected payload: %+v", payload)
	}
}
