package upstream

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMockClient_ReturnsPresetResponse(t *testing.T) {
	expectedResp := &http.Response{StatusCode: http.StatusOK}
	mock := &MockClient{Response: expectedResp}

	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	resp, err := mock.Do(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != expectedResp {
		t.Errorf("expected response %v, got %v", expectedResp, resp)
	}
}

func TestMockClient_RecordsRequests(t *testing.T) {
	mock := &MockClient{Response: &http.Response{StatusCode: http.StatusOK}}

	req1 := httptest.NewRequest(http.MethodPost, "/test1", nil)
	req2 := httptest.NewRequest(http.MethodGet, "/test2", nil)

	mock.Do(context.Background(), req1)
	mock.Do(context.Background(), req2)

	if len(mock.Requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(mock.Requests))
	}
	if mock.Requests[0] != req1 {
		t.Errorf("expected first request %v, got %v", req1, mock.Requests[0])
	}
	if mock.Requests[1] != req2 {
		t.Errorf("expected second request %v, got %v", req2, mock.Requests[1])
	}
}

func TestMockClient_ReturnsPresetError(t *testing.T) {
	expectedErr := errors.New("connection failed")
	mock := &MockClient{Error: expectedErr}

	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	resp, err := mock.Do(context.Background(), req)

	if err != expectedErr {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
	if resp != nil {
		t.Errorf("expected nil response, got %v", resp)
	}
}

func TestMockClient_Close(t *testing.T) {
	mock := &MockClient{}

	mock.Close()
}

func TestMockClient_CloseIsNoOp(t *testing.T) {
	mock := &MockClient{
		Response: &http.Response{StatusCode: http.StatusOK},
	}

	mock.Close()
	mock.Close()

	mock.Close()
}

func TestMockClient_MultipleRequests(t *testing.T) {
	mock := &MockClient{Response: &http.Response{StatusCode: http.StatusCreated}}

	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodPost, "/test", nil)
		resp, err := mock.Do(context.Background(), req)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.StatusCode != http.StatusCreated {
			t.Errorf("expected status 201, got %d", resp.StatusCode)
		}
	}

	if len(mock.Requests) != 5 {
		t.Errorf("expected 5 requests, got %d", len(mock.Requests))
	}
}

func TestMockClient_NilResponse(t *testing.T) {
	mock := &MockClient{}

	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	resp, err := mock.Do(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != nil {
		t.Errorf("expected nil response, got %v", resp)
	}
	if len(mock.Requests) != 1 {
		t.Errorf("expected 1 request, got %d", len(mock.Requests))
	}
}
