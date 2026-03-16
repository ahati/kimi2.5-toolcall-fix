#!/bin/bash
# Integration tests for all 6 protocol combinations

BASE_URL="http://localhost:8085"
PASS=0
FAIL=0

test_case() {
    local name="$1"
    local expected_status="$2"
    local actual_status="$3"
    local body="$4"

    if [ "$actual_status" -eq "$expected_status" ]; then
        echo "✅ PASS: $name (status $actual_status)"
        PASS=$((PASS+1))
    else
        echo "❌ FAIL: $name (expected $expected_status, got $actual_status)"
        echo "Response: $body"
        FAIL=$((FAIL+1))
    fi
}

echo "=========================================="
echo "Integration Tests - 6 Protocol Combinations"
echo "=========================================="
echo ""

# Test 1: OpenAI Chat → OpenAI Chat (pass-through)
# /v1/chat/completions with kimi-k2.5 model routes to chutes (OpenAI)
echo "Test 1: OpenAI Chat → OpenAI Chat (kimi-k2.5 → chutes)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50,
    "stream": false
  }')
BODY=$(echo "$RESPONSE" | head -n -1)
STATUS=$(echo "$RESPONSE" | tail -n 1)
test_case "OpenAI Chat → OpenAI Chat" 200 "$STATUS" "$BODY"
echo ""

# Test 2: Anthropic → Anthropic (pass-through)
# /v1/messages with glm-5 model routes to alibaba (Anthropic)
echo "Test 2: Anthropic → Anthropic (glm-5 → alibaba)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/messages" \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "glm-5",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }')
BODY=$(echo "$RESPONSE" | head -n -1)
STATUS=$(echo "$RESPONSE" | tail -n 1)
test_case "Anthropic → Anthropic" 200 "$STATUS" "$BODY"
echo ""

# Test 3: Anthropic → OpenAI Chat (bridge)
# /v1/openai-to-anthropic/messages converts Anthropic input to OpenAI upstream
echo "Test 3: Anthropic → OpenAI Chat (bridge to chutes)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/openai-to-anthropic/messages" \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }')
BODY=$(echo "$RESPONSE" | head -n -1)
STATUS=$(echo "$RESPONSE" | tail -n 1)
test_case "Anthropic → OpenAI Chat (bridge)" 200 "$STATUS" "$BODY"
echo ""

# Test 4: OpenAI Responses → Anthropic
# /v1/anthropic-to-openai/responses converts Responses API to Anthropic upstream
echo "Test 4: OpenAI Responses → Anthropic (glm-5 → alibaba)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/anthropic-to-openai/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "input": "Say hello",
    "max_output_tokens": 50,
    "stream": false
  }')
BODY=$(echo "$RESPONSE" | head -n -1)
STATUS=$(echo "$RESPONSE" | tail -n 1)
test_case "OpenAI Responses → Anthropic" 200 "$STATUS" "$BODY"
echo ""

# Test 5: OpenAI Responses → OpenAI Chat (via /v1/responses)
# /v1/responses with kimi-k2.5 routes to chutes (OpenAI)
echo "Test 5: OpenAI Responses → OpenAI Chat (kimi-k2.5 → chutes)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "input": "Say hello",
    "max_output_tokens": 50,
    "stream": false
  }')
BODY=$(echo "$RESPONSE" | head -n -1)
STATUS=$(echo "$RESPONSE" | tail -n 1)
test_case "OpenAI Responses → OpenAI Chat" 200 "$STATUS" "$BODY"
echo ""

# Test 6: OpenAI Responses → Anthropic (via /v1/responses)
# /v1/responses with glm-5 routes to alibaba (Anthropic)
echo "Test 6: OpenAI Responses → Anthropic (glm-5 → alibaba)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "input": "Say hello",
    "max_output_tokens": 50,
    "stream": false
  }')
BODY=$(echo "$RESPONSE" | head -n -1)
STATUS=$(echo "$RESPONSE" | tail -n 1)
test_case "OpenAI Responses → Anthropic (via /v1/responses)" 200 "$STATUS" "$BODY"
echo ""

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

# Exit with error if any tests failed
if [ $FAIL -gt 0 ]; then
    exit 1
fi