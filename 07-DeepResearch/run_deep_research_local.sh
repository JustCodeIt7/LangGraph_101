#!/usr/bin/env bash
# Run the local-only deep research agent (Ollama + DuckDuckGo search).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUERY="${1:-}"
MODEL_NAME="llama3.2"

echo "Deep Research Agent (Local Ollama) — Example Run"
echo "================================================"

# Check if Ollama is running and the model is available
echo ""
echo "Checking Ollama ..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "[error] Ollama is not running. Start it with: ollama serve"
    exit 1
fi

if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Model '$MODEL_NAME' not found — pulling now ..."
    ollama pull "$MODEL_NAME"
fi
echo "[ok] Ollama model '$MODEL_NAME' is available"

# Validate query argument
if [[ -z "$QUERY" ]]; then
    echo ""
    echo "Usage: $0 \"<research question>\""
    echo ""
    echo "Example research questions:"
    echo "  \"What are the latest developments in LangGraph as of 2025?\""
    echo "  \"Compare the advantages and disadvantages of vector databases\""
    echo "  \"How does reinforcement learning differ from supervised learning?\""
    exit 1
fi

echo ""
echo "[ok] Research query: $QUERY"
echo ""

# Run the local research agent
python "${SCRIPT_DIR}/deep_research_agent_local.py" "$QUERY" --model "$MODEL_NAME"
