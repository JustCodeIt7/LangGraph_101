#!/usr/bin/env bash
# Run the cloud-based deep research agent (OpenAI + DuckDuckGo search).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUERY="${1:-}"

echo "Deep Research Agent (OpenAI) — Example Run"
echo "==========================================="

# Check for OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
        echo "Loading API key from .env ..."
        set -a; source "${SCRIPT_DIR}/../.env"; set +a
    fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[error] OPENAI_API_KEY is not set."
    echo "Set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi
echo "[ok] OPENAI_API_KEY found"

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

# Run the cloud research agent
python "${SCRIPT_DIR}/deep_research_agent.py" "$QUERY"
