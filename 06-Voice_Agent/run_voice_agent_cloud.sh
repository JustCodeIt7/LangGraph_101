#!/usr/bin/env bash
# Run the cloud-based voice agent (OpenAI Whisper STT + ChatGPT LLM + OpenAI TTS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_FILE="${1:-${SCRIPT_DIR}/recording.wav}"
OUTPUT_FILE="${2:-response.mp3}"

echo "Cloud Voice Agent (OpenAI APIs) — Example Run"
echo "============================================="

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

# Validate audio file
if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "[error] Audio file not found: $AUDIO_FILE"
    echo "Usage: $0 [audio_file] [output_file]"
    exit 1
fi
echo "[ok] Input audio: $AUDIO_FILE"

# Run the cloud voice agent
echo ""
echo "Running cloud voice agent ..."
python "${SCRIPT_DIR}/voice_agent.py" "$AUDIO_FILE" --out "$OUTPUT_FILE"

echo ""
echo "[done] Response saved to: ${SCRIPT_DIR}/${OUTPUT_FILE}"
