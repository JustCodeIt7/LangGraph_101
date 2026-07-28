#!/usr/bin/env bash
# Run the local-only voice agent (mlx-whisper STT + Ollama LLM + macOS `say` TTS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_FILE="${1:-}"
OUTPUT_FILE="response.aiff"
MODEL_NAME="llama3.2"

echo "Local Voice Agent (mlx-whisper + Ollama) — Example Run"
echo "======================================================"

# Check for macOS (required for `say` TTS and MLX)
if [[ "$(uname)" != "Darwin" ]]; then
    echo "[error] This local agent requires macOS (uses Apple MLX + built-in 'say' command)."
    exit 1
fi
echo "[ok] Running on macOS"

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

# Determine input: audio file or record from mic
if [[ -n "$AUDIO_FILE" ]]; then
    if [[ ! -f "$AUDIO_FILE" ]]; then
        echo "[error] Audio file not found: $AUDIO_FILE"
        exit 1
    fi
    echo ""
    echo "Running local voice agent with audio: $AUDIO_FILE"
    python "${SCRIPT_DIR}/voice_agent_local.py" "$AUDIO_FILE" --model "$MODEL_NAME" --out "$OUTPUT_FILE"
else
    # Default to recording from mic for 5 seconds if no file provided
    RECORD_SECONDS="${RECORD_SECONDS:-5}"
    echo ""
    echo "No audio file provided — will record ${RECORD_SECONDS}s from microphone."
    python "${SCRIPT_DIR}/voice_agent_local.py" --record "$RECORD_SECONDS" --model "$MODEL_NAME" --out "$OUTPUT_FILE"
fi

echo ""
echo "[done] Response saved to: ${SCRIPT_DIR}/${OUTPUT_FILE}"
