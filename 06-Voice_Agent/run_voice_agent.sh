#!/usr/bin/env bash
# Run the local voice agent with a sample audio file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_FILE="${1:-}"

if [[ -z "$AUDIO_FILE" ]]; then
    echo "Usage: $0 <path_to_audio_file>"
    echo ""
    echo "Provide a path to an audio file (mp3/wav/m4a) for the voice agent."
    exit 1
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    echo "Usage: $0 <path_to_audio_file>"
    exit 1
fi

echo "Running voice agent with: $AUDIO_FILE"
python "${SCRIPT_DIR}/voice_agent_local.py" "$AUDIO_FILE" --model llama3.2