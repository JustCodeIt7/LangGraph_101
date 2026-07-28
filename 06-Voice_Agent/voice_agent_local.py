"""
Local-Only LangChain + LangGraph Voice Agent Example

A tutorial voice agent that runs entirely on-device — no cloud APIs:

  1. Speech-to-Text   → mlx-whisper (local Whisper via Apple MLX)
  2. LLM Reasoning    → Ollama model (e.g., llama3.2) via LangGraph ReAct agent
  3. Text-to-Speech   → macOS built-in `say` command

Usage:
    python voice_agent_local.py [--record SECONDS] <path_to_audio_file> [--model llama3.2]
    python voice_agent_local.py --record 5 --model llama3.2

Requirements (all local):
    pip install mlx-whisper langchain-ollama langgraph rich python-dotenv sounddevice numpy scipy
    - Ollama running locally with a model pulled (e.g., `ollama pull llama3.2`)
    - macOS for the built-in TTS (`say` command)

Learning objectives:
  - Build a fully offline voice agent using LangChain + LangGraph
  - Integrate mlx-whisper for on-device speech recognition
  - Use Ollama as a local LLM backend with create_react_agent tools
  - Leverage OS-native TTS instead of cloud services
"""

import argparse
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich.console import Console

load_dotenv()
console = Console()


# ── 1. Speech-to-Text: transcribe audio with mlx-whisper (local) ─────────────

def record_audio(duration: int, output_path: str = "recording.wav") -> str:
    """Record from the computer microphone using sounddevice."""
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write

    console.print(f"[cyan]Recording[/cyan] for {duration}s ... speak now.")
    sample_rate = 16000
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    write(output_path, sample_rate, recording)
    console.print(f"[green]Recording saved[/green] to {output_path}")
    return output_path


def transcribe_audio(audio_path: str, model_name: str = "mlx-community/whisper-tiny") -> str:
    """Transcribe speech to text using a local Whisper model via MLX."""
    console.print(f"[cyan]Transcribing[/cyan] {audio_path} with mlx-whisper ...")
    import mlx_whisper

    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_name)
    text = result["text"].strip()
    console.print(f"[green]User (voice):[/green] {text}")
    return text


# ── 2. Define tools for the agent ───────────────────────────────────────────

@tool
def get_weather(location: str) -> str:
    """Return a mock weather report for the given location."""
    return f"The weather in {location} is sunny and 72°F."


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ── 3. Build the LangGraph ReAct agent with a local Ollama LLM ───────────────

def build_agent(model_name: str = "ollama:llama3.2"):
    """Create a LangGraph ReAct agent backed by a local Ollama model."""
    from langchain_ollama import ChatOllama

    clean_name = model_name.removeprefix("ollama:")
    llm = ChatOllama(model=clean_name, temperature=0.3)
    tools = [get_weather, calculate]
    return create_react_agent(model=llm, tools=tools)


# ── 4. Text-to-Speech: use macOS built-in `say` command (local) ───────────────

def text_to_speech(text: str, output_path: str = "response.aiff") -> None:
    """Convert text to speech using the macOS built-in TTS engine."""
    console.print(f"[cyan]Generating[/cyan] speech → {output_path} ...")
    subprocess.run(["say", "-v", "Alex", "-o", output_path, text], check=True)
    console.print(f"[green]Speech saved[/green] to {output_path}")


# ── 5. Main voice loop ───────────────────────────────────────────────────────

def run_voice_agent(
    audio_path: str | None = None,
    record_seconds: int | None = None,
    llm_model: str = "ollama:llama3.2",
    whisper_model: str = "mlx-community/whisper-tiny",
    output_path: str = "response.aiff",
) -> None:
    """Run the full local voice agent pipeline: STT → Agent → TTS."""

    # Step 1 — Record or use existing audio, then transcribe (local Whisper via MLX)
    if record_seconds is not None:
        audio_path = record_audio(record_seconds)
    elif audio_path is None:
        console.print("[red]Error:[/red] Provide an audio file path or --record SECONDS")
        return

    user_text = transcribe_audio(audio_path, whisper_model)

    # Step 2 — Run through the LangGraph ReAct agent with tools
    console.print(f"[cyan]Running[/cyan] LangGraph ReAct agent ({llm_model}) ...")
    agent = build_agent(llm_model)
    result = agent.invoke({"messages": [("user", user_text)]})
    response_text = result["messages"][-1].content

    if not isinstance(response_text, str):
        response_text = str(response_text)

    console.print(f"[green]Agent:[/green] {response_text}")

    # Step 3 — Text to speech (macOS built-in `say`)
    text_to_speech(response_text, output_path)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local-Only LangChain + LangGraph Voice Agent"
    )
    parser.add_argument("audio", nargs="?", default=None, help="Path to input audio file (mp3/wav/m4a)")
    parser.add_argument("--record", type=int, metavar="SECONDS", help="Record from mic for N seconds")
    parser.add_argument(
        "--model", default="ollama:llama3.2",
        help="Ollama model name (default: ollama:llama3.2)",
    )
    parser.add_argument(
        "--out", default="response.aiff",
        help="Output speech file path (default: response.aiff)",
    )
    args = parser.parse_args()

    if not args.audio and args.record is None:
        console.print("[red]Error:[/red] Provide an audio file or use --record SECONDS")
        return

    if args.audio and not Path(args.audio).exists():
        console.print(f"[red]Error:[/red] Audio file not found: {args.audio}")
        return

    run_voice_agent(
        audio_path=args.audio,
        record_seconds=args.record,
        llm_model=args.model,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
