"""
Basic LangChain + LangGraph Voice Agent Example

A tutorial voice agent that:
  1. Transcribes speech to text (OpenAI Whisper API)
  2. Runs the transcribed query through a LangGraph ReAct agent with tools
  3. Converts the agent's response back to speech (OpenAI TTS API)

Usage:
    python voice_agent.py <path_to_audio_file> [--out output.mp3]

Requirements:
    pip install langchain-openai openai langgraph rich python-dotenv
    Set OPENAI_API_KEY in your environment or .env file.

Learning objectives:
  - Wire OpenAI audio APIs (Whisper STT + TTS) around a LangGraph agent
  - Use create_react_agent with custom tools for voice-driven queries
  - Keep the full voice loop in one readable file (<200 lines)
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from openai import OpenAI
from rich.console import Console

load_dotenv()

console = Console()


# ── 1. Speech-to-Text: transcribe an audio file with Whisper ────────────────

def transcribe_audio(audio_path: str, client: OpenAI) -> str:
    """Transcribe speech to text using the OpenAI Whisper API."""
    console.print(f"[cyan]Transcribing[/cyan] {audio_path} ...")
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f
        )
    text = transcript.text.strip()
    console.print(f"[green]User (voice):[/green] {text}")
    return text


# ── 2. Define tools for the agent ───────────────────────────────────────────

@tool
def get_weather(location: str) -> str:
    """Return a mock weather report for the given location."""
    # In production, call a real weather API (e.g., OpenWeatherMap).
    return f"The weather in {location} is sunny and 72°F."


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ── 3. Build the LangGraph ReAct agent ───────────────────────────────────────

def build_agent(client: OpenAI):
    """Create a LangGraph ReAct agent with tools, backed by ChatOpenAI."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [get_weather, calculate]
    return create_react_agent(model=llm, tools=tools)


# ── 4. Text-to-Speech: convert text to speech with TTS ───────────────────────

def text_to_speech(text: str, output_path: str, client: OpenAI) -> None:
    """Convert text to speech using the OpenAI TTS API."""
    console.print(f"[cyan]Generating[/cyan] speech → {output_path} ...")
    response = client.audio.speech.create(
        model="tts-1", voice="nova", input=text
    )
    response.stream_to_file(output_path)
    console.print(f"[green]Speech saved[/green] to {output_path}")


# ── 5. Main voice loop ───────────────────────────────────────────────────────

def run_voice_agent(audio_path: str, output_path: str = "response.mp3") -> None:
    """Run the full voice agent pipeline: STT → Agent → TTS."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is required.")
        return

    client = OpenAI(api_key=api_key)

    # Step 1 — Speech to text
    user_text = transcribe_audio(audio_path, client)

    # Step 2 — Run through the LangGraph agent
    console.print("[cyan]Running[/cyan] LangGraph ReAct agent ...")
    agent = build_agent(client)
    result = agent.invoke({"messages": [("user", user_text)]})
    response_text = result["messages"][-1].content

    # Strip any tool-call artifacts for clean TTS output
    if not isinstance(response_text, str):
        response_text = str(response_text)

    console.print(f"[green]Agent:[/green] {response_text}")

    # Step 3 — Text to speech
    text_to_speech(response_text, output_path, client)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangChain + LangGraph Voice Agent"
    )
    parser.add_argument("audio", help="Path to input audio file (mp3/wav/m4a)")
    parser.add_argument(
        "--out", default="response.mp3", help="Output speech file path"
    )
    args = parser.parse_args()

    if not Path(args.audio).exists():
        console.print(f"[red]Error:[/red] Audio file not found: {args.audio}")
        return

    run_voice_agent(args.audio, args.out)


if __name__ == "__main__":
    main()
