#!/usr/bin/env python3
"""
Setup script for Content Creation Pipeline Agent
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return None

def check_ollama():
    """Check if Ollama is running and llama3.2 is available"""
    print("🔍 Checking Ollama status...")
    
    # Check if Ollama is running
    result = run_command("curl -s http://localhost:11434/api/tags", "Checking Ollama service")
    if not result:
        print("⚠️  Ollama doesn't seem to be running.")
        print("Please start Ollama first: ollama serve")
        return False
    
    # Check if llama3.2 model is available
    result = run_command("ollama list | grep llama3.2", "Checking for llama3.2 model")
    if not result:
        print("📥 llama3.2 model not found. Pulling it now...")
        pull_result = run_command("ollama pull llama3.2", "Pulling llama3.2 model")
        if not pull_result:
            print("❌ Failed to pull llama3.2 model")
            return False
    
    print("✅ Ollama and llama3.2 are ready!")
    return True

def install_dependencies():
    """Install Python dependencies"""
    run_command("pip install -r requirements.txt", "Installing Python dependencies")

def main():
    print("🚀 Setting up Content Creation Pipeline Agent")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Check Ollama setup
    ollama_ready = check_ollama()
    
    print("\n" + "=" * 50)
    if ollama_ready:
        print("🎉 Setup complete! You can now run the agent:")
        print("   python app.py")
    else:
        print("⚠️  Setup incomplete. Please:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Start Ollama: ollama serve")
        print("   3. Run this setup again: python setup.py")

if __name__ == "__main__":
    main()