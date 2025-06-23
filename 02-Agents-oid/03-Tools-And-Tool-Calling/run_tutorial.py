#!/usr/bin/env python3
"""
Quick Start Script for LangGraph Tools and Tool Calling Tutorial

This script provides a simplified way to run the tutorial with
environment checks and helpful guidance.
"""

import os
import sys
import subprocess

def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking environment setup...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check for required packages
    required_packages = ['langgraph', 'langchain_core', 'langchain_openai']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not found")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", *missing_packages
            ])
            print("✅ Packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            print("Please run: pip install langgraph langchain-core langchain-openai")
            return False
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not found in environment")
        api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ API key set for this session")
        else:
            print("❌ No API key provided - the tutorial may not work")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            return False
    else:
        print("✅ OPENAI_API_KEY found")
    
    return True

def main():
    """Main function to run the tutorial."""
    print("=" * 60)
    print("LangGraph Tools and Tool Calling Tutorial - Quick Start")
    print("=" * 60)
    
    if not check_environment():
        print("\n❌ Environment setup failed. Please fix the issues above.")
        return
    
    print("\n🚀 Environment ready! Starting tutorial...")
    print("-" * 40)
    
    # Import and run the main tutorial
    try:
        from pathlib import Path
        tutorial_path = Path(__file__).parent / "03-Tools-And-Tool-Calling.py"
        
        if tutorial_path.exists():
            # Execute the tutorial script
            exec(open(tutorial_path).read())
        else:
            print("❌ Tutorial script not found")
            print("Make sure '03-Tools-And-Tool-Calling.py' is in the same directory")
    
    except Exception as e:
        print(f"❌ Error running tutorial: {e}")
        print("\nTry running the tutorial directly:")
        print("python 03-Tools-And-Tool-Calling.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Tutorial interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please report this issue or try running the tutorial directly.")
