#!/usr/bin/env python3
"""
Simple runner script for the Human-in-the-Loop tutorial.
This script provides an easy way to run the tutorial with proper error handling.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import langgraph
        import langchain_openai
        import langchain_core
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set."""
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key is set")
        return True
    else:
        print("❌ OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False

def main():
    """Main runner function."""
    print("🚀 LangGraph Human-in-the-Loop Tutorial Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies, run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check API key
    if not check_api_key():
        print("\n💡 To set your API key, run:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("\n🎯 Starting Human-in-the-Loop tutorial...")
    print("=" * 50)
    
    # Import and run the tutorial
    try:
        from pathlib import Path
        tutorial_path = Path(__file__).parent / "04-Human-in-the-Loop.py"
        
        # Execute the tutorial script
        subprocess.run([sys.executable, str(tutorial_path)], check=True)
        
    except FileNotFoundError:
        print("❌ Tutorial file not found: 04-Human-in-the-Loop.py")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tutorial: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Tutorial interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()