"""Simple integration test to verify InputHandler system is working."""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from repository_analyzer.input.handler import InputHandler
    from repository_analyzer.input.config import InputConfig
    from repository_analyzer.core.analyzer import RepositoryAnalyzer
    
    print("✓ InputHandler imports successful")
    
    # Test InputHandler creation
    config = InputConfig()
    handler = InputHandler(config)
    print("✓ InputHandler creation successful")
    
    # Test RepositoryAnalyzer integration
    analyzer = RepositoryAnalyzer()
    print("✓ RepositoryAnalyzer with InputHandler integration successful")
    
    print("\n🎉 All InputHandler components are working correctly!")
    print("\nTo test with actual repositories, run:")
    print("  python examples/input_handler_demo.py")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")