#!/usr/bin/env python3
"""
HIL-gem Demo Runner
===================

This script provides an easy way to run all Human-in-the-Loop examples
in the HIL-gem directory. You can run individual examples or all at once.

Usage:
    python run_demos.py                    # Interactive menu
    python run_demos.py --all             # Run all demos
    python run_demos.py --example 1       # Run specific example
    python run_demos.py --list            # List available examples
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Main entry point for the demo runner"""
    parser = argparse.ArgumentParser(description="Run HIL-gem examples")
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all examples sequentially"
    )
    parser.add_argument(
        "--example", 
        type=int, 
        choices=[1, 2, 3], 
        help="Run specific example (1, 2, or 3)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available examples"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: {
            "file": "example1_basic_approval.py",
            "name": "Basic Approval Pattern",
            "description": "Simple yes/no approval workflow for sensitive actions"
        },
        2: {
            "file": "example2_content_editing.py", 
            "name": "Content Review and Editing Pattern",
            "description": "Iterative content creation with human review and editing"
        },
        3: {
            "file": "example3_decision_tree.py",
            "name": "Multi-Step Decision Tree Pattern", 
            "description": "Complex workflows with multiple decision points and branching"
        }
    }
    
    if args.list:
        print("Available HIL-gem Examples:")
        print("=" * 50)
        for num, info in examples.items():
            print(f"{num}. {info['name']}")
            print(f"   File: {info['file']}")
            print(f"   Description: {info['description']}")
            print()
        return
    
    if args.all:
        print("🚀 Running all HIL-gem examples sequentially...")
        print("=" * 60)
        for num in sorted(examples.keys()):
            run_example(num, examples[num])
            if num < max(examples.keys()):
                input("\nPress Enter to continue to next example...")
                print("\n" + "=" * 60)
        return
    
    if args.example:
        run_example(args.example, examples[args.example])
        return
    
    # Interactive menu
    while True:
        print("\n🎯 HIL-gem Examples - Interactive Menu")
        print("=" * 50)
        print("Choose an example to run:")
        
        for num, info in examples.items():
            print(f"{num}. {info['name']}")
        
        print("4. Run all examples")
        print("0. Exit")
        
        try:
            choice = int(input("\nEnter your choice (0-4): "))
            
            if choice == 0:
                print("👋 Goodbye!")
                break
            elif choice == 4:
                for num in sorted(examples.keys()):
                    run_example(num, examples[num])
                    if num < max(examples.keys()):
                        input("\nPress Enter to continue...")
            elif choice in examples:
                run_example(choice, examples[choice])
            else:
                print("❌ Invalid choice. Please try again.")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break


def run_example(num: int, info: dict):
    """Run a specific example"""
    print(f"\n🚀 Running Example {num}: {info['name']}")
    print("=" * 60)
    print(f"Description: {info['description']}")
    print(f"File: {info['file']}")
    print("=" * 60)
    
    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent
        example_path = script_dir / info['file']
        
        if not example_path.exists():
            print(f"❌ Error: {info['file']} not found!")
            return
        
        # Run the example
        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=script_dir,
            capture_output=False  # Show output in real-time
        )
        
        if result.returncode == 0:
            print(f"\n✅ Example {num} completed successfully!")
        else:
            print(f"\n❌ Example {num} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running example {num}: {e}")


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['langgraph', 'langchain-core']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


if __name__ == "__main__":
    print("🔄 HIL-gem Demo Runner")
    print("Human-in-the-Loop Examples for LangGraph")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo runner interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
