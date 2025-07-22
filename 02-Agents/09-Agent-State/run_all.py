#!/usr/bin/env python3
"""
Interactive runner for LangGraph Agent State Examples

This script provides a menu-driven interface to run any of the agent state examples.
"""

import sys
import subprocess
import os

def print_banner():
    """Prints the main banner for the script."""
    print("=" * 80)
    print("🤖 LangGraph Agent State Examples Runner")
    print("=" * 80)
    print("Choose from three examples demonstrating different state management patterns:")
    print()

def print_menu():
    """Prints the menu of available examples."""
    print("📋 Available Examples:")
    print()
    print("1. Basic Agent State (`09.1-basic_agent_state.py`)")
    print("   - Manages a simple sequence of messages.")
    print("   - Demonstrates the most fundamental state pattern.")
    print()
    print("2. Task-Tracking Agent State (`09.2-task_agent_state.py`)")
    print("   - Implements task tracking with retries and completion status.")
    print("   - Uses conditional edges for looping logic.")
    print()
    print("3. Complex Agent State (`09.3-complex_agent_state.py`)")
    print("   - Handles nested data structures for subtasks.")
    print("   - Shows how to manage more complex, structured state.")
    print()
    print("0. ❌ Exit")
    print()

def run_example(choice):
    """Run the selected example script."""
    scripts = {
        "1": "09.1-basic_agent_state.py",
        "2": "09.2-task_agent_state.py",
        "3": "09.3-complex_agent_state.py",
    }
    
    if choice not in scripts:
        print("❌ Invalid choice. Please select 1-3 or 0 to exit.")
        return
    
    script_name = scripts[choice]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Script {script_name} not found at {script_path}.")
        print("Make sure all example scripts are in the same directory as this runner.")
        return
    
    print(f"\n🚀 Running {script_name}...")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, script_path], check=True, cwd=current_dir)
        print("-" * 60)
        print(f"✅ {script_name} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
    except KeyboardInterrupt:
        print(f"\n⚠️ Execution of {script_name} was interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

def main():
    """Main interactive loop."""
    print_banner()
    
    while True:
        print_menu()
        try:
            choice = input("Enter your choice (0-3): ").strip()
            if choice == "0":
                print("\n👋 Thanks for exploring the LangGraph Agent State examples!")
                break
            elif choice in ["1", "2", "3"]:
                run_example(choice)
                input("\nPress Enter to return to the menu...")
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 3.")
            print("\n" + "=" * 80)
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()