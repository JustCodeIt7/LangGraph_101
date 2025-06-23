#!/usr/bin/env python3
"""
Interactive runner for LangGraph Advanced Agent Examples

This script provides a menu-driven interface to run any of the advanced agent examples.
"""

import sys
import subprocess
import os

def print_banner():
    print("=" * 80)
    print("🤖 LangGraph Advanced Agent Examples Tutorial v2")
    print("=" * 80)
    print("Choose from four sophisticated agent architectures:")
    print()

def print_menu():
    print("📋 Available Examples:")
    print()
    print("1. 🔬 Research Assistant Agent (Original)")
    print("   - Multi-step planning and execution")
    print("   - Self-reflection and iterative improvement")
    print("   - Complex state management with looping")
    print()
    print("2. 🤝 Collaborative Multi-Agent System")
    print("   - Multiple specialized agents working together")
    print("   - Coordinator, Analyst, Researcher, and Synthesizer agents")
    print("   - Market analysis with cross-agent collaboration")
    print()
    print("3. 🔄 Adaptive Planning Agent")
    print("   - Dynamic plan creation and modification")
    print("   - Real-time adaptation to blockers and issues")
    print("   - Learning from execution patterns")
    print()
    print("4. 🏗️ Hierarchical Reasoning Agent")
    print("   - Multi-level problem decomposition")
    print("   - Strategic → Tactical → Operational reasoning")
    print("   - Business challenge analysis framework")
    print()
    print("0. ❌ Exit")
    print()

def run_example(choice):
    """Run the selected example script."""
    scripts = {
        "1": "05-Advanced-Agent-Examples.py",
        "2": "example1_collaborative_agents.py", 
        "3": "example2_adaptive_planning.py",
        "4": "example3_hierarchical_reasoning.py"
    }
    
    if choice not in scripts:
        print("❌ Invalid choice. Please select 1-4 or 0 to exit.")
        return False
    
    script_name = scripts[choice]
    
    # Check if script exists
    if not os.path.exists(script_name):
        print(f"❌ Script {script_name} not found in current directory.")
        print("Make sure you're running this from the correct directory.")
        return False
    
    print(f"\n🚀 Starting {script_name}...")
    print("=" * 60)
    
    try:
        # Run the selected script
        result = subprocess.run([sys.executable, script_name], check=True)
        print("\n" + "=" * 60)
        print("✅ Example completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Execution of {script_name} was interrupted by user.")
        return True
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ["langgraph", "langchain_openai", "langchain_core"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("Install with: pip install " + " ".join(missing_packages))
        print()
        return False
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OpenAI API key not found in environment variables.")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        print()
        return False
    
    return True

def main():
    """Main interactive loop."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("Please install missing requirements and try again.")
        sys.exit(1)
    
    print("✅ All requirements satisfied!")
    print()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == "0":
                print("\n👋 Thanks for exploring LangGraph Advanced Agent Examples!")
                print("Check out the README.md for more details on each example.")
                break
            elif choice in ["1", "2", "3", "4"]:
                success = run_example(choice)
                if success:
                    print("\n🔄 Returning to main menu...")
                    input("Press Enter to continue...")
                else:
                    print("\n❓ Would you like to try another example?")
                    continue_choice = input("Continue? (y/n): ").strip().lower()
                    if continue_choice not in ["y", "yes"]:
                        break
            else:
                print("❌ Invalid choice. Please enter a number between 0-4.")
                
            print("\n" + "=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()