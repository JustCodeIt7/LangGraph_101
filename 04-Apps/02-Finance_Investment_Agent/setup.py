#!/usr/bin/env python3
"""
Finance & Investment Agent Setup Script
======================================

This script helps set up the Finance & Investment Agent by:
1. Checking and installing dependencies
2. Verifying Ollama installation
3. Pulling the required llama3.2 model
4. Running basic tests
"""

import os
import sys
import subprocess
import platform

def run_command(command, description="", check=True):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ Failed (exit code: {result.returncode})")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("\n🧪 Checking Python Version")
    print("-" * 30)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Dependencies")
    print("-" * 28)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found in current directory")
        return False
    
    # Install dependencies
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def check_ollama_installation():
    """Check if Ollama is installed."""
    print("\n🤖 Checking Ollama Installation")
    print("-" * 35)
    
    # Check if ollama command exists
    result = run_command("ollama --version", "Checking Ollama version", check=False)
    
    if not result:
        print("\n💡 Ollama is not installed. Install it from: https://ollama.ai")
        print("   For macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    
    return True

def start_ollama_service():
    """Start Ollama service if not running."""
    print("\n🔄 Starting Ollama Service")
    print("-" * 29)
    
    # Check if Ollama is already running
    check_result = run_command(
        "curl -s http://localhost:11434/api/tags > /dev/null 2>&1",
        "Checking if Ollama is running",
        check=False
    )
    
    if check_result:
        print("   ✅ Ollama is already running")
        return True
    
    # Try to start Ollama
    print("   Starting Ollama service...")
    
    system = platform.system().lower()
    if system == "darwin":  # macOS
        result = run_command(
            "ollama serve &",
            "Starting Ollama service (macOS)",
            check=False
        )
    elif system == "linux":
        result = run_command(
            "ollama serve &",
            "Starting Ollama service (Linux)",
            check=False
        )
    else:  # Windows
        print("   Please start Ollama manually on Windows")
        return False
    
    # Wait a moment and check again
    import time
    time.sleep(3)
    
    return run_command(
        "curl -s http://localhost:11434/api/tags > /dev/null 2>&1",
        "Verifying Ollama is running",
        check=False
    )

def pull_llama_model():
    """Pull the llama3.2 model."""
    print("\n📥 Pulling llama3.2 Model")
    print("-" * 27)
    
    # Check if model already exists
    check_result = run_command(
        "ollama list | grep llama3.2",
        "Checking if llama3.2 is already available",
        check=False
    )
    
    if check_result:
        print("   ✅ llama3.2 model is already available")
        return True
    
    # Pull the model
    print("   This may take several minutes depending on your internet connection...")
    return run_command(
        "ollama pull llama3.2",
        "Pulling llama3.2 model"
    )

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running Tests")
    print("-" * 17)
    
    if not os.path.exists("test_agent.py"):
        print("❌ test_agent.py not found")
        return False
    
    return run_command(
        f"{sys.executable} test_agent.py",
        "Running agent test suite"
    )

def create_sample_data():
    """Create a sample data file for testing."""
    print("\n📊 Creating Sample Data")
    print("-" * 25)
    
    sample_data = """# Sample Finance Data for Testing

## Sample Expenses
- Groceries: $45.50 (food)
- Gas: $25.00 (transport)
- Movie tickets: $30.00 (entertainment)
- Coffee: $5.75 (food)

## Sample Budget
- Food: $500/month
- Transportation: $300/month
- Entertainment: $200/month
- Utilities: $150/month

## Sample Portfolio
- AAPL: 10 shares
- GOOGL: 5 shares
- TSLA: 3 shares
- Cash: $2,500
"""
    
    try:
        with open("sample_data.md", "w") as f:
            f.write(sample_data)
        print("✅ Created sample_data.md")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def main():
    """Main setup function."""
    print("🏦 Finance & Investment Agent Setup")
    print("=" * 45)
    print("This script will help you set up the Finance Agent")
    print("by installing dependencies and configuring Ollama.")
    print("=" * 45)
    
    # Track success of each step
    steps = []
    
    # Step 1: Check Python version
    steps.append(("Python Version Check", check_python_version))
    
    # Step 2: Install dependencies
    steps.append(("Install Dependencies", install_dependencies))
    
    # Step 3: Check Ollama
    steps.append(("Check Ollama", check_ollama_installation))
    
    # Step 4: Start Ollama service
    steps.append(("Start Ollama Service", start_ollama_service))
    
    # Step 5: Pull model
    steps.append(("Pull llama3.2 Model", pull_llama_model))
    
    # Step 6: Create sample data
    steps.append(("Create Sample Data", create_sample_data))
    
    # Step 7: Run tests
    steps.append(("Run Tests", run_tests))
    
    # Execute all steps
    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except KeyboardInterrupt:
            print(f"\n⚠️ Setup interrupted by user during: {step_name}")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            results[step_name] = False
    
    # Print summary
    print("\n" + "=" * 45)
    print("📊 SETUP SUMMARY")
    print("=" * 45)
    
    successful_steps = sum(1 for result in results.values() if result)
    total_steps = len(results)
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} - {step_name}")
    
    print(f"\n📈 Setup Results: {successful_steps}/{total_steps} steps completed")
    
    if successful_steps == total_steps:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("  • Test the agent: python test_agent.py")
        print("  • Run demo: python demo_finance_agent.py")
        print("  • Start interactive mode: python finance_investment_agent.py")
    else:
        print("\n⚠️ Setup completed with some issues.")
        print("Please review the failed steps above.")
        
        # Provide specific help for common issues
        if not results.get("Check Ollama", True):
            print("\n💡 Ollama Help:")
            print("  • Install from: https://ollama.ai")
            print("  • macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        
        if not results.get("Install Dependencies", True):
            print("\n💡 Dependencies Help:")
            print("  • Try: pip install --upgrade pip")
            print("  • Then: pip install -r requirements.txt")
    
    return successful_steps == total_steps

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user. Goodbye!")
        sys.exit(1)