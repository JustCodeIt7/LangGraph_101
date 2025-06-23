#!/usr/bin/env python3
"""
Test script to verify RAG Agent setup and dependencies
"""

import sys
import subprocess
import importlib.util
import requests
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible (requires Python 3.8+)")
        return False

def test_package_import(package_name, friendly_name=None):
    """Test if a package can be imported"""
    if friendly_name is None:
        friendly_name = package_name
    
    try:
        __import__(package_name)
        print(f"✅ {friendly_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {friendly_name}: {e}")
        return False

def test_ollama_connection():
    """Test connection to Ollama service"""
    print("🦙 Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            
            # Check if llama3.2 model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if any('llama3.2' in name for name in model_names):
                print("✅ llama3.2 model is available")
                return True
            else:
                print("⚠️  llama3.2 model not found. Available models:", model_names)
                print("   Run: ollama pull llama3.2")
                return False
        else:
            print(f"❌ Ollama service responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama service")
        print("   Make sure Ollama is installed and running: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False

def test_file_permissions():
    """Test file system permissions for database creation"""
    print("📁 Testing file system permissions...")
    test_dir = Path("./test_permissions")
    try:
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        test_dir.rmdir()
        print("✅ File system permissions are OK")
        return True
    except Exception as e:
        print(f"❌ File system permission error: {e}")
        return False

def run_basic_functionality_test():
    """Run a basic functionality test"""
    print("🧪 Running basic functionality test...")
    try:
        # Import the RAG agent
        sys.path.append(str(Path(__file__).parent))
        from rag_agent import RAGAgent
        
        # Try to initialize (this will test Ollama connection)
        agent = RAGAgent(model_name="llama3.2", persist_directory="./test_db")
        print("✅ RAG Agent initialized successfully")
        
        # Clean up test database
        import shutil
        if Path("./test_db").exists():
            shutil.rmtree("./test_db")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAG Agent Setup Test")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("LangChain", lambda: test_package_import("langchain")),
        ("LangChain Community", lambda: test_package_import("langchain_community")),
        ("LangGraph", lambda: test_package_import("langgraph")),
        ("ChromaDB", lambda: test_package_import("chromadb")),
        ("PyPDF", lambda: test_package_import("pypdf")),
        ("Unstructured", lambda: test_package_import("unstructured")),
        ("Requests", lambda: test_package_import("requests")),
        ("File Permissions", test_file_permissions),
        ("Ollama Connection", test_ollama_connection),
        ("Basic Functionality", run_basic_functionality_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG Agent setup is ready to use.")
        print("\nNext steps:")
        print("1. Run: python rag_agent.py --load /path/to/your/documents")
        print("2. Or try: python example_usage.py")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        
        # Provide specific help for common issues
        if not test_ollama_connection():
            print("\n🔧 Ollama Setup Help:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Start Ollama: ollama serve")
            print("3. Pull model: ollama pull llama3.2")
        
        missing_packages = []
        for pkg in ["langchain", "langchain_community", "langgraph", "chromadb", "pypdf", "unstructured"]:
            if not test_package_import(pkg):
                missing_packages.append(pkg)
        
        if missing_packages:
            print("\n📦 Install missing packages:")
            print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()