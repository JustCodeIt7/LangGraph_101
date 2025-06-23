#!/usr/bin/env python3
"""
Quick Test Script for Finance & Investment Agent
===============================================

This script performs basic tests to ensure the agent is working correctly
before running the full demo or interactive mode.
"""

import sys
import traceback
from finance_investment_agent import create_finance_agent, UserData, Portfolio
from langchain_core.messages import HumanMessage

def test_agent_creation():
    """Test that the agent can be created successfully."""
    print("🧪 Test 1: Agent Creation")
    print("-" * 30)
    
    try:
        agent = create_finance_agent()
        print("✅ Agent created successfully!")
        return agent
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        print(f"📍 Error details: {traceback.format_exc()}")
        return None

def test_basic_conversation(agent):
    """Test basic conversation with the agent."""
    print("\n🧪 Test 2: Basic Conversation")
    print("-" * 35)
    
    if agent is None:
        print("❌ Skipping - agent not available")
        return False
    
    try:
        # Create minimal user data
        user_data = UserData(
            name="Test User",
            transactions=[],
            budget=[],
            portfolio=Portfolio(holdings={}, cash_balance=0.0)
        )
        
        # Test simple greeting
        initial_state = {
            "messages": [HumanMessage(content="Hello, what can you help me with?")],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        result = agent.invoke(initial_state)
        
        if result and "messages" in result and result["messages"]:
            response = result["messages"][-1].content
            print(f"✅ Agent responded: {response[:100]}...")
            return True
        else:
            print("❌ Agent did not respond properly")
            return False
            
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        print(f"📍 Error details: {traceback.format_exc()}")
        return False

def test_intent_classification(agent):
    """Test intent classification with different message types."""
    print("\n🧪 Test 3: Intent Classification")
    print("-" * 38)
    
    if agent is None:
        print("❌ Skipping - agent not available")
        return False
    
    test_messages = [
        ("I spent $20 on lunch", "expense tracking"),
        ("Set a budget for groceries", "budget creation"),
        ("What's the price of Apple stock?", "market data"),
        ("Analyze my portfolio", "portfolio analysis")
    ]
    
    user_data = UserData(
        name="Test User",
        transactions=[],
        budget=[],
        portfolio=Portfolio(holdings={"AAPL": 10}, cash_balance=1000.0)
    )
    
    success_count = 0
    
    for message, expected_intent in test_messages:
        try:
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_data": user_data,
                "intent": "",
                "current_request": {},
                "market_data": {}
            }
            
            result = agent.invoke(initial_state)
            
            if result and "messages" in result and result["messages"]:
                response = result["messages"][-1].content
                print(f"✅ '{message}' → Response received")
                success_count += 1
            else:
                print(f"❌ '{message}' → No response")
                
        except Exception as e:
            print(f"❌ '{message}' → Error: {e}")
    
    print(f"\n📊 Intent classification results: {success_count}/{len(test_messages)} successful")
    return success_count == len(test_messages)

def test_tools_import():
    """Test that all tools can be imported and used."""
    print("\n🧪 Test 4: Tools Import & Basic Usage")
    print("-" * 42)
    
    try:
        from finance_investment_agent import (
            track_expense_tool,
            create_budget_item_tool,
            fetch_market_news_tool,
            fetch_stock_price_tool,
            analyze_portfolio_tool
        )
        
        # Test expense tool
        result = track_expense_tool.invoke({
            "amount": 25.0,
            "category": "food",
            "description": "test lunch"
        })
        print(f"✅ Expense tool: {result}")
        
        # Test budget tool
        result = create_budget_item_tool.invoke({
            "category": "food",
            "allocated_amount": 500.0
        })
        print(f"✅ Budget tool: {result}")
        
        # Test market news tool
        result = fetch_market_news_tool.invoke({"topic": "AAPL"})
        print(f"✅ Market news tool: Working (truncated output)")
        
        # Test stock price tool
        result = fetch_stock_price_tool.invoke({"symbol": "AAPL"})
        print(f"✅ Stock price tool: {result}")
        
        print("✅ All tools imported and working!")
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        print(f"📍 Error details: {traceback.format_exc()}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n🧪 Test 5: Dependency Check")
    print("-" * 32)
    
    required_modules = [
        "langgraph",
        "langchain_core",
        "langchain_litellm",
        "pydantic"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - Missing!")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n📋 Missing dependencies: {', '.join(missing_modules)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available!")
        return True

def check_ollama_connection():
    """Check if Ollama is running and llama3.2 is available."""
    print("\n🧪 Test 6: Ollama Connection")
    print("-" * 33)
    
    try:
        from langchain_litellm import ChatLiteLLM
        
        llm = ChatLiteLLM(
            model="ollama/llama3.2",
            api_base="http://localhost:11434",
            temperature=0.3
        )
        
        # Try a simple test
        response = llm.invoke([HumanMessage(content="Hello")])
        print("✅ Ollama connection successful!")
        print(f"✅ llama3.2 model responding: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull model: ollama pull llama3.2")
        print("  3. Check model: ollama list")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 Finance & Investment Agent - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependency Check", test_dependencies),
        ("Ollama Connection", check_ollama_connection),
        ("Agent Creation", test_agent_creation),
    ]
    
    results = {}
    agent = None
    
    # Run initial tests
    for test_name, test_func in tests:
        if test_name == "Agent Creation":
            agent = test_func()
            results[test_name] = agent is not None
        else:
            results[test_name] = test_func()
    
    # Run agent-dependent tests if agent was created
    if agent:
        agent_tests = [
            ("Basic Conversation", lambda: test_basic_conversation(agent)),
            ("Intent Classification", lambda: test_intent_classification(agent)),
            ("Tools Import & Usage", test_tools_import),
        ]
        
        for test_name, test_func in agent_tests:
            results[test_name] = test_func()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Finance Agent is ready to use.")
        print("\n🚀 Next steps:")
        print("  • Run the demo: python demo_finance_agent.py")
        print("  • Start interactive mode: python finance_investment_agent.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n💡 Common solutions:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Start Ollama: ollama serve")
        print("  • Pull model: ollama pull llama3.2")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)