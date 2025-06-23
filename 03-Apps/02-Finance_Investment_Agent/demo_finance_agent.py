"""
Finance & Investment Agent Demo
==============================

This script demonstrates all the capabilities of the Finance & Investment Agent
including expense tracking, budget creation, market data fetching, and portfolio analysis.
"""

import json
from finance_investment_agent import create_finance_agent, UserData, Portfolio, ExpenseDetails, BudgetItem
from langchain_core.messages import HumanMessage

def demo_expense_tracking(agent, user_data):
    """Demonstrate expense tracking functionality."""
    print("\n" + "="*60)
    print("💰 DEMO 1: Expense Tracking")
    print("="*60)
    
    test_expenses = [
        "I spent $45.50 on groceries at Whole Foods",
        "Paid $25 for gas today",
        "Bought lunch for $12.75",
        "Entertainment expense of $30 for movie tickets"
    ]
    
    for expense_msg in test_expenses:
        print(f"\n👤 User: {expense_msg}")
        
        initial_state = {
            "messages": [HumanMessage(content=expense_msg)],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        result = agent.invoke(initial_state)
        if result["messages"]:
            response = result["messages"][-1].content
            print(f"🤖 Agent: {response}")

def demo_budget_creation(agent, user_data):
    """Demonstrate budget creation functionality."""
    print("\n" + "="*60)
    print("📊 DEMO 2: Budget Creation")
    print("="*60)
    
    budget_requests = [
        "Set up a monthly budget: $500 for food, $300 for transportation, $200 for entertainment",
        "I want to allocate $150 for utilities and $100 for healthcare",
        "Create a budget with $1000 for rent and $400 for groceries"
    ]
    
    for budget_msg in budget_requests:
        print(f"\n👤 User: {budget_msg}")
        
        initial_state = {
            "messages": [HumanMessage(content=budget_msg)],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        result = agent.invoke(initial_state)
        if result["messages"]:
            response = result["messages"][-1].content
            print(f"🤖 Agent: {response}")

def demo_market_data(agent, user_data):
    """Demonstrate market data fetching functionality."""
    print("\n" + "="*60)
    print("📈 DEMO 3: Market Data & News")
    print("="*60)
    
    market_requests = [
        "What's the current price of Apple stock?",
        "Get me news about Tesla",
        "Show me Google stock information",
        "What's happening with Microsoft in the market?"
    ]
    
    for market_msg in market_requests:
        print(f"\n👤 User: {market_msg}")
        
        initial_state = {
            "messages": [HumanMessage(content=market_msg)],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        result = agent.invoke(initial_state)
        if result["messages"]:
            response = result["messages"][-1].content
            print(f"🤖 Agent: {response}")

def demo_portfolio_analysis(agent, user_data):
    """Demonstrate portfolio analysis functionality."""
    print("\n" + "="*60)
    print("🏦 DEMO 4: Portfolio Analysis")
    print("="*60)
    
    portfolio_requests = [
        "Analyze my current portfolio",
        "How is my portfolio performing?",
        "Give me insights on my holdings"
    ]
    
    for portfolio_msg in portfolio_requests:
        print(f"\n👤 User: {portfolio_msg}")
        
        initial_state = {
            "messages": [HumanMessage(content=portfolio_msg)],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        result = agent.invoke(initial_state)
        if result["messages"]:
            response = result["messages"][-1].content
            print(f"🤖 Agent: {response}")

def demo_general_help(agent, user_data):
    """Demonstrate general help and conversation functionality."""
    print("\n" + "="*60)
    print("❓ DEMO 5: General Help & Conversation")
    print("="*60)
    
    general_requests = [
        "What can you help me with?",
        "How should I start investing?",
        "What are some good budgeting tips?",
        "Tell me about diversification"
    ]
    
    for general_msg in general_requests:
        print(f"\n👤 User: {general_msg}")
        
        initial_state = {
            "messages": [HumanMessage(content=general_msg)],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        result = agent.invoke(initial_state)
        if result["messages"]:
            response = result["messages"][-1].content
            print(f"🤖 Agent: {response}")

def create_sample_user_data():
    """Create sample user data for demonstration."""
    # Sample expenses
    sample_expenses = [
        ExpenseDetails(50.0, "food", "Grocery shopping", "2024-01-15"),
        ExpenseDetails(25.0, "transport", "Gas", "2024-01-16"),
        ExpenseDetails(15.0, "food", "Lunch", "2024-01-17"),
        ExpenseDetails(100.0, "entertainment", "Concert tickets", "2024-01-18")
    ]
    
    # Sample budget
    sample_budget = [
        BudgetItem("food", 500.0, 65.0),
        BudgetItem("transport", 300.0, 25.0),
        BudgetItem("entertainment", 200.0, 100.0),
        BudgetItem("utilities", 150.0, 0.0)
    ]
    
    # Sample portfolio
    sample_portfolio = Portfolio(
        holdings={
            "AAPL": 10.0,
            "GOOGL": 5.0,
            "TSLA": 3.0,
            "MSFT": 8.0,
            "AMZN": 2.0
        },
        cash_balance=2500.0
    )
    
    # Create user data
    user_data = UserData(
        name="Demo User",
        transactions=sample_expenses,
        budget=sample_budget,
        portfolio=sample_portfolio,
        income=5000.0,
        savings_goals={"emergency_fund": 10000.0, "vacation": 3000.0}
    )
    
    return user_data

def run_full_demo():
    """Run the complete demonstration of the Finance Agent."""
    print("🏦 Finance & Investment Agent - Complete Demo")
    print("=" * 80)
    print("This demo showcases all the capabilities of the Finance Agent:")
    print("• Expense tracking and categorization")
    print("• Budget creation and management") 
    print("• Market data and financial news")
    print("• Portfolio analysis and insights")
    print("• General financial advice and help")
    print("=" * 80)
    
    try:
        # Create the agent
        print("\n🔧 Creating Finance Agent...")
        agent = create_finance_agent()
        print("✅ Agent created successfully!")
        
        # Create sample user data
        print("\n📊 Setting up sample user data...")
        user_data = create_sample_user_data()
        print("✅ Sample data created!")
        
        # Display user data summary
        print(f"\n👤 Demo User Profile:")
        print(f"• Name: {user_data.name}")
        print(f"• Monthly Income: ${user_data.income:,.2f}")
        print(f"• Portfolio Holdings: {len(user_data.portfolio.holdings)} stocks")
        print(f"• Cash Balance: ${user_data.portfolio.cash_balance:,.2f}")
        print(f"• Expense History: {len(user_data.transactions)} transactions")
        print(f"• Budget Categories: {len(user_data.budget)} categories")
        
        # Run all demos
        demo_expense_tracking(agent, user_data)
        demo_budget_creation(agent, user_data)
        demo_market_data(agent, user_data)
        demo_portfolio_analysis(agent, user_data)
        demo_general_help(agent, user_data)
        
        print("\n" + "="*80)
        print("🎉 Demo completed successfully!")
        print("="*80)
        print("📋 Summary of demonstrated features:")
        print("✅ Intent classification and routing")
        print("✅ Expense tracking with natural language")
        print("✅ Budget creation and management")
        print("✅ Market data aggregation")
        print("✅ Portfolio performance analysis")
        print("✅ General financial advice")
        print("✅ LangGraph workflow orchestration")
        print("✅ Ollama llama3.2 model integration")
        
        print("\n🚀 Ready to use the Finance Agent!")
        print("Run 'python finance_investment_agent.py' to start the interactive session.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("• Ensure Ollama is running: 'ollama serve'")
        print("• Check if llama3.2 model is available: 'ollama list'")
        print("• Install requirements: 'pip install -r requirements.txt'")

def run_interactive_demo():
    """Run an interactive demo where user can test specific features."""
    print("\n🎮 Interactive Demo Mode")
    print("=" * 40)
    print("Choose a demo to run:")
    print("1. Expense Tracking")
    print("2. Budget Creation")
    print("3. Market Data")
    print("4. Portfolio Analysis")
    print("5. General Help")
    print("6. Run All Demos")
    print("0. Exit")
    
    agent = create_finance_agent()
    user_data = create_sample_user_data()
    
    while True:
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            demo_expense_tracking(agent, user_data)
        elif choice == "2":
            demo_budget_creation(agent, user_data)
        elif choice == "3":
            demo_market_data(agent, user_data)
        elif choice == "4":
            demo_portfolio_analysis(agent, user_data)
        elif choice == "5":
            demo_general_help(agent, user_data)
        elif choice == "6":
            run_full_demo()
            break
        else:
            print("❌ Invalid choice. Please enter 0-6.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_demo()
    else:
        run_full_demo()