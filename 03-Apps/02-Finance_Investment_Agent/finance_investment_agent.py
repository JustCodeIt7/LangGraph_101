"""
Finance & Investment Agent
=========================

This agent helps users with personal finance management and investment analysis.
It can track expenses, create budgets, aggregate market data (news, prices, reports),
and provide portfolio analysis and investment insights.

Features:
- Track expenses and manage budgets
- Fetch market data and financial news
- Analyze portfolio performance
- Generate investment insights
- Use LangGraph for agent orchestration
- Use Ollama with llama3.2 model
"""

import os
import json
import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from decimal import Decimal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, Field

# Data Models
@dataclass
class ExpenseDetails:
    amount: float
    category: str
    description: str
    date: str
    
@dataclass
class BudgetItem:
    category: str
    allocated_amount: float
    spent_amount: float = 0.0
    
@dataclass
class Portfolio:
    holdings: Dict[str, float]  # symbol -> quantity
    cash_balance: float = 0.0
    
@dataclass
class UserData:
    name: str
    transactions: List[ExpenseDetails]
    budget: List[BudgetItem]
    portfolio: Portfolio
    income: float = 0.0
    savings_goals: Dict[str, float] = None
    
    def __post_init__(self):
        if self.savings_goals is None:
            self.savings_goals = {}

# Agent State
class FinanceAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_data: UserData
    intent: str
    current_request: Dict[str, Any]
    market_data: Dict[str, Any]

# Finance Tools
@tool("track_expense")
def track_expense_tool(amount: float, category: str, description: str) -> str:
    """
    Track a new expense.
    
    Args:
        amount: The expense amount
        category: The expense category (e.g., 'food', 'transport', 'entertainment')
        description: Description of the expense
    """
    expense = ExpenseDetails(
        amount=amount,
        category=category,
        description=description,
        date=datetime.datetime.now().strftime("%Y-%m-%d")
    )
    return f"Expense tracked: ${amount:.2f} for {category} - {description}"

@tool("create_budget_item")
def create_budget_item_tool(category: str, allocated_amount: float) -> str:
    """
    Create a new budget item for a category.
    
    Args:
        category: The budget category
        allocated_amount: The amount allocated for this category
    """
    return f"Budget item created: ${allocated_amount:.2f} allocated for {category}"

@tool("fetch_market_news")
def fetch_market_news_tool(topic: str) -> str:
    """
    Fetch market news for a specific topic or stock.
    
    Args:
        topic: The topic or stock symbol to get news for
    """
    # Mock implementation - in real scenario would call news API
    mock_news = [
        f"{topic} stock shows strong performance in Q4 earnings report",
        f"Market analysts upgrade {topic} rating to 'Buy'",
        f"Industry trends favor {topic} sector growth",
        f"{topic} announces new product launch, shares up 5%",
        f"Institutional investors increase stake in {topic}"
    ]
    return f"Latest news for {topic}:\n" + "\n".join(f"• {news}" for news in mock_news[:3])

@tool("fetch_stock_price")
def fetch_stock_price_tool(symbol: str) -> str:
    """
    Fetch current stock price for a symbol.
    
    Args:
        symbol: The stock symbol (e.g., 'AAPL', 'GOOGL')
    """
    # Mock implementation - in real scenario would call stock API
    import random
    price = round(random.uniform(50, 500), 2)
    change = round(random.uniform(-5, 5), 2)
    change_percent = round((change / price) * 100, 2)
    
    return f"{symbol}: ${price:.2f} ({change:+.2f}, {change_percent:+.2f}%)"

@tool("analyze_portfolio")
def analyze_portfolio_tool(holdings_json: str) -> str:
    """
    Analyze portfolio performance and provide insights.
    
    Args:
        holdings_json: JSON string of portfolio holdings
    """
    try:
        holdings = json.loads(holdings_json)
        total_value = sum(float(quantity) * 100 for quantity in holdings.values())  # Mock calculation
        num_holdings = len(holdings)
        
        analysis = f"""
Portfolio Analysis:
- Total Holdings: {num_holdings} positions
- Estimated Value: ${total_value:.2f}
- Diversification: {'Good' if num_holdings > 5 else 'Consider more diversification'}
- Top Holdings: {', '.join(list(holdings.keys())[:3])}
        """
        return analysis.strip()
    except:
        return "Error analyzing portfolio. Please check the holdings format."

@tool("calculate_budget_summary")
def calculate_budget_summary_tool(budget_json: str, expenses_json: str) -> str:
    """
    Calculate budget summary and spending analysis.
    
    Args:
        budget_json: JSON string of budget items
        expenses_json: JSON string of expenses
    """
    try:
        budget = json.loads(budget_json)
        expenses = json.loads(expenses_json)
        
        # Calculate spending by category
        spending_by_category = {}
        for expense in expenses:
            category = expense['category']
            amount = expense['amount']
            spending_by_category[category] = spending_by_category.get(category, 0) + amount
        
        summary = "Budget Summary:\n"
        for item in budget:
            category = item['category']
            allocated = item['allocated_amount']
            spent = spending_by_category.get(category, 0)
            remaining = allocated - spent
            summary += f"• {category}: ${spent:.2f}/${allocated:.2f} (${remaining:.2f} remaining)\n"
        
        return summary.strip()
    except:
        return "Error calculating budget summary. Please check the data format."

# Agent Nodes
def intent_classifier_node(state: FinanceAgentState) -> Dict[str, Any]:
    """Classify the user's intent based on their message."""
    messages = state["messages"]
    if not messages:
        return {"intent": "unknown"}
    
    last_message = messages[-1].content.lower()
    
    if any(word in last_message for word in ["expense", "spent", "bought", "paid", "cost"]):
        intent = "track_expense"
    elif any(word in last_message for word in ["budget", "allocate", "plan", "spending"]):
        intent = "create_budget"
    elif any(word in last_message for word in ["market", "news", "stock", "price", "ticker"]):
        intent = "get_market_data"
    elif any(word in last_message for word in ["portfolio", "holdings", "analyze", "performance"]):
        intent = "analyze_portfolio"
    else:
        intent = "general_help"
    
    return {"intent": intent}

def expense_tracker_node(state: FinanceAgentState) -> Dict[str, Any]:
    """Handle expense tracking requests."""
    messages = state["messages"]
    user_message = messages[-1].content
    
    # Create LLM to extract expense details
    llm = ChatLiteLLM(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
        temperature=0.3
    )
    
    extraction_prompt = f"""
    Extract expense details from this message: "{user_message}"
    
    Please identify:
    1. Amount (number)
    2. Category (food, transport, entertainment, utilities, etc.)
    3. Description (brief description)
    
    If any information is missing, make reasonable assumptions.
    Format your response as: amount|category|description
    """
    
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    
    try:
        parts = response.content.strip().split('|')
        if len(parts) >= 3:
            amount = float(parts[0].replace('$', '').strip())
            category = parts[1].strip()
            description = parts[2].strip()
            
            # Track the expense
            result = track_expense_tool.invoke({"amount": amount, "category": category, "description": description})
            
            return {"messages": [AIMessage(content=f"✅ {result}")]}
    except:
        pass
    
    return {"messages": [AIMessage(content="I need more information to track your expense. Please specify the amount, category, and description.")]}

def budget_creator_node(state: FinanceAgentState) -> Dict[str, Any]:
    """Handle budget creation requests."""
    messages = state["messages"]
    user_message = messages[-1].content
    
    llm = ChatLiteLLM(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
        temperature=0.3
    )
    
    extraction_prompt = f"""
    Extract budget information from this message: "{user_message}"
    
    Please identify budget categories and amounts. Common categories include:
    - Food/Groceries
    - Transportation
    - Entertainment
    - Utilities
    - Rent/Housing
    - Healthcare
    - Savings
    
    Format your response as category|amount for each item, one per line.
    """
    
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    
    budget_items = []
    for line in response.content.strip().split('\n'):
        try:
            if '|' in line:
                category, amount = line.split('|', 1)
                amount = float(amount.replace('$', '').strip())
                result = create_budget_item_tool.invoke({"category": category.strip(), "allocated_amount": amount})
                budget_items.append(result)
        except:
            continue
    
    if budget_items:
        return {"messages": [AIMessage(content="✅ Budget created:\n" + "\n".join(budget_items))]}
    else:
        return {"messages": [AIMessage(content="I need more specific information to create your budget. Please specify categories and amounts.")]}

def market_data_node(state: FinanceAgentState) -> Dict[str, Any]:
    """Handle market data requests."""
    messages = state["messages"]
    user_message = messages[-1].content
    
    llm = ChatLiteLLM(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
        temperature=0.3
    )
    
    extraction_prompt = f"""
    Extract stock symbols or market topics from this message: "{user_message}"
    
    Look for:
    - Stock symbols (like AAPL, GOOGL, TSLA)
    - Company names
    - Market topics
    
    Return just the symbol or topic, nothing else.
    """
    
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    topic = response.content.strip().upper()
    
    # Fetch market data
    news_result = fetch_market_news_tool.invoke({"topic": topic})
    price_result = fetch_stock_price_tool.invoke({"symbol": topic})
    
    market_response = f"📊 Market Data for {topic}:\n\n📈 Price: {price_result}\n\n📰 News:\n{news_result}"
    
    return {"messages": [AIMessage(content=market_response)]}

def portfolio_analyzer_node(state: FinanceAgentState) -> Dict[str, Any]:
    """Handle portfolio analysis requests."""
    user_data = state.get("user_data")
    
    if not user_data or not user_data.portfolio.holdings:
        return {"messages": [AIMessage(content="You don't have a portfolio set up yet. Please add some holdings first.")]}
    
    holdings_json = json.dumps(user_data.portfolio.holdings)
    analysis_result = analyze_portfolio_tool.invoke({"holdings_json": holdings_json})
    
    return {"messages": [AIMessage(content=f"📊 Portfolio Analysis:\n{analysis_result}")]}

def general_help_node(state: FinanceAgentState) -> Dict[str, Any]:
    """Handle general help and other requests."""
    messages = state["messages"]
    
    llm = ChatLiteLLM(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
        temperature=0.7
    )
    
    system_prompt = """You are a helpful finance and investment assistant. You can help users with:

1. 💰 Expense Tracking - Track your daily expenses and categorize them
2. 📊 Budget Planning - Create and manage budgets for different categories
3. 📈 Market Data - Get latest stock prices and financial news
4. 🏦 Portfolio Analysis - Analyze your investment portfolio performance

Please provide helpful financial advice and guide users on how to use these features.
"""
    
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages_with_system)
    
    return {"messages": [AIMessage(content=response.content)]}

# Routing Logic
def route_intent(state: FinanceAgentState) -> str:
    """Route to the appropriate node based on intent."""
    intent = state.get("intent", "general_help")
    
    if intent == "track_expense":
        return "expense_tracker"
    elif intent == "create_budget":
        return "budget_creator"
    elif intent == "get_market_data":
        return "market_data"
    elif intent == "analyze_portfolio":
        return "portfolio_analyzer"
    else:
        return "general_help"

# Create the Finance Agent Graph
def create_finance_agent():
    """Create the finance agent with LangGraph."""
    
    # Create the graph
    workflow = StateGraph(FinanceAgentState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("expense_tracker", expense_tracker_node)
    workflow.add_node("budget_creator", budget_creator_node)
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("portfolio_analyzer", portfolio_analyzer_node)
    workflow.add_node("general_help", general_help_node)
    
    # Set entry point
    workflow.set_entry_point("intent_classifier")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "expense_tracker": "expense_tracker",
            "budget_creator": "budget_creator",
            "market_data": "market_data",
            "portfolio_analyzer": "portfolio_analyzer",
            "general_help": "general_help"
        }
    )
    
    # All nodes end the conversation
    workflow.add_edge("expense_tracker", END)
    workflow.add_edge("budget_creator", END)
    workflow.add_edge("market_data", END)
    workflow.add_edge("portfolio_analyzer", END)
    workflow.add_edge("general_help", END)
    
    return workflow.compile()

def main():
    """Main function to run the finance agent."""
    print("🏦 Welcome to the Finance & Investment Agent!")
    print("=" * 60)
    print("I can help you with:")
    print("💰 Expense tracking")
    print("📊 Budget planning")
    print("📈 Market data and news")
    print("🏦 Portfolio analysis")
    print("\nType 'exit' to quit, 'help' for more information.")
    print("=" * 60)
    
    # Create the agent
    agent = create_finance_agent()
    
    # Initialize user data
    user_data = UserData(
        name="User",
        transactions=[],
        budget=[],
        portfolio=Portfolio(holdings={"AAPL": 10, "GOOGL": 5, "TSLA": 3}, cash_balance=1000.0)
    )
    
    # Conversation loop
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() == 'exit':
            print("\n👋 Thanks for using the Finance Agent!")
            break
        
        if user_input.lower() == 'help':
            print("\n📋 Available commands:")
            print("• Track expense: 'I spent $50 on groceries'")
            print("• Create budget: 'Set $500 budget for food'")
            print("• Get market data: 'What's the price of AAPL?'")
            print("• Analyze portfolio: 'Analyze my portfolio'")
            continue
        
        if not user_input:
            continue
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_data": user_data,
            "intent": "",
            "current_request": {},
            "market_data": {}
        }
        
        try:
            # Run the agent
            result = agent.invoke(initial_state)
            
            # Get the response
            if result["messages"]:
                response = result["messages"][-1].content
                print(f"\n🤖 Agent: {response}")
            else:
                print("\n🤖 Agent: I'm sorry, I couldn't process your request.")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")

if __name__ == "__main__":
    main()