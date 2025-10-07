from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from litellm import completion
import os
from typing import TypedDict, Annotated, List, Dict, Optional
import operator
import re
import datetime
import yfinance as yf
import requests # For news API

# Set up LiteLLM to use Ollama
os.environ["LITELLM_MODEL_OLLAMA_LLAMA3_2"] = "ollama/llama3.2" # Ensure this matches your Ollama model name if different

class Expense(TypedDict):
    description: str
    amount: float
    category: str
    date: str

class AgentState(TypedDict):
    input: str
    chat_history: list
    result: str
    expenses: List[Expense]
    budgets: Dict[str, float]
    portfolio: List[Dict[str, any]] # Added for portfolio holdings
    action_result: str # To store messages from expense/budget operations
    stock_data: Optional[dict] # To store fetched stock data
    news_data: Optional[list]  # To store fetched news articles

# Helper functions for expense tracking and budgeting
def _add_expense(state: AgentState, description: str, amount: float, category: str = "uncategorized") -> str:
    """Adds an expense to the state."""
    if not isinstance(state.get("expenses"), list):
        state["expenses"] = []
    new_expense: Expense = {
        "description": description,
        "amount": amount,
        "category": category.lower(),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    state["expenses"].append(new_expense)
    return f"Expense added: {description} (${amount:.2f}) in category '{category}'."

def _set_budget(state: AgentState, category: str, amount: float) -> str:
    """Sets a budget for a category."""
    if not isinstance(state.get("budgets"), dict):
        state["budgets"] = {}
    state["budgets"][category.lower()] = amount
    return f"Budget for '{category}' set to ${amount:.2f}."

def _view_expenses(state: AgentState) -> str:
    """Views all recorded expenses."""
    expenses = state.get("expenses", [])
    if not expenses:
        return "No expenses recorded yet."
    response = "Recorded Expenses:\n"
    for exp in expenses:
        response += f"- {exp['date']}: {exp['description']} (${exp['amount']:.2f}) [Category: {exp['category']}]\n"
    return response

def _view_budget_status(state: AgentState) -> str:
    """Views spending against budgets."""
    budgets = state.get("budgets", {})
    expenses = state.get("expenses", [])
    if not budgets:
        return "No budgets set yet."

    response = "Budget Status:\n"
    for category, budget_amount in budgets.items():
        spent_on_category = sum(exp['amount'] for exp in expenses if exp['category'] == category)
# Helper functions for market data
def _get_stock_price(state: AgentState, ticker_symbol: str) -> str:
    """Fetches the current stock price for a given ticker symbol."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if hist.empty or 'Close' not in hist:
            return f"Could not retrieve current price for {ticker_symbol}. The ticker might be invalid or delisted."
        current_price = hist['Close'].iloc[-1]
        state["stock_data"] = {"ticker": ticker_symbol, "price": current_price, "currency": ticker.info.get("currency", "USD")}
        return f"The current price of {ticker_symbol} is {current_price:.2f} {state['stock_data']['currency']}."
    except Exception as e:
        return f"Error fetching stock price for {ticker_symbol}: {e}"

def _get_financial_news(state: AgentState, query: str) -> str:
    """Fetches recent financial news for a given company or ticker symbol."""
    # IMPORTANT: Replace 'YOUR_NEWS_API_KEY' with your actual NewsAPI.org API key.
    # You can obtain a free API key from https://newsapi.org/
    api_key = os.environ.get("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
    if api_key == "YOUR_NEWS_API_KEY":
        return "News API key not configured. Please set the NEWS_API_KEY environment variable or update it in the code."

    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&sortBy=publishedAt&language=en&pageSize=5"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        news_data = response.json()
        articles = news_data.get("articles", [])
        if not articles:
            return f"No recent news found for '{query}'."

        state["news_data"] = articles
        response_str = f"Recent news for '{query}':\n"
        for i, article in enumerate(articles[:3]): # Display top 3 articles
            response_str += f"{i+1}. {article['title']} ({article['source']['name']}) - {article['url']}\n"
        return response_str.strip()
    except requests.exceptions.RequestException as e:
        return f"Error fetching news for '{query}': {e}"
    except Exception as e:
        return f"An unexpected error occurred while fetching news: {e}"

# Helper functions for portfolio management
def _add_holding(state: AgentState, ticker_symbol: str, quantity: float, purchase_price: float) -> str:
    """Adds a holding to the portfolio."""
    if not isinstance(state.get("portfolio"), list):
        state["portfolio"] = []
    
    ticker_symbol = ticker_symbol.upper()
    # Check if holding already exists, if so, update quantity and average price
    for holding in state["portfolio"]:
        if holding["ticker"] == ticker_symbol:
            total_quantity = holding["quantity"] + quantity
            new_avg_price = ((holding["quantity"] * holding["purchase_price"]) + (quantity * purchase_price)) / total_quantity
            holding["quantity"] = total_quantity
            holding["purchase_price"] = new_avg_price
            return f"Updated holding: {quantity} shares of {ticker_symbol} added. New average price: ${new_avg_price:.2f}."

    new_holding = {
        "ticker": ticker_symbol,
        "quantity": quantity,
        "purchase_price": purchase_price,
        "date_added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    state["portfolio"].append(new_holding)
    return f"Added to portfolio: {quantity} shares of {ticker_symbol} at ${purchase_price:.2f} per share."

def _remove_holding(state: AgentState, ticker_symbol: str, quantity_to_remove: float) -> str:
    """Removes a specified quantity of a holding from the portfolio."""
    if not isinstance(state.get("portfolio"), list) or not state["portfolio"]:
        return f"Portfolio is empty. Cannot remove {ticker_symbol}."

    ticker_symbol = ticker_symbol.upper()
    holding_found = False
    for i, holding in enumerate(state["portfolio"]):
        if holding["ticker"] == ticker_symbol:
            holding_found = True
            if holding["quantity"] > quantity_to_remove:
                holding["quantity"] -= quantity_to_remove
                return f"Removed {quantity_to_remove} shares of {ticker_symbol}. Remaining: {holding['quantity']} shares."
            elif holding["quantity"] == quantity_to_remove:
                state["portfolio"].pop(i)
                return f"Removed all {quantity_to_remove} shares of {ticker_symbol} from portfolio."
            else:
                return f"Cannot remove {quantity_to_remove} shares of {ticker_symbol}. You only own {holding['quantity']} shares."
    if not holding_found:
        return f"Holding {ticker_symbol} not found in portfolio."

def _view_portfolio(state: AgentState) -> str:
    """Views all holdings in the portfolio."""
    portfolio = state.get("portfolio", [])
    if not portfolio:
        return "Your portfolio is currently empty."
    
    response = "Current Portfolio Holdings:\n"
    total_portfolio_value = 0
    overall_pnl = 0
    
    # Fetch current prices for all tickers in portfolio
    current_prices = {}
    for holding in portfolio:
        ticker_symbol = holding["ticker"]
        try:
            ticker_data = yf.Ticker(ticker_symbol)
            hist = ticker_data.history(period="1d")
            if not hist.empty and 'Close' in hist:
                current_prices[ticker_symbol] = hist['Close'].iloc[-1]
            else:
                current_prices[ticker_symbol] = None # Mark as not found
        except Exception:
            current_prices[ticker_symbol] = None # Mark as error

    for holding in portfolio:
        ticker = holding["ticker"]
        quantity = holding["quantity"]
        purchase_price = holding["purchase_price"]
        current_price = current_prices.get(ticker)
        
        response += f"- {ticker}: {quantity} shares, Avg. Purchase Price: ${purchase_price:.2f}"
        if current_price is not None:
            current_value = quantity * current_price
            pnl_per_holding = (current_price - purchase_price) * quantity
            total_portfolio_value += current_value
            overall_pnl += pnl_per_holding
            response += f", Current Price: ${current_price:.2f}, Value: ${current_value:.2f}, P/L: ${pnl_per_holding:+.2f}\n"
        else:
            response += f", Current Price: N/A (Could not fetch)\n"
            # If price can't be fetched, use purchase value for total value calculation for now, or skip
            # For simplicity, we'll add the cost basis to total value if current price is unavailable
            total_portfolio_value += quantity * purchase_price


    response += f"\nTotal Estimated Portfolio Value: ${total_portfolio_value:.2f}\n"
    response += f"Overall Portfolio P/L: ${overall_pnl:+.2f}\n"
    
    # Basic LLM Insight (very simple for now)
    if portfolio: # Only add insight if there's a portfolio
        news_summary_for_llm = ""
        if state.get("news_data"):
            news_summary_for_llm = " Based on recent news: "
            for article in state["news_data"][:2]: # Use first 2 news items if available
                news_summary_for_llm += f"'{article['title']}'. "
        
        llm_insight_prompt = f"My portfolio has a total value of ${total_portfolio_value:.2f} with an overall P/L of ${overall_pnl:+.2f}.{news_summary_for_llm}Provide a brief, general comment on this portfolio status."
        try:
            llm_response = completion(model="ollama/llama3.2", messages=[{"role": "user", "content": llm_insight_prompt}])
            insight = llm_response.choices[0].message.content
            response += f"\nLLM Insight: {insight}\n"
        except Exception as e:
            response += f"\nLLM Insight: Could not generate insight at this time ({e}).\n"
            
    return response

def _get_portfolio_value(state: AgentState) -> str:
    """Calculates and returns the current total value of the portfolio."""
    portfolio = state.get("portfolio", [])
    if not portfolio:
        return "Your portfolio is empty. Value is $0.00."

    total_value = 0
    for holding in portfolio:
        try:
            ticker = yf.Ticker(holding["ticker"])
            hist = ticker.history(period="1d")
            if not hist.empty and 'Close' in hist:
                current_price = hist['Close'].iloc[-1]
                total_value += holding["quantity"] * current_price
            else: # If price not found, add cost basis as a fallback for value
                total_value += holding["quantity"] * holding["purchase_price"]
        except Exception as e:
            # If error fetching, add cost basis
            total_value += holding["quantity"] * holding["purchase_price"]
            print(f"Could not fetch price for {holding['ticker']} for value calculation: {e}")
    return f"The current estimated total value of your portfolio is ${total_value:.2f}."

def _get_portfolio_pnl(state: AgentState) -> str:
    """Calculates and returns the overall profit/loss of the portfolio."""
    portfolio = state.get("portfolio", [])
    if not portfolio:
        return "Your portfolio is empty. P/L is $0.00."

    total_pnl = 0
    for holding in portfolio:
        try:
            ticker = yf.Ticker(holding["ticker"])
            hist = ticker.history(period="1d")
            if not hist.empty and 'Close' in hist:
                current_price = hist['Close'].iloc[-1]
                pnl_for_holding = (current_price - holding["purchase_price"]) * holding["quantity"]
                total_pnl += pnl_for_holding
            # If price not found, P/L for this holding is considered 0 for this calculation
        except Exception as e:
            print(f"Could not fetch price for {holding['ticker']} for P/L calculation: {e}")
            # P/L for this holding is $0 if current price cannot be fetched
    return f"Your overall portfolio P/L is ${total_pnl:+.2f}."

def _get_holding_pnl(state: AgentState, ticker_symbol: str) -> str:
    """Calculates and returns the profit/loss for a specific holding."""
    portfolio = state.get("portfolio", [])
    ticker_symbol = ticker_symbol.upper()
    
    for holding in portfolio:
        if holding["ticker"] == ticker_symbol:
            try:
                ticker_data = yf.Ticker(ticker_symbol)
                hist = ticker_data.history(period="1d")
                if not hist.empty and 'Close' in hist:
                    current_price = hist['Close'].iloc[-1]
                    pnl = (current_price - holding["purchase_price"]) * holding["quantity"]
                    current_value = current_price * holding["quantity"]
                    
                    # LLM insight for specific holding
                    news_for_holding_str = ""
                    # Attempt to fetch news specifically for this stock if not already broadly available
                    # For simplicity, we'll rely on existing news_data if it seems relevant, or fetch new
                    # This could be refined to fetch news specifically for `ticker_symbol` here.
                    if state.get("news_data"):
                         # Check if any news item mentions the ticker or a related term
                        for article in state["news_data"]:
                            if ticker_symbol.lower() in article['title'].lower() or ticker_symbol.lower() in article.get('description', '').lower():
                                news_for_holding_str += f" Relevant news: '{article['title']}'. "
                                break # Take first relevant news
                    
                    llm_insight_prompt = f"My holding of {holding['quantity']} shares of {ticker_symbol} (bought at ${holding['purchase_price']:.2f}) is currently valued at ${current_value:.2f}, with a P/L of ${pnl:+.2f}.{news_for_holding_str}Provide a brief comment on this specific holding."
                    insight_message = ""
                    try:
                        llm_response = completion(model="ollama/llama3.2", messages=[{"role": "user", "content": llm_insight_prompt}])
                        insight = llm_response.choices[0].message.content
                        insight_message = f"\nLLM Insight: {insight}"
                    except Exception as e_llm:
                        insight_message = f"\nLLM Insight: Could not generate insight ({e_llm})."

                    return f"For {ticker_symbol}: Quantity: {holding['quantity']}, Avg. Purchase Price: ${holding['purchase_price']:.2f}, Current Price: ${current_price:.2f}, Current Value: ${current_value:.2f}, P/L: ${pnl:+.2f}.{insight_message}"
                else:
                    return f"Could not retrieve current price for {ticker_symbol} to calculate P/L."
            except Exception as e:
                return f"Error fetching data for {ticker_symbol}: {e}"
    return f"Holding {ticker_symbol} not found in your portfolio."

def call_llm_and_finance_tools(state: AgentState):
    """
    Handles user input. If it's a finance command, executes it.
    Otherwise, calls the LLM.
    """
    user_input = state["input"].lower()
    current_chat_history = state.get("chat_history", [])
    action_taken = False
    action_message = ""

    # Ensure expenses and budgets are initialized in the state if not present
    if "expenses" not in state:
        state["expenses"] = []
    if "budgets" not in state:
        state["budgets"] = {}
    if "stock_data" not in state: # Initialize new state fields
        state["stock_data"] = None
    if "news_data" not in state: # Initialize new state fields
        state["news_data"] = None
    if "portfolio" not in state: # Initialize portfolio
        state["portfolio"] = []
 
    # Attempt to parse and execute finance commands
    # Add expense: "I spent $20 on groceries" or "add expense $20 for coffee category food"
    spent_match = re.search(r"(?:i spent|add expense)\s*\$?(\d+(?:\.\d{1,2})?)\s*(?:on|for)\s*(.+?)(?:\s*category\s*(.+))?$", user_input)
    if spent_match:
        try:
            amount = float(spent_match.group(1))
            description_full = spent_match.group(2).strip()
            category = spent_match.group(3).strip() if spent_match.group(3) else "uncategorized"
            
            # If category was part of description_full and not explicitly stated
            # e.g. "add expense $20 for coffee category food" -> desc: coffee, cat: food
            # e.g. "i spent $20 on groceries" -> desc: groceries, cat: uncategorized (unless LLM categorizes later)
            # For simplicity, we'll assume if "category" keyword is not used, the whole part is description.
            # A more advanced version would use LLM for categorization here.
            
            action_message = _add_expense(state, description_full, amount, category)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing expense: {e}. Please use format like 'I spent $AMOUNT on DESCRIPTION [category CATEGORY]'."
            action_taken = True # Still an action, albeit a failed one

    # Set budget: "set budget for food to $200" or "budget $200 for utilities"
    budget_match = re.search(r"(?:set budget|budget)\s*(?:for\s*)?(.+?)\s*(?:to|is)\s*\$?(\d+(?:\.\d{1,2})?)", user_input)
    if not action_taken and budget_match:
        try:
            category = budget_match.group(1).strip()
            amount = float(budget_match.group(2))
            action_message = _set_budget(state, category, amount)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing budget: {e}. Please use format like 'set budget for CATEGORY to $AMOUNT'."
            action_taken = True
    
    # View expenses: "show expenses" or "list my expenses"
    if not action_taken and ("show expenses" in user_input or "list expenses" in user_input or "view expenses" in user_input):
        action_message = _view_expenses(state)
        action_taken = True

    # View budget status: "show budget" or "budget status"
    if not action_taken and ("show budget" in user_input or "budget status" in user_input or "view budget" in user_input):
        action_message = _view_budget_status(state)
        action_taken = True

    # Get stock price: "what is the price of AAPL?" or "get stock price for MSFT"
    stock_price_match = re.search(r"(?:what is the price of|get stock price for|price of)\s+([A-Z\.]+)", user_input, re.IGNORECASE)
    if not action_taken and stock_price_match:
        try:
            ticker = stock_price_match.group(1).upper()
            action_message = _get_stock_price(state, ticker)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing stock price request: {e}"
            action_taken = True

    # Get financial news: "get news for Tesla" or "news about GOOGL"
    news_match = re.search(r"(?:get news for|news about|financial news for)\s+(.+)", user_input, re.IGNORECASE)
    if not action_taken and news_match:
        try:
            query = news_match.group(1).strip()
            action_message = _get_financial_news(state, query)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing news request: {e}"
            action_taken = True
            
    # Portfolio commands
    # Add holding: "I own 10 shares of AAPL bought at $150" or "add 10 AAPL at 150 to portfolio"
    add_holding_match = re.search(r"(?:i own|add)\s*(\d+(?:\.\d*)?)\s*(?:shares of)?\s*([A-Z\.]+)\s*(?:bought at|at)\s*\$?(\d+(?:\.\d{1,2})?)(?:\s*to portfolio)?", user_input, re.IGNORECASE)
    if not action_taken and add_holding_match:
        try:
            quantity = float(add_holding_match.group(1))
            ticker = add_holding_match.group(2).upper()
            price = float(add_holding_match.group(3))
            action_message = _add_holding(state, ticker, quantity, price)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing add holding command: {e}. Use format like 'add 10 AAPL at 150'."
            action_taken = True

    # Remove holding: "remove 5 shares of MSFT" or "sell 5 MSFT"
    remove_holding_match = re.search(r"(?:remove|sell)\s*(\d+(?:\.\d*)?)\s*(?:shares of)?\s*([A-Z\.]+)", user_input, re.IGNORECASE)
    if not action_taken and remove_holding_match:
        try:
            quantity = float(remove_holding_match.group(1))
            ticker = remove_holding_match.group(2).upper()
            action_message = _remove_holding(state, ticker, quantity)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing remove holding command: {e}. Use format like 'remove 5 MSFT'."
            action_taken = True

    # View portfolio: "show my portfolio" or "view portfolio"
    if not action_taken and ("show my portfolio" in user_input or "view portfolio" in user_input or "my holdings" in user_input):
        action_message = _view_portfolio(state)
        action_taken = True

    # Get portfolio value: "what is my portfolio value" or "portfolio worth"
    if not action_taken and ("portfolio value" in user_input or "portfolio worth" in user_input):
        action_message = _get_portfolio_value(state)
        action_taken = True

    # Get portfolio P/L: "portfolio pnl" or "overall profit loss"
    if not action_taken and ("portfolio pnl" in user_input or "overall profit" in user_input or "overall loss" in user_input):
        action_message = _get_portfolio_pnl(state)
        action_taken = True

    # Analyze holding / Get holding P/L: "analyze my AAPL holding" or "pnl for MSFT"
    analyze_holding_match = re.search(r"(?:analyze|pnl for|details for)\s*(?:my)?\s*([A-Z\.]+)\s*(?:holding)?", user_input, re.IGNORECASE)
    if not action_taken and analyze_holding_match:
        try:
            ticker = analyze_holding_match.group(1).upper()
            action_message = _get_holding_pnl(state, ticker)
            action_taken = True
        except Exception as e:
            action_message = f"Error parsing holding analysis request: {e}."
            action_taken = True

    if action_taken:
        updated_chat_history = current_chat_history + [
            {"role": "user", "content": state["input"]},
            {"role": "assistant", "content": action_message}
        ]
        return {
            "result": action_message,
            "chat_history": updated_chat_history,
            "expenses": state["expenses"],
            "budgets": state["budgets"],
            "portfolio": state.get("portfolio"), # Added portfolio
            "stock_data": state.get("stock_data"),
            "news_data": state.get("news_data"),
            "action_result": action_message
        }
 
    # If no finance command was handled, call the LLM
    messages = current_chat_history + [{"role": "user", "content": state["input"]}]
    try:
        response = completion(
            model="ollama/llama3.2",
            messages=messages
        )
        result_content = response.choices[0].message.content
        updated_chat_history = messages + [{"role": "assistant", "content": result_content}]
        return {
            "result": result_content,
            "chat_history": updated_chat_history,
            "expenses": state["expenses"],
            "budgets": state["budgets"],
            "portfolio": state.get("portfolio"), # Added portfolio
            "stock_data": state.get("stock_data"),
            "news_data": state.get("news_data"),
            "action_result": ""
        }
    except Exception as e:
        print(f"Error calling LiteLLM: {e}")
        error_message = f"Error: {str(e)}"
        updated_chat_history = messages + [{"role": "assistant", "content": error_message}]
        return {
            "result": error_message,
            "chat_history": updated_chat_history,
            "expenses": state["expenses"],
            "budgets": state["budgets"],
            "portfolio": state.get("portfolio"), # Added portfolio
            "stock_data": state.get("stock_data"),
            "news_data": state.get("news_data"),
            "action_result": error_message
        }

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("processor", call_llm_and_finance_tools)

# Set entry and finish points
workflow.set_entry_point("processor")
workflow.add_edge("processor", END)

# Compile the graph
memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)

# Example usage (can be run directly or imported)
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "expense-tracker-thread"}}
    
    # Initial state with empty expenses and budgets
    initial_app_state = {"input": "", "chat_history": [], "expenses": [], "budgets": {}, "portfolio": [], "action_result": "", "stock_data": None, "news_data": None}
 
    print("Starting financial agent. Type 'exit' to quit.")

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Exiting agent.")
            break
        
        # Update the input in the state for the stream
        # The checkpointer should maintain expenses and budgets across calls for the same thread_id
        current_state_for_stream = {"input": user_input}

        final_event_result = None
        for event in app.stream(current_state_for_stream, config=config, stream_mode="values"):
            # The 'values' stream_mode yields the full state after each node.
            # We are interested in the 'result' or 'action_result' from our processor node.
            if "processor" in event: # Check if the event is from our node
                processed_state = event["processor"]
                if processed_state.get("action_result"):
                    final_event_result = processed_state["action_result"]
                elif processed_state.get("result"): # Fallback to general result if no specific action_result
                    final_event_result = processed_state["result"]
        
        if final_event_result:
            print(f"Agent: {final_event_result}")
        else:
            # This case might happen if the stream doesn't yield the expected structure
            # or if the last event doesn't contain the result.
            # For robust error handling, one might inspect the full event structure.
            print("Agent: (No specific result processed for this input)")

    # Example interactions:
    # User: I spent $25.50 on lunch category food
    # User: set budget for entertainment to $100
    # User: I spent $30 on movie tickets category entertainment
    # User: show expenses
    # User: show budget status
    # User: What is the capital of France?