# Finance & Investment Agent 🏦

A comprehensive AI-powered finance and investment assistant built with **LangGraph** and **Ollama** using the **llama3.2** model. This agent helps users with personal finance management, investment analysis, expense tracking, and market data aggregation.

## 🌟 Features

### 💰 Expense Tracking
- **Natural Language Processing**: Track expenses using conversational input
- **Automatic Categorization**: Intelligent categorization of expenses
- **Transaction History**: Maintain detailed expense records
- **Date Tracking**: Automatic timestamp for all transactions

### 📊 Budget Management
- **Smart Budget Creation**: Create budgets from natural language descriptions
- **Category Management**: Support for multiple budget categories
- **Spending Analysis**: Compare actual spending vs. budgeted amounts
- **Budget Recommendations**: AI-powered budget suggestions

### 📈 Market Data & News
- **Real-time Stock Prices**: Get current stock prices for any ticker
- **Financial News**: Aggregate latest market news and updates
- **Market Analysis**: Insights on market trends and movements
- **Multi-symbol Support**: Track multiple stocks simultaneously

### 🏦 Portfolio Analysis
- **Performance Metrics**: Calculate portfolio returns and performance
- **Diversification Analysis**: Assess portfolio diversification
- **Risk Assessment**: Evaluate portfolio risk levels
- **Investment Insights**: AI-generated investment recommendations

### 🤖 Intelligent Conversation
- **Intent Recognition**: Understand user requests in natural language
- **Context Awareness**: Maintain conversation context
- **Helpful Guidance**: Provide financial advice and tips
- **Multi-turn Conversations**: Support complex financial discussions

## 🛠️ Technical Architecture

### LangGraph Workflow
The agent uses a sophisticated LangGraph workflow with the following nodes:

1. **Intent Classifier**: Determines user intent from natural language
2. **Expense Tracker**: Handles expense tracking requests
3. **Budget Creator**: Manages budget creation and updates
4. **Market Data Fetcher**: Retrieves market information and news
5. **Portfolio Analyzer**: Analyzes investment portfolios
6. **General Helper**: Provides financial advice and guidance

### State Management
```python
class FinanceAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_data: UserData
    intent: str
    current_request: Dict[str, Any]
    market_data: Dict[str, Any]
```

### Tools & Functions
- **Expense Tracking Tools**: `track_expense_tool`
- **Budget Management Tools**: `create_budget_item_tool`, `calculate_budget_summary_tool`
- **Market Data Tools**: `fetch_market_news_tool`, `fetch_stock_price_tool`
- **Portfolio Analysis Tools**: `analyze_portfolio_tool`

## 📋 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- llama3.2 model pulled in Ollama

### Installation Steps

1. **Navigate to the project directory**:
   ```bash
   cd 03-Apps/02-Finance_Investment_Agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull the llama3.2 model
   ollama pull llama3.2
   ```

4. **Verify installation**:
   ```bash
   # Check if llama3.2 is available
   ollama list
   ```

## 🚀 Usage

### Interactive Mode
Run the main agent for interactive conversations:

```bash
python finance_investment_agent.py
```

### Demo Mode
Run comprehensive demonstrations of all features:

```bash
# Run full demo
python demo_finance_agent.py

# Run interactive demo
python demo_finance_agent.py --interactive
```

### Example Interactions

#### Expense Tracking
```
👤 User: I spent $45.50 on groceries at Whole Foods
🤖 Agent: ✅ Expense tracked: $45.50 for food - groceries at Whole Foods
```

#### Budget Creation
```
👤 User: Set up a monthly budget: $500 for food, $300 for transportation
🤖 Agent: ✅ Budget created:
• Budget item created: $500.00 allocated for food
• Budget item created: $300.00 allocated for transportation
```

#### Market Data
```
👤 User: What's the current price of Apple stock?
🤖 Agent: 📊 Market Data for AAPL:

📈 Price: AAPL: $175.25 (+2.15, +1.24%)

📰 News:
• AAPL stock shows strong performance in Q4 earnings report
• Market analysts upgrade AAPL rating to 'Buy'
• Industry trends favor AAPL sector growth
```

#### Portfolio Analysis
```
👤 User: Analyze my portfolio
🤖 Agent: 📊 Portfolio Analysis:
• Total Holdings: 5 positions
• Estimated Value: $12,500.00
• Diversification: Good
• Top Holdings: AAPL, GOOGL, TSLA
```

## 📁 Project Structure

```
03-Apps/02-Finance_Investment_Agent/
├── README.md                    # This documentation
├── finance_investment_agent.py  # Main agent implementation
├── demo_finance_agent.py        # Demonstration script
└── requirements.txt             # Python dependencies
```

## 🧪 Testing

### Running the Demo
```bash
# Full feature demonstration
python demo_finance_agent.py

# Interactive testing
python demo_finance_agent.py --interactive
```

### Testing Individual Features
The demo script includes separate functions for testing each feature:
- `demo_expense_tracking()` - Test expense tracking functionality
- `demo_budget_creation()` - Test budget creation features
- `demo_market_data()` - Test market data retrieval
- `demo_portfolio_analysis()` - Test portfolio analysis
- `demo_general_help()` - Test general conversation and help

## 🔧 Configuration

### Model Configuration
The agent is configured to use Ollama with llama3.2. To change the model, update the `ChatLiteLLM` configuration:

```python
llm = ChatLiteLLM(
    model="ollama/llama3.2",              # Change model here
    api_base="http://localhost:11434",    # Ollama server
    temperature=0.3
)
```

### API Endpoints
For production use, replace mock functions with real API calls:
- **Stock Data**: Yahoo Finance, Alpha Vantage, or Finnhub
- **News Data**: NewsAPI, Financial Modeling Prep
- **Portfolio Data**: Broker APIs or financial data providers

## 🌐 Architecture Overview

The Finance Agent uses a node-based architecture with LangGraph:

```
User Input → Intent Classifier → Route to Appropriate Node → Tool Execution → Response
```

**Workflow Nodes:**
1. **Intent Classifier**: Analyzes user input to determine intent
2. **Expense Tracker**: Processes expense tracking requests
3. **Budget Creator**: Handles budget creation and management
4. **Market Data**: Fetches stock prices and financial news
5. **Portfolio Analyzer**: Analyzes investment portfolios
6. **General Helper**: Provides financial advice and guidance

## 📈 Implementation Details

### Core Components

**Data Models:**
- `ExpenseDetails`: Stores expense information
- `BudgetItem`: Represents budget categories and allocations
- `Portfolio`: Manages investment holdings
- `UserData`: Comprehensive user financial profile

**Agent State:**
- `FinanceAgentState`: Maintains conversation state and user data
- Message history with LangGraph's `add_messages` annotation
- Intent tracking and request context

**Tools:**
- Expense tracking and categorization
- Budget creation and analysis
- Market data retrieval (mock implementation)
- Portfolio performance analysis

### LangGraph Integration

The agent uses LangGraph's `StateGraph` for workflow management:

```python
workflow = StateGraph(FinanceAgentState)
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("expense_tracker", expense_tracker_node)
# ... additional nodes
workflow.add_conditional_edges("intent_classifier", route_intent, {...})
```

### Ollama Integration

Uses LiteLLM for seamless Ollama integration:

```python
llm = ChatLiteLLM(
    model="ollama/llama3.2",
    api_base="http://localhost:11434",
    temperature=0.3
)
```

## 📈 Future Enhancements

### Planned Features
- **Real API Integration**: Connect to actual financial data providers
- **Data Persistence**: Save user data to database
- **Advanced Analytics**: More sophisticated portfolio analysis
- **Goal Tracking**: Financial goal setting and tracking
- **Alerts & Notifications**: Price alerts and budget notifications
- **Export Functionality**: Export data to Excel/CSV
- **Multi-user Support**: Support multiple user profiles

### Technical Improvements
- **Error Handling**: Enhanced error recovery and user feedback
- **Performance Optimization**: Caching and async operations
- **Security**: Data encryption and secure API handling
- **Testing**: Comprehensive unit and integration tests
- **Logging**: Detailed logging for debugging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is part of the LangGraph 101 tutorial series. Please refer to the main repository license.

## 🆘 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   ollama serve
   
   # Verify model availability
   ollama list
   ```

2. **Model Not Found**:
   ```bash
   # Pull the required model
   ollama pull llama3.2
   ```

3. **Import Errors**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Port Issues**:
   - Default Ollama port is 11434
   - Update `api_base` in code if using different port

### Performance Tips

- **Model Selection**: llama3.2 provides good balance of performance and speed
- **Temperature Settings**: Lower temperature (0.3) for more consistent responses
- **Caching**: Consider implementing response caching for repeated queries
- **Batch Processing**: Group similar operations for better performance

### Support
For issues and questions:
- Check the troubleshooting section above
- Review the demo script for usage examples
- Ensure all dependencies are properly installed
- Verify Ollama setup and model availability

---

**Built with ❤️ using LangGraph, Ollama, and llama3.2**

## 📊 Original Pseudocode Reference

The implementation follows this original pseudocode structure:

```
FUNCTION handle_finance_request(user_query, user_data)
  intent = IDENTIFY_INTENT(user_query)
  
  IF intent == "track_expense" THEN
    expense_details = PARSE_EXPENSE_DETAILS(user_query)
    ADD_EXPENSE(user_data, expense_details)
    response = "Expense tracked successfully."
  ELSEIF intent == "create_budget" THEN
    budget_parameters = PARSE_BUDGET_PARAMETERS(user_query)
    budget = CREATE_BUDGET(user_data, budget_parameters)
    response = GENERATE_BUDGET_SUMMARY(budget)
  ELSEIF intent == "get_market_data" THEN
    topic = PARSE_MARKET_TOPIC(user_query)
    market_news = FETCH_MARKET_NEWS(topic)
    stock_prices = FETCH_STOCK_PRICES(topic)
    response = AGGREGATE_MARKET_DATA(market_news, stock_prices)
  ELSEIF intent == "analyze_portfolio" THEN
    portfolio = GET_USER_PORTFOLIO(user_data)
    IF portfolio IS EMPTY THEN
      response = "You don't have a portfolio set up yet."
    ELSE
      analysis_results = ANALYZE_PORTFOLIO_PERFORMANCE(portfolio)
      investment_insights = GENERATE_INVESTMENT_INSIGHTS(analysis_results)
      response = COMBINE_ANALYSIS_AND_INSIGHTS(analysis_results, investment_insights)
    ENDIF
  ELSE
    response = "Sorry, I can't help with that finance request."
  ENDIF
  
  RETURN response
END FUNCTION
```

The LangGraph implementation enhances this pseudocode with:
- Sophisticated state management
- Tool-based architecture
- Natural language processing
- Conversational AI capabilities
- Extensible node-based workflow
