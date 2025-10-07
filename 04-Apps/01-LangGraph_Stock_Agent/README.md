# 📈 AI Stock Analysis Agent

A beginner-friendly stock analysis application built with **Streamlit**, **Ollama**, **LangChain**, and **LangGraph**. This app fetches real-time stock data and uses AI to provide financial analysis and investment recommendations.

![Stock Analysis Agent](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 📊 **Real-time Stock Data**: Current prices, market cap, volume, and daily changes
- 💰 **Financial Statements**: Balance sheets, income statements, and cash flow (yearly/quarterly)
- 🤖 **AI-Powered Analysis**: Uses local LLM (Ollama) for intelligent financial analysis
- 💡 **Investment Recommendations**: Get AI-generated buy/hold/sell recommendations
- 🔄 **LangGraph Workflow**: Orchestrated agent workflow with clear data processing pipeline
- 🎨 **Beautiful UI**: Clean Streamlit interface with tabs and metrics

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (llama2)
- **Agent Framework**: LangChain + LangGraph
- **Data Source**: Yahoo Finance (yfinance)
- **Language**: Python 3.10+

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Python 3.10+** installed
2. **Ollama** installed and running locally
   - Download from: https://ollama.ai
   - Install llama2 model: `ollama pull llama2`

## 🚀 Installation

### Step 1: Clone the Repository

```bash
cd 01-LangGraph_Stock_Agent
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Ollama Installation

```bash
# Check if Ollama is running
ollama list

# Pull llama2 model if not already installed
ollama pull llama2
```

## 🎯 Usage

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Enter Stock Ticker**: Type any valid stock symbol (e.g., AAPL, MSFT, GOOGL, TSLA)
2. **Select Period**: Choose between yearly or quarterly financial data
3. **Click Analyze**: The AI agent will:
   - Fetch real-time stock prices
   - Retrieve financial statements
   - Analyze financial health
   - Generate investment recommendations

### Example Tickers to Try

- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- **TSLA** - Tesla Inc.
- **NVDA** - NVIDIA Corporation
- **AMZN** - Amazon.com Inc.

## 📁 Project Structure

```
01-LangGraph_Stock_Agent/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Code Structure (Tutorial Sections)

The code is organized into 5 clear sections for easy learning:

### Section 1: Imports and Configuration
- All necessary imports
- Basic setup

### Section 2: Data Fetching Functions
- `fetch_stock_price()`: Gets current price and basic info
- `fetch_financial_statements()`: Retrieves balance sheet, income statement, cash flow

### Section 3: LangGraph State and Nodes
- `AgentState`: Defines the state structure
- `fetch_data_node()`: Node for fetching stock data
- `analyze_financials_node()`: Node for AI analysis
- `generate_recommendation_node()`: Node for generating recommendations

### Section 4: LangGraph Workflow
- `create_stock_analysis_graph()`: Builds and compiles the agent workflow

### Section 5: Streamlit UI
- `main()`: Complete Streamlit interface with tabs and visualizations

## 🎓 Learning Path

This project is perfect for learning:

1. **Streamlit Basics**: Building interactive web apps
2. **LangChain**: Working with LLMs and prompts
3. **LangGraph**: Creating stateful agent workflows
4. **API Integration**: Fetching real-time data
5. **Financial Data**: Understanding stock metrics

## ⚙️ Configuration

### Change LLM Model

To use a different Ollama model, modify the `llm` initialization in `app.py`:

```python
llm = Ollama(model="mistral", temperature=0.3)  # Use mistral instead of llama2
```

Available models:
- `llama2` (default)
- `mistral`
- `codellama`
- `phi`

### Adjust Analysis Temperature

Higher temperature = more creative, Lower = more focused:

```python
llm = Ollama(model="llama2", temperature=0.7)  # More creative
```

## 🐛 Troubleshooting

### "Connection Error" when running app

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found" error

**Solution**: Pull the required model:
```bash
ollama pull llama2
```

### Yahoo Finance data not loading

**Solution**: Check your internet connection and try a different ticker symbol. Some tickers may have limited data.

### Slow LLM responses

**Solution**:
- Use a smaller model (phi)
- Reduce the amount of data being analyzed
- Ensure Ollama has sufficient system resources

## 📊 Sample Output

The app provides:

1. **Price Data Tab**
   - Current price, change %, day high/low, market cap

2. **Financials Tab**
   - Balance sheet: Assets, liabilities, equity
   - Income statement: Revenue, net income, EBITDA
   - Cash flow: Operating, investing, financing cash flows

3. **Analysis Tab**
   - AI-generated financial health assessment
   - Key trends and observations
   - Strengths and weaknesses

4. **Recommendation Tab**
   - Overall sentiment (Bullish/Bearish/Neutral)
   - Risk level
   - Buy/Hold/Sell recommendation

## 🤝 Contributing

This is a tutorial project! Feel free to:
- Add new features
- Improve the UI
- Add more financial metrics
- Integrate additional data sources

## 📝 License

MIT License - Feel free to use this code for learning and projects!

## 🙏 Acknowledgments

- **Streamlit** for the amazing framework
- **LangChain** for LLM orchestration
- **Ollama** for local LLM inference
- **Yahoo Finance** for free stock data

## 📺 YouTube Tutorial

*Coming soon!*

## 💡 Future Enhancements

- [ ] Add historical price charts
- [ ] Compare multiple stocks
- [ ] Technical indicators (RSI, MACD, etc.)
- [ ] Save analysis history
- [ ] Export reports to PDF
- [ ] Add more LLM providers (OpenAI, Anthropic)
- [ ] Stock screening functionality

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Ensure all prerequisites are installed
3. Verify Ollama is running with llama2 model

---

**Happy Stock Analyzing! 📈🤖**
