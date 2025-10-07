import streamlit as st
import yfinance as yf
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# model = ChatOpenAI(model="gpt-4.1-nano", max_tokens=500)
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# Section 1: Setup
# Install dependencies: pip install streamlit yfinance langchain langgraph ollama
# Ensure Ollama is installed and running locally with a model like 'llama2'

# Initialize Ollama LLM (assuming 'llama2' is available)
llm = ChatOllama(model='llama3.2', temperature=0.2)
embedding = OllamaEmbeddings(model='nomic-embed-text')


# Section 2: Data Fetching
def fetch_stock_data(ticker, period='1y'):
    """
    Fetch stock data using yfinance.
    Returns: info dict, historical prices, financials (income, balance, cashflow)
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period=period)
    income = stock.income_stmt
    balance = stock.balance_sheet
    cashflow = stock.cashflow

    return info, hist, income, balance, cashflow


# Section 3: Agent Creation
# Define a simple analysis prompt
analysis_prompt = PromptTemplate(
    input_variables=['ticker', 'summary', 'income', 'balance', 'cashflow'],
    template="""
    Analyze the following stock data for {ticker}:
    Company Summary: {summary}
    Income Statement: {income}
    Balance Sheet: {balance}
    Cash Flow: {cashflow}
    
    Provide a summary of financial health, key trends, and investment recommendations.
    """,
)

# Create LLM chain for analysis
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)


# LangGraph workflow (simple sequential)
def analyze_stock(state):
    ticker = state['ticker']
    info, hist, income, balance, cashflow = state['data']

    # Convert financials to string summaries
    summary = info.get('longBusinessSummary', 'No summary available')
    income_str = income.to_string() if not income.empty else 'No data'
    balance_str = balance.to_string() if not balance.empty else 'No data'
    cashflow_str = cashflow.to_string() if not cashflow.empty else 'No data'

    # Run analysis
    analysis = analysis_chain.run(
        ticker=ticker, summary=summary, income=income_str, balance=balance_str, cashflow=cashflow_str
    )

    state['analysis'] = analysis
    return state


# Build graph
graph = StateGraph(dict)
graph.add_node('analyze', analyze_stock)
graph.set_entry_point('analyze')
graph.add_edge('analyze', END)
app = graph.compile()

# Section 4: UI
st.title('Stock Analysis App')
st.write('Enter a stock ticker to analyze (e.g., AAPL, GOOGL)')

ticker = st.text_input('Stock Ticker', value='AAPL').upper()
period = st.selectbox('Period', ['1y', '2y', '5y'], index=0)

if st.button('Analyze'):
    try:
        # Fetch data
        info, hist, income, balance, cashflow = fetch_stock_data(ticker, period)

        # Display basic info
        st.subheader(f'{ticker} Overview')
        st.write(f'**Name:** {info.get("longName", "N/A")}')
        st.write(f'**Current Price:** ${info.get("currentPrice", "N/A")}')
        st.write(f'**Market Cap:** ${info.get("marketCap", "N/A")}')

        # Historical prices
        st.subheader('Historical Prices')
        st.line_chart(hist['Close'])

        # Financials
        st.subheader('Income Statement (Annual)')
        st.dataframe(income)
        st.subheader('Balance Sheet (Annual)')
        st.dataframe(balance)
        st.subheader('Cash Flow (Annual)')
        st.dataframe(cashflow)

        # Run agent analysis
        state = {'ticker': ticker, 'data': (info, hist, income, balance, cashflow)}
        result = app.invoke(state)

        st.subheader('AI Analysis')
        st.write(result['analysis'])

    except Exception as e:
        st.error(f'Error fetching data: {e}')

# Total lines: ~120 (well under 350)
