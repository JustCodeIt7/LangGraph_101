################################ Imports & Configuration ################################

import streamlit as st
from tests import MODEL
import yfinance as yf
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file if present

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL_NAME = 'gpt-oss'
# MODEL_NAME = 'llama3.2'

llm = ChatOllama(model=MODEL_NAME, temperature=0.2, base_url=OLLAMA_BASE_URL)
# llm = ChatOpenAI(model="gpt-5-nano",temperature=0.2)


################################ Data Fetching Functions ################################
def fetch_stock_price(ticker: str) -> dict:
    """Fetch current stock price and basic company information."""
    # Initialize the yfinance Ticker object
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period='1d')  # Get price data for the last trading day

    # Structure and return the essential price data
    return {
        'ticker': ticker,
        'current_price': history['Close'].iloc[-1],
        'previous_close': info.get('previousClose', 0),
        'day_high': history['High'].iloc[-1],
        'day_low': history['Low'].iloc[-1],
        'volume': history['Volume'].iloc[-1],
        'market_cap': info.get('marketCap', 0),
        'company_name': info.get('longName', ticker),
    }

def get_recent_data(df, items):
    """Extract a list of items from the most recent column of a dataframe."""
    # Return an empty dict if the dataframe is empty
    if df.empty:
        return {}
    result = {}
    # Iterate through requested items and extract the latest value
    for item in items:
        if item in df.index:
            result[item] = df.loc[item].iloc[0]  # .iloc[0] gets the most recent data
    return result

def fetch_financial_statements(ticker: str, period: str = 'yearly') -> dict:
    """Fetch balance sheet, income statement, and cash flow for a given period."""
    stock = yf.Ticker(ticker)

    # Select the appropriate financial statements based on the chosen period
    if period == 'quarterly':
        balance_sheet = stock.quarterly_balance_sheet
        income_stmt = stock.quarterly_income_stmt
        cashflow = stock.quarterly_cashflow
    else:
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cashflow = stock.cashflow

    # Define the key financial metrics to extract from each statement
    balance_items = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
    income_items = ['Total Revenue', 'Net Income', 'Operating Income', 'EBITDA']
    cashflow_items = ['Operating Cash Flow', 'Free Cash Flow']

    # Consolidate the financial data into a single dictionary
    return {
        'balance_sheet': get_recent_data(balance_sheet, balance_items),
        'income_statement': get_recent_data(income_stmt, income_items),
        'cash_flow': get_recent_data(cashflow, cashflow_items),
        'period': period,
    }


################################ LangGraph State & Nodes ################################


# Define the state that will be passed between nodes in the graph
class AgentState(TypedDict):
    """Define the state structure for our agent."""

    ticker: str
    period: str
    price_data: dict
    financial_data: dict
    analysis: str
    recommendation: str
    messages: list

def fetch_data_node(state: AgentState) -> AgentState:
    """Node 1: Fetch all required stock data from yfinance."""
    ticker = state['ticker']
    period = state['period']

    # Populate the state with data from the fetching functions
    state['price_data'] = fetch_stock_price(ticker)
    state['financial_data'] = fetch_financial_statements(ticker, period)
    state['messages'].append(f'✓ Fetched data for {ticker}')

    return state


def analyze_financials_node(state: AgentState) -> AgentState:
    """Node 2: Use an LLM to analyze the collected financial data."""
    # Initialize the LLM for the analysis task
    # llm = ChatOllama(model=MODEL_NAME, temperature=0.2, base_url=OLLAMA_BASE_URL)
    price_data = state['price_data']
    financial_data = state['financial_data']

    # Create a detailed prompt for the LLM to perform financial analysis
    prompt = f"""
        Analyze the following stock data for {state['ticker']}:

        CURRENT PRICE INFORMATION:
        - Company: {price_data.get('company_name')}
        - Current Price: ${price_data.get('current_price')}
        - Market Cap: ${price_data.get('market_cap')}

        FINANCIAL STATEMENTS ({financial_data.get('period')}):
        Balance Sheet: {json.dumps(financial_data.get('balance_sheet', {}), indent=2)}
        Income Statement: {json.dumps(financial_data.get('income_statement', {}), indent=2)}
        Cash Flow: {json.dumps(financial_data.get('cash_flow', {}), indent=2)}

        Provide a concise analysis of:
        1. Financial health (profitability, liquidity, solvency)
        2. Key trends and observations
        3. Strengths and weaknesses

        Keep your analysis under 300 words. Return the analysis in markdown format.
        """
    # Invoke the LLM and update the state with the analysis
    state['analysis'] = llm.invoke(prompt).content
    state['messages'].append('✓ Completed financial analysis')

    return state


def generate_recommendation_node(state: AgentState) -> AgentState:
    """Node 3: Use an LLM to generate an investment recommendation."""
    # Initialize a separate LLM instance for the recommendation task

    # llm = ChatOllama(model='gpt-oss:latest', temperature=0.2, base_url=OLLAMA_BASE_URL)
    # Create a prompt that uses the prior analysis to generate a recommendation
    prompt = f"""
        Based on the following analysis for {state['ticker']}:

        {state['analysis']}

        Provide a clear investment recommendation:
        1. Overall sentiment (Bullish/Bearish/Neutral)
        2. Risk level (Low/Medium/High)
        3. Brief recommendation (Buy/Hold/Sell) with reasoning

        Keep your recommendation under 200 words and be direct. Return the recommendation in markdown format.
        """
    # Invoke the LLM and update the state with the recommendation
    state['recommendation'] = llm.invoke(prompt).content
    state['messages'].append('✓ Generated investment recommendation')

    return state


################################ LangGraph Workflow Construction ################################


def create_stock_analysis_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register the functions as nodes in the graph
    workflow.add_node('fetch_data', fetch_data_node)
    workflow.add_node('analyze_financials', analyze_financials_node)
    workflow.add_node('generate_recommendation', generate_recommendation_node)

    # Define the sequential flow of the agent's tasks
    workflow.set_entry_point('fetch_data')
    workflow.add_edge('fetch_data', 'analyze_financials')
    workflow.add_edge('analyze_financials', 'generate_recommendation')
    workflow.add_edge('generate_recommendation', END)  # Mark the final node
    app = workflow.compile()
    # save the compiled graph for debugging
    app.get_graph().draw_mermaid_png(output_file_path='agent_graph.png')
    # Compile the graph into a runnable object
    return app


################################ Streamlit UI ################################


def main():
    """Define the main Streamlit application layout and logic."""
    st.set_page_config(page_title='Stock Analysis Agent', page_icon='📈', layout='wide')

    st.title('📈 AI Stock Analysis Agent')
    st.markdown('Powered by **Ollama** + **LangChain** + **LangGraph** + **Streamlit**')

    # Configure user inputs in the sidebar
    st.sidebar.header('⚙️ Configuration')
    ticker = st.sidebar.text_input('Stock Ticker', value='INTC').upper()
    period = st.sidebar.radio('Financial Period', ['yearly', 'quarterly'])
    analyze_button = st.sidebar.button('🔍 Analyze Stock', type='primary')

    # Execute the analysis when the user clicks the button
    if analyze_button:
        # Display a spinner while the agent is working
        with st.spinner('🤖 AI Agent is analyzing...'):
            # Define the initial state for the LangGraph agent
            initial_state = {
                'ticker': ticker,
                'period': period,
                'price_data': {},
                'financial_data': {},
                'analysis': '',
                'recommendation': '',
                'messages': [],
            }

            # Create and run the analysis graph
            graph = create_stock_analysis_graph()
            result = graph.invoke(initial_state)

            # Display a success message and the agent's progress log
            st.success('Analysis complete!')
            with st.expander('📋 Agent Progress Log'):
                for msg in result['messages']:
                    st.write(msg)

            with st.expander('View Full Results:'):
                st.write(result)


            # Organize and display the results in separate tabs
            tab1, tab2, tab3, tab4 = st.tabs(['📊 Price Data', '💰 Financials', '🔍 Analysis', '💡 Recommendation'])

            with tab1:
                st.subheader(f'{result["price_data"].get("company_name", ticker)}')
                col1, col2, col3 = st.columns(3)

                # Display key price metrics
                with col1:
                    st.metric('Current Price', f'${result["price_data"].get("current_price"):.2f}')
                with col2:
                    prev = result['price_data'].get('previous_close', 0)
                    curr = result['price_data'].get('current_price', 0)
                    change = ((curr - prev) / prev) * 100 if prev else 0
                    st.metric('Change %', f'{change:.2f}%', delta=f'{change:.2f}%')
                with col3:
                    market_cap = result['price_data'].get('market_cap', 0)
                    st.metric('Market Cap', f'${market_cap / 1e9:.2f}B')  # Format in billions

            with tab2:
                st.subheader(f'Financial Statements ({period.capitalize()})')
                col1, col2, col3 = st.columns(3)

                # Display balance sheet data
                with col1:
                    st.write('**Balance Sheet**')
                    for key, value in result['financial_data'].get('balance_sheet', {}).items():
                        st.write(f'{key}: ${value / 1e9:.2f}B')  # Format in billions

                # Display income statement data
                with col2:
                    st.write('**Income Statement**')
                    for key, value in result['financial_data'].get('income_statement', {}).items():
                        st.write(f'{key}: ${value / 1e9:.2f}B')  # Format in billions

                # Display cash flow data
                with col3:
                    st.write('**Cash Flow**')
                    for key, value in result['financial_data'].get('cash_flow', {}).items():
                        st.write(f'{key}: ${value / 1e9:.2f}B')  # Format in billions

            with tab3:
                st.subheader('🔍 AI Financial Analysis')
                st.markdown(result['analysis'])

            with tab4:
                st.subheader('💡 Investment Recommendation')
                st.markdown(result['recommendation'])


# Define the entry point for the script
if __name__ == '__main__':
    main()
