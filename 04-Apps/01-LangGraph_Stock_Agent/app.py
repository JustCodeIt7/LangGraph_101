"""
Stock Analysis App with Streamlit, Ollama, LangChain & LangGraph
A beginner-friendly tutorial for building an AI-powered stock analysis agent
"""

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import TypedDict, Annotated
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import json


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def fetch_stock_price(ticker: str) -> dict:
    """Fetch current stock price and basic info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period='1d')

        return {
            'ticker': ticker,
            'current_price': history['Close'].iloc[-1] if not history.empty else 'N/A',
            'previous_close': info.get('previousClose', 'N/A'),
            'day_high': history['High'].iloc[-1] if not history.empty else 'N/A',
            'day_low': history['Low'].iloc[-1] if not history.empty else 'N/A',
            'volume': history['Volume'].iloc[-1] if not history.empty else 'N/A',
            'market_cap': info.get('marketCap', 'N/A'),
            'company_name': info.get('longName', ticker),
        }
    except Exception as e:
        return {'error': str(e)}


def fetch_financial_statements(ticker: str, period: str = 'yearly') -> dict:
    """Fetch balance sheet, income statement, and cash flow"""
    try:
        stock = yf.Ticker(ticker)

        # Select period (yearly or quarterly)
        if period == 'quarterly':
            balance_sheet = stock.quarterly_balance_sheet
            income_stmt = stock.quarterly_income_stmt
            cashflow = stock.quarterly_cashflow
        else:
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cashflow = stock.cashflow

        # Get most recent data (first column)
        def get_recent_data(df, items):
            if df.empty:
                return {}
            result = {}
            for item in items:
                if item in df.index:
                    result[item] = df.loc[item].iloc[0] if not df.loc[item].empty else 'N/A'
            return result

        # Key metrics to extract
        balance_items = [
            'Total Assets',
            'Total Liabilities Net Minority Interest',
            'Stockholders Equity',
            'Cash And Cash Equivalents',
        ]
        income_items = ['Total Revenue', 'Net Income', 'Operating Income', 'Gross Profit', 'EBITDA']
        cashflow_items = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']

        return {
            'balance_sheet': get_recent_data(balance_sheet, balance_items),
            'income_statement': get_recent_data(income_stmt, income_items),
            'cash_flow': get_recent_data(cashflow, cashflow_items),
            'period': period,
        }
    except Exception as e:
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: LANGGRAPH STATE AND NODES
# ═══════════════════════════════════════════════════════════════════════════


# Define the state structure for our agent
class AgentState(TypedDict):
    ticker: str
    period: str
    price_data: dict
    financial_data: dict
    analysis: str
    recommendation: str
    messages: list


def fetch_data_node(state: AgentState) -> AgentState:
    """Node 1: Fetch all stock data"""
    ticker = state['ticker']
    period = state['period']

    # Fetch price and financial data
    price_data = fetch_stock_price(ticker)
    financial_data = fetch_financial_statements(ticker, period)

    state['price_data'] = price_data
    state['financial_data'] = financial_data
    state['messages'].append(f'✓ Fetched data for {ticker}')

    return state


def analyze_financials_node(state: AgentState) -> AgentState:
    """Node 2: Analyze financial health using LLM"""

    # Initialize Ollama LLM (using llama3.2 by default)
    llm = Ollama(model='llama3.2', temperature=0.3)

    # Prepare context for analysis
    price_data = state['price_data']
    financial_data = state['financial_data']

    prompt = f"""
Analyze the following stock data for {state['ticker']}:

CURRENT PRICE INFORMATION:
- Company: {price_data.get('company_name', 'N/A')}
- Current Price: ${price_data.get('current_price', 'N/A')}
- Market Cap: ${price_data.get('market_cap', 'N/A')}

FINANCIAL STATEMENTS ({financial_data.get('period', 'N/A')}):
Balance Sheet: {json.dumps(financial_data.get('balance_sheet', {}), indent=2)}
Income Statement: {json.dumps(financial_data.get('income_statement', {}), indent=2)}
Cash Flow: {json.dumps(financial_data.get('cash_flow', {}), indent=2)}

Provide a concise analysis of:
1. Financial health (profitability, liquidity, solvency)
2. Key trends and observations
3. Strengths and weaknesses

Keep your analysis under 200 words.
"""

    # Get LLM analysis
    analysis = llm.invoke(prompt)
    state['analysis'] = analysis
    state['messages'].append('✓ Completed financial analysis')

    return state


def generate_recommendation_node(state: AgentState) -> AgentState:
    """Node 3: Generate investment recommendation"""

    llm = Ollama(model='llama3.2', temperature=0.3)

    prompt = f"""
Based on the following analysis for {state['ticker']}:

{state['analysis']}

Provide a clear investment recommendation:
1. Overall sentiment (Bullish/Bearish/Neutral)
2. Risk level (Low/Medium/High)
3. Brief recommendation (Buy/Hold/Sell) with reasoning

Keep your recommendation under 100 words and be direct.
"""

    recommendation = llm.invoke(prompt)
    state['recommendation'] = recommendation
    state['messages'].append('✓ Generated investment recommendation')

    return state


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: LANGGRAPH WORKFLOW CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


def create_stock_analysis_graph():
    """Build the LangGraph workflow"""

    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node('fetch_data', fetch_data_node)
    workflow.add_node('analyze_financials', analyze_financials_node)
    workflow.add_node('generate_recommendation', generate_recommendation_node)

    # Define the flow
    workflow.set_entry_point('fetch_data')
    workflow.add_edge('fetch_data', 'analyze_financials')
    workflow.add_edge('analyze_financials', 'generate_recommendation')
    workflow.add_edge('generate_recommendation', END)

    # Compile the graph
    app = workflow.compile()

    return app


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════


def format_large_number(num):
    """Format large numbers for display"""
    if isinstance(num, (int, float)):
        if abs(num) >= 1e9:
            return f'${num / 1e9:.2f}B'
        elif abs(num) >= 1e6:
            return f'${num / 1e6:.2f}M'
        else:
            return f'${num:,.2f}'
    return num


def main():
    """Main Streamlit application"""

    # Page configuration
    st.set_page_config(page_title='Stock Analysis Agent', page_icon='📈', layout='wide')

    # Header
    st.title('📈 AI Stock Analysis Agent')
    st.markdown("""
    Powered by **Ollama** + **LangChain** + **LangGraph** + **Streamlit**

    Enter a stock ticker to get real-time data and AI-powered analysis!
    """)

    # Sidebar inputs
    st.sidebar.header('⚙️ Configuration')
    ticker = st.sidebar.text_input(
        'Stock Ticker', value='AAPL', help='Enter ticker symbol (e.g., AAPL, MSFT, TSLA)'
    ).upper()
    period = st.sidebar.radio('Financial Period', ['yearly', 'quarterly'])

    analyze_button = st.sidebar.button('🔍 Analyze Stock', type='primary')

    # Main analysis section
    if analyze_button:
        with st.spinner('🤖 AI Agent is analyzing...'):
            # Create initial state
            initial_state = {
                'ticker': ticker,
                'period': period,
                'price_data': {},
                'financial_data': {},
                'analysis': '',
                'recommendation': '',
                'messages': [],
            }

            # Create and run the graph
            graph = create_stock_analysis_graph()

            try:
                # Execute the workflow
                result = graph.invoke(initial_state)

                # Display progress messages
                st.success('Analysis complete!')
                with st.expander('📋 Agent Progress Log'):
                    for msg in result['messages']:
                        st.write(msg)

                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(['📊 Price Data', '💰 Financials', '🔍 Analysis', '💡 Recommendation'])

                with tab1:
                    st.subheader(f'{result["price_data"].get("company_name", ticker)} - Current Price')

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric('Current Price', f'${result["price_data"].get("current_price", "N/A"):.2f}')
                    with col2:
                        prev_close = result['price_data'].get('previous_close', 0)
                        current = result['price_data'].get('current_price', 0)
                        if prev_close and current:
                            change = ((current - prev_close) / prev_close) * 100
                            st.metric('Change', f'{change:.2f}%')
                    with col3:
                        st.metric('Day High', f'${result["price_data"].get("day_high", "N/A"):.2f}')
                    with col4:
                        st.metric('Market Cap', format_large_number(result['price_data'].get('market_cap', 'N/A')))

                with tab2:
                    st.subheader(f'Financial Statements ({period.capitalize()})')

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write('**Balance Sheet**')
                        bs = result['financial_data'].get('balance_sheet', {})
                        for key, value in bs.items():
                            st.write(f'• {key}: {format_large_number(value)}')

                    with col2:
                        st.write('**Income Statement**')
                        inc = result['financial_data'].get('income_statement', {})
                        for key, value in inc.items():
                            st.write(f'• {key}: {format_large_number(value)}')

                    with col3:
                        st.write('**Cash Flow**')
                        cf = result['financial_data'].get('cash_flow', {})
                        for key, value in cf.items():
                            st.write(f'• {key}: {format_large_number(value)}')

                with tab3:
                    st.subheader('🔍 AI Financial Analysis')
                    st.write(result['analysis'])

                with tab4:
                    st.subheader('💡 Investment Recommendation')
                    st.write(result['recommendation'])

            except Exception as e:
                st.error(f'Error during analysis: {str(e)}')
                st.info('Make sure Ollama is running and llama3.2 model is installed.')


if __name__ == '__main__':
    main()
