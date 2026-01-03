################################ Imports & Configuration ################################

import streamlit as st
import yfinance as yf
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL_NAME = 'gpt-oss:20b'

# Initialize the Large Language Model with a low temperature for consistent analysis
llm = ChatOllama(model=MODEL_NAME, temperature=0.2, base_url=OLLAMA_BASE_URL)


################################ Data Fetching Functions ################################

def fetch_stock_price(ticker: str) -> dict:
    """Fetch current stock price and basic company information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period='1y')

    # Return empty dict if no market data is found to prevent downstream errors
    if history.empty:
        return {}

    return {
        'ticker': ticker,
        'current_price': history['Close'].iloc[-1], # Get the most recent closing price
        'previous_close': info.get('previousClose', 0),
        'day_high': history['High'].iloc[-1],
        'day_low': history['Low'].iloc[-1],
        'volume': history['Volume'].iloc[-1],
        'market_cap': info.get('marketCap', 0),
        'company_name': info.get('longName', ticker),
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'peg_ratio': info.get('pegRatio'),
        'dividend_yield': info.get('dividendYield'),
        'target_mean_price': info.get('targetMeanPrice'),
        'recommendation_key': info.get('recommendationKey'),
        'fifty_day_avg': info.get('fiftyDayAverage'),
        'two_hundred_day_avg': info.get('twoHundredDayAverage'),
        'price_history': history['Close'],
    }


def get_recent_data(df, items):
    """Extract a list of items from the most recent column of a dataframe."""
    if df.empty:
        return {}
    # Use dictionary comprehension to map specific rows to their latest values
    return {item: df.loc[item].iloc[0] for item in items if item in df.index}


def fetch_financial_statements(ticker: str, period: str = 'yearly') -> dict:
    """Fetch balance sheet, income statement, and cash flow for a given period."""
    stock = yf.Ticker(ticker)

    # Toggle between quarterly and annual report data
    if period == 'quarterly':
        balance_sheet = stock.quarterly_balance_sheet
        income_stmt = stock.quarterly_income_stmt
        cashflow = stock.quarterly_cashflow
    else:
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cashflow = stock.cashflow

    # Define the specific accounting line items needed for analysis
    balance_items = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
    income_items = ['Total Revenue', 'Net Income', 'Operating Income', 'EBITDA']
    cashflow_items = ['Operating Cash Flow', 'Free Cash Flow']

    return {
        'balance_sheet': get_recent_data(balance_sheet, balance_items),
        'income_statement': get_recent_data(income_stmt, income_items),
        'cash_flow': get_recent_data(cashflow, cashflow_items),
        'period': period,
    }


################################ LangGraph State & Nodes ################################


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

    # Populate the state with fresh market and financial data
    state['price_data'] = fetch_stock_price(ticker)
    state['financial_data'] = fetch_financial_statements(ticker, period)
    state['messages'].append(f'✓ Fetched data for {ticker}')

    return state


def analyze_financials_node(state: AgentState) -> AgentState:
    """Node 2: Use an LLM to analyze the collected financial data."""
    price_data = state['price_data']
    financial_data = state['financial_data']

    # Construct a detailed prompt including the raw data context
    prompt = f"""
        You are a seasoned Wall Street financial analyst. Your task is to analyze the following stock data for {state['ticker']} and provide a professional assessment.

        ### 1. MARKET DATA
        - Company: {price_data.get('company_name')}
        - Current Price: ${price_data.get('current_price')}
        - Market Cap: ${price_data.get('market_cap')}
        - 50-Day Avg: ${price_data.get('fifty_day_avg')}
        - 200-Day Avg: ${price_data.get('two_hundred_day_avg')}
        
        ### 2. VALUATION & ANALYST SENTIMENT
        - P/E Ratio: {price_data.get('pe_ratio')}
        - Forward P/E: {price_data.get('forward_pe')}
        - PEG Ratio: {price_data.get('peg_ratio')}
        - Dividend Yield: {price_data.get('dividend_yield')}
        - Analyst Target: ${price_data.get('target_mean_price')}
        - Analyst Consensus: {price_data.get('recommendation_key')}

        ### 3. FINANCIAL STATEMENTS ({financial_data.get('period')})
        Balance Sheet Highlights: {json.dumps(financial_data.get('balance_sheet', {}), indent=2)}
        Income Statement Highlights: {json.dumps(financial_data.get('income_statement', {}), indent=2)}
        Cash Flow Highlights: {json.dumps(financial_data.get('cash_flow', {}), indent=2)}

        ### INSTRUCTIONS
        Provide a comprehensive but concise analysis (approx 400 words) covering:
        
        1. **Financial Health**: Assess profitability margins, liquidity, and solvency based on the statements.
        2. **Valuation Check**: Compare current price to moving averages and analyst targets. Interpret the P/E and PEG ratios.
        3. **Trend Analysis**: Identify positive or negative trends in revenue, net income, or cash flow.
        4. **Risk Assessment**: Highlight any potential red flags or areas of concern.

        Format your response in clean Markdown with clear headings.
        """
    state['analysis'] = llm.invoke(prompt).content # Store the LLM response in state
    state['messages'].append('✓ Completed financial analysis')

    return state


def generate_recommendation_node(state: AgentState) -> AgentState:
    """Node 3: Use an LLM to generate an investment recommendation."""
    # Build strategy based on the qualitative analysis from the previous node
    prompt = f"""
        You are a Senior Portfolio Manager. Based on the detailed analysis below for {state['ticker']}, formulate a strategic investment recommendation.

        ### FINANCIAL ANALYSIS
        {state['analysis']}

        ### INSTRUCTIONS
        Provide a structured recommendation including:
        
        1. **Executive Summary**: A 1-sentence verdict.
        2. **Investment Stance**: Clearly state Bullish, Bearish, or Neutral.
        3. **Actionable Rating**: Buy, Sell, or Hold.
        4. **Risk Profile**: Low, Medium, or High risk, with a brief explanation why.
        5. **Key Rationale**: The top 3 reasons supporting your decision.
        6. **Trading Strategy**: Suggest a specific trading approach (e.g., Dollar Cost Averaging, Buy the Dip, Options Strategy, or Immediate Entry/Exit) tailored to the current volatility and price action.

        Keep the tone professional and decisive. Limit to 250 words. Return in Markdown.
        """
    state['recommendation'] = llm.invoke(prompt).content
    state['messages'].append('✓ Generated investment recommendation')

    return state


################################ LangGraph Workflow Construction ################################


def create_stock_analysis_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register the functions as workflow nodes
    workflow.add_node('fetch_data', fetch_data_node)
    workflow.add_node('analyze_financials', analyze_financials_node)
    workflow.add_node('generate_recommendation', generate_recommendation_node)

    # Define the execution sequence (Directed Acyclic Graph)
    workflow.set_entry_point('fetch_data')
    workflow.add_edge('fetch_data', 'analyze_financials')
    workflow.add_edge('analyze_financials', 'generate_recommendation')
    workflow.add_edge('generate_recommendation', END)

    return workflow.compile()


################################ Streamlit UI ################################


def main():
    """Define the main Streamlit application layout and logic."""
    st.set_page_config(page_title='Stock Analysis Agent', page_icon='📈', layout='wide')

    st.title('📈 AI Stock Analysis Agent')
    st.markdown('Powered by **Ollama** + **LangChain** + **LangGraph** + **Streamlit**')

    # Sidebar for user inputs
    st.sidebar.header('⚙️ Configuration')
    ticker = st.sidebar.text_input('Stock Ticker', value='INTC').upper()
    period = st.sidebar.radio('Financial Period', ['yearly', 'quarterly'])
    analyze_button = st.sidebar.button('🔍 Analyze Stock', type='primary')

    # Maintain state across user interactions to prevent re-running the graph
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    if analyze_button:
        with st.spinner('🤖 AI Agent is analyzing...'):
            # Define the initial data structure to kick off the graph
            initial_state = {
                'ticker': ticker,
                'period': period,
                'price_data': {},
                'financial_data': {},
                'analysis': '',
                'recommendation': '',
                'messages': [],
            }
            graph = create_stock_analysis_graph()
            result = graph.invoke(initial_state)

            st.session_state.analysis_result = result
            st.session_state.chat_history = []

    # Check if a successful analysis exists before rendering results
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        st.success('Analysis complete!')
        # Show the step-by-step progress from the agent graph
        with st.expander('📋 Agent Progress Log'):
            for msg in result['messages']:
                st.write(msg)
            st.markdown("---")
            st.write("### Session State")
            st.write(st.session_state)

        tab1, tab2, tab3, tab4 = st.tabs(['📊 Price Data', '💰 Financials', '🔍 Analysis', '💡 Recommendation'])

        with tab1:
            st.subheader(f'{result["price_data"].get("company_name", ticker)}')

            st.markdown('### Key Metrics')
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('P/E Ratio', f'{result["price_data"].get("pe_ratio", "N/A"):.2f}')

            m3.metric('50-Day Avg', f'${result["price_data"].get("fifty_day_avg", 0):.2f}')
            m4.metric('Analyst Target', f'${result["price_data"].get("target_mean_price", 0):.2f}')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Current Price', f'${result["price_data"].get("current_price", 0):.2f}')
            with col2:
                # Calculate percentage change from previous day close
                prev = result['price_data'].get('previous_close', 0)
                curr = result['price_data'].get('current_price', 0)
                change = ((curr - prev) / prev) * 100 if prev else 0
                st.metric('Change %', f'{change:.2f}%', delta=f'{change:.2f}%')
            with col3:
                market_cap = result['price_data'].get('market_cap', 0)
                st.metric('Market Cap', f'${market_cap / 1e9:.2f}B')

            st.subheader('Price History (1 Year)')
            st.line_chart(result['price_data']['price_history'], height=400)

        with tab2:
            st.subheader(f'Financial Statements ({result["period"].capitalize()})')
            col1, col2, col3 = st.columns(3)
            # Display financial items normalized to Billions for readability
            with col1:
                st.write('**Balance Sheet**')
                for key, value in result['financial_data'].get('balance_sheet', {}).items():
                    st.write(f'{key}: ${value / 1e9:.2f}B')
            with col2:
                st.write('**Income Statement**')
                for key, value in result['financial_data'].get('income_statement', {}).items():
                    st.write(f'{key}: ${value / 1e9:.2f}B')
            with col3:
                st.write('**Cash Flow**')
                for key, value in result['financial_data'].get('cash_flow', {}).items():
                    st.write(f'{key}: ${value / 1e9:.2f}B')

        with tab3:
            st.subheader('🔍 AI Financial Analysis')
            st.markdown(result['analysis'])

        with tab4:
            st.subheader('💡 Investment Recommendation')
            st.markdown(result['recommendation'])

        ################################ Chat Functionality ################################
        st.markdown('---')
        
        col1, col2 = st.columns([6, 1])
        with col1:
            st.subheader(f'💬 Chat with {ticker} Analyst')
        with col2:
            if st.button('Clear Chat'):
                st.session_state.chat_history = []
                st.rerun()

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Render the existing conversation thread
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        # Handle user interaction and agent response
        if prompt := st.chat_input('Ask questions about the analysis...'):
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)

            with st.chat_message('assistant'):
                # Provide all gathered data as background context for the chat LLM
                context = f"""
                Ticker: {result['ticker']}
                Price Data: {result['price_data']}
                Financials: {result['financial_data']}
                Analysis: {result['analysis']}
                Recommendation: {result['recommendation']}
                """
                
                # Format previous messages to maintain continuity
                history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

                # Query the model with full situational awareness
                response = llm.invoke(f'Context:\n{context}\n\nChat History:\n{history}\n\nUser Question: {prompt}').content
                st.markdown(response)

            st.session_state.chat_history.append({'role': 'assistant', 'content': response})


if __name__ == '__main__':
    main()
