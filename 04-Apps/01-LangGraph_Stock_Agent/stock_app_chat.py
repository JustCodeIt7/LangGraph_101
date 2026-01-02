########### Imports & Configuration ########

import streamlit as st
import yfinance as yf
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file if present

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL_NAME = 'llama3.2'

llm = ChatOllama(model=MODEL_NAME, temperature=0.2, base_url=OLLAMA_BASE_URL)


################################ Data Fetching Functions ###################
def fetch_stock_price(ticker: str) -> dict:
    """Fetch current stock price and basic company information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period='1d')

    if history.empty:
        return {}

    return {
        'ticker': ticker,
        'current_price': history['Close'].iloc[-1],
        'previous_close': info.get('previousClose', 0),
        'day_high': history['High'].iloc[-1],
        'day_low': history['Low'].iloc[-1],
        'volume': history['Volume'].iloc[-1],
        'market_cap': info.get('marketCap', 0),
        'company_name': info.get('longName', ticker),
        # Valuation Metrics
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'peg_ratio': info.get('pegRatio'),
        'dividend_yield': info.get('dividendYield'),
        # Analyst Data
        'target_mean_price': info.get('targetMeanPrice'),
        'recommendation_key': info.get('recommendationKey'),
        # Technicals
        'fifty_day_avg': info.get('fiftyDayAverage'),
        'two_hundred_day_avg': info.get('twoHundredDayAverage'),
    }


def get_recent_data(df, items):
    """Extract a list of items from the most recent column of a dataframe."""
    if df.empty:
        return {}
    return {item: df.loc[item].iloc[0] for item in items if item in df.index}


def fetch_financial_statements(ticker: str, period: str = 'yearly') -> dict:
    """Fetch balance sheet, income statement, and cash flow for a given period."""
    stock = yf.Ticker(ticker)

    if period == 'quarterly':
        balance_sheet = stock.quarterly_balance_sheet
        income_stmt = stock.quarterly_income_stmt
        cashflow = stock.quarterly_cashflow
    else:
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cashflow = stock.cashflow

    balance_items = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
    income_items = ['Total Revenue', 'Net Income', 'Operating Income', 'EBITDA']
    cashflow_items = ['Operating Cash Flow', 'Free Cash Flow']

    return {
        'balance_sheet': get_recent_data(balance_sheet, balance_items),
        'income_statement': get_recent_data(income_stmt, income_items),
        'cash_flow': get_recent_data(cashflow, cashflow_items),
        'period': period,
    }


################################ LangGraph State & Nodes ######################


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

    state['price_data'] = fetch_stock_price(ticker)
    state['financial_data'] = fetch_financial_statements(ticker, period)
    state['messages'].append(f'✓ Fetched data for {ticker}')

    return state


def analyze_financials_node(state: AgentState) -> AgentState:
    """Node 2: Use an LLM to analyze the collected financial data."""
    price_data = state['price_data']
    financial_data = state['financial_data']

    prompt = f"""
        Analyze the following stock data for {state['ticker']}:

        CURRENT PRICE INFORMATION:
        - Company: {price_data.get('company_name')}
        - Current Price: ${price_data.get('current_price')}
        - Market Cap: ${price_data.get('market_cap')}
        
        VALUATION & TECHNICALS:
        - P/E Ratio: {price_data.get('pe_ratio')}
        - Forward P/E: {price_data.get('forward_pe')}
        - PEG Ratio: {price_data.get('peg_ratio')}
        - Dividend Yield: {price_data.get('dividend_yield')}
        - 50-Day Avg: ${price_data.get('fifty_day_avg')}
        - 200-Day Avg: ${price_data.get('two_hundred_day_avg')}
        - Analyst Target: ${price_data.get('target_mean_price')}
        - Analyst Rec: {price_data.get('recommendation_key')}

        FINANCIAL STATEMENTS ({financial_data.get('period')}):
        Balance Sheet: {json.dumps(financial_data.get('balance_sheet', {}), indent=2)}
        Income Statement: {json.dumps(financial_data.get('income_statement', {}), indent=2)}
        Cash Flow: {json.dumps(financial_data.get('cash_flow', {}), indent=2)}

        Provide a concise analysis of:
        1. Financial health (profitability, liquidity, solvency)
        2. Valuation analysis (is it over/undervalued?)
        3. Key trends and observations
        4. Strengths and weaknesses

        Keep your analysis under 300 words. Return the analysis in markdown format.
        """
    state['analysis'] = llm.invoke(prompt).content
    state['messages'].append('✓ Completed financial analysis')

    return state


def generate_recommendation_node(state: AgentState) -> AgentState:
    """Node 3: Use an LLM to generate an investment recommendation."""
    prompt = f"""
        Based on the following analysis for {state['ticker']}:

        {state['analysis']}

        Provide a clear investment recommendation:
        1. Overall sentiment (Bullish/Bearish/Neutral)
        2. Risk level (Low/Medium/High)
        3. Brief recommendation (Buy/Hold/Sell) with reasoning

        Keep your recommendation under 200 words and be direct. Return the recommendation in markdown format.
        """
    state['recommendation'] = llm.invoke(prompt).content
    state['messages'].append('✓ Generated investment recommendation')

    return state


################################ LangGraph Workflow Construction #################


def create_stock_analysis_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node('fetch_data', fetch_data_node)
    workflow.add_node('analyze_financials', analyze_financials_node)
    workflow.add_node('generate_recommendation', generate_recommendation_node)

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

    st.sidebar.header('⚙️ Configuration')
    ticker = st.sidebar.text_input('Stock Ticker', value='INTC').upper()
    period = st.sidebar.radio('Financial Period', ['yearly', 'quarterly'])
    analyze_button = st.sidebar.button('🔍 Analyze Stock', type='primary')

    # Initialize session state for results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    if analyze_button:
        with st.spinner('🤖 AI Agent is analyzing...'):
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

            # Store result in session state and reset chat history
            st.session_state.analysis_result = result
            st.session_state.chat_history = []

    # Display results if they exist in session state
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        st.success('Analysis complete!')
        with st.expander('📋 Agent Progress Log'):
            for msg in result['messages']:
                st.write(msg)

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
                prev = result['price_data'].get('previous_close', 0)
                curr = result['price_data'].get('current_price', 0)
                change = ((curr - prev) / prev) * 100 if prev else 0
                st.metric('Change %', f'{change:.2f}%', delta=f'{change:.2f}%')
            with col3:
                market_cap = result['price_data'].get('market_cap', 0)
                st.metric('Market Cap', f'${market_cap / 1e9:.2f}B')

        with tab2:
            st.subheader(f'Financial Statements ({result["period"].capitalize()})')
            col1, col2, col3 = st.columns(3)
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

        # ---------------- CHAT FUNCTIONALITY ----------------
        st.markdown('---')
        st.subheader(f'💬 Chat with {ticker} Analyst')

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        # Chat input
        if prompt := st.chat_input('Ask questions about the analysis...'):
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)

            with st.chat_message('assistant'):
                # Construct context from the analysis result
                context = f"""
                Ticker: {result['ticker']}
                Price Data: {result['price_data']}
                Financials: {result['financial_data']}
                Analysis: {result['analysis']}
                Recommendation: {result['recommendation']}
                """
                
                # Build conversation history
                history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

                # Generate response
                response = llm.invoke(f'Context:\n{context}\n\nChat History:\n{history}\n\nUser Question: {prompt}').content
                st.markdown(response)

            st.session_state.chat_history.append({'role': 'assistant', 'content': response})


if __name__ == '__main__':
    main()
