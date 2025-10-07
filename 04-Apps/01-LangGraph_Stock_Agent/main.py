# Stock Analysis App Tutorial
# This app uses Streamlit for the UI, yfinance for stock data, Ollama for local LLM,
# LangChain for agent logic, and LangGraph for workflow orchestration.
# Prerequisites:
# - Install dependencies: pip install streamlit yfinance langchain langchain-community langgraph ollama langchain-ollama plotly
# - Run Ollama locally with a model, e.g., ollama run llama3.2
# - No API key needed (using yfinance for simplicity; switch to Alpha Vantage if preferred, but add API key handling).
# Run the app: streamlit run stock_analysis_app.py
# Note: If using Alpha Vantage, create a .streamlit/secrets.toml file with your API key, e.g., ALPHA_VANTAGE_API_KEY = "your_key_here"

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_ollama import OllamaLLM  # Updated import to avoid deprecation
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated, Sequence
import operator

# --- Section 1: Setup ---
# Initialize Ollama LLM (assuming local Ollama server is running)
llm = OllamaLLM(model='llama3.2')  # Change to your preferred model

# Define ReAct prompt template
react_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


# Define a simple state for LangGraph
class GraphState(TypedDict):
    messages: Annotated[Sequence[AIMessage | HumanMessage], operator.add]
    stock_data: dict
    analysis: str


# --- Section 2: Data Fetching ---
@tool
def fetch_stock_data(ticker: str) -> dict:
    """Fetches real-time stock price, historical data, balance sheet, income statement, and cash flow for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')  # Fetch 1 year of historical data for charts
        data = {
            'price': stock.history(period='1d')['Close'].iloc[-1] if not stock.history(period='1d').empty else None,
            'historical': hist.to_dict() if not hist.empty else {},
            'balance_sheet_yearly': stock.balance_sheet.to_dict() if not stock.balance_sheet.empty else {},
            'balance_sheet_quarterly': stock.quarterly_balance_sheet.to_dict()
            if not stock.quarterly_balance_sheet.empty
            else {},
            'income_stmt_yearly': stock.financials.to_dict() if not stock.financials.empty else {},
            'income_stmt_quarterly': stock.quarterly_financials.to_dict()
            if not stock.quarterly_financials.empty
            else {},
            'cashflow_yearly': stock.cashflow.to_dict() if not stock.cashflow.empty else {},
            'cashflow_quarterly': stock.quarterly_cashflow.to_dict() if not stock.quarterly_cashflow.empty else {},
        }
        return data
    except Exception as e:
        return {'error': str(e)}


# --- Section 3: Agent Creation ---
# Define the analysis tool using LLM
@tool
def analyze_stock(data: dict) -> str:
    """Analyzes stock data and provides insights on financial health, trends, and recommendations."""
    if 'error' in data:
        return f'Error fetching data: {data["error"]}'
    prompt = PromptTemplate.from_template(
        'Analyze the following stock data for financial health, trends, and recommendations. Provide a structured summary:\n{data}'
    )
    chain = prompt | llm
    return chain.invoke({'data': str(data)})


# Create ReAct agent with tools and prompt
tools = [fetch_stock_data, analyze_stock]
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)  # Set verbose to False for cleaner output


# LangGraph workflow
def fetch_node(state):
    result = agent_executor.invoke({'input': f'Fetch data for {state["messages"][-1].content}'})
    return {'stock_data': result['output']}


def analyze_node(state):
    result = agent_executor.invoke({'input': f'Analyze the data: {state["stock_data"]}'})
    return {'analysis': result['output']}


def end_node(state):
    return state


workflow = StateGraph(GraphState)
workflow.add_node('fetch', fetch_node)
workflow.add_node('analyze', analyze_node)
workflow.add_node('end', end_node)

workflow.set_entry_point('fetch')
workflow.add_edge('fetch', 'analyze')
workflow.add_edge('analyze', 'end')

app = workflow.compile()

# --- Section 4: UI with Streamlit ---
st.set_page_config(page_title='Stock Analysis App', page_icon='📈', layout='wide')

# Sidebar for inputs
st.sidebar.title('📊 Stock Analyzer')
st.sidebar.markdown('Enter a stock ticker and select options for analysis.')

ticker = st.sidebar.text_input('Stock Ticker (e.g., AAPL)', placeholder='AAPL')
period = st.sidebar.selectbox('Historical Period', ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
analyze_button = st.sidebar.button('🔍 Analyze Stock')

# Main content
st.title('📈 Advanced Stock Analysis App')
st.markdown('Analyze stocks with AI-powered insights and visualizations.')

if analyze_button:
    if ticker.strip() == '':
        st.sidebar.error('Please enter a valid ticker symbol.')
    else:
        with st.spinner('Fetching and analyzing data...'):
            try:
                # Fetch historical data separately for charting
                stock = yf.Ticker(ticker.upper())
                hist = stock.history(period=period)

                # Run the LangGraph workflow for data and analysis
                initial_state = {'messages': [HumanMessage(content=ticker.upper())]}
                final_state = app.invoke(initial_state)

                stock_data = final_state.get('stock_data', {})
                analysis = final_state.get('analysis', 'No analysis available.')

                # Create tabs for organized display
                tab1, tab2, tab3 = st.tabs(['📊 Overview', '📈 Charts', '🤖 AI Analysis'])

                with tab1:
                    st.subheader(f'Overview for {ticker.upper()}')
                    if 'error' in stock_data:
                        st.error(f'Error: {stock_data["error"]}')
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                'Current Price',
                                f'${stock_data.get("price", "N/A"):.2f}' if stock_data.get('price') else 'N/A',
                            )
                        with col2:
                            if not hist.empty:
                                change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
                                pct_change = (change / hist['Close'].iloc[0]) * 100
                                st.metric(
                                    f'Change ({period})',
                                    f'{change:.2f} ({pct_change:.2f}%)',
                                    delta=f'{pct_change:.2f}%',
                                )
                            else:
                                st.metric(f'Change ({period})', 'N/A')

                        with st.expander('Financial Statements'):
                            st.json(stock_data)

                with tab2:
                    st.subheader(f'Price Chart for {ticker.upper()}')
                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
                        fig.update_layout(
                            title=f'{ticker.upper()} Stock Price ({period})',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning('No historical data available for charting.')

                with tab3:
                    st.subheader('AI-Powered Analysis')
                    st.write(analysis)

            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
                st.info('Please check the ticker symbol and try again.')
