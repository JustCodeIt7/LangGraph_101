# Stock Analysis App Tutorial
# This app uses Streamlit for the UI, yfinance for stock data, Ollama for local LLM,
# LangChain for agent logic, and LangGraph for workflow orchestration.
# Prerequisites:
# - Install dependencies: pip install streamlit yfinance langchain langchain-community langgraph ollama langchain-ollama
# - Run Ollama locally with a model, e.g., ollama run llama2
# - No API key needed (using yfinance for simplicity; switch to Alpha Vantage if preferred, but add API key handling).
# Run the app: streamlit run this_file.py
# Note: If using Alpha Vantage, create a .streamlit/secrets.toml file with your API key, e.g., ALPHA_VANTAGE_API_KEY = "your_key_here"

import streamlit as st
import yfinance as yf
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
    """Fetches real-time stock price, balance sheet, income statement, and cash flow for a given ticker."""
    stock = yf.Ticker(ticker)
    data = {
        'price': stock.history(period='1d')['Close'].iloc[-1] if not stock.history(period='1d').empty else None,
        'balance_sheet_yearly': stock.balance_sheet.to_dict(),
        'balance_sheet_quarterly': stock.quarterly_balance_sheet.to_dict(),
        'income_stmt_yearly': stock.financials.to_dict(),
        'income_stmt_quarterly': stock.quarterly_financials.to_dict(),
        'cashflow_yearly': stock.cashflow.to_dict(),
        'cashflow_quarterly': stock.quarterly_cashflow.to_dict(),
    }
    return data


# --- Section 3: Agent Creation ---
# Define the analysis tool using LLM
@tool
def analyze_stock(data: dict) -> str:
    """Analyzes stock data and provides insights on financial health, trends, and recommendations."""
    prompt = PromptTemplate.from_template(
        'Analyze the following stock data for financial health, trends, and recommendations:\n{data}'
    )
    chain = prompt | llm
    return chain.invoke({'data': str(data)})


# Create ReAct agent with tools and prompt
tools = [fetch_stock_data, analyze_stock]
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


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
st.title('Stock Analysis App')

ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):')

if st.button('Analyze'):
    if ticker:
        with st.spinner('Fetching and analyzing...'):
            # Run the LangGraph workflow
            initial_state = {'messages': [HumanMessage(content=ticker)]}
            final_state = app.invoke(initial_state)

            # Display results
            st.subheader('Stock Data')
            st.json(final_state['stock_data'])

            st.subheader('Analysis')
            st.write(final_state['analysis'])
    else:
        st.error('Please enter a ticker symbol.')

# End of code. Total lines: ~100 (excluding comments and blanks for brevity)
# Fixes:
# - Added required 'prompt' to create_react_agent().
# - Switched to OllamaLLM to address deprecation warning (requires pip install langchain-ollama).
# - If using Alpha Vantage instead of yfinance, handle API key via st.secrets and modify fetch_stock_data accordingly.
# - For secrets error: Ensure .streamlit/secrets.toml exists in your project dir with the key if needed.
