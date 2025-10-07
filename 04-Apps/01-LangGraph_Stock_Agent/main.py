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
# embedding = OllamaEmbeddings(model='nomic-embed-text')

# llm = ChatOpenAI(model="gpt-4.1-nano", max_tokens=500)
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")


################################ Data Fetching Functions ################################
def fetch_stock_price(ticker: str) -> dict:
    """Fetch current stock price and basic company information."""


def fetch_financial_statements(ticker: str, period: str = 'yearly') -> dict:
    """Fetch balance sheet, income statement, and cash flow for a given period."""


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


def analyze_financials_node(state: AgentState) -> AgentState:
    """Node 2: Use an LLM to analyze the collected financial data."""


def generate_recommendation_node(state: AgentState) -> AgentState:
    """Node 3: Use an LLM to generate an investment recommendation."""


################################ LangGraph Workflow Construction ################################


def create_stock_analysis_graph():
    """Build and compile the LangGraph workflow."""


################################ Streamlit UI ################################


def main():
    """Define the main Streamlit application layout and logic."""


# Define the entry point for the script
if __name__ == '__main__':
    main()
