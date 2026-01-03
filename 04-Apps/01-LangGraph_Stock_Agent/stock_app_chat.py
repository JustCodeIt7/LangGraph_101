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


def get_recent_data(df, items):
    """Extract a list of items from the most recent column of a dataframe."""


def fetch_financial_statements(ticker: str, period: str = 'yearly') -> dict:
    """Fetch balance sheet, income statement, and cash flow for a given period."""


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


if __name__ == '__main__':
    main()