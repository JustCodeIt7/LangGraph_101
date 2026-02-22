# 04 - Applications

This directory contains practical applications built with LangGraph, demonstrating real-world use cases and complete implementations of AI-powered applications.

## Table of Contents

1. [Stock Trading Agent (`01-Stock_Agent`)](#01-stock_agent)
2. [Travel Planning Agent (`02-Travel_Planning`)] (#02-travel_planning)
3. [Customer Support Bot (`03-Customer_Support_Bot`)](#03-customer_support_bot)
4. [Crawl4AI Web Scraper (`04-Crawl4AI`)](##-crawl4ai-web-scraper)
5. [Conditional Routing (`05-Conditional_Routing`)](##-conditional-routing)
6. [Looping Logic (`06-Looping_Logic`)](##-looping-logic)
7. [Basic Chat Graph (`07-Basic_Chat_Graph`)](##-basic-chat-graph)
8. [Running Agents (`08-Running_Agents`)](##-running-agents)
9. [Agent State (`09-Agent-State`)](##-agent-state)
10. [Travel Planning Agent v2 (`10-Travel_Planning_v2`)](##-travel-planning-agent-v2)
11. [Personalized Health/Fitness Planner (`11-Personalized_Health_Fitness_Planner`)](##-personalized-health-fitness-planner)
12. [E-commerce Recommendation Agent (`12-E-commerce_Recommendation_Agent`)](##-e-commerce-recommendation-agent)
13. [DeepResearch Agent (`13-DeepResearch_Agent`)](##-deepresearch-agent)
14. [Invoice Parser (`14-Invoice_Parser`)](##-invoice-parser)
15. [Coding Agent (`15-Codeing_Agent`)](##-coding-agent)

---

## 01. Stock Trading Agent

A sophisticated trading application that uses AI agents to analyze market data and execute trades.

### Key Features

- **Market Data Analysis**: Real-time stock price monitoring and historical trend analysis.
- **Technical Indicators**: Implementation of common trading indicators (RSI, MACD, Moving Averages).
- **Trading Strategy**: Configurable strategies based on technical analysis.
- **Portfolio Management**: Track holdings, calculate returns, and manage risk.

### How It Works

1. Agent fetches current market data for specified stocks
2. Technical indicators are calculated from historical prices
3. Trading strategy evaluates conditions and generates signals
4. Risk management checks ensure compliance with portfolio rules
5. Trade execution is triggered based on the analysis

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Market Data │ ──► │ Analysis Engine  │ ──► │   Strategy  │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                   │
                    ┌──────────────────┐           ▼
                    │ Risk Management  │ ◄── Decision
                    └──────────────────┘
```

---

## 02. Research Assistant

An AI-powered research tool that helps users gather, analyze, and synthesize information from multiple sources.

### Key Features

- **Multi-source Search**: Query multiple data sources simultaneously.
- **Information Synthesis**: Combine findings into coherent summaries.
- **Citation Tracking**: Maintain references for all gathered information.
- **Follow-up Questions**: Ability to dig deeper into specific topics.

### How It Works

1. User submits a research query
2. Agent decomposes the question into searcheable components
3. Multiple sources are queried in parallel
4. Results are analyzed and synthesized
5. A comprehensive response is generated with proper citations

---

## 03. Customer Support Bot

An intelligent customer service agent that handles support inquiries with human-like conversation capabilities.

### Key Features

- **Intent Recognition**: Understands what the user needs (technical support, billing, general inquiry).
- **Knowledge Base Integration**: Retrieves relevant help articles and documentation.
- **Ticket Escalation**: Identifies when issues need human intervention.
- **Conversation History**: Maintains context across multi-turn conversations.

### How It Works

1. User submits a support request
2. Bot analyzes the intent and determines the category
3. Relevant knowledge base articles are retrieved
4. A response is generated combining AI capabilities with documented solutions
5. If unresolved, the conversation can be escalated to human agents

### Conversation Flow

```
User Query → Intent Classification → Knowledge Retrieval 
                                    ↓
                              Response Generation ← Context
                                    ↓
                              User Response
```

---

## 02. Travel Planning Agent

An AI-powered travel planning assistant that helps users create detailed travel itineraries.

### Key Features
- **Destination Research**: Gathers information about destinations
- **Itinerary Building**: Creates day-by-day schedules
- **Budget Planning**: Estimates costs for flights, hotels, and activities
- **Recommendation Engine**: Suggests attractions, restaurants, and activities

---

## 04. Crawl4AI Web Scraper

A web scraping agent that uses Crawl4AI to gather information from websites.

### Key Features
- **Web Scraping**: Extract content from URLs
- **Content Processing**: Parses and structures scraped data
- **Multi-page Navigation**: Handles pagination and site navigation
- **Data Extraction**: Identifies and extracts relevant information

---

## 05. Conditional Routing

Demonstrates conditional routing patterns in LangGraph for decision-making flows.

### Key Features
- **Dynamic Flow Control**: Routes based on user input or analysis
- **Multiple Paths**: Supports branching logic with different outcomes
- **State-based Decisions**: Makes routing decisions based on current state

---

## 06. Looping Logic

Implements looping patterns in LangGraph for iterative workflows.

### Key Features
- **Iteration Support**: Allows repeated execution of nodes
- **Termination Conditions**: Defines when to exit loops
- **State Accumulation**: Builds up results across iterations

---

## 07. Basic Chat Graph

A foundational chat application demonstrating core LangGraph concepts.

### Key Features
- **Message Handling**: Manages conversation history
- **State Management**: Maintains context across interactions
- **Node Functions**: Processes and responds to user messages

---

## 08. Running Agents

Examples of running different types of agents with LangGraph.

### Key Features
- **Agent Execution**: Demonstrates various agent run patterns
- **Tool Integration**: Shows how to connect tools to agents
- **Response Handling**: Processes agent outputs

---

## 09. Agent State

Advanced state management patterns for complex agent behaviors.

### Key Features
- **State Schemas**: Defines structured state with TypedDict
- **State Transitions**: Manages state changes across nodes
- **Persistence**: Saves and restores agent state

---

## 10. Travel Planning Agent v2

An enhanced version of the travel planning agent with additional features.

### Key Features
- **Advanced Itinerary Building**: More sophisticated scheduling
- **Multi-modal Inputs**: Handles various input types
- **Enhanced Recommendations**: Improved suggestion algorithms

---

## 11. Personalized Health/Fitness Planner

An AI agent that creates customized fitness and nutrition plans.

### Key Features
- **User Profile Analysis**: Considers age, goals, preferences
- **Workout Generation**: Creates exercise routines
- **Meal Planning**: Suggests nutrition plans
- **Progress Tracking**: Monitors and adjusts plans

---

## 12. E-commerce Recommendation Agent

A product recommendation system for e-commerce applications.

### Key Features
- **Product Analysis**: Understands product attributes
- **User Preferences**: Learns from user behavior
- **Recommendation Engine**: Suggests relevant products
- **Personalization**: Tailors suggestions to individual users

---

## 13. DeepResearch Agent

An advanced research agent for deep information gathering and synthesis.

### Key Features
- **Multi-source Search**: Queries multiple data sources
- **Information Synthesis**: Combines findings into coherent summaries
- **Citation Tracking**: Maintains references for all gathered information
- **Follow-up Questions**: Handles iterative research queries

---

## 14. Invoice Parser

An agent that extracts and processes information from invoice documents.

### Key Features
- **OCR Integration**: Extracts text from scanned invoices
- **Data Extraction**: Parses key fields (dates, amounts, line items)
- **Validation**: Verifies extracted data accuracy
- **Export**: Outputs structured data for further processing

---

## 15. Coding Agent

An AI agent designed to assist with code generation and debugging.

### Key Features
- **Code Generation**: Writes code based on requirements
- **Debugging Assistance**: Identifies and fixes bugs
- **Code Review**: Analyzes code quality
- **Explanation**: Provides detailed code explanations

---

## Common Patterns Used

These applications demonstrate several common LangGraph patterns:

- **State Management**: Using `TypedDict` to maintain application state across interactions.
- **Tool Integration**: Connecting external APIs (market data, search engines, databases).
- **Conditional Routing**: Directing flow based on user intent or analysis results.
- **Checkpointing**: Saving conversation history for persistence and recovery.

## Running the Applications

Each subdirectory contains its own setup instructions. Generally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Check individual README files in each directory for specific configuration requirements.