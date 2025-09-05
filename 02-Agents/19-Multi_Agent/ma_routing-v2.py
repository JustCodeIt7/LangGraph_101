from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import re


# Define the state structure
class AgentState(TypedDict):
    query: str
    route: str
    response: str
    reasoning: str


# Mock LLM responses (replace with actual LLM calls)
def code_llm(query: str) -> str:
    """Specialized LLM for coding questions"""
    return f"[CODE LLM] Here's a programming solution for: {query}"


def math_llm(query: str) -> str:
    """Specialized LLM for math questions"""
    return f'[MATH LLM] Mathematical analysis of: {query}'


def general_llm(query: str) -> str:
    """General purpose LLM"""
    return f'[GENERAL LLM] General response to: {query}'


def creative_llm(query: str) -> str:
    """Specialized LLM for creative tasks"""
    return f'[CREATIVE LLM] Creative response to: {query}'


# Router function
def route_query(state: AgentState) -> AgentState:
    """Analyze query and determine which LLM to use"""
    query = state['query'].lower()

    # Code-related keywords
    code_keywords = [
        'python',
        'javascript',
        'code',
        'function',
        'programming',
        'algorithm',
        'debug',
        'api',
        'class',
        'variable',
    ]

    # Math-related keywords
    math_keywords = [
        'calculate',
        'equation',
        'math',
        'formula',
        'solve',
        'probability',
        'statistics',
        'derivative',
        'integral',
    ]

    # Creative keywords
    creative_keywords = ['story', 'poem', 'creative', 'write', 'imagine', 'fictional', 'character', 'plot', 'narrative']

    # Determine route based on keyword matching
    if any(keyword in query for keyword in code_keywords):
        route = 'code'
        reasoning = 'Detected programming/coding related query'
    elif any(keyword in query for keyword in math_keywords):
        route = 'math'
        reasoning = 'Detected mathematical/calculation query'
    elif any(keyword in query for keyword in creative_keywords):
        route = 'creative'
        reasoning = 'Detected creative writing query'
    else:
        route = 'general'
        reasoning = 'No specific specialization detected, using general LLM'

    state['route'] = route
    state['reasoning'] = reasoning
    return state


# Individual LLM nodes
def handle_code_query(state: AgentState) -> AgentState:
    """Process query with code-specialized LLM"""
    response = code_llm(state['query'])
    state['response'] = response
    return state


def handle_math_query(state: AgentState) -> AgentState:
    """Process query with math-specialized LLM"""
    response = math_llm(state['query'])
    state['response'] = response
    return state


def handle_creative_query(state: AgentState) -> AgentState:
    """Process query with creative-specialized LLM"""
    response = creative_llm(state['query'])
    state['response'] = response
    return state


def handle_general_query(state: AgentState) -> AgentState:
    """Process query with general LLM"""
    response = general_llm(state['query'])
    state['response'] = response
    return state


# Conditional edge function
def determine_next_node(state: AgentState) -> Literal['code', 'math', 'creative', 'general']:
    """Return the next node based on routing decision"""
    return state['route']


# Build the graph
def create_routing_agent():
    """Create and return the LangGraph routing agent"""

    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node('router', route_query)
    workflow.add_node('code', handle_code_query)
    workflow.add_node('math', handle_math_query)
    workflow.add_node('creative', handle_creative_query)
    workflow.add_node('general', handle_general_query)

    # Set entry point
    workflow.set_entry_point('router')

    # Add conditional edges from router to specialized nodes
    workflow.add_conditional_edges(
        'router', determine_next_node, {'code': 'code', 'math': 'math', 'creative': 'creative', 'general': 'general'}
    )

    # Add edges from specialized nodes to END
    workflow.add_edge('code', END)
    workflow.add_edge('math', END)
    workflow.add_edge('creative', END)
    workflow.add_edge('general', END)

    # Compile the graph
    app = workflow.compile()
    return app


# Example usage
def run_example():
    """Demonstrate the LLM-powered routing agent with different query types"""

    agent = create_routing_agent()

    test_queries = [
        'How do I implement a binary search algorithm in Python?',
        "What's the integral of 2x³ + 5x² - 3x + 7?",
        'Write a mysterious short story about a lighthouse keeper',
        'What causes the northern lights phenomenon?',
        'Help me optimize this SQL query performance',
        'Solve for x: 3x² - 12x + 9 = 0',
    ]

    print('LangGraph LLM-Powered Routing Agent Demo')
    print('=' * 50)
    print('Note: Set your OpenAI API key to see actual LLM responses')
    print('=' * 50)

    for i, query in enumerate(test_queries, 1):
        print(f'\n{i}. Query: {query}')

        try:
            # Run the agent
            result = agent.invoke({'query': query})

            print(f'   Route: {result["route"]}')
            print(f'   Reasoning: {result["reasoning"]}')
            print(f'   Response: {result["response"][:100]}...')  # Truncate for demo

        except Exception as e:
            print(f'   Error: {str(e)}')

        print('-' * 50)


if __name__ == '__main__':
    run_example()
