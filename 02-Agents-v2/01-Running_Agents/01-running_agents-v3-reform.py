#%%
# tutorial_langgraph_agents.py
# A Python script for a YouTube tutorial on running LangGraph agents
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from rich import print
#%%
# Constants
MODEL_NAME = "ollama:llama3.2"
# MODEL_NAME = "openai:o3-mini"
DEFAULT_WEATHER_TEMP = 72
MAX_ITERATIONS = 3
RECURSION_LIMIT = 2 * MAX_ITERATIONS + 1
WEATHER_QUERY = "What's the weather in SF?"
SF_LOCATION = "San Francisco"
#%%
def get_weather(query):
    """
    A mock function to fetch weather information for a given location.
    This is a placeholder for the tutorial and returns a static response.
    Args:
        query (str): The location to fetch weather for.
    Returns:
        str: A string describing the weather in the queried location.
    """
    print(f"Tool called: Fetching weather for {query}")
    return f"The weather in {query} is sunny with a high of {DEFAULT_WEATHER_TEMP}°F."
#%%
def create_agent():
    """Create and return a basic LangGraph agent with weather tool."""
    print("Creating a basic LangGraph agent...")
    return create_react_agent(model=MODEL_NAME, tools=[get_weather])

def create_weather_input(content=None):
    """Create a standard weather query input dictionary."""
    query = content or WEATHER_QUERY
    return {"messages": [{"role": "user", "content": query}]}

def demonstrate_sync_invocation(agent):
    """Demonstrate synchronous agent invocation."""
    print("\n=== Synchronous Invocation ===")
    sync_input = create_weather_input(f"What's the weather in {SF_LOCATION}?")
    sync_response = agent.invoke(sync_input)
    print("Synchronous Response:")
    print(sync_response)

def demonstrate_async_invocation():
    """Demonstrate asynchronous agent invocation (explanation only)."""
    print("\n=== Asynchronous Invocation ===")
    print("Asynchronous invocation would work similarly to sync but with 'await' in an async context.")

def demonstrate_input_formats():
    """Show different supported input formats."""
    print("\n=== Input Formats ===")
    input_examples = {
        "String input": {"messages": "Hello"},
        "Dictionary input": {"messages": {"role": "user", "content": "Hi there"}},
        "List of messages": {"messages": [{"role": "user", "content": "Hey!"}]},
        "Custom state input": {"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}
    }

    print("Different input formats are supported:")
    for format_name, example in input_examples.items():
        print(f"- {format_name}: {example}")

    print("Output is a dictionary with 'messages' and optional custom state fields.")

def demonstrate_streaming(agent):
    """Demonstrate streaming output functionality."""
    print("\n=== Streaming Output (Synchronous) ===")
    stream_input = create_weather_input()
    for chunk in agent.stream(stream_input, stream_mode="updates"):
        print("Stream chunk:", chunk)

def demonstrate_max_iterations(agent):
    """Demonstrate recursion limit and max iterations control."""
    print("\n=== Max Iterations with Recursion Limit ===")
    agent_with_limit = agent.with_config(recursion_limit=RECURSION_LIMIT)

    try:
        limit_input = create_weather_input()
        limit_response = agent_with_limit.invoke(limit_input)
        print("Response with recursion limit:", limit_response)
    except GraphRecursionError:
        print("Agent stopped due to max iterations.")

def print_tutorial_summary():
    """Print the tutorial conclusion and summary."""
    print("\n=== Tutorial Summary ===")
    print("In this tutorial, we covered running LangGraph agents with synchronous and asynchronous methods,")
    print("different input/output formats, streaming for incremental updates, and setting max iterations.")
    print("Experiment with these concepts to build powerful agent-based applications!")

def main():
    """Main tutorial execution function."""
    agent = create_agent()

    demonstrate_sync_invocation(agent)
    demonstrate_async_invocation()
    demonstrate_input_formats()
    demonstrate_streaming(agent)
    demonstrate_max_iterations(agent)
    print_tutorial_summary()

if __name__ == "__main__":
    main()