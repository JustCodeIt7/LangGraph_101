# %%
# Import necessary modules for building and running a LangGraph agent
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from rich import print
import asyncio
import nest_asyncio  # Allows nested event loops in Jupyter environments


# %%
####################################################################
########## Step 1: Create a Basic Agent ############################
####################################################################
# Define a mock tool function to simulate fetching weather data
def get_weather(query):
    """
    A mock function to fetch weather information for a given location.
    Returns a static response for demonstration purposes.

    Args:
        query (str): The location to fetch weather for.
    Returns:
        str: A description of the weather at the specified location.
    """
    print(f'[Tool] Fetching weather for {query}')  # Log invocation
    return f'The weather in {query} is sunny with a high of 72°F.'


def get_greetings(query):
    """
    A mock function to handle greetings.
    Returns how the llm should greet the user
    Args:
        query (str): The user's input.

    Returns:
        str: A greeting response.
    """
    print(f'[Tool] Responding to greeting: {query}')
    return f'responding to greeting: {query}'


# Instantiate a LangGraph agent using the mock weather tool
print('Creating a LangGraph agent with a weather tool...')
agent = create_react_agent(
    model='ollama:llama3.2',  # Replace with your preferred model endpoint
    tools=[get_weather, get_greetings],  # Supply the get_weather and greetings functions as tools
)

# %%
####################################################################
########## Step 2: Synchronous Invocation ##########################
####################################################################
print('\n=== Synchronous Invocation ===')
# Prepare input in the format expected by the agent (.invoke)
sync_input = {'messages': [{'role': 'user', 'content': "What's the weather in San Francisco?"}]}
# sync_input = {'messages': [{'role': 'user', 'content': 'hello how are you?'}]}
# Invoke the agent synchronously and capture the full response
sync_response = agent.invoke(sync_input)
print('Synchronous Response:')
print(sync_response)


# %%
####################################################################
########## Step 3: Input Formats ########################
####################################################################
input_variants = {
    'string': {'messages': 'Hello, how are you?'},
    'dict': {'messages': {'role': 'user', 'content': 'Hi there'}},
    'list': {
        'messages': [{'role': 'user', 'content': 'Hey!'}, {'role': 'user', 'content': 'What is the weather in SF?'}]
    },
}

# Loop through each format, invoke the agent, and print the response
type_responses = {}
for fmt_name, fmt_input in input_variants.items():
    print('#' * (70))
    print(f'-- Testing input format: {fmt_name} --')
    response = agent.invoke(fmt_input)
    print('Agent Response:', response)
    type_responses[fmt_name] = response

# %%
####################################################################
########## Step 4: Streaming Output ###############################
####################################################################
print('\n=== Streaming Output Example ===')
stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}

# 1) Stream updates: emit after each agent step
for chunk in agent.stream(stream_input, stream_mode='updates'):
    print('[Update]', chunk)
# %%
# 2) Stream tokens: outputs tokens as the LLM generates them
for token, metadata in agent.stream(stream_input, stream_mode='messages'):
    print(token.content, end='')

# %%
# 3) Stream custom: includes tool outputs and metadata
for token, metadata in agent.stream(stream_input, stream_mode='custom'):
    print(token)
full_response = agent.invoke({'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]})
print(full_response)


# %%
####################################################################
########## Step 6: Max Iterations to Prevent Infinite Loops ########
####################################################################
# stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}
stream_input = {'messages': [{'role': 'user', 'content': 'hello how are you?'}]}
print('\n=== Recursion Limit Configuration ===')
# Define maximum agent steps to avoid runaway loops
recursion_limit = 5
# recursion_limit = 5
# Create a new agent instance with the specified recursion limit
agent_with_limit = agent.with_config(recursion_limit=recursion_limit)

try:
    limit_response = agent_with_limit.invoke(stream_input)
    print('Response within recursion limit:', limit_response)
except GraphRecursionError:
    print('Error: Agent stopped to prevent infinite recursion')

# %%
