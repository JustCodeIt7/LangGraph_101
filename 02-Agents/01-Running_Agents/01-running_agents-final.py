# %%
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from rich import print
import asyncio
import nest_asyncio

# %%
####################################################################
########## Step 1: Create a Basic Agent ############################
####################################################################


# For this tutorial, we'll simulate a simple weather tool function
# In a real scenario, this would call an API or perform a real task
def get_weather(query):
    """
    A mock function to fetch weather information for a given location.
    This is a placeholder for the tutorial and returns a static response.
    Args:
        query (str): The location to fetch weather for.
    Returns:
        str: A string describing the weather in the queried location.
    """
    print(f'Tool called: Fetching weather for {query}')
    return f'The weather in {query} is sunny with a high of 72°F.'


# Step 1: Create a basic agent with a model and tools
# Replace 'model' with an actual model in a real application
print('Creating a basic LangGraph agent...')
agent = create_react_agent(
    # model="openai:gpt-4.1-nano",  # Placeholder model
    model='ollama:llama3.2',  # Placeholder model
    tools=[get_weather],
)

# %%
####################################################################
########## Step 2: Synchronous Invocation ##########################
####################################################################

# Using .invoke() to get a full response in one go
print('\n=== Synchronous Invocation ===')
sync_input = {'messages': [{'role': 'user', 'content': "What's the weather in San Francisco?"}]}
sync_response = agent.invoke(sync_input)
print('Synchronous Response:')
print(sync_response)


# %%


# %%
####################################################################
########## Step 3: Input and Output Formats ########################
####################################################################

# Demonstrating different input formats
print('\n=== Input Formats ===')
input_string = {'messages': 'Hello'}  # Converted to HumanMessage
input_dict = {'messages': {'role': 'user', 'content': 'Hi there'}}
input_list = {'messages': [{'role': 'user', 'content': 'Hey!'}]}
input_custom = {'messages': [{'role': 'user', 'content': 'Hello'}], 'user_name': 'Alice'}
print('Different input formats are supported:')
print('- String input:', input_string)
print('- Dictionary input:', input_dict)
print('- List of messages:', input_list)
print('- Custom state input:', input_custom)
# Output format includes all messages exchanged during execution
print("Output is a dictionary with 'messages' and optional custom state fields.")

# %%
####################################################################
########## Step 4: Streaming Output ###############################
####################################################################

# Using .stream() for incremental updates
# %%
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from rich import print
import asyncio
import nest_asyncio

# %%
####################################################################
########## Step 1: Create a Basic Agent ############################
####################################################################


# For this tutorial, we'll simulate a simple weather tool function
# In a real scenario, this would call an API or perform a real task
def get_weather(query):
    """
    A mock function to fetch weather information for a given location.
    This is a placeholder for the tutorial and returns a static response.
    Args:
        query (str): The location to fetch weather for.
    Returns:
        str: A string describing the weather in the queried location.
    """
    print(f'Tool called: Fetching weather for {query}')
    return f'The weather in {query} is sunny with a high of 72°F.'


# Step 1: Create a basic agent with a model and tools
# Replace 'model' with an actual model in a real application
print('Creating a basic LangGraph agent...')
agent = create_react_agent(
    # model="openai:gpt-4.1-nano",  # Placeholder model
    model='ollama:llama3.2',  # Placeholder model
    tools=[get_weather],
)

# %%
####################################################################
########## Step 2: Synchronous Invocation ##########################
####################################################################

# Using .invoke() to get a full response in one go
print('\n=== Synchronous Invocation ===')
sync_input = {'messages': [{'role': 'user', 'content': "What's the weather in San Francisco?"}]}
sync_response = agent.invoke(sync_input)
print('Synchronous Response:')
print(sync_response)


# %%


# %%
####################################################################
########## Step 3: Input and Output Formats ########################
####################################################################

# Demonstrating different input formats
print('\n=== Input Formats ===')
print('\n=== Input Formats and Agent Responses ===')
# Define various input formats to test how the agent handles them
input_variants = {
    'string': {'messages': 'Hello, how are you?'},
    'dict': {'messages': {'role': 'user', 'content': 'Hi there'}},
    'list': {'messages': [{'role': 'user', 'content': 'Hey!'}]},
    'custom': {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}], 'user_name': 'Alice'},
}

# Loop through each format, invoke the agent, and print the response
type_responses = {}
for fmt_name, fmt_input in input_variants.items():
    print(f'\n-- Testing input format: {fmt_name} --')

    response = agent.invoke(fmt_input)
    print('Agent Response:', response)
    type_responses[fmt_name] = response


# %%
####################################################################
########## Step 4: Streaming Output ###############################
####################################################################

# Using .stream() for incremental updates
# update: emits an event after every agent step
print('\n=== Streaming Output (Synchronous) ===')
stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}

for chunk in agent.stream(stream_input, stream_mode='updates'):
    print('Stream chunk:', chunk)


# %%
# messages: stream tokens as they are produced by the LLM
stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}
for token, metadata in agent.stream(stream_input, stream_mode='messages'):
    print(token.content, end='')

# %%
# custom: stream updates from tools as they are executed
stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}
for token, metadata in agent.stream(
    {'messages': [{'role': 'user', 'content': 'what is the weather in sf'}]}, stream_mode='custom'
):
    print(token.content, end='')
    # print('Metadata', metadata)


# %%
# %%
####################################################################
########## Step 5: Asynchronous Invocation #########################
####################################################################

# Using await .ainvoke() for async execution (simulated here with a comment)
print('\n=== Asynchronous Invocation ===')


async def async_example():
    # streaming modes updates , messages, custom
    async for chunk in agent.astream(
        {'messages': [{'role': 'user', 'content': 'what is the weather in sf'}]}, stream_mode='custom'
    ):
        print('chunk', chunk)
        print('\n')
    # In a real async environment, you would use:
    response = await agent.ainvoke({'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]})
    return response


nest_asyncio.apply()

asyncio.run(async_example())

# %%
####################################################################
########## Step 6: Max Iterations to Prevent Infinite Loops ########
####################################################################

# Setting a recursion limit to control execution
print('\n=== Max Iterations with Recursion Limit ===')
max_iterations = 3
recursion_limit = 2 * max_iterations + 1
agent_with_limit = agent.with_config(recursion_limit=recursion_limit)
try:
    limit_response = agent_with_limit.invoke({'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]})
    print('Response with recursion limit:', limit_response)
except GraphRecursionError:
    print('Agent stopped due to max iterations.')


# %%
print('\n=== Streaming Output (Synchronous) ===')
stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}

for chunk in agent.stream(stream_input, stream_mode='updates'):
    print('Stream chunk:', chunk)


# %%

stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}
for token, metadata in agent.stream(stream_input, stream_mode='messages'):
    print(token.content, end='')

# %%
stream_input = {'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]}
for token, metadata in agent.stream(
    {'messages': [{'role': 'user', 'content': 'what is the weather in sf'}]}, stream_mode='custom'
):
    print(token.content, end='')
    # print('Metadata', metadata)


# %%
# %%
####################################################################
########## Step 5: Asynchronous Invocation #########################
####################################################################

# Using await .ainvoke() for async execution (simulated here with a comment)
print('\n=== Asynchronous Invocation ===')


async def async_example():
    # streaming modes updates , messages, custom
    # update: emits an event after every agent step
    # messages: stream tokens as they are produced by the LLM
    # custom: stream updates from tools as they are executed
    async for chunk in agent.astream(
        {'messages': [{'role': 'user', 'content': 'what is the weather in sf'}]}, stream_mode='custom'
    ):
        print('chunk', chunk)
        print('\n')
    # In a real async environment, you would use:
    response = await agent.ainvoke({'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]})
    return response


nest_asyncio.apply()

asyncio.run(async_example())

# %%
####################################################################
########## Step 6: Max Iterations to Prevent Infinite Loops ########
####################################################################

# Setting a recursion limit to control execution
print('\n=== Max Iterations with Recursion Limit ===')
max_iterations = 3
recursion_limit = 2 * max_iterations + 1
agent_with_limit = agent.with_config(recursion_limit=recursion_limit)
try:
    limit_response = agent_with_limit.invoke({'messages': [{'role': 'user', 'content': "What's the weather in SF?"}]})
    print('Response with recursion limit:', limit_response)
except GraphRecursionError:
    print('Agent stopped due to max iterations.')


# %%
