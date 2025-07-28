import asyncio
from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# 1. Define the state for our graph
#    'messages' will hold the conversation history.
class State(TypedDict):
    messages: List[BaseMessage]


# 2. Define a mock LLM node
#    This function simulates an LLM call. It checks the conversation
#    history for the user's name and responds accordingly.
def mock_llm(state: State) -> dict:
    """A mock LLM node that responds based on conversation history."""
    last_message = state['messages'][-1].content.lower()
    if 'my name is' in last_message:
        name = last_message.split('my name is')[-1].strip().capitalize()
        response = f'Nice to meet you, {name}!'
    elif "what's my name" in last_message:
        # Search for the name in the message history
        name_found = ''
        for msg in reversed(state['messages'][:-1]):
            if 'my name is' in msg.content.lower():
                name_found = msg.content.lower().split('my name is')[-1].strip().capitalize()
                break
        if name_found:
            response = f'Your name is {name_found}.'
        else:
            response = "I don't know your name."
    else:
        response = "Hello! I'm a simple bot. Tell me your name!"

    return {'messages': [AIMessage(content=response)]}


# 3. Set up the graph
#    We use 'add_messages' as the reducer for the 'messages' state key.
#    This appends new messages to the existing list.
graph_builder = StateGraph(State)
graph_builder.add_node('mock_llm', mock_llm)
graph_builder.set_entry_point('mock_llm')
graph_builder.add_edge('mock_llm', END)

# 4. Create the checkpointer and compile the graph
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# 5. Run the graph with a specific thread_id
async def main():
    # Use a unique ID for each conversation thread
    config = {'configurable': {'thread_id': 'youtube-thread-1'}}

    print('--- First Interaction ---')
    response = await graph.ainvoke({'messages': [HumanMessage(content='Hi! My name is Alex.')]}, config)
    print(f'AI: {response["messages"][-1].content}')

    print('\n--- Second Interaction (in the same thread) ---')
    response = await graph.ainvoke({'messages': [HumanMessage(content="What's my name?")]}, config)
    print(f'AI: {response["messages"][-1].content}')

    print('\n--- Interaction in a NEW thread ---')
    new_config = {'configurable': {'thread_id': 'youtube-thread-2'}}
    response = await graph.ainvoke({'messages': [HumanMessage(content="What's my name?")]}, new_config)
    print(f'AI: {response["messages"][-1].content}')


if __name__ == '__main__':
    # asyncio.run(main())
    main()
