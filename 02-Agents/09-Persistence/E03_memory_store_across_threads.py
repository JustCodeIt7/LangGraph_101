# %%
import os
from typing import Annotated, TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import base64
# %%
# Clean up any previous database
if os.path.exists('simple_chat.sqlite'):
    os.remove('simple_chat.sqlite')


# Define the state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# %%
# Create the chatbot node
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# %%
def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}

# %%
# Build and run the graph with persistence
with SqliteSaver.from_conn_string('simple_chat.sqlite') as memory:
    # Create the graph
    workflow = StateGraph(State)
    workflow.add_node('chatbot', chatbot)
    workflow.add_edge(START, 'chatbot')
    workflow.add_edge('chatbot', END)

    # Compile with persistence
    graph = workflow.compile(checkpointer=memory)

    # Generate and save diagram for graph
    diagram = graph.get_graph().draw_mermaid_png()
    with open('graph3_diagram.png', 'wb') as f:
        f.write(diagram)
    print('Saved graph3 diagram to graph3_diagram.png')

    # Configuration for this conversation thread
    config = {'configurable': {'thread_id': 'conversation-1'}}

    # First message
    print('🤖 Starting conversation...')
    result1 = graph.invoke({'messages': [HumanMessage(content='Hi! My name is Alice.')]}, config)
    print(f'AI: {result1["messages"][-1].content}')

    # Second message - AI should remember the name
    print('\n🤖 Continuing conversation...')
    result2 = graph.invoke({'messages': [HumanMessage(content='What is my name?')]}, config)
    print(f'AI: {result2["messages"][-1].content}')

    # Show conversation history
    print('\n📜 Full conversation:')
    final_state = graph.get_state(config)
    for i, msg in enumerate(final_state.values['messages']):
        speaker = 'Human' if msg.type == 'human' else 'AI'
        print(f'{i + 1}. {speaker}: {msg.content}')
