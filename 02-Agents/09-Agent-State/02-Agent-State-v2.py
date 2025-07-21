import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich import print
# --- Environment Setup (Optional: Load from .env or similar in real projects) ---
# Ensure you have OPENAI_API_KEY set in your environment
# from dotenv import load_dotenv
# load_dotenv()
# if os.getenv("OPENAI_API_KEY") is None:
#     print("OPENAI_API_KEY not set. Please set it to run this script.")
#     exit()


# 1. Define the Enhanced Agent State
class EnhancedAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_name: str
    turn_count: int


# 2. Define Agent Nodes
def state_initializer_node(state: EnhancedAgentState):
    """
    Initializes specific parts of the state if they aren't already set.
    This node is more for demonstration of state manipulation.
    In many cases, initial state is passed directly when invoking the graph.
    """
    print('---INITIALIZING/CHECKING STATE---')
    if not state.get('user_name'):
        # This is a fallback, user_name should ideally be set at the start
        state['user_name'] = 'Guest'
    if state.get('turn_count') is None:  # Check for None specifically
        state['turn_count'] = 0

    # No messages are added here, they come from the initial input or user interaction
    print(f'State Initialized/Checked: User: {state["user_name"]}, Turns: {state["turn_count"]}')
    return state  # Return the modified state


def greet_user_node(state: EnhancedAgentState):
    """
    Greets the user if it's the first turn and updates the turn count.
    Illustrates reading and updating different parts of the state.
    """
    print('---GREETING USER NODE---')
    turn_count = state.get('turn_count', 0)
    user_name = state.get('user_name', 'User')

    greeting_message = ''
    if turn_count == 0:  # Only greet on the very first interaction with this node
        greeting_message = f"Hello {user_name}! I'm your stateful assistant. This is our first proper exchange."
        # Add a system message or an AI message to reflect this greeting
        # For simplicity, we'll just print it and let the LLM handle the main interaction.
        print(f'Greeting: {greeting_message}')

    # Increment turn count for the next interaction
    # This demonstrates updating a non-message part of the state.
    # The actual response to the user will come from the LLM.
    return {'turn_count': turn_count + 1}


def call_llm_node(state: EnhancedAgentState):
    """
    Calls the LLM to generate a response based on the current conversation history.
    Also demonstrates awareness of the turn count from the state.
    """
    print('---CALLING LLM NODE---')
    messages = state['messages']
    turn_count = state.get('turn_count', 0)  # Get current turn_count
    user_name = state.get('user_name', 'User')

    print(f'LLM Call for {user_name} | Turn: {turn_count}')

    # Append a system message that includes state information, if desired
    # For this example, we'll keep the prompt simple.
    # system_prompt = f"You are a helpful AI. It is turn number {turn_count} with {user_name}."
    # current_messages = [SystemMessage(content=system_prompt)] + messages

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    llm = ChatOllama(model='llama3.2', temperature=0.7)  # Use Ollama model

    # Create a prompt template (optional, but good practice)
    # For this example, we'll just pass the messages directly
    # prompt = ChatPromptTemplate.from_messages(messages)
    # chain = prompt | llm
    # response = chain.invoke({}) # Invoke with empty dict if prompt has no input_variables

    response = llm.invoke(messages)  # Pass the direct list of messages
    print(f'LLM Response: {response.content}')

    # The LLM response is added to messages by add_messages due to the annotation
    return {'messages': [AIMessage(content=response.content)]}


# 3. Define the Graph
workflow = StateGraph(EnhancedAgentState)

# Add nodes
workflow.add_node('initializer', state_initializer_node)
workflow.add_node('greeter', greet_user_node)
workflow.add_node('agent', call_llm_node)

# Define edges
workflow.set_entry_point('initializer')
workflow.add_edge('initializer', 'greeter')  # Initialize then greet
workflow.add_edge('greeter', 'agent')  # After greeting, call the agent
workflow.add_edge('agent', END)  # Agent responds, then ends this run

# 4. Compile the Graph
agent_with_state_graph = workflow.compile()

# 5. Interactive Loop
if __name__ == '__main__':
    print('Starting Stateful LangGraph Agent...')

    user_name_input = input('What is your name? ')
    if not user_name_input.strip():
        user_name_input = 'Guest'  # Default if empty

    # Initial state for the first run
    # 'messages' will be populated by the first user input inside the loop
    # 'turn_count' is explicitly set to None to let initializer_node handle it first
    current_state = {
        'messages': [],
        'user_name': user_name_input,
        'turn_count': None,  # Let initializer set it to 0
    }

    print("\nAsk the agent something, or type 'exit' to quit.")

    while True:
        user_input = input(f'\n{current_state.get("user_name", "User")} ({current_state.get("turn_count", 0)}): ')
        if user_input.lower() == 'exit':
            print('Exiting agent.')
            break
        if not user_input.strip():
            print('Please enter some text.')
            continue

        # Add the new human message to the current list of messages in the state
        # The `add_messages` utility in the state definition will handle appending correctly
        current_state['messages'] = [HumanMessage(content=user_input)]

        print('---SENDING TO GRAPH---')
        print(f'Current state before graph: {current_state}')

        # Stream events to see the state evolve
        final_event = None
        for event_part in agent_with_state_graph.stream(current_state, stream_mode='values'):
            # event_part is the full state after each node's execution in this mode
            print('---GRAPH EVENT---')
            # print(f"Node: {list(event_part.keys())[0]}") # This is tricky with stream_mode="values"
            print(f'Full state in event: {event_part}')
            final_event = event_part  # Keep the last state

        if final_event and 'messages' in final_event:
            # The AIMessage from the LLM is the last one added by call_llm_node
            ai_response_message = final_event['messages'][-1]
            if isinstance(ai_response_message, AIMessage):
                print(f'\\nAI: {ai_response_message.content}')

            # Persist the full conversation history and other state for the next turn
            current_state['messages'] = final_event['messages']
            current_state['user_name'] = final_event.get('user_name', user_name_input)  # Should persist
            current_state['turn_count'] = final_event.get('turn_count', 0)  # Greeter/LLM node updates this
        else:
            print('AI did not return a message.')
            # Reset messages if something went wrong to avoid issues on next turn
            current_state['messages'] = []

    print('\n---FINAL STATE (after loop ends)---')
    print(current_state)
