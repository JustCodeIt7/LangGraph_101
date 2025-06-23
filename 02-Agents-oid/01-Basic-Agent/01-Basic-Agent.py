from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define the Agent (a function that will be a node in the graph)
def call_llm(state: AgentState):
    """Calls the LLM to generate a response based on the current conversation history."""
    print("---CALLING LLM---")
    messages = state['messages']
    # For this basic example, we'll use a pre-defined LLM and prompt
    # In a real scenario, you might load these from a config or pass them in
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Simple prompt - just pass the messages
    # More complex agents would have more sophisticated prompt engineering here
    response = llm.invoke(messages)
    print(f"LLM Response: {response.content}")
    # We return a dictionary with the 'messages' key,
    # and the value is a list containing the new AIMessage
    return {"messages": [AIMessage(content=response.content)]}

# 3. Define the Graph
# The StateGraph is a graph where the nodes are functions that modify the AgentState
workflow = StateGraph(AgentState)

# Add the 'call_llm' function as a node in the graph
# This node will be named "agent"
workflow.add_node("agent", call_llm)

# Set the entry point of the graph to the "agent" node
workflow.set_entry_point("agent")

# All paths from "agent" lead to END, meaning the graph finishes after the agent node.
# For more complex agents, you'd have conditional edges or more nodes.
workflow.add_edge("agent", END)

# 4. Compile the Graph
# This creates a runnable LangGraph object
basic_agent_graph = workflow.compile()

# 5. Run the Agent
if __name__ == "__main__":
    print("Starting Basic LangGraph Agent...")

    # Initial input: a list of messages
    # The first message is often a SystemMessage to set the context for the AI
    # Followed by a HumanMessage with the user's query
    initial_input = {
        "messages": [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Hello! What is the capital of France?")
        ]
    }

    print(f"Initial Input: {initial_input['messages']}")

    # Stream the events from the graph execution
    for event in basic_agent_graph.stream(initial_input, stream_mode="values"):
        # 'event' will be the full state after each step.
        # The final event will contain the complete conversation history.
        print(f"---EVENT---")
        print(event)
        final_response = event

    print("\\n---FINAL RESPONSE---")
    if final_response and 'messages' in final_response:
        # The last message in the list is typically the AI's response
        ai_message = final_response['messages'][-1]
        if isinstance(ai_message, AIMessage):
            print(f"AI: {ai_message.content}")
        else:
            print("No AIMessage found in the final response.")
    else:
        print("No response generated.")

    print("\\n---Running with another input---")
    second_input = {
        "messages": [
            SystemMessage(content="You are a poet who responds in haikus."),
            HumanMessage(content="Tell me about large language models.")
        ]
    }
    print(f"Second Input: {second_input['messages']}")
    for event in basic_agent_graph.stream(second_input, stream_mode="values"):
        print(f"---EVENT---")
        print(event)
        final_response_two = event

    print("\\n---FINAL RESPONSE 2---")
    if final_response_two and 'messages' in final_response_two:
        ai_message_two = final_response_two['messages'][-1]
        if isinstance(ai_message_two, AIMessage):
            print(f"AI: {ai_message_two.content}")
        else:
            print("No AIMessage found in the final response.")
    else:
        print("No response generated.")
