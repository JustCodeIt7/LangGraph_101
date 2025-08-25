"""
Basic Human in the Loop LangGraph App
A simple chatbot that can pause execution to ask humans for input when needed.
"""

import uuid
from typing import TypedDict, Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama



# Define the graph state
class AgentState(MessagesState):
    """State for our human-in-the-loop agent."""
    needs_human_input: bool = False
    human_feedback: Optional[str] = None


# Initialize the LLM (you can replace with any LLM)
llm = ChatOllama(model="llama3.2", temperature=0.7)


# Define tools
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulate weather data
    return f"The weather in {location} is sunny with 75°F temperature."


@tool
def ask_human(question: str) -> str:
    """Ask a human for input when the AI needs clarification or approval."""
    # This will trigger an interrupt for human input
    human_response = interrupt({
        "type": "human_input_request",
        "question": question,
        "instruction": "Please provide your response below:"
    })
    return human_response


# Define the tools list
tools = [get_weather, ask_human]
tool_node = ToolNode(tools)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# Define nodes
def agent_node(state: AgentState):
    """Main agent node that decides what to do next."""
    system_message = SystemMessage(
        content="""You are a helpful assistant. You can:
        1. Get weather information for any location
        2. Ask humans for input when you need clarification or approval
        
        When you're unsure about something or need human approval for an action,
        use the ask_human tool to get clarification.
        
        Be conversational and helpful!"""
    )
    
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue with tools or end."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


def create_graph():
    """Create and compile the LangGraph workflow."""
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.add_edge(START, "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with checkpointer for interrupt support
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def run_conversation():
    """Run an interactive conversation with the agent."""
    graph = create_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤖 Human-in-the-Loop ChatBot")
    print("Type 'quit' to exit, 'help' for examples")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'help':
            print("""
📝 Example prompts:
- "What's the weather in New York?"
- "Should I invest in stocks? I have $1000 to invest."
- "Help me plan a vacation to Europe"
- "I need approval to delete important files"
            """)
            continue
        
        if not user_input:
            continue
        
        try:
            # Create human message
            human_msg = HumanMessage(content=user_input)
            
            # Stream the response
            print("\n🤖 Assistant: ", end="", flush=True)
            
            for event in graph.stream(
                {"messages": [human_msg]}, 
                config=config,
                stream_mode="values"
            ):
                # Check for interrupts (human input requests)
                if "__interrupt__" in event:
                    interrupt_data = event["__interrupt__"][0]
                    question = interrupt_data.value.get("question", "Please provide input:")
                    
                    print(f"\n\n❓ {question}")
                    human_response = input("👤 Your response: ").strip()
                    
                    # Resume with human input
                    print("\n🤖 Assistant: ", end="", flush=True)
                    for resume_event in graph.stream(
                        Command(resume=human_response),
                        config=config,
                        stream_mode="values"
                    ):
                        if "messages" in resume_event and resume_event["messages"]:
                            last_msg = resume_event["messages"][-1]
                            if hasattr(last_msg, 'content') and isinstance(last_msg, AIMessage):
                                print(last_msg.content, flush=True)
                    break
                
                # Print regular messages
                elif "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    if hasattr(last_msg, 'content') and isinstance(last_msg, AIMessage):
                        print(last_msg.content, flush=True)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def demo_interrupt_workflow():
    """Demonstrate the interrupt functionality with a simple example."""
    print("\n🔧 Running Demo: Human Approval Workflow")
    print("-" * 50)
    
    graph = create_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Test message that should trigger human input
    test_message = HumanMessage(
        content="I want to delete all my important files. Should I proceed?"
    )
    
    print("👤 User: I want to delete all my important files. Should I proceed?")
    print("\n🤖 Assistant will ask for human approval...")
    
    # Run until interrupt
    result = graph.invoke({"messages": [test_message]}, config=config)
    
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        print(f"\n❓ Agent asks: {interrupt_data.value.get('question', 'Need input')}")
        
        # Simulate human response
        human_decision = "No, don't delete the files"
        print(f"👤 Human responds: {human_decision}")
        
        # Resume with human input
        final_result = graph.invoke(
            Command(resume=human_decision),
            config=config
        )
        
        if "messages" in final_result:
            final_message = final_result["messages"][-1]
            print(f"\n🤖 Final response: {final_message.content}")


if __name__ == "__main__":
    print("🚀 Basic Human in the Loop LangGraph App")
    print("=" * 50)
    
    # Show available modes
    print("\nChoose mode:")
    print("1. Interactive chat")
    print("2. Demo workflow")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_conversation()
    elif choice == "2":
        demo_interrupt_workflow()
    else:
        print("Invalid choice. Running interactive chat...")
        run_conversation()
