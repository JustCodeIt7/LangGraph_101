#%%
# ===========================================
# IMPORTS AND DEPENDENCIES
# ==========================================
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from rich.console import Console
from rich.prompt import Prompt
from IPython.display import Image, display

#%%
# =============================================
# STATE DEFINITIONS AND DATA STRUCTURES
# =============================================

# Define the state structure
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    current_response: str
    exit_requested: bool
    verbose_mode: bool


# Initialize the console for rich output
console = Console()

#%%
# ===============================================
# LLM INITIALIZATION AND CONFIGURATION
# ==============================================

def create_llm(model='qwen3:0.6b') -> ChatLiteLLM:
    """Create a LiteLLM instance using Ollama with llama3.2 model."""
    console.print("🤖 Initializing LLM with Ollama (llama3.2)...", style="bold blue")

    llm = ChatLiteLLM(
        model= model,
        api_base="http://localhost:11434",  # Default Ollama local server
        temperature=0.7,
        max_tokens=1000,
    )

    console.print("✅ LLM initialized successfully!", style="bold green")
    return llm
create_llm()

#%%
# =============================================
# STATE MANAGEMENT FUNCTIONS
# ============================================
def initialize_state(state: ChatState) -> ChatState:
    """Initialize the chat state."""
    state["messages"] = []
    state["current_response"] = ""
    state["exit_requested"] = False
    state["verbose_mode"] = False
    return state

#%%
# ===========================================
# USER INPUT PROCESSING
# ==========================================

def process_user_input(state: ChatState) -> ChatState:
    """Process user input and add it to the messages."""
    try:
        # Get user input
        user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

        # Check for exit command
        if user_input.lower() in ["/exit", "/quit", "/bye"]:
            state["exit_requested"] = True
            return state

        # Check for verbose mode toggle
        if user_input.lower() == "verbose":
            state["verbose_mode"] = not state["verbose_mode"]
            console.print(f"[bold yellow]Verbose mode {'enabled' if state['verbose_mode'] else 'disabled'}[/bold yellow]")
            return state

        # Add user message to the conversation history
        state["messages"].append({"role": "user", "content": user_input})

        return state
    except KeyboardInterrupt:
        state["exit_requested"] = True
        return state


#%%
# =========================================
# AI RESPONSE GENERATION
# ========================================

def generate_ai_response(state: ChatState, llm: ChatLiteLLM) -> ChatState:
    """Generate AI response using the LLM."""
    # Skip AI response if exit requested, command was processed, or no messages
    if (state["exit_requested"] or
        state["command_processed"] or
        not state["messages"] or
        len(state["messages"]) == 0):
        return state

    # Also skip if the last message is not from user
    if state["messages"] and state["messages"][-1].get("role") != "user":
        return state


    # Display thinking indicator
    console.print("[bold yellow]Thinking...[/bold yellow]")

    # Generate response from LLM
    response = llm.invoke(state["messages"])

    # Extract the response content
    ai_message = response.content

    # Add AI message to the conversation history
    state["messages"].append({"role": "assistant", "content": ai_message})
    state["current_response"] = ai_message

    # Display the response
    console.print(f"\n[bold green]Assistant[/bold green]: {ai_message}")

    # Display verbose information if enabled
    if state["verbose_mode"]:
        console.print("\n[bold magenta]Debug Info:[/bold magenta]")
        console.print(f"Message count: {len(state['messages'])}")
        if len(state['messages']) >= 2:
            console.print(f"Last user message: {state['messages'][-2]['content'][:50]}...")
        console.print(f"Current response length: {len(ai_message)} characters")

    return state


#%%
# =======================================
# CONVERSATION FLOW CONTROL
# ======================================

def should_continue(state: ChatState) -> str:
    """Conditional logic to determine whether to continue the conversation or end."""
    if state["exit_requested"]:
        console.print("[bold yellow]Exiting chat...[/bold yellow]")
        return "end"
    else:
        return "continue"

#%%
# =====================================================
# MAIN APPLICATION LOGIC AND GRAPH CONFIGURATION
# ====================================================
def decide_after_user_input(state: ChatState) -> str:
    """Decide what to do after processing user input."""
    if state["exit_requested"]:
        return "end"
    elif state["command_processed"]:
        return "continue_input"
    else:
        return "ai_response"

def main():
    """Main function to run the terminal chat application."""
    # ========================================
    # Application Initialization
    # ========================================
    console.print('[bold]Welcome to LangGraph Terminal Chat![/bold]', style='bold blue')
    console.print("Type 'exit', 'quit', or 'bye' to end the conversation.", style='dim')
    console.print("Type 'verbose' to toggle verbose mode.", style='dim')

    # Create the LLM
    llm = create_llm()

    # ========================================
    # Graph Construction - Building the Conversation Flow
    # ========================================

    # STEP 1: Create the main graph object
    # StateGraph is like a flowchart that manages conversation state
    # Think of it as a blueprint for how the conversation should flow
    graph = StateGraph(ChatState)

    # STEP 2: Add nodes (the "work stations" in our conversation flow)
    # Each node is a function that does a specific job and can modify the state

    # Node 1: Initialize the conversation
    # This sets up the initial state when the chat starts
    graph.add_node('initialize', initialize_state)

    # Node 2: Handle user input
    # This processes what the user types and updates the state
    graph.add_node('user_input', process_user_input)

    # Node 3: Generate AI response
    # This creates the AI's reply using the LLM
    # lambda state: creates an anonymous function that calls generate_ai_response with our LLM
    graph.add_node('ai_response', lambda state: generate_ai_response(state, llm))

    # STEP 3: Add simple edges (direct connections)
    # Edges tell the graph how to move from one node to another
    # This creates a direct path: initialize → user_input (always happens)
    graph.add_edge('initialize', 'user_input')

    # STEP 4: Add conditional edges (decision points)
    # These are like "if-then" statements that decide where to go next based on the state
    # Decision point 1: After user input, what should we do?
    def decide_after_user_input(state: ChatState) -> str:
        """
        Decide what to do after processing user input.

        Returns:
            "end" - User wants to exit
            "continue_input" - User entered a command, stay in input loop
            "ai_response" - User entered a message, generate AI response
        """
        if state['exit_requested']:
            return 'end'
        elif state['command_processed']:
            return 'continue_input'
        else:
            return 'ai_response'

    # CONDITIONAL EDGE 1: After processing user input, where should we go?
    graph.add_conditional_edges(
        source="user_input",                    # Starting from user_input node
        path=decide_after_user_input,           # Use our decision function
        path_map={                              # Map decisions to destinations
            "ai_response": "ai_response",       # Generate AI response
            "continue_input": "user_input",     # Go back to user input
            "end": END,                         # Exit the conversation
        },
    )

    # CONDITIONAL EDGE 2: After AI response, should we continue or stop?
    # Decision point 2: After AI response, continue or end?
    def decide_after_ai_response(state: ChatState) -> str:
        """
        Decide what to do after AI generates a response.

        Returns:
            "continue" - Keep the conversation going
            "end" - End the conversation
        """
        if state['exit_requested']:
            return 'end'
        else:
            return 'continue'

    graph.add_conditional_edges(
        source='ai_response',  # Starting from ai_response node
        path=decide_after_ai_response,  # Use our decision function
        path_map={  # Map decisions to destinations
            'continue': 'user_input',  # Continue conversation
            'end': END,  # Exit the conversation
        },
    )
    # graph.add_conditional_edges(
    #     'ai_response',  # Starting point: AI just generated a response
    #     # Decision function: should_continue is defined elsewhere
    #     # It returns either 'continue' or 'end' based on the state
    #     should_continue,
    #     # Path mapping: what to do with each possible return value
    #     {
    #         'continue': 'user_input',  # Keep chatting → go back to user input
    #         'end': END,  # Stop chatting → END the graph
    #     },
    # )
    #
    # STEP 5: Set the starting point
    # This tells the graph which node to run first when we start the conversation
    graph.set_entry_point('initialize')

    # STEP 6: Compile the graph
    # This "bakes" the graph into an executable application
    # After compilation, we can't add more nodes or edges, but we can run it
    app = graph.compile()

    # print the graph structure for debugging
    console.print("\n[bold green]Graph Structure:[/bold green]")


    # save the graph to a file for later use
    with open('langgraph_flow.png', 'wb') as f:
        f.write(app.get_graph().draw_mermaid_png())

    # ========================================
    # Application Execution - Running the Graph
    # ========================================
    app.invoke({})

    console.print('\n[bold blue]Thank you for using LangGraph Terminal Chat![/bold blue]')


# ===================================================
# APPLICATION ENTRY POINT
# ===================================================
if __name__ == '__main__':
    main()


#%%
# ===================================================
# APPLICATION ENTRY POINT
# ==================================================
if __name__ == "__main__":
    main()
#%%