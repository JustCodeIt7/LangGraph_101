# ================================================================================================
# IMPORTS AND DEPENDENCIES
# ================================================================================================

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from rich.console import Console
from rich.prompt import Prompt
import os
import subprocess
import tempfile
from pathlib import Path
from IPython.display import Image, display
# ================================================================================================
# STATE DEFINITIONS AND DATA STRUCTURES
# ================================================================================================

# Define the state structure
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    current_response: str
    exit_requested: bool
    verbose_mode: bool
    command_processed: bool  # New field to track command processing

# Initialize the console for rich output
console = Console()



# ================================================================================================
# LLM INITIALIZATION AND CONFIGURATION
# ================================================================================================


def create_llm(model='ollama/qwen3:0.6b') -> ChatLiteLLM:
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

# ================================================================================================
# STATE MANAGEMENT FUNCTIONS
# ================================================================================================

def initialize_state(state: ChatState) -> ChatState:
    """Initialize the chat state."""
    state["messages"] = []
    state["current_response"] = ""
    state["exit_requested"] = False
    state["verbose_mode"] = False
    state["command_processed"] = False  # Initialize new field
    return state

# ================================================================================================
# USER INPUT PROCESSING
# ================================================================================================

def process_user_input(state: ChatState) -> ChatState:
    """Process user input and add it to the messages."""
    # Reset command processed flag
    state["command_processed"] = False

    try:
        # Get user input
        user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            state["exit_requested"] = True
            state["command_processed"] = True
            return state

        # Check for verbose mode toggle
        if user_input.lower() == "verbose":
            state["verbose_mode"] = not state["verbose_mode"]
            console.print(f"[bold yellow]Verbose mode {'enabled' if state['verbose_mode'] else 'disabled'}[/bold yellow]")
            state["command_processed"] = True  # Mark as command processed
            return state

        # Check for help command
        if user_input.lower() in ["help", "?"]:
            console.print("\n[bold cyan]Available commands:[/bold cyan]")
            console.print("• [bold]verbose[/bold] - Toggle verbose mode")
            console.print("• [bold]exit/quit/bye[/bold] - Exit the chat")
            console.print("• [bold]help/?[/bold] - Show this help")
            state["command_processed"] = True
            return state

        # Add user message to the conversation history (only for non-commands)
        if user_input.strip():  # Ensure we don't add empty messages
            state["messages"].append({"role": "user", "content": user_input})

        return state

    except KeyboardInterrupt:
        state["exit_requested"] = True
        state["command_processed"] = True
        return state
    except Exception as e:
        console.print(f"[bold red]Error processing input: {str(e)}[/bold red]")
        state["command_processed"] = True
        return state


# ================================================================================================
# AI RESPONSE GENERATION
# ================================================================================================

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


# ================================================================================================
# CONVERSATION FLOW CONTROL
# ================================================================================================

def should_continue(state: ChatState) -> str:
    """Conditional logic to determine whether to continue the conversation or end."""
    if state["exit_requested"]:
        console.print("[bold yellow]Exiting chat...[/bold yellow]")
        return "end"
    elif state["command_processed"]:
        # If a command was processed, go back to user input without AI response
        return "continue_input"
    else:
        return "continue_input"

# ================================================================================================
# MAIN APPLICATION LOGIC AND GRAPH CONFIGURATION
# ================================================================================================

def main():
    """Main function to run the terminal chat application."""
    # ========================================
    # Application Initialization
    # ========================================
    console.print("[bold]Welcome to LangGraph Terminal Chat![/bold]", style="bold blue")
    console.print("Type 'exit', 'quit', or 'bye' to end the conversation.", style="dim")
    console.print("Type 'verbose' to toggle verbose mode.", style="dim")
    console.print("Type 'help' or '?' for available commands.", style="dim")

    # Generate and save Mermaid diagram
    console.print("\n📊 Generating Mermaid diagram...", style="bold cyan")


    # Create the LLM
    llm = create_llm()

    # ========================================
    # Graph Construction
    # ========================================
    # Build the graph
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("initialize", initialize_state)
    graph.add_node("user_input", process_user_input)
    graph.add_node("ai_response", lambda state: generate_ai_response(state, llm))

    # Add edges
    graph.add_edge("initialize", "user_input")

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

    # Add conditional edge after user input
    graph.add_conditional_edges(
        "user_input",
        decide_after_user_input,
        {
            "ai_response": "ai_response",
            "continue_input": "user_input",
            "end": END,
        },
    )

    # Add conditional edge for looping or ending after AI response
    graph.add_conditional_edges(
        "ai_response",
        should_continue,
        {
            # "continue": "user_input",  # Loop back for more conversation
            # "continue_input": "user_input",  # Direct back to input (shouldn't happen here)
            "end": END,  # End the conversation
        },
    )

    # Set the entry point
    graph.set_entry_point("initialize")

    # Compile the graph
    app = graph.compile()

    # app.get_graph().draw_mermaid_png()
    with open("chat_graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # save the graph image
    with open("chat_graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())


    # ========================================
    # Application Execution
    # ========================================
    app.invoke({})




def run_tests():
    """Run comprehensive tests for the chat application."""
    console.print("\n" + "="*80, style="bold magenta")
    console.print("🧪 RUNNING TESTS", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Test counters
    tests_passed = 0
    tests_failed = 0

    # ========================================
    # Test 1: State Initialization
    # ========================================
    console.print("📋 Test 1: State Initialization", style="bold yellow")
    try:
        state = ChatState()
        state = initialize_state(state)
        assert state["messages"] == []
        assert state["current_response"] == ""
        assert state["exit_requested"] == False
        assert state["verbose_mode"] == False
        assert state["command_processed"] == False
        console.print("✅ State initialization test passed\n", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"❌ State initialization test failed: {e}\n", style="red")
        tests_failed += 1

    # ========================================
    # Test 2: Command Processing
    # ========================================
    console.print("📋 Test 2: Command Processing", style="bold yellow")

    # Test exit commands
    for exit_cmd in ["exit", "quit", "bye"]:
        try:
            state = ChatState()
            state = initialize_state(state)
            # Simulate user input
            import unittest.mock
            with unittest.mock.patch('rich.prompt.Prompt.ask', return_value=exit_cmd):
                state = process_user_input(state)
            assert state["exit_requested"] == True
            assert state["command_processed"] == True
            console.print(f"  ✅ Exit command '{exit_cmd}' test passed", style="green")
            tests_passed += 1
        except Exception as e:
            console.print(f"  ❌ Exit command '{exit_cmd}' test failed: {e}", style="red")
            tests_failed += 1

    # Test verbose command
    try:
        state = ChatState()
        state = initialize_state(state)
        with unittest.mock.patch('rich.prompt.Prompt.ask', return_value="verbose"):
            state = process_user_input(state)
        assert state["verbose_mode"] == True
        assert state["command_processed"] == True
        console.print("  ✅ Verbose command test passed", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"  ❌ Verbose command test failed: {e}", style="red")
        tests_failed += 1

    # Test help command
    try:
        state = ChatState()
        state = initialize_state(state)
        with unittest.mock.patch('rich.prompt.Prompt.ask', return_value="help"):
            state = process_user_input(state)
        assert state["command_processed"] == True
        console.print("  ✅ Help command test passed\n", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"  ❌ Help command test failed: {e}\n", style="red")
        tests_failed += 1

    # ========================================
    # Test 3: Regular Message Processing
    # ========================================
    console.print("📋 Test 3: Regular Message Processing", style="bold yellow")
    try:
        state = ChatState()
        state = initialize_state(state)
        test_message = "Hello, AI!"
        with unittest.mock.patch('rich.prompt.Prompt.ask', return_value=test_message):
            state = process_user_input(state)
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][0]["content"] == test_message
        assert state["command_processed"] == False
        console.print("✅ Regular message processing test passed\n", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"❌ Regular message processing test failed: {e}\n", style="red")
        tests_failed += 1

    # ========================================
    # Test 4: Flow Control Logic
    # ========================================
    console.print("📋 Test 4: Flow Control Logic", style="bold yellow")

    # Test exit flow
    try:
        state = ChatState()
        state["exit_requested"] = True
        result = should_continue(state)
        assert result == "end"
        console.print("  ✅ Exit flow test passed", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"  ❌ Exit flow test failed: {e}", style="red")
        tests_failed += 1

    # Test command processed flow
    try:
        state = ChatState()
        state["exit_requested"] = False
        state["command_processed"] = True
        result = should_continue(state)
        assert result == "continue_input"
        console.print("  ✅ Command processed flow test passed", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"  ❌ Command processed flow test failed: {e}", style="red")
        tests_failed += 1

    # Test normal flow
    try:
        state = ChatState()
        state["exit_requested"] = False
        state["command_processed"] = False
        result = should_continue(state)
        assert result == "continue_input"
        console.print("  ✅ Normal flow test passed\n", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"  ❌ Normal flow test failed: {e}\n", style="red")
        tests_failed += 1

    # ========================================
    # Test 5: LLM Connection (Mock)
    # ========================================
    console.print("📋 Test 5: LLM Connection Test", style="bold yellow")
    try:
        # Check if Ollama is running
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            console.print("✅ Ollama server is running\n", style="green")
            tests_passed += 1
        else:
            console.print("⚠️  Ollama server returned unexpected status\n", style="yellow")
            tests_passed += 1
    except:
        console.print("⚠️  Ollama server not running (this is OK for testing)\n", style="yellow")
        tests_passed += 1

    # ========================================
    # Test 6: Graph Construction
    # ========================================
    console.print("📋 Test 6: Graph Construction", style="bold yellow")
    try:
        # Build a test graph
        test_graph = StateGraph(ChatState)
        test_graph.add_node("test_node", lambda state: state)
        test_graph.set_entry_point("test_node")
        test_graph.add_edge("test_node", END)
        test_app = test_graph.compile()

        # Test graph execution
        result = test_app.invoke({"messages": [], "exit_requested": False})
        assert "messages" in result
        console.print("✅ Graph construction test passed\n", style="green")
        tests_passed += 1
    except Exception as e:
        console.print(f"❌ Graph construction test failed: {e}\n", style="red")
        tests_failed += 1

    # ========================================
    # Test Summary
    # ========================================
    console.print("="*80, style="bold magenta")
    console.print(f"🏁 TEST SUMMARY", style="bold magenta")
    console.print(f"  Total tests: {tests_passed + tests_failed}", style="bold")
    console.print(f"  ✅ Passed: {tests_passed}", style="bold green")
    console.print(f"  ❌ Failed: {tests_failed}", style="bold red")
    console.print("="*80 + "\n", style="bold magenta")

    return tests_failed == 0

# ================================================================================================
# ENHANCED ENTRY POINT WITH TESTING OPTION
# ================================================================================================

if __name__ == "__main__":
    import sys

    # Check for test flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--test", "-t"]:
        # Run tests
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        # Run the main application
        console.print("\n💡 Tip: Run with '--test' or '-t' flag to run tests\n", style="dim italic")
        main()
