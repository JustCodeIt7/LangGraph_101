import uuid
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

################################ Configuration & Setup ################################

# Initialize the Ollama large language model
llm = ChatOllama(model="llama3.2", temperature=0.7)


# Define the dictionary structure for the graph's state
class PromptState(TypedDict):
    original_prompt: str
    improved_prompt: Optional[str]
    final_prompt: Optional[str]
    final_answer: Optional[str]
    step: str  # Track the current step for UI feedback


################################ Core Logic & Helper Functions ################################

def call_llm(prompt: str, system_message: str = "") -> str:
    """
    Call Ollama LLM with a system message and a user prompt.
    """
    messages = []
    # Prepend a system message if one is provided
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=prompt))

    # Invoke the model and return the text content
    response = llm.invoke(messages)
    return response.content


################################ Graph Nodes ################################

def improve_prompt_node(state: PromptState) -> PromptState:
    """
    Use the LLM to refine and improve the user's original prompt.
    """
    print(f"🤖 Improving prompt with Ollama: {state['original_prompt']}")

    # Define the expert persona and instructions for the LLM
    system_message = """You are a prompt engineering expert. Take the user's original prompt and improve it to be:
1. More specific and detailed
2. Clear about the expected output format
3. Include relevant context that would help get better responses
4. Maintain the original intent but make it more effective

Return only the improved prompt, no explanations or additional text."""

    # Generate the improved prompt
    improved_prompt = call_llm(state['original_prompt'], system_message)

    # Update the graph's state with the new prompt
    return {
        **state,
        "improved_prompt": improved_prompt,
        "step": "prompt_improved"
    }


def human_review_node(state: PromptState) -> PromptState:
    """
    Pause the graph's execution to allow for human review and editing.
    """
    print("👤 Requesting human review...")

    # Interrupt execution, passing necessary data to the user for review
    human_response = interrupt({
        "task": "Please review and edit the improved prompt if needed",
        "original_prompt": state["original_prompt"],
        "improved_prompt": state["improved_prompt"],
        "instructions": "You can either approve the improved prompt or provide an edited version",
        "step": "awaiting_human_review"
    })

    # Process the human's response after the graph resumes
    if isinstance(human_response, dict):
        # Handle a structured dictionary response
        final_prompt = human_response.get("edited_prompt", state["improved_prompt"]) # Default to original if no edit
        action = human_response.get("action", "approve")
    else:
        # Handle a direct string input from the user
        final_prompt = human_response if human_response.strip() else state["improved_prompt"]
        action = "edit" if human_response.strip() else "approve"

    print(f"✅ Human {action}ed the prompt")

    # Update the state with the finalized prompt
    return {
        **state,
        "final_prompt": final_prompt,
        "step": "human_reviewed"
    }


def answer_prompt_node(state: PromptState) -> PromptState:
    """
    Use the LLM to generate a final answer based on the approved prompt.
    """
    print(f"🤖 Generating final answer with Ollama for: {state['final_prompt']}")

    # Define a helpful persona for the final response generation
    system_message = """You are a helpful AI assistant. Provide a comprehensive, well-structured answer to the user's prompt. Be thorough, accurate, and helpful. Format your response clearly with proper structure."""

    # Generate the final answer
    final_answer = call_llm(state['final_prompt'], system_message)

    # Update the state with the final answer
    return {
        **state,
        "final_answer": final_answer,
        "step": "completed"
    }


################################ Graph Definition & Compilation ################################

def create_app():
    """Create and compile the LangGraph application."""

    # Initialize a new state graph
    builder = StateGraph(PromptState)

    # Add the defined functions as nodes in the graph
    builder.add_node("improve_prompt", improve_prompt_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("answer_prompt", answer_prompt_node)

    # Define the execution flow by connecting the nodes with edges
    builder.add_edge(START, "improve_prompt")
    builder.add_edge("improve_prompt", "human_review")
    builder.add_edge("human_review", "answer_prompt")
    builder.add_edge("answer_prompt", END)

    # Set up an in-memory checkpointer to save state and support interruption
    checkpointer = InMemorySaver()

    # Compile the graph into a runnable application
    app = builder.compile(checkpointer=checkpointer)

    return app


################################ Application Execution ################################

def interactive_run():
    """Run an interactive console session for the Human-in-the-Loop app."""

    app = create_app()
    # Create a unique thread ID to manage conversation state
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Get the initial prompt from the user
    user_prompt = input("Enter your prompt: ").strip()
    if not user_prompt:
        user_prompt = "Tell me about machine learning" # Use a default prompt if none is provided

    # Define the initial state to start the graph
    initial_state = {
        "original_prompt": user_prompt,
        "step": "starting"
    }

    print(f"\n🚀 Processing with Ollama: {user_prompt}")
    print("-" * 50)

    # Start the graph execution, which will run until an interrupt is encountered
    result = app.invoke(initial_state, config=config)

    # Check if the graph was interrupted for human review
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        print(f"\n📝 Original prompt: {interrupt_data['original_prompt']}")
        print(f"🔧 Improved prompt: {interrupt_data['improved_prompt']}")
        print(f"\n{interrupt_data['task']}")

        # Get input from the human reviewer
        print("\nOptions:")
        print("1. Press Enter to approve as-is")
        print("2. Type a new version to edit")
        user_input = input("\nYour choice: ").strip()

        # Prepare the response payload based on user input
        if user_input:
            human_response = {"action": "edit", "edited_prompt": user_input}
        else:
            human_response = {"action": "approve"}

        # Resume the graph execution with the human's feedback
        print("\n🤖 Generating final answer with Ollama...")
        final_result = app.invoke(Command(resume=human_response), config=config)

        # Print the final results of the completed graph run
        print("\n" + "=" * 50)
        print("📊 RESULTS")
        print("=" * 50)
        print(f"Final prompt: {final_result['final_prompt']}")
        print(f"\n🎯 Answer:\n{final_result['final_answer']}")


# Define the main entry point for the script
if __name__ == "__main__":
    print("🚀 Advanced Human-in-the-Loop LangGraph App with Ollama")
    interactive_run()
