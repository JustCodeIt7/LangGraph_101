"""
Basic Human in the Loop LangGraph App with Ollama
- Takes user prompt
- LLM creates improved prompt using Ollama
- Human approves/edits the improved prompt
- LLM answers the final improved prompt using Ollama
"""

import uuid
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

# Initialize Ollama LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)


# Define the graph state
class PromptState(TypedDict):
    original_prompt: str
    improved_prompt: Optional[str]
    final_prompt: Optional[str]
    final_answer: Optional[str]
    step: str  # Track current step for UI feedback


def call_llm(prompt: str, system_message: str = "") -> str:
    """
    Call Ollama LLM with system message and user prompt
    """
    try:
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return f"Error: Could not generate response. Please check if Ollama is running and the model 'llama3.2' is available."


def improve_prompt_node(state: PromptState) -> PromptState:
    """
    Node that uses Ollama to improve the original prompt
    """
    print(f"🤖 Improving prompt with Ollama: {state['original_prompt']}")
    
    system_message = """You are a prompt engineering expert. Take the user's original prompt and improve it to be:
1. More specific and detailed
2. Clear about the expected output format
3. Include relevant context that would help get better responses
4. Maintain the original intent but make it more effective

Return only the improved prompt, no explanations or additional text."""
    
    improved_prompt = call_llm(state['original_prompt'], system_message)
    
    return {
        **state,
        "improved_prompt": improved_prompt,
        "step": "prompt_improved"
    }


def human_review_node(state: PromptState) -> PromptState:
    """
    Node that pauses for human review and editing of the improved prompt
    """
    print("👤 Requesting human review...")
    
    # Interrupt execution for human review
    human_response = interrupt({
        "task": "Please review and edit the improved prompt if needed",
        "original_prompt": state["original_prompt"],
        "improved_prompt": state["improved_prompt"],
        "instructions": "You can either approve the improved prompt or provide an edited version",
        "step": "awaiting_human_review"
    })
    
    # Process human response
    if isinstance(human_response, dict):
        # Human provided structured response
        final_prompt = human_response.get("edited_prompt", state["improved_prompt"])
        action = human_response.get("action", "approve")
    else:
        # Human provided direct text response
        final_prompt = human_response if human_response.strip() else state["improved_prompt"]
        action = "edit" if human_response.strip() else "approve"
    
    print(f"✅ Human {action}ed the prompt")
    
    return {
        **state,
        "final_prompt": final_prompt,
        "step": "human_reviewed"
    }


def answer_prompt_node(state: PromptState) -> PromptState:
    """
    Node that uses Ollama to answer the final improved prompt
    """
    print(f"🤖 Generating final answer with Ollama for: {state['final_prompt']}")
    
    system_message = """You are a helpful AI assistant. Provide a comprehensive, well-structured answer to the user's prompt. Be thorough, accurate, and helpful. Format your response clearly with proper structure."""
    
    final_answer = call_llm(state['final_prompt'], system_message)
    
    return {
        **state,
        "final_answer": final_answer,
        "step": "completed"
    }


# Build the graph
def create_app():
    """Create and compile the LangGraph application"""
    
    # Initialize state graph
    builder = StateGraph(PromptState)
    
    # Add nodes
    builder.add_node("improve_prompt", improve_prompt_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("answer_prompt", answer_prompt_node)
    
    # Add edges
    builder.add_edge(START, "improve_prompt")
    builder.add_edge("improve_prompt", "human_review")
    builder.add_edge("human_review", "answer_prompt")
    builder.add_edge("answer_prompt", END)
    
    # Set up checkpointer for interrupt support
    checkpointer = InMemorySaver()
    
    # Compile the graph
    app = builder.compile(checkpointer=checkpointer)
    
    return app


def test_ollama_connection():
    """Test if Ollama is working properly"""
    print("🔍 Testing Ollama connection...")
    try:
        test_response = llm.invoke([HumanMessage(content="Say 'Ollama is working!'")])
        print(f"✅ Ollama test successful: {test_response.content}")
        return True
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        print("Please ensure:")
        print("1. Ollama is installed and running")
        print("2. The model 'llama3.2' is available (run: ollama pull llama3.2)")
        return False


def run_app_example():
    """Example of how to run the app"""
    
    if not test_ollama_connection():
        return
    
    app = create_app()
    
    # Configuration with thread ID for state persistence
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Initial prompt from user
    initial_state = {
        "original_prompt": "Tell me about Python",
        "step": "starting"
    }
    
    print("\n🚀 Starting Human in the Loop Prompt Improvement App with Ollama")
    print(f"📝 Original prompt: {initial_state['original_prompt']}")
    print("-" * 50)
    
    # Run until first interrupt
    result = app.invoke(initial_state, config=config)
    
    # Check if we hit an interrupt
    if "__interrupt__" in result:
        print("\n⏸️  App paused for human review")
        print("Interrupt details:")
        interrupt_data = result["__interrupt__"][0].value
        print(f"  Original: {interrupt_data['original_prompt']}")
        print(f"  Improved: {interrupt_data['improved_prompt']}")
        print(f"  Task: {interrupt_data['task']}")
        
        # Simulate human response (in real app, this would come from UI)
        print("\n👤 Human reviewing...")
        
        # Option 1: Approve as-is
        human_input = {"action": "approve"}
        
        # Option 2: Edit the prompt (uncomment to test)
        # human_input = {
        #     "action": "edit", 
        #     "edited_prompt": "Explain Python programming language comprehensively, including its key features, main use cases, advantages for beginners, and provide practical code examples with explanations."
        # }
        
        print(f"Human decision: {human_input['action']}")
        
        # Resume execution with human input
        print("\n▶️  Resuming execution with Ollama...")
        final_result = app.invoke(Command(resume=human_input), config=config)
        
        # Display final results
        print("\n" + "="*50)
        print("📊 FINAL RESULTS")
        print("="*50)
        print(f"Original prompt: {final_result['original_prompt']}")
        print(f"Improved prompt: {final_result['improved_prompt']}")
        print(f"Final prompt: {final_result['final_prompt']}")
        print(f"\n🎯 Final Answer:\n{final_result['final_answer']}")
        
    return result


def interactive_run():
    """Interactive version where user provides real input"""
    
    if not test_ollama_connection():
        return
    
    app = create_app()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Get user input
    user_prompt = input("Enter your prompt: ").strip()
    if not user_prompt:
        user_prompt = "Tell me about machine learning"
    
    initial_state = {
        "original_prompt": user_prompt,
        "step": "starting"
    }
    
    print(f"\n🚀 Processing with Ollama: {user_prompt}")
    print("-" * 50)
    
    # Run until interrupt
    result = app.invoke(initial_state, config=config)
    
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        
        print(f"\n📝 Original prompt: {interrupt_data['original_prompt']}")
        print(f"🔧 Improved prompt: {interrupt_data['improved_prompt']}")
        print(f"\n{interrupt_data['task']}")
        
        # Get human input
        print("\nOptions:")
        print("1. Press Enter to approve as-is")
        print("2. Type a new version to edit")
        
        user_input = input("\nYour choice: ").strip()
        
        if user_input:
            human_response = {"action": "edit", "edited_prompt": user_input}
        else:
            human_response = {"action": "approve"}
        
        # Resume
        print("\n🤖 Generating final answer with Ollama...")
        final_result = app.invoke(Command(resume=human_response), config=config)
        
        print("\n" + "="*50)
        print("📊 RESULTS")
        print("="*50)
        print(f"Final prompt: {final_result['final_prompt']}")
        print(f"\n🎯 Answer:\n{final_result['final_answer']}")


if __name__ == "__main__":
    print("Human in the Loop LangGraph App with Ollama")
    print("Model: llama3.2")
    print("Temperature: 0.7")
    print("\n1. Run example")
    print("2. Interactive mode")
    
    choice = input("Choose (1 or 2): ").strip()
    
    if choice == "2":
        interactive_run()
    else:
        run_app_example()
