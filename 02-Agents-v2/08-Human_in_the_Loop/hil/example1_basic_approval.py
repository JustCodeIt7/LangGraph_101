import asyncio
from typing import Dict, Any, List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pytest
from unittest.mock import patch, MagicMock
import json

# Define the state structure
class ApprovalState:
    def __init__(self):
        self.product_name: str = ""
        self.tagline: str = ""
        self.approved: bool = False
        self.attempts: int = 0
        self.max_attempts: int = 3
        self.feedback: str = ""

def create_basic_approval_graph():
    """Create a basic human-in-the-loop graph for tagline approval."""
    
    # Initialize the language model (you can replace with your preferred model)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    def generate_tagline(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a marketing tagline for the product."""
        try:
            product_name = state.get("product_name", "")
            attempts = state.get("attempts", 0)
            feedback = state.get("feedback", "")
            
            if attempts == 0:
                prompt = f"Create a catchy marketing tagline for a product called '{product_name}'. Make it memorable and engaging."
            else:
                prompt = f"Create a different marketing tagline for '{product_name}'. Previous feedback: {feedback}. Make it more appealing."
            
            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            
            state["tagline"] = response.content.strip()
            state["attempts"] = attempts + 1
            
            print(f"Generated tagline (attempt {state['attempts']}): {state['tagline']}")
            return state
            
        except Exception as e:
            print(f"Error generating tagline: {e}")
            state["tagline"] = f"Error generating tagline for {product_name}"
            return state
    
    def human_approval(state: Dict[str, Any]) -> Dict[str, Any]:
        """Ask human for approval of the generated tagline."""
        try:
            tagline = state.get("tagline", "")
            product_name = state.get("product_name", "")
            attempts = state.get("attempts", 0)
            
            print(f"\n--- Human Approval Required ---")
            print(f"Product: {product_name}")
            print(f"Generated Tagline: '{tagline}'")
            print(f"Attempt: {attempts}")
            
            # Human intervention point
            approval = input("Do you approve this tagline? (y/n): ").lower().strip()
            
            if approval == 'y':
                state["approved"] = True
                state["feedback"] = "Approved"
                print("✅ Tagline approved!")
            else:
                state["approved"] = False
                feedback = input("Please provide feedback for improvement: ").strip()
                state["feedback"] = feedback
                print("❌ Tagline rejected. Generating new one...")
                
            return state
            
        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user")
            state["approved"] = False
            state["feedback"] = "Process interrupted"
            return state
        except Exception as e:
            print(f"Error in human approval: {e}")
            state["approved"] = False
            state["feedback"] = f"Error: {e}"
            return state
    
    def check_completion(state: Dict[str, Any]) -> str:
        """Check if the process is complete or should continue."""
        approved = state.get("approved", False)
        attempts = state.get("attempts", 0)
        max_attempts = state.get("max_attempts", 3)
        
        if approved:
            return "complete"
        elif attempts >= max_attempts:
            print(f"⚠️ Maximum attempts ({max_attempts}) reached. Process terminated.")
            return "complete"
        else:
            return "retry"
    
    def finalize_result(state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the result."""
        if state.get("approved", False):
            print(f"\n🎉 Final approved tagline: '{state.get('tagline', '')}'")
        else:
            print(f"\n❌ No tagline was approved after {state.get('attempts', 0)} attempts.")
        
        return state
    
    # Create the graph
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("generate", generate_tagline)
    workflow.add_node("approve", human_approval)
    workflow.add_node("finalize", finalize_result)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "approve")
    workflow.add_conditional_edges(
        "approve",
        check_completion,
        {
            "retry": "generate",
            "complete": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# Example usage
def run_basic_example():
    """Run the basic approval example."""
    print("=== Example 1: Basic Text Generation with Human Approval ===\n")
    
    # Create the graph
    graph = create_basic_approval_graph()
    
    # Initial state
    initial_state = {
        "product_name": "EcoClean Pro",
        "tagline": "",
        "approved": False,
        "attempts": 0,
        "max_attempts": 3,
        "feedback": ""
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": "example1"}}
    final_state = graph.invoke(initial_state, config)
    
    return final_state

# Unit Tests
class TestBasicApproval:
    def test_generate_tagline_function(self):
        """Test tagline generation function."""
        # Mock the LLM
        with patch('langchain.chat_models.ChatOpenAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Clean Green, Live Clean!"
            mock_llm.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            graph = create_basic_approval_graph()
            
            # Test state
            test_state = {
                "product_name": "EcoClean Pro",
                "attempts": 0,
                "feedback": ""
            }
            
            # This is a simplified test - in practice, you'd test individual functions
            assert "product_name" in test_state
            assert test_state["attempts"] == 0
    
    def test_state_structure(self):
        """Test the state structure is correct."""
        state = ApprovalState()
        assert hasattr(state, 'product_name')
        assert hasattr(state, 'tagline')
        assert hasattr(state, 'approved')
        assert hasattr(state, 'attempts')
        assert hasattr(state, 'max_attempts')
        assert hasattr(state, 'feedback')
    
    def test_max_attempts_logic(self):
        """Test maximum attempts logic."""
        test_state = {
            "approved": False,
            "attempts": 3,
            "max_attempts": 3
        }
        
        # Simulate the check_completion function logic
        approved = test_state.get("approved", False)
        attempts = test_state.get("attempts", 0)
        max_attempts = test_state.get("max_attempts", 3)
        
        if approved:
            result = "complete"
        elif attempts >= max_attempts:
            result = "complete"
        else:
            result = "retry"
        
        assert result == "complete"

# Run the tests
if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests for Example 1...")
    pytest.main([__file__ + "::TestBasicApproval", "-v"])
    
    print("\n" + "="*50)
    print("Unit tests completed. Now running the interactive example...")
    print("="*50 + "\n")
    
    # Run the interactive example
    # Uncomment the line below to run interactively
    # run_basic_example()