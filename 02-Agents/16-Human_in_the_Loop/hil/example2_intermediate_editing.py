import asyncio
from typing import Dict, Any, List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pytest
from unittest.mock import patch, MagicMock
import textwrap

class SummaryState:
    def __init__(self):
        self.document: str = ""
        self.original_summary: str = ""
        self.edited_summary: str = ""
        self.follow_up_questions: List[str] = []
        self.human_edited: bool = False
        self.edit_feedback: str = ""

def create_summary_editing_graph():
    """Create an intermediate human-in-the-loop graph for document summarization."""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    def generate_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an initial summary of the document."""
        try:
            document = state.get("document", "")
            
            if not document.strip():
                raise ValueError("Document is empty")
            
            # Split document if it's too long
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(document)
            
            if len(chunks) == 1:
                prompt = f"Please provide a concise but comprehensive summary of the following document:\n\n{document}"
            else:
                # Handle multiple chunks
                prompt = f"Please provide a concise but comprehensive summary of the following document (showing first part):\n\n{chunks[0]}"
            
            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            
            state["original_summary"] = response.content.strip()
            
            print("📄 Original Summary Generated:")
            print("-" * 40)
            print(state["original_summary"])
            print("-" * 40)
            
            return state
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            state["original_summary"] = f"Error: Could not generate summary - {e}"
            return state
    
    def human_edit_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """Allow human to edit the generated summary."""
        try:
            original_summary = state.get("original_summary", "")
            
            print(f"\n--- Human Editing Required ---")
            print("Original Summary:")
            print(original_summary)
            print("\nOptions:")
            print("1. Keep the summary as-is (press Enter)")
            print("2. Edit the summary")
            print("3. Provide feedback for regeneration")
            
            choice = input("\nChoose option (1/2/3) or press Enter for option 1: ").strip()
            
            if choice == "2":
                print("\nPlease edit the summary below:")
                print("(Copy the original and modify as needed)")
                edited_summary = input("Edited summary: ").strip()
                
                if edited_summary:
                    state["edited_summary"] = edited_summary
                    state["human_edited"] = True
                    state["edit_feedback"] = "Human edited the summary"
                    print("✅ Summary updated with your edits!")
                else:
                    state["edited_summary"] = original_summary
                    state["human_edited"] = False
                    state["edit_feedback"] = "No edits provided, keeping original"
                    
            elif choice == "3":
                feedback = input("What should be changed in the summary? ").strip()
                state["edit_feedback"] = feedback
                state["edited_summary"] = original_summary  # Will be regenerated
                state["human_edited"] = False
                print("📝 Feedback noted. Summary will be regenerated.")
                return state  # Will trigger regeneration
                
            else:  # Default: keep original
                state["edited_summary"] = original_summary
                state["human_edited"] = False
                state["edit_feedback"] = "Summary approved as-is"
                print("✅ Original summary approved!")
            
            return state
            
        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user")
            state["edited_summary"] = state.get("original_summary", "")
            state["human_edited"] = False
            state["edit_feedback"] = "Process interrupted"
            return state
        except Exception as e:
            print(f"Error in human editing: {e}")
            state["edited_summary"] = state.get("original_summary", "")
            state["human_edited"] = False
            state["edit_feedback"] = f"Error: {e}"
            return state
    
    def generate_follow_up_questions(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate follow-up questions based on the final summary."""
        try:
            summary = state.get("edited_summary", "")
            
            prompt = f"""Based on the following summary, generate 3-5 thoughtful follow-up questions that would help deepen understanding of the topic:

Summary: {summary}

Please provide questions that:
1. Explore implications or consequences
2. Ask about potential solutions or alternatives
3. Seek clarification on complex points
4. Connect to broader contexts or related topics

Format as a numbered list."""

            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            
            # Parse the response to extract questions
            questions_text = response.content.strip()
            # Simple parsing - split by lines and filter numbered items
            lines = questions_text.split('\n')
            questions = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean up
                    clean_question = line.split('.', 1)[-1].strip()
                    if clean_question:
                        questions.append(clean_question)
            
            state["follow_up_questions"] = questions
            
            print("\n🤔 Follow-up Questions Generated:")
            print("-" * 40)
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            print("-" * 40)
            
            return state
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            state["follow_up_questions"] = [f"Error generating questions: {e}"]
            return state
    
    def check_summary_status(state: Dict[str, Any]) -> str:
        """Check if summary needs regeneration or can proceed."""
        edit_feedback = state.get("edit_feedback", "")
        human_edited = state.get("human_edited", False)
        
        # If feedback suggests regeneration is needed
        if "regenerated" in edit_feedback.lower() or "change" in edit_feedback.lower():
            return "regenerate"
        else:
            return "proceed"
    
    def regenerate_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """Regenerate summary based on feedback."""
        try:
            document = state.get("document", "")
            feedback = state.get("edit_feedback", "")
            
            prompt = f"""Please provide a revised summary of the following document, incorporating this feedback: {feedback}

Document: {document}

Please address the feedback while maintaining accuracy and comprehensiveness."""

            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            
            state["original_summary"] = response.content.strip()
            
            print("📄 Regenerated Summary:")
            print("-" * 40)
            print(state["original_summary"])
            print("-" * 40)
            
            return state
            
        except Exception as e:
            print(f"Error regenerating summary: {e}")
            return state
    
    def finalize_summary_process(state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the summary process."""
        edited_summary = state.get("edited_summary", "")
        human_edited = state.get("human_edited", False)
        questions = state.get("follow_up_questions", [])
        
        print(f"\n🎉 Summary Process Complete!")
        print(f"Human editing: {'Yes' if human_edited else 'No'}")
        print(f"Follow-up questions generated: {len(questions)}")
        
        print(f"\n📋 Final Summary:")
        print("=" * 50)
        print(edited_summary)
        print("=" * 50)
        
        return state
    
    # Create the graph
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("human_edit", human_edit_summary)
    workflow.add_node("regenerate", regenerate_summary)
    workflow.add_node("generate_questions", generate_follow_up_questions)
    workflow.add_node("finalize", finalize_summary_process)
    
    # Add edges
    workflow.set_entry_point("generate_summary")
    workflow.add_edge("generate_summary", "human_edit")
    workflow.add_conditional_edges(
        "human_edit",
        check_summary_status,
        {
            "regenerate": "regenerate",
            "proceed": "generate_questions"
        }
    )
    workflow.add_edge("regenerate", "human_edit")
    workflow.add_edge("generate_questions", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# Example usage
def run_intermediate_example():
    """Run the intermediate summary editing example."""
    print("=== Example 2: Document Summarization with Human Editing ===\n")
    
    # Sample document
    sample_document = textwrap.dedent("""
    Climate change represents one of the most pressing challenges of our time, with far-reaching implications 
    for environmental sustainability, economic stability, and social equity. The phenomenon, primarily driven 
    by increased greenhouse gas emissions from human activities, has led to rising global temperatures, 
    melting ice caps, and more frequent extreme weather events.
    
    Recent scientific studies indicate that global temperatures have risen by approximately 1.1 degrees 
    Celsius since the late 19th century, with the most rapid warming occurring in the past four decades. 
    This warming trend has triggered a cascade of environmental changes, including sea-level rise, ocean 
    acidification, and shifts in precipitation patterns.
    
    The economic implications are equally significant. Climate-related disasters have caused billions of 
    dollars in damages annually, affecting infrastructure, agriculture, and human settlements. Developing 
    nations are particularly vulnerable, often lacking the resources to adapt to changing conditions or 
    recover from climate-related disasters.
    
    Addressing climate change requires coordinated global action, including rapid decarbonization of energy 
    systems, protection and restoration of natural ecosystems, and development of climate-resilient 
    infrastructure. The transition to renewable energy sources, implementation of carbon pricing mechanisms, 
    and investment in green technologies are critical components of effective climate action.
    """).strip()
    
    # Create the graph
    graph = create_summary_editing_graph()
    
    # Initial state
    initial_state = {
        "document": sample_document,
        "original_summary": "",
        "edited_summary": "",
        "follow_up_questions": [],
        "human_edited": False,
        "edit_feedback": ""
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": "example2"}}
    final_state = graph.invoke(initial_state, config)
    
    return final_state

# Unit Tests
class TestSummaryEditing:
    def test_summary_state_structure(self):
        """Test the summary state structure."""
        state = SummaryState()
        assert hasattr(state, 'document')
        assert hasattr(state, 'original_summary')
        assert hasattr(state, 'edited_summary')
        assert hasattr(state, 'follow_up_questions')
        assert hasattr(state, 'human_edited')
        assert hasattr(state, 'edit_feedback')
    
    def test_document_processing(self):
        """Test document processing logic."""
        test_document = "This is a test document for summarization."
        
        # Test basic document validation
        assert len(test_document.strip()) > 0
        assert isinstance(test_document, str)
    
    def test_summary_status_check(self):
        """Test the summary status checking logic."""
        # Test proceed scenario
        state1 = {
            "edit_feedback": "Summary approved as-is",
            "human_edited": False
        }
        
        edit_feedback = state1.get("edit_feedback", "")
        if "regenerated" in edit_feedback.lower() or "change" in edit_feedback.lower():
            result1 = "regenerate"
        else:
            result1 = "proceed"
        
        assert result1 == "proceed"
        
        # Test regenerate scenario
        state2 = {
            "edit_feedback": "Please change the focus to economic impacts",
            "human_edited": False
        }
        
        edit_feedback = state2.get("edit_feedback", "")
        if "regenerated" in edit_feedback.lower() or "change" in edit_feedback.lower():
            result2 = "regenerate"
        else:
            result2 = "proceed"
        
        assert result2 == "regenerate"
    
    def test_questions_parsing(self):
        """Test follow-up questions parsing logic."""
        sample_response = """1. What are the main economic impacts?
2. How can developing nations adapt?
3. What role does technology play?"""
        
        lines = sample_response.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                clean_question = line.split('.', 1)[-1].strip()
                if clean_question:
                    questions.append(clean_question)
        
        assert len(questions) == 3
        assert "What are the main economic impacts?" in questions
        assert "How can developing nations adapt?" in questions
        assert "What role does technology play?" in questions

# Run the tests
if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests for Example 2...")
    pytest.main([__file__ + "::TestSummaryEditing", "-v"])
    
    print("\n" + "="*50)
    print("Unit tests completed. Now running the interactive example...")
    print("="*50 + "\n")
    
    # Run the interactive example
    # Uncomment the line below to run interactively
    # run_intermediate_example()