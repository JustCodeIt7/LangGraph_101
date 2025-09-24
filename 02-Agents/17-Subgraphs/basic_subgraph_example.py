"""
Basic LangGraph Subgraph Example

This example demonstrates:
1. Creating a subgraph with its own state and logic
2. Integrating the subgraph into a parent graph
3. State management between parent and subgraph
4. Sequential execution flow

Key Concepts:
- Subgraphs are independent, reusable components
- Each graph has its own state structure
- Parent graph invokes subgraph like any other node
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# --- SUBGRAPH: Text Processing Module ---
class TextProcessingState(TypedDict):
    """State for the text processing subgraph"""
    raw_text: str
    processed_text: str


def clean_text(state: TextProcessingState) -> dict:
    """Remove extra whitespace and normalize"""
    print("🔧 Cleaning text...")
    raw = state['raw_text']
    cleaned = ' '.join(raw.split())
    return {'processed_text': cleaned}


def add_formatting(state: TextProcessingState) -> dict:
    """Add basic formatting"""
    print("✨ Adding formatting...")
    text = state['processed_text']
    formatted = f"📝 {text.title()}"
    return {'processed_text': formatted}


# Build the text processing subgraph
text_processor = StateGraph(TextProcessingState)
text_processor.add_node('clean', clean_text)
text_processor.add_node('format', add_formatting)

text_processor.add_edge(START, 'clean')
text_processor.add_edge('clean', 'format')
text_processor.add_edge('format', END)

# Compile the subgraph
text_processing_graph = text_processor.compile()


# --- PARENT GRAPH: Document Workflow ---
class DocumentState(TypedDict):
    """State for the main document processing workflow"""
    user_input: str
    document_title: str
    final_document: str


def validate_input(state: DocumentState) -> dict:
    """Validate user input"""
    print("🔍 Validating input...")
    user_text = state['user_input']
    if not user_text.strip():
        raise ValueError("Input cannot be empty")
    return {'document_title': 'Untitled Document'}


def process_with_subgraph(state: DocumentState) -> dict:
    """Invoke the text processing subgraph"""
    print("⚡ Processing text with subgraph...")
    
    # Prepare input for subgraph
    subgraph_input = {'raw_text': state['user_input']}
    
    # Execute subgraph
    result = text_processing_graph.invoke(subgraph_input)
    
    # Extract result and update parent state
    processed_text = result['processed_text']
    return {'final_document': processed_text}


def finalize_document(state: DocumentState) -> dict:
    """Add document metadata"""
    print("📄 Finalizing document...")
    content = state['final_document']
    title = state['document_title']
    
    final_doc = f"""
DOCUMENT: {title}
{'='*40}
{content}
{'='*40}
Status: Complete ✅
    """.strip()
    
    return {'final_document': final_doc}


# Build the main workflow graph
main_workflow = StateGraph(DocumentState)
main_workflow.add_node('validate', validate_input)
main_workflow.add_node('process', process_with_subgraph)
main_workflow.add_node('finalize', finalize_document)

main_workflow.add_edge(START, 'validate')
main_workflow.add_edge('validate', 'process')
main_workflow.add_edge('process', 'finalize')
main_workflow.add_edge('finalize', END)

# Compile the main graph
document_processor = main_workflow.compile()


# --- EXECUTION ---
if __name__ == '__main__':
    # Test input with messy formatting
    test_input = {
        'user_input': '   hello    world   this is    a test   document   '
    }
    
    print("🚀 Starting document processing workflow\n")
    
    # Stream execution to see each step
    for event in document_processor.stream(test_input):
        for node_name, output in event.items():
            print(f"Node '{node_name}' completed:")
            print(f"  Output: {output}")
            print("-" * 50)
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    
    # Get final result
    final_result = document_processor.invoke(test_input)
    print(final_result['final_document'])