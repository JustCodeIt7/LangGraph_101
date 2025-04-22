from typing import Annotated, Dict, List, Literal, TypedDict
import operator
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END


# Define our state schema using TypedDict
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # Accumulate messages
    query_type: str  # Will be used for conditional routing
    result: str  # Final result to return to user


# Node 1: Classifier that determines query type
def classify_query(state: AgentState) -> Dict:
    """Determine what type of query the user has sent."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Get the last message from the user
    user_message = state["messages"][-1].content
    
    # Create a prompt to classify the query
    prompt = f"""
    Classify the following user query into EXACTLY ONE of these categories:
    - 'factual': User is asking for information or facts
    - 'creative': User is asking for creative content like stories or ideas
    - 'action': User is asking to perform a specific action
    
    User query: {user_message}
    
    Return ONLY the category name without explanation or additional text.
    """
    
    # Get the classification
    messages = [SystemMessage(content=prompt)]
    response = llm.invoke(messages)
    
    # The classification will be used to route to the appropriate node
    query_type = response.content.strip().lower()
    
    # Return the update to state
    return {"query_type": query_type}


# Node 2: Handle factual queries
def process_factual_query(state: AgentState) -> Dict:
    """Process queries that ask for factual information."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Extract the user's question
    user_message = state["messages"][-1].content
    
    # Create a prompt focused on factual, accurate information
    messages = [
        SystemMessage(content="You are a helpful assistant that provides factual, accurate information. Cite sources when possible."),
        HumanMessage(content=user_message)
    ]
    
    # Generate a factual response
    response = llm.invoke(messages)
    
    # Add the AI's response to the messages and set the result
    return {
        "messages": [AIMessage(content=response.content)],
        "result": response.content
    }


# Node 3: Handle creative queries
def process_creative_query(state: AgentState) -> Dict:
    """Process queries that ask for creative content."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Extract the user's request
    user_message = state["messages"][-1].content
    
    # Create a prompt focused on creativity
    messages = [
        SystemMessage(content="You are a creative assistant that generates imaginative, engaging content. Feel free to be original and think outside the box."),
        HumanMessage(content=user_message)
    ]
    
    # Generate a creative response
    response = llm.invoke(messages)
    
    # Add the AI's response to the messages and set the result
    return {
        "messages": [AIMessage(content=response.content)],
        "result": response.content
    }


# Node 4: Handle action queries
def process_action_query(state: AgentState) -> Dict:
    """Process queries that request an action."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Extract the user's request
    user_message = state["messages"][-1].content
    
    # Create a prompt focused on actions
    messages = [
        SystemMessage(content="You are an assistant that helps users understand how to perform actions. Explain the steps clearly and thoroughly."),
        HumanMessage(content=user_message)
    ]
    
    # Generate an action-oriented response
    response = llm.invoke(messages)
    
    # Add the AI's response to the messages and set the result
    return {
        "messages": [AIMessage(content=response.content)],
        "result": response.content
    }


# Now let's create our graph with conditional edges
def create_agent_graph():
    """Create a graph with conditional edges based on query type."""
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("factual", process_factual_query)
    graph.add_node("creative", process_creative_query)
    graph.add_node("action", process_action_query)
    
    # Set the entry point
    graph.set_entry_point("classify")
    
    # Define the condition function that will determine which path to take
    def route_by_query_type(state: AgentState) -> str:
        """Route to the appropriate node based on query type."""
        return state["query_type"]
    
    # Add conditional edges from the classifier node
    graph.add_conditional_edges(
        "classify",  # Source node
        route_by_query_type,  # Conditional function (positional)
        {  # Path map (positional)
            "factual": "factual",
            "creative": "creative",
            "action": "action",
            # Default case if none of the above match
            "_default": "factual"
        }
    )
    
    # All processing nodes go to END
    graph.add_edge("factual", END)
    graph.add_edge("creative", END)
    graph.add_edge("action", END)
    
    # Compile the graph
    return graph.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    agent = create_agent_graph()
    
    # Example queries to test different paths
    examples = [
        "What is the capital of France?",                 # factual
        "Write a short poem about artificial intelligence", # creative
        "How do I reset my router?"                       # action
    ]
    
    # Test each example
    for example in examples:
        print(f"\n\n--- PROCESSING: {example} ---")
        
        # Initialize state with the user query
        initial_state = {
            "messages": [HumanMessage(content=example)],
            "query_type": "",  # Will be filled by classifier
            "result": ""       # Will be filled by processing node
        }
        
        # Process the query
        result = agent.invoke(initial_state)
        
        # Print the results
        print(f"Query type: {result['query_type']}")
        print(f"Result: {result['result']}")
