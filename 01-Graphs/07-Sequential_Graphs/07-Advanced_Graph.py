# %% [markdown]
# # LangGraph Tutorial: Comprehensive Graph Patterns
# 
# ## Introduction
# 
# This tutorial demonstrates key LangGraph concepts through a practical chatbot example:
# - **Nodes**: Individual processing units
# - **Edges**: Connections between nodes
# - **Conditional Routing**: Dynamic path selection based on state
# - **Looping**: Repeating execution patterns
# - **State Management**: Shared data across the graph
# 
# We'll build a smart customer service chatbot that demonstrates all these concepts.

# %%
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from rich import print
from IPython.display import Image, display
import json
import random

# Load environment variables from .env file
load_dotenv("../.env")

# %% [markdown]
# ## 1. State Definition
# 
# The state is shared across all nodes and contains all the information needed for our graph execution.

# %%
class CustomerServiceState(TypedDict):
    """
    Comprehensive state for our customer service chatbot.
    This demonstrates how to structure state for complex workflows.
    """
    # User information
    user_name: str
    user_email: str
    user_sentiment: str  # positive, negative, neutral
    
    # Conversation flow
    conversation_history: list[str]
    current_message: str
    bot_response: str
    
    # Routing logic
    issue_type: str  # technical, billing, general
    escalate_to_human: bool
    max_attempts: int
    current_attempt: int
    
    # Final state
    conversation_resolved: bool
    satisfaction_score: int

# %% [markdown]
# ## 2. Node Functions
# 
# Each node represents a specific step in our customer service workflow.

# %%
def welcome_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 1: Welcome and initial information gathering.
    Demonstrates basic node functionality and state initialization.
    """
    print("🤖 Welcome Node: Starting customer service session...")
    
    # Initialize default values
    state["conversation_history"] = []
    state["current_attempt"] = 1
    state["max_attempts"] = 3
    state["escalate_to_human"] = False
    state["conversation_resolved"] = False
    state["satisfaction_score"] = 0
    
    # Simulate user input (in real app, this would come from UI)
    if not state.get("user_name"):
        state["user_name"] = "John Doe"  # In tutorial, we'll use sample data
    
    welcome_message = f"Hello {state['user_name']}! I'm here to help you today. How can I assist you?"
    state["bot_response"] = welcome_message
    state["conversation_history"].append(f"Bot: {welcome_message}")
    
    print(f"✅ Welcome complete for {state['user_name']}")
    return state

def sentiment_analysis_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 2: Analyze user sentiment to determine response strategy.
    Demonstrates processing logic within nodes.
    """
    print("🎭 Sentiment Analysis Node: Analyzing user emotion...")
    
    # Simulate sentiment analysis (in real app, use actual NLP)
    current_msg = state.get("current_message", "").lower()
    
    # Simple keyword-based sentiment analysis for demo
    if any(word in current_msg for word in ["angry", "frustrated", "terrible", "awful", "hate"]):
        state["user_sentiment"] = "negative"
    elif any(word in current_msg for word in ["happy", "great", "love", "awesome", "excellent"]):
        state["user_sentiment"] = "positive"
    else:
        state["user_sentiment"] = "neutral"
    
    print(f"✅ Sentiment detected: {state['user_sentiment']}")
    return state

def classify_issue_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 3: Classify the type of issue for proper routing.
    Demonstrates classification logic that affects routing.
    """
    print("🏷️ Issue Classification Node: Determining issue type...")
    
    current_msg = state.get("current_message", "").lower()
    
    # Classify issue type based on keywords
    if any(word in current_msg for word in ["password", "login", "error", "bug", "technical", "broken"]):
        state["issue_type"] = "technical"
    elif any(word in current_msg for word in ["bill", "charge", "payment", "refund", "money", "price"]):
        state["issue_type"] = "billing"
    else:
        state["issue_type"] = "general"
    
    print(f"✅ Issue classified as: {state['issue_type']}")
    return state

def technical_support_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 4a: Handle technical issues.
    Demonstrates specialized processing for specific routes.
    """
    print("🔧 Technical Support Node: Providing technical assistance...")
    
    responses = [
        "Let me help you with that technical issue. Have you tried restarting the application?",
        "I understand you're experiencing technical difficulties. Let's troubleshoot step by step.",
        "Technical issues can be frustrating. Let me guide you through some common solutions."
    ]
    
    response = random.choice(responses)
    state["bot_response"] = response
    state["conversation_history"].append(f"Bot: {response}")
    
    # Simulate some resolution logic
    if state["current_attempt"] >= 2:
        state["conversation_resolved"] = True
        state["satisfaction_score"] = 8
    
    print("✅ Technical support response generated")
    return state

def billing_support_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 4b: Handle billing issues.
    Demonstrates alternative routing path.
    """
    print("💳 Billing Support Node: Addressing billing concerns...")
    
    responses = [
        "I'd be happy to help with your billing question. Let me look into your account.",
        "I understand billing issues can be concerning. Let me explain the charges.",
        "Let's resolve this billing matter together. I'll review your account details."
    ]
    
    response = random.choice(responses)
    state["bot_response"] = response
    state["conversation_history"].append(f"Bot: {response}")
    
    # Billing issues might need human escalation
    if state["user_sentiment"] == "negative":
        state["escalate_to_human"] = True
    else:
        state["conversation_resolved"] = True
        state["satisfaction_score"] = 7
    
    print("✅ Billing support response generated")
    return state

def general_support_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 4c: Handle general inquiries.
    Demonstrates default routing path.
    """
    print("💬 General Support Node: Providing general assistance...")
    
    responses = [
        "I'm here to help with your inquiry. Could you provide more details?",
        "Thank you for reaching out. Let me assist you with your question.",
        "I'd be glad to help you today. What specific information do you need?"
    ]
    
    response = random.choice(responses)
    state["bot_response"] = response
    state["conversation_history"].append(f"Bot: {response}")
    
    state["conversation_resolved"] = True
    state["satisfaction_score"] = 6
    
    print("✅ General support response generated")
    return state

def escalation_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 5: Handle escalation to human agents.
    Demonstrates conditional node execution.
    """
    print("🚨 Escalation Node: Connecting to human agent...")
    
    escalation_response = (
        "I understand this requires additional attention. "
        "I'm connecting you with one of our human specialists who can better assist you."
    )
    
    state["bot_response"] = escalation_response
    state["conversation_history"].append(f"Bot: {escalation_response}")
    state["conversation_resolved"] = True
    state["satisfaction_score"] = 5  # Lower score due to escalation
    
    print("✅ Escalated to human agent")
    return state

def follow_up_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 6: Follow up if issue not resolved (demonstrates looping).
    This node can loop back for multiple attempts.
    """
    print("🔄 Follow-up Node: Checking if more help is needed...")
    
    state["current_attempt"] += 1
    
    follow_up_responses = [
        "Is there anything else I can help you with regarding this issue?",
        "Would you like me to try a different approach to solve this problem?",
        "Let me know if you need any additional assistance."
    ]
    
    response = random.choice(follow_up_responses)
    state["bot_response"] = response
    state["conversation_history"].append(f"Bot: {response}")
    
    print(f"✅ Follow-up attempt {state['current_attempt']}")
    return state

def satisfaction_survey_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node 7: Final satisfaction survey.
    Demonstrates final processing before END.
    """
    print("📊 Satisfaction Survey Node: Collecting feedback...")
    
    survey_message = (
        f"Thank you for using our service! "
        f"On a scale of 1-10, how would you rate your experience? "
        f"(Auto-rated: {state['satisfaction_score']}/10)"
    )
    
    state["bot_response"] = survey_message
    state["conversation_history"].append(f"Bot: {survey_message}")
    
    print(f"✅ Satisfaction survey completed: {state['satisfaction_score']}/10")
    return state

# %% [markdown]
# ## 3. Conditional Routing Functions
# 
# These functions determine which path the graph should take based on the current state.

# %%
def route_by_issue_type(state: CustomerServiceState) -> Literal["technical", "billing", "general"]:
    """
    Conditional routing based on issue classification.
    This demonstrates how to implement dynamic routing in LangGraph.
    """
    issue_type = state.get("issue_type", "general")
    print(f"🔀 Routing to {issue_type} support based on issue type")
    return issue_type

def check_escalation_needed(state: CustomerServiceState) -> Literal["escalate", "continue"]:
    """
    Conditional routing for escalation logic.
    Demonstrates complex conditional routing based on multiple factors.
    """
    should_escalate = (
        state.get("escalate_to_human", False) or
        state.get("user_sentiment") == "negative" and state.get("current_attempt", 1) >= 2
    )
    
    route = "escalate" if should_escalate else "continue"
    print(f"🔀 Escalation check: {route}")
    return route

def check_resolution_status(state: CustomerServiceState) -> Literal["resolved", "follow_up", "max_attempts"]:
    """
    Conditional routing for resolution and looping logic.
    Demonstrates looping patterns in LangGraph.
    """
    if state.get("conversation_resolved", False):
        return "resolved"
    elif state.get("current_attempt", 1) >= state.get("max_attempts", 3):
        return "max_attempts"
    else:
        return "follow_up"

# %% [markdown]
# ## 4. Graph Construction
# 
# Now we'll build the complete graph with all the nodes, edges, and conditional routing.

# %%
def create_customer_service_graph():
    """
    Create a comprehensive customer service graph demonstrating:
    - Sequential flow
    - Conditional routing
    - Looping patterns
    - Error handling paths
    """
    
    # Initialize the graph with our state schema
    graph = StateGraph(CustomerServiceState)
    
    # Add all nodes to the graph
    graph.add_node("welcome", welcome_node)
    graph.add_node("sentiment_analysis", sentiment_analysis_node)
    graph.add_node("classify_issue", classify_issue_node)
    graph.add_node("technical_support", technical_support_node)
    graph.add_node("billing_support", billing_support_node)
    graph.add_node("general_support", general_support_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("follow_up", follow_up_node)
    graph.add_node("satisfaction_survey", satisfaction_survey_node)
    
    # Sequential edges (guaranteed flow)
    graph.add_edge(START, "welcome")
    graph.add_edge("welcome", "sentiment_analysis")
    graph.add_edge("sentiment_analysis", "classify_issue")
    
    # Conditional routing based on issue type
    graph.add_conditional_edges(
        "classify_issue",
        route_by_issue_type,
        {
            "technical": "technical_support",
            "billing": "billing_support", 
            "general": "general_support"
        }
    )
    
    # Check for escalation after each support type
    for support_type in ["technical_support", "billing_support", "general_support"]:
        graph.add_conditional_edges(
            support_type,
            check_escalation_needed,
            {
                "escalate": "escalation",
                "continue": "follow_up"
            }
        )
    
    # Resolution check with looping logic
    graph.add_conditional_edges(
        "follow_up",
        check_resolution_status,
        {
            "resolved": "satisfaction_survey",
            "follow_up": "sentiment_analysis",  # Loop back for another attempt
            "max_attempts": "escalation"  # Escalate if max attempts reached
        }
    )
    
    # Final edges to END
    graph.add_edge("escalation", "satisfaction_survey")
    graph.add_edge("satisfaction_survey", END)
    
    return graph.compile()

# Create the compiled graph
app = create_customer_service_graph()

# %% [markdown]
# ## 5. Visualization
# 
# Let's visualize our complex graph to see the flow, conditional routing, and loops.

# %%
# Visualize the Graph
try:
    graph_image = app.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    
    # Save the graph visualization
    os.makedirs("../output", exist_ok=True)
    with open("../output/07-Sequential_Graphs_Tutorial.png", "wb") as f:
        f.write(graph_image)
    print("📊 Graph visualization saved to ../output/07-Sequential_Graphs_Tutorial.png")
except Exception as e:
    print(f"⚠️ Could not generate graph visualization: {e}")

# %% [markdown]
# ## 6. Demonstration Scenarios
# 
# Let's run through different scenarios to demonstrate various graph patterns.

# %%
def run_scenario(scenario_name: str, initial_state: CustomerServiceState):
    """Helper function to run and display scenario results."""
    print(f"\n{'='*60}")
    print(f"🎬 SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    result = app.invoke(initial_state)
    
    print(f"\n📋 Final State Summary:")
    print(f"   User: {result['user_name']}")
    print(f"   Issue Type: {result['issue_type']}")
    print(f"   Sentiment: {result['user_sentiment']}")
    print(f"   Attempts: {result['current_attempt']}")
    print(f"   Resolved: {result['conversation_resolved']}")
    print(f"   Satisfaction: {result['satisfaction_score']}/10")
    print(f"   Escalated: {result['escalate_to_human']}")
    
    print(f"\n💬 Conversation History:")
    for msg in result['conversation_history']:
        print(f"   {msg}")
    
    return result

# %% [markdown]
# ### Scenario 1: Technical Issue (Happy Path)

# %%
# Scenario 1: Technical issue with positive resolution
technical_scenario = {
    "user_name": "Alice Johnson",
    "user_email": "alice@email.com",
    "current_message": "I'm having trouble logging into my account. Can you help?"
}

result1 = run_scenario("Technical Support - Happy Path", technical_scenario)

# %% [markdown]
# ### Scenario 2: Billing Issue (Escalation Path)

# %%
# Scenario 2: Billing issue with negative sentiment leading to escalation
billing_scenario = {
    "user_name": "Bob Smith", 
    "user_email": "bob@email.com",
    "current_message": "I'm angry about these unexpected charges on my bill. This is terrible service!"
}

result2 = run_scenario("Billing Issue - Escalation Path", billing_scenario)

# %% [markdown]
# ### Scenario 3: General Inquiry (Quick Resolution)

# %%
# Scenario 3: General inquiry with quick resolution
general_scenario = {
    "user_name": "Carol Davis",
    "user_email": "carol@email.com", 
    "current_message": "Can you tell me about your service hours?"
}

result3 = run_scenario("General Inquiry - Quick Resolution", general_scenario)

# %% [markdown]
# ### Scenario 4: Complex Technical Issue (Looping Pattern)

# %%
# Scenario 4: Technical issue that requires multiple attempts (demonstrates looping)
complex_scenario = {
    "user_name": "David Wilson",
    "user_email": "david@email.com",
    "current_message": "I have a complex technical error that keeps happening",
    "max_attempts": 2  # Lower max attempts to demonstrate looping and escalation
}

result4 = run_scenario("Complex Technical Issue - Looping Pattern", complex_scenario)

# %% [markdown]
# ## 7. Key Learning Points
# 
# This tutorial demonstrates several important LangGraph concepts:

# %%
print("""
🎓 KEY LANGGRAPH CONCEPTS DEMONSTRATED:

1. 📦 NODES:
   - Individual processing units (welcome_node, sentiment_analysis_node, etc.)
   - Each node receives state and returns modified state
   - Nodes can contain complex business logic

2. 🔗 EDGES:
   - Simple edges: Direct connections (START → welcome)
   - Conditional edges: Dynamic routing based on state
   - Loops: Edges that cycle back to previous nodes

3. 🔀 CONDITIONAL ROUTING:
   - route_by_issue_type(): Routes based on issue classification
   - check_escalation_needed(): Dynamic escalation logic
   - check_resolution_status(): Multi-condition routing with loops

4. 🔄 LOOPING PATTERNS:
   - follow_up → sentiment_analysis: Retry mechanism
   - Max attempt limits prevent infinite loops
   - State tracking (current_attempt) manages loop behavior

5. 📊 STATE MANAGEMENT:
   - TypedDict for structure and type safety
   - Shared state across all nodes
   - State evolution throughout graph execution

6. 🛣️ EXECUTION PATHS:
   - Happy path: welcome → ... → satisfaction_survey → END
   - Escalation path: ... → escalation → satisfaction_survey → END  
   - Loop path: ... → follow_up → sentiment_analysis → ...

7. 🏗️ GRAPH ARCHITECTURE:
   - Modular design with specialized nodes
   - Separation of concerns (routing logic vs. business logic)
   - Scalable pattern for complex workflows
""")

# %% [markdown]
# ## 8. Advanced Patterns and Best Practices

# %%
def demonstrate_advanced_patterns():
    """
    Demonstrate advanced LangGraph patterns and best practices.
    """
    print("🚀 ADVANCED LANGGRAPH PATTERNS:\n")
    
    print("1. 🔧 ERROR HANDLING:")
    print("   - Use try/catch in nodes for graceful error handling")
    print("   - Add error states to handle failures")
    print("   - Implement fallback routing paths\n")
    
    print("2. 🔄 LOOP MANAGEMENT:")
    print("   - Always include exit conditions (max_attempts)")
    print("   - Track iteration state (current_attempt)")
    print("   - Provide escape routes (escalation)\n")
    
    print("3. 🎯 STATE DESIGN:")
    print("   - Use TypedDict for type safety")
    print("   - Keep state flat and simple when possible")
    print("   - Include metadata for routing decisions\n")
    
    print("4. 🔀 ROUTING STRATEGIES:")
    print("   - Separate routing logic from business logic")
    print("   - Use enums/literals for routing return types")
    print("   - Consider all possible state combinations\n")
    
    print("5. 🧪 TESTING APPROACHES:")
    print("   - Test individual nodes in isolation")
    print("   - Create scenario-based integration tests")
    print("   - Verify all routing paths are reachable\n")
    
    print("6. 📈 SCALABILITY:")
    print("   - Design for easy addition of new nodes")
    print("   - Use consistent naming conventions")
    print("   - Document node purposes and routing logic")

demonstrate_advanced_patterns()

# %% [markdown]
# ## 9. Next Steps
# 
# To further explore LangGraph, consider:
# 
# 1. **Add LLM Integration**: Replace simulated responses with actual language models
# 2. **Implement Persistence**: Save conversation state to databases
# 3. **Add Human-in-the-Loop**: Real human escalation workflows
# 4. **Error Recovery**: Robust error handling and recovery mechanisms
# 5. **Monitoring**: Add logging and metrics for production use
# 6. **Parallel Processing**: Explore concurrent node execution
# 7. **Dynamic Graphs**: Build graphs that modify themselves based on runtime conditions
# 
# This tutorial provides a solid foundation for building complex, production-ready LangGraph applications!

# %%
print("\n🎉 Tutorial Complete!")
print("You've learned how to build sophisticated LangGraph applications with:")
print("   ✅ Nodes and Edges")
print("   ✅ Conditional Routing") 
print("   ✅ Looping Patterns")
print("   ✅ State Management")
print("   ✅ Real-world Application Structure")
print("\nHappy coding with LangGraph! 🚀")

# %%

