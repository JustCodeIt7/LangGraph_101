import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from pydantic import BaseModel , Field
from rich import print
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

################################ Application Setup & Configuration ################################

# Load environment variables from a .env file
load_dotenv()

# Initialize the language model for all tasks
# Note: Using 'gpt-4.1-nano' as a suitable, efficient alternative to a non-existent model.
llm = ChatOpenAI(model="gpt-4.1-nano", max_tokens=500)

################################ State & Schema Definitions ################################

# Define a Pydantic model for structured LLM output during evaluation
class Evaluation(BaseModel):
    """Evaluation of the email response."""
    tone_ok: bool = Field(description="Is the tone professional and polite?")
    complete: bool = Field(description="Does the response address all points from the original email?")
    clear: bool = Field(description="Is the response easy to understand?")

# Define the state object that will be passed between nodes in the graph
class GraphState(TypedDict):
    """Represents the state of our graph."""
    original_email: str
    response_draft: Optional[str]
    evaluation: Optional[Evaluation]
    optimization_feedback: Optional[str]
    revision_count: int

################################ Graph Node Definitions ################################

# Define the node to generate or revise the email response
def generate_response_node(state: GraphState):
    """Generates or revises an email response."""
    print("--- DRAFTING RESPONSE ---")
    revision_count = state['revision_count']
    feedback = state.get('optimization_feedback')

    # Construct the prompt, including feedback if it's a revision
    prompt = f"""You are a professional assistant. Your task is to generate a polite and complete response to the following email.

Original Email:
---
{state['original_email']}
---
"""
    # Append revision feedback to the prompt if it exists
    if feedback:
        prompt += f"\nThis is revision number {revision_count}. Please improve the previous draft based on this feedback: {feedback}"

    # Invoke the LLM to generate the response
    response_draft = llm.invoke(prompt).content
    return {"response_draft": response_draft, "revision_count": revision_count + 1}

# Define the node to evaluate the quality of the generated response
def evaluate_response_node(state: GraphState):
    """Evaluates the generated response against quality criteria."""
    print("--- EVALUATING RESPONSE ---")

    # Bind the Pydantic schema to the LLM to force structured output
    structured_llm = llm.with_structured_output(Evaluation)

    # Construct the prompt for the evaluator
    prompt = f"""Evaluate the following email response based on three criteria: tone, completeness, and clarity.

Original Email:
---
{state['original_email']}
---

Generated Response:
---
{state['response_draft']}
---

Evaluation Criteria:
1. Tone: Is the response professional and polite?
2. Completeness: Does it address all questions or points from the original email?
3. Clarity: Is the language clear and easy to understand?
"""
    # Invoke the structured LLM to get an evaluation object
    evaluation = structured_llm.invoke(prompt)
    print(f"Evaluation: Tone OK? {evaluation.tone_ok}, Complete? {evaluation.complete}, Clear? {evaluation.clear}")
    return {"evaluation": evaluation}

# Define the node to generate actionable feedback for improvement
def optimize_response_node(state: GraphState):
    """Generates specific feedback for improving the response."""
    print("--- GENERATING OPTIMIZATION FEEDBACK ---")
    evaluation = state['evaluation']

    # Construct the prompt for the optimizer LLM
    feedback_prompt = f"""The following email response was found to have issues.

Original Email:
---
{state['original_email']}
---

Generated Response:
---
{state['response_draft']}
---

Evaluation Issues:
- Tone was appropriate: {evaluation.tone_ok}
- Response was complete: {evaluation.complete}
- Response was clear: {evaluation.clear}

Provide specific, actionable feedback to the writer on how to fix these issues. Focus only on what needs improvement.
"""
    # Invoke the LLM to generate optimization feedback
    optimization_feedback = llm.invoke(feedback_prompt).content
    return {"optimization_feedback": optimization_feedback}

################################ Conditional Logic (Quality Gate) ################################

MAX_REVISIONS = 3 # Set a limit for the number of revision cycles

# Define the conditional edge that routes based on evaluation results
def quality_gate(state: GraphState):
    """Determines whether to accept the response or send for revision."""
    evaluation = state['evaluation']

    # Check if the response meets all quality criteria
    if evaluation.tone_ok and evaluation.complete and evaluation.clear:
        print("--- QUALITY GATE: PASSED ---")
        return "end" # Finish the workflow

    # Check if the revision limit has been reached
    if state['revision_count'] >= MAX_REVISIONS:
        print("--- QUALITY GATE: MAX REVISIONS REACHED ---")
        return "end" # Finish the workflow to avoid infinite loops

    print("--- QUALITY GATE: FAILED, ROUTING TO OPTIMIZER ---")
    return "optimize" # Send for revision

################################ Graph Assembly ################################

# Initialize the state graph
workflow = StateGraph(GraphState)

# Add the defined functions as nodes in the graph
workflow.add_node("email_generator", generate_response_node)
workflow.add_node("email_evaluator", evaluate_response_node)
workflow.add_node("email_optimizer", optimize_response_node)

# Set the entry point for the graph
workflow.set_entry_point("email_generator")

# Define the directed edges that connect the nodes
workflow.add_edge("email_generator", "email_evaluator")
workflow.add_conditional_edges(
    "email_evaluator",
    quality_gate,
    {
        "end": END,                 # If quality gate returns "end", stop
        "optimize": "email_optimizer" # If it returns "optimize", go to optimizer
    }
)
workflow.add_edge("email_optimizer", "email_generator") # Loop back for revision

# Compile the graph into a runnable LangChain application
app = workflow.compile()

################################ Workflow Execution ################################

# Define a dictionary of sample emails to test the workflow
sample_emails = {
    "Complaint": "Hi, I'm writing to complain about my recent order #123. The item arrived damaged and it's not the color I ordered. This is unacceptable.",
    "Inquiry": "Hello, I was wondering if your software supports integration with Salesforce? Also, what are your pricing tiers?",
    "Meeting Request": "Hi team, can we schedule a meeting for next week to discuss the Q3 project plan? Please let me know your availability."
}

# Iterate through the sample emails and run the workflow for each
for name, email_content in sample_emails.items():
    print(f"\n\n--- Running Workflow for: {name} ---")
    # Prepare the initial state for the graph
    inputs = {"original_email": email_content, "revision_count": 0}

    # Execute the compiled graph with the initial inputs
    final_state = app.invoke(inputs)

    # Print the final generated response
    print("\n--- FINAL RESPONSE ---")
    print(final_state['response_draft'])
    print("------------------------\n")
