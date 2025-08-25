import os
from typing import Dict, TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic.v1 import BaseModel, Field  # Updated import to fix deprecation warning
import getpass

# Set your OpenAI API key


# Define the state
class ResearchState(TypedDict):
    prompt: str
    plan: str
    approved: bool
    research_results: str
    messages: Annotated[List[Dict[str, str]], "append"]

# LLM setup
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Node 1: Generate Research Plan
def generate_plan(state: ResearchState) -> ResearchState:
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="Create a detailed research plan for the following prompt: {prompt}. "
                 "The plan should include steps like: 1. Key questions to answer. "
                 "2. Sources to consult. 3. Methods for gathering information."
    )
    chain = prompt_template | llm
    response = chain.invoke({"prompt": state["prompt"]})
    return {
        "plan": response.content,
        "messages": [{"role": "assistant", "content": f"Generated plan: {response.content}"}]
    }

# Node 2: Human Approval
def human_approval(state: ResearchState) -> ResearchState:
    print("\nResearch Plan:\n")
    print(state["plan"])
    print("\nApprove (y), Edit (e), or Reject (n)?")
    user_input = input().strip().lower()
    
    if user_input == "y":
        return {"approved": True, "messages": [{"role": "human", "content": "Approved"}]}
    elif user_input == "e":
        print("Enter your edited plan:")
        edited_plan = input()
        return {
            "plan": edited_plan,
            "approved": True,
            "messages": [{"role": "human", "content": f"Edited plan: {edited_plan}"}]
        }
    else:
        return {"approved": False, "messages": [{"role": "human", "content": "Rejected"}]}

# Node 3: Perform Research
class ResearchOutput(BaseModel):
    results: str = Field(description="The research findings based on the plan.")

def perform_research(state: ResearchState) -> ResearchState:
    prompt_template = PromptTemplate(
        input_variables=["prompt", "plan"],
        template="Perform research on the prompt: {prompt} following this plan: {plan}. "
                 "Summarize the key findings."
    )
    structured_llm = llm.with_structured_output(ResearchOutput)
    chain = prompt_template | structured_llm
    response = chain.invoke({"prompt": state["prompt"], "plan": state["plan"]})
    return {
        "research_results": response.results,
        "messages": [{"role": "assistant", "content": f"Research results: {response.results}"}]
    }

# Conditional edge: Check approval
def check_approval(state: ResearchState) -> str:
    if state["approved"]:
        return "perform_research"
    else:
        return END

# Build the graph
workflow = StateGraph(ResearchState)

workflow.add_node("generate_plan", generate_plan)
workflow.add_node("human_approval", human_approval)
workflow.add_node("perform_research", perform_research)

workflow.set_entry_point("generate_plan")
workflow.add_edge("generate_plan", "human_approval")
workflow.add_conditional_edges(
    "human_approval",
    check_approval,
    {
        "perform_research": "perform_research",
        END: END
    }
)
workflow.add_edge("perform_research", END)

# Compile the graph with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

# Function to run the app
def run_research_app(research_prompt: str):
    initial_state = ResearchState(
        prompt=research_prompt,
        plan="",
        approved=False,
        research_results="",
        messages=[]
    )
    
    # Run until human approval node
    config = {"configurable": {"thread_id": "research_thread"}}
    app.invoke(initial_state, config=config)
    
    # Now, resume after human input
    # In a real app, this would be in a loop or handled differently, but for simplicity:
    print("Resuming after human input...")
    app.invoke(None, config=config)  # Resume from checkpoint
    
    # Get final state
    final_checkpoint = app.get_state(config)
    print("\nFinal Research Results:\n")
    print(final_checkpoint.values["research_results"])

# Example usage
if __name__ == "__main__":
    prompt = input("Enter your research prompt: ")
    run_research_app(prompt)