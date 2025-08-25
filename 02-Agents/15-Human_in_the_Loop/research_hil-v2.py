import os
from typing import Dict, TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic.v1 import BaseModel, Field  # Updated import to fix deprecation warning
import getpass

# Define the state
class ResearchState(TypedDict):
    prompt: str
    plan: str
    approved: bool
    final_report: str
    messages: Annotated[List[Dict[str, str]], "append"]

# LLM setup
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Node 1: Generate Research Plan
def generate_plan(state: ResearchState) -> ResearchState:
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template=(
            "Create a detailed research plan for the following prompt: {prompt}.\n"
            "The plan should include:\n"
            "1) Key questions to answer\n"
            "2) Sources to consult\n"
            "3) Methods for gathering information\n"
            "4) A proposed outline for the final report\n"
        )
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

# Node 3: Write Final Report (no actual research performed)
class ReportOutput(BaseModel):
    report: str = Field(description="The final report text generated solely from the plan and prompt.")

def write_report(state: ResearchState) -> ResearchState:
    prompt_template = PromptTemplate(
        input_variables=["prompt", "plan"],
        template=(
            "You are a professional report writer. Do not perform any external research.\n"
            "Write the final report entirely based on the approved plan and the prompt.\n\n"
            "Prompt:\n{prompt}\n\n"
            "Approved Plan:\n{plan}\n\n"
            "Instructions:\n"
            "- Use the plan's outline and key questions to structure the report.\n"
            "- Synthesize and expand the plan into a clear, cohesive, and well-argued report.\n"
            "- If the plan references sources, include them as placeholders (e.g., [Source Placeholder]) "
            "without fabricating citations.\n"
            "- Include an executive summary, main sections following the plan, and a brief conclusion.\n"
            "- Keep the tone professional and concise.\n\n"
            "Now write the final report:"
        )
    )
    structured_llm = llm.with_structured_output(ReportOutput)
    chain = prompt_template | structured_llm
    response = chain.invoke({"prompt": state["prompt"], "plan": state["plan"]})
    return {
        "final_report": response.report,
        "messages": [{"role": "assistant", "content": f"Final report generated."}]
    }

# Conditional edge: Check approval
def check_approval(state: ResearchState) -> str:
    if state["approved"]:
        return "write_report"
    else:
        return END

# Build the graph
workflow = StateGraph(ResearchState)

workflow.add_node("generate_plan", generate_plan)
workflow.add_node("human_approval", human_approval)
workflow.add_node("write_report", write_report)

workflow.set_entry_point("generate_plan")
workflow.add_edge("generate_plan", "human_approval")
workflow.add_conditional_edges(
    "human_approval",
    check_approval,
    {
        "write_report": "write_report",
        END: END
    }
)
workflow.add_edge("write_report", END)

# Compile the graph with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

# Function to run the app
def run_research_app(research_prompt: str):
    initial_state = ResearchState(
        prompt=research_prompt,
        plan="",
        approved=False,
        final_report="",
        messages=[]
    )

    # Run until human approval node
    config = {"configurable": {"thread_id": "research_thread"}}
    app.invoke(initial_state, config=config)

    # Now, resume after human input
    print("Resuming after human input...")
    app.invoke(None, config=config)  # Resume from checkpoint

    # Get final state
    final_checkpoint = app.get_state(config)
    print("\nFinal Report:\n")
    print(final_checkpoint.values["final_report"])

# Example usage
if __name__ == "__main__":
    prompt = input("Enter your research prompt: ")
    run_research_app(prompt)
