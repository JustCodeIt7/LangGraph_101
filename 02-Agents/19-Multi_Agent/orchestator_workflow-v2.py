from typing import TypedDict, List, Annotated, Optional
import operator
from langgraph.graph import StateGraph, END, Send
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

"""
Orchestrator (planner/dispatcher) workflow with true per-section fan-out.

Key ideas:
- planner_agent creates the list of section titles.
- dispatch_agent fans out work by returning a list of Send objects (one per section).
- writer_agent handles ONE section (state provided via partial state in the Send payload).
- drafts field uses Annotated[List[str], operator.add] so LangGraph merges lists from parallel writers.
- compiler_agent may run multiple times (once per completed writer). It only assembles the final
  report when all drafts are present (len(drafts) == len(sections)), making it idempotent.
"""

# --- Shared State Definition ---


class ReportState(TypedDict, total=False):
    topic: str
    sections: List[str]
    # drafts merged via list concatenation from parallel writers
    drafts: Annotated[List[str], operator.add]
    final_report: str
    # section_title only exists inside writer executions (payload from dispatch)
    section_title: str


# --- LLM Initialization ---

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)


# --- Agents / Nodes ---


def planner_agent(state: ReportState) -> ReportState:
    """Plans the structure of the report by defining sections."""
    topic = state['topic']
    planning_prompt = f"""
    You are a report planning expert. Given the topic: "{topic}"

    Create a logical outline with 3-4 main sections for a comprehensive report.
    Return only the section titles, one per line, without numbering.
    """
    response = llm.invoke([HumanMessage(content=planning_prompt)])
    sections = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
    print(f'[planner] Planned {len(sections)} sections: {sections}')
    # Initialize drafts as empty list
    return {**state, 'sections': sections, 'drafts': []}


def dispatch_agent(state: ReportState):
    """
    Orchestrator dispatch node.
    Returns a list of Send objects, each targeting 'writer' with a partial state containing:
      - topic
      - section_title
    LangGraph will process these in parallel (subject to executor constraints).
    """
    sends = [
        Send(
            'writer',
            {
                'topic': state['topic'],
                'section_title': section_title,
                'sections': state['sections'],  # Keep sections available for compiler alignment
            },
        )
        for section_title in state['sections']
    ]
    print(f'[dispatch] Dispatching {len(sends)} writer tasks.')
    # Special key "_" instructs LangGraph to treat the list as parallel sends
    return {'_': sends}


def writer_agent(state: ReportState) -> ReportState:
    """Writes a single section (one invocation per section)."""
    topic = state['topic']
    section_title = state['section_title']

    if any(k in section_title.lower() for k in ['introduction', 'background', 'analysis', 'current']):
        focus = 'research-backed information and analysis'
        tone = 'professional and factual'
    else:
        focus = 'strategic insights and actionable recommendations'
        tone = 'forward-thinking and suggestive'

    writer_prompt = f"""
    You are an expert report writer. Write a detailed section for a report on "{topic}".

    Section: "{section_title}"

    Requirements:
    - 2-3 substantial paragraphs
    - Focus on {focus}
    - Use a {tone}
    - Do NOT repeat the section title explicitly
    """
    resp = llm.invoke([HumanMessage(content=writer_prompt)])
    content = resp.content.strip()
    print(f"[writer] Completed section: '{section_title[:50]}' (length={len(content)} chars)")
    # Return drafts as a single-item list so operator.add merges them
    return {'drafts': [content]}


def compiler_agent(state: ReportState) -> ReportState:
    """
    Compiles all drafted sections into the final report once all drafts are present.
    Idempotent: if not all drafts are ready yet, leaves final_report unchanged (or empty).
    """
    sections = state.get('sections', [])
    drafts = state.get('drafts', [])
    current_final: Optional[str] = state.get('final_report')

    if len(drafts) != len(sections) or len(sections) == 0:
        print(f'[compiler] Waiting: drafts={len(drafts)}/{len(sections)}')
        # Preserve any existing final_report if mid-way
        return {**state, 'final_report': current_final or ''}

    print(f'[compiler] Assembling final report. drafts={len(drafts)} sections={len(sections)}')
    topic = state['topic']
    final_report_content = f'# Report: {topic}\n\n'
    for i, section_title in enumerate(sections):
        if i < len(drafts):
            final_report_content += f'## {section_title}\n\n{drafts[i]}\n\n'

    return {**state, 'final_report': final_report_content}


# --- Graph Construction ---

workflow = StateGraph(ReportState)

workflow.add_node('planner', planner_agent)
workflow.add_node('dispatch', dispatch_agent)
workflow.add_node('writer', writer_agent)
workflow.add_node('compiler', compiler_agent)

workflow.set_entry_point('planner')
workflow.add_edge('planner', 'dispatch')
workflow.add_edge('dispatch', 'writer')
# Each writer execution will proceed to compiler; compiler will finalize only when complete
workflow.add_edge('writer', 'compiler')
workflow.add_edge('compiler', END)

app = workflow.compile()


# --- Execution Helper ---


def generate_report_orchestrated(topic: str) -> str:
    initial_state: ReportState = {'topic': topic, 'sections': [], 'drafts': [], 'final_report': ''}
    result = app.invoke(initial_state)
    return result['final_report']



def node1(state):
    pass
workflow.add_node("node_1", node_1)  # Add node for node_1
if __name__ == '__main__':
    topic = 'The Impact of Artificial Intelligence on Healthcare'
    print(f'[main] Generating orchestrated report for: {topic}')
    report = generate_report_orchestrated(topic)
    print(report)
