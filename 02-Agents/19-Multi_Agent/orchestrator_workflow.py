import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, List

# Load environment variables
load_dotenv()


# 1. Define the State
class ReportState(TypedDict):
    topic: str
    plan: List[str]
    drafts: List[str]
    report: str


# 2. Define the Nodes (Agents)
def create_llm(model_name='gpt-4o', temperature=0.1):
    """Factory function to create an LLM instance."""
    return ChatOpenAI(model=model_name, temperature=temperature)


def planner_node(state: ReportState):
    """Generates a plan of sections for the report."""
    topic = state['topic']
    prompt = (
        f"You are a master planner. Create a concise plan for a report on the topic: '{topic}'. "
        'Just output a list of 3-5 section titles, each on a new line. Do not add any other text.'
    )
    llm = create_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    plan = [line.strip() for line in response.content.split('\n') if line.strip()]
    print(f'Generated Plan: {plan}')
    return {'plan': plan}


def worker_node(section_title: str, state: ReportState):
    """Writes a draft for a specific section of the report."""
    topic = state['topic']
    prompt = (
        f"You are an expert writer. Write a detailed draft for the section titled '{section_title}' "
        f"as part of a larger report on '{topic}'. Focus only on this section's content."
    )
    llm = create_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f'Wrote draft for section: {section_title}')
    return {'drafts': [(section_title, response.content)]}  # Return tuple of (title, content)


def compiler_node(state: ReportState):
    """Compiles the individual drafts into a final report."""
    topic = state['topic']
    drafts = state['drafts']

    report_str = f'<h1>Report on: {topic}</h1>\n\n'
    for section_title, draft_content in drafts:
        report_str += f'<h2>{section_title}</h2>\n'
        report_str += f'<p>{draft_content}</p>\n\n'

    print('Compiled final report.')
    return {'report': report_str}


# 3. Build the Graph
workflow = StateGraph(ReportState)

# Add nodes
workflow.add_node('planner', planner_node)
workflow.add_node('compiler', compiler_node)


# The worker node is special, it will be dynamically invoked for each plan item.
workflow.add_node('worker', worker_node)

# Set up the edges
workflow.set_entry_point('planner')
workflow.add_edge('planner', 'worker', map='plan')
workflow.add_edge('worker', 'compiler')
workflow.add_edge('compiler', END)

# Compile the graph
app = workflow.compile()

# 4. Run the Workflow
if __name__ == '__main__':
    topic = 'the future of artificial intelligence'
    inputs = {'topic': topic, 'drafts': []}  # Initial state

    final_report = None
    for output in app.stream(inputs, stream_mode='values'):
        if 'report' in output:
            final_report = output['report']

    print('\n' + '=' * 30)
    print('          FINAL REPORT')
    print('=' * 30 + '\n')
    print(final_report)
