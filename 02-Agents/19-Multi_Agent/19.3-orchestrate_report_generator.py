from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import asyncio

import os
# Define the shared state
class ReportState(TypedDict):
    topic: str
    sections: List[str]  # This will be the list of section titles
    section_drafts: Dict[str, str]  # Maps section titles to their content
    final_report: str


# Initialize the LLM
llm = ChatOpenAI(model='qwen/qwen3-30b-a3b-instruct-2507', base_url='https://openrouter.ai/api/v1',temperature=0.7, max_tokens=250, api_key=os.getenv('OPENROUTER_API_KEY'))


def planner_agent(state: ReportState) -> ReportState:
    """Plans the structure of the report by defining sections"""
    topic = state['topic']

    planning_prompt = f"""
    You are a report planning expert. Given the topic: "{topic}"
    
    Create a logical outline with 3-4 main sections for a comprehensive report.
    Return only the section titles, one per line, without numbering.
    
    Example format:
    Introduction and Background
    Current State Analysis
    Future Implications
    Conclusion and Recommendations
    """

    response = llm.invoke([HumanMessage(content=planning_prompt)])
    sections = [line.strip() for line in response.content.strip().split('\n') if line.strip()]

    return {**state, 'sections': sections, 'section_drafts': {}}


def write_section(section_title: str, topic: str) -> str:
    """Helper function to write a single section"""
    if any(keyword in section_title.lower() for keyword in ['introduction', 'background', 'analysis', 'current']):
        focus = 'research-backed information and analysis'
        tone = 'professional and factual'
    else:
        focus = 'strategic insights and actionable recommendations'
        tone = 'forward-thinking and suggestive'

    writer_prompt = f"""
    You are an expert report writer. Write a detailed section for a report on "{topic}".
    
    Section to write: "{section_title}"
    
    Requirements:
    - Write 2-3 substantial paragraphs.
    - Focus on {focus}.
    - Use a {tone}.
    - Do not include the section title itself in your response, just the content.
    """

    response = llm.invoke([HumanMessage(content=writer_prompt)])
    return response.content.strip()


def writer_coordinator(state: ReportState) -> ReportState:
    """Coordinates the writing of all sections"""
    topic = state['topic']
    sections = state['sections']
    section_drafts = {}

    # Write each section (this could be parallelized with threading/asyncio if needed)
    for section_title in sections:
        section_content = write_section(section_title, topic)
        section_drafts[section_title] = section_content

    return {**state, 'section_drafts': section_drafts}


def compiler_agent(state: ReportState) -> ReportState:
    """Compiles all drafted sections into a final report"""
    topic = state['topic']
    section_drafts = state['section_drafts']
    sections = state['sections']

    final_report_content = f'# Report: {topic}\n\n'

    # Compile sections in the planned order
    for section_title in sections:
        if section_title in section_drafts:
            final_report_content += f'## {section_title}\n\n{section_drafts[section_title]}\n\n'

    return {**state, 'final_report': final_report_content}


# Alternative: Async version for true parallel execution
async def write_section_async(section_title: str, topic: str) -> str:
    """Async helper function to write a single section"""
    return write_section(section_title, topic)


async def writer_coordinator_async(state: ReportState) -> ReportState:
    """Async coordinator for parallel section writing"""
    topic = state['topic']
    sections = state['sections']

    # Create tasks for parallel execution
    tasks = [write_section_async(section, topic) for section in sections]

    # Execute all tasks in parallel
    section_contents = await asyncio.gather(*tasks)

    # Map results back to section titles
    section_drafts = dict(zip(sections, section_contents))

    return {**state, 'section_drafts': section_drafts}


# --- Graph Construction ---

workflow = StateGraph(ReportState)

# Add nodes
workflow.add_node('planner', planner_agent)
workflow.add_node('writer_coordinator', writer_coordinator)
workflow.add_node('compiler', compiler_agent)

# Set the entry point
workflow.set_entry_point('planner')

# Add edges (simple sequential flow)
workflow.add_edge('planner', 'writer_coordinator')
workflow.add_edge('writer_coordinator', 'compiler')
workflow.add_edge('compiler', END)

# Compile the graph
app = workflow.compile()


# --- Alternative: Conditional workflow with dynamic node creation ---


def create_dynamic_workflow():
    """Creates a workflow that dynamically adds writer nodes for each section"""

    def router_agent(state: ReportState) -> str:
        """Routes to appropriate writer based on sections"""
        if not state.get('section_drafts'):
            return 'start_writing'
        elif len(state['section_drafts']) < len(state['sections']):
            return 'continue_writing'
        else:
            return 'compile'

    def section_writer_factory(section_index: int):
        """Factory to create writer functions for specific sections"""

        def section_writer(state: ReportState) -> ReportState:
            sections = state['sections']
            if section_index < len(sections):
                section_title = sections[section_index]
                topic = state['topic']

                section_content = write_section(section_title, topic)

                section_drafts = state.get('section_drafts', {}).copy()
                section_drafts[section_title] = section_content

                return {**state, 'section_drafts': section_drafts}
            return state

        return section_writer

    # This approach would require knowing sections in advance or using conditional edges
    # For simplicity, we'll stick with the coordinator approach

    return workflow


# --- Execution Functions ---


def generate_report(topic: str) -> str:
    """Generate a report using the sequential workflow."""
    initial_state = {'topic': topic, 'sections': [], 'section_drafts': {}, 'final_report': ''}
    result = app.invoke(initial_state, debug=True)
    return result['final_report']


async def generate_report_async(topic: str) -> str:
    """Generate a report with parallel section writing."""
    # Create async workflow
    async_workflow = StateGraph(ReportState)
    async_workflow.add_node('planner', planner_agent)
    async_workflow.add_node('writer_coordinator', writer_coordinator_async)
    async_workflow.add_node('compiler', compiler_agent)

    async_workflow.set_entry_point('planner')
    async_workflow.add_edge('planner', 'writer_coordinator')
    async_workflow.add_edge('writer_coordinator', 'compiler')
    async_workflow.add_edge('compiler', END)

    async_app = async_workflow.compile()

    initial_state = {'topic': topic, 'sections': [], 'section_drafts': {}, 'final_report': ''}
    result = await async_app.ainvoke(initial_state)
    return result['final_report']


if __name__ == '__main__':
    topic = 'The Impact of Artificial Intelligence on Healthcare'

    print(f'Generating report on: {topic}...')
    report = generate_report(topic)
    print(report)

    # For async version:
    # import asyncio
    # report_async = asyncio.run(generate_report_async(topic))
    # print(report_async)
