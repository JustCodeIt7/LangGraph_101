#%% md
# LangGraph 101: Multi-Agent Parallel Agents (Teaching Notebook in Percent-Format)

# This file is a Jupyter-compatible percent script: each cell is marked with `#%%` (code) or `#%% md` (markdown).
# You can open it as a notebook in:
# - Jupyter: jupytext, or run cells in e.g., VS Code/Jupyter extension
# - JetBrains IDEs (PyCharm/IDEA with Python plugin): Run cells and view markdown
# - VS Code with the Python extension: It will treat `#%%` blocks as notebook cells
#
# Purpose:
# - Teach how to build a simple multi-agent system in LangGraph where three agents run in parallel:
#   - Questions about a topic
#   - Jokes about the topic
#   - Related topics
# - Each agent uses a different OpenAI model to highlight trade-offs (capability, creativity, cost).
#
# If you prefer a pure .py script without explanations, see the sibling file:
# 19.2-multiagent_parallel_agents.py
#
# ---
#
# ## Learning Goals
# - Understand LangGraph state and nodes
# - Build a parallel workflow by setting multiple entry points
# - Structure LLM outputs with JSON parsing for robustness
# - Swap models to meet budget/latency/creativity goals
#
# ---


#%% md
# LangGraph 101: Multi-Agent Parallel Agents (Teaching Notebook in Percent-Format)

# This file is a Jupyter-compatible percent script: each cell is marked with `#%%` (code) or `#%% md` (markdown).
# You can open it as a notebook in:
# - Jupyter: jupytext, or run cells in e.g., VS Code/Jupyter extension
# - JetBrains IDEs (PyCharm/IDEA with Python plugin): Run cells and view markdown
# - VS Code with the Python extension: It will treat `#%%` blocks as notebook cells
#
# Purpose:
# - Teach how to build a simple multi-agent system in LangGraph where three agents run in parallel:
#   - Questions about a topic
#   - Jokes about the topic
#   - Related topics
# - Each agent uses a different OpenAI model to highlight trade-offs (capability, creativity, cost).
#
# If you prefer a pure .py script without explanations, see the sibling file:
# 19.2-multiagent_parallel_agents.py
#
# ---
#
# ## Learning Goals
# - Understand LangGraph state and nodes
# - Build a parallel workflow by setting multiple entry points
# - Structure LLM outputs with JSON parsing for robustness
# - Swap models to meet budget/latency/creativity goals
#
# ---

#%% md
## 0. Prerequisites

# - Python 3.10+ recommended.
# - An OpenAI API key set in your environment or stored in a `.env` file (see below).
# - Dependencies (LangChain, LangGraph, python-dotenv, etc.).
#
# Optional: If OpenAI is unavailable, you can adapt to use Ollama locally.

#%% md
## 1. Install or verify dependencies

# You might not need to install anything if you're running inside this repository.
# If you do, uncomment the pip command below (prefer using your project's requirements).

#%%
# If running in a fresh environment, uncomment to install:
# %pip install langchain langchain-openai langgraph python-dotenv langchain-ollama
print('If packages are missing, uncomment the %pip line above and re-run this cell.')

#%% md
## 2. Imports

# We import the building blocks from LangChain and LangGraph,
# and dotenv for configuration. JsonOutputParser helps parse structured JSON.

#%%
import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# Optional: local models with Ollama (unused by default, but shown for reference)
from langchain_ollama import ChatOllama, OllamaEmbeddings

print('Imports loaded.')

#%% md
## 3. Environment setup (.env and keys)

# We load environment variables from a `.env` file (if present) and set keys in os.environ.
# Required: OPENAI_API_KEY
# Optional: LANGCHAIN_API_KEY if you use LangSmith/LangChain tracing.
# Tip: Never hard-code credentials in notebooks. Use `.env` or environment vars.

#%%
# Load environment variables from .env (if present)
load_dotenv()

# Export keys to environment for libraries to detect
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key

# Simple check and friendly message
if not os.environ.get('OPENAI_API_KEY'):
    print('Warning: OPENAI_API_KEY is not set. You will need it to run OpenAI models.')
else:
    print('OPENAI_API_KEY detected. Ready to run OpenAI models.')

#%% md
## 4. Define the State schema for our graph

# LangGraph passes state between nodes. Here we define a TypedDict describing the shape.
# Fields:
# - topic: input topic string
# - questions: list of question strings
# - jokes: list of joke strings
# - related_topics: list of related topic strings

#%%
class AgentState(TypedDict):
    topic: str
    questions: List[str]
    jokes: List[str]
    related_topics: List[str]

print('AgentState schema defined.')

#%% md
## 5. Choose and initialize models for each agent

# Three OpenAI chat models with different temperatures to emphasize their roles:
# - gpt-4o (low temperature) for high-quality questions
# - gpt-4o-mini (higher temperature) for creative jokes
# - gpt-3.5-turbo (moderate temperature) for related topics
# Tip: You can swap models to fit budget and latency.
# Optional (Local): You could switch to ChatOllama models if desired.

#%%
llm_questions = ChatOpenAI(model='gpt-4o', temperature=0.1)  # Questions
llm_jokes = ChatOpenAI(model='gpt-4o-mini', temperature=0.8)  # Jokes
llm_related = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.3)  # Related topics

print('Models initialized (OpenAI).')

#%% md
## 6. Create a reusable generation chain helper

# A helper to build Prompt -> LLM -> JSON parser pipelines.
# Why JSON? It makes downstream handling robust and predictable.

#%%
def create_generation_chain(llm, prompt_template_str: str):
    """Creates a chain for generating content based on a topic."""
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    parser = JsonOutputParser()
    return prompt | llm | parser

# Build three chains, one per agent role
question_chain = create_generation_chain(
    llm_questions,
    'Generate 5 questions about the following topic: {topic}. '
    'Output as a JSON object with a single key "output" that contains a list of strings.',
)

joke_chain = create_generation_chain(
    llm_jokes,
    'Generate 3 jokes about the following topic: {topic}. '
    'Output as a JSON object with a single key "output" that contains a list of strings.',
)

related_topics_chain = create_generation_chain(
    llm_related,
    'Generate 4 related topics for the following topic: {topic}. '
    'Output as a JSON object with a single key "output" that contains a list of strings.',
)

print('Chains created.')

#%% md
## 7. Define agent node functions

# Each agent:
# 1) Receives the current state (uses only `topic`)
# 2) Invokes its chain
# 3) Returns a partial state update under its key
# Simple try/except keeps demos robust.

#%%
def question_agent(state: AgentState):
    """Generates questions based on the topic using GPT-4o."""
    print('---GENERATING QUESTIONS (GPT-4o)---')
    try:
        result = question_chain.invoke({'topic': state['topic']})
        return {'questions': result['output']}
    except Exception as e:
        print(f'Error in question_agent: {e}')
        return {'questions': [f'Error generating questions: {str(e)}']}


def joke_agent(state: AgentState):
    """Generates jokes based on the topic using GPT-4o-mini."""
    print('---GENERATING JOKES (GPT-4o-mini)---')
    try:
        result = joke_chain.invoke({'topic': state['topic']})
        return {'jokes': result['output']}
    except Exception as e:
        print(f'Error in joke_agent: {e}')
        return {'jokes': [f'Error generating jokes: {str(e)}']}


def related_topics_agent(state: AgentState):
    """Generates related topics based on the topic using GPT-3.5-turbo."""
    print('---GENERATING RELATED TOPICS (GPT-3.5-turbo)---')
    try:
        result = related_topics_chain.invoke({'topic': state['topic']})
        return {'related_topics': result['output']}
    except Exception as e:
        print(f'Error in related_topics_agent: {e}')
        return {'related_topics': [f'Error generating related topics: {str(e)}']}

print('Agent functions defined.')

#%% md
## 8. Build the LangGraph workflow

# We construct a StateGraph, add nodes, and wire edges. We start all three agents in parallel
# by setting multiple entry points; each one goes to END.

#%%
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node('question_agent', question_agent)
workflow.add_node('joke_agent', joke_agent)
workflow.add_node('related_topics_agent', related_topics_agent)

# The entry point is the topic, which then triggers the parallel agents
workflow.set_entry_point('question_agent')
workflow.set_entry_point('joke_agent')
workflow.set_entry_point('related_topics_agent')

# All parallel agents lead to the end
workflow.add_edge('question_agent', END)
workflow.add_edge('joke_agent', END)
workflow.add_edge('related_topics_agent', END)

# Compile the graph into an app we can invoke
app = workflow.compile()

print('Workflow compiled.')

#%% md
## 9. Run the app on a single topic (quick test)

# A small smoke test to confirm everything is wired correctly before a full demo.

#%%
test_state = {
    'topic': 'Artificial Intelligence',
    'questions': [],
    'jokes': [],
    'related_topics': []
}

try:
    demo_result = app.invoke(test_state)
    print('Quick test result keys:', list(demo_result.keys()))
    print('Questions count:', len(demo_result.get('questions', [])))
    print('Jokes count:', len(demo_result.get('jokes', [])))
    print('Related topics count:', len(demo_result.get('related_topics', [])))
except Exception as e:
    print('Quick test failed:', e)

#%% md
## 10. Full demo over multiple topics

# Iterate over a few topics and print the outputs in a readable format.

#%%
print('🚀 LangGraph Multi-Agent Parallel Processing with Different LLMs')
print('=' * 65)
print('🤖 Question Agent: GPT-4o (Most capable for thoughtful questions)')
print('😂 Joke Agent: GPT-4o-mini (Creative and cost-effective for humor)')
print('🔗 Related Topics Agent: GPT-3.5-turbo (Balanced for connections)')
print('=' * 65)

test_topics = ['Artificial Intelligence', 'Climate Change', 'Space Exploration']

for topic in test_topics:
    print(f'\n🎯 Processing Topic: {topic}')
    print('⚡ Running parallel agents with different LLMs...')

    initial_state = {'topic': topic, 'questions': [], 'jokes': [], 'related_topics': []}

    try:
        final_state = app.invoke(initial_state)

        print(f'\n📊 RESULTS FOR: {final_state["topic"].upper()}')
        print('-' * 50)

        print(f'\n🤔 QUESTIONS (Generated by GPT-4o):')
        for i, q in enumerate(final_state['questions'], 1):
            print(f'  {i}. {q}')

        print(f'\n😂 JOKES (Generated by GPT-4o-mini):')
        for i, j in enumerate(final_state['jokes'], 1):
            print(f'  {i}. {j}')

        print(f'\n🔗 RELATED TOPICS (Generated by GPT-3.5-turbo):')
        for i, t in enumerate(final_state['related_topics'], 1):
            print(f'  {i}. {t}')

        print('\n' + '=' * 65)

    except Exception as e:
        print(f'❌ Error processing topic "{topic}": {str(e)}')

print('\n✅ Demo completed! Each agent used a different LLM model.')

#%% md
## 11. Exercises for Students

# 1. Change the number of generated items (e.g., 10 questions) and compare results.
# 2. Swap models (e.g., use gpt-4o-mini for all three) and compare quality/cost/latency.
# 3. Add a fourth agent that summarizes the combined outputs into a short lesson plan.
# 4. Replace OpenAI with a local Ollama model and observe differences.
# 5. Add input widgets (e.g., ipywidgets) to let users choose a topic interactively.

#%% md
## 12. Troubleshooting

# - If you see authentication errors, ensure OPENAI_API_KEY is set.
# - If JSON parsing fails, the model may have deviated from the format. Strengthen instructions
#   or consider StructuredOutputParser and schemas.
# - If latency is high, reduce the number of generated items or choose lighter models.
# - For more deterministic behavior, lower temperature. True determinism is not guaranteed.
