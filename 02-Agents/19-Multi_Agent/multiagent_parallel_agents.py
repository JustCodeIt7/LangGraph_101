import os
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings


# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key


# Define the state for our graph
class AgentState(TypedDict):
    topic: str
    questions: List[str]
    jokes: List[str]
    related_topics: List[str]


# Initialize the language model
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


# Helper function to create a generation chain
def create_generation_chain(prompt_template_str: str):
    """Creates a chain for generating content based on a topic."""
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    parser = JsonOutputParser()
    return prompt | llm | parser


# Create chains for each agent
question_chain = create_generation_chain(
    'Generate 5 questions about the following topic: {topic}. Output as a JSON list of strings.'
)
joke_chain = create_generation_chain(
    'Generate 3 jokes about the following topic: {topic}. Output as a JSON list of strings.'
)
related_topics_chain = create_generation_chain(
    'Generate 4 related topics for the following topic: {topic}. Output as a JSON list of strings.'
)


# Define the nodes for the graph
def question_agent(state: AgentState):
    """Generates questions based on the topic."""
    print('---GENERATING QUESTIONS---')
    questions = question_chain.invoke({'topic': state['topic']})
    return {'questions': questions}


def joke_agent(state: AgentState):
    """Generates jokes based on the topic."""
    print('---GENERATING JOKES---')
    jokes = joke_chain.invoke({'topic': state['topic']})
    return {'jokes': jokes}


def related_topics_agent(state: AgentState):
    """Generates related topics based on the topic."""
    print('---GENERATING RELATED TOPICS---')
    related_topics = related_topics_chain.invoke({'topic': state['topic']})
    return {'related_topics': related_topics}


# Build the graph
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

# Compile the graph
app = workflow.compile()

# Run the graph with a topic
if __name__ == '__main__':
    topic = 'Artificial Intelligence'
    initial_state = {'topic': topic, 'questions': [], 'jokes': [], 'related_topics': []}

    final_state = app.invoke(initial_state)

    print('\n---FINAL RESULTS---')
    print(f'Topic: {final_state["topic"]}')
    print('\nQuestions:')
    for q in final_state['questions']:
        print(f'- {q}')

    print('\nJokes:')
    for j in final_state['jokes']:
        print(f'- {j}')

    print('\nRelated Topics:')
    for t in final_state['related_topics']:
        print(f'- {t}')
