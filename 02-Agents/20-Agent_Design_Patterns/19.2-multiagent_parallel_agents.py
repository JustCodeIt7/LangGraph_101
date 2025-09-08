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


# Initialize different language models for each agent
llm_questions = ChatOpenAI(model='gpt-4o', temperature=0.1)  # Most capable for thoughtful questions
llm_jokes = ChatOpenAI(model='gpt-4o-mini', temperature=0.8)  # Creative for humor
llm_related = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.3)  # Balanced for topic connections


# Helper function to create a generation chain
def create_generation_chain(llm, prompt_template_str: str):
    """Creates a chain for generating content based on a topic."""
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    parser = JsonOutputParser()
    return prompt | llm | parser


# Create chains for each agent, passing the specific LLM
question_chain = create_generation_chain(
    llm_questions,
    'Generate 5 questions about the following topic: {topic}. Output as a JSON object with a single key "output" that contains a list of strings.',
)
joke_chain = create_generation_chain(
    llm_jokes,
    'Generate 3 jokes about the following topic: {topic}. Output as a JSON object with a single key "output" that contains a list of strings.',
)
related_topics_chain = create_generation_chain(
    llm_related,
    'Generate 4 related topics for the following topic: {topic}. Output as a JSON object with a single key "output" that contains a list of strings.',
)


# Define the nodes for the graph
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
    print('🚀 LangGraph Multi-Agent Parallel Processing with Different LLMs')
    print('=' * 65)
    print('🤖 Question Agent: GPT-4o (Most capable for thoughtful questions)')
    print('😂 Joke Agent: GPT-4o-mini (Creative and cost-effective for humor)')
    print('🔗 Related Topics Agent: GPT-3.5-turbo (Balanced for connections)')
    print('=' * 65)

    # Test with multiple topics
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
