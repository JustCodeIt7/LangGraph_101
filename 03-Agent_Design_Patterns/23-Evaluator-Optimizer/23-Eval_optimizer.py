import os
from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from rich import print
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

################################ Model Initialization ################################

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL_NAME = 'llama3.2'

# Initialize the LLM client with local Ollama settings
model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,  # Lower temperature for more consistent evaluations
    base_url=OLLAMA_BASE_URL,
)

################################ Data Schema Definitions ################################


# Define the shared state for the agentic workflow
class AgentState(TypedDict):
    topic: str  # Initial user prompt
    joke: str  # Current iteration of generated content
    feedback: str  # Critique from the evaluator node
    score: int  # Numeric quality metric
    iteration: int  # Safety counter to prevent infinite loops


# Enforce structured output from the evaluator LLM
class Evaluation(BaseModel):
    score: int = Field(description='Score from 1 to 10, where 10 is the best.')
    feedback: str = Field(description='Specific feedback on how to improve the joke.')


################################ Graph Node Logic ################################


def generator_node(state: AgentState):
    topic = state['topic']
    feedback = state.get('feedback')
    iteration = state.get('iteration', 0)

    # Adjust prompt based on whether this is a revision or a fresh start
    if feedback:
        prompt = f'The user wants a joke about {topic}. \n\nPrevious attempt: {state["joke"]}\nFeedback: {feedback}\n\nGenerate a BETTER joke.'
    else:
        prompt = f'Generate a funny joke about {topic}.'

    response = model.invoke([HumanMessage(content=prompt)])

    print(f'\n--- GENERATOR (Iteration {iteration + 1}) ---\nJoke: {response.content}')

    return {'joke': response.content, 'iteration': iteration + 1}


def evaluator_node(state: AgentState):
    joke = state['joke']

    prompt = f"""Evaluate this joke for humor and clarity:
    
    Joke: {joke}
    
    Rate it 1-10 and provide feedback."""

    # Bind the Pydantic schema to the model for reliable JSON parsing
    evaluator = model.with_structured_output(Evaluation)
    result = evaluator.invoke([HumanMessage(content=prompt)])

    print(f'--- EVALUATOR ---\nScore: {result.score}/10\nFeedback: {result.feedback}')

    return {'score': result.score, 'feedback': result.feedback}


################################ Workflow Control Flow ################################


def should_continue(state: AgentState):
    # Terminate the loop if the quality threshold is met
    if state['score'] >= 8:
        print('\n--- DECISION: GOOD ENOUGH ---')
        return END

    # Terminate the loop if the maximum retry limit is reached
    if state['iteration'] >= 3:
        print('\n--- DECISION: MAX ITERATIONS REACHED ---')
        return END

    print('\n--- DECISION: TRY AGAIN ---')
    return 'generator'


################################ Graph Assembly & Execution ################################

builder = StateGraph(AgentState)

# Register the functional components of the agent
builder.add_node('generator', generator_node)
builder.add_node('evaluator', evaluator_node)

# Define the execution path and loop logic
builder.set_entry_point('generator')
builder.add_edge('generator', 'evaluator')
builder.add_conditional_edges(
    'evaluator',
    should_continue,
    ['generator', END],  # Map return values to graph destinations
)

graph = builder.compile()

if __name__ == '__main__':
    # Define the starting parameters for the workflow
    initial_state = {'topic': 'Quantum Physics', 'iteration': 0}

    print('Starting Evaluator-Optimizer Agent...')
    result = graph.invoke(initial_state)

    print('\nFinal Result:')
    print(result['joke'])
