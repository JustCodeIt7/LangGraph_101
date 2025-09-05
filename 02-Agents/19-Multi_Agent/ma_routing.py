import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List
import operator

load_dotenv()


# Define the state for our graph
class AgentState(TypedDict):
    task: str
    result: str
    history: Annotated[list, operator.add]


# Define the router logic
class RouteQuery(BaseModel):
    """Route the query to the appropriate specialist."""

    destination: str = Field(
        description="The destination for the query, either 'creative_writer' or 'coder'.",
        enum=['creative_writer', 'coder'],
    )


def create_llm(model_name='gpt-4o'):
    """Factory function to create an LLM instance."""
    return ChatOpenAI(model=model_name, temperature=0)


def get_router(llm):
    """Returns a function that routes queries based on their content."""
    structured_llm = llm.with_structured_output(RouteQuery, method='function_calling')

    def router(state: AgentState):
        query = state['task']
        route = structured_llm.invoke([HumanMessage(content=f"Route the following query: '{query}'")])
        return route.destination

    return router


# Define the specialist nodes
def creative_writer_node(state: AgentState):
    """Invokes the creative writer LLM to generate a creative response."""
    query = state['task']
    prompt = f"Write a creative piece, like a short poem or story, based on the following topic: '{query}'. Be imaginative and concise."
    llm = create_llm()
    result = llm.invoke([HumanMessage(content=prompt)])
    return {'result': result.content, 'history': [('creative_writer', result.content)]}


def coder_node(state: AgentState):
    """Invokes the coder LLM to generate a code snippet."""
    query = state['task']
    prompt = f'Generate a Python code snippet to {query}. Provide only the code, with a brief explanation if necessary.'
    llm = create_llm()
    result = llm.invoke([HumanMessage(content=prompt)])
    return {'result': result.content, 'history': [('coder', result.content)]}


# Build the graph
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node('creative_writer', creative_writer_node)
    workflow.add_node('coder', coder_node)

    # Set up the router
    llm = create_llm()
    router_func = get_router(llm)
    workflow.add_conditional_edges(
        '__start__',
        router_func,
        {
            'creative_writer': 'creative_writer',
            'coder': 'coder',
        },
    )

    # Define the end
    workflow.add_edge('creative_writer', END)
    workflow.add_edge('coder', END)

    # Compile the graph
    return workflow.compile()


# Main execution block
if __name__ == '__main__':
    app = build_graph()

    # Example 1: Creative Query
    creative_query = 'the feeling of a sunrise'
    creative_inputs = {'task': creative_query, 'history': []}
    creative_output = app.invoke(creative_inputs)
    print('--- Creative Query ---')
    print(f'Query: {creative_query}')
    print(f'Agent Used: {creative_output["history"][0][0]}')
    print(f"""Result:
{creative_output['result']}""")
    print('\n' + '=' * 20 + '\n')

    # Example 2: Coding Query
    code_query = 'create a function that calculates the factorial of a number'
    code_inputs = {'task': code_query, 'history': []}
    code_output = app.invoke(code_inputs)
    print('--- Coding Query ---')
    print(f'Query: {code_query}')
    print(f'Agent Used: {code_output["history"][0][0]}')
    print(f"""Result:
{code_output['result']}""")
