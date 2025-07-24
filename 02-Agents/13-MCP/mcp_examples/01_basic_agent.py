
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph_mcp import McpServer
from typing import TypedDict, Annotated
import operator

# 1. Define the state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 2. Define the agent
def my_agent(state):
    messages = state['messages']
    # Agent logic goes here
    # For this basic example, it just responds with a fixed message
    response = "Hello from the basic agent!"
    return {"messages": [response]}

# 3. Define the graph
def create_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", my_agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=MemorySaver())

# 4. Run the MCP server
async def main():
    """
    Main function to run the MCP server.
    
    To run this example:
    1. Make sure you have the required packages installed: `uv pip install langgraph langchain_openai langgraph_mcp`
    2. Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`
    3. Run the script: `python 01_basic_agent.py`
    4. In a separate terminal, you can use curl to interact with the agent:
       `curl -X POST -H "Content-Type: application/json" -d '{"input": {"messages": ["Hello, agent!"]}}' http://localhost:8000/invoke`
    """
    app = create_graph()
    server = McpServer(app)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
