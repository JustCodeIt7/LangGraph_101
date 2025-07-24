
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph_mcp import McpServer
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# 1. Define a tool
from langchain_core.tools import tool

@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for a real web search tool
    return ["The weather in SF is sunny!"]

# 2. Define the state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# 3. Define the agent
class Agent:
    def __init__(self, model, tools):
        self.model = model.bind_tools(tools)

    def __call__(self, state: AgentState):
        messages = state['messages']
        response = self.model.invoke(messages)
        return {"messages": [response]}

# 4. Define the graph
def create_graph():
    llm = ChatOpenAI(model="gpt-4o")
    agent = Agent(llm, [search])
    tool_node = ToolNode([search])

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())

# 5. Run the MCP server
async def main():
    """
    Main function to run the MCP server.
    
    To run this example:
    1. Make sure you have the required packages installed: `uv pip install langgraph langchain_openai langgraph_mcp`
    2. Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`
    3. Run the script: `python 02_agent_with_tools.py`
    4. In a separate terminal, you can use curl to interact with the agent:
       `curl -X POST -H "Content-Type: application/json" -d '{"input": {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}}' http://localhost:8000/invoke`
    """
    app = create_graph()
    server = McpServer(app)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
