
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph_mcp import McpServer
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define tools
from langchain_core.tools import tool

@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for a real web search tool
    return ["The weather in SF is sunny!"]

# 2. Define the state
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    sender: str

# 3. Define the agents
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

# 4. Define the multi-agent graph
def create_graph():
    researcher_model = ChatOpenAI(model="gpt-4o")
    research_agent = Agent(researcher_model, [search], "You are a research assistant.")
    
    chart_generator_model = ChatOpenAI(model="gpt-4o")
    chart_generator_agent = Agent(chart_generator_model, [], "You are a chart generator.")

    builder = StateGraph(AgentState)
    builder.add_node("researcher", researcher_agent.graph)
    builder.add_node("chart_generator", chart_generator_agent.graph)
    
    def router(state):
        return state["sender"]

    builder.add_conditional_edges(
        "__start__",
        router,
        {
            "researcher": "researcher",
            "chart_generator": "chart_generator",
        },
    )
    builder.add_edge("researcher", END)
    builder.add_edge("chart_generator", END)

    return builder.compile(checkpointer=MemorySaver())

# 5. Run the MCP server
async def main():
    """
    Main function to run the MCP server.
    
    To run this example:
    1. Make sure you have the required packages installed: `uv pip install langgraph langchain_openai`
    2. Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`
    3. Run the script: `python 03_multi_agent_collaboration.py`
    4. In a separate terminal, you can use curl to interact with the researcher agent:
       `curl -X POST -H "Content-Type: application/json" -d '{"input": {"messages": [{"role": "user", "content": "What is the weather in SF?"}], "sender": "researcher"}}' http://localhost:8000/invoke`
    5. Or interact with the chart generator agent:
       `curl -X POST -H "Content-Type: application/json" -d '{"input": {"messages": [{"role": "user", "content": "Generate a chart for the weather in SF."}], "sender": "chart_generator"}}' http://localhost:8000/invoke`
    """
    app = create_graph()
    server = McpServer(app)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
