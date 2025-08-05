import asyncio
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


################################ Agent State ################################
class AgentState(TypedDict):
    """State for the agent workflow."""
    # The list of messages accumulates over time
    messages: Annotated[List[BaseMessage], add_messages]


################################ Local Tools ################################
@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression."""


################################ Agent Definition ################################
class SimpleAgent:
    def __init__(self):
        self.tools = []
        self.llm = None
        self.workflow = None
        self.mcp_contexts = []  # Store contexts for later cleanup

    async def setup(self):
        """Setup tools and compile the agent workflow."""

    async def cleanup(self):
        """Clean up any active MCP resources."""
        # Close contexts in reverse order of creation

    async def agent_node(self, state: AgentState) -> AgentState:
        """Invoke the LLM to decide the next action."""

    async def tools_node(self, state: AgentState) -> AgentState:
        """Execute the tools requested by the agent."""

    def should_continue(self, state: AgentState) -> str:
        """Determine whether to continue with another tool call or end."""

    async def ask(self, question: str):
        """Present a question to the agent and get a final answer."""


################################ Main Execution ################################
async def main():
    """Initialize and run the agent."""


if __name__ == '__main__':
    asyncio.run(main())
