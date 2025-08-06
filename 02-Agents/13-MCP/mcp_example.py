import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI  # Or use another LLM like ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import datetime
from langchain_mcp_adapters.client import MultiServerMCPClient

################################ Configure MCP Server ################################
# Define how to start the external MCP server process
# server_params = StdioServerParameters(
#     command='python',
#     args=['math_server.py'],  # Path to the server script
#     env=None,
# )
# Example for installing the MCP server fetch tool: pip install mcp-server-fetch
server_params = StdioServerParameters(
    command='python',
    args=['-m', 'mcp_server_fetch'],
    env=None,
)
################################ Define Custom Local Tools ################################
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate what percentage of a value is.
    Args:
        value: The base value
        percentage: The percentage to calculate (e.g., 20 for 20%)
    """
    return (value * percentage) / 100
################################ Main Agent Logic ################################
async def main():
    # Start the MCP server as a subprocess
    async with stdio_client(server_params) as (read, write):
        # Establish a client session with the running server
        async with ClientSession(read, write) as session:
            await session.initialize()  # Finalize the connection and handshake
            # Load tools exposed by the remote MCP server
            mcp_tools = await load_mcp_tools(session)
            print('MCP tools:', [tool.name for tool in mcp_tools])
            # Define additional tools available in this local script
            custom_tools = [
                get_current_time,
                calculate_percentage,
            ]
            # Combine remote and local tools into a single list for the agent
            all_tools = mcp_tools + custom_tools
            print('All available tools:', [tool.name for tool in all_tools])
            # Configure the Large Language Model
            llm = ChatOpenAI(model='gpt-4o', temperature=0)
            # llm = ChatOllama(model='llama3.2', temperature=0)  # Or use a local Ollama model
            # Create a ReAct agent that can use the combined toolset
            agent = create_react_agent(llm, all_tools)
            # Send a complex, multi-tool query to the agent
            # response = await agent.ainvoke({
            #     'messages': [
            #         {
            #             'role': 'user',
            #             'content': "What's the current time? Also calculate (3 + 5) * 12 and then find 15% of that result.",
            #         }
            #     ]
            # })
            # Example query for the web fetch tool
            response = await agent.ainvoke({
                'messages': [
                    {
                        'role': 'user',
                        'content': 'fetch the website https://langchain-ai.github.io/langgraph/agents/mcp/ and summarize it',
                    }
                ]
            })
            # Print the agent's final response
            print('Agent response:', response['messages'][-1].content)
################################ Run the Application ################################
if __name__ == '__main__':
    asyncio.run(main())