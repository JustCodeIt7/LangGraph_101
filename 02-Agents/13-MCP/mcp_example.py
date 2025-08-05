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

# Set up server parameters (update path to your math_server.py)
server_params = StdioServerParameters(
    command='python',
    args=['math_server.py'],  # might need to adjust this path
    transport='stdio',
    definition='Math tools for calculations',
    env=None,
)


# Define custom tools
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate what percentage of a value is.

    Args:
        value: The base value
        percentage: The percentage to calculate (e.g., 20 for 20%)
    """
    return (value * percentage) / 100


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # Initialize connection

            # Load tools from the MCP server
            mcp_tools = await load_mcp_tools(session)
            print('MCP tools:', [tool.name for tool in mcp_tools])  # Should show ['add', 'multiply']

            # Define additional custom tools
            custom_tools = [
                get_current_time,
                calculate_percentage,
                # DuckDuckGoSearchRun(),  # Uncomment if you want web search
            ]

            # Combine MCP tools with custom tools
            all_tools = mcp_tools + custom_tools
            print('All available tools:', [tool.name for tool in all_tools])

            # Set up the LLM and agent
            llm = ChatOpenAI(model='gpt-4o', temperature=0)  # Replace with your key
            # llm = ChatOllama(model='llama3.2', temperature=0)  # Or use Ollama model
            agent = create_react_agent(llm, all_tools)

            # Test with multiple tools
            response = await agent.ainvoke({
                'messages': [
                    {
                        'role': 'user',
                        'content': "What's the current time? Also calculate (3 + 5) * 12 and then find 15% of that result.",
                    }
                ]
            })
            print('Agent response:', response['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())
