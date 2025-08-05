import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI  # Or use another LLM like ChatGoogleGenerativeAI

# Set up server parameters (update path to your math_server.py)
server_params = StdioServerParameters(
    command='python',
    args=['math_server.py'],  # Replace with absolute path
    env=None,
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # Initialize connection

            # Load tools from the MCP server
            tools = await load_mcp_tools(session)
            print('Available tools:', [tool.name for tool in tools])  # Should show ['add', 'multiply']

            # Set up the LLM and agent
            llm = ChatOpenAI(model='gpt-4o', temperature=0)  # Replace with your key
            agent = create_react_agent(llm, tools)

            # Invoke the agent with a query
            response = await agent.ainvoke({'messages': [{'role': 'user', 'content': "What's (3 + 5) * 12?"}]})
            print('Agent response:', response['messages'][-1].content)  # Example output: "The result is 96."


if __name__ == '__main__':
    asyncio.run(main())
