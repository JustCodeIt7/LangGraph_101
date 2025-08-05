# Example 1: Basic MCP Server Usage with LangGraph
# This example shows the simplest way to use an MCP server with LangGraph

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def basic_mcp_example():
    """
    Basic example using the fetch MCP server to get web content
    """

    # Step 1: Set up the MCP client with the fetch server
    client = MultiServerMCPClient({
        'fetch': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-fetch'],
            'transport': 'stdio',
        }
    })

    # Step 2: Get available tools from the MCP server
    tools = await client.get_tools()
    print(f'Available tools: {[tool.name for tool in tools]}')

    # Step 3: Create a simple React agent with the tools
    agent = create_react_agent(
        'anthropic:claude-3-5-sonnet-latest',  # or your preferred model
        tools,
    )

    # Step 4: Use the agent to fetch web content
    response = await agent.ainvoke({
        'messages': [
            {'role': 'user', 'content': 'Fetch the content from https://httpbin.org/json and tell me what you find'}
        ]
    })

    print('Agent response:')
    print(response['messages'][-1].content)

    # Clean up
    await client.cleanup()


if __name__ == '__main__':
    asyncio.run(basic_mcp_example())
