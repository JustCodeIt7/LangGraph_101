import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

client = MultiServerMCPClient({
    'math': {
        'command': 'python',
        'args': ['math_server.py'],  # Adjust path as needed
        'transport': 'stdio',
    },
    'weather': {
        'url': 'http://localhost:8000/mcp',
        'transport': 'streamable_http',  # Changed from 'sse' to match docs/examples
    },
})


async def main():
    tools = client.get_tools()  # Remove 'await' here—it's likely sync in your env
    print('Available tools:', [tool.name for tool in tools])  # E.g., ['add', 'multiply', 'subtract', 'get_weather']

    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    agent = create_react_agent(llm, tools)

    response = await agent.ainvoke({
        'messages': [{'role': 'user', 'content': "What's the weather in NYC, then add 5 to 10?"}]
    })
    print('Agent response:', response['messages'][-1].content)  # Agent handles both tools in sequence.


if __name__ == '__main__':
    asyncio.run(main())
