import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools  # Import this for explicit loading
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

client = MultiServerMCPClient({
    'math': {
        'command': 'python',
        'args': ['math_server.py'],  # Use FULL absolute path
        'transport': 'stdio',
    },
    'weather': {
        'url': 'http://localhost:8000/mcp',
        'transport': 'streamable_http',  # Recommended in docs for reliability
    },
})


async def main():
    all_tools = []  # Collect tools from all servers

    # Explicitly load from math server
    async with client.sessions('math') as math_session:
        print('Math session initialized:', math_session.initialized)  # Debug: Check if session started
        math_tools = await load_mcp_tools(math_session)
        all_tools.extend(math_tools)
        print('Math tools loaded:', [tool.name for tool in math_tools])

    # Explicitly load from weather server
    async with client.sessions('weather') as weather_session:
        print('Weather session initialized:', weather_session.initialized)
        weather_tools = await load_mcp_tools(weather_session)
        all_tools.extend(weather_tools)
        print('Weather tools loaded:', [tool.name for tool in weather_tools])

    print('Available tools:', [tool.name for tool in all_tools])  # Should now list tools like ['add', 'get_weather']

    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    agent = create_react_agent(llm, all_tools)

    response = await agent.ainvoke({
        'messages': [{'role': 'user', 'content': "What's the weather in NYC, then add 5 to 10?"}]
    })
    print('Agent response:', response['messages'][-1].content)  # Should now use tools properly


if __name__ == '__main__':
    asyncio.run(main())
