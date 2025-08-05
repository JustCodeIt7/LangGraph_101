"""
Examples of different ways to add tools to a LangGraph agent
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Type
from pydantic import BaseModel, Field
import requests
import json


# Method 1: Using @tool decorator (Simplest)
@tool
def weather_lookup(city: str) -> str:
    """Get current weather for a city."""
    # This is a mock implementation - replace with real API
    return f"The weather in {city} is sunny and 72°F"

@tool
def file_writer(filename: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


# Method 2: Creating a custom tool class
class CurrencyConverterInput(BaseModel):
    """Input for currency converter tool."""
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="Source currency code (e.g., USD)")
    to_currency: str = Field(description="Target currency code (e.g., EUR)")

class CurrencyConverterTool(BaseTool):
    name: str = "currency_converter"
    description: str = "Convert currency from one type to another"
    args_schema: Type[BaseModel] = CurrencyConverterInput

    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        # Mock conversion - replace with real API
        rate = 0.85 if from_currency == "USD" and to_currency == "EUR" else 1.0
        result = amount * rate
        return f"{amount} {from_currency} = {result:.2f} {to_currency}"

    async def _arun(self, amount: float, from_currency: str, to_currency: str) -> str:
        return self._run(amount, from_currency, to_currency)


# Method 3: Using existing LangChain community tools
def get_community_tools():
    """Get pre-built community tools."""
    # Web search tool
    # tools.append(DuckDuckGoSearchRun())
    
    # Wikipedia search
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    # Tavily search (requires API key)
    # tools.append(TavilySearchResults(max_results=3))
    
    return [wikipedia]


# Method 4: Tools with more complex logic
@tool
def code_executor(language: str, code: str) -> str:
    """Execute code in a safe environment (Python only for this example)."""
    if language.lower() != "python":
        return "Only Python code execution is supported"
    
    try:
        # In production, use a sandboxed environment
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            exec(code)
        output = f.getvalue()
        return output or "Code executed successfully (no output)"
    except Exception as e:
        return f"Error executing code: {str(e)}"


# Method 5: API-based tools
@tool
def joke_generator() -> str:
    """Get a random programming joke."""
    try:
        response = requests.get("https://official-joke-api.appspot.com/jokes/programming/random")
        if response.status_code == 200:
            joke = response.json()[0]
            return f"{joke['setup']} - {joke['punchline']}"
        return "Sorry, couldn't fetch a joke right now."
    except Exception:
        return "Here's a fallback joke: Why do programmers prefer dark mode? Because light attracts bugs!"


async def demonstrate_tools():
    """Demonstrate different ways to add tools to a LangGraph agent."""
    
    # You can also run without MCP server if you don't have one
    try:
        server_params = StdioServerParameters(
            command='python',
            args=['math_server.py'],
            transport='stdio',
            definition='Math tools',
            env=None,
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)
                print("MCP tools loaded:", [t.name for t in mcp_tools])
    except Exception:
        print("No MCP server available, using only custom tools")
        mcp_tools = []

    # Combine different types of tools
    custom_tools = [
        weather_lookup,
        file_writer,
        CurrencyConverterTool(),
        code_executor,
        joke_generator,
    ]
    
    community_tools = get_community_tools()
    
    all_tools = mcp_tools + custom_tools + community_tools
    
    print(f"\nAll available tools ({len(all_tools)}):")
    for agent_tool in all_tools:
        print(f"  - {agent_tool.name}: {agent_tool.description}")
    
    # Create agent with all tools
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    agent = create_react_agent(llm, all_tools)
    
    # Test the agent
    test_queries = [
        "What's the weather in New York?",
        "Convert 100 USD to EUR",
        "Write 'Hello World' to a file called test.txt",
        "Tell me a programming joke",
        "Search Wikipedia for information about Python programming language",
    ]
    
    for query in test_queries:
        print(f"\n🤖 Query: {query}")
        print("=" * 50)
        try:
            response = await agent.ainvoke({'messages': [{'role': 'user', 'content': query}]})
            print(f"Response: {response['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print()


if __name__ == '__main__':
    asyncio.run(demonstrate_tools())
