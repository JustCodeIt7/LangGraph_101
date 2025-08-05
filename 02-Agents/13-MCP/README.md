# Adding Tools to LangGraph Agents

This directory demonstrates various ways to add tools to your LangGraph agents. Here are the main approaches:

## 1. Using MCP (Model Context Protocol) Tools + Custom Tools (Recommended)

This is what your current `mcp_example.py` demonstrates - combining MCP server tools with custom LangChain tools:

```python
# Load tools from MCP server
mcp_tools = await load_mcp_tools(session)

# Define custom tools
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Combine them
all_tools = mcp_tools + custom_tools
agent = create_react_agent(llm, all_tools)
```

**Pros:**
- Best of both worlds - MCP standardization + custom flexibility
- Easy to extend
- Type-safe with proper schemas

## 2. Pure Custom Tools (@tool decorator)

The simplest way to add custom functionality:

```python
from langchain_core.tools import tool

@tool
def weather_lookup(city: str) -> str:
    """Get current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate what percentage of a value is."""
    return (value * percentage) / 100
```

**Pros:**
- Simple and intuitive
- Automatic schema generation
- Type hints provide validation

## 3. Custom Tool Classes

For more complex tools with advanced validation:

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CurrencyConverterInput(BaseModel):
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="Source currency code")
    to_currency: str = Field(description="Target currency code")

class CurrencyConverterTool(BaseTool):
    name: str = "currency_converter"
    description: str = "Convert currency from one type to another"
    args_schema: Type[BaseModel] = CurrencyConverterInput
    
    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        # Your implementation here
        pass
```

**Pros:**
- Full control over validation
- Custom error handling
- Complex input schemas

## 4. Pre-built Community Tools

Use existing tools from the LangChain community:

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
]
```

**Available Community Tools:**
- Web search (DuckDuckGo, Tavily, SerpAPI)
- Wikipedia search
- Database connections
- File operations
- API integrations
- And many more!

## 5. API-based Tools

Tools that call external APIs:

```python
@tool
def joke_generator() -> str:
    """Get a random programming joke."""
    response = requests.get("https://official-joke-api.appspot.com/jokes/programming/random")
    if response.status_code == 200:
        joke = response.json()[0]
        return f"{joke['setup']} - {joke['punchline']}"
    return "Sorry, couldn't fetch a joke right now."
```

## 6. File and System Tools

Tools that interact with the file system or execute code:

```python
@tool
def file_writer(filename: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def code_executor(language: str, code: str) -> str:
    """Execute code in a safe environment."""
    # Implementation with proper sandboxing
    pass
```

## Examples in This Directory

- `mcp_example.py` - Your original MCP + custom tools example
- `tool_examples.py` - Comprehensive examples of all tool types
- `math_server.py` - MCP server providing math operations

## Best Practices

1. **Always provide clear descriptions** - The LLM uses these to decide when to use tools
2. **Use type hints** - They improve reliability and provide automatic validation
3. **Handle errors gracefully** - Return meaningful error messages, don't let exceptions bubble up
4. **Keep tools focused** - Each tool should do one thing well
5. **Test tools individually** - Make sure they work before adding to an agent
6. **Use async when possible** - For better performance with I/O operations

## Running the Examples

1. **Basic MCP + Custom Tools:**
   ```bash
   python mcp_example.py
   ```

2. **Comprehensive Tool Demo:**
   ```bash
   python tool_examples.py
   ```

## Adding Your Own Tools

1. Choose the appropriate method based on complexity
2. Define clear input/output schemas
3. Add proper error handling
4. Test the tool individually
5. Add it to your tools list
6. Update your agent with the new tools

Remember: The key is to combine different types of tools to create a comprehensive agent that can handle a wide variety of tasks!
