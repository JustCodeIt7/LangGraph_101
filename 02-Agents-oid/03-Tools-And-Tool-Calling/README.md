# LangGraph Tools and Tool Calling Tutorial

This tutorial demonstrates how to create and use tools with LangGraph agents, covering tool calling, execution, and integration into conversation flows.

## 📚 What You'll Learn

- **Custom Tool Creation**: Using the `@tool` decorator to define functions the AI can call
- **Tool Execution**: How LangGraph handles tool invocation and result processing
- **State Management**: Tracking tool usage and conversation history
- **Conditional Routing**: Directing graph flow based on tool requirements
- **Error Handling**: Robust tool execution with fallback mechanisms
- **Interactive Agents**: Building conversational AI with tool capabilities

## 🔧 Tools Included

The tutorial includes 5 custom tools:

1. **get_current_time()** - Returns current date and time
2. **calculate_math(expression)** - Safely evaluates mathematical expressions
3. **get_random_fact()** - Provides interesting random facts
4. **weather_simulator(location)** - Simulates weather data for any location
5. **text_analyzer(text)** - Analyzes text and provides statistics

## 🏗️ Architecture

The tutorial uses a StateGraph with the following flow:

```
initialize → agent → [tools needed?] → execute_tools → final_response → END
                   ↘ [no tools] → END
```

### Key Components

- **ToolAgentState**: Manages conversation history, user info, and tool usage stats
- **Tool Executor**: Handles safe tool invocation and error handling
- **Conditional Routing**: Determines whether tools need to be executed
- **State Persistence**: Maintains context across multiple interactions

## 🚀 Running the Tutorial

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install langgraph langchain_openai langchain_core
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Execution Modes

Run the script and choose from 4 modes:

```bash
python 03-Tools-And-Tool-Calling.py
```

1. **Interactive Mode**: Chat with the tool-enabled agent
2. **Run Demonstrations**: See pre-defined examples
3. **Test Individual Tools**: Test each tool separately
4. **Show All Examples**: Complete demonstration suite

## 💡 Example Interactions

### Tool Usage Examples

**Math Calculation:**
```
User: Calculate 25 * 4 + 10
AI: I'll calculate that for you.
Tool: calculate_math(expression="25 * 4 + 10")
Result: The result of '25 * 4 + 10' is: 110
```

**Weather Query:**
```
User: What's the weather in Paris?
AI: Let me check the weather for Paris.
Tool: weather_simulator(location="Paris")
Result: Weather in Paris: Sunny, 22°C
```

**Text Analysis:**
```
User: Analyze this text: "LangGraph is amazing!"
AI: I'll analyze that text for you.
Tool: text_analyzer(text="LangGraph is amazing!")
Result: Text Analysis Results:
- Characters (with spaces): 22
- Words: 3
- Average word length: 5.7 characters
```

## 🎯 Key Learning Points

### 1. Tool Definition
```python
@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    # Tool implementation with error handling
    return result
```

### 2. Tool Binding
```python
llm = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(tools)
```

### 3. Tool Execution
```python
tool_invocation = ToolInvocation(tool=tool_name, tool_input=args)
result = tool_executor.invoke(tool_invocation)
```

### 4. State Management
```python
class ToolAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    tools_used: List[str]
    total_tool_calls: int
```

## 🔄 Workflow Details

1. **Initialization**: Set up agent state and user context
2. **LLM Call**: Agent decides if tools are needed
3. **Tool Execution**: Execute requested tools safely
4. **Response Generation**: Create final response with tool results
5. **State Update**: Track tool usage and conversation history

## 🛡️ Error Handling

The tutorial includes comprehensive error handling:

- **Safe Math Evaluation**: Prevents code injection
- **Tool Execution Errors**: Graceful failure with error messages
- **Input Validation**: Checks for valid inputs
- **State Recovery**: Maintains conversation flow despite errors

## 🎮 Interactive Features

- **Real-time Tool Tracking**: See which tools are used
- **Session Statistics**: Track tool usage across conversation
- **Multiple Input Modes**: Support for various question types
- **Demo Mode**: Built-in examples to explore capabilities

## 📈 Advanced Concepts

- **Conditional Routing**: Graph flow based on tool requirements
- **Message Types**: Handling different message types (Human, AI, Tool, System)
- **State Persistence**: Maintaining context across interactions
- **Tool Result Integration**: Seamless incorporation of tool outputs

## 🔍 Debugging Tips

- Use the verbose output to see graph execution flow
- Check tool call logs for debugging
- Use 'stats' command to see session information
- Tool execution steps are clearly marked with emojis

## 🚀 Next Steps

After completing this tutorial, you can:

1. Create custom tools for your specific use cases
2. Build more complex tool chains
3. Implement tool authentication and external API calls
4. Add tool result caching and optimization
5. Create specialized agents for different domains

## 📝 Notes

- The tutorial uses simulated tools for demonstration
- Real applications would connect to actual APIs and services
- Error handling examples show production-ready patterns
- State management demonstrates scalable conversation tracking
