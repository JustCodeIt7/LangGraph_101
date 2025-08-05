# LangGraph + MCP Servers Examples

This directory contains 3 comprehensive examples demonstrating how to use Model Context Protocol (MCP) servers with LangGraph for building powerful AI agents.

## Examples Overview

### Example 1: Basic MCP Agent (`example_1_basic_mcp_agent.py`)
- **Purpose**: Introduction to MCP with LangGraph's prebuilt agent
- **Features**:
  - Uses `create_react_agent` for simple setup
  - Integrates math and weather MCP servers
  - Demonstrates basic tool usage patterns
  - Shows error handling basics

### Example 2: Custom Workflow (`example_2_custom_workflow.py`)
- **Purpose**: Custom StateGraph workflow with MCP tools
- **Features**:
  - Custom workflow using `StateGraph`
  - Conditional routing and tool execution
  - Step-by-step logging and debugging
  - Complex multi-step task handling

### Example 3: Advanced File Operations (`example_3_advanced_file_ops.py`)
- **Purpose**: Sophisticated workflow with state management and error handling
- **Features**:
  - File system operations through MCP
  - Advanced error handling and recovery
  - State persistence with memory
  - Complex project management tasks

## MCP Servers Included

### Math Server (`math_server.py`)
Provides mathematical operations:
- `add(a, b)` - Addition
- `subtract(a, b)` - Subtraction  
- `multiply(a, b)` - Multiplication
- `divide(a, b)` - Division with zero-check
- `power(base, exponent)` - Exponentiation
- `square_root(number)` - Square root with negative check

### Weather Server (`weather_server.py`)
Provides weather information:
- `get_weather(location)` - Current weather
- `get_forecast(location, days)` - Multi-day forecast
- `compare_weather(location1, location2)` - Weather comparison

### File Operations Server (`file_server.py`)
Provides secure file operations:
- `create_file(filename, content)` - Create files
- `read_file(filename)` - Read file contents
- `write_file(filename, content)` - Write/overwrite files
- `append_to_file(filename, content)` - Append content
- `list_files(directory)` - List directory contents
- `delete_file(filename)` - Delete files
- `create_directory(directory)` - Create directories
- `delete_directory(directory)` - Delete directories
- `file_info(filename)` - Get file metadata
- `search_files(pattern, directory)` - Search for files

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Weather Server** (for Examples 1 & 2):
   ```bash
   python weather_server.py
   ```
   This starts an HTTP server on port 8000.

3. **Run Examples**:
   ```bash
   # Basic example
   python example_1_basic_mcp_agent.py
   
   # Custom workflow
   python example_2_custom_workflow.py
   
   # Advanced file operations
   python example_3_advanced_file_ops.py
   ```

## Key Concepts Demonstrated

### 1. MCP Client Setup
```python
client = MultiServerMCPClient({
    "server_name": {
        "command": "python",
        "args": ["server_file.py"],
        "transport": "stdio",  # or "streamable_http"
    }
})
```

### 2. Tool Integration
```python
tools = await client.get_tools()
model_with_tools = model.bind_tools(tools)
```

### 3. Custom Workflows
```python
builder = StateGraph(StateClass)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("call_model", should_continue)
```

### 4. Error Handling
- Graceful error recovery
- Operation logging
- Maximum error limits
- Fallback strategies

### 5. State Management
- Persistent state across operations
- Memory checkpointing
- Progress tracking
- Operation history

## Transport Types

### STDIO Transport
- Best for: Simple command-line tools
- Communication: Standard input/output
- Usage: Local Python scripts

### Streamable HTTP Transport  
- Best for: Web services and remote servers
- Communication: HTTP requests
- Usage: Network-accessible services

## Error Handling Patterns

The examples demonstrate several error handling approaches:

1. **Basic Error Checking**: Simple try/catch with user feedback
2. **Error Counting**: Track errors and stop after threshold
3. **Recovery Strategies**: Attempt alternative approaches on failure
4. **Graceful Degradation**: Continue with partial functionality

## Security Considerations

The file operations server includes security measures:
- **Sandboxing**: All operations restricted to `./mcp_sandbox/` directory
- **Path Validation**: Prevents directory traversal attacks
- **Safe Operations**: Error handling for invalid operations
- **Resource Limits**: Controlled access to file system

## YouTube Video Structure

These examples are designed for a comprehensive tutorial covering:

1. **Introduction** (Example 1)
   - What is MCP?
   - Basic setup and first agent
   - Simple tool usage

2. **Intermediate** (Example 2) 
   - Custom workflows
   - StateGraph concepts
   - Advanced routing

3. **Advanced** (Example 3)
   - Complex state management
   - Error handling strategies
   - Real-world applications

## Troubleshooting

### Common Issues:

1. **"Connection refused"**: Make sure weather server is running on port 8000
2. **"Module not found"**: Install all dependencies with `pip install -r requirements.txt`
3. **"Permission denied"**: Check file permissions in the sandbox directory
4. **"Server timeout"**: Ensure MCP servers are responding properly

### Debug Tips:

- Check server logs for detailed error messages
- Use the logging features in Example 2 and 3
- Verify tool registration with `print([tool.name for tool in tools])`
- Test MCP servers individually before integration