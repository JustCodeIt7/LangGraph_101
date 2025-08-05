# Example 2: Custom LangGraph Workflow with MCP Tools
# This example shows how to build a custom workflow using MCP tools

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

async def custom_workflow_example():
    """
    Custom workflow that uses MCP tools in a controlled manner
    """
    
    # Step 1: Initialize the chat model
    model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    
    # Step 2: Set up MCP client with fetch server
    client = MultiServerMCPClient({
        "fetch": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-fetch"],
            "transport": "stdio",
        }
    })
    
    # Step 3: Get tools and bind them to the model
    tools = await client.get_tools()
    model_with_tools = model.bind_tools(tools)
    
    # Step 4: Create ToolNode for executing tools
    tool_node = ToolNode(tools)
    
    # Step 5: Define workflow functions
    def should_continue(state: MessagesState):
        """Decide whether to continue with tools or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, execute them
        if last_message.tool_calls:
            return "tools"
        return END
    
    async def call_model(state: MessagesState):
        """Call the model with current messages"""
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    # Step 6: Build the workflow graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    
    # Add edges
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )
    builder.add_edge("tools", "call_model")
    
    # Step 7: Compile the graph
    graph = builder.compile()
    
    # Step 8: Test the workflow
    print("Testing custom workflow with MCP fetch tool...")
    
    response = await graph.ainvoke({
        "messages": [
            {
                "role": "user", 
                "content": "Fetch https://api.github.com/users/octocat and summarize the user profile"
            }
        ]
    })
    
    print("\nFinal response:")
    print(response["messages"][-1].content)
    
    # Clean up
    await client.cleanup()

if __name__ == "__main__":
    asyncio.run(custom_workflow_example())