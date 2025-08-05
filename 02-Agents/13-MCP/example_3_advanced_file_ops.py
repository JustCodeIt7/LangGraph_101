# Example 3: Advanced Multi-Server MCP Setup with Error Handling
# This example shows how to use multiple MCP servers with proper error handling

import asyncio
import logging
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def advanced_mcp_example():
    """
    Advanced example with multiple MCP servers and error handling
    """

    # Step 1: Initialize model
    model = init_chat_model('anthropic:claude-3-5-sonnet-latest')

    # Step 2: Set up multiple MCP servers
    server_configs = {
        'fetch': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-fetch'],
            'transport': 'stdio',
        },
        # You can add more servers here, for example:
        # "filesystem": {
        #     "command": "npx",
        #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        #     "transport": "stdio",
        # }
    }

    client = None
    try:
        # Step 3: Initialize client with error handling
        logger.info('Initializing MCP client...')
        client = MultiServerMCPClient(server_configs)

        # Step 4: Get tools from all servers
        tools = await client.get_tools()
        logger.info(f'Loaded {len(tools)} tools: {[tool.name for tool in tools]}')

        if not tools:
            logger.error('No tools were loaded!')
            return

        # Step 5: Set up the workflow with enhanced error handling
        model_with_tools = model.bind_tools(tools)
        tool_node = ToolNode(tools)

        def should_continue(state: MessagesState):
            """Enhanced decision function with logging"""
            messages = state['messages']
            last_message = messages[-1]

            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f'Executing {len(last_message.tool_calls)} tool calls')
                return 'tools'

            logger.info('Workflow complete')
            return END

        async def call_model_with_error_handling(state: MessagesState):
            """Model call with error handling"""
            try:
                messages = state['messages']
                logger.info('Calling model...')
                response = await model_with_tools.ainvoke(messages)
                return {'messages': [response]}
            except Exception as e:
                logger.error(f'Error calling model: {e}')
                # Return error message to user
                error_msg = {'role': 'assistant', 'content': f'I encountered an error: {str(e)}'}
                return {'messages': [error_msg]}

        async def execute_tools_with_error_handling(state: MessagesState):
            """Tool execution with error handling"""
            try:
                logger.info('Executing tools...')
                return await tool_node.ainvoke(state)
            except Exception as e:
                logger.error(f'Error executing tools: {e}')
                # Add error message to state
                error_msg = {'role': 'tool', 'content': f'Tool execution failed: {str(e)}', 'tool_call_id': 'error'}
                return {'messages': state['messages'] + [error_msg]}

        # Step 6: Build the enhanced workflow
        builder = StateGraph(MessagesState)
        builder.add_node('call_model', call_model_with_error_handling)
        builder.add_node('tools', execute_tools_with_error_handling)

        builder.add_edge(START, 'call_model')
        builder.add_conditional_edges('call_model', should_continue)
        builder.add_edge('tools', 'call_model')

        graph = builder.compile()

        # Step 7: Test with multiple queries
        test_queries = [
            'Fetch https://httpbin.org/json and tell me about the data structure',
            'Get the content from https://api.github.com/repos/microsoft/vscode and summarize the repository information',
        ]

        for i, query in enumerate(test_queries, 1):
            print(f'\n{"=" * 50}')
            print(f'Test Query {i}: {query}')
            print('=' * 50)

            try:
                response = await graph.ainvoke({'messages': [{'role': 'user', 'content': query}]})

                print(f'\nResponse {i}:')
                print(response['messages'][-1].content)

            except Exception as e:
                logger.error(f'Error processing query {i}: {e}')
                print(f'Query {i} failed: {e}')

    except Exception as e:
        logger.error(f'Failed to initialize MCP client: {e}')
        print(f'Setup failed: {e}')

    finally:
        # Step 8: Clean up resources
        if client:
            try:
                await client.cleanup()
                logger.info('MCP client cleaned up successfully')
            except Exception as e:
                logger.error(f'Error during cleanup: {e}')


async def demonstrate_tool_info():
    """Bonus function to demonstrate how to inspect available tools"""
    client = MultiServerMCPClient({
        'fetch': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-fetch'],
            'transport': 'stdio',
        }
    })

    try:
        tools = await client.get_tools()

        print('\nAvailable MCP Tools:')
        print('=' * 30)
        for tool in tools:
            print(f'Name: {tool.name}')
            print(f'Description: {tool.description}')
            if hasattr(tool, 'parameters'):
                print(f'Parameters: {tool.parameters}')
            print('-' * 30)

    except Exception as e:
        print(f'Error inspecting tools: {e}')
    finally:
        await client.cleanup()


if __name__ == '__main__':
    # Run the advanced example
    asyncio.run(advanced_mcp_example())

    # Optionally, demonstrate tool inspection
    print('\n' + '=' * 60)
    print('BONUS: Tool Inspection')
    print('=' * 60)
    asyncio.run(demonstrate_tool_info())
