import asyncio
from typing import TypedDict, Annotated, Sequence
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import datetime

# Set up server parameters (update path to your math_server.py)
server_params = StdioServerParameters(
    command='python',
    args=['math_server.py'],  # might need to adjust this path
    env=None,
)


# Define custom tools
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate what percentage of a value is.
    Args:
        value: The base value
        percentage: The percentage to calculate (e.g., 20 for 20%)
    """
    return (value * percentage) / 100


# Define the state for our workflow
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_workflow_agent(llm, tools):
    """Create a LangGraph workflow agent with the given LLM and tools."""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            """You are a helpful assistant with access to various tools.
        When you need to use a tool, call it with the appropriate parameters.
        Always provide a clear and helpful response to the user's question.
        
        Available tools: {tool_names}
        
        Think step by step and use tools as needed to answer the user's question completely.""",
        ),
        MessagesPlaceholder(variable_name='messages'),
    ])

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create the tool node for executing tools
    tool_node = ToolNode(tools)

    def should_continue(state: WorkflowState):
        """Determine if we should continue to tools or end."""
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'tools'
        return END

    def call_model(state: WorkflowState):
        """Call the model with the current state."""
        # Format the prompt with available tool names
        tool_names = [tool.name for tool in tools]
        formatted_prompt = prompt.format_messages(tool_names=', '.join(tool_names), messages=state['messages'])

        # Call the LLM
        response = llm_with_tools.invoke(formatted_prompt)
        return {'messages': [response]}

    # Build the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node('agent', call_model)
    workflow.add_node('tools', tool_node)

    # Add edges
    workflow.add_edge(START, 'agent')
    workflow.add_conditional_edges('agent', should_continue, ['tools', END])
    workflow.add_edge('tools', 'agent')

    # Compile the graph
    return workflow.compile()


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # Initialize connection

            # Load tools from the MCP server
            mcp_tools = await load_mcp_tools(session)
            print('MCP tools:', [tool.name for tool in mcp_tools])  # Should show ['add', 'multiply']

            # Define additional custom tools
            custom_tools = [
                get_current_time,
                calculate_percentage,
                # DuckDuckGoSearchRun(),  # Uncomment if you want web search
            ]

            # Combine MCP tools with custom tools
            all_tools = mcp_tools + custom_tools
            print('All available tools:', [tool.name for tool in all_tools])

            # Set up the LLM
            # llm = ChatOpenAI(model='gpt-4o', temperature=0)  # Replace with your key
            llm = ChatOllama(model='llama3.2', temperature=0)  # Or use Ollama model

            # Create the workflow agent
            workflow_agent = create_workflow_agent(llm, all_tools)

            # Test with multiple tools
            initial_state = {
                'messages': [
                    HumanMessage(
                        content="What's the current time? Also calculate (3 + 5) * 12 and then find 15% of that result."
                    )
                ]
            }

            print('Starting workflow execution...')

            # Execute the workflow
            final_state = await workflow_agent.ainvoke(initial_state)

            # Print the final response
            print('Agent response:', final_state['messages'][-1].content)

            # Optionally, print the full conversation flow
            print('\n--- Full Conversation Flow ---')
            for i, message in enumerate(final_state['messages']):
                if isinstance(message, HumanMessage):
                    print(f'{i + 1}. Human: {message.content}')
                elif isinstance(message, AIMessage):
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f'{i + 1}. AI: [Calling tools: {[tc["name"] for tc in message.tool_calls]}]')
                    else:
                        print(f'{i + 1}. AI: {message.content}')
                elif isinstance(message, ToolMessage):
                    print(f'{i + 1}. Tool ({message.name}): {message.content}')


if __name__ == '__main__':
    asyncio.run(main())
