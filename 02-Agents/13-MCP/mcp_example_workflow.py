import asyncio
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import datetime

################################ Configuration ################################
# Simple configuration
QUESTIONS = [
    'fetch the website https://langchain-ai.github.io/langgraph/ and summarize it',
    "What's the current time? Also calculate (3 + 5) * 12 and then find 15% of that result.",
]


################################ Agent State ################################
# Agent state
class AgentState(TypedDict):
    # The `add_messages` function appends new messages to the existing list
    messages: Annotated[List[BaseMessage], add_messages]


################################ Tools ################################
# Simple tools
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate what percentage of a value is.
    Args:
        value: The base value
        percentage: The percentage to calculate (e.g., 20 for 20%)
    """
    return (value * percentage) / 100


################################ Agent Class ################################
class SimpleAgent:
    def __init__(self):
        self.tools = []
        self.llm = None
        self.workflow = None
        # MCP session storage
        self.mcp_stdio_context = None
        self.mcp_session_context = None

    async def setup(self):
        """Setup tools and workflow."""
        # Basic tools defined locally
        basic_tools = [get_current_time, add_numbers, multiply_numbers, calculate_percentage]

        # Add MCP fetch tool, which runs in a separate process
        try:
            print('🔧 Setting up MCP fetch tool...')
            # Define how to start the MCP server process
            server_params = StdioServerParameters(
                command='python',
                args=['-m', 'mcp_server_fetch'],  # Run the fetch server module
                env=None,
            )

            # Establish a connection to the server process
            self.mcp_stdio_context = stdio_client(server_params)
            read, write = await self.mcp_stdio_context.__aenter__()

            # Start a client session over the established connection
            self.mcp_session_context = ClientSession(read, write)
            session = await self.mcp_session_context.__aenter__()
            await session.initialize()

            # Load the tools exposed by the MCP server
            mcp_tools = await load_mcp_tools(session)
            print(f'✅ MCP tools loaded: {[tool.name for tool in mcp_tools]}')

            # Combine local and remote tools
            self.tools = basic_tools + mcp_tools

        except Exception as e:
            # Fallback to only basic tools if MCP setup fails
            print(f'⚠️ MCP setup failed: {e}')
            print('Using basic tools only')
            self.tools = basic_tools

        # Setup LLM and bind the available tools for the agent to use
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0).bind_tools(self.tools)

        # Define the agent's graph structure
        workflow = StateGraph(AgentState)
        workflow.add_node('agent', self.agent_node)
        workflow.add_node('tools', self.tools_node)

        # Define the control flow
        workflow.set_entry_point('agent')
        workflow.add_conditional_edges('agent', self.should_continue, {'continue': 'tools', 'end': END})
        workflow.add_edge('tools', 'agent')  # Loop back to the agent after tool execution

        self.workflow = workflow.compile()  # Compile the graph into a runnable object
        print(f'📦 All tools ready: {[tool.name for tool in self.tools]}')

    async def cleanup(self):
        """Clean up MCP resources to terminate the server process."""
        try:
            # Gracefully close the session and the process connection
            if self.mcp_session_context:
                await self.mcp_session_context.__aexit__(None, None, None)
            if self.mcp_stdio_context:
                await self.mcp_stdio_context.__aexit__(None, None, None)
            print('🧹 Cleanup complete')
        except Exception as e:
            print(f'⚠️ Cleanup error: {e}')

    async def agent_node(self, state: AgentState) -> AgentState:
        """Main agent reasoning node. The LLM decides what to do next."""
        response = await self.llm.ainvoke(state['messages'])
        return {'messages': [response]}

    async def tools_node(self, state: AgentState) -> AgentState:
        """Execute tools called by the agent."""
        last_message = state['messages'][-1]
        tool_messages = []

        # Iterate over all tool calls requested by the LLM
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f'🔧 {tool_name}({tool_args})')

            # Find the corresponding tool function and execute it
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    result = await tool.ainvoke(tool_args)
                    # Format the result as a ToolMessage to be sent back to the LLM
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

                    # Show a preview of the tool's output
                    preview = f'{str(result)[:100]}...' if len(str(result)) > 100 else str(result)
                    print(f'✅ {preview}')

                except Exception as e:  # type: ignore
                    # Handle tool execution errors
                    error_msg = f'Error: {e}' if str(e) else f'Unknown error with {tool_name}'
                    tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
                    print(f'❌ {error_msg}')
            else:
                # Handle cases where the LLM hallucinates a non-existent tool
                tool_messages.append(ToolMessage(content=f'Tool {tool_name} not found', tool_call_id=tool_call['id']))
                print(f'❌ Tool {tool_name} not found')

        return {'messages': tool_messages}

    def should_continue(self, state: AgentState) -> str:
        """Check if the agent should continue with tool execution or end."""
        last_message = state['messages'][-1]
        # If the last message has tool calls, continue to the 'tools' node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'continue'
        # Otherwise, end the conversation
        return 'end'

    async def run(self, question: str):
        """Run the agent on a single question."""
        print(f'\n🤖 Question: {question}')
        print('-' * 60)

        # Invoke the compiled graph with the initial state
        result = await self.workflow.ainvoke({'messages': [HumanMessage(content=question)]})

        # Print the final response from the agent
        print(f'\n✨ Answer:')
        print(result['messages'][-1].content)
        print('=' * 60)


################################ Main Execution ################################
async def main():
    """Run the simple agent with MCP."""
    agent = SimpleAgent()

    try:
        print('🚀 Simple LangGraph Agent with MCP')
        await agent.setup()  # Initialize the agent and its tools

        # Run the agent for each question in the list
        for i, question in enumerate(QUESTIONS, 1):
            print(f'\n📝 Question {i}/{len(QUESTIONS)}')
            await agent.run(question)

            # Pause between questions for better readability
            if i < len(QUESTIONS):
                await asyncio.sleep(1)

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback

        traceback.print_exc()
    finally:
        # Ensure resources are cleaned up even if errors occur
        await agent.cleanup()


# Run the main asynchronous function
if __name__ == '__main__':
    asyncio.run(main())
