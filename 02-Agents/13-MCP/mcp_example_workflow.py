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


################################ Agent State ################################
class AgentState(TypedDict):
    # The list of messages accumulates over time
    messages: Annotated[List[BaseMessage], add_messages]


################################ Local Tools ################################
@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression."""
    try:
        # Use a whitelist to prevent arbitrary code execution
        allowed_chars = set('0123456789+-*/.(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f'{expression} = {result}'
        else:
            return 'Invalid expression'
    except Exception as e:
        return f'Calculation error: {e}'


################################ Agent Definition ################################
class SimpleAgent:
    def __init__(self):
        self.tools = []
        self.llm = None
        self.workflow = None
        self.mcp_contexts = []  # Store contexts for later cleanup

    async def setup(self):
        """Setup tools and compile the agent workflow."""
        # Define local tools available to the agent
        local_tools = [calculate]

        # Attempt to add remote tools via MCP
        try:
            print('🔧 Loading MCP fetch tool...')
            # Configure the MCP server to run as a Python module subprocess
            server_params = StdioServerParameters(command='python', args=['-m', 'mcp_server_fetch'])

            # Establish a connection to the MCP server
            stdio_context = stdio_client(server_params)
            read, write = await stdio_context.__aenter__()
            self.mcp_contexts.append(stdio_context)

            # Create a client session for communication
            session_context = ClientSession(read, write)
            session = await session_context.__aenter__()
            self.mcp_contexts.append(session_context)

            # Initialize the session and load remote tools
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            print(f'✅ MCP tools: {[t.name for t in mcp_tools]}')

            self.tools = local_tools + mcp_tools
        except Exception as e:
            # Fall back to using only local tools if MCP connection fails
            print(f'⚠️ MCP failed: {e}. Using local tools only.')
            self.tools = local_tools

        # Configure the LLM and grant it access to the available tools
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0).bind_tools(self.tools)

        # Define the agent's graph structure
        workflow = StateGraph(AgentState)
        workflow.add_node('agent', self.agent_node)
        workflow.add_node('tools', self.tools_node)
        workflow.set_entry_point('agent')

        # Define the control flow logic
        workflow.add_conditional_edges('agent', self.should_continue, {'continue': 'tools', 'end': END})
        workflow.add_edge('tools', 'agent')

        # Compile the graph into a runnable workflow
        self.workflow = workflow.compile()

    async def cleanup(self):
        """Clean up any active MCP resources."""
        # Close contexts in reverse order of creation
        for context in reversed(self.mcp_contexts):
            try:
                await context.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup

    async def agent_node(self, state: AgentState) -> AgentState:
        """Invoke the LLM to decide the next action."""
        response = await self.llm.ainvoke(state['messages'])
        return {'messages': [response]}

    async def tools_node(self, state: AgentState) -> AgentState:
        """Execute the tools requested by the agent."""
        last_message = state['messages'][-1]
        tool_messages = []

        # Iterate through all tool calls made by the LLM
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f'🔧 {tool_name}({tool_args})')

            # Find the corresponding tool implementation
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    # Invoke the tool and capture its result
                    result = await tool.ainvoke(tool_args)
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
                    print(f'✅ {str(result)[:100]}...' if len(str(result)) > 100 else f'✅ {result}')
                except Exception as e:
                    # Handle any errors during tool execution
                    error_msg = f'Error: {e}'
                    tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
                    print(f'❌ {error_msg}')

        return {'messages': tool_messages}

    def should_continue(self, state: AgentState) -> str:
        """Determine whether to continue with another tool call or end."""
        last_message = state['messages'][-1]
        # Continue if the agent requested a tool, otherwise end
        return 'continue' if hasattr(last_message, 'tool_calls') and last_message.tool_calls else 'end'

    async def ask(self, question: str):
        """Present a question to the agent and get a final answer."""
        print(f'\n🤖 Question: {question}')
        # Invoke the compiled workflow with the user's message
        initial_state = {'messages': [HumanMessage(content=question)]}
        result = await self.workflow.ainvoke(initial_state)
        # Extract the final response from the agent
        answer = result['messages'][-1].content
        print(f'✨ Answer: {answer}\n')
        return answer


################################ Main Execution ################################
async def main():
    """Initialize and run the agent."""
    agent = SimpleAgent()

    # Use a try/finally block to ensure resources are always cleaned up
    try:
        await agent.setup()

        # Ask the agent a series of questions
        await agent.ask("What's 15% of 96?")
        await agent.ask('fetch the website https://langchain-ai.github.io/langgraph/ and summarize it')

    except Exception as e:
        print(f'❌ Error: {e}')
    finally:
        await agent.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
