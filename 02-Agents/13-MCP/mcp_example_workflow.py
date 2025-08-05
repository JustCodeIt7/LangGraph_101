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


# Agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# Simple local tool
@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression."""
    try:
        # Only allow basic math operations for safety
        allowed_chars = set('0123456789+-*/.(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f'{expression} = {result}'
        else:
            return 'Invalid expression'
    except Exception as e:
        return f'Calculation error: {e}'


class SimpleAgent:
    def __init__(self):
        self.tools = []
        self.llm = None
        self.workflow = None
        self.mcp_contexts = []

    async def setup(self):
        """Setup tools and workflow."""
        # Start with local tools
        local_tools = [calculate]

        # Add MCP fetch tool
        try:
            print('🔧 Loading MCP fetch tool...')
            server_params = StdioServerParameters(command='python', args=['-m', 'mcp_server_fetch'])

            stdio_context = stdio_client(server_params)
            read, write = await stdio_context.__aenter__()
            self.mcp_contexts.append(stdio_context)

            session_context = ClientSession(read, write)
            session = await session_context.__aenter__()
            self.mcp_contexts.append(session_context)

            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            print(f'✅ MCP tools: {[t.name for t in mcp_tools]}')

            self.tools = local_tools + mcp_tools
        except Exception as e:
            print(f'⚠️ MCP failed: {e}. Using local tools only.')
            self.tools = local_tools

        # Setup LLM and workflow
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0).bind_tools(self.tools)

        workflow = StateGraph(AgentState)
        workflow.add_node('agent', self.agent_node)
        workflow.add_node('tools', self.tools_node)
        workflow.set_entry_point('agent')
        workflow.add_conditional_edges('agent', self.should_continue, {'continue': 'tools', 'end': END})
        workflow.add_edge('tools', 'agent')

        self.workflow = workflow.compile()

    async def cleanup(self):
        """Clean up MCP resources."""
        for context in reversed(self.mcp_contexts):
            try:
                await context.__aexit__(None, None, None)
            except Exception:
                pass

    async def agent_node(self, state: AgentState) -> AgentState:
        """Agent decides what to do next."""
        response = await self.llm.ainvoke(state['messages'])
        return {'messages': [response]}

    async def tools_node(self, state: AgentState) -> AgentState:
        """Execute requested tools."""
        last_message = state['messages'][-1]
        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f'🔧 {tool_name}({tool_args})')

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    result = await tool.ainvoke(tool_args)
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
                    print(f'✅ {str(result)[:100]}...' if len(str(result)) > 100 else f'✅ {result}')
                except Exception as e:
                    error_msg = f'Error: {e}'
                    tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
                    print(f'❌ {error_msg}')

        return {'messages': tool_messages}

    def should_continue(self, state: AgentState) -> str:
        """Check if agent should continue or end."""
        last_message = state['messages'][-1]
        return 'continue' if hasattr(last_message, 'tool_calls') and last_message.tool_calls else 'end'

    async def ask(self, question: str):
        """Ask the agent a question."""
        print(f'\n🤖 Question: {question}')
        result = await self.workflow.ainvoke({'messages': [HumanMessage(content=question)]})
        answer = result['messages'][-1].content
        print(f'✨ Answer: {answer}\n')
        return answer


async def main():
    """Run the agent."""
    agent = SimpleAgent()

    try:
        await agent.setup()

        # Example questions
        await agent.ask("What's 15% of 96?")
        await agent.ask('fetch the website https://langchain-ai.github.io/langgraph/ and summarize it')

    except Exception as e:
        print(f'❌ Error: {e}')
    finally:
        await agent.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
