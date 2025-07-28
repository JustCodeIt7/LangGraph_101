from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# Define the LLM (as provided)
llm = init_chat_model("ollama:llama3.2", temperature=0)


# Define a simple tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


# Create the agent with tools
tools = [add_numbers]
agent = create_react_agent(model=llm, tools=tools)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 5 + 3?"}]
})
print(result['messages'][-1].content)
