"""
LangGraph Agents Tutorial: Running Agents
==========================================

This tutorial demonstrates how to run LangGraph agents with various execution modes,
input/output formats, streaming, and execution controls.

YouTube Tutorial: Running LangGraph Agents
"""

import asyncio
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.tools import tool


# Mock weather tool for demonstration
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    weather_data = {
        "san francisco": "Sunny, 72°F",
        "new york": "Cloudy, 65°F", 
        "london": "Rainy, 58°F",
        "tokyo": "Clear, 70°F"
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def print_section(title: str):
    """Helper function to print section headers."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_messages(messages: List[Dict[str, Any]]):
    """Helper function to print messages in a readable format."""
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        print(f"{role.upper()}: {content}")


class LangGraphAgentTutorial:
    """
    A comprehensive tutorial class demonstrating LangGraph agent execution patterns.
    """
    
    def __init__(self):
        """Initialize the tutorial with a basic agent."""
        # Note: In a real implementation, you would use an actual LLM
        # For this tutorial, we'll simulate the agent creation
        self.tools = [get_weather, calculate]
        
        # Create a React agent with tools
        # In practice, you would provide an actual model here
        # self.agent = create_react_agent(
        #     model="your-model-here",  # e.g., "openai:gpt-4"
        #     tools=self.tools
        # )
        
        print("🤖 LangGraph Agent Tutorial Initialized")
        print(f"📋 Available tools: {[tool.name for tool in self.tools]}")
    
    def demonstrate_basic_usage(self):
        """Demonstrate basic synchronous and asynchronous agent execution."""
        print_section("1. Basic Agent Usage")
        
        print("\n📝 Synchronous Execution (.invoke())")
        print("Code example:")
        print("""
        # Synchronous execution
        response = agent.invoke({
            "messages": [{"role": "user", "content": "what is the weather in sf"}]
        })
        """)
        
        # Simulate the response
        sample_input = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        print(f"Input: {sample_input}")
        
        # Simulated response structure
        simulated_response = {
            "messages": [
                {"role": "user", "content": "what is the weather in sf"},
                {"role": "assistant", "content": "I'll check the weather in San Francisco for you."},
                {"role": "tool", "content": "get_weather(location='san francisco')"},
                {"role": "assistant", "content": "The weather in San Francisco is sunny and 72°F."}
            ]
        }
        
        print(f"\nSimulated Response Structure:")
        print_messages(simulated_response["messages"])
        
        print("\n📝 Asynchronous Execution (await .ainvoke())")
        print("Code example:")
        print("""
        # Asynchronous execution
        response = await agent.ainvoke({
            "messages": [{"role": "user", "content": "what is the weather in sf"}]
        })
        """)
    
    def demonstrate_input_formats(self):
        """Demonstrate different input formats supported by agents."""
        print_section("2. Input Formats")
        
        input_examples = [
            {
                "name": "String Input",
                "input": {"messages": "Hello"},
                "description": "Interpreted as HumanMessage"
            },
            {
                "name": "Message Dictionary",
                "input": {"messages": {"role": "user", "content": "Hello"}},
                "description": "Single message as dictionary"
            },
            {
                "name": "List of Messages",
                "input": {"messages": [{"role": "user", "content": "Hello"}]},
                "description": "Multiple messages in a list"
            },
            {
                "name": "Custom State",
                "input": {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "user_name": "Alice"
                },
                "description": "With additional state fields"
            }
        ]
        
        for example in input_examples:
            print(f"\n🔸 {example['name']}")
            print(f"   Description: {example['description']}")
            print(f"   Input: {json.dumps(example['input'], indent=2)}")
    
    def demonstrate_output_format(self):
        """Demonstrate the structure of agent outputs."""
        print_section("3. Output Format")
        
        print("Agent outputs are dictionaries containing:")
        print("• messages: List of all messages during execution")
        print("• structured_response: (Optional) If structured output is configured")
        print("• Custom state fields: (Optional) If using custom state schema")
        
        sample_output = {
            "messages": [
                {"role": "user", "content": "Calculate 15 * 8"},
                {"role": "assistant", "content": "I'll calculate that for you."},
                {"role": "tool", "content": "calculate(expression='15 * 8')"},
                {"role": "assistant", "content": "The result of 15 * 8 is 120."}
            ]
        }
        
        print(f"\n📤 Example Output Structure:")
        print(json.dumps(sample_output, indent=2))
    
    def demonstrate_streaming(self):
        """Demonstrate streaming output capabilities."""
        print_section("4. Streaming Output")
        
        print("Streaming provides:")
        print("• Progress updates after each step")
        print("• LLM tokens as they're generated")
        print("• Custom tool messages during execution")
        
        print("\n📝 Synchronous Streaming:")
        print("""
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
            stream_mode="updates"
        ):
            print(chunk)
        """)
        
        print("\n📝 Asynchronous Streaming:")
        print("""
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
            stream_mode="updates"
        ):
            print(chunk)
        """)
        
        # Simulate streaming chunks
        print("\n🔄 Simulated Streaming Chunks:")
        streaming_chunks = [
            {"step": 1, "update": "Processing user input..."},
            {"step": 2, "update": "Calling weather tool..."},
            {"step": 3, "update": "Tool response received"},
            {"step": 4, "update": "Generating final response..."},
            {"step": 5, "update": "Response complete"}
        ]
        
        import time
        for chunk in streaming_chunks:
            print(f"   Chunk {chunk['step']}: {chunk['update']}")
            time.sleep(0.5)  # Simulate streaming delay
    
    def demonstrate_max_iterations(self):
        """Demonstrate recursion limits and iteration controls."""
        print_section("5. Max Iterations Control")
        
        print("Recursion limits prevent infinite loops and control execution steps.")
        print("Formula: recursion_limit = 2 * max_iterations + 1")
        
        print("\n📝 Runtime Configuration:")
        print("""
        max_iterations = 3
        recursion_limit = 2 * max_iterations + 1  # = 7
        
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": "question"}]},
                {"recursion_limit": recursion_limit}
            )
        except GraphRecursionError:
            print("Agent stopped due to max iterations.")
        """)
        
        print("\n📝 Using .with_config():")
        print("""
        agent_with_limit = agent.with_config(recursion_limit=7)
        
        try:
            response = agent_with_limit.invoke(
                {"messages": [{"role": "user", "content": "question"}]}
            )
        except GraphRecursionError:
            print("Agent stopped due to max iterations.")
        """)
        
        # Demonstrate the calculation
        max_iterations = 3
        recursion_limit = 2 * max_iterations + 1
        print(f"\n🔢 Example Calculation:")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Recursion limit: {recursion_limit}")
        print(f"   This allows the agent to make {max_iterations} attempts with tool calls")
    
    async def demonstrate_async_patterns(self):
        """Demonstrate asynchronous execution patterns."""
        print_section("6. Asynchronous Execution Patterns")
        
        print("🔄 Async execution allows for better performance and concurrency")
        
        # Simulate async agent calls
        async def simulate_agent_call(query: str, delay: float = 1.0):
            """Simulate an async agent call with delay."""
            print(f"   🚀 Starting: {query}")
            await asyncio.sleep(delay)  # Simulate processing time
            print(f"   ✅ Completed: {query}")
            return {"messages": [{"role": "assistant", "content": f"Response to: {query}"}]}
        
        print("\n📝 Sequential Async Calls:")
        queries = [
            "What's the weather in New York?",
            "Calculate 25 + 17",
            "What's the weather in London?"
        ]
        
        start_time = asyncio.get_event_loop().time()
        for query in queries:
            await simulate_agent_call(query, 0.5)
        sequential_time = asyncio.get_event_loop().time() - start_time
        
        print(f"   Sequential execution time: {sequential_time:.2f} seconds")
        
        print("\n📝 Concurrent Async Calls:")
        start_time = asyncio.get_event_loop().time()
        tasks = [simulate_agent_call(query, 0.5) for query in queries]
        results = await asyncio.gather(*tasks)
        concurrent_time = asyncio.get_event_loop().time() - start_time
        
        print(f"   Concurrent execution time: {concurrent_time:.2f} seconds")
        print(f"   Performance improvement: {(sequential_time/concurrent_time):.1f}x faster")
    
    def demonstrate_error_handling(self):
        """Demonstrate error handling patterns."""
        print_section("7. Error Handling")
        
        print("🛡️ Common error handling patterns for LangGraph agents:")
        
        error_examples = [
            {
                "name": "GraphRecursionError",
                "description": "When agent exceeds recursion limit",
                "code": """
try:
    response = agent.invoke(input_data, {"recursion_limit": 5})
except GraphRecursionError:
    print("Agent hit recursion limit - consider simplifying the task")
                """
            },
            {
                "name": "Tool Execution Error",
                "description": "When a tool fails during execution",
                "code": """
try:
    response = agent.invoke(input_data)
except Exception as e:
    print(f"Tool execution failed: {e}")
    # Implement fallback logic
                """
            },
            {
                "name": "Input Validation",
                "description": "Validate input format before execution",
                "code": """
def validate_input(input_data):
    if "messages" not in input_data:
        raise ValueError("Input must contain 'messages' key")
    return True

if validate_input(input_data):
    response = agent.invoke(input_data)
                """
            }
        ]
        
        for example in error_examples:
            print(f"\n🔸 {example['name']}")
            print(f"   {example['description']}")
            print(f"   Code:{example['code']}")
    
    def demonstrate_best_practices(self):
        """Demonstrate best practices for running agents."""
        print_section("8. Best Practices")
        
        practices = [
            {
                "title": "🎯 Set Appropriate Recursion Limits",
                "description": "Balance between allowing complex reasoning and preventing infinite loops",
                "example": "Use 2 * expected_steps + 1 as a starting point"
            },
            {
                "title": "🔄 Use Streaming for Long Tasks",
                "description": "Provide user feedback during lengthy agent execution",
                "example": "Enable streaming for tasks taking > 2 seconds"
            },
            {
                "title": "⚡ Leverage Async for Multiple Calls",
                "description": "Use async execution when making multiple agent calls",
                "example": "Process user queries concurrently when possible"
            },
            {
                "title": "📊 Monitor and Log Execution",
                "description": "Track agent performance and tool usage",
                "example": "Log execution time, tool calls, and error rates"
            },
            {
                "title": "🛡️ Implement Robust Error Handling",
                "description": "Handle various failure modes gracefully",
                "example": "Catch recursion errors, tool failures, and timeouts"
            }
        ]
        
        for practice in practices:
            print(f"\n{practice['title']}")
            print(f"   {practice['description']}")
            print(f"   Example: {practice['example']}")
    
    async def run_complete_tutorial(self):
        """Run the complete tutorial demonstration."""
        print("🎬 Starting LangGraph Agents Tutorial: Running Agents")
        print("="*60)
        
        # Run all demonstrations
        self.demonstrate_basic_usage()
        self.demonstrate_input_formats()
        self.demonstrate_output_format()
        self.demonstrate_streaming()
        self.demonstrate_max_iterations()
        await self.demonstrate_async_patterns()
        self.demonstrate_error_handling()
        self.demonstrate_best_practices()
        
        print_section("Tutorial Complete! 🎉")
        print("\n📚 Key Takeaways:")
        print("• Agents support both sync (.invoke) and async (.ainvoke) execution")
        print("• Multiple input formats are supported (string, dict, list)")
        print("• Streaming enables real-time progress updates")
        print("• Recursion limits prevent infinite loops")
        print("• Async execution improves performance for multiple calls")
        print("• Proper error handling is essential for production use")
        
        print("\n🔗 Additional Resources:")
        print("• LangGraph Documentation: https://langchain-ai.github.io/langgraph/")
        print("• Async Programming in LangChain")
        print("• Agent Streaming Guide")


def main():
    """Main function to run the tutorial."""
    tutorial = LangGraphAgentTutorial()
    
    # Run the complete tutorial
    asyncio.run(tutorial.run_complete_tutorial())


if __name__ == "__main__":
    main()