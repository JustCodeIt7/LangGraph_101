"""
LangGraph Running Agents - Complete Tutorial
===========================================

This tutorial demonstrates how to run LangGraph agents based on the official documentation:
https://langchain-ai.github.io/langgraph/agents/run_agents/

Topics covered:
1. Basic Usage (sync/async)
2. Input/Output Formats
3. Streaming Responses
4. Max Iterations Control
5. Error Handling
6. Real Implementation Examples

YouTube Tutorial: LangGraph Running Agents
"""

import asyncio
import json
import os
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # You'll need to install: pip install langchain-openai

# For environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# TOOLS DEFINITION
# ============================================================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        Weather information as a string
    """
    # Simulated weather data for demo purposes
    weather_data = {
        "san francisco": "Sunny, 72°F (22°C). Light breeze from the west.",
        "new york": "Cloudy, 65°F (18°C). Chance of rain this afternoon.",
        "london": "Rainy, 58°F (14°C). Heavy clouds with light drizzle.",
        "tokyo": "Clear, 70°F (21°C). Perfect day with blue skies.",
        "paris": "Partly cloudy, 68°F (20°C). Gentle morning fog clearing.",
        "sydney": "Sunny, 75°F (24°C). Perfect beach weather!",
        "berlin": "Overcast, 62°F (17°C). Cool with light winds."
    }
    
    location_key = location.lower().strip()
    return weather_data.get(location_key, f"Weather data not available for {location}. Try: San Francisco, New York, London, Tokyo, Paris, Sydney, or Berlin.")


@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The calculation result as a string
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return f"Error: Expression contains invalid characters. Only numbers, +, -, *, /, (, ), and spaces are allowed."
        
        result = eval(expression)
        return f"The result of {expression} is: {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in expression '{expression}'"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current timestamp as a formatted string
    """
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*width}")
    print(f" {title}")
    print(f"{'='*width}")


def print_messages(messages: List[Dict[str, Any]], show_metadata: bool = False) -> None:
    """Print messages in a readable format."""
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content', '')
        
        print(f"\n[{i+1}] {role}:")
        if isinstance(content, str):
            print(f"    {content}")
        else:
            print(f"    {json.dumps(content, indent=4)}")
        
        if show_metadata and 'additional_kwargs' in msg:
            print(f"    Metadata: {msg['additional_kwargs']}")


def simulate_typing(text: str, delay: float = 0.03) -> None:
    """Simulate typing effect for demonstrations."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


# ============================================================================
# AGENT STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Define the agent state structure."""
    messages: Annotated[List[Any], add_messages]
    user_name: Optional[str]
    session_id: Optional[str]
    execution_count: int


# ============================================================================
# MAIN TUTORIAL CLASS
# ============================================================================

class RunningAgentsTutorial:
    """
    Complete tutorial for running LangGraph agents.
    Demonstrates all patterns from the official documentation.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the tutorial with specified model.
        
        Args:
            model_name: The OpenAI model to use (default: gpt-3.5-turbo)
        """
        self.model_name = model_name
        self.tools = [get_weather, calculate, get_current_time]
        
        # Initialize the language model
        # Note: You need to set OPENAI_API_KEY in your environment
        try:
            self.llm = ChatOpenAI(model=model_name, temperature=0)
            self.agent = self._create_agent()
            self.model_available = True
        except Exception as e:
            print(f"⚠️  Model initialization failed: {e}")
            print("   This tutorial will run in demonstration mode without actual LLM calls.")
            self.model_available = False
            self.agent = None
        
        # Setup memory for stateful conversations
        self.memory = MemorySaver()
        
        print(f"🤖 LangGraph Running Agents Tutorial Initialized")
        print(f"📋 Available tools: {[tool.name for tool in self.tools]}")
        print(f"🧠 Model: {model_name} ({'Available' if self.model_available else 'Demo Mode'})")
    
    def _create_agent(self):
        """Create a React agent with tools."""
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory
        )
    
    # ========================================================================
    # 1. BASIC USAGE DEMONSTRATIONS
    # ========================================================================
    
    def demonstrate_basic_usage(self):
        """Demonstrate basic synchronous and asynchronous execution."""
        print_section("1. Basic Usage: Sync vs Async Execution")
        
        print("\n🔄 Synchronous Execution with .invoke()")
        print("=" * 45)
        
        sync_code = '''
# Synchronous execution - blocks until complete
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=tools)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

print(response["messages"][-1]["content"])
        '''
        
        print("Code example:")
        print(sync_code)
        
        if self.model_available:
            try:
                print("\n🚀 Running synchronous example...")
                response = self.agent.invoke({
                    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
                })
                print("\nActual Response:")
                print_messages(response["messages"])
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print("\n🎭 Simulated Response:")
            simulated_response = {
                "messages": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                    {"role": "assistant", "content": "I'll check the weather in San Francisco for you."},
                    {"role": "assistant", "content": "Let me use the weather tool to get current conditions."},
                    {"role": "tool", "name": "get_weather", "content": "get_weather(location='San Francisco')"},
                    {"role": "assistant", "content": "The weather in San Francisco is sunny and 72°F (22°C) with a light breeze from the west."}
                ]
            }
            print_messages(simulated_response["messages"])
        
        print("\n⚡ Asynchronous Execution with await .ainvoke()")
        print("=" * 50)
        
        async_code = '''
# Asynchronous execution - non-blocking
import asyncio

async def run_agent():
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Calculate 15 * 8 + 7"}]
    })
    return response

# Run the async function
response = asyncio.run(run_agent())
        '''
        
        print("Code example:")
        print(async_code)
        
        print("\n💡 Key Differences:")
        print("   • Sync (.invoke): Blocks execution until agent completes")
        print("   • Async (.ainvoke): Allows other operations while agent runs")
        print("   • Use async for better performance with multiple concurrent calls")
    
    async def demonstrate_async_execution(self):
        """Show actual async execution if model is available."""
        if not self.model_available:
            print("\n🎭 Async execution requires a real model - skipping actual demo")
            return
        
        print("\n🚀 Running async example...")
        try:
            response = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": "Calculate 25 * 4 and tell me the current time"}]
            })
            print("\nAsync Response:")
            print_messages(response["messages"])
        except Exception as e:
            print(f"   Error in async execution: {e}")
    
    # ========================================================================
    # 2. INPUT/OUTPUT FORMATS
    # ========================================================================
    
    def demonstrate_input_formats(self):
        """Show all supported input formats."""
        print_section("2. Input Formats - Flexibility in Message Handling")
        
        input_examples = [
            {
                "name": "🔤 String Input (Simplest)",
                "input": {"messages": "Hello, what can you help me with?"},
                "description": "Automatically converted to HumanMessage",
                "use_case": "Quick testing and simple interactions"
            },
            {
                "name": "📝 Message Dictionary",
                "input": {"messages": {"role": "user", "content": "What's 2 + 2?"}},
                "description": "Single message as a dictionary",
                "use_case": "When you need explicit role control"
            },
            {
                "name": "📋 List of Messages",
                "input": {"messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                    {"role": "user", "content": "What's the weather like?"}
                ]},
                "description": "Multi-turn conversation history",
                "use_case": "Continuing existing conversations"
            },
            {
                "name": "⚙️ Custom State Fields",
                "input": {
                    "messages": [{"role": "user", "content": "Hello Alice!"}],
                    "user_name": "Alice",
                    "session_id": "session_123"
                },
                "description": "Additional context and state data",
                "use_case": "Personalized responses and session management"
            },
            {
                "name": "🔧 System + User Messages",
                "input": {"messages": [
                    {"role": "system", "content": "You are a helpful math tutor."},
                    {"role": "user", "content": "Explain multiplication to me."}
                ]},
                "description": "Custom system prompts with user input",
                "use_case": "Role-specific behavior and context setting"
            }
        ]
        
        for i, example in enumerate(input_examples, 1):
            print(f"\n{example['name']}")
            print("-" * 40)
            print(f"Description: {example['description']}")
            print(f"Use case: {example['use_case']}")
            print("Input format:")
            print(json.dumps(example['input'], indent=2))
            
            if i < len(input_examples):
                print()
    
    def demonstrate_output_format(self):
        """Show the structure and content of agent outputs."""
        print_section("3. Output Format - Understanding Agent Responses")
        
        print("🔍 Agent outputs are dictionaries containing:")
        print("   • messages: Complete conversation history")
        print("   • structured_response: (Optional) Parsed structured data")
        print("   • Custom state: (Optional) Additional state fields")
        
        # Example of a complete output structure
        sample_output = {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo and calculate 15 * 8?"
                },
                {
                    "role": "assistant", 
                    "content": "I'll help you with both requests. Let me check the weather in Tokyo and calculate 15 * 8."
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"name": "get_weather", "args": {"location": "Tokyo"}, "id": "call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "content": "Clear, 70°F (21°C). Perfect day with blue skies.",
                    "tool_call_id": "call_1"
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"name": "calculate", "args": {"expression": "15 * 8"}, "id": "call_2"}
                    ]
                },
                {
                    "role": "tool",
                    "content": "The result of 15 * 8 is: 120",
                    "tool_call_id": "call_2"
                },
                {
                    "role": "assistant",
                    "content": "Great! Here are your results:\n\n🌤️ **Weather in Tokyo**: Clear, 70°F (21°C). Perfect day with blue skies.\n\n🔢 **Calculation**: 15 × 8 = 120\n\nIs there anything else you'd like to know?"
                }
            ]
        }
        
        print(f"\n📤 Complete Output Structure Example:")
        print(json.dumps(sample_output, indent=2))
        
        print(f"\n🎯 Key Points About Output:")
        print("   • Messages include full conversation flow")
        print("   • Tool calls and responses are preserved")
        print("   • Final assistant message contains the complete response")
        print("   • Each message has a role: 'user', 'assistant', or 'tool'")
        print("   • Tool calls include function name, arguments, and unique IDs")
    
    # ========================================================================
    # 4. STREAMING DEMONSTRATIONS
    # ========================================================================
    
    def demonstrate_streaming(self):
        """Show streaming capabilities with different modes."""
        print_section("4. Streaming Output - Real-time Progress Updates")
        
        print("🌊 Streaming provides:")
        print("   • Real-time progress updates")
        print("   • LLM tokens as they're generated")
        print("   • Tool execution notifications")
        print("   • Better user experience for long-running tasks")
        
        print("\n📝 Synchronous Streaming:")
        sync_streaming_code = '''
# Stream with progress updates
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in London?"}]},
    stream_mode="updates"  # Options: "updates", "values", "debug"
):
    print(f"Step update: {chunk}")
        '''
        print(sync_streaming_code)
        
        print("\n📝 Asynchronous Streaming:")
        async_streaming_code = '''
# Async streaming for concurrent operations
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "Calculate 100 * 25"}]},
    stream_mode="values"
):
    if "messages" in chunk:
        latest_message = chunk["messages"][-1]
        print(f"Latest: {latest_message}")
        '''
        print(async_streaming_code)
        
        print("\n🔄 Stream Modes Explained:")
        stream_modes = [
            {
                "mode": "updates",
                "description": "Shows incremental changes after each step",
                "use_case": "Progress tracking and debugging"
            },
            {
                "mode": "values", 
                "description": "Shows complete state after each step",
                "use_case": "Getting full context at each stage"
            },
            {
                "mode": "debug",
                "description": "Detailed execution information",
                "use_case": "Development and troubleshooting"
            }
        ]
        
        for mode_info in stream_modes:
            print(f"   • {mode_info['mode']}: {mode_info['description']}")
            print(f"     Use case: {mode_info['use_case']}")
        
        # Simulate streaming output
        if self.model_available:
            print(f"\n🚀 Live Streaming Demo:")
            self._demonstrate_live_streaming()
        else:
            print(f"\n🎭 Simulated Streaming Demo:")
            self._simulate_streaming()
    
    def _simulate_streaming(self):
        """Simulate streaming output for demonstration."""
        streaming_chunks = [
            {"node": "agent", "update": "Processing user request..."},
            {"node": "agent", "update": "Analyzing: 'weather in Paris'"},
            {"node": "tools", "update": "Calling get_weather tool..."},
            {"node": "tools", "update": "Tool execution complete"},
            {"node": "agent", "update": "Generating response..."},
            {"node": "agent", "update": "Response complete ✅"}
        ]
        
        print("   Streaming output:")
        for i, chunk in enumerate(streaming_chunks, 1):
            print(f"   [{i}] {chunk['node']}: {chunk['update']}")
            time.sleep(0.8)  # Simulate real-time updates
    
    def _demonstrate_live_streaming(self):
        """Show actual streaming if model is available."""
        try:
            print("   Starting live stream...")
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": "What's the current time and weather in Berlin?"}]},
                stream_mode="updates"
            ):
                print(f"   Stream chunk: {chunk}")
        except Exception as e:
            print(f"   Streaming error: {e}")
    
    # ========================================================================
    # 5. MAX ITERATIONS AND RECURSION CONTROL
    # ========================================================================
    
    def demonstrate_max_iterations(self):
        """Show how to control agent execution limits."""
        print_section("5. Max Iterations - Preventing Infinite Loops")
        
        print("🔄 Recursion limits control agent execution steps:")
        print("   • Prevents infinite loops in complex reasoning")
        print("   • Balances thoroughness with performance")
        print("   • Essential for production environments")
        
        print(f"\n📐 Calculation Formula:")
        print("   recursion_limit = 2 × max_iterations + 1")
        print("   • Each iteration includes: thought → action → observation")
        print("   • +1 accounts for the final response generation")
        
        # Practical examples with different limits
        examples = [
            {"max_iter": 2, "limit": 5, "use_case": "Simple single-tool tasks"},
            {"max_iter": 5, "limit": 11, "use_case": "Multi-step reasoning"},
            {"max_iter": 10, "limit": 21, "use_case": "Complex problem solving"}
        ]
        
        print(f"\n📊 Common Configurations:")
        for ex in examples:
            print(f"   • Max iterations: {ex['max_iter']}, Limit: {ex['limit']} → {ex['use_case']}")
        
        print(f"\n📝 Runtime Configuration:")
        runtime_config_code = '''
from langgraph.errors import GraphRecursionError

max_iterations = 3
recursion_limit = 2 * max_iterations + 1  # = 7

try:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Complex multi-step task"}]},
        config={"recursion_limit": recursion_limit}
    )
    print("✅ Task completed successfully")
except GraphRecursionError:
    print("⚠️ Agent stopped - reached max iterations")
    # Implement fallback logic here
        '''
        print(runtime_config_code)
        
        print(f"\n📝 Pre-configured Agent:")
        preconfig_code = '''
# Set limit when creating the agent
agent_with_limit = agent.with_config(recursion_limit=7)

try:
    response = agent_with_limit.invoke({
        "messages": [{"role": "user", "content": "Your question here"}]
    })
except GraphRecursionError:
    print("Agent exceeded maximum allowed steps")
        '''
        print(preconfig_code)
        
        # Demonstrate the practical impact
        if self.model_available:
            self._demonstrate_recursion_limits()
        else:
            self._simulate_recursion_scenarios()
    
    def _simulate_recursion_scenarios(self):
        """Simulate different recursion limit scenarios."""
        print(f"\n🎭 Recursion Limit Scenarios:")
        
        scenarios = [
            {
                "task": "Simple weather query",
                "expected_steps": 1,
                "limit": 3,
                "result": "✅ Completes successfully"
            },
            {
                "task": "Multi-city weather comparison",
                "expected_steps": 3,
                "limit": 7,
                "result": "✅ Completes with multiple tool calls"
            },
            {
                "task": "Complex calculation chain",
                "expected_steps": 8,
                "limit": 5,
                "result": "⚠️ Hits recursion limit"
            }
        ]
        
        for scenario in scenarios:
            print(f"   Task: {scenario['task']}")
            print(f"   Expected steps: {scenario['expected_steps']}")
            print(f"   Recursion limit: {scenario['limit']}")
            print(f"   Result: {scenario['result']}")
            print()
    
    def _demonstrate_recursion_limits(self):
        """Show actual recursion limit behavior."""
        print(f"\n🚀 Testing recursion limits...")
        
        # Test with a low limit
        low_limit_agent = self.agent.with_config(recursion_limit=3)
        
        try:
            response = low_limit_agent.invoke({
                "messages": [{"role": "user", "content": "Get weather for 3 cities and calculate their average temperature"}]
            })
            print("✅ Task completed within limit")
        except GraphRecursionError:
            print("⚠️ Hit recursion limit - task too complex for current setting")
        except Exception as e:
            print(f"   Other error: {e}")
    
    # ========================================================================
    # 6. ERROR HANDLING PATTERNS
    # ========================================================================
    
    def demonstrate_error_handling(self):
        """Show comprehensive error handling strategies."""
        print_section("6. Error Handling - Robust Agent Operations")
        
        print("🛡️ Common error scenarios and handling strategies:")
        
        error_patterns = [
            {
                "error": "GraphRecursionError",
                "cause": "Agent exceeds recursion limit",
                "handling": "Catch and provide fallback response",
                "prevention": "Set appropriate limits for task complexity"
            },
            {
                "error": "Tool Execution Failure",
                "cause": "Tool raises exception during execution",
                "handling": "Graceful degradation and user notification",
                "prevention": "Validate tool inputs and add error handling"
            },
            {
                "error": "Model API Errors",
                "cause": "LLM service unavailable or rate limited",
                "handling": "Retry with exponential backoff",
                "prevention": "Monitor usage and implement circuit breakers"
            },
            {
                "error": "Invalid Input Format",
                "cause": "Malformed input data",
                "handling": "Validate and sanitize inputs",
                "prevention": "Input validation schemas"
            }
        ]
        
        for i, pattern in enumerate(error_patterns, 1):
            print(f"\n{i}. {pattern['error']}")
            print(f"   Cause: {pattern['cause']}")
            print(f"   Handling: {pattern['handling']}")
            print(f"   Prevention: {pattern['prevention']}")
        
        print(f"\n📝 Comprehensive Error Handling Example:")
        error_handling_code = '''
import logging
from langgraph.errors import GraphRecursionError
from typing import Dict, Any, Optional

class RobustAgentRunner:
    def __init__(self, agent, max_retries: int = 3):
        self.agent = agent
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    def safe_invoke(self, input_data: Dict[str, Any], 
                   recursion_limit: int = 10) -> Optional[Dict[str, Any]]:
        """Safely invoke agent with comprehensive error handling."""
        
        # Input validation
        if not self._validate_input(input_data):
            return {"error": "Invalid input format"}
        
        for attempt in range(self.max_retries):
            try:
                response = self.agent.invoke(
                    input_data,
                    config={"recursion_limit": recursion_limit}
                )
                return response
                
            except GraphRecursionError:
                self.logger.warning(f"Recursion limit hit on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    return self._fallback_response(input_data)
                # Try with higher limit
                recursion_limit = min(recursion_limit * 1.5, 50)
                
            except Exception as e:
                self.logger.error(f"Agent error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Agent failed after {self.max_retries} attempts"}
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {"error": "Max retries exceeded"}
    
    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input format."""
        return "messages" in input_data and input_data["messages"]
    
    def _fallback_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback when agent fails."""
        return {
            "messages": input_data["messages"] + [{
                "role": "assistant",
                "content": "I apologize, but this task is too complex for me to complete right now. Could you try breaking it into smaller parts?"
            }],
            "fallback_used": True
        }

# Usage
robust_runner = RobustAgentRunner(agent)
result = robust_runner.safe_invoke({
    "messages": [{"role": "user", "content": "Your complex task here"}]
})
        '''
        print(error_handling_code)
    
    # ========================================================================
    # 7. ADVANCED PATTERNS AND BEST PRACTICES
    # ========================================================================
    
    def demonstrate_best_practices(self):
        """Show advanced patterns and production best practices."""
        print_section("7. Best Practices - Production-Ready Agents")
        
        practices = [
            {
                "category": "🎯 Execution Control",
                "practices": [
                    "Set conservative recursion limits initially (5-10)",
                    "Monitor execution time and adjust limits dynamically",
                    "Implement circuit breakers for failed operations",
                    "Use timeouts for long-running agent calls"
                ]
            },
            {
                "category": "⚡ Performance Optimization", 
                "practices": [
                    "Use async execution for concurrent agent calls",
                    "Enable streaming for tasks taking > 2 seconds",
                    "Cache frequently used tool results",
                    "Batch similar requests when possible"
                ]
            },
            {
                "category": "🔍 Monitoring & Observability",
                "practices": [
                    "Log all agent executions with timing",
                    "Track tool usage patterns and success rates",
                    "Monitor recursion limit hits and adjust",
                    "Set up alerts for high error rates"
                ]
            },
            {
                "category": "🛡️ Safety & Security",
                "practices": [
                    "Validate and sanitize all inputs",
                    "Implement proper authentication for agent access",
                    "Use structured outputs to prevent injection",
                    "Rate limit agent calls per user/session"
                ]
            },
            {
                "category": "💾 State Management",
                "practices": [
                    "Use persistent checkpointers for long conversations",
                    "Implement session cleanup and TTL",
                    "Handle state corruption gracefully",
                    "Version your agent configurations"
                ]
            }
        ]
        
        for practice_group in practices:
            print(f"\n{practice_group['category']}")
            print("-" * 40)
            for practice in practice_group['practices']:
                print(f"   • {practice}")
        
        print(f"\n📝 Production-Ready Agent Setup:")
        production_code = '''
import logging
from datetime import datetime, timedelta
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class ProductionAgent:
    def __init__(self, model, tools, config=None):
        self.logger = self._setup_logging()
        self.config = config or self._default_config()
        
        # Agent setup with persistence
        self.checkpointer = MemorySaver()
        self.agent = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=self.checkpointer
        )
        
        # Metrics tracking
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_execution_time": 0
        }
    
    async def execute(self, input_data, session_id=None):
        """Execute agent with full production features."""
        start_time = datetime.now()
        self.metrics["total_calls"] += 1
        
        try:
            # Add session tracking
            config = {"configurable": {"thread_id": session_id or "default"}}
            
            # Execute with monitoring
            response = await self.agent.ainvoke(
                input_data,
                config={**config, "recursion_limit": self.config["recursion_limit"]}
            )
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(execution_time, success=True)
            
            # Log successful execution
            self.logger.info(f"Agent execution successful in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.metrics["failed_calls"] += 1
            self.logger.error(f"Agent execution failed: {e}")
            return self._error_response(str(e))
    
    def _setup_logging(self):
        """Configure structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f"{__name__}.ProductionAgent")
    
    def _default_config(self):
        """Default production configuration."""
        return {
            "recursion_limit": 10,
            "timeout_seconds": 30,
            "max_retries": 3
        }
    
    def _update_metrics(self, execution_time, success=True):
        """Update performance metrics."""
        if success:
            self.metrics["successful_calls"] += 1
        
        # Update average execution time
        total_successful = self.metrics["successful_calls"]
        current_avg = self.metrics["avg_execution_time"]
        self.metrics["avg_execution_time"] = (
            (current_avg * (total_successful - 1) + execution_time) / total_successful
        )
    
    def get_health_status(self):
        """Get agent health and performance metrics."""
        total = self.metrics["total_calls"]
        if total == 0:
            return {"status": "No calls yet", "metrics": self.metrics}
        
        success_rate = (self.metrics["successful_calls"] / total) * 100
        
        return {
            "status": "healthy" if success_rate > 95 else "degraded",
            "success_rate": f"{success_rate:.1f}%",
            "metrics": self.metrics
        }

# Usage
agent = ProductionAgent(model=your_model, tools=your_tools)
response = await agent.execute({"messages": [{"role": "user", "content": "Hello!"}]})
health = agent.get_health_status()
        '''
        print(production_code)
    
    # ========================================================================
    # 8. COMPLETE TUTORIAL RUNNER
    # ========================================================================
    
    async def run_complete_tutorial(self):
        """Run the complete tutorial demonstration."""
        print("🎬 LangGraph Running Agents - Complete Tutorial")
        print("=" * 60)
        print(f"📖 Based on: https://langchain-ai.github.io/langgraph/agents/run_agents/")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tutorial sections
        sections = [
            ("Basic Usage", self.demonstrate_basic_usage),
            ("Input Formats", self.demonstrate_input_formats),
            ("Output Format", self.demonstrate_output_format),
            ("Streaming", self.demonstrate_streaming),
            ("Max Iterations", self.demonstrate_max_iterations),
            ("Error Handling", self.demonstrate_error_handling),
            ("Best Practices", self.demonstrate_best_practices)
        ]
        
        for section_name, section_func in sections:
            try:
                if asyncio.iscoroutinefunction(section_func):
                    await section_func()
                else:
                    section_func()
                print(f"\n✅ {section_name} section completed")
            except Exception as e:
                print(f"\n❌ Error in {section_name} section: {e}")
        
        # Run async demonstration if available
        if self.model_available:
            print_section("Async Execution Demo")
            await self.demonstrate_async_execution()
        
        # Tutorial summary
        print_section("🎉 Tutorial Complete!")
        
        print("\n📚 Key Takeaways:")
        takeaways = [
            "Agents support both .invoke() (sync) and .ainvoke() (async) execution",
            "Multiple input formats: string, dict, list, with custom state",
            "Streaming enables real-time progress with different stream modes",
            "Recursion limits prevent infinite loops: limit = 2 × iterations + 1",
            "Comprehensive error handling is essential for production",
            "Async execution dramatically improves performance for multiple calls",
            "Production agents need monitoring, logging, and safety measures"
        ]
        
        for i, takeaway in enumerate(takeaways, 1):
            print(f"   {i}. {takeaway}")
        
        print(f"\n🔗 Additional Resources:")
        resources = [
            "LangGraph Documentation: https://langchain-ai.github.io/langgraph/",
            "Agent Streaming Guide: https://langchain-ai.github.io/langgraph/agents/streaming/",
            "Error Handling Patterns: https://python.langchain.com/docs/concepts/async",
            "Production Deployment: https://langchain-ai.github.io/langgraph/tutorials/deployment/"
        ]
        
        for resource in resources:
            print(f"   • {resource}")
        
        print(f"\n⭐ Tutorial completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete tutorial."""
    print("🚀 Starting LangGraph Running Agents Tutorial")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OpenAI API key found in environment")
        print("   Set OPENAI_API_KEY to enable live agent demonstrations")
        print("   Tutorial will run in demonstration mode\n")
    
    # Initialize and run tutorial
    tutorial = RunningAgentsTutorial()
    
    # Run the complete tutorial
    asyncio.run(tutorial.run_complete_tutorial())


if __name__ == "__main__":
    main()