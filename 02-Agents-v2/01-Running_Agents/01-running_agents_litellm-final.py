"""
LangGraph React Agent Tutorial - LiteLLM Version
A comprehensive guide to building and using React agents with LangGraph using LiteLLM
"""

import os
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_litellm import ChatLiteLLM
from rich import print
# Set up your API keys as needed
# For OpenAI: os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# For Anthropic: os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
# For local models like Ollama: no API key needed


class WeatherTool:
    """A simple weather tool for demonstration purposes."""

    @staticmethod
    def get_weather(query: str) -> str:
        """
        Mock weather function that simulates fetching weather data.

        Args:
            query (str): The location to get weather for

        Returns:
            str: Weather information for the location
        """
        print(f"🌤️  Tool called: Fetching weather for {query}")
        # In a real application, this would call a weather API
        weather_data = {
            "san francisco": "sunny with a high of 72°F",
            "new york": "cloudy with a high of 68°F",
            "london": "rainy with a high of 60°F",
            "tokyo": "partly cloudy with a high of 75°F",
        }

        location = query.lower()
        if any(city in location for city in weather_data.keys()):
            for city, weather in weather_data.items():
                if city in location:
                    return f"The weather in {query} is {weather}."

        return f"The weather in {query} is sunny with a high of 72°F."


def create_weather_agent():
    """Create a React agent with weather capabilities using LiteLLM."""
    print("🤖 Creating LangGraph React Agent with LiteLLM...")

    # Create LiteLLM instance - you can switch between different models easily
    llm = ChatLiteLLM(
        model="ollama/phi4-mini",  # Lopenai:gpt-4o-miniocal Ollama model
        api_base="http://100.95.194.160:11434",  # Ollama local server
        # model="gpt-4o-mini",    # OpenAI model
        # model="claude-3-haiku-20240307",  # Anthropic model
        # model="gemini/gemini-pro",  # Google model
        temperature=0.5,
        max_tokens=1000,
        # top_p=0.9,  # Available for some models
    )

    agent = create_react_agent(
        model=llm,  # Use the LiteLLM instance
        tools=[WeatherTool.get_weather],
        debug=True,  # Helpful for tutorials
    )

    print("✅ Agent created successfully with LiteLLM!")
    return agent


def demo_synchronous_invocation(agent):
    """Demonstrate synchronous agent invocation."""
    print("\n" + "=" * 50)
    print("🔄 DEMO 1: Synchronous Invocation")
    print("=" * 50)

    user_message = "What's the weather like in San Francisco?"
    print(f"User: {user_message}")

    response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Extract the final response
    final_message = response["messages"][-1]
    print(f"Agent: {final_message.content}")

    return response


def demo_streaming_output(agent):
    """Demonstrate streaming agent responses."""
    print("\n" + "=" * 50)
    print("🌊 DEMO 2: Streaming Output")
    print("=" * 50)

    user_message = "What's the weather in New York and London?"
    print(f"User: {user_message}")
    print("Agent response (streaming):")

    stream_input = {"messages": [{"role": "user", "content": user_message}]}

    for chunk in agent.stream(stream_input, stream_mode="updates"):
        if chunk:
            # Pretty print streaming updates
            for node_name, node_data in chunk.items():
                if "messages" in node_data:
                    message = node_data["messages"][-1]
                    if hasattr(message, "content") and message.content:
                        print(f"  📝 {node_name}: {message.content[:100]}...")


def demo_input_formats(agent):
    """Demonstrate different input formats."""
    print("\n" + "=" * 50)
    print("📝 DEMO 3: Different Input Formats")
    print("=" * 50)

    # Format 1: Simple string
    print("1. String input:")
    response1 = agent.invoke({"messages": "Hello, I'm interested in weather!"})
    print(f"   Response: {response1['messages'][-1].content[:50]}...")

    # Format 2: Dictionary format
    print("\n2. Dictionary input:")
    response2 = agent.invoke({"messages": {"role": "user", "content": "What can you help me with?"}})
    print(f"   Response: {response2['messages'][-1].content[:50]}...")

    # Format 3: List of messages (conversation history)
    print("\n3. Conversation history:")
    response3 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ]
        }
    )
    print(f"   Response: {response3['messages'][-1].content[:50]}...")


def demo_recursion_limits(agent):
    """Demonstrate recursion limits and error handling."""
    print("\n" + "=" * 50)
    print("🛡️ DEMO 4: Recursion Limits & Safety")
    print("=" * 50)

    # Set a low recursion limit for demonstration
    max_iterations = 2
    recursion_limit = 2 * max_iterations + 1

    print(f"Setting recursion limit to: {recursion_limit}")

    agent_with_limit = agent.with_config(recursion_limit=recursion_limit)

    try:
        response = agent_with_limit.invoke({"messages": [{"role": "user", "content": "What's the weather in SF?"}]})
        print("✅ Agent completed within recursion limit")
        print(f"Final response: {response['messages'][-1].content[:100]}...")

    except GraphRecursionError as e:
        print("⚠️ Agent stopped due to recursion limit")
        print(f"Error: {e}")


def demo_model_switching():
    """Demonstrate how easy it is to switch models with LiteLLM."""
    print("\n" + "=" * 50)
    print("🔄 DEMO 5: Model Switching with LiteLLM")
    print("=" * 50)

    models_to_test = [
        "ollama/llama3.2",  # Local model
        # "gpt-4o-mini",    # Uncomment if you have OpenAI API key
        # "claude-3-haiku-20240307",  # Uncomment if you have Anthropic API key
    ]

    for model_name in models_to_test:
        try:
            print(f"\n🧠 Testing model: {model_name}")

            llm = ChatLiteLLM(model=model_name, temperature=0.5, max_tokens=100)

            agent = create_react_agent(model=llm, tools=[WeatherTool.get_weather], debug=False)

            response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]})

            print(f"✅ {model_name}: {response['messages'][-1].content[:60]}...")

        except Exception as e:
            print(f"❌ {model_name} failed: {e}")


def main():
    """Main tutorial function."""
    print("🎬 Welcome to the LangGraph React Agent Tutorial with LiteLLM!")
    print("=" * 70)

    try:
        # Create the agent
        agent = create_weather_agent()

        # Run all demos
        demo_synchronous_invocation(agent)
        demo_streaming_output(agent)
        demo_input_formats(agent)
        demo_recursion_limits(agent)
        demo_model_switching()

        print("\n" + "=" * 70)
        print("🎉 Tutorial completed successfully!")
        print("📚 Key takeaways:")
        print("   • React agents can use tools to solve complex tasks")
        print("   • LiteLLM makes it easy to switch between different LLM providers")
        print("   • Multiple invocation methods: sync, async, streaming")
        print("   • Flexible input formats for different use cases")
        print("   • Safety features like recursion limits prevent infinite loops")
        print("   • Model switching is as simple as changing the model parameter")

        print("\n🔧 LiteLLM Benefits:")
        print("   • Unified interface for 100+ LLM providers")
        print("   • Easy switching between OpenAI, Anthropic, Google, local models")
        print("   • Cost tracking and logging capabilities")
        print("   • Fallback and retry mechanisms")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("💡 Make sure your chosen model is available!")
        print("   • For Ollama: ensure the model is pulled locally")
        print("   • For cloud providers: set the appropriate API keys")


if __name__ == "__main__":
    main()
