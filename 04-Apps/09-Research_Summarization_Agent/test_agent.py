"""
Test script for the Research & Summarization Agent.

This script tests the agent with a simple research request.
"""

from research_summarization_agent import create_research_agent, ResearchScope, OutputFormat
from langchain_core.messages import HumanMessage

def test_research_agent():
    """Test the research agent with a simple request."""
    print("🧪 Testing Research & Summarization Agent")
    print("=" * 60)
    
    # Create the agent
    agent = create_research_agent()
    
    # Test cases
    test_cases = [
        "Research the impact of artificial intelligence on healthcare",
        "Find scholarly articles about climate change and provide an executive summary",
        "Give me bullet points about renewable energy from web sources"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}: '{test_case}'")
        print("-" * 60)
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=test_case)],
            "request": None,
            "web_sources": [],
            "scholarly_sources": [],
            "scraped_content": [],
            "key_findings": [],
            "synthesized_information": "",
            "final_report": "",
            "current_step": ""
        }
        
        try:
            # Run the agent
            result = agent.invoke(initial_state)
            
            # Print the final state
            print(f"✅ Final state: {result['current_step']}")
            
            # Print the messages
            for message in result["messages"]:
                print(f"\n🤖 {message.content}")
                
            # Print the final report if available
            if result.get("final_report"):
                print("\n📄 Final Report:")
                print("-" * 60)
                print(result["final_report"])
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_research_agent()