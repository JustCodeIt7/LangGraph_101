#!/usr/bin/env python3
"""
Content Creation Pipeline Agent
Using LangGraph and Ollama with llama3.2
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# State management
class ContentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    request_type: str
    target_audience: str
    existing_content: Optional[str]
    research_data: Optional[str]
    final_output: Optional[str]

# Content request types
class ContentType(Enum):
    BLOG_OUTLINE = "generate_blog_outline"
    VIDEO_SCRIPT = "generate_video_script_outline"
    SOCIAL_MEDIA = "adapt_for_social_media"
    NEWSLETTER = "curate_newsletter"

class ContentCreationAgent:
    def __init__(self):
        self.llm = Ollama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        self.graph = self._create_graph()

    def _create_graph(self) -> Graph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ContentState)
        
        # Add nodes
        workflow.add_node("parse_request", self._parse_request)
        workflow.add_node("research_topic", self._research_topic)
        workflow.add_node("generate_blog_outline", self._generate_blog_outline)
        workflow.add_node("generate_video_script", self._generate_video_script)
        workflow.add_node("adapt_social_media", self._adapt_social_media)
        workflow.add_node("curate_newsletter", self._curate_newsletter)
        
        # Set entry point
        workflow.set_entry_point("parse_request")
        
        # Add edges based on request type
        workflow.add_conditional_edges(
            "parse_request",
            self._route_request,
            {
                ContentType.BLOG_OUTLINE.value: "research_topic",
                ContentType.VIDEO_SCRIPT.value: "research_topic", 
                ContentType.SOCIAL_MEDIA.value: "adapt_social_media",
                ContentType.NEWSLETTER.value: "research_topic",
                "unknown": END
            }
        )
        
        # Route from research to specific generators
        workflow.add_conditional_edges(
            "research_topic",
            self._route_after_research,
            {
                ContentType.BLOG_OUTLINE.value: "generate_blog_outline",
                ContentType.VIDEO_SCRIPT.value: "generate_video_script",
                ContentType.NEWSLETTER.value: "curate_newsletter"
            }
        )
        
        # All generators end the workflow
        workflow.add_edge("generate_blog_outline", END)
        workflow.add_edge("generate_video_script", END)
        workflow.add_edge("adapt_social_media", END)
        workflow.add_edge("curate_newsletter", END)
        
        return workflow.compile()

    def _parse_request(self, state: ContentState) -> ContentState:
        """Parse the user's content creation request"""
        last_message = state["messages"][-1].content
        
        # Simple parsing - in production, use more sophisticated NLP
        if "blog" in last_message.lower() or "outline" in last_message.lower():
            state["request_type"] = ContentType.BLOG_OUTLINE.value
        elif "video" in last_message.lower() or "script" in last_message.lower():
            state["request_type"] = ContentType.VIDEO_SCRIPT.value
        elif "social" in last_message.lower() or "twitter" in last_message.lower() or "linkedin" in last_message.lower():
            state["request_type"] = ContentType.SOCIAL_MEDIA.value
        elif "newsletter" in last_message.lower():
            state["request_type"] = ContentType.NEWSLETTER.value
        else:
            state["request_type"] = "unknown"
            
        # Extract topic (simplified)
        words = last_message.split()
        if "about" in words:
            topic_start = words.index("about") + 1
            state["topic"] = " ".join(words[topic_start:topic_start+3])
        else:
            state["topic"] = " ".join(words[-3:])  # Last 3 words as fallback
            
        state["target_audience"] = "general"  # Default audience
        
        return state

    def _route_request(self, state: ContentState) -> str:
        """Route based on request type"""
        return state.get("request_type", "unknown")

    def _route_after_research(self, state: ContentState) -> str:
        """Route after research is complete"""
        return state["request_type"]

    def _research_topic(self, state: ContentState) -> ContentState:
        """Research the given topic"""
        topic = state["topic"]
        
        research_prompt = f"""
        Research the topic: {topic}
        
        Provide:
        1. Key trends and insights
        2. Important statistics or data points
        3. Expert opinions or perspectives
        4. Current market context
        5. Audience interests and pain points
        
        Keep the research concise but comprehensive.
        """
        
        try:
            research_data = self.llm.invoke(research_prompt)
            state["research_data"] = research_data
            state["messages"].append(AIMessage(content=f"Research completed for topic: {topic}"))
        except Exception as e:
            state["research_data"] = f"Basic research on {topic} - no external data available"
            state["messages"].append(AIMessage(content=f"Research note: {str(e)}"))
        
        return state

    def _generate_blog_outline(self, state: ContentState) -> ContentState:
        """Generate a blog post outline"""
        topic = state["topic"]
        research = state.get("research_data", "")
        
        outline_prompt = f"""
        Create a detailed blog post outline for: {topic}
        
        Research context: {research}
        
        Format the outline with:
        1. Compelling headline
        2. Introduction hook
        3. 3-5 main sections with subpoints
        4. Conclusion
        5. Call-to-action
        
        Also suggest 5-7 SEO keywords related to this topic.
        """
        
        try:
            outline = self.llm.invoke(outline_prompt)
            state["final_output"] = outline
            state["messages"].append(AIMessage(content=outline))
        except Exception as e:
            state["final_output"] = f"Error generating blog outline: {str(e)}"
            state["messages"].append(AIMessage(content=state["final_output"]))
        
        return state

    def _generate_video_script(self, state: ContentState) -> ContentState:
        """Generate a video script outline"""
        topic = state["topic"]
        research = state.get("research_data", "")
        
        script_prompt = f"""
        Create a video script outline for: {topic}
        
        Research context: {research}
        
        Structure the script with:
        1. Hook (first 15 seconds)
        2. Introduction and preview
        3. Main content sections (3-4 key points)
        4. Visual suggestions for each section
        5. Strong conclusion and CTA
        
        Include timing estimates and engagement techniques.
        """
        
        try:
            script = self.llm.invoke(script_prompt)
            state["final_output"] = script
            state["messages"].append(AIMessage(content=script))
        except Exception as e:
            state["final_output"] = f"Error generating video script: {str(e)}"
            state["messages"].append(AIMessage(content=state["final_output"]))
        
        return state

    def _adapt_social_media(self, state: ContentState) -> ContentState:
        """Adapt content for social media platforms"""
        topic = state["topic"]
        existing_content = state.get("existing_content", "")
        
        if not existing_content:
            # Generate basic social media content
            social_prompt = f"""
            Create social media posts for the topic: {topic}
            
            Generate 3 different versions:
            1. Twitter/X post (280 characters max, include relevant hashtags)
            2. LinkedIn post (professional tone, 1-2 paragraphs)
            3. Instagram caption (engaging, visual-friendly with emojis)
            
            Make each post engaging and platform-appropriate.
            """
        else:
            social_prompt = f"""
            Adapt this content for social media: {existing_content}
            
            Create versions for:
            1. Twitter/X (concise, hashtags)
            2. LinkedIn (professional)
            3. Instagram (visual, engaging)
            
            Topic: {topic}
            """
        
        try:
            social_content = self.llm.invoke(social_prompt)
            state["final_output"] = social_content
            state["messages"].append(AIMessage(content=social_content))
        except Exception as e:
            state["final_output"] = f"Error adapting for social media: {str(e)}"
            state["messages"].append(AIMessage(content=state["final_output"]))
        
        return state

    def _curate_newsletter(self, state: ContentState) -> ContentState:
        """Curate a newsletter"""
        topic = state["topic"]
        research = state.get("research_data", "")
        
        newsletter_prompt = f"""
        Create a newsletter draft for the topic: {topic}
        
        Research context: {research}
        
        Include:
        1. Engaging subject line
        2. Personal introduction
        3. 3-5 curated content sections with:
           - Brief summary
           - Why it matters
           - Key takeaway
        4. Community spotlight or tip
        5. Call-to-action
        6. Sign-off
        
        Keep it concise but valuable.
        """
        
        try:
            newsletter = self.llm.invoke(newsletter_prompt)
            state["final_output"] = newsletter
            state["messages"].append(AIMessage(content=newsletter))
        except Exception as e:
            state["final_output"] = f"Error curating newsletter: {str(e)}"
            state["messages"].append(AIMessage(content=state["final_output"]))
        
        return state

    def process_request(self, user_message: str) -> str:
        """Process a content creation request"""
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "topic": "",
            "request_type": "",
            "target_audience": "",
            "existing_content": None,
            "research_data": None,
            "final_output": None
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        return result.get("final_output", "Sorry, I couldn't process your request.")

def main():
    """Main CLI interface"""
    print("🎯 Content Creation Pipeline Agent")
    print("Powered by LangGraph + Ollama (llama3.2)")
    print("-" * 50)
    print("I can help you with:")
    print("• Blog post outlines")
    print("• Video script outlines") 
    print("• Social media content adaptation")
    print("• Newsletter curation")
    print("-" * 50)
    
    agent = ContentCreationAgent()
    
    while True:
        try:
            user_input = input("\n💬 What content would you like me to help you create? (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
                
            print("\n🔄 Processing your request...")
            result = agent.process_request(user_input)
            print("\n📝 Content Created:")
            print("=" * 60)
            print(result)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different request.")

if __name__ == "__main__":
    main()