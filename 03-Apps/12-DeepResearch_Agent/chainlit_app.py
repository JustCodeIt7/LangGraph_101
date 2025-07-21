# chainlit_app.py
# This script creates a Chainlit chatbot interface for the LangGraph-based research agent.
# It allows users to input a research topic and optional configurations (report length, acceptable sources).
# The agent runs asynchronously, providing progress updates in the chat.
# Run this with: chainlit run chainlit_app.py

# Required packages (in addition to those in research_agent.py):
# - chainlit==1.1.402 (latest as of 2025-07-21)

import chainlit as cl
from research_agent import ResearchConfig, run_research_agent  # Import from the previous script
import asyncio

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Research Agent Chatbot! Provide a research topic to get started. You can also specify options like: 'report_length: medium' or 'acceptable_sources: edu,gov'.").send()

@cl.on_message
async def main(message: cl.Message):
    # Parse user input for topic and configs
    inputs = message.content.strip()
    topic = inputs  # Default to entire message as topic
    report_length = "medium"
    acceptable_sources = ["all"]
    
    # Simple parsing for configs (e.g., "Topic here | report_length: short | acceptable_sources: edu,gov")
    parts = inputs.split("|")
    if len(parts) > 1:
        topic = parts[0].strip()
        for part in parts[1:]:
            if "report_length:" in part:
                report_length = part.split(":")[1].strip()
            elif "acceptable_sources:" in part:
                sources = part.split(":")[1].strip().split(",")
                acceptable_sources = [s.strip() for s in sources]
    
    if not topic:
        await cl.Message(content="Please provide a valid research topic.").send()
        return
    
    config = ResearchConfig(
        topic=topic,
        report_length=report_length,
        acceptable_sources=acceptable_sources,
        max_api_calls=20  # Default; can be made configurable if needed
    )
    
    # Inform user that research is starting
    await cl.Message(content=f"Starting research on '{topic}' with report length '{report_length}' and sources '{', '.join(acceptable_sources)}'... This may take a few minutes.").send()
    
    # Run the agent asynchronously
    try:
        # Since run_research_agent is sync, wrap in async executor
        report = await cl.make_async(run_research_agent)(config)
        
        # Send the report
        await cl.Message(content="Research complete! Here's the report:").send()
        await cl.Message(content=report).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

# Optional: Add a button or actions if needed, but keeping it simple for now