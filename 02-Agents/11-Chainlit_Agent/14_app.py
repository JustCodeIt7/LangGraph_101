from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import chainlit as cl
import os
from dotenv import load_dotenv

# If you don't have it yet: pip install gnews
from gnews import GNews as GoogleNews  # Alias to match provided tool code


################################## Setup ##################################
load_dotenv()  # Load environment variables from a .env file

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')


############################# Tools Definition ############################
# ----------------------------------
# Tool Definition
# ----------------------------------
@tool
def get_top_news() -> str:
    """Get the top news stories for the current country and language."""
    try:
        gn = GoogleNews(lang='en', country='US')
        top_stories = gn.top_news()
        articles = []
        articles.extend(f'• {entry.title}\n  {entry.link}' for entry in top_stories['entries'][:10])
        return '\n\n'.join(articles) if articles else 'No top news found.'
    except Exception as e:
        return f'Error fetching top news: {e}'


@tool
def get_topic_headlines(topic: str) -> str:
    """Get news headlines for a specific topic.

    Accepted topics: WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT, SCIENCE, SPORTS, HEALTH
    """
    try:
        valid_topics = ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS', 'HEALTH']
        topic_upper = topic.upper()

        if topic_upper not in valid_topics:
            return f'Invalid topic. Please use one of: {", ".join(valid_topics)}'

        gn = GoogleNews(lang='en', country='US')
        headlines = gn.topic_headlines(topic_upper)
        articles = []
        articles.extend(f'• {entry.title}\n  {entry.link}' for entry in headlines['entries'][:10])
        body = '\n\n'.join(articles) if articles else 'No headlines found.'
        return f'Headlines for {topic}:\n\n' + body
    except Exception as e:
        return f'Error fetching topic headlines: {e}'


@tool
def get_geo_headlines(location: str) -> str:
    """Get news headlines for a specific geographic location."""
    try:
        gn = GoogleNews(lang='en', country='US')
        geo_news = gn.geo_headlines(location)
        articles = []
        articles.extend(f'• {entry.title}\n  {entry.link}' for entry in geo_news['entries'][:10])
        body = '\n\n'.join(articles) if articles else 'No local news found.'
        return f'News for {location}:\n\n' + body
    except Exception as e:
        return f'Error fetching geo headlines: {e}'


@tool
def search_news(query: str, when: str = None) -> str:
    """Search for news articles using a custom query.

    Args:
        query: Search terms. Supports Google search operators like:
               - "exact phrase" for phrase search
               - boeing OR airbus for OR search
               - boeing -airbus to exclude terms
               - intitle:boeing to search in titles
               - allintitle:multiple words to search all words in title
        when: Time range filter (e.g., '1h', '12h', '1d', '7d', '1m')
    """
    try:
        gn = GoogleNews(lang='en', country='US')
        search_results = gn.search(query, when=when)

        if not search_results['entries']:
            return f'No news found for query: {query}'

        articles = []
        for entry in search_results['entries'][:10]:  # Limit to top 10
            published = getattr(entry, 'published', 'Unknown date')
            articles.append(f'• {entry.title}\n  Published: {published}\n  {entry.link}')

        time_filter = f' (last {when})' if when else ''
        return f"Search results for '{query}'{time_filter}:\n\n" + '\n\n'.join(articles)
    except Exception as e:
        return f'Error searching news: {e}'


@tool
def search_recent_news(query: str, hours: int = 1) -> str:
    """Search for recent news articles within the last specified hours.

    Args:
        query: Search terms
        hours: Number of hours to look back (1-24)
    """
    try:
        if hours < 1 or hours > 24:
            return 'Hours must be between 1 and 24'

        gn = GoogleNews(lang='en', country='US')
        search_results = gn.search(query, when=f'{hours}h')

        if not search_results['entries']:
            return f'No recent news found for query: {query} in the last {hours} hours'

        articles = []
        for entry in search_results['entries'][:10]:
            published = getattr(entry, 'published', 'Unknown date')
            articles.append(f'• {entry.title}\n  Published: {published}\n  {entry.link}')

        return f"Recent news for '{query}' (last {hours} hours):\n\n" + '\n\n'.join(articles)
    except Exception as e:
        return f'Error searching recent news: {e}'


tools = [get_top_news, get_topic_headlines, get_geo_headlines, search_news, search_recent_news]
tool_node = ToolNode(tools=tools)


############################# Graph Definition ############################
# Initialize the state graph with a schema for storing messages
workflow = StateGraph(state_schema=MessagesState)

# Base LLM
model = ChatOpenAI(
    model='google/gemini-2.5-flash-lite',
    base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
    api_key=openrouter_api_key,
    temperature=0,
)

# LLM bound with tools (enables function/tool calling)
llm_with_tools = model.bind_tools(tools)


def call_model(state: MessagesState):
    """Invokes the model (with tools bound) with the current state and returns the new message."""
    response = llm_with_tools.invoke(state['messages'])
    # Ensure response is appended as a list entry
    return {'messages': [response]}


def should_continue(state: MessagesState):
    """Route to tools if the last AI message requested tool calls, else end."""
    messages = state['messages']
    if not messages:
        return 'end'
    last = messages[-1]
    # OpenAI-compatible tool calls are found on AIMessage.tool_calls
    has_tool_calls = getattr(last, 'tool_calls', None)
    return 'tools' if has_tool_calls else 'end'


# Nodes
workflow.add_node('model', call_model)
workflow.add_node('tools', tool_node)

# Edges
workflow.add_edge(START, 'model')
workflow.add_conditional_edges('model', should_continue, {'tools': 'tools', 'end': END})
workflow.add_edge('tools', 'model')

############################ Graph Compilation ############################
memory = MemorySaver()  # In-memory storage for conversation history

# Compile the graph into a runnable, adding the memory checkpointer
app = workflow.compile(checkpointer=memory)


############################### Chainlit UI ###############################
@cl.on_message
async def main(message: cl.Message):
    """Process incoming user messages and stream back the AI's response."""
    answer = cl.Message(content='')
    await answer.send()

    # Configure the runnable to associate the conversation with the user's session
    config: RunnableConfig = {'configurable': {'thread_id': cl.context.session.thread_id}}

    # Stream the graph's output
    for msg, _ in app.stream(
        {'messages': [HumanMessage(content=message.content)]},
        config,
        stream_mode='messages',
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()
