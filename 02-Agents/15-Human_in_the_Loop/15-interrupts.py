# %%
from typing_extensions import TypedDict, NotRequired
from datetime import datetime
import re

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display

################################ State Definition ################################


# Define the central state object that will be passed between nodes in the graph.
class PostState(TypedDict):
    title: str
    body: str
    # Define optional fields to be populated during the workflow.
    flagged_reasons: NotRequired[list[str]]
    approved: NotRequired[bool]
    review_type: NotRequired[str]
    moderator_notes: NotRequired[str]
    published_url: NotRequired[str]
    published_at: NotRequired[str]


################################ Graph Node Definitions ################################


def ingest(state: PostState):
    """A starting node to perform initial data cleaning or normalization."""
    print('---Ingest---')
    # In a real app, normalize input, strip whitespace, etc.
    return {}


def moderate(state: PostState):
    """Check the post content against a set of moderation rules."""
    print('---Moderate---')
    reasons: list[str] = []
    text = f'{state["title"]} {state["body"]}'.lower()

    # Check for banned keywords in the combined text.
    banned_keywords = ['explicit', 'hate', 'violence', 'gambling', 'buy now']
    for kw in banned_keywords:
        if kw in text:
            reasons.append(f'keyword:{kw}')

    # Flag content if it contains any web links.
    if re.search(r'https?://', state['body'], flags=re.I):
        reasons.append('contains_link')

    # Flag content that exceeds the character limit.
    if len(state['body']) > 280:
        reasons.append('length_exceeded')

    print(f'reasons={reasons}')
    return {'flagged_reasons': reasons}


def moderation_route(state: PostState):
    """Route the workflow based on whether the content was flagged."""
    # If any reasons were added, send for human review.
    if state.get('flagged_reasons'):
        return 'human_review'
    # Otherwise, approve it automatically.
    return 'auto_approve'


def human_review(state: PostState):
    """Pause the graph to wait for a manual review decision."""
    print('---Human Review (interrupt)---')
    reasons = ', '.join(state.get('flagged_reasons', [])) or 'none'

    # Use 'interrupt' to pause execution and await external input.
    resp = interrupt(
        f"Content flagged for: {reasons}. Enter 'approve' or 'reject', optionally add notes (e.g., 'approve: allowed for campaign')."
    )

    # Parse the flexible input format from the resume payload.
    decision = None
    notes = ''
    if isinstance(resp, dict):
        decision = str(resp.get('decision', '')).strip().lower()
        notes = str(resp.get('notes', '')).strip()
    else:
        # Handle simple string input or string with notes after a colon.
        txt = str(resp).strip()
        if ':' in txt:
            decision, notes = txt.split(':', 1)
            decision = decision.strip().lower()
            notes = notes.strip()
        else:
            decision = txt.lower()

    # Determine approval status based on common affirmative words.
    approved = decision in ('approve', 'approved', 'yes', 'y')
    return {'approved': approved, 'review_type': 'manual', 'moderator_notes': notes}


def auto_approve(state: PostState):
    """Automatically approve content that passed moderation checks."""
    print('---Auto-Approve---')
    return {'approved': True, 'review_type': 'auto'}


def post_review_route(state: PostState):
    """Route the workflow after a review decision has been made."""
    return 'publish' if state.get('approved') else 'reject'


def _to_slug(s: str) -> str:
    """A helper function to generate a URL-friendly slug from a string."""
    # Replace non-alphanumeric characters with hyphens.
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


def publish(state: PostState):
    """Simulate publishing the approved content."""
    print('---Publish---')
    slug = _to_slug(state['title'])
    url = f'https://example.com/posts/{slug}'
    ts = datetime.utcnow().isoformat()
    print(f'url={url}')
    return {'published_url': url, 'published_at': ts}


def reject(state: PostState):
    """Handle rejected content, e.g., by logging it or notifying the user."""
    print('---Reject---')
    # This node could trigger notifications, log metrics, etc.
    return {}


################################ Graph Construction ################################

# Initialize the state graph builder.
builder = StateGraph(PostState)

# Add all the functions as nodes in the graph.
builder.add_node('ingest', ingest)
builder.add_node('moderate', moderate)
builder.add_node('human_review', human_review)
builder.add_node('auto_approve', auto_approve)
builder.add_node('publish', publish)
builder.add_node('reject', reject)

# Define the workflow's entry point and initial edges.
builder.add_edge(START, 'ingest')
builder.add_edge('ingest', 'moderate')

# Add a conditional edge after moderation to decide the next step.
builder.add_conditional_edges(
    'moderate',
    moderation_route,
    # Map the routing function's output strings to the corresponding node names.
    path_map=['human_review', 'auto_approve'],
)

# Add a conditional edge after human review to publish or reject.
builder.add_conditional_edges(
    'human_review',
    post_review_route,
    path_map=['publish', 'reject'],
)

# Define the path for content that is auto-approved.
builder.add_edge('auto_approve', 'publish')

# Define the terminal nodes of the graph.
builder.add_edge('publish', END)
builder.add_edge('reject', END)

# Set up an in-memory checkpointer to save and resume graph state.
memory = InMemorySaver()

# Compile the graph, enabling state checkpointing.
graph = builder.compile(checkpointer=memory)

# Generate and display a visual representation of the graph.
display(Image(graph.get_graph().draw_mermaid_png()))


################################ Graph Execution & Interaction ################################

# --- Run 1: Content that requires human review ---

# Define an initial input designed to be flagged by the moderation node.
initial_input = {
    'title': 'Limited-time Offer!',
    'body': 'BUY NOW: http://example.com — best deal today!!',
}

# Assign a unique ID to this execution thread for state management.
thread = {'configurable': {'thread_id': 'post-1'}}

# Stream the events from the graph run, which will pause at the interrupt.
for event in graph.stream(initial_input, thread, stream_mode='updates'):
    print(event)
    print()

# --- Resume Run 1: Approve the content ---

# Resume the paused graph by sending a Command with the moderator's decision.
for event in graph.stream(
    Command(resume='approve: allowed for campaign'),  # The payload for the 'interrupt'
    thread,
    stream_mode='values',
):
    print(event)
    print()

# --- Run 2: Safe content that gets auto-approved ---

# Define an input that should pass moderation checks without issue.
safe_input = {
    'title': 'Changelog 2025.08.26',
    'body': 'Minor bug fixes and performance improvements.',
}
# Use a new thread ID for this independent run.
thread2 = {'configurable': {'thread_id': 'post-2'}}

# Stream the events for the second run, which should complete without pausing.
for event in graph.stream(safe_input, thread2, stream_mode='values'):
    print(event)
    print()

# %%
