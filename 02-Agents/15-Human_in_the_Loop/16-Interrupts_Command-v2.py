"""
LangGraph Tutorial: Interrupts and Commands
===========================================

This tutorial demonstrates two key LangGraph features:
1. **Interrupts**: Pause execution to wait for human input
2. **Commands**: Dynamically control graph flow and state updates

We'll build a simple content moderation system that showcases both concepts.
"""

from typing_extensions import TypedDict, NotRequired, Literal
from datetime import datetime
import re

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# ===============================================================================
# PART 1: STATE DEFINITION
# ===============================================================================


class ContentState(TypedDict):
    """Central state object passed between nodes in the graph."""

    content: str
    user_id: str
    # Optional fields populated during workflow
    flagged_reasons: NotRequired[list[str]]
    approved: NotRequired[bool]
    review_type: NotRequired[str]
    moderator_notes: NotRequired[str]
    final_status: NotRequired[str]


# ===============================================================================
# PART 2: INTERRUPT EXAMPLE - Human-in-the-Loop Moderation
# ===============================================================================


def initial_check(state: ContentState):
    """Initial content validation."""
    print(f'🔍 Initial Check - User: {state["user_id"]}')
    print(f"   Content: '{state['content'][:50]}...'")
    return {}


def moderate_content(state: ContentState):
    """Analyze content and flag potential issues."""
    print('🛡️  Moderating Content')

    reasons = []
    content_lower = state['content'].lower()

    # Simple moderation rules
    banned_words = ['spam', 'inappropriate', 'harmful']
    for word in banned_words:
        if word in content_lower:
            reasons.append(f'banned_word:{word}')

    # Check for excessive caps
    if sum(1 for c in state['content'] if c.isupper()) > len(state['content']) * 0.7:
        reasons.append('excessive_caps')

    # Check length
    if len(state['content']) > 500:
        reasons.append('too_long')

    print(f'   Flagged reasons: {reasons}')
    return {'flagged_reasons': reasons}


def content_router(state: ContentState) -> Command[Literal['human_review', 'auto_approve']]:
    """Route based on moderation results using Command."""
    print('🔀 Routing Decision')

    if state.get('flagged_reasons'):
        print('   → Sending to human review')
        return Command(goto='human_review')
    else:
        print('   → Auto-approving')
        return Command(goto='auto_approve')


def human_review(state: ContentState):
    """Demonstrate interrupt - pause for human decision."""
    print('👨‍💼 Human Review Required (INTERRUPT)')

    reasons = ', '.join(state.get('flagged_reasons', []))
    prompt = (
        f'Content flagged for: {reasons}\n'
        f"Content: '{state['content']}'\n"
        f'User: {state["user_id"]}\n\n'
        'Decision? (approve/reject) [optional notes after colon]'
    )

    # This pauses execution until external input is provided
    response = interrupt(prompt)

    # Parse response
    decision = 'reject'  # default
    notes = ''

    if isinstance(response, dict):
        decision = str(response.get('decision', 'reject')).lower()
        notes = str(response.get('notes', ''))
    else:
        response_str = str(response).strip()
        if ':' in response_str:
            decision, notes = response_str.split(':', 1)
            decision = decision.strip().lower()
            notes = notes.strip()
        else:
            decision = response_str.lower()

    approved = decision in ('approve', 'approved', 'yes', 'y')

    print(f'   Decision: {"APPROVED" if approved else "REJECTED"}')
    if notes:
        print(f'   Notes: {notes}')

    return {'approved': approved, 'review_type': 'manual', 'moderator_notes': notes}


def auto_approve(state: ContentState) -> Command[Literal['finalize']]:
    """Auto-approve clean content using Command."""
    print('✅ Auto-Approved')

    return Command(update={'approved': True, 'review_type': 'automatic', 'final_status': 'published'}, goto='finalize')


def approval_router(state: ContentState) -> Command[Literal['finalize']]:
    """Route after human review using Command."""
    status = 'published' if state.get('approved') else 'rejected'

    return Command(update={'final_status': status}, goto='finalize')


def finalize(state: ContentState):
    """Final processing step."""
    print('🏁 Finalizing')

    status = state.get('final_status', 'unknown')
    review_type = state.get('review_type', 'unknown')

    print(f'   Final Status: {status}')
    print(f'   Review Type: {review_type}')

    if state.get('moderator_notes'):
        print(f'   Notes: {state["moderator_notes"]}')

    return {}


# ===============================================================================
# PART 3: GRAPH CONSTRUCTION
# ===============================================================================


def create_moderation_graph():
    """Build the complete moderation graph."""

    builder = StateGraph(ContentState)

    # Add all nodes
    builder.add_node('initial_check', initial_check)
    builder.add_node('moderate_content', moderate_content)
    builder.add_node('human_review', human_review)
    builder.add_node('auto_approve', auto_approve)
    builder.add_node('finalize', finalize)

    # Define the flow
    builder.add_edge(START, 'initial_check')
    builder.add_edge('initial_check', 'moderate_content')

    # After moderation, route using Command (no explicit edges needed)
    builder.add_node('content_router', content_router)
    builder.add_edge('moderate_content', 'content_router')

    # After human review, route using Command
    builder.add_node('approval_router', approval_router)
    builder.add_edge('human_review', 'approval_router')

    # End connections
    builder.add_edge('finalize', END)

    return builder.compile(checkpointer=InMemorySaver())


# ===============================================================================
# PART 4: DEMONSTRATION
# ===============================================================================


def main():
    """Run the tutorial demonstrations."""

    print('=' * 60)
    print('LANGGRAPH TUTORIAL: INTERRUPTS & COMMANDS')
    print('=' * 60)

    graph = create_moderation_graph()

    # ===== DEMO 1: Content requiring human review (INTERRUPT) =====
    print('\n' + '=' * 40)
    print('DEMO 1: Content Requiring Human Review')
    print('=' * 40)

    flagged_content = {'content': 'This is SPAM content that is inappropriate!', 'user_id': 'user123'}

    thread1 = {'configurable': {'thread_id': 'content-1'}}

    print('\n🚀 Starting graph execution...')

    # This will pause at the interrupt
    for event in graph.stream(flagged_content, thread1, stream_mode='updates'):
        for node_name, node_output in event.items():
            if node_output:
                print(f'📋 {node_name}: {node_output}')

    print('\n⏸️  Graph paused for human input!')
    print("💡 To resume, call: graph.stream(Command(resume='approve: looks ok'), thread1)")

    # ===== DEMO 2: Auto-approved content (COMMAND routing) =====
    print('\n' + '=' * 40)
    print('DEMO 2: Clean Content (Auto-Approved)')
    print('=' * 40)

    clean_content = {'content': 'This is a perfectly normal and clean message.', 'user_id': 'user456'}

    thread2 = {'configurable': {'thread_id': 'content-2'}}

    print('\n🚀 Starting graph execution...')

    # This should complete without interruption
    for event in graph.stream(clean_content, thread2, stream_mode='values'):
        print(f'📋 Current state: {event}')

    # ===== DEMO 3: Resume the interrupted graph =====
    print('\n' + '=' * 40)
    print('DEMO 3: Resuming Interrupted Graph')
    print('=' * 40)

    print('🔄 Resuming with approval decision...')

    # Resume the first graph with moderator decision
    for event in graph.stream(Command(resume='approve: Content is actually fine'), thread1, stream_mode='updates'):
        for node_name, node_output in event.items():
            if node_output:
                print(f'📋 {node_name}: {node_output}')

    print('\n✨ Tutorial Complete!')
    print('\nKey Takeaways:')
    print('🔸 INTERRUPTS: Use interrupt() to pause execution for human input')
    print('🔸 COMMANDS: Use Command() to dynamically route and update state')
    print('🔸 CHECKPOINTING: Enable with InMemorySaver() for stateful execution')
    print('🔸 ROUTING: Commands eliminate need for explicit conditional edges')


# ===============================================================================
# PART 5: ADVANCED USAGE EXAMPLES
# ===============================================================================


def advanced_command_example():
    """Show more advanced Command usage patterns."""

    print('\n' + '=' * 40)
    print('ADVANCED: Command Patterns')
    print('=' * 40)

    class AdvancedState(TypedDict):
        counter: int
        messages: list[str]
        should_continue: bool

    def increment_node(state: AdvancedState) -> Command[Literal['decision_node']]:
        """Demonstrate Command with complex state updates."""
        new_counter = state.get('counter', 0) + 1
        new_messages = state.get('messages', []) + [f'Incremented to {new_counter}']

        return Command(
            update={'counter': new_counter, 'messages': new_messages, 'should_continue': new_counter < 3},
            goto='decision_node',
        )

    def decision_node(state: AdvancedState) -> Command[Literal['increment_node', 'end_node']]:
        """Route based on state using Command."""
        if state.get('should_continue', True):
            return Command(goto='increment_node')
        else:
            return Command(update={'messages': state.get('messages', []) + ['Finished!']}, goto='end_node')

    def end_node(state: AdvancedState):
        """Terminal node."""
        print('📊 Final Results:')
        print(f'   Counter: {state.get("counter", 0)}')
        for msg in state.get('messages', []):
            print(f'   📝 {msg}')
        return {}

    # Build and run advanced example
    builder = StateGraph(AdvancedState)
    builder.add_node('increment_node', increment_node)
    builder.add_node('decision_node', decision_node)
    builder.add_node('end_node', end_node)

    builder.add_edge(START, 'increment_node')
    builder.add_edge('end_node', END)

    advanced_graph = builder.compile()

    result = advanced_graph.invoke({'counter': 0, 'messages': []})
    print(f'\n🎯 Final State: {result}')


if __name__ == '__main__':
    main()
    advanced_command_example()
