#!/usr/bin/env python3
"""
Human-in-the-Loop: Collaborative Creation

This example demonstrates:
1. AI and human working together to create content
2. Iterative refinement process with human feedback
3. Real-time collaboration and suggestion system
4. Perfect for content creation, design, or creative workflows

Pattern: AI Draft → Human Feedback → AI Refine → Collaborate → Finalize
"""

import json
from typing import TypedDict, Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# State definition
class CreationState(TypedDict):
    messages: Annotated[list, add_messages]
    current_version: int
    content_history: List[dict]
    feedback_history: List[str]
    collaboration_mode: str  # "drafting", "reviewing", "refining", "finalizing"
    content_type: str  # "blog", "email", "report", "story"
    quality_score: float


# Define tools for collaborative creation
@tool
def generate_content_outline(topic: str, content_type: str, target_audience: str) -> dict:
    """Generate a content outline based on topic and type."""
    outlines = {
        'blog': {
            'introduction': 'Hook and main thesis',
            'main_points': ['Point 1 with evidence', 'Point 2 with examples', 'Point 3 with analysis'],
            'conclusion': 'Summary and call to action',
        },
        'email': {
            'subject': 'Clear and compelling subject line',
            'greeting': 'Personalized greeting',
            'body': 'Main message with key points',
            'closing': 'Professional closing and signature',
        },
        'report': {
            'executive_summary': 'Brief overview of findings',
            'methodology': 'How the analysis was conducted',
            'findings': 'Key results and data',
            'recommendations': 'Action items based on findings',
        },
        'story': {
            'opening': 'Character and setting introduction',
            'conflict': 'Main problem or challenge',
            'development': 'Plot progression and character growth',
            'resolution': 'Climax and conclusion',
        },
    }

    return outlines.get(
        content_type, {'introduction': 'Introduction', 'body': 'Main content', 'conclusion': 'Conclusion'}
    )


@tool
def analyze_content_quality(content: str, content_type: str) -> dict:
    """Analyze content quality and provide suggestions."""
    # Simple quality analysis based on content characteristics
    word_count = len(content.split())
    sentences = content.split('.')

    quality_metrics = {
        'word_count': word_count,
        'sentence_count': len([s for s in sentences if s.strip()]),
        'readability_score': max(0, min(100, 100 - (word_count / 10))),  # Simple readability
        'suggestions': [],
    }

    # Generate suggestions based on analysis
    if word_count < 100:
        quality_metrics['suggestions'].append('Consider adding more detail to fully develop the topic')
    elif word_count > 1000:
        quality_metrics['suggestions'].append('Consider breaking into smaller sections for better readability')

    if len(sentences) < 5:
        quality_metrics['suggestions'].append('Add more sentence variety for better flow')

    # Content-specific suggestions
    if content_type == 'blog':
        quality_metrics['suggestions'].append('Include examples or case studies to support your points')
    elif content_type == 'email':
        quality_metrics['suggestions'].append('Ensure clear call to action is included')

    return quality_metrics


@tool
def suggest_improvements(content: str, feedback: str, content_type: str) -> List[str]:
    """Generate specific improvement suggestions based on feedback."""
    # Parse feedback for key themes
    feedback_lower = feedback.lower()

    suggestions = []

    if 'long' in feedback_lower or 'short' in feedback_lower:
        suggestions.append('Adjust content length based on feedback')

    if 'tone' in feedback_lower:
        suggestions.append('Modify tone to better match audience expectations')

    if 'structure' in feedback_lower or 'organization' in feedback_lower:
        suggestions.append('Reorganize content for better flow and clarity')

    if 'examples' in feedback_lower or 'evidence' in feedback_lower:
        suggestions.append('Add specific examples or evidence to support claims')

    if 'clarity' in feedback_lower or 'clear' in feedback_lower:
        suggestions.append('Simplify language and improve sentence structure')

    # Add content-specific suggestions
    if content_type == 'blog':
        suggestions.append('Consider adding subheadings to break up text')
    elif content_type == 'email':
        suggestions.append('Ensure subject line is compelling and relevant')

    return suggestions if suggestions else ['Review content for general improvements']


@tool
def create_final_version(content: str, improvements: List[str]) -> str:
    """Create final version incorporating improvements."""
    # In a real implementation, this would use AI to rewrite the content
    # For this example, we'll simulate the improvement process

    improved_content = content

    # Apply some basic improvements based on suggestions
    for improvement in improvements:
        if 'length' in improvement.lower():
            # Simulate length adjustment
            if len(improved_content) > 500:
                improved_content = improved_content[:400] + '... [truncated for brevity]'
            else:
                improved_content += ' [expanded with additional details]'

        if 'structure' in improvement.lower():
            # Simulate better structure
            improved_content = f'Introduction:\n{improved_content}\n\nConclusion:\nThis content has been restructured for better clarity.'

    return improved_content


# Create tool list
tools = [generate_content_outline, analyze_content_quality, suggest_improvements, create_final_version]


# Node functions
def start_creation(state: CreationState):
    """Start the collaborative creation process."""
    system_message = SystemMessage(
        content='You are a collaborative content creator. Work with the human to create '
        'high-quality content. Start by understanding their needs and generating an outline. '
        'Be open to feedback and willing to refine the content iteratively.'
    )

    messages = state['messages']
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages

    model = ChatOpenAI(model='gpt-4o')
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    return {
        'messages': [response],
        'current_version': 1,
        'content_history': [],
        'feedback_history': [],
        'collaboration_mode': 'drafting',
        'quality_score': 0.0,
    }


def collaborative_review(state: CreationState):
    """Handle collaborative review and feedback."""
    print('\n' + '=' * 60)
    print('COLLABORATIVE REVIEW SESSION')
    print('=' * 60)

    mode = state['collaboration_mode']
    version = state['current_version']

    print(f'\nCurrent Mode: {mode.upper()}')
    print(f'Version: {version}')

    # Show current content if available
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and len(msg.content) > 50:
            print(f'\nCurrent Content:\n{msg.content}')
            break

    # Show previous feedback
    if state['feedback_history']:
        print(f'\nPrevious Feedback:')
        for i, feedback in enumerate(state['feedback_history'][-3:], 1):
            print(f'{i}. {feedback}')

    # Get human input based on mode
    if mode == 'drafting':
        print('\nOptions:')
        print('1. Approve outline and proceed to drafting')
        print('2. Request changes to outline')
        print('3. Provide topic clarification')

        choice = input('\nChoose option (1/2/3): ').strip()

        if choice == '1':
            feedback = 'Outline approved. Please proceed with drafting.'
            new_mode = 'reviewing'
        elif choice == '2':
            feedback = input('What changes would you like to make to the outline? ').strip()
            new_mode = 'drafting'
        else:
            feedback = input('Please provide additional topic information: ').strip()
            new_mode = 'drafting'

    elif mode == 'reviewing':
        print('\nReview Options:')
        print('1. Approve content')
        print('2. Request revisions')
        print('3. Suggest improvements')
        print('4. Ask for different approach')

        choice = input('\nChoose option (1/2/3/4): ').strip()

        if choice == '1':
            feedback = 'Content approved. Ready for finalization.'
            new_mode = 'finalizing'
        elif choice == '2':
            feedback = input('What revisions would you like? ').strip()
            new_mode = 'refining'
        elif choice == '3':
            feedback = input('What improvements would you suggest? ').strip()
            new_mode = 'refining'
        else:
            feedback = input('What different approach would you prefer? ').strip()
            new_mode = 'drafting'

    elif mode == 'refining':
        print('\nRefinement Options:')
        print('1. Accept refinements')
        print('2. Request further changes')
        print('3. Add specific requirements')

        choice = input('\nChoose option (1/2/3): ').strip()

        if choice == '1':
            feedback = 'Refinements look good. Ready for final review.'
            new_mode = 'reviewing'
        else:
            feedback = input('What additional changes are needed? ').strip()
            new_mode = 'refining'

    else:  # finalizing
        print('\nFinalization Options:')
        print('1. Approve final version')
        print('2. Request final tweaks')
        print('3. Start over with new approach')

        choice = input('\nChoose option (1/2/3): ').strip()

        if choice == '1':
            feedback = 'Final version approved. Content creation complete!'
            new_mode = 'completed'
        elif choice == '2':
            feedback = input('What final tweaks are needed? ').strip()
            new_mode = 'refining'
        else:
            feedback = "Let's start over with a fresh approach."
            new_mode = 'drafting'

    # Update feedback history
    feedback_history = state['feedback_history'] + [feedback]

    print(f'\n✓ Feedback recorded: {feedback}')

    return {
        'feedback_history': feedback_history,
        'collaboration_mode': new_mode,
        'messages': [HumanMessage(content=feedback)],
        'current_version': version + 1 if new_mode != 'completed' else version,
    }


def refine_content(state: CreationState):
    """Refine content based on feedback."""
    print('\n🔧 Refining content based on feedback...')

    # Get the latest feedback
    latest_feedback = state['feedback_history'][-1] if state['feedback_history'] else ''

    # Get current content
    current_content = ''
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and len(msg.content) > 50:
            current_content = msg.content
            break

    # Generate improvement suggestions
    content_type = state.get('content_type', 'blog')
    suggestions = suggest_improvements.invoke(
        {'content': current_content, 'feedback': latest_feedback, 'content_type': content_type}
    )

    print(f'\n💡 Improvement Suggestions:')
    for i, suggestion in enumerate(suggestions, 1):
        print(f'{i}. {suggestion}')

    # Create refined version
    refined_content = create_final_version.invoke({'content': current_content, 'improvements': suggestions})

    # Analyze quality
    quality_analysis = analyze_content_quality.invoke({'content': refined_content, 'content_type': content_type})

    # Save to history
    content_history = state['content_history'] + [
        {
            'version': state['current_version'],
            'content': current_content,
            'feedback': latest_feedback,
            'suggestions': suggestions,
            'quality_score': quality_analysis.get('readability_score', 0),
        }
    ]

    print(f'\n📊 Quality Score: {quality_analysis.get("readability_score", 0):.1f}/100')

    return {
        'content_history': content_history,
        'quality_score': quality_analysis.get('readability_score', 0),
        'messages': [AIMessage(content=refined_content)],
    }


def finalize_content(state: CreationState):
    """Finalize the content and generate summary."""
    print('\n🎯 Finalizing content...')

    # Get final content
    final_content = ''
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and len(msg.content) > 50:
            final_content = msg.content
            break

    # Generate final summary
    summary = {
        'total_versions': state['current_version'],
        'content_type': state.get('content_type', 'blog'),
        'final_quality_score': state.get('quality_score', 0),
        'revision_count': len(state['feedback_history']),
        'content_length': len(final_content.split()),
        'creation_summary': f'Content created through {state["current_version"]} versions with {len(state["feedback_history"])} feedback iterations.',
    }

    print(f'\n📋 Creation Summary:')
    print(json.dumps(summary, indent=2))

    print(f'\n📝 Final Content:\n{final_content}')

    return {
        'collaboration_mode': 'completed',
        'messages': [AIMessage(content=f'Content creation completed! Summary: {summary["creation_summary"]}')],
    }


def process_creation_tools(state: CreationState):
    """Process tools for content creation."""
    messages = state['messages']
    last_message = messages[-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {'messages': []}

    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        if tool_name == 'generate_content_outline':
            result = generate_content_outline.invoke(tool_args)
            # Set content type based on outline generation
            content_type = tool_args.get('content_type', 'blog')
        elif tool_name == 'analyze_content_quality':
            result = analyze_content_quality.invoke(tool_args)
        elif tool_name == 'suggest_improvements':
            result = suggest_improvements.invoke(tool_args)
        elif tool_name == 'create_final_version':
            result = create_final_version.invoke(tool_args)
        else:
            result = f'Unknown tool: {tool_name}'

        tool_message = ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call['id'])
        tool_messages.append(tool_message)

    return {'messages': tool_messages}


# Router functions
def route_after_start(state: CreationState):
    """Route after starting creation."""
    mode = state['collaboration_mode']
    if mode == 'drafting':
        return 'collaborate'
    return 'end'


def route_after_collaboration(state: CreationState):
    """Route after collaborative review."""
    mode = state['collaboration_mode']

    if mode == 'completed':
        return 'finalize'
    elif mode == 'refining':
        return 'refine'
    elif mode == 'drafting':
        return 'start'  # Restart with new feedback
    else:
        return 'collaborate'  # Continue collaboration


def route_after_refinement(state: CreationState):
    """Route after content refinement."""
    mode = state['collaboration_mode']
    if mode == 'refining':
        return 'collaborate'
    return 'end'


def route_after_tools(state: CreationState):
    """Route after tool processing."""
    return 'collaborate'


# Create the graph
graph = StateGraph(CreationState)

# Add nodes
graph.add_node('start', start_creation)
graph.add_node('collaborate', collaborative_review)
graph.add_node('refine', refine_content)
graph.add_node('finalize', finalize_content)
graph.add_node('tools', process_creation_tools)

# Add edges
graph.add_edge('finalize', END)

# Add conditional edges
graph.add_conditional_edges('start', route_after_start, {'collaborate': 'collaborate', 'end': END})

graph.add_conditional_edges(
    'collaborate',
    route_after_collaboration,
    {'finalize': 'finalize', 'refine': 'refine', 'start': 'start', 'collaborate': 'collaborate', 'end': END},
)

graph.add_conditional_edges('refine', route_after_refinement, {'collaborate': 'collaborate', 'end': END})

graph.add_conditional_edges('tools', route_after_tools, {'collaborate': 'collaborate'})

# Set entry point
graph.set_entry_point('start')

# Compile graph with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# Example usage
if __name__ == '__main__':
    print('=== Collaborative Content Creation ===')
    print('Try creating:')
    print("- 'Write a blog post about the benefits of remote work'")
    print("- 'Create an email announcing a new product launch'")
    print("- 'Draft a report on quarterly sales performance'")
    print("- 'Write a short story about a time traveler'")

    config = {'configurable': {'thread_id': 'collaborative_creation_1'}}

    while True:
        user_input = input('\nWhat would you like to create? ').strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Determine content type from user input
        content_type = 'blog'  # default
        if 'email' in user_input.lower():
            content_type = 'email'
        elif 'report' in user_input.lower():
            content_type = 'report'
        elif 'story' in user_input.lower():
            content_type = 'story'

        # Initialize state
        initial_state = {
            'messages': [HumanMessage(content=user_input)],
            'current_version': 1,
            'content_history': [],
            'feedback_history': [],
            'collaboration_mode': 'drafting',
            'content_type': content_type,
            'quality_score': 0.0,
        }

        # Run the collaborative creation
        result = app.invoke(initial_state, config=config)

        # Show final status
        print(f'\n🎯 Creation Status: {result.get("collaboration_mode", "completed")}')
        print(f'Final Quality Score: {result.get("quality_score", 0):.1f}/100')
        print(f'Total Versions: {result.get("current_version", 1)}')
