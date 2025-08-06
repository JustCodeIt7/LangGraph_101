"""
Example 2: Content Review and Editing Pattern
==============================================

This example demonstrates a more sophisticated HITL workflow where humans can
review, edit, and iterate on AI-generated content before final approval.
Perfect for content creation, document drafting, and creative workflows.

Key Features:
- Multi-turn review and editing cycles
- Human can modify AI-generated content
- Iterative refinement process
- Final approval with editing history
"""

from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime


class ContentEditingState(TypedDict):
    """State for content review and editing workflow"""
    content_type: str
    initial_prompt: str
    current_content: str
    content_history: List[dict]
    human_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    final_approved: bool
    user_id: str


def generate_initial_content(state: ContentEditingState) -> ContentEditingState:
    """
    AI generates initial content based on the prompt
    """
    print(f"\n🤖 AI: Generating {state['content_type']} based on your prompt...")
    print(f"   Prompt: {state['initial_prompt']}")
    
    # Simulate AI content generation based on type
    if state["content_type"] == "email":
        content = f"""Subject: {state['initial_prompt'].split('about')[1].strip() if 'about' in state['initial_prompt'] else 'Important Update'}

Dear Team,

I hope this email finds you well. I wanted to reach out regarding the recent developments in our project.

As we move forward, it's important that we maintain our focus on delivering quality results while keeping our timelines in mind. The upcoming milestones will require coordinated effort from all team members.

Please let me know if you have any questions or concerns.

Best regards,
AI Assistant"""
    
    elif state["content_type"] == "blog_post":
        topic = state['initial_prompt'].replace("Write a blog post about", "").strip()
        content = f"""# {topic.title()}

## Introduction

In today's rapidly evolving landscape, {topic.lower()} has become increasingly important for businesses and individuals alike. This post explores the key aspects and practical implications.

## Key Points

1. **Understanding the Basics**: {topic.title()} fundamentals everyone should know
2. **Practical Applications**: Real-world scenarios and use cases
3. **Best Practices**: Proven strategies for success
4. **Future Outlook**: What to expect in the coming years

## Conclusion

As we've explored in this post, {topic.lower()} presents both opportunities and challenges. By understanding these concepts and applying best practices, you can navigate this landscape effectively.

What are your thoughts on {topic.lower()}? Share your experiences in the comments below!"""
    
    elif state["content_type"] == "social_media":
        content = f"""🚀 Exciting news! {state['initial_prompt']}

Did you know that innovative solutions can transform the way we work? Here's what makes the difference:

✅ Strategic thinking
✅ Collaborative approach  
✅ Continuous learning

What's your take on this? Drop a comment below! 👇

#Innovation #Growth #Success"""
    
    else:
        content = f"Generated content for: {state['initial_prompt']}\n\nThis is AI-generated content that requires human review and potential editing."
    
    # Add to history
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "version": 1,
        "content": content,
        "source": "AI_generated",
        "feedback": None
    }
    
    print(f"\n📝 Generated Content:")
    print(f"{'-' * 50}")
    print(content)
    print(f"{'-' * 50}")
    
    return {
        **state,
        "current_content": content,
        "content_history": [history_entry],
        "iteration_count": 1
    }


def request_human_review(state: ContentEditingState) -> ContentEditingState:
    """
    Request human review and feedback on the current content
    """
    print(f"\n👤 Human Review Required (Iteration {state['iteration_count']}):")
    print("=" * 60)
    print("Please review the content above and choose an action:")
    print("1. Approve as-is (type: 'approve')")
    print("2. Edit content (type: 'edit')")
    print("3. Request revision with feedback (type: 'revise')")
    print("4. Reject completely (type: 'reject')")
    
    # Get human input
    action = input("\nYour action: ").strip().lower()
    
    if action in ['approve', 'a', '1']:
        print("✅ Content approved!")
        return {
            **state,
            "final_approved": True,
            "human_feedback": "Approved by human reviewer"
        }
    
    elif action in ['edit', 'e', '2']:
        print("\n✏️ Please provide your edited version:")
        edited_content = input("Edited content: ").strip()
        
        if edited_content:
            # Add edited version to history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "version": len(state["content_history"]) + 1,
                "content": edited_content,
                "source": "human_edited",
                "feedback": "Direct human edit"
            }
            
            print("✅ Content updated with your edits!")
            
            return {
                **state,
                "current_content": edited_content,
                "content_history": state["content_history"] + [history_entry],
                "human_feedback": "Content edited by human",
                "final_approved": True
            }
        else:
            print("❌ No edits provided, keeping current version")
            return state
    
    elif action in ['revise', 'r', '3']:
        print("\n💭 Please provide feedback for AI revision:")
        feedback = input("Your feedback: ").strip()
        
        if feedback:
            print(f"✅ Feedback recorded: {feedback}")
            return {
                **state,
                "human_feedback": feedback,
                "final_approved": False
            }
        else:
            print("❌ No feedback provided")
            return state
    
    elif action in ['reject', 'reject_all', '4']:
        print("❌ Content rejected by human reviewer")
        return {
            **state,
            "final_approved": False,
            "human_feedback": "Content rejected - start over"
        }
    
    else:
        print(f"❓ Unknown action '{action}', treating as revision request")
        return {
            **state,
            "human_feedback": f"Unknown action: {action}",
            "final_approved": False
        }


def revise_content(state: ContentEditingState) -> ContentEditingState:
    """
    AI revises content based on human feedback
    """
    feedback = state.get("human_feedback", "")
    current_content = state["current_content"]
    
    print(f"\n🤖 AI: Revising content based on feedback: '{feedback}'")
    
    # Simulate AI revision based on feedback keywords
    if "shorter" in feedback.lower() or "brief" in feedback.lower():
        # Make content shorter
        lines = current_content.split('\n')
        revised_content = '\n'.join(lines[:len(lines)//2]) + "\n\n[Content abbreviated based on feedback]"
    
    elif "longer" in feedback.lower() or "more detail" in feedback.lower():
        # Add more content
        revised_content = current_content + f"\n\nAdditional Details:\nBased on your feedback, here are more insights and detailed explanations that provide greater depth to the topic..."
    
    elif "professional" in feedback.lower() or "formal" in feedback.lower():
        # Make more professional
        revised_content = current_content.replace("!", ".").replace("exciting", "noteworthy").replace("amazing", "significant")
    
    elif "casual" in feedback.lower() or "friendly" in feedback.lower():
        # Make more casual
        revised_content = current_content.replace(".", "!") + "\n\nHope this helps! Let me know what you think! 😊"
    
    else:
        # Generic revision
        revised_content = f"[REVISED] {current_content}\n\n[AI Note: Content revised based on feedback: {feedback}]"
    
    # Add revision to history
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "version": len(state["content_history"]) + 1,
        "content": revised_content,
        "source": "AI_revised",
        "feedback": feedback
    }
    
    print(f"\n📝 Revised Content:")
    print(f"{'-' * 50}")
    print(revised_content)
    print(f"{'-' * 50}")
    
    return {
        **state,
        "current_content": revised_content,
        "content_history": state["content_history"] + [history_entry],
        "iteration_count": state["iteration_count"] + 1,
        "human_feedback": None  # Reset feedback
    }


def should_continue_editing(state: ContentEditingState) -> Literal["review", "end"]:
    """
    Determine if we should continue the editing process
    """
    # Check if approved
    if state.get("final_approved", False):
        return "end"
    
    # Check if max iterations reached
    if state["iteration_count"] >= state["max_iterations"]:
        print(f"\n⚠️ Maximum iterations ({state['max_iterations']}) reached!")
        return "end"
    
    # Check if rejected completely
    if state.get("human_feedback") == "Content rejected - start over":
        return "end"
    
    # Continue if we have feedback to work with
    return "review" if state.get("human_feedback") else "end"


def create_content_editing_graph():
    """
    Create the content editing workflow graph
    """
    workflow = StateGraph(ContentEditingState)
    
    # Add nodes
    workflow.add_node("generate", generate_initial_content)
    workflow.add_node("review", request_human_review)
    workflow.add_node("revise", revise_content)
    
    # Define the workflow
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "review")
    
    # Conditional routing after review
    workflow.add_conditional_edges(
        "review",
        lambda state: "revise" if not state.get("final_approved", False) and state.get("human_feedback") and "rejected" not in state.get("human_feedback", "") else "end",
        {
            "revise": "revise",
            "end": END
        }
    )
    
    # After revision, go back to review
    workflow.add_conditional_edges(
        "revise",
        should_continue_editing,
        {
            "review": "review",
            "end": END
        }
    )
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def demo_email_editing():
    """
    Demo: Email content editing workflow
    """
    print("=" * 60)
    print("DEMO 1: Email Content Editing")
    print("=" * 60)
    
    app = create_content_editing_graph()
    
    initial_state = ContentEditingState(
        content_type="email",
        initial_prompt="Write an email about the upcoming team meeting",
        current_content="",
        content_history=[],
        human_feedback=None,
        iteration_count=0,
        max_iterations=3,
        final_approved=False,
        user_id="user123"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Final Results:")
    print(f"   Approved: {final_state['final_approved']}")
    print(f"   Iterations: {final_state['iteration_count']}")
    print(f"   History versions: {len(final_state['content_history'])}")


def demo_blog_post_editing():
    """
    Demo: Blog post content editing workflow
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Blog Post Content Editing")
    print("=" * 60)
    
    app = create_content_editing_graph()
    
    initial_state = ContentEditingState(
        content_type="blog_post",
        initial_prompt="Write a blog post about artificial intelligence in healthcare",
        current_content="",
        content_history=[],
        human_feedback=None,
        iteration_count=0,
        max_iterations=5,
        final_approved=False,
        user_id="blogger456"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Final Results:")
    print(f"   Approved: {final_state['final_approved']}")
    print(f"   Iterations: {final_state['iteration_count']}")
    print(f"   History versions: {len(final_state['content_history'])}")


if __name__ == "__main__":
    print("🔄 LangGraph Human-in-the-Loop: Content Review & Editing")
    print("=" * 60)
    print("This example demonstrates iterative content creation with")
    print("human review, editing, and approval capabilities.")
    print("=" * 60)
    
    # Run demos
    demo_email_editing()
    demo_blog_post_editing()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("Note: Content history and iterations preserved throughout the process.")
    print("=" * 60)
