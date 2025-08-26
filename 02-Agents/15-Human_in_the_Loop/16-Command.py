# %%
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from IPython.display import display, Image

# %%
# Simplified customer service workflow with Command routing


# Define simplified graph state
class CustomerServiceState(TypedDict):
    customer_id: str
    issue_description: str
    priority: str
    status: str
    resolution: str


# Simplified routing function
def router(state: CustomerServiceState) -> Command[Literal['support_agent', 'manager']]:
    print('Called Router')

    # Simple routing logic based on keywords
    issue_keywords = state['issue_description'].lower()

    if 'urgent' in issue_keywords or 'critical' in issue_keywords or 'refund' in issue_keywords:
        # High priority issues go to manager
        priority = 'high'
        goto = 'manager'
        status = 'escalated'
    else:
        # Standard issues go to support agent
        priority = 'medium'
        goto = 'support_agent'
        status = 'assigned'

    return Command(
        update={'priority': priority, 'status': status},
        goto=goto,
    )


def support_agent(state: CustomerServiceState):
    print('Called Support Agent')
    # Support agent handles standard issues
    resolution = f'Support agent resolved: {state["issue_description"]}'
    return {'status': 'completed', 'resolution': resolution}


def manager(state: CustomerServiceState):
    print('Called Manager')
    # Manager handles escalated issues
    resolution = f'Manager resolved high-priority issue: {state["issue_description"]}'
    return {'status': 'completed', 'resolution': resolution}


# %%

# Build the simplified graph
builder = StateGraph(CustomerServiceState)
builder.add_edge(START, 'router')
builder.add_node(router)
builder.add_node(support_agent)
builder.add_node(manager)

graph = builder.compile()

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
# Test examples

print('=== Example 1: Standard Issue ===')
result1 = graph.invoke(
    {
        'customer_id': 'cust123',
        'issue_description': 'I need help with my account settings',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result1)
# %%

print('\n=== Example 2: Urgent Issue ===')
result2 = graph.invoke(
    {
        'customer_id': 'cust456',
        'issue_description': 'Urgent: I need a refund for my recent purchase',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result2)
# %%
print('\n=== Example 3: Critical Issue ===')
result3 = graph.invoke(
    {
        'customer_id': 'cust789',
        'issue_description': 'Critical system error preventing me from logging in',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result3)

# %%
print('\n=== Example 4: Billing Inquiry ===')
result4 = graph.invoke(
    {
        'customer_id': 'cust101',
        'issue_description': 'I have questions about my monthly billing statement',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result4)

# %%
print('\n=== Example 5: Urgent Password Reset ===')
result5 = graph.invoke(
    {
        'customer_id': 'cust202',
        'issue_description': 'Urgent: I cannot reset my password and need immediate access',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result5)

print('\n=== Example 6: Feature Request ===')
result6 = graph.invoke(
    {
        'customer_id': 'cust303',
        'issue_description': 'I would love a dark mode feature in the app',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result6)

print('\n=== Example 7: Refund Requested (Non-urgent) ===')
result7 = graph.invoke(
    {
        'customer_id': 'cust404',
        'issue_description': 'I am requesting a refund for an accidental duplicate order',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result7)

# %%
