# %%
import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from IPython.display import display, Image

# %%


# Define graph state for customer service workflow
class CustomerServiceState(TypedDict):
    customer_id: str
    issue_type: str
    issue_description: str
    priority: str
    status: str
    resolution: str


# Define the nodes


def triage_agent(state: CustomerServiceState) -> Command[Literal['billing_agent', 'technical_agent', 'general_agent']]:
    print('Called Triage Agent')
    # Analyze the issue type and route to appropriate specialist
    issue_keywords = state['issue_description'].lower()

    if 'bill' in issue_keywords or 'payment' in issue_keywords or 'refund' in issue_keywords:
        goto = 'billing_agent'
        priority = 'high' if 'urgent' in issue_keywords else 'medium'
    elif 'technical' in issue_keywords or 'error' in issue_keywords or 'bug' in issue_keywords:
        goto = 'technical_agent'
        priority = 'high' if 'critical' in issue_keywords else 'medium'
    else:
        goto = 'general_agent'
        priority = 'low'

    # Update state with priority and route to next node
    return Command(
        update={'priority': priority, 'status': 'triaged'},
        goto=goto,
    )


def billing_agent(state: CustomerServiceState) -> Command[Literal['escalate_manager', 'resolve_issue']]:
    print('Called Billing Agent')
    # Handle billing-related issues
    issue_description = state['issue_description'].lower()

    if 'refund' in issue_description and state['priority'] == 'high':
        # High priority refund requests need manager approval
        resolution = 'Refund request escalated to manager for approval'
        goto = 'escalate_manager'
    else:
        # Standard billing issues can be resolved directly
        resolution = 'Billing issue resolved with standard procedure'
        goto = 'resolve_issue'

    return Command(
        update={'status': 'processed', 'resolution': resolution},
        goto=goto,
    )


def technical_agent(state: CustomerServiceState) -> Command[Literal['escalate_manager', 'resolve_issue']]:
    print('Called Technical Agent')
    # Handle technical issues
    issue_description = state['issue_description'].lower()

    if 'critical' in issue_description or 'system down' in issue_description:
        # Critical technical issues need escalation
        resolution = 'Critical technical issue escalated to engineering team'
        goto = 'escalate_manager'
    else:
        # Standard technical issues
        resolution = 'Technical issue resolved with troubleshooting steps'
        goto = 'resolve_issue'

    return Command(
        update={'status': 'processed', 'resolution': resolution},
        goto=goto,
    )


def general_agent(state: CustomerServiceState) -> Command[Literal['escalate_manager', 'resolve_issue']]:
    print('Called General Agent')
    # Handle general inquiries
    issue_description = state['issue_description'].lower()

    if 'complaint' in issue_description or 'unhappy' in issue_description:
        # Customer complaints need manager attention
        resolution = 'Customer complaint escalated to manager'
        goto = 'escalate_manager'
    else:
        # General inquiries can be resolved
        resolution = 'General inquiry resolved with information provided'
        goto = 'resolve_issue'

    return Command(
        update={'status': 'processed', 'resolution': resolution},
        goto=goto,
    )


def escalate_manager(state: CustomerServiceState):
    print('Called Escalate Manager')
    # Manager handles escalated issues
    resolution = f'Manager resolved escalated issue: {state["resolution"]}'
    return {'status': 'resolved', 'resolution': resolution}


def resolve_issue(state: CustomerServiceState):
    print('Called Resolve Issue')
    # Final resolution step
    return {'status': 'completed', 'resolution': f'Issue resolved: {state["resolution"]}'}


# %% [markdown]
# This is a markdown cell explaining the customer service workflow
# %%
builder = StateGraph(CustomerServiceState)
builder.add_edge(START, 'triage_agent')
builder.add_node(triage_agent)
builder.add_node(billing_agent)
builder.add_node(technical_agent)
builder.add_node(general_agent)
builder.add_node(escalate_manager)
builder.add_node(resolve_issue)
# NOTE: there are no edges between nodes except from START to triage_agent!

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# This is another markdown cell showing example invocations
# %%
# Example 1: Billing issue
print('=== Example 1: Billing Issue ===')
result1 = graph.invoke(
    {
        'customer_id': 'cust123',
        'issue_type': 'billing',
        'issue_description': 'I have an urgent question about my recent bill',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result1)

# Example 2: Technical issue
print('\n=== Example 2: Technical Issue ===')
result2 = graph.invoke(
    {
        'customer_id': 'cust456',
        'issue_type': 'technical',
        'issue_description': 'I encountered a critical error in the system',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result2)

# Example 3: General inquiry
print('\n=== Example 3: General Inquiry ===')
result3 = graph.invoke(
    {
        'customer_id': 'cust789',
        'issue_type': 'general',
        'issue_description': 'I need information about your services',
        'priority': '',
        'status': 'new',
        'resolution': '',
    }
)
print(result3)


# %% [markdown]
# Navigate to a node in a parent graph with subgraph example
# %%
def escalation_node(state: CustomerServiceState) -> Command[Literal['manager_review']]:
    return Command(
        update={'status': 'escalated', 'resolution': 'Issue requires manager review'},
        goto='manager_review',  # where 'manager_review' is a node in the parent graph
        graph=Command.PARENT,
    )


# %%
import operator
from typing_extensions import Annotated


class SubgraphState(TypedDict):
    # NOTE: we define a reducer here to accumulate resolution steps
    resolution_steps: Annotated[list[str], operator.add]
    final_resolution: str


def initial_triage(state: SubgraphState) -> Command[Literal['specialist_agent', 'escalate_node']]:
    print('Called Initial Triage in Subgraph')
    # Randomly decide if we need escalation or can handle with specialist
    needs_escalation = random.choice([True, False])

    if needs_escalation:
        goto = 'escalate_node'
        resolution_step = 'Issue identified as requiring escalation'
    else:
        goto = 'specialist_agent'
        resolution_step = 'Issue routed to specialist agent'

    return Command(
        update={'resolution_steps': [resolution_step]},
        goto=goto,
        # This tells LangGraph to navigate to the specified node in the parent graph
        graph=Command.PARENT,
    )


def specialist_agent(state: SubgraphState):
    print('Called Specialist Agent in Subgraph')
    resolution_step = 'Specialist agent processed the issue'
    return {'resolution_steps': [resolution_step], 'final_resolution': 'Issue resolved by specialist'}


# Create the subgraph
subgraph = (
    StateGraph(SubgraphState)
    .add_node(initial_triage)
    .add_node(specialist_agent)
    .add_edge(START, 'initial_triage')
    .compile()
)


def manager_review(state: CustomerServiceState):
    print('Called Manager Review in Parent Graph')
    resolution = f'Manager reviewed and resolved: {state["resolution"]}'
    return {'status': 'resolved', 'resolution': resolution}


def customer_service_agent(state: CustomerServiceState):
    print('Called Customer Service Agent in Parent Graph')
    resolution = 'Customer service agent handled the inquiry'
    return {'status': 'resolved', 'resolution': resolution}


# Build the parent graph
builder = StateGraph(CustomerServiceState)
builder.add_edge(START, 'subgraph')
builder.add_node('subgraph', subgraph)
builder.add_node(manager_review)
builder.add_node(customer_service_agent)

parent_graph = builder.compile()

# %%
# Test the parent graph with subgraph
print('\n=== Testing Parent Graph with Subgraph ===')
result = parent_graph.invoke(
    {
        'customer_id': 'cust999',
        'issue_type': 'complex',
        'issue_description': 'Complex issue requiring subgraph processing',
        'priority': 'high',
        'status': 'new',
        'resolution': '',
    }
)
print(result)

# %%
