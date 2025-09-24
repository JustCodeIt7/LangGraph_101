# %%
from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from rich import print


# %%
################################ State Definition #################################
# Define the schema for the shared state that flows through the graph.
class TodoState(TypedDict):
    """
    Represents the application's state, including the list of tasks and reminders.
    """

    # Use operator.add to accumulate items into the list across graph runs.
    todos: Annotated[List[str], operator.add]
    reminder: str
    new_todo: str


# %%
################################ Main Graph Nodes #################################


# Define a node to add a new task to the to-do list.
def add_todo(state: TodoState):
    """
    Appends a new task from the 'new_todo' field to the 'todos' list.
    """
    print('--- ADDING TODO ---')
    new_task = state.get('new_todo', '').strip()

    # Handle cases where the input task is empty.
    if not new_task:
        print('No new task provided.')
        return {'todos': []}  # Return an empty list to signify no change.

    print(f'Adding task: {new_task}')
    # Return the new task in a list to be appended to the state.
    return {'todos': [new_task]}


# Define a node to print the final state of the to-do list and reminder.
def display_final_state(state: TodoState):
    """
    Displays the final list of tasks and the generated reminder to the console.
    """
    print('\n--- FINAL TO-DO LIST & REMINDER ---')
    print(f'Tasks: {state["todos"]}')
    print(f'Reminder: {state["reminder"]}')
    print('------------------------------------')
    return {}  # Return an empty dictionary as this node doesn't modify state.


# %%
################################ Subgraph Nodes #################################


# Define a node that uses an LLM to prioritize the current list of tasks.
def prioritize_tasks_llm(state: TodoState):
    """
    Invokes an LLM to generate a prioritized list from the current 'todos'.
    """
    print('\n--- SUBGRAPH: PRIORITIZING TASKS ---')
    llm = ChatOllama(model='llama3.2')  # Initialize the language model.

    # If there are no tasks, return a default message.
    if not state['todos']:
        return {'reminder': 'No tasks to prioritize.'}

    # Format the list of tasks into a string for the LLM prompt.
    task_list_str = '\n'.join(f'- {task}' for task in state['todos'])

    # Construct the prompt with system and human messages.
    prompt = [
        SystemMessage(content='You are a helpful assistant. Your task is to prioritize a list of to-do items.'),
        HumanMessage(
            content=f'Please prioritize the following tasks and return them as a numbered list, with the most important task first:\n\n{task_list_str}'
        ),
    ]

    response = llm.invoke(prompt)
    print('LLM generated prioritized list.')
    # Update the 'reminder' field with the raw LLM output.
    return {'reminder': response.content}


# Define a node to format the LLM's output into a user-friendly message.
def format_reminder_text(state: TodoState):
    """
    Wraps the prioritized task list in a friendly reminder message.
    """
    print('--- SUBGRAPH: FORMATTING REMINDER ---')
    prioritized_tasks = state.get('reminder', 'No tasks were prioritized.')
    formatted_reminder = f'Here is your prioritized reminder for today:\n{prioritized_tasks}'
    # Overwrite the 'reminder' field with the final formatted string.
    return {'reminder': formatted_reminder}


# %%
################################ Graph Construction #################################

# Create the state graph for the reminder-generation subgraph.
reminder_subgraph = StateGraph(TodoState)
# Add nodes for prioritizing tasks and formatting the output.
reminder_subgraph.add_node('prioritize', prioritize_tasks_llm)
reminder_subgraph.add_node('format_reminder', format_reminder_text)

# Define the data flow within the subgraph.
reminder_subgraph.add_edge(START, 'prioritize')
reminder_subgraph.add_edge('prioritize', 'format_reminder')
reminder_subgraph.add_edge('format_reminder', END)

# Compile the subgraph into a runnable application.
reminder_app = reminder_subgraph.compile()

# Create the main application graph.
main_graph = StateGraph(TodoState)

# Add nodes for the main workflow.
main_graph.add_node('add_todo', add_todo)
# Add the compiled subgraph as a single, callable node.
main_graph.add_node('reminder_subgraph', reminder_app)
main_graph.add_node('display_todos', display_final_state)

# Define the data flow for the main graph.
main_graph.add_edge(START, 'add_todo')
main_graph.add_edge('add_todo', 'reminder_subgraph')
main_graph.add_edge('reminder_subgraph', 'display_todos')
main_graph.add_edge('display_todos', END)

# Compile the complete main graph into the final application.
app = main_graph.compile()

# %%
# display the structure of the main application graph.
print('Main application graph structure:')
print('Subgraph structure:')
# Optional: Display a visualization of the graph's structure.
try:
    from IPython.display import Image, display

    img = Image(app.get_graph().draw_mermaid_png())
    display(img)
    # save the image locally
    img.save('todo_app_graph.png')

except Exception:
    pass
# %%
################################ Application Execution ################################

if __name__ == '__main__':
    # Initialize an empty state for the first run.
    current_state = {'todos': [], 'reminder': ''}

    # Execute the graph sequentially to add multiple tasks and update the state.
    # First invocation adds 'Finish LangGraph tutorial video'.
    inputs1 = {'new_todo': 'Finish LangGraph tutorial video'}
    current_state = app.invoke(
        inputs1, config={'recursion_limit': 100}, state=current_state
    )  # Pass previous state to accumulate tasks.

    # Second invocation adds 'Buy groceries'.
    inputs2 = {'new_todo': 'Buy groceries'}
    current_state = app.invoke(inputs2, config={'recursion_limit': 100}, state=current_state)  # Pass previous state.

    # Third invocation adds 'Schedule dentist appointment' and gets the final result.
    inputs3 = {'new_todo': 'Schedule dentist appointment'}
    final_state = app.invoke(inputs3, config={'recursion_limit': 100}, state=current_state)  # Pass previous state.

# %%
