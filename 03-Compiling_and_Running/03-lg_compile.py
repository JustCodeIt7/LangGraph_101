# %%
# --- Video 5: Compiling & Running Your Graph ---
# Demonstrates compiling a graph and using invoke, stream, batch, and config.
import asyncio
import time
from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END

# %%
# --- 1. Define the State (Reusing from Video 3) ---
class WorkflowState(TypedDict):
    """
    Represents the shared state of our simple workflow.
    """
    input_message: str
    processing_log: List[str]
    step: int

# %%
# --- 2. Define Node Functions (Reusing/Adapting from Video 3) ---
def start_node(state: WorkflowState) -> dict:
    """
    The first node in the workflow. Logs the start.
    """
    print(f"--- [Thread: {state.get('thread_id', 'N/A')}] Executing Start Node ---")
    initial_input = state.get("input_message", "No input provided")
    print(f"Input received: {initial_input}")
    current_log = state.get("processing_log", [])
    log_update = ["Workflow started."]
    time.sleep(0.5)  # Simulate work
    return {"processing_log": log_update, "step": 1}

# %%
def processing_step_node(state: WorkflowState) -> dict:
    """
    A node representing an intermediate processing step. Logs its execution.
    """
    print(f"--- [Thread: {state.get('thread_id', 'N/A')}] Executing Processing Step Node ---")
    current_log = state.get("processing_log", [])
    log_update = current_log + ["Processing step executed."]
    current_step = state.get("step", 0)
    time.sleep(0.5)  # Simulate work
    return {"processing_log": log_update, "step": current_step + 1}

# %%
# --- 3. Build the Graph (Reusing from Video 3) ---
graph_builder = StateGraph(WorkflowState)
graph_builder.add_node("start", start_node)
graph_builder.add_node("processing_step", processing_step_node)
graph_builder.set_entry_point("start")
graph_builder.add_edge("start", "processing_step")
graph_builder.add_edge("processing_step", END)
# %%
# --- 4. Compile the Graph ---
# The compile() method finalizes the graph structure, performs validation, and returns a runnable 'app' object.
print("Compiling the graph...")
app = graph_builder.compile()
print("Graph compiled successfully.")

print("\n--- Graph Structure ---")
print(app.get_graph().draw_ascii())
# %%
# --- 5. Define Initial State & Config ---
initial_state_base = {"input_message": "Hello Execution Methods!", "processing_log": [], "step": 0}

# Configuration dictionary - used for things like thread ID (for persistence), recursion limits, etc.
config_example = {"configurable": {"thread_id": "example-thread-1"}}
config_example_batch_1 = {"configurable": {"thread_id": "batch-thread-A"}}
config_example_batch_2 = {"configurable": {"thread_id": "batch-thread-B"}}
# %%
# --- 6. Demonstrate invoke() ---
# Executes the graph synchronously from entry point to END for a single input. Returns the final state.
print("\n" + "=" * 20 + " invoke() Demo " + "=" * 20)
print("Running app.invoke()...")
final_state_invoke = app.invoke(initial_state_base, config=config_example)
print("\ninvoke() finished.")
print("Final State from invoke():")
print(final_state_invoke)
print("=" * 55)
# %%
# --- 7. Demonstrate stream() ---
# Executes the graph step-by-step, yielding intermediate results (chunks). Useful for real-time updates in UIs.
print("\n" + "=" * 20 + " stream() Demo " + "=" * 20)
print("Running app.stream()...")
# Reset state for stream demo if needed (invoke modified the state if persistence was on)
initial_state_stream = initial_state_base.copy()  # Use a copy

stream_chunks = []
for chunk in app.stream(initial_state_stream, config=config_example, stream_mode="updates"):
    print(f"--- Stream Chunk ({type(chunk).__name__}) ---")
    # chunk is a dictionary usually containing the name of the node that just ran and the updates it returned. Format depends on stream_mode.
    print(chunk)
    stream_chunks.append(chunk)
    print("-" * 10)

print("\nstream() finished.")
# The final state isn't directly returned by stream, but the last chunk often contains i or you can reconstruct it from the chunks.
print(f"Total chunks received: {len(stream_chunks)}")
print("=" * 55)

# %%
# --- 8. Demonstrate batch() ---
# Executes the graph concurrently for multiple inputs.
# Takes a list of inputs, returns a list of final states.
print("\n" + "=" * 20 + " batch() Demo " + "=" * 20)
initial_state_batch_1 = {
    "input_message": "Input for Batch 1",
    "processing_log": [],
    "step": 0,
    "thread_id": "batch-thread-A",
}
initial_state_batch_2 = {
    "input_message": "Input for Batch 2",
    "processing_log": [],
    "step": 0,
    "thread_id": "batch-thread-B",
}
batch_inputs = [initial_state_batch_1, initial_state_batch_2]
batch_configs = [config_example_batch_1, config_example_batch_2]  # Configs per input

print(f"Running app.batch() with {len(batch_inputs)} inputs...")
# Note: For batch, config can be a single dict applied to all, or a list matching inputs
final_states_batch = app.batch(batch_inputs, config=batch_configs)
print("\nbatch() finished.")
print("Final States from batch():")
for i, final_state in enumerate(final_states_batch):
    print(f"--- Input {i + 1} ---")
    print(final_state)
print("=" * 55)
# %%
# --- 9. Demonstrate Async Methods (ainvoke, astream, abatch) ---
# These require an async context (e.g., using asyncio).
print("\n" + "=" * 20 + " Async Methods Demo " + "=" * 20)


async def run_async_demos():
    # ainvoke
    print("\nRunning await app.ainvoke()...")
    final_state_ainvoke = await app.ainvoke(initial_state_base, config=config_example)
    print("ainvoke() finished.")
    print("Final State from ainvoke():", final_state_ainvoke)

    # astream
    print("\nRunning async for chunk in app.astream()...")
    initial_state_astream = initial_state_base.copy()
    astream_chunks = []
    async for chunk in app.astream(initial_state_astream, config=config_example, stream_mode="updates"):
        print(f"--- Async Stream Chunk ({type(chunk).__name__}) ---")
        print(chunk)
        astream_chunks.append(chunk)
        print("-" * 10)
    print("astream() finished.")
    print(f"Total async chunks received: {len(astream_chunks)}")

    # abatch
    print("\nRunning await app.abatch()...")
    final_states_abatch = await app.abatch(batch_inputs, config=batch_configs)
    print("abatch() finished.")
    print("Final States from abatch():")
    for i, final_state in enumerate(final_states_abatch):
        print(f"--- Input {i + 1} ---")
        print(final_state)


# Run the async demos
asyncio.run(run_async_demos())
print("=" * 55)

print("\nVideo 5 Demo Complete.")
