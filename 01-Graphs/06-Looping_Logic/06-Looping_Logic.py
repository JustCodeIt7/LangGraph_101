# %% [markdown]
# # LangGraph Looping Logic
# 
# ## Key Concepts
# 
# - **Conditional Edges**: These help decide the flow direction based on the current state.
# - **Self-loops**: Nodes that loop back to themselves or other nodes to enable iterative behavior.
# - **State Management**: Keeps track of data across multiple loop iterations.
# 
# We'll showcase two distinct methods for implementing loops in LangGraph:
# 
# 1. **Direct Conditional Routing**: Utilizing conditional edges straight from a processing node.
# 2. **Separate Decision Node**: Employing a dedicated node to handle routing decisions.
# 

# %% [markdown]
# ## 1.  Setup and Imports
# 

# %%
# * 1.  Setup and Imports
from langgraph.graph import StateGraph, END
import random
from typing import TypedDict, List
from IPython.display import Image, display

# %% [markdown]
# ## State Definition
# 
# We'll create a simple state that tracks a list of numbers and their running total. Our loop will continue adding random numbers until the total reaches a threshold.
# 

# %%
#  Define the shape of our state
class SumState(TypedDict):
    numbers: List[int]
    total: int

# %% [markdown]
# ## Approach 1: Direct Conditional Routing
# 
# In this approach, we'll create a graph where the conditional edges are attached directly to the processing node. This creates a more compact graph structure.
# 
# ### Node Functions
# 

# %%


# %% [markdown]
# ### Building the Graph
# 
# Now, we'll build the graph using direct conditional routing. Observe how the `add` node features conditional edges that can loop back to itself, forming a cycle.
# 

# %%


# %% [markdown]
# ### Visualizing the Graph
# 
# Let's visualize the graph structure to understand the flow better.
# 

# %%


# %% [markdown]
# ### Running the Graph
# 

# %%


# %% [markdown]
# ## Approach 2: Separate Decision Node
# 
# In this approach, we'll use a dedicated decision node that acts as a router. This pattern separates the processing logic from the routing logic, which can be useful for more complex decision-making scenarios.
# 
# ### Node Functions
# 

# %%


# %% [markdown]
# ### Building the Graph with Decision Node
# 
# This time we'll create a more explicit flow: `init → add → decide → (add or END)`
# 

# %%


# %% [markdown]
# ### Visualizing the Second Graph
# 

# %%


# %% [markdown]
# ### Running the Second Graph
# 

# %%



