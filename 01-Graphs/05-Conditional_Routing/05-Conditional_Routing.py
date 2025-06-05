# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: py312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LangGraph 101: Conditional Routing
#
# ## Imports and State Definition
#


# %%
# ~ Imports
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from rich import print

# %% [markdown]
# ## State Definition
#

# %%
class NumberState(TypedDict):
    number: int
    result: int

# %% [markdown]
# ## Node Functions and Router
#
# We've set up three processing nodes to handle different types of numbers:
#
# - **Positive numbers**: We square the number.
# - **Negative numbers**: We take the absolute value.
# - **Zero**: We leave it as zero.
#
# Additionally, we've built a router function that decides which path to follow based on the sign of the input number.
#

# %%



# %%


# %% [markdown]
# ## Building the Conditional Routing Graph
#
# 1.  Start → Router node
# 2.  Router evaluates the number's sign
# 3.  Routes to the appropriate processing node
# 4.  All branches lead to END
#

# %%


# %% [markdown]
# ## Visualizing the Graph
#

# %%


# %% [markdown]
# ## Testing the Conditional Routing Graph
#

# %%


# %% [markdown]
# # Example 2: Text Message Processing Router
#
# Let's create another conditional routing example using text messages. This will demonstrate how to route based on text length and apply different processing strategies.
#

# %% [markdown]
# ## Text State Definition
#

# %%
class MessageState(TypedDict):
    # * test 
    message: str
    processed_message: str
    word_count: int

# %% [markdown]
# ## Text Processing Nodes and Router
#
# We'll route messages based on their length and apply different processing:
#
# - **Short messages** (≤ 10 chars): Convert to uppercase and add exclamation
# - **Medium messages** (11-50 chars): Add a friendly greeting prefix
# - **Long messages** (> 50 chars): Summarize by counting words and truncating
#

# %%


# %%


# %% [markdown]
# ## Building the Text Processing Graph
#

# %%


# %% [markdown]
# ## Visualizing the Text Processing Graph
#

# %%


# %% [markdown]
# ## Testing the Text Processing Router
#

# %%

