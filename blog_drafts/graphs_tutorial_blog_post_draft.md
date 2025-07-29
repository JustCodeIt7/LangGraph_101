# Exploring LangGraph Graphs

LangGraph makes it simple to build stateful workflows around large language models. The **01-Graphs** tutorials walk through the essential features step by step.

## Graph Basics
These examples introduce the `StateGraph` concept and show how to define a state schema, create nodes, and compile a graph into a runnable application. The tutorial culminates in a tiny graph that greets the user.

## Handling Multiple Inputs
Next, you'll see how to design a graph that consumes several pieces of state at once. The example grade calculator demonstrates reading multiple fields from the state to perform a computation, and how to visualize the graph.

## Conditional Routing
Here we learn how router functions can alter the flow depending on the state. The conditional routing examples show both number processing and text processing workflows that take different branches based on custom logic.

## Looping Logic
LangGraph also supports loops. You'll experiment with looping directly back to a node or using a decision node to break out once a condition is met. The tutorial adds random numbers to a list until a threshold is reached, generating diagrams for each approach.

## Basic Chat Graph
Finally, all of the concepts come together in a simple command line chat app. The state tracks conversation history while nodes parse user input, generate LLM responses, and handle commands like `exit` or `verbose`.

These tutorials form the foundation for building more complex LangGraph applications.
