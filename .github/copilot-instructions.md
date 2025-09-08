# AI Assistant Working Agreements for LangGraph_101

Concise, project-specific guidance to help AI coding agents be productive quickly. Focus on actual patterns in this repo—avoid generic boilerplate.

## 1. Repository Purpose & Learning Flow

This repo is an educational sandbox for mastering LangGraph: building stateful, multi-agent, tool-using LLM workflows. Content progresses: Intro ➜ Graph primitives ➜ Agents & patterns ➜ Full apps. Many examples live only in notebooks / folders; there is no monolithic package.

High-value directories:

- `01-Graphs/` – Core graph patterns (multiple inputs, conditional routing, loops, chat graph)
- `02-Agents/` – Agent state, tools, streaming, memory, multi-agent coordination, MCP
- `03-Agent_Design_Patterns/` – RAG agent, orchestration, multi-agent routing/parallelism
- `03-Apps/` – End-to-end applied agents (finance, travel, stock, support, etc.)
- `research_agent/` – A concrete Python module-style agent example (importable code)
- `core_architecture_standards_for_python.md` – General architectural principles (reference only; don’t enforce retroactively unless editing related code)

## 2. Tech & Dependencies

Lightweight Python project (no heavy framework). Key libs actually used:

- `langgraph`, `langchain-core`, `langchain`, provider-specific modules (`langchain_openai`, `langchain_ollama`, `litellm`)
- `rich` for pretty console output
- `yfinance` in finance/stock agents
- `python-dotenv` for local config
  No central `src/` package; examples are script/notebook oriented. Keep additions similarly minimal unless creating reusable utilities.

## 3. Patterns & Conventions

State & Flow:

- Prefer `StateGraph` with strongly-typed state via `TypedDict` / Pydantic style or custom dataclass-like classes when scaling.
- Message-based state uses `MessagesState` + list of `HumanMessage`/`AIMessage` objects.
- Routing via returning `Command(goto=..., update={...})` from node functions.
- Multi-agent patterns: supervisor router, peer network, hierarchical subgraphs (see `02-Agents/19-Multi_Agent` and related pattern folders). When adding new examples, mirror naming: `something_agent`, `*_team`, `supervisor`.

Code Style:

- Keep examples short & didactic; prioritize readability over abstraction.
- Inline small prompts; avoid giant prompt builder frameworks.
- Use explicit model names (e.g., `ollama/llama3.2`, `gpt-4o-mini`) rather than indirection.

File Layout:

- Each concept folder self-contained (code, optional output/visualization). Do NOT centralize utilities unless duplication becomes noisy (>3 repeats across distinct folders).
- When adding a new pattern, create a clearly numbered directory that fits progression.

## 4. Execution & Dev Workflow

Typical usage is: open a notebook or run a script directly.
General workflow:

1. Set environment variables (API keys) via `.env` (create if needed). Example keys: `OPENAI_API_KEY`, OLLAMA running locally.
2. Install deps: `pip install -r requirements.txt` (or editable: `pip install -e .` for experimentation with local module pieces).
3. Run a sample: `python research_agent/agent.py` or open a notebook in Jupyter.
   No test suite present—if adding one, colocate under `tests/` with pytest and keep fast.

## 5. Adding New Material (Important)

When you generate new examples:

- Provide a top-of-file docstring summarizing the learning objective.
- Show minimal runnable path (define state ➜ nodes ➜ build graph ➜ invoke once).
- If multi-agent: demonstrate at least two specializations + a router.
- If tool use: clearly isolate tool definition vs model invocation.
- Avoid hidden global side effects (no implicit singletons). Pass config via function parameters or small config objects if needed.

## 6. Prompts & Models

- Keep prompts concise and instructional ("Answer briefly", "Return final numeric result only").
- For math/tool reliability, strip extraneous formatting from model outputs before storing in state.
- If adding streaming examples, show both standard invoke and a streaming iterator.

## 7. Data / External Calls

- Finance agents use `yfinance`—do not hardcode live responses; keep logic resilient to missing fields.
- Any retrieval/RAG example should inline tiny mock corpus unless demonstrating a real vector store (then clearly mark dependency).

## 8. Quality & Safety Boundaries

- Do not introduce network calls beyond what existing examples illustrate without a comment explaining purpose.
- Keep dependencies lean—avoid adding new major libs (vector DBs, web frameworks) unless explicitly requested.
- Retain MIT license header when copying substantial blocks.

## 9. Useful Snippet Template

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

class ChatState(MessagesState):
    pass

def echo(state: ChatState):
    last = [m for m in state["messages"] if getattr(m, "type", getattr(m, "role", "")) in ("human","user")][-1]
    return Command(goto=END, update={"messages": [AIMessage(content=f"You said: {last.content}")]})

b = StateGraph(ChatState)
b.add_node("echo", echo)
b.add_edge(START, "echo")
app = b.compile()
print(app.invoke({"messages": [HumanMessage("Hi!")]})["messages"][-1].content)
```

## 10. When Unsure

Prefer showing a minimal new folder example mirroring existing naming. Ask for clarification only if architectural direction truly ambiguous.

---

End of instructions. Refine after feedback if any section feels incomplete.
