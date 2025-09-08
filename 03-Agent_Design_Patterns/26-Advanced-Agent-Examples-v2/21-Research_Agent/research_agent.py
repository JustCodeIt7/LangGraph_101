# Research Assistant Agent
# Minimal single-agent LangGraph workflow demonstrating:
# 1. Multi‑step phased planning
# 2. Tool-integrated sequential task execution
# 3. Per‑phase reflection & critique
# 4. One-shot refinement (plan adjustment) based on reflection
# 5. Explicit, transparent state management & final report synthesis
#
# Constraints:
# - <200 non-comment, non-blank code lines
# - Deterministic (no external API calls)
# - Single file; simple, readable functional decomposition

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Simulated Tool Layer (deterministic placeholders)
# ---------------------------------------------------------------------------
def search_tool(query: str) -> str:
    tokens = query.split()
    return f"SEARCH_RESULTS: {' | '.join(tokens[:6])} :: facts"

def summarize_tool(text: str) -> str:
    w = text.split()
    core = " ".join(w[:12])
    return ("SUMMARY: " + core + ("..." if len(w) > 12 else ""))

def cite_tool(fact: str) -> str:
    key = "".join(ch for ch in fact.lower() if ch.isalnum())[:10] or "ref"
    return f"[CITE:{key}]"

# ---------------------------------------------------------------------------
# Research State Dataclass
# ---------------------------------------------------------------------------
@dataclass
class ResearchState:
    topic: str
    plan: List[Dict[str, Any]] = field(default_factory=list)
    current_phase_index: int = 0
    current_task_index: int = 0
    completed: List[Dict[str, Any]] = field(default_factory=list)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    refined: bool = False
    final_report: Optional[str] = None

# ---------------------------------------------------------------------------
# Plan Generation: 5 Phases (id, title, objectives, tasks)
# ---------------------------------------------------------------------------
def generate_plan(topic: str) -> List[Dict[str, Any]]:
    blueprint = [
        ("scoping", "Clarify scope & baseline",
         ["search baseline context", "identify key subtopics", "draft initial angle"]),
        ("collection", "Gather structured evidence",
         ["search academic sources", "search industry reports", "review recent data"]),
        ("analysis", "Synthesize & compare findings",
         ["summary key patterns", "compare contrasting viewpoints", "assess reliability"]),
        ("impacts", "Extract implications & gaps",
         ["analyze practical impacts", "identify limitations", "map future opportunities"]),
        ("synthesis", "Integrate & recommend",
         ["summary integrated narrative", "draft recommendations", "prepare next steps"])
    ]
    phases: List[Dict[str, Any]] = []
    for pid, title, tasks in blueprint:
        phases.append({
            "id": pid,
            "title": title,
            "objectives": [title, f"Advance understanding of {topic}"],
            "tasks": tasks
        })
    return phases

# ---------------------------------------------------------------------------
# Node: Plan creation (idempotent)
# ---------------------------------------------------------------------------
def plan_node(state: ResearchState) -> ResearchState:
    if not state.plan:
        state.plan = generate_plan(state.topic)
        state.current_phase_index = 0
        state.current_task_index = 0
    return state

# ---------------------------------------------------------------------------
# Helper: Tool routing heuristics
# ---------------------------------------------------------------------------
def run_tool_for_task(task: str, topic: str, prior_outputs: List[Dict[str, Any]]) -> str:
    t = task.lower()
    if "search" in t:
        return search_tool(f"{topic} {task}")
    if "compare" in t or "contrast" in t:
        merged = " ".join(c['output'] for c in prior_outputs[-3:])
        return summarize_tool(merged or topic) + " " + cite_tool(task)
    if "summary" in t or "summarize" in t:
        merged = " ".join(c['output'] for c in prior_outputs[-2:])
        return summarize_tool(merged or topic)
    if "analyze" in t or "assess" in t or "review" in t:
        base = " ".join(c['output'] for c in prior_outputs[-2:])
        return summarize_tool(base or topic) + " ANALYSIS"
    if "draft" in t or "prepare" in t or "identify" in t or "map" in t:
        return search_tool(f"{task} {topic}") + " " + cite_tool(task)
    return summarize_tool(task + " " + topic)

# ---------------------------------------------------------------------------
# Node: Execute single task
# ---------------------------------------------------------------------------
def execute_task_node(state: ResearchState) -> ResearchState:
    phase = state.plan[state.current_phase_index]
    tasks = phase["tasks"]
    if state.current_task_index >= len(tasks):
        return state  # Safeguard; flow control will handle
    task = tasks[state.current_task_index]
    output = run_tool_for_task(task, state.topic, state.completed)
    state.completed.append({
        "phase_id": phase["id"],
        "task": task,
        "output": output
    })
    state.current_task_index += 1
    return state

# ---------------------------------------------------------------------------
# Node: Reflection after phase completion
# ---------------------------------------------------------------------------
def reflection_node(state: ResearchState) -> ResearchState:
    phase = state.plan[state.current_phase_index]
    phase_tasks = phase["tasks"]
    done = [c for c in state.completed if c["phase_id"] == phase["id"]]
    coverage = len(done) / max(1, len(phase_tasks))
    critique = ("Strong coverage" if coverage > 0.9 else
                "Adequate coverage" if coverage > 0.6 else
                "Limited coverage; broaden evidence")
    missing_compare = not any("compare" in t.lower() for t in phase_tasks)
    adjustment = ""
    if missing_compare and not state.refined:
        adjustment = "Consider adding cross-compare sources task"
    state.reflections.append({
        "phase_id": phase["id"],
        "critique": critique,
        "adjustments": adjustment
    })
    # Advance to next phase
    state.current_phase_index += 1
    state.current_task_index = 0
    return state

# ---------------------------------------------------------------------------
# Node: Optional one-shot plan refinement based on reflections
# ---------------------------------------------------------------------------
def refine_node(state: ResearchState) -> ResearchState:
    if state.refined:
        return state
    if not state.reflections:
        return state
    last = state.reflections[-1]
    if "cross-compare" not in last.get("adjustments", "").lower():
        return state
    # Insert a new task in each remaining phase if absent
    for idx in range(state.current_phase_index, len(state.plan)):
        tasks = state.plan[idx]["tasks"]
        if not any("cross-compare sources" in t for t in tasks):
            # Insert after first task for visibility
            tasks.insert(1 if len(tasks) > 1 else 0, "cross-compare sources")
    state.refined = True
    return state

# ---------------------------------------------------------------------------
# Node: Final report synthesis
# ---------------------------------------------------------------------------
def report_node(state: ResearchState) -> ResearchState:
    # Aggregate simple evidence slices
    by_phase: Dict[str, List[str]] = {}
    for c in state.completed:
        by_phase.setdefault(c["phase_id"], []).append(c["output"])
    sections = []
    sections.append(f"# Research Report: {state.topic}")
    sections.append("## Introduction")
    sections.append(f"This report explores {state.topic} through a structured, phased inquiry.")
    sections.append("## Methodology")
    sections.append("A deterministic multi-phase plan executed simulated search, comparison, and summarization tools with a single refinement opportunity.")
    sections.append("## Key Findings")
    for ph in state.plan:
        outputs = by_phase.get(ph["id"], [])
        snippet = summarize_tool(" ".join(outputs)) if outputs else "No data."
        sections.append(f"- {ph['title']}: {snippet}")
    sections.append("## Synthesized Insights")
    insights_base = " ".join(c['output'] for c in state.completed[-6:])
    sections.append(summarize_tool(insights_base or state.topic))
    sections.append("## Limitations")
    sections.append("Findings derive from simulated tools; real-world variability and source validation are absent.")
    sections.append("## Recommended Next Steps")
    steps = ["Augment with real literature retrieval",
             "Quantitatively validate key claims",
             "Engage domain experts for qualitative review"]
    sections.extend([f"- {s}" for s in steps])
    sections.append("## Provenance Summary")
    sections.append(f"Phases executed: {len(state.plan)} | Tasks completed: {len(state.completed)} | Refined: {state.refined}")
    state.final_report = "\n".join(sections)
    return state

# ---------------------------------------------------------------------------
# Routing Logic Helpers (pure functions)
# ---------------------------------------------------------------------------
def route_after_execute(state: ResearchState) -> str:
    phase = state.plan[state.current_phase_index]
    if state.current_task_index < len(phase["tasks"]):
        return "execute"
    return "reflect"

def route_after_reflection(state: ResearchState) -> str:
    if state.current_phase_index >= len(state.plan):
        return "report"
    return "refine"

def route_after_refine(state: ResearchState) -> str:
    if state.current_phase_index >= len(state.plan):
        return "report"
    return "execute"

# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------
def build_graph() -> Any:
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_task_node)
    g.add_node("reflect", reflection_node)
    g.add_node("refine", refine_node)
    g.add_node("report", report_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", route_after_execute,
                            {"execute": "execute", "reflect": "reflect"})
    g.add_conditional_edges("reflect", route_after_reflection,
                            {"refine": "refine", "report": "report"})
    g.add_conditional_edges("refine", route_after_refine,
                            {"execute": "execute", "report": "report"})
    g.add_edge("report", END)
    return g.compile()

# ---------------------------------------------------------------------------
# Convenience Runner
# ---------------------------------------------------------------------------
def run_agent(topic: str) -> Dict[str, Any]:
    workflow = build_graph()
    init = ResearchState(topic=topic)
    final: ResearchState = workflow.invoke(init)
    return {
        "report": final.final_report,
        "plan": final.plan,
        "reflections": final.reflections,
        "completed_count": len(final.completed),
        "refined": final.refined
    }

# ---------------------------------------------------------------------------
# __main__ entry (sample)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_agent("Impacts of urban green roofs on microclimate")
    print(result["report"])