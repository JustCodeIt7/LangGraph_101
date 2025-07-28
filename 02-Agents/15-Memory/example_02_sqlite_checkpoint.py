"""
example_02_sqlite_checkpoint.py  ────────────────────────────────────
A drop‑in SQLite replacement for InMemorySaver. It serializes the
entire checkpoint tuple as JSON so you can reopen past sessions.

Run:
    python example_02_sqlite_checkpoint.py
"""

from __future__ import annotations
import json, sqlite3, uuid, datetime as _dt
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseSaver, CheckpointTuple
from langgraph.graph import StateGraph, MessagesState, START


# ── ❶ Minimal SQLite checkpointer ──────────────────────────────────
class TinySQLiteSaver(BaseSaver):
    """
    Lightweight drop‑in saver that persists LangGraph checkpoints in
    a single SQLite table called `ckpt`.
    """

    def __init__(self, path: str = 'checkpoints.db'):
        self._cx = sqlite3.connect(path, check_same_thread=False)
        self.setup()

    # YT: highlight table schema creation
    def setup(self) -> None:  # idempotent
        self._cx.execute('CREATE TABLE IF NOT EXISTS ckpt(id TEXT PRIMARY KEY, thread TEXT, ts TEXT, data TEXT)')
        self._cx.commit()

    # -- BaseSaver interface ----------------------------------------
    def put_tuple(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> str:
        _id = checkpoint.get('id') or str(uuid.uuid4())
        row = (
            _id,
            config['configurable']['thread_id'],
            _dt.datetime.utcnow().isoformat(),
            json.dumps(checkpoint),
        )
        self._cx.execute('REPLACE INTO ckpt VALUES (?,?,?,?)', row)
        self._cx.commit()
        return _id

    def get_tuple(self, config: Dict[str, Any]) -> CheckpointTuple | None:
        tid = config['configurable']['thread_id']
        row = self._cx.execute('SELECT data FROM ckpt WHERE thread=? ORDER BY ts DESC LIMIT 1', (tid,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete_thread(self, thread_id: str) -> None:
        self._cx.execute('DELETE FROM ckpt WHERE thread=?', (thread_id,))
        self._cx.commit()


# ── ❷ Tiny chat graph (same as Example 1) ──────────────────────────
def talk(state: MessagesState):
    reply = ChatOpenAI(model_name='gpt-3.5-turbo').invoke(state['messages'])
    return {'messages': [reply]}


builder = StateGraph(MessagesState)
builder.add_node(talk)
builder.add_edge(START, 'talk')
graph = builder.compile(checkpointer=TinySQLiteSaver())  # ★ local DB

# ── ❸ Demo two separate runs sharing the same DB ───────────────────
if __name__ == '__main__':
    cfg = {'configurable': {'thread_id': 'persistent'}}
    graph.invoke({'messages': 'Remember my favorite color is teal'}, cfg)
    # YT: stop recording, restart script, ask the question again
