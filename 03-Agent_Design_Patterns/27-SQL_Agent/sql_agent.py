from __future__ import annotations

import re
import sqlite3
import sys
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph importjbmour
jbmour
END, START, StateGraph

DB = 'tutorial_sales.db'
MODEL = 'qwen3:8b'  # try llama3.1:8b or qwen2.5-coder:7b
TOP_K = 5


def setup_db() -> None:
    with sqlite3.connect(DB) as con:
        con.executescript("""
        DROP TABLE IF EXISTS jobs; DROP TABLE IF EXISTS services; DROP TABLE IF EXISTS customers;
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, industry TEXT);
        CREATE TABLE services(service_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE jobs(job_id INTEGER PRIMARY KEY, customer_id INTEGER, service_id INTEGER,
            quantity INTEGER, job_date TEXT, status TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(service_id) REFERENCES services(service_id));
        """)
        con.executemany(
            'INSERT INTO customers VALUES (?, ?, ?, ?)',
            [
                (1, 'Acme Retail', 'Cincinnati', 'Retail'),
                (2, 'Blue Sky Bank', 'Columbus', 'Financial'),
                (3, 'Northstar Foods', 'Cleveland', 'Food Service'),
                (4, 'Metro Health', 'Dayton', 'Healthcare'),
            ],
        )
        con.executemany(
            'INSERT INTO services VALUES (?, ?, ?, ?)',
            [
                (1, 'ATM Install', 'Field Service', 950),
                (2, 'Locker Install', 'Field Service', 725),
                (3, 'Storage', 'Logistics', 125),
                (4, 'Site Survey', 'Preflight', 275),
                (5, 'White Glove Delivery', 'Logistics', 450),
            ],
        )
        con.executemany(
            'INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)',
            [
                (1, 1, 2, 3, '2026-01-15', 'paid'),
                (2, 1, 4, 3, '2026-01-15', 'paid'),
                (3, 2, 1, 2, '2026-02-02', 'paid'),
                (4, 2, 3, 6, '2026-02-02', 'paid'),
                (5, 2, 5, 2, '2026-02-18', 'paid'),
                (6, 3, 3, 12, '2026-03-05', 'open'),
                (7, 3, 4, 1, '2026-03-05', 'open'),
                (8, 4, 2, 2, '2026-03-22', 'paid'),
                (9, 4, 5, 1, '2026-03-22', 'paid'),
            ],
        )


@tool
def list_tables() -> str:
    """Return all table names in the SQLite database."""
    with sqlite3.connect(DB) as con:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return ', '.join(row[0] for row in rows)


@tool
def get_schema(table_names: str) -> str:
    """Return schemas and sample rows for comma-separated table names."""
    chunks = []
    with sqlite3.connect(DB) as con:
        valid = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in [t.strip() for t in table_names.split(',') if t.strip()]:
            if table not in valid:
                chunks.append(f'Missing table: {table}')
                continue
            q = '"' + table.replace('"', '""') + '"'
            ddl = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
            sample = con.execute(f'SELECT * FROM {q} LIMIT 3')
            chunks.append(f'{ddl}\nColumns: {[d[0] for d in sample.description]}\nRows: {sample.fetchall()}')
    return '\n\n'.join(chunks)


def safe_select(sql: str) -> bool:
    q = sql.strip().rstrip(';').lower()
    blocked = r'\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma)\b'
    return (q.startswith('select') or q.startswith('with')) and not re.search(blocked, q)


@tool
def run_sql(sql: str) -> str:
    """Run a read-only SQL query and return rows as dictionaries."""
    if not safe_select(sql):
        return 'Blocked: only read-only SELECT/WITH queries are allowed.'
    try:
        with sqlite3.connect(DB) as con:
            con.row_factory = sqlite3.Row
            return str([dict(row) for row in con.execute(sql).fetchall()])
    except Exception as e:
        return f'SQL error: {e}'


def clean_sql(text: str) -> str:
    text = text.strip().replace('```sql', '').replace('```', '').strip()
    match = re.search(r'\b(with|select)\b[\s\S]*', text, re.I)
    text = match.group(0) if match else text
    return text.split(';')[0].strip() + ';'


class State(TypedDict):
    question: str
    schema: str
    sql: str
    result: str
    answer: str


llm = ChatOllama(model=MODEL, temperature=0)
parser = StrOutputParser()

write_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a SQLite expert. Write ONE safe read-only query. Use only this schema. Limit results to {top_k} rows unless asked otherwise.\n\n{schema}',
        ),
        ('human', 'Question: {question}\nReturn SQL only, no markdown.'),
    ]
)
check_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Fix this SQLite query if needed. Keep it read-only. Return SQL only.'),
        ('human', 'Schema:\n{schema}\n\nSQL:\n{sql}'),
    ]
)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Answer the user from the SQL result. Be concise and mention key numbers.'),
        ('human', 'Question: {question}\nSQL: {sql}\nResult: {result}'),
    ]
)


def inspect_db(state: State) -> dict:
    tables = list_tables.invoke({})
    return {'schema': get_schema.invoke({'table_names': tables})}


def write_sql(state: State) -> dict:
    raw = (write_prompt | llm | parser).invoke({'question': state['question'], 'schema': state['schema'], 'top_k': TOP_K})
    return {'sql': clean_sql(raw)}


def check_sql(state: State) -> dict:
    raw = (check_prompt | llm | parser).invoke({'schema': state['schema'], 'sql': state['sql']})
    return {'sql': clean_sql(raw)}


def execute_sql(state: State) -> dict:
    return {'result': run_sql.invoke({'sql': state['sql']})}


def answer(state: State) -> dict:
    return {'answer': (answer_prompt | llm | parser).invoke(state)}


def build_graph():
    graph = StateGraph(State)
    for name, node in [('inspect_db', inspect_db), ('write_sql', write_sql), ('check_sql', check_sql), ('execute_sql', execute_sql), ('answer', answer)]:
        graph.add_node(name, node)
    graph.add_edge(START, 'inspect_db')
    graph.add_edge('inspect_db', 'write_sql')
    graph.add_edge('write_sql', 'check_sql')
    graph.add_edge('check_sql', 'execute_sql')
    graph.add_edge('execute_sql', 'answer')
    graph.add_edge('answer', END)
    return graph.compile()


if __name__ == '__main__':
    setup_db()
    question = ' '.join(sys.argv[1:]) or 'Which customer spent the most money?'
    state = build_graph().invoke({'question': question})
    print('\nQUESTION:', question)
    print('\nSQL:\n', state['sql'])
    print('\nRESULT:\n', state['result'])
    print('\nANSWER:\n', state['answer'])
