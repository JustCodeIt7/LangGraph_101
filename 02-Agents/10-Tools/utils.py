import sqlite3

def create_mock_database():
    """Create a mock SQLite database with fake employee data for testing."""
    conn = sqlite3.connect('company.db')
    cursor = conn.cursor()

    # Create employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)

    # Insert mock data
    employees = [
        (1, 'Alice Johnson', 'Engineering', 95000, '2020-01-15'),
        (2, 'Bob Smith', 'Marketing', 75000, '2019-03-22'),
        (3, 'Carol Davis', 'Engineering', 105000, '2018-07-10'),
        (4, 'David Wilson', 'Sales', 65000, '2021-05-30'),
        (5, 'Eve Brown', 'Marketing', 80000, '2020-11-12'),
        (6, 'Frank Miller', 'Engineering', 92000, '2019-09-05'),
        (7, 'Grace Lee', 'HR', 70000, '2021-02-18'),
        (8, 'Henry Taylor', 'Sales', 72000, '2020-08-24'),
    ]

    cursor.executemany('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)', employees)
    conn.commit()
    conn.close()
