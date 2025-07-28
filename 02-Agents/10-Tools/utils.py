import sqlite3
import pathlib
import os

# set working directory to the current file's directory
pathlib.Path(__file__).parent.resolve()
os.chdir(pathlib.Path(__file__).parent.resolve())

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
        (9, 'Ivy Anderson', 'Engineering', 98000, '2018-12-01'),
        (10, 'Jack Thomas', 'Marketing', 77000, '2021-04-15'),
        (11, 'Kathy White', 'HR', 68000, '2019-10-30'),
        (12, 'Liam Harris', 'Sales', 69000, '2020-06-20'),
        (13, 'Mia Clark', 'Engineering', 91000, '2018-08-14'),
        (14, 'Noah Lewis', 'Marketing', 76000, '2021-01-10'),
        (15, 'Olivia Walker', 'HR', 72000, '2020-03-05'),
        (16, 'Paul Young', 'Sales', 73000, '2019-11-25'),
        (17, 'Quinn King', 'Engineering', 94000, '2018-06-30'),
        (18, 'Rita Scott', 'Marketing', 78000, '2021-07-15'),
        (19, 'Sam Green', 'HR', 71000, '2020-02-20'),
        (20, 'Tina Adams', 'Sales', 70000, '2019-04-10'),
        (21, 'Uma Baker', 'Engineering', 90000, '2018-10-05'),
        (22, 'Vera Nelson', 'Marketing', 74000, '2021-08-01'),
        (23, 'Will Carter', 'HR', 73000, '2020-12-15'),
        (24, 'Xena Mitchell', 'Sales', 72000, '2019-03-30'),
        (25, 'Yara Perez', 'Engineering', 95000, '2018-05-20'),
        (26, 'Zane Roberts', 'Marketing', 77000, '2021-09-10'),
        (27, 'Aaron Turner', 'HR', 70000, '2020-04-25'),
        (28, 'Bella Phillips', 'Sales', 68000, '2019-07-05'),
    ]

    cursor.executemany('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)', employees)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_mock_database()
    print('Mock database created with employee data.')