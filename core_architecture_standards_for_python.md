# Core Architecture Standards for Python

> From [CodingRules.ai](https://codingrules.ai/rules/core-architecture-standards-for-python)

**Tags:** Python

---

# Core Architecture Standards for Python

This document outlines the core architectural standards for Python development. It provides guidance on fundamental design patterns, project structure, and organization principles, tailored for optimal maintainability, performance, and security. These standards are informed by the latest Python version and best practices within the Python ecosystem.

## 1. Project Structure and Organization

A well-defined project structure is crucial for maintainability, scalability, and collaboration. This section provides guidelines on how to organize your Python projects.

### 1.1. Standard Directory Layout

**Do This:** Adopt a standard directory layout to ensure consistency across projects.

**Why:** A consistent structure makes it easier for developers to navigate and understand the codebase, regardless of the project. It facilitates code reuse and simplifies deployment.

**Example:**

"""
project_name/
├── pyproject.toml              # Project metadata and dependencies (PEP 621)
├── src/                        # Source code
│   └── project_name/           # Main application package
│       ├── __init__.py        # Initializes the package
│       ├── module1.py         # Module files
│       ├── module2.py
│       └── ...
├── tests/                      # Unit and integration tests
│   ├── __init__.py            # Initializes the test package
│   ├── test_module1.py        # Tests for module1
│   ├── test_module2.py        # Tests for module2
│   └── ...
├── docs/                       # Documentation
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst              # Root documentation file
│   └── ...
├── .gitignore                   # Specifies intentionally untracked files that Git should ignore
├── README.md                    # Project description and usage instructions
├── LICENSE                      # Project license
└── requirements.txt             # Historical Dependency management (can be used alongside pyproject.toml)
"""

**Don't Do This:** Use a flat or disorganized directory structure.  Avoid scattering source code files directly in the root directory.

### 1.2. Package Design

**Do This:** Break down your application into smaller, cohesive packages and modules.

**Why:** Modularity enhances code reusability, simplifies testing, and reduces the complexity of individual components.

**Example:**

Suppose you're building a system for managing user accounts. Structure your packages like this:

"""
src/
└── user_management/
    ├── __init__.py
    ├── models/          # Data models (e.g., User, Profile)
    │   ├── __init__.py
    │   ├── user.py
    │   └── profile.py
    ├── views/           # API endpoints or UI components
    │   ├── __init__.py
    │   ├── user_views.py
    │   └── profile_views.py
    ├── services/        # Business logic (e.g., user registration, authentication)
    │   ├── __init__.py
    │   ├── user_service.py
    │   └── auth_service.py
    └── utils/           # Utility functions (e.g., email sending, password hashing)
        ├── __init__.py
        ├── email_utils.py
        └── password_utils.py
"""

**Don't Do This:** Create monolithic packages with tightly coupled components. Avoid circular dependencies between packages.

### 1.3. Layered Architecture

**Do This:** Apply a layered architecture (e.g., presentation, business logic, data access) to separate concerns.

**Why:** Layered architecture improves maintainability, testability, and flexibility.  Changes in one layer have minimal impact on other layers.

**Example:**

*   **Presentation Layer:** Handles user interaction and presents data (e.g., web UI, API endpoints).
*   **Business Logic Layer:** Contains the core application logic and rules.
*   **Data Access Layer:** Manages data persistence and retrieval (e.g., database operations, file system interactions).

**Code Example (simplified):**

"""python
# data_access/user_repository.py
class UserRepository:
    def get_user_by_id(self, user_id: int):
        # Database interaction code using SQLAlchemy or similar
        ...

# business_logic/user_service.py
from data_access.user_repository import UserRepository

class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def get_user(self, user_id: int):
        return self.user_repository.get_user_by_id(user_id)

# presentation/user_api.py
from business_logic.user_service import UserService
from data_access.user_repository import UserRepository

user_repository = UserRepository()
user_service = UserService(user_repository)

def get_user_api(user_id: int):
    user = user_service.get_user(user_id)
    # Return user data as JSON or HTML
    ...
"""

**Don't Do This:** Mix presentation logic directly with business logic or data access code. Avoid tightly coupling layers.

### 1.4. Dependency Management with "pyproject.toml" and "poetry/pip"

**Do This:** Use "pyproject.toml" (PEP 621) for managing dependencies with tools like Poetry or Pip. Specify exact versions to avoid unexpected behavior due to dependency updates.  Consider using virtual environments.

**Why:**  "pyproject.toml" provides a standardized way to manage project metadata and dependencies, ensuring reproducibility and consistency across different environments. Tools like Poetry and Pip simplify dependency resolution and installation.

**Example ("pyproject.toml"):**

"""toml
[project]
name = "my_project"
version = "0.1.0"
description = "A description of my project"
authors = [{name = "Your Name", email = "your.email@example.com"}]
dependencies = [
    "fastapi>=0.100.0,<0.101.0",
    "uvicorn[standard]>=0.20.0,<0.21.0",
    "SQLAlchemy>=2.0.0,<2.1.0"
]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""

Using Poetry:

"""bash
poetry install
poetry add requests  # Add a new dependency
poetry update        # Update dependencies
"""

Using Pip:

"""bash
pip install -r requirements.txt
"""

**Don't Do This:** Rely on system-wide packages without explicit versioning.  Neglect to update dependencies regularly, as this often leads to security vulnerabilities.  Avoid pinning dependency versions without a good reason (understand the implications).

## 2. Design Patterns

Design patterns are reusable solutions to common software design problems. Effective use of design patterns leads to more maintainable, scalable, and understandable code.

### 2.1. Dependency Injection

**Do This:** Use dependency injection to decouple components and improve testability.

**Why:** Dependency injection allows you to inject dependencies into a class rather than creating them within the class itself. This makes it easier to swap implementations and mock dependencies for testing.

**Example:**

"""python
# Interface
class MessageService:
    def send(self, message: str, recipient: str):
        raise NotImplementedError

# Concrete implementation
class EmailService(MessageService):
    def send(self, message: str, recipient: str):
        print(f"Sending email to {recipient}: {message}")
        # Code to send email using an email library

# Client class
class NotificationService:
    def __init__(self, message_service: MessageService):
        self.message_service = message_service

    def send_notification(self, message: str, user_email: str):
        self.message_service.send(message, user_email)

# Usage
email_service = EmailService()
notification_service = NotificationService(email_service)
notification_service.send_notification("Hello, user!", "user@example.com")

# For testing, you can inject a mock MessageService
class MockMessageService(MessageService):
    def send(self, message: str, recipient: str):
        print(f"Mock sending message to {recipient}: {message}")

mock_service = MockMessageService()
notification_service = NotificationService(mock_service)
notification_service.send_notification("Test message", "test@example.com")

"""

**Don't Do This:** Hardcode dependencies within classes. Create tight coupling between components.

### 2.2. Factory Pattern

**Do This:** Use the factory pattern to create objects without specifying their concrete classes.

**Why:** The factory pattern decouples object creation from the client code, allowing you to change the concrete class being instantiated without modifying the client code. This enhances flexibility and maintainability, especially when handling complex object creation logic.

**Example:**

"""python
from abc import ABC, abstractmethod

# Abstract Product
class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

# Concrete Products
class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Abstract Factory
class AnimalFactory(ABC):
    @abstractmethod
    def create_animal(self):
        pass

# Concrete Factories
class DogFactory(AnimalFactory):
    def create_animal(self):
        return Dog()

class CatFactory(AnimalFactory):
    def create_animal(self):
        return Cat()

# Client Code
def make_animal_speak(factory: AnimalFactory):
    animal = factory.create_animal()
    return animal.speak()

# Usage
dog_factory = DogFactory()
cat_factory = CatFactory()

print(make_animal_speak(dog_factory))  # Output: Woof!
print(make_animal_speak(cat_factory))  # Output: Meow!

# An alternative use of the factory pattern, using a simple function

def create_animal(animal_type: str) -> Animal:
    if animal_type == "dog":
        return Dog()
    elif animal_type == "cat":
        return Cat()
    else:
        raise ValueError("Invalid animal type")

dog = create_animal("dog")
print(dog.speak())
"""

**Don't Do This:**  Directly instantiate concrete classes throughout your code, creating tight coupling and hindering maintainability.

### 2.3. Observer Pattern

**Do This:** Implement the observer pattern to define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**Why:** The observer pattern facilitates loose coupling between objects. It's useful for implementing event-driven systems, pub-sub mechanisms, and reactive programming.

**Example:**

"""python
from abc import ABC, abstractmethod

# Subject (Observable)
class Subject(ABC):
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self)

# Observer (Abstract Observer)
class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

# Concrete Subject
class ConcreteSubject(Subject):
    def __init__(self):
        super().__init__()
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

# Concrete Observers
class ConcreteObserverA(Observer):
    def update(self, subject):
        print("ConcreteObserverA: Subject's state has changed to", subject.state)

class ConcreteObserverB(Observer):
    def update(self, subject):
        print("ConcreteObserverB: Subject's state has changed to", subject.state)

# Usage
subject = ConcreteSubject()

observer_a = ConcreteObserverA()
observer_b = ConcreteObserverB()

subject.attach(observer_a)
subject.attach(observer_b)

subject.state = "New State"
# Output:
# ConcreteObserverA: Subject's state has changed to New State
# ConcreteObserverB: Subject's state has changed to New State

subject.detach(observer_a)

subject.state = "Another State"
# Output:
# ConcreteObserverB: Subject's state has changed to Another State
"""

**Don't Do This:** Implement tight coupling between subjects and observers. Directly modify observer state within the subject.

## 3. Concurrency & Asynchronous Programming

Python's asynchronous programming capabilities have improved significantly, and proper use is essential for I/O-bound tasks.

### 3.1. "asyncio" for Asynchronous Operations

**Do This:** Use "asyncio" and "async"/"await" syntax for concurrent I/O-bound operations.

**Why:**  "asyncio" enables efficient concurrency without the overhead of threads. This is crucial for applications that handle many concurrent connections or perform frequent I/O operations.

**Example:**

"""python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://www.example.com",
        "https://www.python.org",
        "https://realpython.com"
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

        for url, content in zip(urls, results):
            print(f"Content from {url}: {content[:50]}...")  # Print first 50 characters

if __name__ == "__main__":
    asyncio.run(main())
"""

**Don't Do This:** Use blocking I/O operations in the main thread of your application. Avoid mixing synchronous and asynchronous code without careful consideration.

### 3.2. Threading and Multiprocessing for CPU-Bound Tasks

**Do This:**  Use "threading" for I/O-bound tasks where concurrency is needed, and "multiprocessing" for CPU-bound tasks to achieve true parallelism.

**Why:**  The Global Interpreter Lock (GIL) in CPython prevents true parallelism for CPU-bound tasks with threads.  "multiprocessing" bypasses the GIL by creating separate processes.

**Example (Multiprocessing):**

"""python
import multiprocessing
import time

def square(number):
    result = number * number
    print(f"Square of {number} is {result}")
    time.sleep(1) #simulate intensive operation
    return result

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(square, numbers)
    print ("All squares calculated")
    print(results)
"""

**Don't Do This:** Overuse threads for CPU-bound tasks, expecting significant performance gains. Neglect to properly manage shared resources when using threads or processes, leading to race conditions or deadlocks.

### 3.3.  Asynchronous Context Managers

**Do This:** Use asynchronous context managers when working with resources in asynchronous code.

**Why:**  Asynchronous context managers (using "async with") ensure proper resource management in asynchronous environments, similar to regular context managers (using "with") in synchronous code.

**Example:**

"""python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource...")
        await asyncio.sleep(1)  # Simulate resource acquisition
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource...")
        await asyncio.sleep(1)  # Simulate resource release

    async def do_something(self):
        print("Doing something with the resource...")
        await asyncio.sleep(0.5)

async def main():
    async with AsyncResource() as resource:
        await resource.do_something()

asyncio.run(main())
"""

**Don't Do This:**  Manage resources manually in asynchronous code without using asynchronous context managers.
## 4. Exception Handling

Robust exception handling is crucial for preventing application crashes and providing informative error messages.

### 4.1. Specific Exception Handling

**Do This:** Catch specific exceptions rather than broad exception classes like "Exception".

**Why:** Catching specific exceptions allows you to handle different error conditions in a targeted manner.  Broad exception handling can mask unexpected errors and make debugging difficult.

**Example:**

"""python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: Cannot divide by zero. {e}")
except TypeError as e:
    print(f"Error: Invalid type. {e}")
else:
    print(f"Result: {result}")
finally:
    print("Operation complete.")
"""

**Don't Do This:** Use "except Exception:" as a general catch-all. This can hide bugs and make debugging very difficult. Blindly catch exceptions and ignore them, leading to silent failures.

### 4.2. Context Managers for Resource Management

**Do This:** Use context managers ("with" statement) to ensure resources are properly released, even in the event of exceptions.

**Why:** Context managers guarantee that resources (e.g., files, network connections) are properly cleaned up, preventing resource leaks and ensuring data integrity.

**Example:**

"""python
try:
    with open("my_file.txt", "r") as f:
        content = f.read()
        # Process content
except FileNotFoundError as e:
    print(f"Error: File not found. {e}")
except IOError as e:
    print(f"Error: I/O error. {e}")
"""

**Don't Do This:** Manually open and close resources without using context managers. Rely on garbage collection to release resources.

### 4.3. Custom Exceptions

**Do This:** Define custom exception classes for application-specific error conditions.

**Why:** Custom exceptions provide more context and clarity for error handling. They make it easier to differentiate between different types of errors and handle them appropriately.

**Example:**

"""python
class InsufficientFundsError(Exception):
    """Raised when an account has insufficient funds for a transaction."""
    pass

def withdraw(account, amount):
    if account.balance < amount:
        raise InsufficientFundsError(f"Insufficient funds: Balance = {account.balance}, Amount = {amount}")
    account.balance -= amount
    return account.balance

class Account:
  def __init__(self, balance: int):
    self.balance = balance

try:
    my_account = Account(100)
    new_balance = withdraw(my_account, 200)
    print(f"New balance: {new_balance}")
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")
"""

**Don't Do This:**  Rely solely on built-in exception classes for all error conditions. Reuse generic exceptions for unrelated error conditions.

## 5. Security Best Practices

Security should be a primary concern throughout the application development lifecycle.

### 5.1. Input Validation and Sanitization

**Do This:** Validate and sanitize all user inputs to prevent injection attacks (e.g., SQL injection, cross-site scripting).

**Why:** Untrusted user input can be exploited to execute arbitrary code or access sensitive data. Input validation and sanitization ensure that data conforms to expected formats and does not contain malicious content. Libraries such as "bleach" for sanitizing HTML and parameterized queries for database interactions can be extremely helpful.

**Example:**

"""python
import bleach

def sanitize_html(html_content):
    """Sanitize HTML content to prevent XSS attacks."""
    allowed_tags = ['p', 'a', 'strong', 'em', 'ul', 'ol', 'li'] # Restrict html to minimal set
    allowed_attributes = {'a': ['href', 'title']}
    return bleach.clean(html_content, tags=allowed_tags, attributes=allowed_attributes)

def process_user_input(user_input):
    """Process user input after sanitization."""
    sanitized_input = sanitize_html(user_input)
    # Further processing of sanitized input
    return sanitized_input

input_data = "<script>alert('XSS');</script><p>This is some <strong>safe</strong> content.</p>"
safe_data = process_user_input(input_data)
print(safe_data)
"""

**Don't Do This:** Trust user input without validation. Concatenate user input directly into SQL queries or system commands.

### 5.2. Secure Password Handling

**Do This:** Use strong password hashing algorithms (e.g., bcrypt, scrypt, Argon2) to store passwords securely. Never store passwords in plain text. Avoid using "md5" or "sha1", which are now considered weak.

**Why:** Password hashing protects user credentials in the event of a data breach. Strong hashing algorithms make it computationally infeasible for attackers to recover the original passwords from the hashes.

**Example (using "bcrypt"):**

"""python
import bcrypt

def hash_password(password):
    """Hash a password using bcrypt."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_password.decode('utf-8')

def verify_password(password, hashed_password):
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# Example Usage
password = "mysecretpassword"
hashed = hash_password(password)

# Simulate password verification
is_valid = verify_password(password, hashed)

print("Is password valid:", is_valid)

"""

**Don't Do This:** Store passwords in plain text. Use weak hashing algorithms. Use the same salt for all passwords.

### 5.3. Keep Dependencies Up-to-Date

**Do This:** Regularly update project dependencies to patch security vulnerabilities.

**Why:** Software vulnerabilities are constantly being discovered. Keeping dependencies up-to-date ensures that you have the latest security fixes.

**Example (using Poetry):**

"""bash
poetry update
"""

## 6. Documentation and Code Comments

Well-documented code is essential for maintainability, collaboration, and knowledge sharing.

### 6.1. Docstrings

**Do This:** Write comprehensive docstrings for all modules, classes, functions, and methods.  Use reStructuredText or Google style docstrings.

**Why:** Docstrings provide a concise description of the purpose, arguments, and return values of code elements, making it easier for others (and your future self) to understand and use your code.

**Example (Google Style):**

"""python
def calculate_area(length, width):
    """Calculate the area of a rectangle.

    Args:
        length (int): The length of the rectangle.
        width (int): The width of the rectangle.

    Returns:
        int: The area of the rectangle.

    Raises:
        TypeError: If either length or width is not an integer.
        ValueError: If either length or width is negative.
    """
    if not isinstance(length, int) or not isinstance(width, int):
        raise TypeError("Length and width must be integers.")
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative.")
    return length * width
"""

**Don't Do This:** Write superficial or outdated docstrings. Omit docstrings altogether.

### 6.2. Code Comments

**Do This:** Add comments to explain complex logic, non-obvious code sections, and design decisions. Avoid over-commenting obvious code.

**Why:**  Comments provide additional context and explanations that are not readily apparent from the code itself.

**Example:**

"""python
def process_data(data):
    # Sort the data by timestamp in descending order.  This is crucial for
    # ensuring the most recent entries are processed first.
    sorted_data = sorted(data, key=lambda x: x["timestamp"], reverse=True)
    return sorted_data
"""

**Don't Do This:** Write redundant or misleading comments. Use comments to compensate for poorly written code; refactor instead.

This document provides a foundation for building robust and maintainable Python applications using current best practices. Adherence to these standards will improve code quality, promote consistency, and facilitate collaboration within development teams. Remember to consult the official Python documentation and other resources for more in-depth information.
