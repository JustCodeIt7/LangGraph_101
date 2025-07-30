#!/usr/bin/env python3
"""
Utility functions for the simple Python project.
"""

def add(a, b):
    """Add two numbers together."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def multiply(a, b):
    """Multiply two numbers together."""
    return a * b


def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b