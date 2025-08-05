"""
Math MCP Server - provides basic arithmetic operations
Run this as: python math_server.py
"""

from mcp.server.fastmcp import FastMCP

############################ Server Initialization ############################
# Create MCP server instance
mcp = FastMCP("Math Server")

############################## Tool Definitions ##############################
@mcp.tool()  # Register the function as a callable tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract second number from first number."""

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide first number by second number."""
    # Prevent division by zero


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""

@mcp.tool()
def square_root(number: float) -> float:
    """Calculate square root of a number."""
    # Prevent taking the square root of a negative number


############################## Server Execution ##############################
if __name__ == "__main__":
    print("🧮 Starting Math MCP Server...")
    mcp.run(transport='stdio')  # Use standard I/O for communication