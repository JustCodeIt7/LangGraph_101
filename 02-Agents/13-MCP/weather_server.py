from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Weather')


@mcp.tool()
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    # In a real app, this could call an API; for demo, return a static response
    return f'The weather in {location} is sunny and 75°F.'


if __name__ == '__main__':
    mcp.run(transport='sse')  # Run on HTTP for multi-server ease.