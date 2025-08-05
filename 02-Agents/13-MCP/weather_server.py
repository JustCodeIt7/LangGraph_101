"""
Weather MCP Server - provides weather information
Run this as: python weather_server.py
"""

import json
import random
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Weather Server")

# Mock weather data
WEATHER_DATA = {
    "new york": {"temp": 72, "condition": "Partly cloudy", "humidity": 65},
    "san francisco": {"temp": 68, "condition": "Foggy", "humidity": 85},
    "los angeles": {"temp": 78, "condition": "Sunny", "humidity": 45},
    "chicago": {"temp": 65, "condition": "Windy", "humidity": 70},
    "miami": {"temp": 84, "condition": "Hot and humid", "humidity": 90},
    "seattle": {"temp": 62, "condition": "Rainy", "humidity": 80},
    "boston": {"temp": 69, "condition": "Clear", "humidity": 55},
    "denver": {"temp": 71, "condition": "Sunny", "humidity": 30},
}

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get current weather for a specified location."""
    location_key = location.lower().strip()
    
    if location_key in WEATHER_DATA:
        weather = WEATHER_DATA[location_key]
        return f"Weather in {location.title()}: {weather['temp']}°F, {weather['condition']}, Humidity: {weather['humidity']}%"
    else:
        # Return random weather for unknown locations
        temp = random.randint(55, 85)
        conditions = ["Sunny", "Cloudy", "Partly cloudy", "Rainy", "Clear"]
        condition = random.choice(conditions)
        humidity = random.randint(30, 90)
        return f"Weather in {location.title()}: {temp}°F, {condition}, Humidity: {humidity}%"

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for specified location and number of days."""
    if days > 7:
        days = 7
    
    location_key = location.lower().strip()
    base_weather = WEATHER_DATA.get(location_key, {
        "temp": random.randint(60, 80),
        "condition": "Variable",
        "humidity": random.randint(40, 80)
    })
    
    forecast = [f"Weather forecast for {location.title()}:"]
    
    for day in range(1, days + 1):
        temp_variation = random.randint(-5, 5)
        temp = base_weather["temp"] + temp_variation
        conditions = ["Sunny", "Cloudy", "Partly cloudy", "Rainy", "Clear", "Overcast"]
        condition = random.choice(conditions)
        humidity = base_weather["humidity"] + random.randint(-10, 10)
        
        forecast.append(f"Day {day}: {temp}°F, {condition}, Humidity: {max(20, min(100, humidity))}%")
    
    return "\n".join(forecast)

@mcp.tool()
async def compare_weather(location1: str, location2: str) -> str:
    """Compare weather between two locations."""
    weather1 = await get_weather(location1)
    weather2 = await get_weather(location2)
    
    return f"Weather comparison:\n{weather1}\n{weather2}"

if __name__ == "__main__":
    print("🌤️ Starting Weather MCP Server on port 8000...")
    mcp.run(transport="streamable-http", port=8000)