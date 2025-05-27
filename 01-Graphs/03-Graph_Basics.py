#%%
# =============================================================================
# PART 1: RECIPE PLANNER GRAPH
# =============================================================================

print("=" * 60)
print("PART 1: RECIPE PLANNER GRAPH")
print("=" * 60)

# Step 1: Import required modules
from typing import TypedDict, List
from langgraph.graph import StateGraph
from IPython.display import Image, display
#%%
# Step 2: Define the Agent State Schema
class RecipeState(TypedDict):
    ingredient: str

print("✅ RecipeState defined with 'ingredient' field")

# Step 3: Define the recipe planning node
def recipe_planner_node(state: RecipeState) -> RecipeState:
    """
    Simple node that suggests a recipe based on a main ingredient.
    Takes an ingredient and returns a recipe suggestion.
    """
    print(f"📥 Input to recipe_planner_node: {state}")
    
    ingredient = state["ingredient"]
    
    # Create recipe suggestions based on ingredient
    recipes = {
        "chicken": "Herb-Roasted Chicken with garlic and rosemary",
        "pasta": "Creamy Carbonara with parmesan and bacon",
        "tomatoes": "Fresh Caprese Salad with mozzarella and basil",
        "rice": "Spanish Paella with saffron and seafood",
        "potatoes": "Crispy Garlic Roasted Potatoes with herbs"
    }
    
    recipe = recipes.get(ingredient.lower(), f"Creative {ingredient} stir-fry with vegetables")
    state["ingredient"] = f"Recipe suggestion: {recipe}"
    
    print(f"📤 Output from recipe_planner_node: {state}")
    return state

# Step 4: Build and compile the graph
def build_recipe_graph():
    """Build and compile the Recipe Planner graph"""
    print("\n🔧 Building Recipe Planner Graph...")
    
    # Create the graph with our state schema
    graph = StateGraph(state_schema=RecipeState)
    
    # Add our recipe planning node
    graph.add_node("planner", recipe_planner_node)
    
    # Set entry and finish points
    graph.set_entry_point("planner")
    graph.set_finish_point("planner")
    
    # Compile the graph
    app = graph.compile()
    display(Image(app.get_graph().draw_mermaid_png()))
    print("✅ Graph compiled successfully!")
    
    return app

# Step 5: Test the Recipe Planner graph
def test_recipe_planner():
    print("\n🚀 Testing Recipe Planner Graph...")
    
    app = build_recipe_graph()
    display(Image(app.get_graph().draw_mermaid_png()))
    # Test with different ingredients
    test_ingredients = ["chicken", "pasta", "broccoli", "salmon"]
    
    for ingredient in test_ingredients:
        print(f"\n--- Planning recipe for: {ingredient} ---")
        result = app.invoke({"ingredient": ingredient})
        print(f"🍽️ {result['ingredient']}")
#%%
# Run Recipe Planner example
test_recipe_planner()

print("\n" + "=" * 60)
print("EXERCISE 1: WORKOUT RECOMMENDATION AGENT")
print("=" * 60)
#%%
# Exercise 1 Solution
def workout_recommender_node(state: RecipeState) -> RecipeState:
    """
    Node that creates workout recommendations based on activity type.
    """
    print(f"📥 Input to workout_recommender_node: {state}")
    
    # Get the activity type from the current ingredient field (reusing structure)
    activity = state["ingredient"]
    
    # Create workout recommendations
    workouts = {
        "cardio": "30-minute HIIT workout with burpees, jumping jacks, and mountain climbers",
        "strength": "Full-body strength training with squats, deadlifts, and bench press",
        "yoga": "Flow yoga session focusing on flexibility and mindfulness",
        "running": "5K interval training with 1-minute sprints and 2-minute recovery",
        "swimming": "Lap swimming workout: 10x100m freestyle with 30-second rest"
    }
    
    workout = workouts.get(activity.lower(), f"Beginner-friendly {activity} workout routine")
    state["ingredient"] = f"Workout plan: {workout}"
    
    print(f"📤 Output from workout_recommender_node: {state}")
    return state

def build_workout_graph():
    """Build the workout recommendation graph for Exercise 1"""
    print("\n🔧 Building Workout Recommendation Graph...")
    
    graph = StateGraph(state_schema=RecipeState)
    graph.add_node("trainer", workout_recommender_node)
    graph.set_entry_point("trainer")
    graph.set_finish_point("trainer")
    
    app = graph.compile()
    print("✅ Workout Graph compiled!")
    
    return app

def test_workout_agent():
    print("\n🚀 Testing Workout Recommendation Agent...")
    
    app = build_workout_graph()
    display(Image(app.get_graph().draw_mermaid_png()))
    
    test_activities = ["cardio", "strength", "yoga", "dancing"]
    
    for activity in test_activities:
        print(f"\n--- Workout for: {activity} ---")
        result = app.invoke({"ingredient": activity})
        print(f"💪 {result['ingredient']}")
#%%
# Run Exercise 1
test_workout_agent()
#%%
# =============================================================================
# PART 2: DATA ANALYZER GRAPH
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: DATA ANALYZER GRAPH")
print("=" * 60)

# Step 1: Define extended Agent State Schema
class DataAnalyzerState(TypedDict):
    numbers: List[int]
    dataset_name: str
    analysis_result: str

print("✅ DataAnalyzerState defined with 'numbers', 'dataset_name', and 'analysis_result' fields")

# Step 2: Define the data analysis node
def analyze_data_node(state: DataAnalyzerState) -> DataAnalyzerState:
    """
    Analyzes a dataset and provides statistical insights.
    Takes a list of numbers and generates comprehensive analysis.
    """
    print(f"📥 Input to analyze_data_node: {state}")
    
    numbers = state["numbers"]
    dataset_name = state["dataset_name"]
    
    # Perform statistical analysis
    total_count = len(numbers)
    average = sum(numbers) / total_count
    maximum = max(numbers)
    minimum = min(numbers)
    range_val = maximum - minimum
    
    # Create comprehensive analysis report
    state["analysis_result"] = (
        f"Analysis of {dataset_name} dataset:\n"
        f"• Count: {total_count} data points\n"
        f"• Average: {average:.2f}\n"
        f"• Range: {minimum} to {maximum} (span: {range_val})\n"
        f"• Trend: {'Increasing' if numbers[-1] > numbers[0] else 'Decreasing'}"
    )
    
    print(f"📤 Output from analyze_data_node: {state}")
    return state

# Step 3: Build and compile the data analyzer graph
def build_data_analyzer_graph():
    """Build and compile the Data Analyzer graph"""
    print("\n🔧 Building Data Analyzer Graph...")
    
    graph = StateGraph(state_schema=DataAnalyzerState)
    graph.add_node("analyzer", analyze_data_node)
    graph.set_entry_point("analyzer")
    graph.set_finish_point("analyzer")
    
    app = graph.compile()
    print("✅ Data Analyzer Graph compiled!")
    
    return app

# Step 4: Test the data analyzer graph
def test_data_analyzer():
    print("\n🚀 Testing Data Analyzer Graph...")
    
    app = build_data_analyzer_graph()
    
    # Test cases with different datasets
    test_cases = [
        {"numbers": [85, 92, 78, 96, 88], "dataset_name": "Student Test Scores", "analysis_result": ""},
        {"numbers": [120, 135, 110, 145, 130], "dataset_name": "Daily Sales ($)", "analysis_result": ""},
        {"numbers": [22, 25, 19, 28, 24, 21], "dataset_name": "Temperature (°C)", "analysis_result": ""},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Dataset {i}: {test_case['dataset_name']} ---")
        print(f"Raw data: {test_case['numbers']}")
        result = app.invoke(test_case)
        print(f"📊 {result['analysis_result']}")
#%%
# Run Data Analyzer example
test_data_analyzer()
#%%
print("\n" + "=" * 60)
print("EXERCISE 2: WEATHER PATTERN ANALYZER")
print("=" * 60)

# Exercise 2 Solution
class WeatherState(TypedDict):
    temperatures: List[int]
    location: str
    analysis_type: str  # "weekly" or "monthly"
    weather_report: str

def weather_analyzer_node(state: WeatherState) -> WeatherState:
    """
    Analyzes weather patterns and provides different insights based on analysis type.
    """
    print(f"📥 Input to weather_analyzer_node: {state}")
    
    temperatures = state["temperatures"]
    location = state["location"]
    analysis_type = state["analysis_type"]
    
    avg_temp = sum(temperatures) / len(temperatures)
    max_temp = max(temperatures)
    min_temp = min(temperatures)
    
    if analysis_type == "weekly":
        # Weekly analysis focuses on daily variations
        daily_changes = [abs(temperatures[i] - temperatures[i-1]) for i in range(1, len(temperatures))]
        avg_change = sum(daily_changes) / len(daily_changes) if daily_changes else 0
        
        state["weather_report"] = (
            f"Weekly Weather Report for {location}:\n"
            f"🌡️ Average temperature: {avg_temp:.1f}°C\n"
            f"🔥 Highest: {max_temp}°C | 🧊 Lowest: {min_temp}°C\n"
            f"📈 Daily variation: {avg_change:.1f}°C average change\n"
            f"☀️ Weather stability: {'Stable' if avg_change < 3 else 'Variable'}"
        )
    elif analysis_type == "monthly":
        # Monthly analysis focuses on overall trends
        temp_trend = "warming" if temperatures[-1] > temperatures[0] else "cooling"
        variance = sum((t - avg_temp) ** 2 for t in temperatures) / len(temperatures)
        
        state["weather_report"] = (
            f"Monthly Weather Summary for {location}:\n"
            f"🌡️ Monthly average: {avg_temp:.1f}°C\n"
            f"📊 Temperature range: {min_temp}°C to {max_temp}°C\n"
            f"📈 Overall trend: {temp_trend}\n"
            f"🔄 Consistency: {'Very consistent' if variance < 10 else 'Highly variable'}"
        )
    
    print(f"📤 Output from weather_analyzer_node: {state}")
    return state

def build_weather_graph():
    """Build the weather analyzer graph for Exercise 2"""
    print("\n🔧 Building Weather Analyzer Graph...")
    
    graph = StateGraph(state_schema=WeatherState)
    graph.add_node("weather_analyst", weather_analyzer_node)
    graph.set_entry_point("weather_analyst")
    graph.set_finish_point("weather_analyst")
    
    app = graph.compile()
    
    print("✅ Weather Analyzer Graph compiled!")
    
    return app

def test_weather_agent():
    print("\n🚀 Testing Weather Pattern Analyzer...")
    
    app = build_weather_graph()
    display(Image(app.get_graph().draw_mermaid_png()))
    # Test cases for both weekly and monthly analysis
    test_cases = [
        {
            "temperatures": [22, 24, 21, 25, 23, 20, 26], 
            "location": "San Francisco", 
            "analysis_type": "weekly", 
            "weather_report": ""
        },
        {
            "temperatures": [15, 18, 20, 22, 25, 23, 21, 19], 
            "location": "London", 
            "analysis_type": "monthly", 
            "weather_report": ""
        },
        {
            "temperatures": [30, 32, 35, 33, 31, 34, 36], 
            "location": "Phoenix", 
            "analysis_type": "weekly", 
            "weather_report": ""
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Weather Analysis {i} ---")
        print(f"Location: {test_case['location']} | Type: {test_case['analysis_type']}")
        print(f"Temperatures: {test_case['temperatures']}")
        result = app.invoke(test_case)
        print(f"🌤️ {result['weather_report']}")
#%%
# Run Exercise 2
test_weather_agent()
#%%