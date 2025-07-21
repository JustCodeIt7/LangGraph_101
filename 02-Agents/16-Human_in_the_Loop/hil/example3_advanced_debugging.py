import asyncio
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, List, Tuple
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pytest
from unittest.mock import patch, MagicMock
import traceback
import textwrap

class CodeDebugState:
    def __init__(self):
        self.problem_description: str = ""
        self.generated_code: str = ""
        self.test_code: str = ""
        self.test_results: str = ""
        self.test_passed: bool = False
        self.debug_attempts: int = 0
        self.max_debug_attempts: int = 3
        self.human_corrections: List[str] = []
        self.final_working_code: str = ""
        self.error_messages: List[str] = []

def create_code_debug_graph():
    """Create an advanced human-in-the-loop graph for code generation and debugging."""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    def generate_initial_code(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial code based on the problem description."""
        try:
            problem = state.get("problem_description", "")
            previous_corrections = state.get("human_corrections", [])
            
            if not problem.strip():
                raise ValueError("Problem description is empty")
            
            # Include previous corrections in the prompt
            correction_context = ""
            if previous_corrections:
                correction_context = f"\n\nPrevious corrections to consider:\n" + "\n".join(f"- {correction}" for correction in previous_corrections[-3:])  # Last 3 corrections
            
            prompt = f"""Write a Python function to solve the following problem:

{problem}

Requirements:
1. Write clean, well-documented code
2. Include proper error handling
3. Use descriptive variable names
4. Add docstrings to explain the function
5. Make the code efficient and readable

{correction_context}

Please provide only the Python code without any explanatory text."""

            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            
            # Clean up the response to extract only the code
            code = response.content.strip()
            if code.startswith("```python"):
                code = code[9:]  # Remove ```python
            if code.endswith("```"):
                code = code[:-3]  # Remove ```
            
            state["generated_code"] = code.strip()
            
            print("🔧 Generated Code:")
            print("-" * 50)
            print(state["generated_code"])
            print("-" * 50)
            
            return state
            
        except Exception as e:
            print(f"Error generating code: {e}")
            state["generated_code"] = f"# Error generating code: {e}"
            state["error_messages"] = state.get("error_messages", []) + [str(e)]
            return state
    
    def generate_test_code(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test cases for the generated code."""
        try:
            problem = state.get("problem_description", "")
            code = state.get("generated_code", "")
            
            prompt = f"""Given this problem and solution, create comprehensive unit tests using pytest:

Problem: {problem}

Code:
{code}

Create test cases that:
1. Test normal cases
2. Test edge cases
3. Test error conditions
4. Validate the function works as expected

Please provide complete test code that can be run with pytest. Include imports and make sure the tests are thorough."""

            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            
            # Clean up the response
            test_code = response.content.strip()
            if test_code.startswith("```python"):
                test_code = test_code[9:]
            if test_code.endswith("```"):
                test_code = test_code[:-3]
            
            state["test_code"] = test_code.strip()
            
            print("🧪 Generated Test Code:")
            print("-" * 50)
            print(state["test_code"])
            print("-" * 50)
            
            return state
            
        except Exception as e:
            print(f"Error generating test code: {e}")
            state["test_code"] = f"# Error generating test code: {e}"
            state["error_messages"] = state.get("error_messages", []) + [str(e)]
            return state
    
    def run_tests(state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the generated tests against the code."""
        try:
            code = state.get("generated_code", "")
            test_code = state.get("test_code", "")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                code_file.write(code)
                code_file_path = code_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
                # Combine code and tests
                combined_content = f"{code}\n\n{test_code}"
                test_file.write(combined_content)
                test_file_path = test_file.name
            
            try:
                # Run pytest on the test file
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', test_file_path, '-v', '--tb=short'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                test_output = result.stdout + result.stderr
                state["test_results"] = test_output
                state["test_passed"] = result.returncode == 0
                
                print("🔍 Test Results:")
                print("-" * 50)
                print(test_output)
                print("-" * 50)
                
                if state["test_passed"]:
                    print("✅ All tests passed!")
                    state["final_working_code"] = code
                else:
                    print("❌ Tests failed. Human debugging needed.")
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(code_file_path)
                    os.unlink(test_file_path)
                except:
                    pass
            
            return state
            
        except subprocess.TimeoutExpired:
            print("⏰ Test execution timed out")
            state["test_results"] = "Test execution timed out"
            state["test_passed"] = False
            return state
        except Exception as e:
            print(f"Error running tests: {e}")
            state["test_results"] = f"Error running tests: {str(e)}\n{traceback.format_exc()}"
            state["test_passed"] = False
            state["error_messages"] = state.get("error_messages", []) + [str(e)]
            return state
    
    def human_debug_assistance(state: Dict[str, Any]) -> Dict[str, Any]:
        """Ask human for debugging assistance when tests fail."""
        try:
            code = state.get("generated_code", "")
            test_results = state.get("test_results", "")
            debug_attempts = state.get("debug_attempts", 0)
            
            print(f"\n--- Human Debug Assistance Required (Attempt {debug_attempts + 1}) ---")
            print("Current Code:")
            print("-" * 30)
            print(code)
            print("-" * 30)
            
            print("\nTest Results:")
            print("-" * 30)
            print(test_results)
            print("-" * 30)
            
            print("\nDebugging Options:")
            print("1. Provide corrected code")
            print("2. Provide debugging hints/corrections")
            print("3. Skip this iteration")
            
            choice = input("\nChoose option (1/2/3): ").strip()
            
            if choice == "1":
                print("\nPlease provide the corrected code:")
                corrected_code = input("Corrected code (or 'multiline' for multi-line input): ").strip()
                
                if corrected_code.lower() == 'multiline':
                    print("Enter corrected code (type 'END' on a new line when finished):")
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == 'END':
                            break
                        lines.append(line)
                    corrected_code = '\n'.join(lines)
                
                if corrected_code:
                    state["generated_code"] = corrected_code
                    state["human_corrections"].append(f"Direct code correction provided")
                    print("✅ Code updated with your corrections!")
                else:
                    print("❌ No correction provided")
                    
            elif choice == "2":
                hint = input("What's wrong with the code? Provide debugging hints: ").strip()
                if hint:
                    state["human_corrections"].append(hint)
                    print("📝 Debugging hint noted. Code will be regenerated.")
                else:
                    print("❌ No hint provided")
                    
            else:  # choice == "3" or default
                print("⏭️ Skipping this iteration")
                state["human_corrections"].append("Skipped debugging iteration")
            
            state["debug_attempts"] = debug_attempts + 1
            
            return state
            
        except KeyboardInterrupt:
            print("\n⚠️ Debug process interrupted by user")
            state["human_corrections"].append("Process interrupted during debugging")
            state["debug_attempts"] = state.get("debug_attempts", 0) + 1
            return state
        except Exception as e:
            print(f"Error in human debugging: {e}")
            state["human_corrections"].append(f"Error during debugging: {e}")
            state["debug_attempts"] = state.get("debug_attempts", 0) + 1
            return state
    
    def check_debug_status(state: Dict[str, Any]) -> str:
        """Check if debugging should continue or stop."""
        test_passed = state.get("test_passed", False)
        debug_attempts = state.get("debug_attempts", 0)
        max_attempts = state.get("max_debug_attempts", 3)
        
        if test_passed:
            return "success"
        elif debug_attempts >= max_attempts:
            return "max_attempts_reached"
        else:
            return "continue_debugging"
    
    def finalize_code_process(state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the code generation and debugging process."""
        test_passed = state.get("test_passed", False)
        debug_attempts = state.get("debug_attempts", 0)
        final_code = state.get("final_working_code", state.get("generated_code", ""))
        corrections = state.get("human_corrections", [])
        
        print(f"\n🎉 Code Generation Process Complete!")
        print(f"Tests passed: {'Yes' if test_passed else 'No'}")
        print(f"Debug attempts: {debug_attempts}")
        print(f"Human corrections: {len(corrections)}")
        
        if test_passed:
            print(f"\n🎯 Final Working Code:")
            print("=" * 60)
            print(final_code)
            print("=" * 60)
        else:
            print(f"\n⚠️ Final code (tests may still fail):")
            print("=" * 60)
            print(final_code)
            print("=" * 60)
        
        if corrections:
            print(f"\n📚 Human Corrections Applied:")
            for i, correction in enumerate(corrections, 1):
                print(f"{i}. {correction}")
        
        return state
    
    # Create the graph
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("generate_code", generate_initial_code)
    workflow.add_node("generate_tests", generate_test_code)
    workflow.add_node("run_tests", run_tests)
    workflow.add_node("human_debug", human_debug_assistance)
    workflow.add_node("finalize", finalize_code_process)
    
    # Add edges
    workflow.set_entry_point("generate_code")
    workflow.add_edge("generate_code", "generate_tests")
    workflow.add_edge("generate_tests", "run_tests")
    workflow.add_conditional_edges(
        "run_tests",
        check_debug_status,
        {
            "success": "finalize",
            "max_attempts_reached": "finalize",
            "continue_debugging": "human_debug"
        }
    )
    workflow.add_edge("human_debug", "generate_code")  # Regenerate code after human input
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# Example usage
def run_advanced_example():
    """Run the advanced code debugging example."""
    print("=== Example 3: Advanced Code Generation with Human Debug Assistance ===\n")
    
    # Sample problem
    problem_description = textwrap.dedent("""
    Create a Python function called 'fibonacci_sequence' that:
    1. Takes an integer n as input
    2. Returns a list containing the first n numbers in the Fibonacci sequence
    3. Handles edge cases: n <= 0 should return an empty list
    4. For n = 1, return [0]
    5. For n = 2, return [0, 1]
    6. For n > 2, return the full sequence
    
    Example: fibonacci_sequence(6) should return [0, 1, 1, 2, 3, 5]
    """).strip()
    
    # Create the graph
    graph = create_code_debug_graph()
    
    # Initial state
    initial_state = {
        "problem_description": problem_description,
        "generated_code": "",
        "test_code": "",
        "test_results": "",
        "test_passed": False,
        "debug_attempts": 0,
        "max_debug_attempts": 3,
        "human_corrections": [],
        "final_working_code": "",
        "error_messages": []
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": "example3"}}
    final_state = graph.invoke(initial_state, config)
    
    return final_state

# Unit Tests
class TestCodeDebugging:
    def test_code_debug_state_structure(self):
        """Test the code debug state structure."""
        state = CodeDebugState()
        assert hasattr(state, 'problem_description')
        assert hasattr(state, 'generated_code')
        assert hasattr(state, 'test_code')
        assert hasattr(state, 'test_results')
        assert hasattr(state, 'test_passed')
        assert hasattr(state, 'debug_attempts')
        assert hasattr(state, 'max_debug_attempts')
        assert hasattr(state, 'human_corrections')
        assert hasattr(state, 'final_working_code')
        assert hasattr(state, 'error_messages')
    
    def test_debug_status_logic(self):
        """Test the debug status checking logic."""
        # Test success scenario
        state1 = {
            "test_passed": True,
            "debug_attempts": 1,
            "max_debug_attempts": 3
        }
        
        test_passed = state1.get("test_passed", False)
        debug_attempts = state1.get("debug_attempts", 0)
        max_attempts = state1.get("max_debug_attempts", 3)
        
        if test_passed:
            result1 = "success"
        elif debug_attempts >= max_attempts:
            result1 = "max_attempts_reached"
        else:
            result1 = "continue_debugging"
        
        assert result1 == "success"
        
        # Test max attempts scenario
        state2 = {
            "test_passed": False,
            "debug_attempts": 3,
            "max_debug_attempts": 3
        }
        
        test_passed = state2.get("test_passed", False)
        debug_attempts = state2.get("debug_attempts", 0)
        max_attempts = state2.get("max_debug_attempts", 3)
        
        if test_passed:
            result2 = "success"
        elif debug_attempts >= max_attempts:
            result2 = "max_attempts_reached"
        else:
            result2 = "continue_debugging"
        
        assert result2 == "max_attempts_reached"
        
        # Test continue debugging scenario
        state3 = {
            "test_passed": False,
            "debug_attempts": 1,
            "max_debug_attempts": 3
        }
        
        test_passed = state3.get("test_passed", False)
        debug_attempts = state3.get("debug_attempts", 0)
        max_attempts = state3.get("max_debug_attempts", 3)
        
        if test_passed:
            result3 = "success"
        elif debug_attempts >= max_attempts:
            result3 = "max_attempts_reached"
        else:
            result3 = "continue_debugging"
        
        assert result3 == "continue_debugging"
    
    def test_code_cleanup(self):
        """Test code cleanup functionality."""
        # Test removing code block markers
        raw_code = "```python\ndef test_function():\n    return 'hello'\n```"
        
        code = raw_code.strip()
        if code.startswith("```python"):
            code = code[9:]  # Remove ```python
        if code.endswith("```"):
            code = code[:-3]  # Remove ```
        
        clean_code = code.strip()
        
        assert clean_code == "def test_function():\n    return 'hello'"
        assert not clean_code.startswith("```")
        assert not clean_code.endswith("```")
    
    def test_fibonacci_function(self):
        """Test a sample Fibonacci function that could be generated."""
        def fibonacci_sequence(n):
            """Generate the first n numbers in the Fibonacci sequence."""
            if n <= 0:
                return []
            elif n == 1:
                return [0]
            elif n == 2:
                return [0, 1]
            else:
                fib = [0, 1]
                for i in range(2, n):
                    fib.append(fib[i-1] + fib[i-2])
                return fib
        
        # Test the function
        assert fibonacci_sequence(0) == []
        assert fibonacci_sequence(1) == [0]
        assert fibonacci_sequence(2) == [0, 1]
        assert fibonacci_sequence(6) == [0, 1, 1, 2, 3, 5]
        assert fibonacci_sequence(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    
    def test_error_handling(self):
        """Test error handling in the process."""
        error_messages = []
        
        try:
            # Simulate an error
            raise ValueError("Test error")
        except Exception as e:
            error_messages.append(str(e))
        
        assert len(error_messages) == 1
        assert "Test error" in error_messages[0]

# Run the tests
if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests for Example 3...")
    pytest.main([__file__ + "::TestCodeDebugging", "-v"])
    
    print("\n" + "="*50)
    print("Unit tests completed. Now running the interactive example...")
    print("="*50 + "\n")
    
    # Run the interactive example
    # Uncomment the line below to run interactively
    # run_advanced_example()