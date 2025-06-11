# LangGraph Tools Tutorial

This tutorial series covers comprehensive usage of tools in LangGraph, from basic concepts to advanced integration patterns.

## Files Overview

### 01_basic_tools.py
- Simple function tools
- Using the @tool decorator
- Custom Pydantic schemas
- Data processing tools

### 02_advanced_tool_features.py
- Hiding arguments from the model using state and config
- Disabling parallel tool calling
- Return direct functionality
- Force tool use with safety measures

### 03_error_handling.py
- Default error handling (enabled)
- Disabled error handling
- Custom error messages and handlers
- Error recovery strategies

### 04_prebuilt_tools_and_integration.py
- Simulated prebuilt tools
- Database integration patterns
- File system tools
- API integration examples
- Comprehensive agent composition

## Setup Requirements

```bash
pip install langgraph langchain langchain-openai langchain-community