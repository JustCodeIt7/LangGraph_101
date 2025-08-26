# Crush Configuration for LangGraph Human-in-the-Loop Project

## Commands
- Run main application: `python 16-hil_advanced.py`
- Run specific Python file: `python <filename>.py`
- No specific lint/test commands found in codebase

## Code Style Guidelines
- Use TypedDict for state definitions
- Follow function naming: `snake_case`
- Use descriptive docstrings for all functions
- Structure code in sections with commented headers
- Use rich library for enhanced print statements
- Error handling: Use interrupt for human-in-the-loop flows
- Imports: Standard library first, then third-party, then local
- Use of ellipsis (...) for code organization

## Project Structure Notes
- Main implementation in 16-hil_advanced.py
- Uses LangGraph for workflow orchestration
- Ollama integration for LLM calls
- Human-in-the-loop intervention via interrupt/resume