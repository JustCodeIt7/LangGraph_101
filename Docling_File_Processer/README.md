# Docling File Processor

This module provides document processing capabilities using Docling, an AI-powered document understanding tool.

## Overview

Docling is a powerful document processing library that converts various document formats (PDF, images, etc.) into structured data. This module integrates Docling with LangGraph to create intelligent document processing workflows.

## Files

- `main.py` - Main entry point for document processing
- `main-cp.py` - Checkpoint-based processing implementation  
- `main-gem.py` - Gemini-enhanced document processing
- `main-cc.py` - Claude Code enhanced version
- `test.ipynb` - Jupyter notebook with examples and tests

## Features

- **Multi-format Support**: Process PDF, images, and other document types
- **AI Enhancement**: Uses LLMs (Gemini, Claude) for improved understanding
- **Structured Output**: Converts documents to structured JSON/Markdown
- **Graph-based Workflows**: LangGraph integration for complex processing pipelines

## Usage

### Basic Processing

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
print(result.to_markdown())
```

### With LLM Enhancement

For enhanced understanding, use the LLM-enhanced versions:

```bash
# Using Gemini enhancement
python main-gem.py input.pdf

# Using Claude Code enhancement  
python main-cp.py input.pdf
```

## Architecture

The processing pipeline follows this flow:

1. **Input**: Load document (PDF/image)
2. **Parse**: Extract text and structure using Docling
3. **Enhance** (optional): Use LLM to improve understanding
4. **Output**: Return structured data (JSON/Markdown)

### Graph Nodes

- `load_document` - Loads and validates input files
- `extract_content` - Parses document structure
- `enhance_with_llm` - Applies AI enhancement when needed
- `format_output` - Converts to desired output format

## Dependencies

Required packages:
```bash
pip install docling langchain-google-genai anthropic
```

Environment variables:
- `GOOGLE_API_KEY` - For Gemini enhancement
- `ANTHROPIC_API_KEY` - For Claude enhancement

## Testing

Run the test notebook:
```bash
jupyter notebook test.ipynb
```

This contains examples of processing various document types and demonstrates the LLM-enhanced features.