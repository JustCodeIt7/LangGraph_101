# RAG Agent with LangGraph

A powerful CLI application that allows you to chat with your documents using Retrieval-Augmented Generation (RAG). Built with LangGraph for sophisticated agent orchestration, Ollama Llama3.2 for language modeling, and ChromaDB for vector storage.

## Features

- 📚 **Document Support**: Load and chat with PDF and Markdown files
- 🗂️ **Directory Processing**: Process entire directories of documents
- 🧠 **Smart Retrieval**: Uses vector similarity search to find relevant content
- 💬 **Interactive Chat**: Maintains conversation context and history
- 🔄 **LangGraph Orchestration**: Sophisticated agent workflow management
- 🚀 **Local LLM**: Uses Ollama Llama3.2 for privacy and performance
- 💾 **Persistent Storage**: ChromaDB for efficient vector storage and retrieval

## Architecture

The RAG Agent uses LangGraph to orchestrate a multi-step workflow:

1. **Document Loading**: Processes PDF and Markdown files into chunks
2. **Vector Storage**: Creates embeddings and stores in ChromaDB
3. **Query Processing**: Retrieves relevant document chunks
4. **Response Generation**: Uses Ollama Llama3.2 to generate contextual responses

## Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai
```

### 2. Pull Llama3.2 Model

```bash
ollama pull llama3.2
```

### 3. Start Ollama Service

```bash
ollama serve
```

## Installation

1. **Clone and navigate to the directory**:
```bash
cd 02-Agents-v2/10-RAG_Agent
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Load documents and start interactive chat
python rag_agent.py --load /path/to/documents

# Load a single PDF file
python rag_agent.py --load document.pdf

# Load a directory of documents
python rag_agent.py --load /path/to/document/folder
```

### Advanced Usage

```bash
# Use a different Ollama model
python rag_agent.py --model llama2 --load documents/

# Specify custom database path
python rag_agent.py --db-path ./custom_db --load documents/

# Single query mode (non-interactive)
python rag_agent.py --query "What is the main topic of these documents?"
```

### Interactive Commands

Once in interactive mode, you can use these commands:

- **Chat**: Simply type your question
- **Load documents**: `load /path/to/documents`
- **Clear history**: `clear`
- **Exit**: `quit`

## Example Session

```bash
$ python rag_agent.py --load ./research_papers/

🤖 RAG Agent - Chat with your documents
==================================================
✅ Initialized RAG Agent with model: llama3.2

📚 Loading documents from: ./research_papers/
✅ Successfully loaded documents! (127 chunks in database)

💬 Interactive Chat Mode
Commands:
  - Type your question to chat
  - 'load <path>' to load documents
  - 'clear' to clear chat history
  - 'quit' to exit
--------------------------------------------------

👤 You: What are the main findings in the research papers?

🤖 Thinking...
🤖 Assistant: Based on the research papers you've loaded, I can identify several key findings:

1. **Machine Learning Performance**: The studies show significant improvements in model accuracy when using ensemble methods, with average performance gains of 15-20%.

2. **Data Quality Impact**: Research consistently demonstrates that data preprocessing and cleaning can improve model performance by up to 30%.

3. **Computational Efficiency**: New optimization techniques reduced training time by 40% while maintaining comparable accuracy levels.

The papers particularly emphasized the importance of feature engineering and cross-validation in achieving reliable results.

👤 You: Can you elaborate on the ensemble methods mentioned?

🤖 Assistant: Certainly! The ensemble methods discussed in the research papers include several key approaches:

[Detailed response based on document content...]
```

## Configuration

### Model Configuration

You can use different Ollama models by specifying the `--model` parameter:

```bash
# Use Llama2
python rag_agent.py --model llama2

# Use CodeLlama for code-related documents
python rag_agent.py --model codellama

# Use Mistral
python rag_agent.py --model mistral
```

### Database Configuration

The agent uses ChromaDB for vector storage with the following default settings:

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Similarity Search**: Top 4 most relevant chunks
- **Persistence**: Local directory (`./chroma_db`)

## File Support

### Supported Formats

- **PDF Files**: `.pdf`
- **Markdown Files**: `.md`, `.markdown`

### Directory Processing

The agent recursively processes directories, automatically detecting and loading all supported file types.

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   ❌ Failed to initialize agent: Could not connect to Ollama
   ```
   **Solution**: Ensure Ollama is running (`ollama serve`)

2. **Model Not Found**
   ```
   ❌ Model 'llama3.2' not found
   ```
   **Solution**: Pull the model (`ollama pull llama3.2`)

3. **No Documents Loaded**
   ```
   ❌ No documents loaded. Use --load to load documents first.
   ```
   **Solution**: Use `--load` parameter or `load <path>` command

4. **Permission Errors**
   ```
   ❌ Permission denied accessing file
   ```
   **Solution**: Check file permissions and ensure read access

### Debug Mode

For detailed logging, set the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Document Size**: Large documents are automatically chunked for optimal processing
2. **Memory Usage**: ChromaDB efficiently manages vector storage
3. **Query Optimization**: More specific questions yield better results
4. **Model Selection**: Choose appropriate models based on your use case

## Advanced Features

### Custom Prompts

The agent uses sophisticated prompts that include:
- Document context
- Chat history
- Clear instructions for handling missing information

### Error Handling

Robust error handling with graceful degradation:
- Network connectivity issues
- Document parsing errors
- Model unavailability
- Invalid file formats

### State Management

LangGraph manages complex state transitions:
- Document loading and processing
- Vector store initialization
- Query processing and retrieval
- Response generation and formatting

## Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is part of the LangGraph tutorial series.

---

**Happy chatting with your documents! 🚀📚**