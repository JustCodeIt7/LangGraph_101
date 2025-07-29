# LangGraph Document Chat CLI

A powerful CLI application that allows you to chat with your documents (PDF and Markdown) using LangGraph, Ollama Llama3.2, and ChromaDB.

## Features

- 🔍 **Document Processing**: Support for PDF and Markdown files
- 💬 **Interactive Chat**: Natural language conversations about your documents
- 🧠 **Advanced AI**: Powered by Ollama Llama3.2 model
- 📚 **Vector Storage**: ChromaDB for efficient document retrieval
- 🌐 **LangGraph**: Sophisticated conversation flow management
- 📁 **Flexible Input**: Process single files or entire directories
- 🎨 **Rich Interface**: Beautiful CLI with syntax highlighting

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **Llama3.2 model** pulled in Ollama

### Install Ollama and Llama3.2

```bash
# Install Ollama (macOS)
brew install ollama

# Or install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull Llama3.2 model
ollama pull llama3.2
```

## Installation

1. Clone or create the project:
```bash
mkdir langchat
cd langchat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make the script executable:
```bash
chmod +x main.py
```

## Usage

### Basic Chat with Documents

Chat with a single document:
```bash
python main.py chat document.pdf
```

Chat with all documents in a directory:
```bash
python main.py chat ./documents/
```

### Advanced Options

Specify custom settings:
```bash
python main.py chat ./docs/ --collection my-docs --model llama3.2 --chunk-size 1500
```

### Index Documents (Without Chat)

Pre-index documents for faster chat sessions:
```bash
python main.py index ./documents/ --collection my-collection
```

### List Collections

View all available document collections:
```bash
python main.py list-collections
```

## Chat Commands

Once in a chat session, you can use these commands:

- `help` - Show available commands
- `clear` - Clear the screen
- `quit` or `exit` - End the chat session

## Example Usage

```bash
$ python main.py chat ./research-papers/

┌─────────────────────────────────────┐
│         Starting Chat Session       │
├─────────────────────────────────────┤
│ Path: ./research-papers/            │
│ Model: llama3.2                     │
│ Collection: default                 │
└─────────────────────────────────────┘

✓ Processed 45 document chunks

Chat started! Type 'quit' or 'exit' to end the session.
Type 'help' for available commands.

You: What are the main findings in the machine learning papers?

┌─────────────── Assistant ───────────────┐
│ Based on the research papers in your    │
│ collection, the main findings include:  │
│                                         │
│ 1. **Deep Learning Advances**: Several  │
│    papers discuss improvements in       │
│    neural network architectures...      │
│                                         │
│ 2. **Transfer Learning**: Multiple      │
│    studies show that pre-trained       │
│    models can be effectively...         │
└─────────────────────────────────────────┘

You: Can you summarize the methodology from the transformer paper?
```

## Supported File Types

- **PDF**: `.pdf`
- **Markdown**: `.md`, `.markdown`
- **Text**: `.txt`

## Configuration

The application stores its data in `~/.langchat/`:
- `chromadb/` - Vector database storage
- Configuration files and logs

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │───▶│ Document Processor│───▶│   Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                                              │
          ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chat Agent    │◀───│   LangGraph      │◀───│   ChromaDB      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │
          ▼
┌─────────────────┐
│ Ollama Llama3.2 │
└─────────────────┘
```

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Model Not Found
```bash
# Pull the required model
ollama pull llama3.2

# Check available models
ollama list
```

### ChromaDB Issues
```bash
# Clear ChromaDB data if corrupted
rm -rf ~/.langchat/chromadb/
```

## Development

### Project Structure
```
langchat/
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── src/
│   ├── __init__.py
│   ├── chat_agent.py      # LangGraph chat agent
│   ├── document_processor.py # Document processing
│   ├── vector_store.py    # ChromaDB management
│   └── config.py          # Configuration
└── README.md
```

### Adding New File Types

To add support for new file types, modify `document_processor.py`:

1. Add the extension to `supported_extensions`
2. Create a new processing method
3. Add the method to `_process_file()`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
