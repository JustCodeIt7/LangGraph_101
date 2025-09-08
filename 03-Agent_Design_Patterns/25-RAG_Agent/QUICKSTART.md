# Quick Start Guide - RAG Agent

Get up and running with the RAG Agent in 5 minutes!

## 🚀 Quick Setup

### 1. Prerequisites Check
```bash
# Test your setup
python test_setup.py
```

### 2. Install Ollama (if not already installed)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

### 3. Pull the Model
```bash
ollama pull llama3.2
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Quick Usage Examples

### Example 1: Interactive Chat with Documents
```bash
# Load a PDF and start chatting
python rag_agent.py --load document.pdf

# Load a directory of documents
python rag_agent.py --load /path/to/documents/
```

### Example 2: Single Query Mode
```bash
# Ask a single question
python rag_agent.py --load document.pdf --query "What is this document about?"
```

### Example 3: Programmatic Usage
```bash
# Run the example script
python example_usage.py
```

## 💡 Common Use Cases

### Research Papers
```bash
python rag_agent.py --load ./research_papers/
```

### Documentation
```bash
python rag_agent.py --load ./project_docs/
```

### Mixed Document Types
```bash
# Load directory with PDFs and markdown files
python rag_agent.py --load ./mixed_documents/
```

## 🛠️ Troubleshooting

### Issue: "Could not connect to Ollama"
**Solution:**
```bash
# Start Ollama service
ollama serve

# In another terminal, test
curl http://localhost:11434/api/tags
```

### Issue: "Model not found"
**Solution:**
```bash
# Pull the model
ollama pull llama3.2

# List available models
ollama list
```

### Issue: "No documents loaded"
**Solution:**
- Use `--load` parameter when starting
- Or use `load <path>` command in interactive mode
- Check file permissions and supported formats (PDF, MD)

## 📚 Interactive Commands

Once in chat mode:
- **Ask questions**: Just type your question
- **Load more docs**: `load /path/to/more/documents`
- **Clear history**: `clear`
- **Exit**: `quit`

## 🎉 You're Ready!

Your RAG Agent is now ready to chat with your documents! 

For more details, see the full [README.md](README.md).