# Quick Start Guide 🚀

Get the Finance & Investment Agent up and running in minutes!

## ⚡ Fast Setup (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Ollama & Model
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull the model (in a new terminal)
ollama pull llama3.2
```

### 3. Test the Agent
```bash
python test_agent.py
```

## 🎮 Try It Out

### Interactive Mode
```bash
python finance_investment_agent.py
```

### Demo Mode
```bash
python demo_finance_agent.py
```

## 💬 Example Conversations

```
You: I spent $45 on groceries today
Agent: ✅ Expense tracked: $45.00 for food - groceries today

You: What's the price of Apple stock?
Agent: 📊 Market Data for AAPL:
       📈 Price: AAPL: $175.25 (+2.15, +1.24%)

You: Create a budget with $500 for food and $300 for transport
Agent: ✅ Budget created:
       • Budget item created: $500.00 allocated for food
       • Budget item created: $300.00 allocated for transport
```

## 🆘 Troubleshooting

**Ollama Connection Issues:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

**Model Not Found:**
```bash
# Check available models
ollama list

# Pull llama3.2 if missing
ollama pull llama3.2
```

**Import Errors:**
```bash
# Upgrade pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt
```

## 🛠️ Automated Setup

For a fully automated setup experience:
```bash
python setup.py
```

This script will:
- ✅ Check Python version
- ✅ Install dependencies
- ✅ Verify Ollama installation
- ✅ Start Ollama service
- ✅ Pull llama3.2 model
- ✅ Run tests
- ✅ Create sample data

## 📚 Next Steps

Once everything is working:
1. **Read the full [README.md](README.md)** for detailed documentation
2. **Explore [demo_finance_agent.py](demo_finance_agent.py)** to see all features
3. **Customize the agent** by modifying tools and prompts
4. **Add real API integrations** for production use

## 🏆 Features You Can Try

- 💰 **Expense Tracking**: "I bought coffee for $5"
- 📊 **Budget Planning**: "Set $200 budget for entertainment"
- 📈 **Market Data**: "What's Tesla's stock price?"
- 🏦 **Portfolio Analysis**: "Analyze my portfolio"
- ❓ **Financial Advice**: "How should I start investing?"

Happy finance management! 🏦✨