#!/bin/bash

# Example script to run the LangGraph Document Chat CLI

echo "LangGraph Document Chat CLI - Example Usage"
echo "==========================================="

# Check if Ollama is running
echo "Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running. Please start it with: ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# Check if llama3.2 model is available
echo "Checking for llama3.2 model..."
if ! ollama list | grep -q "llama3.2"; then
    echo "❌ llama3.2 model not found. Pulling it now..."
    ollama pull llama3.2
fi

echo "✅ llama3.2 model is available"

# Create sample documents directory
mkdir -p sample_docs

# Create sample markdown file
cat > sample_docs/sample.md << 'EOF'
# Machine Learning Basics

## Introduction
Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

## Types of Machine Learning

### Supervised Learning
- Uses labeled training data
- Examples: Classification, Regression

### Unsupervised Learning
- Finds patterns in unlabeled data
- Examples: Clustering, Dimensionality Reduction

### Reinforcement Learning
- Learns through interaction with environment
- Uses rewards and penalties

## Popular Algorithms
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
EOF

# Create sample text file
cat > sample_docs/notes.txt << 'EOF'
Deep Learning Notes
==================

Deep learning is a subset of machine learning that uses neural networks with multiple layers.

Key concepts:
- Backpropagation
- Gradient descent
- Activation functions
- Convolutional layers
- Recurrent layers

Applications:
- Computer vision
- Natural language processing
- Speech recognition
- Game playing (AlphaGo, etc.)
EOF

echo "📁 Created sample documents in ./sample_docs/"
echo ""
echo "🚀 Starting chat with sample documents..."
echo "   You can ask questions like:"
echo "   - 'What is machine learning?'"
echo "   - 'Explain the types of machine learning'"
echo "   - 'What are some popular algorithms?'"
echo "   - 'Tell me about deep learning'"
echo ""

# Run the chat application
python main.py chat sample_docs/