# LangGraph Multi-Agent Systems Tutorial

This tutorial demonstrates three progressively sophisticated multi-agent architectures using LangGraph, showcasing different patterns of agent collaboration and coordination.

## 📚 Learning Progression: Easy → Intermediate → Advanced

### Example 1: Basic Multi-Agent System (EASY)
**File**: [`example1_basic_multi_agent.py`](./example1_basic_multi_agent.py)

**Concepts Demonstrated**:
- Simple sequential agent workflow
- Basic message passing between agents
- Clear separation of responsibilities
- Simple routing logic
- Two-agent collaboration pattern

**Architecture**: 
```
User Input → Researcher Agent → Summarizer Agent → Final Output
```

**Scenario**: Question & Answer System
- **Researcher Agent**: Gathers detailed information about a topic
- **Summarizer Agent**: Creates concise summaries of research findings

**Key Features**:
- ✅ Sequential processing with clear handoffs
- ✅ Simple state management
- ✅ Basic agent specialization
- ✅ Easy to understand and modify

**Perfect for**: Learning multi-agent basics, understanding agent communication patterns

---

### Example 2: Supervisor-Worker Pattern (INTERMEDIATE)
**File**: [`example2_supervisor_workers.py`](./example2_supervisor_workers.py)

**Concepts Demonstrated**:
- Supervisor-worker coordination pattern
- Dynamic task assignment based on content analysis
- Specialized tools for different worker types
- Conditional routing and decision making
- Multi-agent tool usage

**Architecture**:
```
                    Supervisor Agent
                    (Task Assignment)
                          |
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
Text Analyst        Data Analyst        Sentiment Analyst
(Topics/Themes)    (Statistics/Data)    (Emotional Tone)
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
                   Final Report
```

**Scenario**: Content Analysis Team
- **Supervisor**: Analyzes requests and assigns appropriate workers
- **Text Analyst**: Extracts topics and themes using specialized tools
- **Data Analyst**: Processes numerical data and statistics
- **Sentiment Analyst**: Evaluates emotional tone and sentiment

**Key Features**:
- ✅ Intelligent task delegation
- ✅ Specialized worker tools and capabilities
- ✅ Dynamic workflow based on content type
- ✅ Comprehensive analysis from multiple perspectives
- ✅ Supervisor coordination and final synthesis

**Perfect for**: Understanding hierarchical agent patterns, tool specialization, dynamic task assignment

---

### Example 3: Collaborative Agent Swarm (ADVANCED)
**File**: [`example3_collaborative_swarm.py`](./example3_collaborative_swarm.py)

**Concepts Demonstrated**:
- Peer-to-peer agent collaboration
- Multi-round consensus building
- Cross-agent knowledge sharing
- Adaptive role assignment
- Complex coordination patterns
- Sophisticated state management

**Architecture**:
```
Round 1: Core Analysis
┌─────────────┬─────────────┬─────────────┐
│  Strategy   │   Market    │  Finance    │
│   Agent     │   Agent     │   Agent     │
└─────────────┴─────────────┴─────────────┘
                     ↓
Round 2: Risk & Innovation
┌─────────────┬─────────────────────────────┐
│    Risk     │        Innovation           │
│   Agent     │         Agent               │
└─────────────┴─────────────────────────────┘
                     ↓
Round 3: Consensus Building
┌───────────────────────────────────────────┐
│            Coordinator Agent             │
│         (Consensus & Synthesis)           │
└───────────────────────────────────────────┘
```

**Scenario**: Strategic Business Planning Swarm
- **Strategy Agent**: Develops high-level strategic approaches
- **Market Agent**: Analyzes market conditions and opportunities
- **Finance Agent**: Evaluates financial implications and projections
- **Risk Agent**: Identifies and assesses potential risks
- **Innovation Agent**: Proposes creative solutions and innovations
- **Coordinator Agent**: Facilitates collaboration and builds consensus

**Key Features**:
- ✅ Multi-round collaborative workflow
- ✅ Cross-agent insight sharing and knowledge building
- ✅ Consensus building and conflict resolution
- ✅ Adaptive collaboration patterns
- ✅ Complex state management across multiple agents
- ✅ Sophisticated coordination and synthesis

**Perfect for**: Advanced multi-agent systems, collaborative decision making, complex business scenarios

## 🚀 Getting Started

### Prerequisites

Install required packages:
```bash
pip install langgraph langchain_openai langchain_core typing_extensions
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Examples

Each example can be run independently:

```bash
# Basic Multi-Agent System
python example1_basic_multi_agent.py

# Supervisor-Worker Pattern  
python example2_supervisor_workers.py

# Collaborative Agent Swarm
python example3_collaborative_swarm.py
```

## 🎯 Use Cases by Example

### Example 1 - Basic Multi-Agent
- **Research & Summarization**: Academic research, market analysis
- **Content Processing**: Document analysis, report generation
- **Simple Workflows**: Any two-step process requiring specialization

### Example 2 - Supervisor-Worker  
- **Content Analysis**: Social media monitoring, customer feedback analysis
- **Document Processing**: Legal document review, technical documentation
- **Quality Assurance**: Multi-perspective content evaluation

### Example 3 - Collaborative Swarm
- **Strategic Planning**: Business strategy development, market entry planning
- **Complex Decision Making**: Investment decisions, product development
- **Multi-Stakeholder Analysis**: Policy development, organizational change

## 🏗️ Architecture Patterns Explained

### 1. Sequential Pattern (Example 1)
- **Flow**: Linear agent-to-agent handoffs
- **Coordination**: Simple message passing
- **State**: Minimal shared state
- **Best for**: Simple, predictable workflows

### 2. Hierarchical Pattern (Example 2)
- **Flow**: Central coordinator with specialized workers
- **Coordination**: Top-down task assignment
- **State**: Structured with role-specific data
- **Best for**: Complex tasks requiring different expertise

### 3. Collaborative Pattern (Example 3)
- **Flow**: Multi-round peer collaboration
- **Coordination**: Consensus building and knowledge synthesis
- **State**: Rich, multi-dimensional agent contributions
- **Best for**: Complex decision making requiring multiple perspectives

## 🔧 Key Technical Concepts

### State Management
- **Basic**: Simple fields for agent handoffs
- **Intermediate**: Role-specific state compartments
- **Advanced**: Multi-dimensional collaboration tracking

### Routing Logic
- **Basic**: Simple next-agent determination
- **Intermediate**: Conditional routing based on content analysis
- **Advanced**: Multi-round workflow with consensus building

### Tool Integration
- **Basic**: No tools (pure LLM reasoning)
- **Intermediate**: Specialized tools per worker type
- **Advanced**: Collaborative tools for consensus building

### Agent Communication
- **Basic**: Direct message passing
- **Intermediate**: Supervisor-mediated coordination
- **Advanced**: Peer-to-peer knowledge sharing

## 💡 Best Practices

### When to Use Each Pattern

**Use Basic Multi-Agent when**:
- You have a simple, linear workflow
- Each step requires different expertise
- You want maximum simplicity and clarity

**Use Supervisor-Worker when**:
- You need dynamic task assignment
- Different tasks require different tools
- You want centralized coordination

**Use Collaborative Swarm when**:
- You need multiple perspectives on complex problems
- Consensus building is important
- The problem requires iterative refinement

### Performance Considerations

- **Basic**: Fastest execution, minimal overhead
- **Intermediate**: Moderate complexity, good for most use cases
- **Advanced**: Highest complexity, use for complex decision making

### Customization Tips

1. **Modify Agent Roles**: Change agent specializations to fit your domain
2. **Adjust Routing Logic**: Customize when agents are activated
3. **Add/Remove Agents**: Scale the number of agents based on complexity
4. **Custom Tools**: Create domain-specific tools for your agents
5. **State Fields**: Add fields to track domain-specific information

## 🎓 Educational Value

This tutorial progression teaches:

1. **Foundational Concepts** (Example 1)
   - Multi-agent basics
   - State management
   - Agent communication

2. **Intermediate Patterns** (Example 2)
   - Hierarchical coordination
   - Tool specialization
   - Dynamic routing

3. **Advanced Techniques** (Example 3)
   - Collaborative workflows
   - Consensus building
   - Complex state management

Each example builds upon the previous one, providing a comprehensive learning path for multi-agent system development with LangGraph.