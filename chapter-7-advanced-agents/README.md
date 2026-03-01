# Chapter 7: Advanced Agent Architectures

Complete implementation of advanced agent patterns, multi-agent systems, and production-ready agent infrastructure.

## Repository Structure
```
chapter-7-advanced-agents/
├── core/              # Base agent classes and state management
├── patterns/          # Agent patterns (ReAct, Reflection, Planning)
├── memory/            # Memory systems (short-term, long-term, semantic)
├── tools/             # Tool implementations and registry
├── multi_agent/       # Multi-agent coordination
├── monitoring/        # Agent monitoring and tracing
├── safety/            # Safety guardrails and cost limiting
├── mcp_integration/   # MCP server implementation
├── evaluation/        # Agent evaluation framework
└── examples/          # Complete working examples
```

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage
```python
from patterns.react_agent import ReActAgent
from tools.search_tools import WebSearchTool
from anthropic import Anthropic

# Initialize
llm = Anthropic(api_key="your-key")
agent = ReActAgent("Assistant", llm, tools=[WebSearchTool()])

# Run task
result = agent.run("What is the capital of France?")
print(result['result'])
```

## Agent Patterns

### ReAct (Reasoning + Acting)
Interleaves thinking and tool use for step-by-step problem solving.

### Reflection
Self-evaluates outputs and iteratively improves responses.

### Planning
Creates explicit plans before execution, manages complex multi-step tasks.

## Memory Systems

- **Short-term**: Conversation buffer for recent context
- **Long-term**: Episodic memory using vector embeddings
- **Semantic**: Factual knowledge storage and retrieval

## Multi-Agent Systems

- **Agent Teams**: Collaborative task execution
- **Hierarchical Teams**: Manager-worker coordination
- **Communication**: Inter-agent messaging protocols

## Examples

Run examples from the examples directory:
```bash
python examples/01_simple_react_agent.py
python examples/08_autonomous_researcher.py
python examples/09_complete_agent_system.py
```

## Safety and Monitoring

- Safety guardrails prevent harmful actions
- Cost limiters manage API budgets
- Comprehensive monitoring tracks agent behavior
- Detailed execution tracing for debugging

## Documentation

See `docs/` directory for:
- Architecture overview
- Agent pattern details
- Tool development guide
- Multi-agent system design

## License

MIT License - see LICENSE file for details