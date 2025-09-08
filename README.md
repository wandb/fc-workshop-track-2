# 🤖 Agentic AI Systems Workshop

> **Learn to build, optimize, and evaluate production-ready multi-agent AI systems**

Build autonomous agents that coordinate across complex scenarios using CrewAI, evaluate them with LLM-as-a-Judge, and optimize for production deployment.

**Powered by [Weave](https://weave-docs.wandb.ai)**: LLM application evaluation and tracing


---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11+
- Google API key (for Gemini models)
- Weights & Biases account (for evaluation tracking)
- Note: Tested with Linux and MacOS. Windows was not tested.

### **1. Setup Environment**
```bash
# Clone and enter the repository
git clone https://github.com/wandb/fc-workshop-track-2
cd fc-workshop-track-2

# Install uv (modern Python package manager)
pip install uv

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### **2. Start Workshop**
```bash
# Option 1: Full workshop experience (recommended)
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
jupyter lab

# Option 2: Individual sessions
jupyter lab morning_session.ipynb    # 3 hours - Building agents
jupyter lab afternoon_session.ipynb  # 3 hours - Optimization & evaluation
```

---

## 🎯 **What You'll Build**

### **Morning Session (3 hours): Architecting and Orchestrating AI Agents**

**🎓 Core Learning**: Design and orchestrate agentic AI systems using modern frameworks, standards, and best practices. Master foundational design principles including tool use, task planning, autonomy, and multi-agent collaboration.

**Key Architectural Concepts:**
- **🏗️ Agent Design Patterns**: Build autonomous agents that make decisions, invoke tools, and accomplish complex tasks without rigid pre-programmed flows
- **🔧 Dynamic Tool Integration**: Learn how emerging standards like Model Context Protocol (MCP) simplify agent discovery and use of external tools
- **🤝 Multi-Agent Coordination**: Orchestrate specialized agents working together through hierarchical and collaborative patterns
- **⚖️ Architecture Comparison**: Quantitatively compare rule-based vs agent-based vs LLM chain approaches
- **🔄 Adaptive Systems**: Create systems that adapt to new scenarios and integrate external systems dynamically

**Practical Implementation:**
- Smart city crisis management system with Grid, Emergency, and Traffic coordination
- Specialized tools with Pydantic models for reliable agent communication
- Real-time decision making under resource constraints and changing conditions

```python
# Example: Building autonomous decision-making agents
crisis_crew = Crew(
    agents=[grid_specialist, emergency_coordinator, traffic_manager, feedback_specialist],
    tasks=[assess_situation, coordinate_response, adapt_to_changes],
    process=Process.hierarchical  # Orchestration pattern
)
```

### **Afternoon Session (3 hours): Optimizing and Evaluating Agentic AI Applications**

**🎓 Core Learning**: Shift from building to optimizing and evaluating agentic AI applications. Implement sophisticated evaluation strategies that measure agent decision-making quality, optimize responsiveness, and incorporate human feedback for continuous adaptation.

**Key Optimization Concepts:**
- **📏 Multi-Dimensional Evaluation**: Implement evaluation strategies beyond simple success rates - measure decision quality, efficiency, and adaptability
- **⚡ Performance Optimization**: Reduce latency through parallel processing, caching strategies, and dynamic model selection  
- **🔄 Human Feedback Integration**: Build systems that learn and adapt from real-world input and iterative enhancement
- **👁️ Agent Observability**: Comprehensive monitoring and tracing of agent behavior and decision-making processes
- **🏗️ Production Scaling**: Use MCP standards to simplify scaling, monitoring, and integration with external systems
- **🎯 Online Evaluation**: Real-time assessment strategies for continuous improvement in production environments

**Practical Implementation:**
- LLM-as-a-Judge evaluation frameworks for decision quality assessment
- Performance optimization techniques achieving sub-2-second response times
- Human feedback loops that modify agent behavior based on real-world outcomes
- Production monitoring with complete observability into agent reasoning

```python
# Example: Comprehensive evaluation and optimization
@weave.op()
async def evaluate_and_optimize_agents(scenario):
    # Multi-dimensional evaluation
    performance_metrics = await evaluate_decision_quality(agent_response)
    
    # Dynamic optimization based on results
    optimized_config = await adapt_based_on_feedback(performance_metrics)
    
    return optimized_agent_system
```

---

## 🏗️ **Technical Architecture**

### **Services & APIs**
- **Grid Management** (`localhost:8002`): Power distribution, load balancing, infrastructure priorities
- **Emergency Response** (`localhost:8003`): Drone deployment, incident management, resource allocation  
- **Traffic Coordination** (`localhost:8004`): Flow optimization, emergency corridors, congestion management
- **Scenario Management** (`localhost:8005`): Crisis simulation and state management

### **Key Technologies**
- **[Weave](https://weave-docs.wandb.ai)**: LLM application evaluation and tracing
- **[CrewAI](https://github.com/joaomdmoura/crewAI)**: Multi-agent orchestration framework
- **[Pydantic](https://docs.pydantic.dev/)**: Structured outputs and data validation
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance async service APIs

### **Workshop Scenarios**
- **Heat Wave Crisis**: Power grid overload, cooling center management
- **Cyber Attack**: Service degradation, resource reallocation
- **Major Earthquake**: Emergency response, infrastructure damage
- **Festival Emergency**: Crowd control, traffic management
- **Multi-Domain Crisis**: Complex coordination across all services

---

## 📁 **Repository Structure**

```
├── morning_session.ipynb           # Multi-agent system development
├── afternoon_session.ipynb         # Optimization and evaluation  
├── workshop/                       # Core framework
├── pyproject.toml                 # Python dependencies (uv compatible)
```

---

## 🎓 **Learning Outcomes**

### **Technical Skills**
- Design and implement multi-agent systems using modern frameworks
- Create reliable agent communication with structured outputs
- Build comprehensive evaluation frameworks beyond simple metrics
- Optimize agent performance through parallel processing and caching
- Integrate human feedback for continuous system improvement
- Deploy production-ready agent systems with proper monitoring

### **Conceptual Understanding**
- When to use agents vs rules vs LLM chains
- Multi-agent coordination patterns and trade-offs
- Evaluation strategies for autonomous decision-making systems
- Human-AI collaboration and feedback loop design
- Production deployment considerations for agentic systems

---

## 🏆 **Workshop Format**

**Interactive Jupyter Notebooks** with:
- 📝 Educational content with step-by-step instructions
- 💻 Modular code cells you can modify and experiment with
- 🛠️ Hands-on exercises building understanding through practice
- 📊 Live evaluation and performance comparison
- 🤝 Group discussions and collaborative learning
- 🏁 Competitive final challenge with leaderboard

**Schedule:**
- **Morning (3h)**: Foundation building → Agent development → System integration
- **Break (1h)**: Lunch and networking  
- **Afternoon (3h)**: Evaluation → Optimization → Competition

---

## 🔧 **Troubleshooting**

### **Common Issues**
```bash
# Environment issues  
uv sync --reinstall

# Jupyter kernel problems
python -m ipykernel install --user --name=.venv
```

### **Requirements**
- **API Limits**: Google Gemini API with sufficient quota for ~100 LLM calls
- **Network**: Stable connection for real-time service interaction

---

## 📈 **Expected Outcomes**

By workshop end, you'll have:
- ✅ Working multi-agent system handling complex coordination
- ✅ Comprehensive evaluation framework with quantitative metrics  
- ✅ Optimized system with sub-2-second response times
- ✅ Production-ready deployment patterns and monitoring
- ✅ Competition score and performance comparison data

**Ready to build the future of autonomous AI systems? Let's get started! 🚀**