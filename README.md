# 🤖 Agentic AI Systems Workshop

## 🏙️ Operation SENTINEL GRID

NeoCatalis, a fully connected smart city, has experienced a systemic AI blackout. This workshop simulates rebuilding the city's autonomy using modern agentic principles through two comprehensive sessions.

---

## 🚀 **Quick Start**

### **Setup**
```bash
# Install uv package manager
pip install uv

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add your AWS credentials to .env (openai for now)
# Add your wandb API Key to .env
```

### **Run Workshop**
```bash
# Morning session (3 hours)
python morning_session.py

# Afternoon session (3 hours) 
python afternoon_session.py
```

---

## 🌅 **MORNING SESSION: Building Agentic AI Systems**

### Learning Objective
Learn to design and orchestrate agentic AI systems using modern frameworks. Cover tool use, task planning, autonomy, multi-agent collaboration, and Model Context Protocol (MCP) for dynamic external system integration.

### What You'll Build
- Multi-agent systems with specialized domain expertise
- Dynamic tool discovery and structured outputs
- Production-ready agent orchestration patterns
- Comparative evaluation: Rules vs Agents vs LLM chains

### Key Phases
1. **Environment Setup** - Service discovery and crisis scenarios
2. **LLM Chain Analysis** - Why sophisticated prompting isn't enough
3. **Service Investigation** - Grid, Emergency, Traffic (Rules → Tools → Agents)
4. **System Comparison** - Full rule-based vs agent-based systems
5. **Adaptability Test** - New scenario evaluation
6. **MCP Integration** - Dynamic tool discovery
7. **Results Analysis** - Performance comparison and insights

---

## 🌆 **AFTERNOON SESSION: Optimization & Evaluation**

### Learning Objective
Shift from building to optimizing agentic AI applications. Implement evaluation strategies, optimize responsiveness, integrate human feedback, and prepare for production deployment.

### What You'll Build
- Comprehensive evaluation frameworks with LLM-as-a-Judge
- Performance optimization techniques
- Human feedback integration systems
- Production monitoring and scalability patterns
- Competitive optimization showcase

### Key Phases
1. **Evaluation Framework** - Multi-dimensional metrics and LLM-as-a-Judge
2. **Baseline Measurement** - Performance testing across scenarios
3. **MCP Integration** - Dynamic service discovery and tool registration
4. **Human Feedback** - Agent adaptation and behavioral modification
5. **Production Patterns** - Scalable architecture design
6. **Final Competition** - 6-dimensional scoring and leaderboard

---

## 🏗️ **Technical Infrastructure**

### SENTINEL GRID Services
- **⚡ Grid Management**: Power distribution, load balancing, infrastructure priorities
- **🚁 Emergency Response**: Drone deployment, incident management, resource allocation  
- **🚦 Traffic Coordination**: Flow optimization, emergency corridors, congestion management

### Key Scenarios
- Heat Wave Crisis, Cyber Attack, Major Earthquake, Festival Emergency, Complex Multi-Domain Crisis

### Evaluation Metrics
- Incident Coverage, Response Time, Tool Failure Handling, Capability Match, Latency, Decision Quality

---

## 📁 **Repository Structure**

```
/
├── morning_session.py              # Morning workshop (building agents)
├── afternoon_session.py            # Afternoon workshop (optimization)
├── workshop/                       # Core infrastructure
│   ├── command.py                  # Command execution system
│   ├── agent_system.py             # Agent orchestration
│   ├── state_management.py         # Service state handling
│   ├── command_evaluator.py        # Performance evaluation
│   ├── agent_converter.py          # Agent result processing
│   ├── afternoon_session_utils.py  # Optimization utilities
│   ├── scenarios.py                # Crisis scenarios
│   ├── state_models.py             # Data models
│   ├── service_management.py       # Service lifecycle
│   ├── config.py                   # Configuration
│   ├── day_seed_generator.py       # Scenario generation
│   └── services/                   # Service implementations
├── results/                        # Workshop analytics
└── .env.example                    # Environment template
```

---

## 🎓 **Learning Objectives**

### Technical Skills
- Design modular agentic systems using CrewAI
- Build specialized tools with structured outputs
- Implement comprehensive evaluation frameworks
- Optimize performance through various techniques
- Integrate human feedback for continuous improvement
- Utilize MCP for dynamic tool discovery

### Conceptual Understanding
- Agent vs Rule-Based Systems trade-offs
- Multi-agent coordination patterns
- Production deployment considerations
- Evaluation methodologies beyond success metrics
- Human-AI interaction and feedback loops

---

## 🏆 **Workshop Format**

- **Morning (3 hours)**: Foundation building and agent development
- **☕ Break (1 hour)**: Lunch and networking
- **Afternoon (3 hours)**: Optimization, evaluation, and competition
- **Interactive**: Jupyter notebook format with hands-on exercises
- **Competitive**: Final leaderboard across 6 evaluation dimensions

