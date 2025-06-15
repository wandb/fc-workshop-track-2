#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# 📊 **Agentic AI Optimization & Evaluation Workshop: 
From Building to Production Excellence**

## ⏰ **WORKSHOP SCHEDULE (3 Hours Total)**

### **Part 1: Foundation & Measurement (75 minutes)**
- **Phase 1**: Evaluation Framework Setup (15 min)
- **Phase 2**: Performance Baseline Measurement (20 min)
- **🛠️ HANDS-ON**: Experiment with evaluation metrics (15 min)
- **💬 DISCUSSION**: What makes good evaluation? (10 min)
- **Phase 3**: Latency Optimization Strategies (15 min)

### **☕ COFFEE BREAK (10 minutes)**

### **Part 2: Advanced Techniques (75 minutes)**
- **Phase 4**: Human Feedback Integration (20 min)
- **🛠️ HANDS-ON**: Customize feedback criteria (15 min)
- **Phase 5**: Model Context Protocol (MCP) Integration (20 min)
- **🛠️ HANDS-ON**: MCP experimentation (15 min)
- **Phase 6**: Production Integration Patterns (5 min)

### **Part 3: OPTIMIZATION COMPETITION (20 minutes)**
- **🏆 FINAL CHALLENGE**: Beat the leaderboard! (15 min)
- **🎯 RESULTS & WINNERS**: Compare scores and techniques (5 min)

---

## 🎯 **Workshop Learning Objectives**

In this session, we shift focus from building to optimizing and evaluating 
agentic AI applications. You'll learn how to implement evaluation strategies 
that measure agent decision-making quality, optimize responsiveness, reduce 
latency, and incorporate human feedback for continuous adaptation. We'll cover 
dynamic model selection, agent observability, online evaluation strategies, 
and how to use standards like MCP to simplify scaling and monitoring. The 
emphasis will be on improving agentic systems through real-world feedback 
and iterative enhancement.

## 🎓 **What You'll Build Today**

By the end of this workshop, you will have:

1. **📏 A Comprehensive Evaluation Framework**: Multi-dimensional metrics that 
   capture agent decision-making quality and system performance
2. **⚡ Optimized Agent Systems**: Reduced latency through parallel processing, 
   caching, and model selection strategies  
3. **🔄 Human Feedback Integration**: Systems that learn and adapt from 
   real-world input
4. **🎯 Dynamic Model Selection**: Automatic optimization based on scenario 
   complexity and requirements
5. **👁️ Production Monitoring**: Complete observability into agent behavior 
   and performance
6. **🏆 Competition Entry**: Your optimized system competing for the highest score!

## 🛠️ **Interactive Notebook Format**

This workshop is designed for **Jupyter notebook interaction**! Each section contains:

- **📝 Markdown cells** with educational content and instructions
- **👨‍💻 Code cells** with adjustable parameters for experimentation
- **🛠️ Interactive exercises** to modify and observe results
- **💬 Discussion prompts** for reflection and team learning
- **📊 Visualization cells** showing performance comparisons

### **🏆 COMPETITION SCORING**
Throughout the workshop, you'll earn points for:
- **Baseline Performance**: Starting score based on initial measurements
- **Optimization Gains**: Points for improving response time and decision quality
- **Innovation Bonus**: Extra points for creative optimization strategies
- **Final Challenge**: Head-to-head competition in the last 20 minutes!

---
"""

# All imports from morning session - reusing existing infrastructure
import logging
import time
from typing import List, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime

# Import existing workshop components
from workshop.command import Command, ServiceType, CommandExecutor
from workshop.command_evaluator import evaluate_scenario_commands
from workshop.agent_system import ScenarioType
from workshop.state_models import ScenarioDefinition
from workshop.agent_converter import convert_and_evaluate_agent_commands

# CrewAI imports - Core framework for building multi-agent systems
from crewai import Crew, Process
from pydantic import BaseModel, Field

# Import state management utilities
from workshop.state_management import (
    reset_all_service_states,
    get_actual_service_ids,
    get_system_status
)

# Import service management utilities
from workshop.service_management import (
    check_environment,
    start_services,
    save_experiment_results
)

# Import from afternoon session utilities
from workshop.afternoon_session_utils import (
    create_evaluation_scenarios,
    create_optimized_grid_agent,
    create_optimized_emergency_agent,
    create_optimized_traffic_agent,
    create_optimized_agent_tasks,
    create_heat_wave_scenario_for_evaluation,
    create_workshop_checkpoint,
    create_discussion_prompt,
    display_workshop_progress,
    create_hands_on_exercise
)

# Suppress LiteLLM debug logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.WARNING)

# Load environment variables and setup console
load_dotenv()
console = Console()

# Structured output models for agent communication (from morning session)
class ZoneAdjustment(BaseModel):
    """Zone capacity adjustment action."""
    zone_id: str = Field(description="ID of the zone to adjust")
    capacity: float = Field(description="New capacity ratio (0.0-1.0)")
    reason: str = Field(description="Reason for the adjustment")


class InfrastructurePriority(BaseModel):
    """Infrastructure priority setting action."""
    infrastructure_id: str = Field(description="ID of infrastructure")
    level: str = Field(description="Priority level (normal, high, critical)")
    reason: str = Field(description="Reason for priority change")


class GridManagementPlan(BaseModel):
    """Structured output for grid management actions."""
    zone_adjustments: List[ZoneAdjustment] = Field(
        description="Zone capacity adjustments to prevent overloads"
    )
    priority_settings: List[InfrastructurePriority] = Field(
        description="Infrastructure priority changes for critical facilities"
    )
    stability_forecast: str = Field(
        description="Expected grid stability after implementing changes"
    )
    coordination_notes: str = Field(
        description="Notes for other agents about grid impacts"
    )


class DroneAssignment(BaseModel):
    """Drone assignment action."""
    drone_id: str = Field(description="ID of the drone to assign")
    incident_id: str = Field(description="ID of the incident to respond to")
    reason: str = Field(description="Reason for this assignment")


class IncidentUpdate(BaseModel):
    """Incident status update action."""
    incident_id: str = Field(description="ID of the incident to update")
    status: str = Field(description="New status (active, assigned, resolved)")
    reason: str = Field(description="Reason for status change")


class EmergencyResponsePlan(BaseModel):
    """Structured output for emergency response actions."""
    drone_assignments: List[DroneAssignment] = Field(
        description="Drone to incident assignments prioritized by urgency"
    )
    incident_updates: List[IncidentUpdate] = Field(
        description="Incident status updates to track response progress"
    )
    resource_allocation: str = Field(
        description="Summary of how limited resources are being allocated"
    )
    coordination_notes: str = Field(
        description="Notes for other agents about emergency operations"
    )


class TrafficRedirection(BaseModel):
    """Traffic redirection action."""
    sector_id: str = Field(description="ID of the traffic sector to redirect")
    target_reduction: float = Field(description="Target congestion reduction")
    reason: str = Field(description="Reason for redirection")


class RouteBlocking(BaseModel):
    """Route blocking action."""
    sector_id: str = Field(description="ID of the sector to block")
    duration_minutes: int = Field(description="Duration to block in minutes")
    reason: str = Field(description="Reason for blocking")


class TrafficManagementPlan(BaseModel):
    """Structured output for traffic management actions."""
    traffic_redirections: List[TrafficRedirection] = Field(
        description="Traffic redirection actions to reduce congestion"
    )
    route_blocks: List[RouteBlocking] = Field(
        description="Route blocking actions for emergency access"
    )
    emergency_corridors: str = Field(
        description="Description of maintained emergency vehicle access routes"
    )
    coordination_notes: str = Field(
        description="Notes for other agents about traffic impacts"
    )


# LLM-as-a-Judge evaluation models
class AgentDecisionAssessment(BaseModel):
    """LLM assessment of agent decision quality."""
    decision_score: float = Field(
        description="Decision quality score (0.0-1.0)",
        ge=0.0, le=1.0
    )
    reasoning_quality: float = Field(
        description="Quality of reasoning behind decisions (0.0-1.0)",
        ge=0.0, le=1.0
    )
    action_appropriateness: float = Field(
        description="How appropriate actions are for scenario (0.0-1.0)",
        ge=0.0, le=1.0
    )
    resource_efficiency: float = Field(
        description="Efficiency of resource utilization (0.0-1.0)",
        ge=0.0, le=1.0
    )
    coordination_effectiveness: float = Field(
        description="How well agent coordinates with others (0.0-1.0)",
        ge=0.0, le=1.0
    )
    improvement_suggestions: List[str] = Field(
        description="Specific suggestions for improvement"
    )
    strengths: List[str] = Field(
        description="Key strengths demonstrated by the agent"
    )
    overall_assessment: str = Field(
        description="Overall qualitative assessment of performance"
    )


# Evaluation and optimization results tracking
evaluation_results = {
    "baseline_performance": {},
    "optimization_metrics": {},
    "human_feedback": {},
    "model_selection": {},
    "observability_data": {},
    "online_evaluation": {},
    "mcp_monitoring": {},
    "llm_judge_evaluations": {}
}

OPTIMIZATION_RESULTS_FILE = "workshop_optimization_results.json"

# %% [markdown]
"""
## 📏 **PHASE 1: Evaluation Framework Setup**

**🎓 Educational Goal**: Establish comprehensive evaluation metrics for 
agent systems

In this phase, we'll:
1. Design multi-dimensional evaluation frameworks
2. Implement decision quality metrics
3. Create performance benchmarking systems
4. Establish baseline measurements

**🔑 Key Learning**: Production agent systems require sophisticated evaluation 
beyond simple success rates - we need to measure decision quality, efficiency, 
and adaptability
"""

# %%
@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for agent systems."""
    decision_quality: float  # How good are the agent's decisions
    response_time: float     # Time to generate response (ms)
    execution_efficiency: float  # Resource utilization efficiency
    adaptability_score: float   # Ability to handle new scenarios
    coordination_quality: float  # Multi-agent coordination effectiveness
    user_satisfaction: float    # Human feedback scores
    
    def overall_score(self) -> float:
        """Calculate weighted overall performance score."""
        weights = {
            "decision_quality": 0.3,
            "response_time": 0.15,
            "execution_efficiency": 0.15,
            "adaptability_score": 0.2,
            "coordination_quality": 0.15,
            "user_satisfaction": 0.05
        }
        
        # Normalize response_time (lower is better)
        normalized_response_time = max(0, 1 - (self.response_time / 10000))
        
        return (
            weights["decision_quality"] * self.decision_quality +
            weights["response_time"] * normalized_response_time +
            weights["execution_efficiency"] * self.execution_efficiency +
            weights["adaptability_score"] * self.adaptability_score +
            weights["coordination_quality"] * self.coordination_quality +
            weights["user_satisfaction"] * self.user_satisfaction
        )


class AdvancedEvaluationFramework:
    """Comprehensive evaluation framework for agent systems."""
    
    def __init__(self):
        self.evaluation_history = []
        self.decision_patterns = {}
        self.performance_trends = {}
        self.llm_judge_history = []
        
    def llm_as_judge_evaluation(self, agent_result, scenario: ScenarioDefinition,
                               commands: List[Command]) -> AgentDecisionAssessment:
        """Use LLM as a judge to evaluate agent decision quality."""
        try:
            import litellm
            
            # Convert commands to readable format for LLM evaluation
            commands_summary = []
            for cmd in commands:
                commands_summary.append({
                    "service": cmd.service.value,
                    "action": cmd.action,
                    "parameters": cmd.parameters
                })
            
            evaluation_prompt = f"""You are an expert evaluator of AI agent 
decision-making in crisis management scenarios.

SCENARIO: {scenario.name}
Description: {scenario.description}

AGENT ACTIONS TAKEN:
{commands_summary}

SCENARIO CONTEXT:
- Grid zones with varying load levels
- Emergency incidents requiring drone response
- Traffic congestion affecting emergency access

Evaluate the agent's performance across multiple dimensions and provide 
structured feedback.

Consider:
1. Decision Quality: Are the decisions logical and well-reasoned?
2. Action Appropriateness: Do the actions fit the scenario requirements?
3. Resource Efficiency: Is the agent using resources optimally?
4. Coordination: How well do actions coordinate across services?
5. Reasoning Quality: Is there clear logic behind decisions?

Provide specific, actionable feedback for improvement."""

            # Use structured output with LiteLLM
            response = litellm.completion(
                model="gpt-4o",
                messages=[
                    {"role": "system", 
                     "content": ("You are an expert evaluator of AI agent "
                               "systems. Provide detailed, structured "
                               "assessments.")},
                    {"role": "user", "content": evaluation_prompt}
                ],
                response_format=AgentDecisionAssessment,
                temperature=0.1
            )
            
            # Parse structured response
            assessment = AgentDecisionAssessment.model_validate(
                response.choices[0].message.content
            )
            
            # Store in history
            self.llm_judge_history.append({
                "timestamp": datetime.now().isoformat(),
                "scenario": scenario.name,
                "assessment": assessment,
                "commands_count": len(commands)
            })
            
            return assessment
            
        except Exception as e:
            console.print(f"[yellow]LLM Judge evaluation failed: {e}[/yellow]")
            # Return default assessment
            return AgentDecisionAssessment(
                decision_score=0.5,
                reasoning_quality=0.5,
                action_appropriateness=0.5,
                resource_efficiency=0.5,
                coordination_effectiveness=0.5,
                improvement_suggestions=["LLM evaluation unavailable"],
                strengths=["Unable to assess"],
                overall_assessment="LLM evaluation failed"
            )
    
    def evaluate_decision_quality(self, commands: List[Command], 
                                scenario: ScenarioDefinition) -> float:
        """Evaluate the quality of agent decisions."""
        if not commands:
            return 0.0
            
        # Use existing evaluation but enhance with decision analysis
        evaluation = evaluate_scenario_commands(
            commands=[{
                "service": cmd.service.value,
                "action": cmd.action,
                "parameters": cmd.parameters,
                "success": True
            } for cmd in commands],
            scenario_type=ScenarioType.GRID_SURGE,
            current_state=get_system_status(),
            scenario_definition=scenario
        )
        
        base_score = evaluation.get('overall_score', 0.0)
        
        # Enhanced decision quality analysis
        quality_factors = self._analyze_decision_patterns(commands, scenario)
        
        # Add LLM judge evaluation
        llm_assessment = self.llm_as_judge_evaluation(None, scenario, commands)
        llm_score = llm_assessment.decision_score
        
        # Combine base score, pattern analysis, and LLM judgment
        enhanced_score = (base_score * 0.5) + (quality_factors * 0.3) + (llm_score * 0.2)
        
        return min(1.0, enhanced_score)
    
    def _analyze_decision_patterns(self, commands: List[Command], 
                                 scenario: ScenarioDefinition) -> float:
        """Analyze decision patterns for quality assessment."""
        if not commands:
            return 0.0
            
        quality_score = 0.0
        
        # Check for service coverage (good decisions cover all relevant services)
        services_used = set(cmd.service for cmd in commands)
        expected_services = {ServiceType.GRID, ServiceType.EMERGENCY, ServiceType.TRAFFIC}
        coverage_score = len(services_used.intersection(expected_services)) / len(expected_services)
        
        # Check for logical action sequences
        grid_commands = [cmd for cmd in commands if cmd.service == ServiceType.GRID]
        emergency_commands = [cmd for cmd in commands if cmd.service == ServiceType.EMERGENCY]
        
        sequence_score = 0.5  # Default
        if grid_commands and emergency_commands:
            # Good pattern: grid stabilization before emergency response
            sequence_score = 0.8
            
        # Check for resource efficiency (not over-allocating)
        efficiency_score = self._evaluate_resource_efficiency(commands)
        
        quality_score = (coverage_score * 0.4 + sequence_score * 0.3 + efficiency_score * 0.3)
        
        return quality_score
    
    def _evaluate_resource_efficiency(self, commands: List[Command]) -> float:
        """Evaluate how efficiently resources are allocated."""
        if not commands:
            return 0.0
            
        # Count drone assignments vs available drones
        actual_ids = get_actual_service_ids()
        available_drones = len(actual_ids.get('drones', []))
        
        drone_assignments = len([cmd for cmd in commands 
                               if cmd.action == "assign_drone"])
        
        if available_drones == 0:
            return 0.5  # Default score if no drones available
            
        # Efficient usage: using most drones but not over-assigning
        efficiency_ratio = min(1.0, drone_assignments / available_drones)
        
        # Penalty for under-utilization or over-assignment
        if efficiency_ratio < 0.5:  # Under-utilization
            return efficiency_ratio
        elif efficiency_ratio > 1.0:  # Over-assignment (impossible but check)
            return 0.3
        else:
            return efficiency_ratio
    
    def measure_response_time(self, agent_function, *args, **kwargs) -> Tuple[Any, float]:
        """Measure response time of agent function execution."""
        start_time = time.perf_counter()
        result = agent_function(*args, **kwargs)
        end_time = time.perf_counter()
        
        response_time_ms = (end_time - start_time) * 1000
        return result, response_time_ms
    
    def evaluate_coordination_quality(self, multi_agent_results: List[Any]) -> float:
        """Evaluate how well multiple agents coordinated."""
        if len(multi_agent_results) < 2:
            return 0.5  # Single agent, no coordination to evaluate
            
        # Check for complementary actions vs conflicting actions
        all_commands = []
        for result in multi_agent_results:
            if hasattr(result, 'commands'):
                all_commands.extend(result.commands)
        
        if not all_commands:
            return 0.0
            
        # Look for coordination patterns
        coordination_score = 0.5  # Default
        
        # Check for balanced service coverage
        services = [cmd.service for cmd in all_commands]
        service_balance = len(set(services)) / len(ServiceType)
        
        # Check for complementary resource usage
        resource_usage = self._analyze_resource_coordination(all_commands)
        
        coordination_score = (service_balance * 0.6 + resource_usage * 0.4)
        
        return min(1.0, coordination_score)
    
    def _analyze_resource_coordination(self, commands: List[Command]) -> float:
        """Analyze how well agents coordinated resource usage."""
        if not commands:
            return 0.0
            
        # Check for drone assignment conflicts
        drone_assignments = {}
        for cmd in commands:
            if cmd.action == "assign_drone":
                drone_id = cmd.parameters.get("drone_id")
                if drone_id:
                    if drone_id in drone_assignments:
                        return 0.2  # Conflict detected
                    drone_assignments[drone_id] = cmd
        
        # Check for zone management coordination
        zone_adjustments = {}
        for cmd in commands:
            if cmd.action == "adjust_zone":
                zone_id = cmd.parameters.get("zone_id")
                if zone_id:
                    if zone_id in zone_adjustments:
                        return 0.3  # Multiple adjustments to same zone
                    zone_adjustments[zone_id] = cmd
        
        return 0.8  # Good coordination if no conflicts
    
    def create_comprehensive_evaluation(self, agent_result, scenario: ScenarioDefinition,
                                      response_time: float) -> EvaluationMetrics:
        """Create comprehensive evaluation of agent performance."""
        
        # Convert agent result to commands for evaluation
        try:
            _, commands, _ = convert_and_evaluate_agent_commands(
                crew_result=agent_result,
                scenario_definition=scenario,
                scenario_type=ScenarioType.GRID_SURGE
            )
        except Exception:
            commands = []
        
        # Evaluate each dimension
        decision_quality = self.evaluate_decision_quality(commands, scenario)
        
        # Response time score (normalize to 0-1, lower time = higher score)
        response_time_score = max(0, 1 - (response_time / 10000))
        
        # Execution efficiency (based on command success rate and resource usage)
        execution_efficiency = self._evaluate_execution_efficiency(commands)
        
        # Adaptability score (placeholder - would be measured across scenarios)
        adaptability_score = 0.7  # Default for now
        
        # Coordination quality
        coordination_quality = self.evaluate_coordination_quality([agent_result])
        
        # User satisfaction (placeholder - would come from human feedback)
        user_satisfaction = 0.75  # Default for now
        
        metrics = EvaluationMetrics(
            decision_quality=decision_quality,
            response_time=response_time,
            execution_efficiency=execution_efficiency,
            adaptability_score=adaptability_score,
            coordination_quality=coordination_quality,
            user_satisfaction=user_satisfaction
        )
        
        # Store in history
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "scenario": scenario.name,
            "commands_count": len(commands)
        })
        
        return metrics
    
    def _evaluate_execution_efficiency(self, commands: List[Command]) -> float:
        """Evaluate execution efficiency of commands."""
        if not commands:
            return 0.0
            
        # Simulate command execution to measure efficiency
        executor = CommandExecutor()
        successful_commands = 0
        
        for cmd in commands:
            try:
                result = executor.execute(cmd)
                if result.success:
                    successful_commands += 1
            except Exception:
                pass
                
        success_rate = successful_commands / len(commands) if commands else 0
        
        # Factor in resource utilization efficiency
        resource_efficiency = self._evaluate_resource_efficiency(commands)
        
        return (success_rate * 0.7 + resource_efficiency * 0.3)


# Infrastructure setup for evaluation
console.print(Panel("📏 Phase 1: Evaluation Framework Setup", 
                   border_style="blue"))

# Display workshop progress
display_workshop_progress(1, 6, "Evaluation Framework Setup")

environment_ok = check_environment()
if environment_ok:
    services_running = start_services()
    status = '✅ All Running' if services_running else '❌ Some Failed'
    console.print(f"Services Status: {status}")
    
    if services_running:
        console.print("\n🔄 Initializing evaluation environment...")
        reset_all_service_states()
        console.print("✅ Evaluation environment ready")
else:
    console.print("[red]Please fix environment issues before continuing[/red]")

# Create evaluation framework
evaluation_framework = AdvancedEvaluationFramework()

console.print(Panel(
    "📊 **Advanced Evaluation Framework Initialized**\n\n"
    "**Evaluation Dimensions:**\n"
    "• Decision Quality: Analyzes decision patterns and logic\n"
    "• Response Time: Measures agent responsiveness\n"
    "• Execution Efficiency: Evaluates resource utilization\n"
    "• Adaptability Score: Measures flexibility across scenarios\n"
    "• Coordination Quality: Assesses multi-agent cooperation\n"
    "• User Satisfaction: Incorporates human feedback\n\n"
    "**Key Features:**\n"
    "✅ Multi-dimensional scoring\n"
    "✅ Pattern analysis for decision quality\n"
    "✅ Resource efficiency evaluation\n"
    "✅ Coordination conflict detection\n"
    "✅ Historical trend tracking",
    title="Evaluation Framework Ready",
    border_style="green"
))

# 👨‍💻 INTERACTIVE WORKSHOP CHECKPOINT
create_workshop_checkpoint(
    "Phase 1 Exploration",
    "Now it's your turn to experiment with the evaluation framework!\n\n"
    "Try these activities:\n"
    "1. Look at the EvaluationMetrics class above\n"
    "2. Consider what other metrics would be valuable for YOUR use case\n"
    "3. Think about how you would weight these different dimensions\n"
    "4. Modify the overall_score() function to use different weights\n\n"
    "💡 Challenge: Add a new metric dimension that's important for your domain!"
)

# Store Phase 1 results
evaluation_results["framework_setup"] = {
    "environment_ok": environment_ok,
    "services_running": services_running,
    "evaluation_dimensions": [
        "decision_quality",
        "response_time", 
        "execution_efficiency",
        "adaptability_score",
        "coordination_quality",
        "user_satisfaction"
    ],
    "framework_features": [
        "pattern_analysis",
        "resource_efficiency",
        "coordination_detection",
        "historical_tracking"
    ]
}

save_experiment_results(evaluation_results)
console.print("✅ Phase 1 Complete: Advanced evaluation framework established")

# 💬 DISCUSSION PROMPT
create_discussion_prompt(
    "Evaluation Framework Design",
    [
        "What metrics are most important for YOUR specific use case?",
        "How would you balance speed vs. quality in agent evaluation?",
        "What real-world constraints should influence our metrics?",
        "How could we make evaluation more aligned with business outcomes?"
    ]
)

# %% [markdown]
"""
## ⚡ **PHASE 2: Performance Baseline Measurement**

**🎓 Educational Goal**: Establish performance baselines for optimization 
targets

In this phase, we'll:
1. Measure current agent system performance across all dimensions
2. Identify performance bottlenecks and optimization opportunities
3. Create baseline metrics for comparison
4. Document performance characteristics

**🔑 Key Learning**: You can't optimize what you don't measure - establishing 
comprehensive baselines is essential for meaningful optimization
"""

# %%
# 👨‍💻 ADJUSTABLE PARAMETERS: Modify these to experiment!
EVALUATION_CONFIG = {
    "scenarios_to_test": ["heat_wave", "cyber_attack", "earthquake"],  # Modify this list!
    "evaluation_weights": {  # Adjust these weights based on your priorities!
        "decision_quality": 0.3,
        "response_time": 0.2, 
        "execution_efficiency": 0.2,
        "adaptability_score": 0.15,
        "coordination_quality": 0.1,
        "user_satisfaction": 0.05
    },
    "baseline_agents_config": {
        "verbose": True,  # Set to False for faster execution
        "model": "gpt-4o-mini",  # Try: "gpt-4o", "gpt-4o-mini", "claude-3-haiku"
        "temperature": 0.1  # Adjust for more/less creative responses
    }
}

# Run baseline measurement with diverse scenarios
console.print(Panel("⚡ Phase 2: Performance Baseline Measurement", 
                   border_style="blue"))

display_workshop_progress(2, 6, "Performance Baseline Measurement")

# Create diverse evaluation scenarios
evaluation_scenarios = create_evaluation_scenarios()

# 🛠️ HANDS-ON EXERCISE: Scenario Analysis
create_hands_on_exercise(
    "Scenario Complexity Analysis",
    "Before running the baseline, let's analyze our scenarios:\n\n"
    "1. Look at the evaluation_scenarios dictionary created above\n"
    "2. Pick one scenario and examine its structure\n"
    "3. Identify what makes it easy vs. difficult for agents\n"
    "4. Predict which scenario will be most challenging and why\n\n"
    "🎯 Write down your predictions - we'll check them against the results!",
    "You should understand what makes scenarios more/less challenging",
    "Easy"
)

# Test baseline performance with configurable scenarios
console.print("📊 Testing baseline performance across multiple scenarios...")

baseline_results = {}
for scenario_name in EVALUATION_CONFIG["scenarios_to_test"]:
    if scenario_name in evaluation_scenarios:
        scenario = evaluation_scenarios[scenario_name]
        console.print(f"\n🎯 Testing: {scenario.name}")
        
        # Create agents with configurable parameters
        grid_agent = create_optimized_grid_agent()
        emergency_agent = create_optimized_emergency_agent()
        traffic_agent = create_optimized_traffic_agent()
        
        # Configure agents based on workshop settings
        # if not EVALUATION_CONFIG["baseline_agents_config"]["verbose"]:
        #     grid_agent.verbose = False
        #     emergency_agent.verbose = False
        #     traffic_agent.verbose = False
            
        # Create crew for this scenario
        crew = Crew(
            agents=[grid_agent, emergency_agent, traffic_agent],
            tasks=create_optimized_agent_tasks(
                grid_agent, emergency_agent, traffic_agent, scenario
            ),
            process=Process.sequential,
            verbose=EVALUATION_CONFIG["baseline_agents_config"]["verbose"]
        )
        
        # Measure performance
        result, response_time = evaluation_framework.measure_response_time(
            crew.kickoff,
            inputs={
                "scenario_name": scenario.name,
                "scenario_description": scenario.description
            }
        )
        
        # Create evaluation with custom weights
        metrics = evaluation_framework.create_comprehensive_evaluation(
            result, scenario, response_time
        )
        
        baseline_results[scenario_name] = {
            "metrics": metrics,
            "scenario": scenario,
            "response_time": response_time
        }
        
        console.print(f"✅ {scenario_name}: {metrics.overall_score():.3f} score")

# Display baseline comparison
baseline_table = Table(title="📊 Baseline Performance Across Scenarios")
baseline_table.add_column("Scenario", style="cyan")
baseline_table.add_column("Overall Score", style="green")
baseline_table.add_column("Response Time", style="yellow")
baseline_table.add_column("Decision Quality", style="blue")

for scenario_name, data in baseline_results.items():
    metrics = data["metrics"]
    baseline_table.add_row(
        scenario_name.replace("_", " ").title(),
        f"{metrics.overall_score():.3f}",
        f"{metrics.response_time:.1f}ms",
        f"{metrics.decision_quality:.3f}"
    )

console.print(baseline_table)

# 💬 DISCUSSION PROMPT
create_discussion_prompt(
    "Baseline Analysis",
    [
        "Which scenario was most challenging and why?",
        "What patterns do you see in the performance data?",
        "How would you adjust the evaluation weights for your use case?",
        "What optimization opportunities do you identify?"
    ]
)

console.print("✅ Phase 2 Complete: Multi-scenario baseline established")

# %% [markdown]
"""
## 🔌 **PHASE 3: Model Context Protocol (MCP) Integration**

**🎓 Educational Goal**: Demonstrate how MCP enables dynamic tool discovery 
and adaptive agent capabilities

**👨‍💻 INTERACTIVE COMPONENT**: See how agents adapt when new tools become 
available at runtime!

In this phase, we'll:
1. Understand Model Context Protocol (MCP) fundamentals
2. Simulate dynamic service discovery and tool registration
3. Show how agents can adapt to new capabilities automatically
4. Build production-ready extensibility patterns

**🔑 Key Learning**: MCP transforms static agent systems into dynamic, 
extensible platforms that can discover and use new tools at runtime, making 
them more adaptable and future-proof.

**🌟 Why MCP Matters for Production**:
- **Dynamic Extensibility**: Add new tools without redeploying agents
- **Service Discovery**: Automatically find and integrate external services
- **Protocol Standardization**: Consistent interface for tool integration
- **Scalable Architecture**: Build systems that grow with your needs
"""

# %%
# 👨‍💻 ADJUSTABLE MCP PARAMETERS
MCP_CONFIG = {
    "discovery_simulation": {
        "external_services": [  # Services that could be discovered at runtime
            {
                "name": "weather_analytics",
                "description": "Real-time weather pattern analysis",
                "capabilities": ["temperature_prediction", "wind_analysis", "storm_tracking"],
                "priority": "high",
                "mcp_version": "1.0"
            },
            {
                "name": "social_intelligence", 
                "description": "Social media crisis sentiment monitoring",
                "capabilities": ["sentiment_analysis", "trend_detection", "crisis_escalation"],
                "priority": "medium",
                "mcp_version": "1.0"
            },
            {
                "name": "predictive_modeling",
                "description": "AI-powered infrastructure failure prediction", 
                "capabilities": ["failure_prediction", "maintenance_scheduling", "risk_assessment"],
                "priority": "high",
                "mcp_version": "1.0"
            },
            {
                "name": "satellite_imagery",
                "description": "Real-time satellite monitoring and analysis",
                "capabilities": ["damage_assessment", "crowd_monitoring", "infrastructure_status"],
                "priority": "medium",
                "mcp_version": "1.0"
            }
        ],
        "discovery_delay": 1.5,  # Simulate network discovery time
        "integration_success_rate": 0.85,  # Some services might fail
        "enable_runtime_discovery": True
    },
    "agent_adaptation": {
        "auto_integrate_tools": True,     # Should agents auto-adopt new tools?
        "adaptation_threshold": 0.7,     # How readily agents adopt new tools
        "learning_rate": 0.4,            # How quickly they improve with new tools
        "capability_weighting": {        # How much to value different capabilities
            "prediction": 0.3,
            "monitoring": 0.25,
            "analysis": 0.25,
            "assessment": 0.2
        }
    }
}

display_workshop_progress(3, 6, "MCP Integration")

console.print(Panel("🔌 Phase 3: Model Context Protocol (MCP) Integration", 
                   border_style="purple"))


class MCPServiceRegistry:
    """Model Context Protocol service registry and discovery system."""
    
    def __init__(self, config):
        self.config = config
        self.static_tools = {}
        self.discovered_services = {}
        self.integration_log = []
        self.capability_matrix = {}
        
    def register_core_tools(self):
        """Register existing workshop tools as MCP-compatible services."""
        console.print("📦 Registering core crisis management tools via MCP...")
        
        core_mcp_tools = [
            {
                "name": "grid_management",
                "capabilities": ["zone_adjustment", "priority_setting", "load_balancing"],
                "mcp_interface": "crisis_management.grid.v1"
            },
            {
                "name": "emergency_response", 
                "capabilities": ["drone_assignment", "incident_tracking", "resource_allocation"],
                "mcp_interface": "crisis_management.emergency.v1"
            },
            {
                "name": "traffic_coordination",
                "capabilities": ["flow_redirection", "route_blocking", "emergency_corridors"],
                "mcp_interface": "crisis_management.traffic.v1"
            }
        ]
        
        for tool in core_mcp_tools:
            self.static_tools[tool["name"]] = {
                "type": "core_mcp",
                "capabilities": tool["capabilities"],
                "reliability": 0.95,
                "interface": tool["mcp_interface"]
            }
        
        console.print(f"✅ Registered {len(core_mcp_tools)} core MCP tools")
    
    def simulate_runtime_discovery(self):
        """Simulate MCP service discovery at runtime."""
        if not self.config["discovery_simulation"]["enable_runtime_discovery"]:
            console.print("⏸️  Runtime discovery disabled")
            return 0
            
        console.print("🔍 MCP Discovery: Scanning for external services...")
        
        import random
        import time
        
        # Simulate network discovery delay - REMOVED to prevent stopping execution
        # time.sleep(self.config["discovery_simulation"]["discovery_delay"])
        
        discovered_count = 0
        success_rate = self.config["discovery_simulation"]["integration_success_rate"]
        
        for service in self.config["discovery_simulation"]["external_services"]:
            # Simulate integration success/failure
            if random.random() < success_rate:
                # Successful MCP integration
                self.discovered_services[service["name"]] = {
                    "description": service["description"],
                    "capabilities": service["capabilities"],
                    "priority": service["priority"],
                    "mcp_version": service["mcp_version"],
                    "integration_time": datetime.now().isoformat(),
                    "status": "active",
                    "tool_count": len(service["capabilities"])
                }
                
                # Update capability matrix
                for capability in service["capabilities"]:
                    if capability not in self.capability_matrix:
                        self.capability_matrix[capability] = []
                    self.capability_matrix[capability].append(service["name"])
                
                log_entry = f"✅ MCP Integration: {service['name']} ({len(service['capabilities'])} capabilities)"
                self.integration_log.append(log_entry)
                console.print(log_entry)
                discovered_count += 1
            else:
                log_entry = f"❌ MCP Integration Failed: {service['name']} (connection timeout)"
                self.integration_log.append(log_entry)
                console.print(log_entry)
        
        console.print(f"🎯 MCP Discovery Complete: {discovered_count}/{len(self.config['discovery_simulation']['external_services'])} services integrated")
        
        return discovered_count
    
    def create_mcp_enhanced_agent(self, base_agent_creator, agent_type="crisis_coordinator"):
        """Create agent with MCP-discovered capabilities."""
        
        # Calculate total available capabilities through MCP
        static_capabilities = sum(len(tool["capabilities"]) for tool in self.static_tools.values())
        dynamic_capabilities = sum(len(service["capabilities"]) for service in self.discovered_services.values())
        total_capabilities = static_capabilities + dynamic_capabilities
        
        # Create enhanced agent with MCP awareness
        base_agent = base_agent_creator()
        
        # Enhance with MCP-specific prompt engineering
        mcp_enhancement = f"""

==== MCP INTEGRATION STATUS ====
You now have access to {total_capabilities} capabilities through Model Context Protocol:

CORE TOOLS ({static_capabilities} capabilities):
{chr(10).join([f"• {name}: {', '.join(info['capabilities'])}" for name, info in self.static_tools.items()])}

RUNTIME-DISCOVERED SERVICES ({dynamic_capabilities} capabilities):
{chr(10).join([f"• {name}: {', '.join(info['capabilities'])}" for name, info in self.discovered_services.items()])}

MCP CAPABILITY MATRIX:
{chr(10).join([f"• {capability}: Available via {', '.join(services)}" for capability, services in self.capability_matrix.items()])}

==== MCP DECISION FRAMEWORK ====
When making decisions:
1. ASSESS available capabilities through MCP registry
2. PRIORITIZE tools based on reliability and relevance
3. COMBINE core tools with discovered services for optimal response
4. ADAPT strategy based on newly available capabilities
5. LOG which MCP services you utilize for transparency

==== DYNAMIC ADAPTATION RULES ====
• If predictive capabilities are available, use them for proactive planning
• If monitoring services are discovered, integrate real-time intelligence  
• If analysis tools become available, enhance decision quality
• Always prefer higher-reliability MCP services when available"""

        # Update agent backstory with MCP integration
        base_agent.backstory = base_agent.backstory + mcp_enhancement
        
        # Update agent goal to reflect MCP capabilities
        original_goal = base_agent.goal
        enhanced_goal = f"{original_goal}, leveraging {total_capabilities} MCP-integrated capabilities for optimal crisis response"
        base_agent.goal = enhanced_goal
        
        return base_agent
    
    def demonstrate_adaptive_behavior(self, scenario):
        """Demonstrate how MCP enables adaptive agent behavior."""
        console.print("🤖 Demonstrating MCP-enabled adaptive behavior...")
        
        # Before MCP Discovery
        console.print("\n📊 BEFORE MCP Discovery:")
        console.print(f"• Available Core Tools: {len(self.static_tools)}")
        console.print(f"• Total Capabilities: {sum(len(tool['capabilities']) for tool in self.static_tools.values())}")
        console.print("• Agent Adaptability: Static, predefined tools only")
        
        # Perform MCP discovery
        discovered_count = self.simulate_runtime_discovery()
        
        # After MCP Discovery
        console.print(f"\n🚀 AFTER MCP Discovery:")
        console.print(f"• Core Tools: {len(self.static_tools)}")
        console.print(f"• Discovered Services: {len(self.discovered_services)}")
        total_capabilities = (sum(len(tool['capabilities']) for tool in self.static_tools.values()) + 
                            sum(len(service['capabilities']) for service in self.discovered_services.values()))
        console.print(f"• Total Capabilities: {total_capabilities}")
        console.print(f"• Agent Adaptability: Dynamic, runtime-extensible")
        
        # Show capability enhancement
        if self.capability_matrix:
            console.print(f"\n🧠 Enhanced Decision-Making Capabilities:")
            for capability, services in list(self.capability_matrix.items())[:5]:
                console.print(f"• {capability.replace('_', ' ').title()}: {len(services)} service(s) available")
        
        # Create MCP-enhanced agent
        enhanced_agent = self.create_mcp_enhanced_agent(
            create_optimized_grid_agent, 
            agent_type="mcp_crisis_coordinator"
        )
        
        return enhanced_agent
    
    def get_mcp_integration_report(self):
        """Generate comprehensive MCP integration report."""
        return {
            "core_tools": len(self.static_tools),
            "discovered_services": len(self.discovered_services),
            "total_capabilities": (sum(len(tool['capabilities']) for tool in self.static_tools.values()) + 
                                 sum(len(service['capabilities']) for service in self.discovered_services.values())),
            "capability_matrix": self.capability_matrix,
            "integration_success_rate": (len(self.discovered_services) / 
                                       len(self.config["discovery_simulation"]["external_services"]) 
                                       if self.config["discovery_simulation"]["external_services"] else 0),
            "integration_log": self.integration_log,
            "mcp_enabled": True
        }


# Create and demonstrate MCP registry
mcp_registry = MCPServiceRegistry(MCP_CONFIG)

# Register core tools as MCP services
mcp_registry.register_core_tools()

# Demonstrate MCP-enabled adaptive behavior
test_scenario = create_heat_wave_scenario_for_evaluation()
mcp_enhanced_agent = mcp_registry.demonstrate_adaptive_behavior(test_scenario)

# Generate integration report
mcp_report = mcp_registry.get_mcp_integration_report()

console.print(Panel(
    f"🔌 **MCP Integration Report**\n\n"
    f"• Core MCP Tools: {mcp_report['core_tools']}\n"
    f"• Runtime Discovered: {mcp_report['discovered_services']}\n"
    f"• Total Capabilities: {mcp_report['total_capabilities']}\n"
    f"• Integration Success: {mcp_report['integration_success_rate']:.1%}\n"
    f"• Capability Categories: {len(mcp_report['capability_matrix'])}\n\n"
    f"**MCP Status**: {'🟢 Active' if mcp_report['mcp_enabled'] else '🔴 Inactive'}\n"
    f"**Recent Integrations**:\n" +
    "\n".join([f"  {log}" for log in mcp_report['integration_log'][-3:]]),
    title="Model Context Protocol Status", 
    border_style="purple"
))

# 🛠️ HANDS-ON EXERCISE: MCP Configuration
create_hands_on_exercise(
    "Customize MCP Discovery",
    "Experiment with Model Context Protocol:\n\n"
    "1. Modify MCP_CONFIG['discovery_simulation']['external_services']\n"
    "2. Add your own domain-specific services and capabilities\n"
    "3. Adjust integration success rates and discovery delays\n"
    "4. Configure agent adaptation parameters\n\n"
    "💡 Try adding services specific to your industry or use case!\n"
    "Examples: 'legal_compliance', 'financial_analysis', 'customer_sentiment'",
    "Understanding how MCP enables dynamic system extensibility",
    "Hard"
)

# Store MCP results for final evaluation
evaluation_results["mcp_integration"] = mcp_report

console.print("✅ Phase 3 Complete: MCP integration enables dynamic agent extensibility")

# %% [markdown]
"""
## 🎯 **PHASE 4: Human Feedback Integration**

**🎓 Educational Goal**: Build systems that learn from human input

**👨‍💻 INTERACTIVE COMPONENT**: Customize feedback criteria and see how 
agents adapt!

In this phase, we'll:
1. Design human feedback collection systems
2. Implement preference learning mechanisms  
3. Create adaptive improvement loops
4. **Integrate feedback into agent configurations for testing**

**🔑 Key Learning**: Production agents must incorporate human feedback 
for continuous improvement and alignment
"""

# %%
# 👨‍💻 ADJUSTABLE FEEDBACK PARAMETERS
FEEDBACK_CONFIG = {
    "feedback_criteria": {  # Modify these weights to match your priorities!
        "action_appropriateness": 0.3,    # How suitable are the actions?
        "safety_priority": 0.25,          # Does it prioritize safety?
        "resource_efficiency": 0.2,       # Efficient use of resources?
        "response_speed": 0.15,           # Fast enough response?
        "coordination_quality": 0.1       # Good team coordination?
    },
    "feedback_simulation": {
        "expert_bias": 0.8,        # How much do simulated experts favor safety?
        "speed_preference": 0.6,   # How much do users prefer speed?
        "variability": 0.2         # Randomness in feedback (0-1)
    },
    "improvement_thresholds": {
        "minimum_score": 0.7,      # Below this, major changes needed
        "target_score": 0.85,      # Target performance level
        "excellent_score": 0.9     # Exceptional performance
    },
    "agent_integration": {        # NEW: Control how feedback affects agents
        "enable_feedback_learning": True,   # Should agents adapt?
        "feedback_influence": 0.3,          # How much feedback changes behavior
        "adaptation_speed": 0.5             # How quickly agents adapt
    }
}

display_workshop_progress(4, 6, "Human Feedback Integration")

console.print(Panel("🔄 Phase 4: Human Feedback Integration", 
                   border_style="blue"))


class InteractiveHumanFeedbackSystem:
    """Interactive human feedback system with configurable criteria."""
    
    def __init__(self, evaluation_framework, config):
        self.evaluation_framework = evaluation_framework
        self.config = config
        self.feedback_history = []
        self.improvement_suggestions = []
        self.agent_adaptations = {}  # NEW: Track how agents adapt
    
    def simulate_expert_feedback(self, agent_actions, scenario):
        """Simulate expert feedback with configurable criteria."""
        console.print("👥 Simulating expert feedback...")
        
        feedback_scores = {}
        criteria = self.config["feedback_criteria"]
        
        # Analyze actions based on configurable criteria
        grid_actions = [cmd for cmd in agent_actions 
                       if hasattr(cmd, 'service') and cmd.service.value == 'grid']
        emergency_actions = [cmd for cmd in agent_actions 
                           if hasattr(cmd, 'service') and cmd.service.value == 'emergency']
        
        # Action appropriateness (configurable weight)
        if grid_actions and emergency_actions:
            feedback_scores["action_appropriateness"] = 0.9
        elif grid_actions or emergency_actions:
            feedback_scores["action_appropriateness"] = 0.7
        else:
            feedback_scores["action_appropriateness"] = 0.3
            
        # Safety priority (expert bias influences this)
        safety_actions = len([cmd for cmd in agent_actions 
                            if hasattr(cmd, 'action') and 
                            cmd.action in ["set_priority", "assign_drone"]])
        base_safety = min(0.9, 0.4 + (safety_actions * 0.15))
        expert_bias = self.config["feedback_simulation"]["expert_bias"]
        feedback_scores["safety_priority"] = base_safety * expert_bias + base_safety * (1 - expert_bias)
        
        # Resource efficiency
        total_actions = len(agent_actions)
        if total_actions > 8:
            feedback_scores["resource_efficiency"] = 0.9  # Comprehensive
        elif total_actions > 4:
            feedback_scores["resource_efficiency"] = 0.7  # Adequate
        else:
            feedback_scores["resource_efficiency"] = 0.4  # Insufficient
            
        # Response speed (based on user preference)
        speed_pref = self.config["feedback_simulation"]["speed_preference"]
        feedback_scores["response_speed"] = 0.8 * speed_pref + 0.6 * (1 - speed_pref)
        
        # Coordination quality
        service_types = len(set(getattr(cmd, 'service', 'unknown').value 
                               for cmd in agent_actions if hasattr(cmd, 'service')))
        feedback_scores["coordination_quality"] = min(0.9, service_types * 0.3)
        
        # Calculate weighted overall score
        overall_score = sum(score * criteria[criterion] 
                          for criterion, score in feedback_scores.items())
        
        # Add some variability
        variability = self.config["feedback_simulation"]["variability"]
        import random
        overall_score += random.uniform(-variability/2, variability/2)
        overall_score = max(0, min(1, overall_score))
        
        return {
            "scores": feedback_scores,
            "overall_score": overall_score,
            "detailed_feedback": self._generate_detailed_feedback(feedback_scores, overall_score),
            "adaptation_suggestions": self._generate_adaptation_suggestions(feedback_scores)  # NEW
        }
    
    def _generate_detailed_feedback(self, scores, overall_score):
        """Generate detailed feedback based on scores."""
        feedback = []
        thresholds = self.config["improvement_thresholds"]
        
        if overall_score >= thresholds["excellent_score"]:
            feedback.append("🌟 Excellent performance! This response demonstrates mastery.")
        elif overall_score >= thresholds["target_score"]:
            feedback.append("✅ Good performance with room for minor improvements.")
        elif overall_score >= thresholds["minimum_score"]:
            feedback.append("⚠️  Acceptable but needs improvement in key areas.")
        else:
            feedback.append("❌ Performance below standards. Major improvements needed.")
        
        # Specific feedback based on lowest scores
        lowest_score = min(scores.items(), key=lambda x: x[1])
        feedback.append(f"🎯 Focus area: {lowest_score[0].replace('_', ' ').title()}")
        
        return feedback
    
    def _generate_adaptation_suggestions(self, scores):
        """NEW: Generate suggestions for how agents should adapt."""
        suggestions = {}
        
        if scores["safety_priority"] < 0.7:
            suggestions["safety_priority"] = {
                "emphasis": "Emphasize safety actions more strongly",
                "specific_commands": [
                    "Always prioritize set_priority commands for critical infrastructure",
                    "Assign drones to highest urgency incidents first",
                    "Include safety justification in every action"
                ],
                "prompt_additions": [
                    "SAFETY FIRST: Every action must consider safety implications",
                    "When in doubt, choose the safer option",
                    "Emergency response takes priority over efficiency"
                ]
            }
        if scores["resource_efficiency"] < 0.6:
            suggestions["resource_efficiency"] = {
                "emphasis": "Optimize resource allocation strategy",
                "specific_commands": [
                    "Check resource availability before assignments",
                    "Batch similar operations for efficiency",
                    "Avoid over-allocating resources to single incidents"
                ],
                "prompt_additions": [
                    "EFFICIENCY FOCUS: Minimize resource waste",
                    "Consider cost-benefit of each action",
                    "Look for opportunities to consolidate operations"
                ]
            }
        if scores["coordination_quality"] < 0.5:
            suggestions["coordination_quality"] = {
                "emphasis": "Improve inter-agent communication",
                "specific_commands": [
                    "Include coordination_notes in all structured outputs",
                    "Reference other agents' actions in decision reasoning",
                    "Avoid conflicting resource assignments"
                ],
                "prompt_additions": [
                    "COORDINATION: Always consider other agents' actions",
                    "Communication is key to effective response",
                    "Avoid duplicate or conflicting operations"
                ]
            }
            
        return suggestions
    
    def create_feedback_enhanced_agent(self, base_agent_creator, agent_type="grid"):
        """NEW: Create agent with comprehensive feedback-driven enhancements."""
        if not self.config["agent_integration"]["enable_feedback_learning"]:
            return base_agent_creator()
        
        # Get recent feedback to influence agent behavior
        recent_feedback = self.feedback_history[-3:] if self.feedback_history else []
        
        # Analyze feedback patterns for specific improvements
        improvement_patterns = self._analyze_improvement_patterns(recent_feedback)
        
        # Create base agent
        base_agent = base_agent_creator()
        
        # Apply comprehensive feedback-driven modifications
        enhanced_agent = self._apply_feedback_enhancements(
            base_agent, improvement_patterns, agent_type
        )
        
        return enhanced_agent
    
    def _analyze_improvement_patterns(self, recent_feedback):
        """Analyze recent feedback to identify specific improvement patterns."""
        patterns = {
            "safety_commands": [],
            "efficiency_commands": [],
            "coordination_commands": [],
            "prompt_enhancements": [],
            "goal_modifications": []
        }
        
        for session in recent_feedback:
            suggestions = session["feedback"].get("adaptation_suggestions", {})
            
            for area, details in suggestions.items():
                if isinstance(details, dict):
                    patterns["safety_commands"].extend(
                        details.get("specific_commands", [])
                    )
                    patterns["prompt_enhancements"].extend(
                        details.get("prompt_additions", [])
                    )
        
        return patterns
    
    def _apply_feedback_enhancements(self, agent, patterns, agent_type):
        """Apply comprehensive feedback enhancements to agent."""
        
        # Enhance agent goal with specific feedback improvements
        original_goal = agent.goal
        goal_enhancements = []
        
        if patterns["safety_commands"]:
            goal_enhancements.append("prioritize safety in all decisions")
        if patterns["efficiency_commands"]:
            goal_enhancements.append("optimize resource utilization")
        if patterns["coordination_commands"]:
            goal_enhancements.append("coordinate closely with other agents")
        
        if goal_enhancements:
            enhanced_goal = f"{original_goal}, while ensuring you {', '.join(goal_enhancements)}"
            agent.goal = enhanced_goal
        
        # Enhance backstory with specific command patterns
        command_patterns = self._generate_command_patterns(patterns, agent_type)
        
        enhanced_backstory = f"""{agent.backstory}

FEEDBACK-DRIVEN COMMAND PATTERNS:
{command_patterns}

SPECIFIC IMPROVEMENT AREAS:
{chr(10).join([f"• {enhancement}" for enhancement in patterns["prompt_enhancements"][:3]])}

DECISION FRAMEWORK:
1. Always justify actions with safety, efficiency, or coordination benefits
2. Include specific resource considerations in reasoning
3. Reference coordination with other agents when relevant
4. Use structured outputs to ensure clear communication"""
        
        agent.backstory = enhanced_backstory
        
        # Track this comprehensive adaptation
        self.agent_adaptations[agent_type] = {
            "goal_enhanced": bool(goal_enhancements),
            "command_patterns_added": len(command_patterns.split('\n')),
            "prompt_enhancements": len(patterns["prompt_enhancements"]),
            "specific_improvements": patterns
        }
        
        return agent
    
    def _generate_command_patterns(self, patterns, agent_type):
        """Generate specific command patterns based on agent type and feedback."""
        
        if agent_type == "grid":
            return f"""
GRID MANAGEMENT COMMAND PRIORITIES:
• adjust_zone: Always check load thresholds before adjustments
• set_priority: Focus on critical infrastructure during high load
• Resource allocation: Balance load across zones efficiently
• Coordination: Share grid status with emergency and traffic agents

DECISION SEQUENCE:
1. Assess current grid load and stability
2. Identify critical zones needing immediate attention
3. Calculate optimal capacity adjustments
4. Set infrastructure priorities for emergency response
5. Communicate grid impacts to other agents"""

        elif agent_type == "emergency":
            return f"""
EMERGENCY RESPONSE COMMAND PRIORITIES:
• assign_drone: Match drone capabilities to incident requirements
• update_incident: Keep status current for coordination
• Resource allocation: Prioritize by urgency and impact
• Coordination: Maintain awareness of grid and traffic constraints

DECISION SEQUENCE:
1. Assess all active incidents by urgency and severity
2. Evaluate available drone resources and capabilities
3. Calculate optimal drone-to-incident assignments
4. Update incident statuses for team coordination
5. Consider grid stability and traffic impacts"""

        elif agent_type == "traffic":
            return f"""
TRAFFIC MANAGEMENT COMMAND PRIORITIES:
• redirect_traffic: Clear emergency corridors first
• block_routes: Minimize civilian impact while enabling response
• Resource allocation: Balance flow with emergency access needs
• Coordination: Support grid stability and emergency operations

DECISION SEQUENCE:
1. Identify traffic congestion affecting emergency response
2. Calculate redirection impacts on overall flow
3. Create emergency corridors for critical services
4. Block problematic routes strategically
5. Coordinate with grid and emergency teams"""
        
        return "Standard command patterns apply"

    def collect_feedback_session(self, scenario, agent_result):
        """Collect comprehensive feedback session."""
        console.print("👥 Conducting feedback session...")
        
        # Convert agent result to commands for evaluation
        try:
            _, commands, _ = convert_and_evaluate_agent_commands(
                crew_result=agent_result,
                scenario_definition=scenario,
                scenario_type=ScenarioType.GRID_SURGE
            )
        except Exception:
            commands = []
        
        # Get expert feedback
        feedback = self.simulate_expert_feedback(commands, scenario)
        
        # Store feedback
        session = {
            "scenario": scenario.name,
            "commands_count": len(commands),
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_history.append(session)
        
        return feedback
    
    def analyze_feedback_patterns(self):
        """Analyze feedback patterns to identify improvement areas."""
        if not self.feedback_history:
            return {}
        
        # Calculate average scores across all criteria
        all_scores = {}
        criteria = self.config["feedback_criteria"].keys()
        
        for criterion in criteria:
            scores = [session["feedback"]["scores"][criterion] 
                     for session in self.feedback_history]
            all_scores[criterion] = sum(scores) / len(scores)
        
        # Identify improvement areas
        thresholds = self.config["improvement_thresholds"]
        improvement_areas = [
            criterion for criterion, score in all_scores.items()
            if score < thresholds["target_score"]
        ]
        
        return {
            "average_scores": all_scores,
            "improvement_areas": improvement_areas,
            "sessions_analyzed": len(self.feedback_history),
            "agent_adaptations": self.agent_adaptations,  # NEW
            "overall_trend": "improving" if len(self.feedback_history) > 1 and 
                           self.feedback_history[-1]["feedback"]["overall_score"] > 
                           self.feedback_history[0]["feedback"]["overall_score"] else "stable"
        }


# Create feedback system
feedback_system = InteractiveHumanFeedbackSystem(evaluation_framework, FEEDBACK_CONFIG)

# 🛠️ HANDS-ON EXERCISE: Create Your Own Feedback-Enhanced Agents
create_hands_on_exercise(
    "Build Feedback-Enhanced Agents",
    "Now you can create agents that learn from feedback!\n\n"
    "STEP 1: Modify FEEDBACK_CONFIG above to set your priorities\n"
    "STEP 2: Try this code to create feedback-enhanced agents:\n\n"
    "```python\n"
    "# Create agents that adapt based on feedback\n"
    "enhanced_grid_agent = feedback_system.create_feedback_enhanced_agent(\n"
    "    create_optimized_grid_agent, 'grid'\n"
    ")\n"
    "enhanced_emergency_agent = feedback_system.create_feedback_enhanced_agent(\n"
    "    create_optimized_emergency_agent, 'emergency'\n"
    ")\n"
    "\n"
    "# Use these agents in your crew for optimization competition!\n"
    "```\n\n"
    "💡 The agents will adapt their behavior based on previous feedback!",
    "Understanding how to integrate human feedback into agent behavior",
    "Medium"
)

# Demonstrate feedback integration
console.print("🤖 Creating feedback-enhanced agents...")

# Create enhanced agents that learn from feedback
enhanced_grid_agent = feedback_system.create_feedback_enhanced_agent(
    create_optimized_grid_agent, 'grid'
)
enhanced_emergency_agent = feedback_system.create_feedback_enhanced_agent(
    create_optimized_emergency_agent, 'emergency'
)
enhanced_traffic_agent = feedback_system.create_feedback_enhanced_agent(
    create_optimized_traffic_agent, 'traffic'
)

console.print("✅ Feedback-enhanced agents created!")

# Run feedback collection on baseline performance
console.print("📋 Testing feedback-enhanced agents...")

test_scenario = create_heat_wave_scenario_for_evaluation()

# Create crew with feedback-enhanced agents
feedback_enhanced_crew = Crew(
    agents=[enhanced_grid_agent, enhanced_emergency_agent, enhanced_traffic_agent],
    tasks=create_optimized_agent_tasks(enhanced_grid_agent, enhanced_emergency_agent, enhanced_traffic_agent, test_scenario),
    process=Process.sequential,
    verbose=True
)

# Execute and collect feedback
agent_result = feedback_enhanced_crew.kickoff(inputs={
    "scenario_name": test_scenario.name,
    "scenario_description": test_scenario.description
})

feedback_result = feedback_system.collect_feedback_session(test_scenario, agent_result)

# Display feedback results
console.print(Panel(
    f"👥 **Human Feedback Results**\n\n"
    f"**Overall Score**: {feedback_result['overall_score']:.3f}\n\n"
    f"**Detailed Scores**:\n" +
    "\n".join([f"• {criterion.replace('_', ' ').title()}: {score:.3f}"
               for criterion, score in feedback_result['scores'].items()]) + "\n\n"
    f"**Feedback**: {feedback_result['detailed_feedback'][0]}\n\n"
    f"**Agent Adaptations Applied**: {list(feedback_system.agent_adaptations.keys())}",
    title="Feedback-Enhanced Agent Results",
    border_style="green"
))

console.print("✅ Phase 4 Complete: Human feedback integration implemented")

# %% [markdown]
"""
## 🎯 **PHASE 5: Production Integration Patterns**

**🎓 Educational Goal**: Implement scalable and adaptable production patterns

**👨‍💻 HANDS-ON TIME**: Experiment with different integration strategies!

In this phase, we'll:
1. Design scalable production patterns
2. Implement modular architecture
3. Create flexible integration points
4. Test different deployment strategies

**🔑 Key Learning**: Production systems require a balance between 
scalability and adaptability - we'll explore different patterns to achieve this.
"""

# %%
# 👨‍💻 ADJUSTABLE PRODUCTION PARAMETERS
PRODUCTION_CONFIG = {
    "modular_architecture": {
        "enable_modular_architecture": True,
        "module_count": 3,
        "module_types": ["grid_management", "emergency_response", "traffic_coordination"]
    },
    "flexible_integration": {
        "enable_flexible_integration": True,
        "integration_points": 5,
        "integration_types": ["API", "Database", "Cloud Service", "Custom Tool", "Legacy System"]
    },
    "deployment_strategies": {
        "cloud_deployment": True,
        "on_premises_deployment": True,
        "hybrid_deployment": True,
        "containerization": True,
        "serverless_architecture": True
    },
    "monitoring_tools": {
        "prometheus": True,
        "grafana": True,
        "elk_stack": True,
        "splunk": True,
        "datadog": True
    }
}

display_workshop_progress(5, 6, "Production Integration Patterns")

console.print(Panel("🎯 Phase 5: Production Integration Patterns", 
                   border_style="blue"))


class ProductionIntegrationSystem:
    """Production integration system with configurable modular architecture."""
    
    def __init__(self, config):
        self.config = config
        self.modules = {}
        self.integration_points = {}
        self.deployment_strategies = {}
        self.monitoring_tools = {}
        
    def configure_modules(self):
        """Configure modules based on modular architecture configuration."""
        for module_type in self.config["modular_architecture"]["module_types"]:
            self.modules[module_type] = self._create_module(module_type)
        
    def _create_module(self, module_type):
        """Create a module based on its type."""
        if module_type == "grid_management":
            return GridManagementModule()
        elif module_type == "emergency_response":
            return EmergencyResponseModule()
        elif module_type == "traffic_coordination":
            return TrafficCoordinationModule()
        else:
            raise ValueError(f"Unknown module type: {module_type}")
    
    def configure_integration_points(self):
        """Configure integration points based on flexible integration configuration."""
        for point_type in self.config["flexible_integration"]["integration_types"]:
            self.integration_points[point_type] = self._create_integration_point(point_type)
    
    def _create_integration_point(self, point_type):
        """Create an integration point based on its type."""
        if point_type == "API":
            return APIIntegrationPoint()
        elif point_type == "Database":
            return DatabaseIntegrationPoint()
        elif point_type == "Cloud Service":
            return CloudServiceIntegrationPoint()
        elif point_type == "Custom Tool":
            return CustomToolIntegrationPoint()
        elif point_type == "Legacy System":
            return LegacySystemIntegrationPoint()
        else:
            raise ValueError(f"Unknown integration point type: {point_type}")
    
    def configure_deployment_strategies(self):
        """Configure deployment strategies based on deployment strategies configuration."""
        for strategy in self.config["deployment_strategies"]:
            self.deployment_strategies[strategy] = self._create_deployment_strategy(strategy)
    
    def _create_deployment_strategy(self, strategy):
        """Create a deployment strategy based on its type."""
        if strategy == "cloud_deployment":
            return CloudDeploymentStrategy()
        elif strategy == "on_premises_deployment":
            return OnPremisesDeploymentStrategy()
        elif strategy == "hybrid_deployment":
            return HybridDeploymentStrategy()
        elif strategy == "containerization":
            return ContainerizationDeploymentStrategy()
        elif strategy == "serverless_architecture":
            return ServerlessArchitectureDeploymentStrategy()
        else:
            raise ValueError(f"Unknown deployment strategy: {strategy}")
    
    def configure_monitoring_tools(self):
        """Configure monitoring tools based on monitoring tools configuration."""
        for tool in self.config["monitoring_tools"]:
            self.monitoring_tools[tool] = self._create_monitoring_tool(tool)
    
    def _create_monitoring_tool(self, tool):
        """Create a monitoring tool based on its type."""
        if tool == "prometheus":
            return PrometheusMonitoringTool()
        elif tool == "grafana":
            return GrafanaMonitoringTool()
        elif tool == "elk_stack":
            return ELKStackMonitoringTool()
        elif tool == "splunk":
            return SplunkMonitoringTool()
        elif tool == "datadog":
            return DatadogMonitoringTool()
        else:
            raise ValueError(f"Unknown monitoring tool: {tool}")
    
    def deploy(self):
        """Deploy the production system using configured strategies and tools."""
        self.configure_modules()
        self.configure_integration_points()
        self.configure_deployment_strategies()
        self.configure_monitoring_tools()
        
        # Deploy modules
        for module in self.modules.values():
            module.deploy()
        
        # Configure integration points
        for point, integration_point in self.integration_points.items():
            integration_point.configure()
        
        # Deploy deployment strategies
        for strategy in self.deployment_strategies.values():
            strategy.deploy()
        
        # Configure monitoring tools
        for tool in self.monitoring_tools.values():
            tool.configure()
        
        console.print("🚀 Production system deployed successfully!")
    
    def monitor(self):
        """Monitor the production system using configured monitoring tools."""
        for tool in self.monitoring_tools.values():
            tool.monitor()
        
        console.print("🔍 Production system monitoring in progress...")


# Initialize production integration system
production_system = ProductionIntegrationSystem(PRODUCTION_CONFIG)

# 🛠️ HANDS-ON EXERCISE: Configure Your Own Production System
create_hands_on_exercise(
    "Design Your Own Production System",
    "Now it's time to design your own production system!\n\n"
    "STEP 1: Modify PRODUCTION_CONFIG above to set your priorities\n"
    "STEP 2: Try this code to configure your production system:\n\n"
    "```python\n"
    "# Configure your production system\n"
    "production_system.configure_modules()\n"
    "production_system.configure_integration_points()\n"
    "production_system.configure_deployment_strategies()\n"
    "production_system.configure_monitoring_tools()\n"
    "\n"
    "# Deploy your production system\n"
    "production_system.deploy()\n"
    "\n"
    "# Monitor your production system\n"
    "production_system.monitor()\n"
    "```\n\n"
    "💡 The production system will be modular, scalable, and adaptable!",
    "Understanding how to design a scalable and adaptable production system",
    "Hard"
)

# 🎯 **FINAL EVALUATION & LEADERBOARD SYSTEM**

# %% [markdown]
"""
## 🏆 **FINAL COMPETITION SYSTEM: Understanding the Evaluation Framework**

**🎓 Educational Deep Dive**: How Multi-Dimensional Agent Evaluation Works

The final competition demonstrates a real-world approach to evaluating and 
optimizing agentic AI systems. This isn't just about scoring - it's about 
understanding what makes agents effective in production environments.

### **🔍 Competition Architecture**

The `ComprehensiveWorkshopEvaluator` implements a sophisticated evaluation 
framework that mirrors how production AI systems are assessed:

**1. Multi-Dimensional Scoring (6 Key Areas)**
- **Baseline Performance (20%)**: How well do your agents perform out-of-the-box?
- **Optimization Improvement (30%)**: What gains did your optimization techniques achieve?
- **Feedback Integration (25%)**: How effectively do agents learn from human input?
- **Innovation Factor (15%)**: Creative approaches and novel combinations
- **Consistency Score (10%)**: Reliable performance across different scenarios
- **Production Readiness (10%)**: Scalability and real-world applicability

**2. Real-World Evaluation Patterns**
- **Scenario-Based Testing**: Multiple crisis scenarios (heat wave, cyber attack, earthquake)
- **Resource Constraint Handling**: Limited drones, grid capacity, traffic sectors
- **Cross-Agent Coordination**: How well agents work together
- **Performance Under Pressure**: Response time vs. decision quality trade-offs

**3. Leaderboard Dynamics**
- **Weighted Scoring**: Different dimensions have different importance
- **Innovation Bonuses**: Rewards for creative optimization approaches
- **Consistency Penalties**: Unreliable performance is penalized
- **Comparative Ranking**: See how your techniques stack up against others

### **🎯 Why This Matters for Production AI**

This competition structure reflects real-world AI deployment challenges:

- **Multi-Stakeholder Evaluation**: Different groups care about different metrics
- **Continuous Improvement**: Systems must adapt and improve over time
- **Resource Optimization**: Balancing performance with computational costs
- **Reliability Requirements**: Consistency often matters more than peak performance
- **Innovation vs. Stability**: Balancing new approaches with proven techniques

### **📊 Reading the Competition Results**

When you see the leaderboard, you're seeing:
- **Final Score**: Weighted combination of all evaluation dimensions
- **Top Strength**: The area where each participant excelled most
- **Strategies**: The optimization techniques that were most effective
- **Comparative Analysis**: How different approaches perform against each other
"""

# %%
# 🎓 EDUCATIONAL NOTE: Competition Evaluation Framework
# 
# This class demonstrates how to build comprehensive evaluation systems for
# agentic AI applications. In production, you'd want similar multi-dimensional
# evaluation to ensure your agents perform well across all important criteria.

class ComprehensiveWorkshopEvaluator:
    """
    🏆 Final evaluation system for all workshop optimization techniques.
    
    🎓 LEARNING OBJECTIVE: Understand how to create comprehensive evaluation
    frameworks that assess agent performance across multiple dimensions,
    similar to how production AI systems are evaluated.
    
    This class implements:
    - Multi-dimensional scoring (6 evaluation criteria)
    - Weighted importance across different performance aspects
    - Innovation recognition and consistency measurement
    - Comparative analysis across different optimization approaches
    """
    
    def __init__(self):
        self.leaderboard = []
        self.evaluation_history = []
        self.evaluation_stats = {
            "total_evaluations": 0,
            "average_scores": {},
            "performance_trends": {}
        }
        
    def run_comprehensive_evaluation(self, participant_name, 
                                   optimization_config, 
                                   participant_agents=None):
        """
        🎯 Run comprehensive evaluation of all workshop techniques.
        
        🎓 EDUCATIONAL NOTE: This method demonstrates how to evaluate
        agent systems across multiple dimensions:
        
        1. **Baseline Measurement**: Establishes performance starting point
        2. **Optimization Assessment**: Measures improvement from techniques
        3. **Feedback Integration**: Evaluates learning and adaptation
        4. **Innovation Scoring**: Rewards creative approaches
        5. **Consistency Measurement**: Ensures reliable performance
        6. **Production Readiness**: Assesses real-world applicability
        
        This approach mirrors how production AI systems are evaluated
        before deployment and during continuous improvement cycles.
        """
        
        console.print(f"🏆 Running comprehensive evaluation for: {participant_name}")
        console.print("📊 This evaluation measures 6 key dimensions of agent performance...")
        
        # Phase 1: Baseline measurement
        console.print("\n📈 Phase 1: Measuring baseline performance...")
        baseline_results = self._measure_baseline_performance(participant_agents)
        console.print(f"   Baseline Score: {baseline_results['overall_score']:.3f}")
        
        # Phase 2: Optimization techniques evaluation
        console.print("\n⚡ Phase 2: Evaluating optimization techniques...")
        optimization_results = self._evaluate_optimization_techniques(optimization_config)
        console.print(f"   Optimization Score: {optimization_results:.3f}")
        
        # Phase 3: Feedback integration assessment
        console.print("\n🔄 Phase 3: Assessing feedback integration...")
        feedback_results = self._evaluate_feedback_integration()
        console.print(f"   Feedback Integration Score: {feedback_results:.3f}")
        
        # Phase 4: Innovation scoring
        console.print("\n💡 Phase 4: Calculating innovation factor...")
        innovation_score = self._calculate_innovation_score(optimization_config)
        console.print(f"   Innovation Score: {innovation_score['score']:.3f}")
        
        # Phase 5: Consistency across scenarios
        console.print("\n📊 Phase 5: Measuring consistency...")
        consistency_score = self._measure_consistency()
        console.print(f"   Consistency Score: {consistency_score:.3f}")
        
        # Phase 6: Production system evaluation
        console.print("\n🏭 Phase 6: Evaluating production readiness...")
        production_score = self._evaluate_production_system()
        console.print(f"   Production Score: {production_score:.3f}")
        
        # Calculate final score
        final_score = self._calculate_final_score(
            baseline_results, optimization_results, feedback_results,
            innovation_score, consistency_score, production_score
        )
        
        console.print(f"\n🎯 Final Score: {final_score:.3f}/1.000")
        
        # Create evaluation entry
        evaluation_entry = {
            "participant": participant_name,
            "final_score": final_score,
            "breakdown": {
                "baseline_performance": baseline_results,
                "optimization_improvement": optimization_results,
                "feedback_integration": feedback_results,
                "innovation_factor": innovation_score,
                "consistency_score": consistency_score,
                "production_system": production_score
            },
            "optimization_strategies": list(optimization_config.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to leaderboard
        self.leaderboard.append(evaluation_entry)
        self.leaderboard.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Update statistics
        self._update_evaluation_stats(evaluation_entry)
        
        return evaluation_entry
    
    def _measure_baseline_performance(self, participant_agents):
        """
        📊 Measure baseline performance using standard agents or participant's.
        
        🎓 EDUCATIONAL NOTE: Baseline measurement is crucial for understanding
        the starting point of your optimization efforts. This method:
        
        1. Uses either standard agents or participant's custom agents
        2. Tests on a representative scenario (heat wave crisis)
        3. Measures multiple performance dimensions
        4. Provides objective comparison point for improvements
        
        In production, you'd establish baselines before deploying optimizations.
        """
        
        if participant_agents:
            # Use participant's custom agents
            test_agents = participant_agents
            console.print("   Using custom participant agents for baseline...")
        else:
            # Use standard baseline agents
            test_agents = [
                create_optimized_grid_agent(),
                create_optimized_emergency_agent(),
                create_optimized_traffic_agent()
            ]
            console.print("   Using standard workshop agents for baseline...")
        
        # Test on heat wave scenario
        test_scenario = create_heat_wave_scenario_for_evaluation()
        crew = Crew(
            agents=test_agents,
            tasks=create_optimized_agent_tasks(*test_agents, test_scenario),
            process=Process.sequential,
            verbose=False  # Keep quiet for evaluation
        )
        
        # Measure performance
        result, response_time = evaluation_framework.measure_response_time(
            crew.kickoff,
            inputs={
                "scenario_name": test_scenario.name,
                "scenario_description": test_scenario.description
            }
        )
        
        metrics = evaluation_framework.create_comprehensive_evaluation(
            result, test_scenario, response_time
        )
        
        return {
            "overall_score": metrics.overall_score(),
            "response_time": response_time,
            "decision_quality": metrics.decision_quality,
            "execution_efficiency": metrics.execution_efficiency
        }
    
    def _evaluate_optimization_techniques(self, optimization_config):
        """
        ⚡ Evaluate the effectiveness of optimization techniques used.
        
        🎓 EDUCATIONAL NOTE: This method demonstrates how to quantify
        the impact of different optimization strategies:
        
        - **Latency Optimization**: How much faster did agents become?
        - **Caching Strategies**: Did caching improve response times?
        - **Model Optimization**: Did smart model selection help?
        - **Parallel Processing**: Did parallelization provide benefits?
        
        The scoring reflects real-world optimization value - improvements
        that matter for production deployment.
        """
        
        optimization_score = 0.0
        techniques_used = 0
        
        # Check for latency optimization
        if "latency_optimization" in optimization_config:
            latency_improvement = optimization_config["latency_optimization"].get("improvement", 0)
            # Normalize to 0-1 scale (50% improvement = 1.0 score)
            optimization_score += min(1.0, latency_improvement / 50.0)
            techniques_used += 1
            console.print(f"   ⚡ Latency improvement: {latency_improvement}%")
        
        # Check for caching strategies
        if "caching_enabled" in optimization_config:
            optimization_score += 0.2  # Bonus for implementing caching
            techniques_used += 1
            console.print("   💾 Caching strategy implemented")
        
        # Check for model optimization
        if "model_optimization" in optimization_config:
            optimization_score += 0.15  # Bonus for model selection
            techniques_used += 1
            console.print("   🧠 Model optimization implemented")
        
        # Check for parallel processing
        if "parallel_processing" in optimization_config:
            optimization_score += 0.25  # Bonus for parallelization
            techniques_used += 1
            console.print("   🔄 Parallel processing implemented")
        
        # Normalize by number of techniques
        if techniques_used > 0:
            optimization_score = optimization_score / techniques_used
        
        return min(1.0, optimization_score)
    
    def _evaluate_feedback_integration(self):
        """
        🔄 Evaluate how well feedback integration was implemented.
        
        🎓 EDUCATIONAL NOTE: This evaluates the sophistication of
        human feedback integration:
        
        - **Adaptation Depth**: How comprehensively do agents adapt?
        - **Behavioral Changes**: Are agents actually changing behavior?
        - **Learning Patterns**: Do agents improve over time?
        
        In production, feedback integration is crucial for continuous
        improvement and alignment with human values.
        """
        
        if not hasattr(feedback_system, 'agent_adaptations'):
            console.print("   ⚠️  No feedback integration detected")
            return 0.5  # Default score if no feedback integration
        
        adaptations = feedback_system.agent_adaptations
        
        if not adaptations:
            console.print("   ❌ No agent adaptations found")
            return 0.3  # Low score for no adaptations
        
        # Score based on adaptation comprehensiveness
        adaptation_score = 0.0
        
        for agent_type, adaptation_data in adaptations.items():
            if adaptation_data.get("goal_enhanced", False):
                adaptation_score += 0.3
            if adaptation_data.get("command_patterns_added", 0) > 0:
                adaptation_score += 0.4
            if adaptation_data.get("prompt_enhancements", 0) > 0:
                adaptation_score += 0.3
        
        console.print(f"   ✅ {len(adaptations)} agents adapted with feedback")
        
        # Average across all adapted agents
        return min(1.0, adaptation_score / len(adaptations))
    
    def _calculate_innovation_score(self, optimization_config):
        """
        💡 Calculate innovation score based on creative approaches.
        
        🎓 EDUCATIONAL NOTE: Innovation scoring rewards participants for:
        
        - **Creative Combinations**: Using multiple techniques together
        - **Custom Approaches**: Implementing novel optimization strategies
        - **Advanced Techniques**: Going beyond basic optimization
        - **Experimental Methods**: Trying new evaluation approaches
        
        This encourages exploration and creative problem-solving,
        which is essential for advancing the field.
        """
        
        innovation_factors = []
        
        # Check for creative combinations
        if len(optimization_config) > 3:
            innovation_factors.append("Multiple optimization techniques combined")
            console.print("   🎨 Creative technique combination detected")
        
        # Check for custom configurations
        if any("custom" in str(key).lower() for key in optimization_config.keys()):
            innovation_factors.append("Custom optimization approach")
            console.print("   🔧 Custom optimization approach detected")
        
        # Check for advanced feedback usage
        if "advanced_feedback" in optimization_config:
            innovation_factors.append("Advanced feedback integration")
            console.print("   🧠 Advanced feedback integration detected")
        
        # Check for novel evaluation metrics
        if "custom_metrics" in optimization_config:
            innovation_factors.append("Custom evaluation metrics")
            console.print("   📊 Custom evaluation metrics detected")
        
        # Score based on innovation factors
        base_score = min(1.0, len(innovation_factors) * 0.25)
        
        console.print(f"   💡 Innovation factors: {len(innovation_factors)}")
        
        return {
            "score": base_score,
            "factors": innovation_factors
        }
    
    def _measure_consistency(self):
        """
        📊 Measure consistency across different scenarios.
        
        🎓 EDUCATIONAL NOTE: Consistency is often more important than
        peak performance in production systems. This method measures:
        
        - **Performance Variance**: How much do scores vary?
        - **Reliability**: Can you count on consistent results?
        - **Stability**: Do agents perform predictably?
        
        High consistency means your optimization is robust and reliable.
        """
        
        # This would ideally test multiple scenarios
        # For workshop purposes, return based on available data
        if hasattr(evaluation_framework, 'evaluation_history'):
            history = evaluation_framework.evaluation_history
            if len(history) > 1:
                scores = [eval_data["metrics"].overall_score() for eval_data in history]
                # Calculate coefficient of variation (lower is more consistent)
                if scores:
                    mean_score = sum(scores) / len(scores)
                    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
                    cv = (variance ** 0.5) / mean_score if mean_score > 0 else 1
                    consistency_score = max(0, 1 - cv)  # Convert to consistency score
                    console.print(f"   📊 Measured across {len(scores)} evaluations")
                    return consistency_score
        
        console.print("   📊 Using default consistency score")
        return 0.7  # Default consistency score
    
    def _evaluate_production_system(self):
        """
        🏭 Evaluate the production system based on modular architecture.
        
        🎓 EDUCATIONAL NOTE: Production readiness assessment considers:
        
        - **Scalability**: Can the system handle increased load?
        - **Modularity**: Is the architecture maintainable?
        - **Monitoring**: Are there observability capabilities?
        - **Deployment**: Is the system deployment-ready?
        
        This reflects real-world deployment considerations.
        """
        
        # Check if production system was configured
        if 'production_system' in locals() or 'PRODUCTION_CONFIG' in globals():
            console.print("   🏭 Production system architecture detected")
            return 0.8  # Good score for production consideration
        else:
            console.print("   ⚠️  Basic production readiness assumed")
            return 0.6  # Default score
    
    def _calculate_final_score(self, baseline, optimization, feedback, 
                             innovation, consistency, production):
        """
        🎯 Calculate weighted final score.
        
        🎓 EDUCATIONAL NOTE: The weighting reflects real-world priorities:
        
        - **Baseline (20%)**: Foundation performance matters
        - **Optimization (30%)**: Improvements are highly valued
        - **Feedback (25%)**: Adaptability is crucial for production
        - **Innovation (15%)**: Creativity drives the field forward
        - **Consistency (10%)**: Reliability is essential
        - **Production (10%)**: Real-world deployment readiness
        
        These weights can be adjusted based on specific use cases.
        """
        
        weights = {
            "baseline": 0.20,
            "optimization": 0.30,
            "feedback": 0.25,
            "innovation": 0.15,
            "consistency": 0.10,
            "production": 0.10
        }
        
        # Handle innovation score structure
        innovation_score = innovation["score"] if isinstance(innovation, dict) else innovation
        
        final_score = (
            weights["baseline"] * baseline["overall_score"] +
            weights["optimization"] * optimization +
            weights["feedback"] * feedback +
            weights["innovation"] * innovation_score +
            weights["consistency"] * consistency +
            weights["production"] * production
        )
        
        return min(1.0, final_score)
    
    def _update_evaluation_stats(self, evaluation_entry):
        """
        📈 Update evaluation statistics for tracking.
        
        🎓 EDUCATIONAL NOTE: Tracking evaluation statistics helps identify:
        - Performance trends across participants
        - Most effective optimization strategies
        - Areas where the workshop could be improved
        
        This data would inform future workshop iterations.
        """
        
        self.evaluation_stats["total_evaluations"] += 1
        
        # Track average scores
        breakdown = evaluation_entry["breakdown"]
        for metric, value in breakdown.items():
            if metric not in self.evaluation_stats["average_scores"]:
                self.evaluation_stats["average_scores"][metric] = []
            
            # Handle different value types
            if isinstance(value, dict) and "overall_score" in value:
                self.evaluation_stats["average_scores"][metric].append(value["overall_score"])
            elif isinstance(value, dict) and "score" in value:
                self.evaluation_stats["average_scores"][metric].append(value["score"])
            elif isinstance(value, (int, float)):
                self.evaluation_stats["average_scores"][metric].append(value)
    
    def display_leaderboard(self):
        """
        🏆 Display the final competition leaderboard.
        
        🎓 EDUCATIONAL NOTE: The leaderboard shows:
        - **Rank**: Comparative performance ordering
        - **Final Score**: Weighted combination of all dimensions
        - **Top Strength**: Area of highest performance
        - **Strategies**: Optimization techniques used
        
        This helps participants understand what approaches work best
        and learn from successful strategies.
        """
        
        if not self.leaderboard:
            console.print("🏆 No entries yet - be the first to submit!")
            return
        
        leaderboard_table = Table(title="🏆 WORKSHOP OPTIMIZATION LEADERBOARD")
        leaderboard_table.add_column("Rank", style="yellow")
        leaderboard_table.add_column("Participant", style="cyan")
        leaderboard_table.add_column("Final Score", style="green")
        leaderboard_table.add_column("Top Strength", style="blue")
        leaderboard_table.add_column("Strategies", style="yellow")
        
        for i, entry in enumerate(self.leaderboard[:10], 1):
            # Determine rank emoji
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            
            # Find top performing area
            breakdown = entry["breakdown"]
            top_area = max(breakdown.items(), 
                          key=lambda x: x[1]["score"] if isinstance(x[1], dict) and "score" in x[1] 
                               else x[1] if isinstance(x[1], (int, float)) 
                               else 0)
            
            # Format strategies
            strategies = ", ".join(entry["optimization_strategies"][:2])
            if len(entry["optimization_strategies"]) > 2:
                strategies += f" +{len(entry['optimization_strategies'])-2}"
            
            leaderboard_table.add_row(
                rank_emoji,
                entry["participant"],
                f"{entry['final_score']:.3f}",
                top_area[0].replace('_', ' ').title(),
                strategies
            )
        
        console.print(leaderboard_table)
        
        # Display evaluation statistics
        self._display_evaluation_statistics()
    
    def _display_evaluation_statistics(self):
        """
        📊 Display comprehensive evaluation statistics.
        
        🎓 EDUCATIONAL NOTE: These statistics help understand:
        - Overall workshop performance trends
        - Most common optimization strategies
        - Average scores across different dimensions
        
        This replaces the confusing "LLM: 0.0%" message with meaningful stats.
        """
        
        if not self.evaluation_stats["total_evaluations"]:
            return
        
        stats_table = Table(title="📊 WORKSHOP EVALUATION STATISTICS")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Average Score", style="green")
        stats_table.add_column("Participants", style="yellow")
        stats_table.add_column("Trend", style="blue")
        
        for metric, scores in self.evaluation_stats["average_scores"].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                trend = "📈 Improving" if len(scores) > 1 and scores[-1] > scores[0] else "📊 Stable"
                
                stats_table.add_row(
                    metric.replace('_', ' ').title(),
                    f"{avg_score:.3f}",
                    f"{len(scores)}",
                    trend
                )
        
        console.print(stats_table)
        
        # Summary insights
        console.print(Panel(
            f"📈 **WORKSHOP PERFORMANCE INSIGHTS**\n\n"
            f"• Total Evaluations: {self.evaluation_stats['total_evaluations']}\n"
            f"• Participants: {len(self.leaderboard)}\n"
            f"• Most Common Strategies: {self._get_most_common_strategies()}\n"
            f"• Average Final Score: {self._get_average_final_score():.3f}\n"
            f"• Top Performing Area: {self._get_top_performing_area()}\n\n"
            f"🎓 **Key Learning**: {self._get_key_learning_insight()}",
            title="Workshop Analytics",
            border_style="blue"
        ))
    
    def _get_most_common_strategies(self):
        """Get the most commonly used optimization strategies."""
        strategy_counts = {}
        for entry in self.leaderboard:
            for strategy in entry["optimization_strategies"]:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            most_common = max(strategy_counts.items(), key=lambda x: x[1])
            return f"{most_common[0]} ({most_common[1]} participants)"
        return "None identified"
    
    def _get_average_final_score(self):
        """Calculate average final score across all participants."""
        if self.leaderboard:
            return sum(entry["final_score"] for entry in self.leaderboard) / len(self.leaderboard)
        return 0.0
    
    def _get_top_performing_area(self):
        """Identify the area where participants performed best on average."""
        if not self.evaluation_stats["average_scores"]:
            return "Not enough data"
        
        avg_by_area = {}
        for metric, scores in self.evaluation_stats["average_scores"].items():
            if scores:
                avg_by_area[metric] = sum(scores) / len(scores)
        
        if avg_by_area:
            top_area = max(avg_by_area.items(), key=lambda x: x[1])
            return f"{top_area[0].replace('_', ' ').title()} ({top_area[1]:.3f})"
        return "Not enough data"
    
    def _get_key_learning_insight(self):
        """Generate a key learning insight based on the data."""
        if len(self.leaderboard) < 2:
            return "More participants needed for meaningful insights"
        
        # Analyze top performer
        winner = self.leaderboard[0]
        num_strategies = len(winner["optimization_strategies"])
        
        if num_strategies >= 4:
            return "Comprehensive optimization approaches tend to perform best"
        elif winner["breakdown"]["innovation_factor"]["score"] > 0.8:
            return "Innovation and creativity are highly rewarded"
        elif winner["breakdown"]["feedback_integration"] > 0.8:
            return "Strong feedback integration leads to better performance"
        else:
            return "Balanced approaches across multiple dimensions work well"
    
    def get_detailed_analysis(self):
        """
        📈 Get detailed analysis of top performers.
        
        🎓 EDUCATIONAL NOTE: This method provides comprehensive analysis of
        the top-performing optimization strategies, helping participants understand:
        
        - What made the winning approach successful
        - Performance breakdown across all evaluation dimensions
        - Specific optimization strategies that were most effective
        - Key success factors that can be applied to other projects
        
        This type of analysis is crucial for learning from successful
        optimization approaches and improving future agent deployments.
        """
        
        if not self.leaderboard:
            return "No entries to analyze"
        
        winner = self.leaderboard[0]
        
        analysis = f"""🏆 **COMPREHENSIVE WORKSHOP ANALYSIS**

**🥇 Top Performer**: {winner['participant']}
**🎯 Final Score**: {winner['final_score']:.3f}/1.000

**📊 Performance Breakdown**:
• Baseline Performance: {winner['breakdown']['baseline_performance']['overall_score']:.3f}
• Optimization Improvement: {winner['breakdown']['optimization_improvement']:.3f}
• Feedback Integration: {winner['breakdown']['feedback_integration']:.3f}
• Innovation Factor: {winner['breakdown']['innovation_factor']['score'] if isinstance(winner['breakdown']['innovation_factor'], dict) else winner['breakdown']['innovation_factor']:.3f}
• Consistency Score: {winner['breakdown']['consistency_score']:.3f}
• Production System: {winner['breakdown']['production_system']:.3f}

**🧠 Optimization Strategies Used**:
{chr(10).join([f"• {strategy}" for strategy in winner['optimization_strategies']])}

**🔑 Key Success Factors**:
• Comprehensive approach using multiple optimization techniques
• Effective integration of human feedback into agent behavior
• Consistent performance across evaluation criteria
• Innovative combinations of workshop techniques
• Scalable and adaptable production system

**📈 Performance Insights**:
• {self._get_winner_strength_analysis(winner)}
• Workshop Average Score: {self._get_average_final_score():.3f}
• Performance Advantage: {(winner['final_score'] - self._get_average_final_score()) * 100:.1f}% above average"""
        
        return analysis
    
    def _get_winner_strength_analysis(self, winner):
        """Analyze what made the winner successful."""
        breakdown = winner["breakdown"]
        
        # Find the strongest area
        max_score = 0
        strongest_area = ""
        for area, value in breakdown.items():
            score = value["score"] if isinstance(value, dict) and "score" in value else value if isinstance(value, (int, float)) else 0
            if score > max_score:
                max_score = score
                strongest_area = area
        
        return f"Excelled in {strongest_area.replace('_', ' ').title()} ({max_score:.3f}/1.000)"


# Initialize comprehensive evaluator
workshop_evaluator = ComprehensiveWorkshopEvaluator()

# 👨‍💻 COMPETITION SUBMISSION INTERFACE
console.print("🏆 **FINAL WORKSHOP EVALUATION & COMPETITION** 🏆\n")

console.print(Panel(
    """🎯 **SUBMIT YOUR OPTIMIZATION RESULTS**

Time to showcase everything you've learned! Create your final submission:

**STEP 1**: Configure your optimization approach
```python
my_optimization_config = {
    "latency_optimization": {"improvement": 35.0},  # Your % improvement
    "caching_enabled": True,                        # Did you use caching?
    "model_optimization": True,                     # Model selection used?
    "parallel_processing": True,                    # Parallel techniques?
    "advanced_feedback": True,                      # Enhanced feedback system?
    "custom_metrics": False,                        # Custom evaluation metrics?
    # Add your own optimization techniques here!
}
```

**STEP 2**: Submit your evaluation
```python
# Submit your results to the leaderboard
my_evaluation = workshop_evaluator.run_comprehensive_evaluation(
    participant_name="Your Name",
    optimization_config=my_optimization_config,
    participant_agents=[enhanced_grid_agent, enhanced_emergency_agent, enhanced_traffic_agent]  # Optional: your custom agents
)

# View the leaderboard
workshop_evaluator.display_leaderboard()
```

**💡 PRO TIPS FOR HIGH SCORES**:
• Combine multiple optimization techniques for higher improvement scores
• Use feedback-enhanced agents for better adaptation scores
• Include innovative approaches for bonus innovation points
• Test your configuration thoroughly for consistency points""",
    title="Final Competition Interface",
    border_style="yellow"
))

# Create example demo entries for the leaderboard
demo_configurations = [
    {
        "participant": "Demo: Speed Specialist",
        "config": {
            "latency_optimization": {"improvement": 45.0},
            "caching_enabled": True,
            "parallel_processing": True,
            "model_optimization": True
        }
    },
    {
        "participant": "Demo: Feedback Master",
        "config": {
            "latency_optimization": {"improvement": 25.0},
            "advanced_feedback": True,
            "custom_metrics": True,
            "caching_enabled": True
        }
    },
    {
        "participant": "Demo: Innovation Leader",
        "config": {
            "latency_optimization": {"improvement": 38.0},
            "caching_enabled": True,
            "parallel_processing": True,
            "advanced_feedback": True,
            "custom_metrics": True,
            "custom_ensemble_approach": True
        }
    }
]

# Submit demo entries
for demo in demo_configurations:
    workshop_evaluator.run_comprehensive_evaluation(
        demo["participant"], 
        demo["config"]
    )

# Display initial leaderboard with demo entries
console.print("\n📊 **CURRENT LEADERBOARD**")
workshop_evaluator.display_leaderboard()

# %% [markdown]
"""
### 🎯 **FINAL RESULTS & WORKSHOP COMPLETION**

**⏰ Final submissions close now!** 

See how your optimization techniques compare and learn from the top performers.
"""

# %%
# Final workshop results and analysis
console.print("\n🏆 **FINAL WORKSHOP RESULTS** 🏆\n")

# Display final leaderboard
workshop_evaluator.display_leaderboard()

# Get detailed analysis
detailed_analysis = workshop_evaluator.get_detailed_analysis()
console.print(Panel(detailed_analysis, title="🏆 WINNER ANALYSIS", border_style="yellow"))

# Create comprehensive scoring table showing all workshop results
comprehensive_table = Table(title="📊 COMPREHENSIVE WORKSHOP SCORING SUMMARY")
comprehensive_table.add_column("Evaluation Phase", style="cyan", no_wrap=True)
comprehensive_table.add_column("Scenario/Test", style="green")
comprehensive_table.add_column("Score", style="yellow")
comprehensive_table.add_column("Key Metric", style="blue")
comprehensive_table.add_column("Notes", style="white")

# Add baseline performance results
if 'baseline_results' in locals():
    for scenario_name, data in baseline_results.items():
        metrics = data["metrics"]
        comprehensive_table.add_row(
            "Baseline Performance",
            scenario_name.replace("_", " ").title(),
            f"{metrics.overall_score():.3f}",
            f"Response: {data['response_time']:.1f}ms",
            f"Decision Quality: {metrics.decision_quality:.3f}"
        )

# Add feedback results
if 'feedback_result' in locals():
    comprehensive_table.add_row(
        "Human Feedback",
        "Feedback-Enhanced Agents",
        f"{feedback_result['overall_score']:.3f}",
        f"Safety: {feedback_result['scores']['safety_priority']:.3f}",
        f"Adaptations: {len(feedback_system.agent_adaptations)} agents"
    )

# Add MCP integration results
if 'mcp_report' in locals():
    comprehensive_table.add_row(
        "MCP Integration",
        "Dynamic Tool Discovery",
        f"{mcp_report['integration_success_rate']:.1%}",
        f"Capabilities: {mcp_report['total_capabilities']}",
        f"Services: {mcp_report['discovered_services']} discovered"
    )

# Add competition entries
if hasattr(workshop_evaluator, 'leaderboard') and workshop_evaluator.leaderboard:
    for i, entry in enumerate(workshop_evaluator.leaderboard[:3], 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        comprehensive_table.add_row(
            "Final Competition",
            f"{rank_emoji} {entry['participant']}",
            f"{entry['final_score']:.3f}",
            f"Techniques: {len(entry['optimization_strategies'])}",
            f"Top performing optimization entry"
        )

console.print(comprehensive_table)

# Combine all workshop results for final summary
final_workshop_results = {
    "baseline_performance": baseline_results if 'baseline_results' in locals() else {},
    "optimization_strategies": optimization_results if 'optimization_results' in locals() else {},
    "human_feedback": feedback_system.analyze_feedback_patterns() if 'feedback_system' in locals() else {},
    "final_leaderboard": workshop_evaluator.leaderboard,
    "evaluation_framework": {
        "total_evaluations": len(evaluation_framework.evaluation_history),
        "llm_judge_assessments": len(evaluation_framework.llm_judge_history)
    },
    "configurations_used": {
        "evaluation": EVALUATION_CONFIG if 'EVALUATION_CONFIG' in locals() else {},
        "optimization": OPTIMIZATION_CONFIG if 'OPTIMIZATION_CONFIG' in locals() else {},
        "feedback": FEEDBACK_CONFIG if 'FEEDBACK_CONFIG' in locals() else {},
        "mcp": MCP_CONFIG if 'MCP_CONFIG' in locals() else {},
        "production": PRODUCTION_CONFIG if 'PRODUCTION_CONFIG' in locals() else {}
    },
    "workshop_completion": datetime.now().isoformat()
}

console.print(Panel(
    f"""🎉 **Agentic AI Optimization Workshop Complete!** 🎉

**🏆 Competition Results:**
• Total Participants: {len(workshop_evaluator.leaderboard)}
• Evaluation Criteria: 6 comprehensive dimensions
• Top Score: {workshop_evaluator.leaderboard[0]['final_score']:.3f} if workshop_evaluator.leaderboard else 'N/A'

**🎓 Workshop Achievements:**
✅ Built comprehensive evaluation frameworks with LLM-as-a-judge
✅ Implemented latency optimization with measurable improvements  
✅ Created sophisticated human feedback integration with specific command patterns
✅ Enhanced agents with feedback-driven prompt modifications
✅ Developed optimization strategies across multiple dimensions
✅ Competed in comprehensive evaluation with detailed scoring
✅ Implemented MCP integration with dynamic tool discovery
✅ Designed scalable and adaptable production system

**📊 Key Workshop Stats:**
• Baseline Performance: {len(baseline_results if 'baseline_results' in locals() else {})} scenarios tested
• Optimization Strategies: {len(optimization_results if 'optimization_results' in locals() else {})} approaches benchmarked  
• Human Feedback: {len(feedback_system.feedback_history) if 'feedback_system' in locals() else 0} sessions analyzed
• Agent Adaptations: {len(feedback_system.agent_adaptations) if 'feedback_system' in locals() else 0} agents enhanced
• Total Evaluations: {len(evaluation_framework.evaluation_history)} comprehensive assessments
• MCP Integration: {len(mcp_registry.discovered_services) if 'mcp_registry' in locals() else 0} services integrated
• Production System: {len(PRODUCTION_CONFIG['modular_architecture']['module_types'])} modules configured

**🔧 Everything remains customizable for continued experimentation!**

🚀 **Congratulations to all participants!**
Keep building amazing agentic AI systems with these optimization techniques!""",
    title="🏆 Workshop & Competition Complete!",
    border_style="green"
))

# Save all results including comprehensive evaluation data
save_experiment_results(final_workshop_results)

console.print("📊 Workshop results saved for future reference!")
console.print("🎓 Thank you for participating in the comprehensive optimization challenge!")