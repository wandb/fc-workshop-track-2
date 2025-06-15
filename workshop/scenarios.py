"""
Scenario definitions and management for the SENTINEL GRID workshop.

This module provides functions to create, run, and evaluate different scenarios
in the SENTINEL GRID smart city simulation.
"""

import time
import logging
import os
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from pydantic import BaseModel, Field
from enum import Enum

from workshop.agent_system import ScenarioType

# Initialize console
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scenarios")

# Base models for scenario evaluation
class MetricType(str, Enum):
    """Types of metrics that can be evaluated."""
    THRESHOLD = "threshold"  # Must reach a specific value
    RANGE = "range"  # Must stay within a range
    TREND = "trend"  # Must follow a specific trend
    COMPARISON = "comparison"  # Must compare favorably to another metric

class MetricDefinition(BaseModel):
    """Definition of a metric to be evaluated."""
    name: str
    description: str
    type: MetricType
    target: Any  # Target value or range
    weight: float = Field(ge=0.0, le=1.0)
    service: str  # Which service this metric belongs to
    calculation: str  # Function name to calculate this metric

class CommandImpact(BaseModel):
    """Impact of a command on metrics."""
    command: Dict[str, Any]
    affected_metrics: List[str]
    expected_impact: Dict[str, float]

class ScenarioEvaluation(BaseModel):
    """Complete scenario evaluation configuration."""
    name: str
    description: str
    metrics: List[MetricDefinition]
    optimal_commands: List[CommandImpact]
    time_limit: Optional[int] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)

def extract_commands_from_output(agent_output):
    """
    Extract commands from agent output, handling CrewAI structured outputs with output_pydantic.
    
    Args:
        agent_output: Output from an agent task (CrewAI TaskOutput with pydantic models)
        
    Returns:
        List of command dictionaries compatible with workshop.command.Command
    """
    commands = []
    
    # Handle CrewAI TaskOutput with pydantic models
    if hasattr(agent_output, 'pydantic'):
        pydantic_output = agent_output.pydantic
        
        # Handle GridManagementPlan
        if hasattr(pydantic_output, 'zone_adjustments'):
            for adjustment in pydantic_output.zone_adjustments:
                if isinstance(adjustment, dict) and 'zone_id' in adjustment and 'capacity' in adjustment:
                    commands.append({
                        "service": "grid",
                        "action": "adjust_zone",
                        "parameters": {
                            "zone_id": adjustment['zone_id'],
                            "capacity": adjustment['capacity']
                        }
                    })
        
        if hasattr(pydantic_output, 'priority_settings'):
            for priority in pydantic_output.priority_settings:
                if isinstance(priority, dict) and 'infrastructure_id' in priority and 'level' in priority:
                    commands.append({
                        "service": "grid",
                        "action": "set_priority",
                        "parameters": {
                            "infrastructure_id": priority['infrastructure_id'],
                            "level": priority['level']
                        }
                    })
        
        # Handle EmergencyResponsePlan
        if hasattr(pydantic_output, 'drone_assignments'):
            for assignment in pydantic_output.drone_assignments:
                if isinstance(assignment, dict) and 'drone_id' in assignment and 'incident_id' in assignment:
                    commands.append({
                        "service": "emergency",
                        "action": "assign_drone",
                        "parameters": {
                            "drone_id": assignment['drone_id'],
                            "incident_id": assignment['incident_id']
                        }
                    })
        
        if hasattr(pydantic_output, 'incident_updates'):
            for update in pydantic_output.incident_updates:
                if isinstance(update, dict) and 'incident_id' in update and 'status' in update:
                    commands.append({
                        "service": "emergency",
                        "action": "update_incident",
                        "parameters": {
                            "incident_id": update['incident_id'],
                            "status": update['status']
                        }
                    })
        
        # Handle TrafficManagementPlan
        if hasattr(pydantic_output, 'traffic_redirections'):
            for redirection in pydantic_output.traffic_redirections:
                if isinstance(redirection, dict) and 'sector_id' in redirection and 'target_reduction' in redirection:
                    commands.append({
                        "service": "traffic",
                        "action": "redirect",
                        "parameters": {
                            "sector_id": redirection['sector_id'],
                            "target_reduction": redirection['target_reduction']
                        }
                    })
        
        if hasattr(pydantic_output, 'route_blocks'):
            for block in pydantic_output.route_blocks:
                if isinstance(block, dict) and 'sector_id' in block and 'duration_minutes' in block:
                    commands.append({
                        "service": "traffic",
                        "action": "block_route",
                        "parameters": {
                            "sector": block['sector_id'],
                            "reason": block.get('reason', 'Emergency blocking'),
                            "duration_minutes": block['duration_minutes']
                        }
                    })
    
    # Handle direct pydantic model access (fallback)
    elif hasattr(agent_output, 'zone_adjustments'):
        for adjustment in agent_output.zone_adjustments:
            if isinstance(adjustment, dict) and 'zone_id' in adjustment and 'capacity' in adjustment:
                commands.append({
                    "service": "grid",
                    "action": "adjust_zone",
                    "parameters": {
                        "zone_id": adjustment['zone_id'],
                        "capacity": adjustment['capacity']
                    }
                })
    
    # Handle dictionary format (legacy support)
    elif isinstance(agent_output, dict):
        for key in ["zone_adjustments", "priority_settings", "drone_assignments", 
                   "incident_updates", "traffic_redirections", "route_blocks",
                   "recommendations", "commands", "actions"]:
            if key in agent_output and isinstance(agent_output[key], list):
                for item in agent_output[key]:
                    if isinstance(item, dict):
                        if key == "zone_adjustments" and 'zone_id' in item:
                            commands.append({
                                "service": "grid",
                                "action": "adjust_zone",
                                "parameters": {
                                    "zone_id": item['zone_id'],
                                    "capacity": item['capacity']
                                }
                            })
                        elif key == "drone_assignments" and 'drone_id' in item:
                            commands.append({
                                "service": "emergency",
                                "action": "assign_drone",
                                "parameters": {
                                    "drone_id": item['drone_id'],
                                    "incident_id": item['incident_id']
                                }
                            })
                        elif "service" in item and "action" in item:
                            commands.append(item)
    
    # Handle string outputs by parsing for command patterns
    elif isinstance(agent_output, str):
        import re
        import json
        
        # Look for JSON-like command structures
        json_pattern = r'\{[^{}]*"service"[^{}]*"action"[^{}]*\}'
        matches = re.findall(json_pattern, agent_output)
        
        for match in matches:
            try:
                cmd = json.loads(match)
                if "service" in cmd and "action" in cmd:
                    commands.append(cmd)
            except json.JSONDecodeError:
                continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_commands = []
    for cmd in commands:
        cmd_key = (cmd.get("service"), cmd.get("action"), 
                  str(sorted(cmd.get("parameters", {}).items())))
        if cmd_key not in seen:
            seen.add(cmd_key)
            unique_commands.append(cmd)
    
    return unique_commands

def run_scenario(scenario_type, scenario_description, crew=None, constraints=None):
    """
    Run a scenario with a CrewAI crew and evaluate performance.
    
    Args:
        scenario_type: Type of scenario (ScenarioType)
        scenario_description: Description of the scenario
        crew: Optional crew to use (if None, creates a new heat wave crew)
        constraints: Dict of scenario constraints (e.g., {"available_drones": 3})
        
    Returns:
        Dictionary with evaluation results
    """
    start_time = time.time()
    
    # Apply constraints if provided
    if constraints:
        log.info(f"Applying scenario constraints: {constraints}")
        # This would normally be implemented by passing constraints to the services
        # For demonstration, we just log them
    
    # Create crew if not provided
    if crew is None:
        from workshop.deprecated_utils.agents import create_heat_wave_crew
        crew = create_heat_wave_crew(scenario_description)
    
    # Display scenario info
    console.print(Panel.fit(
        f"[bold]Running Scenario: {scenario_type.name}[/bold]\n\n{scenario_description}",
        title="SENTINEL GRID Scenario",
        border_style="blue"
    ))
    
    if constraints:
        constraint_text = "\n".join([f"- {k}: {v}" for k, v in constraints.items()])
        console.print(Panel.fit(
            f"[bold yellow]Scenario Constraints:[/bold yellow]\n{constraint_text}",
            border_style="yellow"
        ))
    
    # Setup progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Track overall progress
        scenario_task = progress.add_task("[bold blue]Running scenario...", total=100)
        
        # Run the crew
        progress.update(scenario_task, advance=10, description="[bold blue]Initializing agents...")
        
        # Kickoff the crew
        progress.update(scenario_task, advance=10, description="[bold blue]Agents activated...")
        result = crew.kickoff()
        
        progress.update(scenario_task, advance=30, description="[bold blue]Processing agent outputs...")
        
        # Extract commands from results
        commands = []
        task_metrics = {}
        
        # Process each task output
        for task in crew.tasks:
            task_start = task.started_at if hasattr(task, 'started_at') else 0
            task_end = task.completed_at if hasattr(task, 'completed_at') else time.time()
            task_duration = task_end - task_start if task_start else 0
            
            if task.output:
                task_commands = extract_commands_from_output(task.output)
                commands.extend(task_commands)
                log.info(f"Extracted {len(task_commands)} commands from {task.agent.role}")
                
                # Track task metrics
                task_metrics[task.agent.role] = {
                    "duration_seconds": task_duration,
                    "commands_generated": len(task_commands),
                    "task_description": task.description.split("\n")[0][:50] + "..."
                }
        
        log.info(f"Total commands extracted: {len(commands)}")
        
        # Evaluate performance
        progress.update(scenario_task, advance=30, description="[bold blue]Evaluating performance...")
        # Import at runtime to avoid circular dependency
        from workshop.command_evaluator import evaluate_scenario_commands
        evaluation = evaluate_scenario_commands(commands, scenario_type, constraints)
        
        # Calculate overall scenario duration
        scenario_duration = time.time() - start_time
        
        # Complete progress
        progress.update(scenario_task, advance=20, description="[bold green]Scenario completed!")
    
    # Display evaluation results
    console.print("\n")
    console.print(Panel.fit(
        f"[bold]Overall Score: [green]{evaluation.get('overall_score', 0):.2f}[/green][/bold]\n\n{evaluation.get('feedback', '')}",
        title="Performance Evaluation",
        border_style="green"
    ))
    
    # Return comprehensive results
    return {
        "commands": commands,
        "evaluation": evaluation,
        "crew_result": result,
        "task_metrics": task_metrics,
        "scenario_metrics": {
            "duration_seconds": scenario_duration,
            "total_commands": len(commands),
            "agent_count": len(task_metrics)
        }
    }

# Define scenario creation functions with appropriate constraints
def create_heat_wave_scenario():
    """Create a heat wave (grid surge) scenario."""
    constraints = {
        "available_drones": 5,  # Full drone capacity
        "grid_stability_threshold": 0.7,  # Stability threshold for grid zones
        "weather_condition": "heat_wave",  # Weather condition
        "max_temperature": 40.2,  # Celsius
        "traffic_congestion_level": 0.6,  # Moderate traffic congestion
    }
    
    return {
        "type": ScenarioType.GRID_SURGE,
        "description": "A severe heat wave is causing increased power demand across the city. " +
                       "Grid zones are approaching capacity and temperatures continue to rise.",
        "constraints": constraints
    }

def create_storm_scenario():
    """Create a severe storm (flood disruption) scenario."""
    constraints = {
        "available_drones": 4,  # Reduced drone capacity due to weather
        "grid_stability_threshold": 0.6,  # Lower stability threshold for grid
        "weather_condition": "severe_storm",  # Weather condition
        "flooded_sectors": ["Downtown", "East Side", "River District"],  # Flooded areas
        "traffic_congestion_level": 0.8,  # High traffic congestion due to flooding
        "drone_speed_reduction": 0.7,  # Drones move slower in heavy rain
    }
    
    return {
        "type": ScenarioType.FLOOD_DISRUPTION,
        "description": "A severe storm is causing flooding and disruptions across the city. " +
                       "Multiple sectors are affected and emergency resources must be prioritized.",
        "constraints": constraints
    }

def create_festival_scenario():
    """Create a festival medical emergency scenario."""
    constraints = {
        "available_drones": 5,  # Full drone capacity
        "grid_stability_threshold": 0.8,  # Normal grid stability
        "weather_condition": "clear",  # Weather condition
        "traffic_congestion_level": 0.9,  # Very high traffic congestion due to festival
        "incident_concentration": "Downtown",  # Where most incidents are occurring
        "incident_count": 12,  # High number of simultaneous incidents
    }
    
    return {
        "type": ScenarioType.MEDICAL_EMERGENCY,
        "description": "A festival has led to multiple medical emergencies across downtown. " +
                       "High traffic congestion is delaying response times and resources are stretched thin.",
        "constraints": constraints
    }

def create_drone_crisis_scenario():
    """Create a drone capacity crisis scenario."""
    constraints = {
        "available_drones": 2,  # Severely limited drone capacity
        "grid_stability_threshold": 0.75,  # Normal-ish grid stability
        "weather_condition": "clear",  # Weather condition
        "traffic_congestion_level": 0.5,  # Moderate traffic congestion
        "incident_count": 8,  # High number of incidents but limited drones
        "priority_sectors": ["Hospital District", "Government Center"],  # Areas to prioritize
    }
    
    return {
        "type": ScenarioType.DRONE_CAPACITY_CRISIS,
        "description": "Multiple emergencies have stretched drone resources to their limits. " +
                       "With only 2 operational drones, you must prioritize critical incidents.",
        "constraints": constraints
    }

def create_infrastructure_collapse_scenario():
    """Create an infrastructure collapse scenario."""
    constraints = {
        "available_drones": 3,  # Reduced drone capacity
        "grid_stability_threshold": 0.3,  # Critical grid instability
        "weather_condition": "rain",  # Weather condition
        "traffic_congestion_level": 0.95,  # Extreme traffic congestion
        "failed_infrastructure": ["Main Power Plant", "Central Bridge", "Water Treatment"],
        "affected_zones": ["Downtown", "Industrial Zone", "West Side", "North Side"],
        "communication_delay": 30,  # 30 second delay in communications
    }
    
    return {
        "type": ScenarioType.INFRASTRUCTURE_COLLAPSE,
        "description": "Critical infrastructure is failing, causing cascading effects across the city. " +
                       "Power outages, traffic gridlock, and communications issues require coordinated response.",
        "constraints": constraints
    }

# Function to run specific scenario by type
def run_scenario_by_type(scenario_type):
    """Run a specific scenario by type."""
    # Special handling for infrastructure collapse scenario
    if scenario_type == ScenarioType.INFRASTRUCTURE_COLLAPSE:
        from workshop.deprecated_utils.advanced_patterns import run_infrastructure_collapse_scenario
        return run_infrastructure_collapse_scenario()
    
    # Normal handling for other scenarios
    scenarios = {
        ScenarioType.GRID_SURGE: create_heat_wave_scenario(),
        ScenarioType.FLOOD_DISRUPTION: create_storm_scenario(),
        ScenarioType.MEDICAL_EMERGENCY: create_festival_scenario(),
        ScenarioType.DRONE_CAPACITY_CRISIS: create_drone_crisis_scenario(),
    }
    
    if scenario_type not in scenarios:
        log.error(f"Unknown scenario type: {scenario_type}")
        return None
    
    scenario = scenarios[scenario_type]
    return run_scenario(
        scenario_type=scenario["type"],
        scenario_description=scenario["description"],
        constraints=scenario["constraints"]
    )

def create_scenario(
    name: str,
    description: str,
    metrics: List[MetricDefinition],
    optimal_commands: List[CommandImpact],
    time_limit: Optional[int] = None,
    constraints: Dict[str, Any] = None
) -> ScenarioEvaluation:
    """Create a new scenario with custom metrics and optimal commands."""
    return ScenarioEvaluation(
        name=name,
        description=description,
        metrics=metrics,
        optimal_commands=optimal_commands,
        time_limit=time_limit,
        constraints=constraints or {}
    )

# Example of heat wave metrics and optimal commands
HEAT_WAVE_METRICS = [
    MetricDefinition(
        name="grid_stability",
        description="Overall stability of the power grid",
        type=MetricType.THRESHOLD,
        target=0.8,
        weight=0.4,
        service="grid",
        calculation="grid_stability"
    ),
    MetricDefinition(
        name="power_conservation",
        description="Amount of power conserved",
        type=MetricType.THRESHOLD,
        target=0.2,
        weight=0.3,
        service="grid",
        calculation="power_conservation"
    ),
    MetricDefinition(
        name="incident_response",
        description="Percentage of incidents responded to",
        type=MetricType.THRESHOLD,
        target=0.9,
        weight=0.3,
        service="emergency",
        calculation="incident_response"
    )
]

HEAT_WAVE_OPTIMAL_COMMANDS = [
    CommandImpact(
        command={
            "service": "grid",
            "action": "adjust_zone",
            "parameters": {"zone_id": "zone_a", "capacity": 0.8}
        },
        affected_metrics=["grid_stability", "power_conservation"],
        expected_impact={
            "grid_stability": 0.1,
            "power_conservation": 0.05
        }
    ),
    CommandImpact(
        command={
            "service": "grid",
            "action": "adjust_zone",
            "parameters": {"zone_id": "zone_b", "capacity": 0.7}
        },
        affected_metrics=["grid_stability", "power_conservation"],
        expected_impact={
            "grid_stability": 0.08,
            "power_conservation": 0.07
        }
    ),
    CommandImpact(
        command={
            "service": "emergency",
            "action": "assign_drone",
            "parameters": {"drone_id": "drone_1", "incident_id": "incident_1"}
        },
        affected_metrics=["incident_response"],
        expected_impact={
            "incident_response": 0.2
        }
    )
]

# Update heat wave scenario to use the new structure
def create_heat_wave_scenario_evaluation() -> ScenarioEvaluation:
    """Create a heat wave scenario with custom metrics."""
    return create_scenario(
        name="Heat Wave Crisis",
        description="A severe heat wave is causing grid stress",
        metrics=HEAT_WAVE_METRICS,
        optimal_commands=HEAT_WAVE_OPTIMAL_COMMANDS,
        time_limit=30,
        constraints={
            "available_drones": 5,
            "weather_condition": "heat_wave",
            "max_temperature": 40.2,
            "grid_stability_threshold": 0.7,
            "traffic_congestion_level": 0.6
        }
    ) 