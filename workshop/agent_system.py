from typing import Dict, List, Any, Optional, Union
import time
import json
import os
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from enum import Enum

from .command import Command, CommandResult, CommandExecutor, CommandPlan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScenarioType(str, Enum):
    """Types of scenarios that can be run."""
    GRID_SURGE = "grid_surge_heat_wave"
    MEDICAL_EMERGENCY = "medical_emergency_festival"
    DRONE_CAPACITY = "drone_capacity_crisis"
    FLOOD_DISRUPTION = "flood_advisory_disruption"
    CITY_WIDE_DRILL = "city_wide_drill"

class AgentMetrics(BaseModel):
    """Metrics for evaluating agent performance."""
    incident_coverage: float = Field(0.0, description="Percentage of emergency incidents dispatched to")
    average_eta: float = Field(0.0, description="Average estimated time of arrival in minutes")
    power_zone_mitigation: float = Field(0.0, description="Percentage of critical power zones properly managed")
    tool_failure_handling: float = Field(0.0, description="Percentage of tool failures with successful fallback")
    feedback_response: float = Field(0.0, description="Percentage of feedback items that led to improved behavior")
    capability_match: float = Field(0.0, description="Percentage of incidents with correctly matched drone capabilities")
    weather_adaptation: float = Field(0.0, description="Percentage of weather events properly handled")
    
    # Additional metrics for workshop
    latency_seconds: float = Field(0.0, description="Time taken to complete the scenario")
    steps_taken: int = Field(0, description="Number of steps taken to reach a conclusion")
    final_score: float = Field(0.0, description="Overall score for the scenario")
    command_count: int = Field(0, description="Number of commands issued")
    successful_commands: int = Field(0, description="Number of commands that executed successfully")
    failed_commands: int = Field(0, description="Number of commands that failed to execute")
    avg_command_execution_time: float = Field(0.0, description="Average execution time per command in seconds")

# Standardized output schemas for each scenario
class GridSurgeOutput(BaseModel):
    """Standardized output for grid surge scenario."""
    critical_zones: Dict[str, float] = Field(
        ..., description="Map of critical zone IDs to their current load percentage"
    )
    load_reduction_actions: Dict[str, float] = Field(
        ..., description="Map of zone IDs to percentage load reduction applied"
    )
    priority_infrastructure: Dict[str, str] = Field(
        ..., description="Map of critical infrastructure to their protection status"
    )
    weather_impact: str = Field(..., description="Assessment of weather impact on grid")
    emergency_readiness: str = Field(..., description="Status of emergency response readiness")
    action_plan: str = Field(..., description="Overall action plan for the heat wave")
    executed_commands: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of commands executed for this scenario"
    )

class MedicalEmergencyOutput(BaseModel):
    """Standardized output for medical emergency scenario."""
    incidents_by_priority: Dict[str, int] = Field(
        ..., description="Count of incidents by priority level (critical, severe, moderate, minor)"
    )
    drone_assignments: Dict[str, str] = Field(
        ..., description="Map of drone IDs to incident IDs they're assigned to"
    )
    estimated_response_times: Dict[str, float] = Field(
        ..., description="Map of incident IDs to estimated response times in minutes"
    )
    traffic_conditions: str = Field(..., description="Summary of traffic conditions")
    action_plan: str = Field(..., description="Overall action plan for the medical emergency")
    executed_commands: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of commands executed for this scenario"
    )

class DroneCapacityOutput(BaseModel):
    """Standardized output for drone capacity crisis scenario."""
    available_drones: Dict[str, str] = Field(
        ..., description="Map of available drone IDs to their capabilities"
    )
    prioritized_incidents: Dict[str, int] = Field(
        ..., description="Map of incident IDs to their priority score"
    )
    reassignment_plan: Dict[str, str] = Field(
        ..., description="Map of drone IDs to their new incident assignments"
    )
    downgraded_incidents: Dict[str, str] = Field(
        ..., description="Map of incident IDs to reason for downgrade"
    )
    action_plan: str = Field(..., description="Overall plan for managing the capacity crisis")
    executed_commands: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of commands executed for this scenario"
    )

class FloodDisruptionOutput(BaseModel):
    """Standardized output for flood disruption scenario."""
    flood_risk_areas: Dict[str, str] = Field(
        ..., description="Map of areas to their flood risk level"
    )
    affected_grid_zones: Dict[str, str] = Field(
        ..., description="Map of grid zones to their current status"
    )
    blocked_routes: Dict[str, str] = Field(
        ..., description="Map of blocked route IDs to their status"
    )
    detour_routes: Dict[str, str] = Field(
        ..., description="Map of affected routes to recommended detours"
    )
    weather_forecast: str = Field(..., description="Current weather forecast")
    action_plan: str = Field(..., description="Overall plan for flood response")
    executed_commands: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of commands executed for this scenario"
    )

class CityWideDrillOutput(BaseModel):
    """Standardized output for city-wide drill scenario."""
    grid_status: Dict[str, str] = Field(
        ..., description="Map of grid zones to their status"
    )
    emergency_incidents: Dict[str, str] = Field(
        ..., description="Map of incident IDs to their status"
    )
    traffic_disruptions: Dict[str, str] = Field(
        ..., description="Map of traffic sectors to their status"
    )
    communication_status: Dict[str, str] = Field(
        ..., description="Map of communication systems to their status"
    )
    priority_actions: Dict[str, str] = Field(
        ..., description="Map of priority actions to responsible teams"
    )
    action_plan: str = Field(..., description="Overall response plan for the city-wide emergency")
    executed_commands: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of commands executed for this scenario"
    )

# Union type for all scenario outputs
ScenarioOutput = Union[
    GridSurgeOutput, 
    MedicalEmergencyOutput, 
    DroneCapacityOutput,
    FloodDisruptionOutput,
    CityWideDrillOutput
]

class ScenarioInput(BaseModel):
    """Input for a scenario."""
    scenario_type: ScenarioType
    day: int = Field(1, ge=1, le=31, description="Day to simulate (1-31)")
    duration_minutes: int = Field(60, description="Duration of the scenario in minutes")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the scenario")

class ScenarioResult(BaseModel):
    """Result of running a scenario."""
    scenario_type: ScenarioType
    success: bool = Field(True, description="Whether the scenario completed successfully")
    response: str = Field("", description="The overall response from the agent system")
    structured_output: Optional[Dict[str, Any]] = Field(
        None, description="Structured output data in the format defined for the scenario"
    )
    metrics: AgentMetrics = Field(default_factory=AgentMetrics, description="Performance metrics")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed results and actions taken")
    commands: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of commands issued during the scenario"
    )
    command_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Results of executing commands"
    )
    
    def save_to_file(self, filepath: Optional[str] = None):
        """Save the scenario result to a file."""
        if filepath is None:
            # Create a results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/{self.scenario_type}_{timestamp}.json"
        
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=2))
        
        logger.info(f"Saved scenario result to {filepath}")
        return filepath
    
    def add_command(self, command: Command, result: Optional[CommandResult] = None):
        """
        Add a command and its result to the scenario result.
        
        Args:
            command: The command that was issued
            result: The result of executing the command
        """
        self.commands.append(command.dict())
        if result:
            self.command_results.append(result.dict())
            
            # Update metrics
            self.metrics.command_count += 1
            if result.success:
                self.metrics.successful_commands += 1
            else:
                self.metrics.failed_commands += 1
            
            # Update average execution time
            total_time = self.metrics.avg_command_execution_time * (self.metrics.command_count - 1)
            total_time += result.execution_time
            self.metrics.avg_command_execution_time = total_time / self.metrics.command_count

class AgentSystem(BaseModel):
    """Base class for agent systems."""
    name: str
    description: str
    version: str = "1.0.0"
    initialized: bool = False
    command_executor: Optional[CommandExecutor] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def initialize(self):
        """Initialize the agent system."""
        if self.initialized:
            logger.info(f"Agent system '{self.name}' already initialized")
            return
        
        logger.info(f"Initializing agent system '{self.name}'")
        self.command_executor = CommandExecutor()
        self._initialize()
        self.initialized = True
    
    def _initialize(self):
        """Override this method to implement custom initialization."""
        pass
    
    def invoke(self, scenario: ScenarioInput) -> ScenarioResult:
        """Invoke the agent system with a scenario."""
        if not self.initialized:
            self.initialize()
        
        logger.info(f"Running scenario {scenario.scenario_type} on day {scenario.day}")
        
        # Record start time for latency measurement
        start_time = time.time()
        
        # Run the scenario
        result = self._invoke_scenario(scenario)
        
        # Calculate latency
        latency = time.time() - start_time
        result.metrics.latency_seconds = latency
        
        # Validate the structured output against the expected schema
        self._validate_scenario_output(result)
        
        return result
    
    def _invoke_scenario(self, scenario: ScenarioInput) -> ScenarioResult:
        """Override this method to implement scenario-specific logic."""
        return ScenarioResult(scenario_type=scenario.scenario_type)
    
    def _validate_scenario_output(self, result: ScenarioResult):
        """Validate structured output against the expected schema."""
        if not result.structured_output:
            logger.warning(f"No structured output provided for scenario {result.scenario_type}")
            return
        
        try:
            # Select the appropriate schema based on scenario type
            if result.scenario_type == ScenarioType.GRID_SURGE:
                GridSurgeOutput(**result.structured_output)
            elif result.scenario_type == ScenarioType.MEDICAL_EMERGENCY:
                MedicalEmergencyOutput(**result.structured_output)
            elif result.scenario_type == ScenarioType.DRONE_CAPACITY:
                DroneCapacityOutput(**result.structured_output)
            elif result.scenario_type == ScenarioType.FLOOD_DISRUPTION:
                FloodDisruptionOutput(**result.structured_output)
            elif result.scenario_type == ScenarioType.CITY_WIDE_DRILL:
                CityWideDrillOutput(**result.structured_output)
            
            logger.info(f"Structured output for {result.scenario_type} is valid")
        except Exception as e:
            logger.error(f"Invalid structured output for {result.scenario_type}: {str(e)}")
            # Don't fail the scenario, just log the error
    
    def execute_command(self, command: Command, result: ScenarioResult) -> CommandResult:
        """
        Execute a command and update the scenario result.
        
        Args:
            command: The command to execute
            result: The scenario result to update
            
        Returns:
            The result of executing the command
        """
        if not self.command_executor:
            self.command_executor = CommandExecutor()
        
        # Execute the command
        command_result = self.command_executor.execute(command)
        
        # Update the scenario result
        result.add_command(command, command_result)
        
        # Add the command to the structured output
        if result.structured_output and "executed_commands" in result.structured_output:
            result.structured_output["executed_commands"].append(command.dict())
        
        return command_result
    
    def execute_plan(self, plan: CommandPlan, result: ScenarioResult) -> List[CommandResult]:
        """
        Execute a command plan and update the scenario result.
        
        Args:
            plan: The plan to execute
            result: The scenario result to update
            
        Returns:
            List of command execution results
        """
        results = []
        for command in plan.commands:
            command_result = self.execute_command(command, result)
            results.append(command_result)
        
        return results
    
    def shutdown(self):
        """Shutdown the agent system."""
        logger.info(f"Shutting down agent system '{self.name}'")
        self._shutdown()
        self.initialized = False
    
    def _shutdown(self):
        """Override this method to implement custom shutdown logic."""
        pass

# Helper functions to create structured outputs for scenarios
def create_grid_surge_output(
    critical_zones: Dict[str, float],
    load_reduction_actions: Dict[str, float],
    priority_infrastructure: Dict[str, str],
    weather_impact: str,
    emergency_readiness: str,
    action_plan: str,
    executed_commands: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a structured output for a grid surge scenario."""
    return {
        "critical_zones": critical_zones,
        "load_reduction_actions": load_reduction_actions,
        "priority_infrastructure": priority_infrastructure,
        "weather_impact": weather_impact,
        "emergency_readiness": emergency_readiness,
        "action_plan": action_plan,
        "executed_commands": executed_commands or []
    }

def create_medical_emergency_output(
    incidents_by_priority: Dict[str, int],
    drone_assignments: Dict[str, str],
    estimated_response_times: Dict[str, float],
    traffic_conditions: str,
    action_plan: str,
    executed_commands: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a structured output for a medical emergency scenario."""
    return {
        "incidents_by_priority": incidents_by_priority,
        "drone_assignments": drone_assignments,
        "estimated_response_times": estimated_response_times,
        "traffic_conditions": traffic_conditions,
        "action_plan": action_plan,
        "executed_commands": executed_commands or []
    }

def create_drone_capacity_output(
    available_drones: Dict[str, str],
    prioritized_incidents: Dict[str, int],
    reassignment_plan: Dict[str, str],
    downgraded_incidents: Dict[str, str],
    action_plan: str,
    executed_commands: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a structured output for a drone capacity crisis scenario."""
    return {
        "available_drones": available_drones,
        "prioritized_incidents": prioritized_incidents,
        "reassignment_plan": reassignment_plan,
        "downgraded_incidents": downgraded_incidents,
        "action_plan": action_plan,
        "executed_commands": executed_commands or []
    }

def create_flood_disruption_output(
    flood_risk_areas: Dict[str, str],
    affected_grid_zones: Dict[str, str],
    blocked_routes: Dict[str, str],
    detour_routes: Dict[str, str],
    weather_forecast: str,
    action_plan: str,
    executed_commands: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a structured output for a flood disruption scenario."""
    return {
        "flood_risk_areas": flood_risk_areas,
        "affected_grid_zones": affected_grid_zones,
        "blocked_routes": blocked_routes,
        "detour_routes": detour_routes,
        "weather_forecast": weather_forecast,
        "action_plan": action_plan,
        "executed_commands": executed_commands or []
    }

def create_city_wide_drill_output(
    grid_status: Dict[str, str],
    emergency_incidents: Dict[str, str],
    traffic_disruptions: Dict[str, str],
    communication_status: Dict[str, str],
    priority_actions: Dict[str, str],
    action_plan: str,
    executed_commands: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a structured output for a city-wide drill scenario."""
    return {
        "grid_status": grid_status,
        "emergency_incidents": emergency_incidents,
        "traffic_disruptions": traffic_disruptions,
        "communication_status": communication_status,
        "priority_actions": priority_actions,
        "action_plan": action_plan,
        "executed_commands": executed_commands or []
    } 