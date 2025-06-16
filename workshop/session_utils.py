#!/usr/bin/env python3
"""
Afternoon Session Utilities - Helper functions copied from morning session
"""

import requests
from typing import Dict, List, Any
from rich.console import Console
from rich.panel import Panel
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from workshop.state_management import get_actual_service_ids, SERVICE_URLS
from workshop.state_models import (
    ScenarioDefinition, ServiceState, ZoneState, IncidentState, 
    DroneState, TrafficState, SuccessCriteria
)
from workshop.command import Command, ServiceType, CommandExecutor
from datetime import datetime, timedelta
import weave

console = Console()


# Structured output models (copied from morning session)
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
        description="Description of maintained emergency vehicle access"
    )
    coordination_notes: str = Field(
        description="Notes for other agents about traffic impacts"
    )


# Tool creation functions (copied from morning session)
def create_grid_zone_adjustment_tool():
    """Create GridZoneAdjustmentTool with dynamic description."""
    actual_ids = get_actual_service_ids()
    available_zones = actual_ids.get('grid_zones', ['Z001', 'Z002', 'Z003'])
    
    class DynamicGridZoneAdjustmentTool(BaseTool):
        name: str = "adjust_grid_zone"
        description: str = (
            f"Adjust the capacity of a specific power grid zone.\n\n"
            f"Parameters:\n"
            f"- zone_id: ID of the zone (available zones: "
            f"{', '.join(available_zones)})\n"
            f"- capacity: New capacity level between 0.0 and 1.0\n"
            f"- reason: Reason for the adjustment\n\n"
            f"Returns success/failure status."
        )
        
        def __init__(self):
            super().__init__()
            self._execution_results = []
        
        def _run(self, zone_id: str, capacity: float, reason: str) -> str:
            cmd = Command(
                service=ServiceType.GRID,
                action="adjust_zone",
                parameters={"zone_id": zone_id, "capacity": capacity}
            )
            
            executor = CommandExecutor()
            result = executor.execute(cmd)
            
            self._execution_results.append(result.success)
            
            status = "SUCCESS" if result.success else "FAILED"
            console.print(f"🔧 Grid: {zone_id} → {capacity:.1%} "
                         f"({reason}) - {status}")
            
            return f"Grid zone {zone_id} adjustment: {status}"

    return DynamicGridZoneAdjustmentTool()


def create_infrastructure_priority_tool():
    """Create InfrastructurePriorityTool with dynamic description."""
    try:
        response = requests.get(f"{SERVICE_URLS['grid']}/service/info", 
                               timeout=5)
        if response.status_code == 200:
            available_infrastructure = [
                "hospital", "police", "emergency_services", 
                "water_treatment", "data_center", "emergency_shelter"
            ]
        else:
            available_infrastructure = [
                "hospital", "police", "emergency_services"
            ]
    except Exception:
        available_infrastructure = ["hospital", "police", "emergency_services"]
    
    class DynamicInfrastructurePriorityTool(BaseTool):
        name: str = "set_infrastructure_priority"
        description: str = (
            f"Set priority level for critical infrastructure.\n\n"
            f"Parameters:\n"
            f"- infrastructure_id: ID of infrastructure (available: "
            f"{', '.join(available_infrastructure)})\n"
            f"- level: Priority level ('normal', 'high', 'critical')\n"
            f"- reason: Reason for priority change\n\n"
            f"Returns success/failure status."
        )
        
        def __init__(self):
            super().__init__()
            self._execution_results = []
        
        def _run(self, infrastructure_id: str, level: str, reason: str) -> str:
            cmd = Command(
                service=ServiceType.GRID,
                action="set_priority",
                parameters={"infrastructure_id": infrastructure_id, 
                           "level": level}
            )
            
            executor = CommandExecutor()
            result = executor.execute(cmd)
            
            self._execution_results.append(result.success)
            
            status = "SUCCESS" if result.success else "FAILED"
            console.print(f"⚡ Priority: {infrastructure_id} → {level} "
                         f"({reason}) - {status}")
            
            return f"Infrastructure {infrastructure_id} priority: {status}"

    return DynamicInfrastructurePriorityTool()


def create_drone_assignment_tool():
    """Create DroneAssignmentTool with dynamic description."""
    actual_ids = get_actual_service_ids()
    available_drones = actual_ids.get('drones', 
                                     ['D001', 'D002', 'D003', 'D004'])
    available_incidents = actual_ids.get('incidents', 
                                        ['E-1001', 'E-1002', 'E-1003', 
                                         'E-1004'])
    
    class DynamicDroneAssignmentTool(BaseTool):
        name: str = "assign_emergency_drone"
        description: str = (
            f"Assign an available drone to an emergency incident.\n"
            f"Available drones: {', '.join(available_drones)}\n"
            f"Available incidents: {', '.join(available_incidents)}"
        )
        
        def _run(self, drone_id: str, incident_id: str, reason: str) -> str:
            cmd = Command(
                service=ServiceType.EMERGENCY,
                action="assign_drone",
                parameters={"drone_id": drone_id, "incident_id": incident_id}
            )
            
            executor = CommandExecutor()
            result = executor.execute(cmd)
            
            status = "SUCCESS" if result.success else "FAILED"
            console.print(f"🚁 Drone: {drone_id} → {incident_id} "
                         f"({reason}) - {status}")
            
            return f"Drone {drone_id} assignment: {status}"

    return DynamicDroneAssignmentTool()


def create_incident_update_tool():
    """Create IncidentUpdateTool with dynamic description."""
    actual_ids = get_actual_service_ids()
    available_incidents = actual_ids.get('incidents', 
                                        ['E-1001', 'E-1002', 'E-1003', 
                                         'E-1004'])
    
    class DynamicIncidentUpdateTool(BaseTool):
        name: str = "update_incident_status"
        description: str = (
            f"Update the status of an emergency incident.\n"
            f"Available incidents: {', '.join(available_incidents)}\n"
            f"Valid statuses: 'active', 'assigned', 'in_progress', 'resolved'"
        )
        
        def _run(self, incident_id: str, status: str, reason: str) -> str:
            cmd = Command(
                service=ServiceType.EMERGENCY,
                action="update_incident",
                parameters={"incident_id": incident_id, "status": status}
            )
            
            executor = CommandExecutor()
            result = executor.execute(cmd)
            
            status_result = "SUCCESS" if result.success else "FAILED"
            console.print(f"🚨 Incident: {incident_id} → {status} "
                         f"({reason}) - {status_result}")
            
            return f"Incident {incident_id} update: {status_result}"

    return DynamicIncidentUpdateTool()


def create_traffic_redirection_tool():
    """Create TrafficRedirectionTool with dynamic description."""
    actual_ids = get_actual_service_ids()
    available_sectors = actual_ids.get('traffic_sectors', 
                                      ['S001', 'S002', 'S003'])
    
    class DynamicTrafficRedirectionTool(BaseTool):
        name: str = "redirect_traffic"
        description: str = (
            f"Redirect traffic in congested sectors.\n"
            f"Available sectors: {', '.join(available_sectors)}"
        )
        
        def _run(self, sector_id: str, target_reduction: float, 
                reason: str) -> str:
            cmd = Command(
                service=ServiceType.TRAFFIC,
                action="redirect",
                parameters={"sector_id": sector_id, 
                           "target_reduction": target_reduction}
            )
            
            executor = CommandExecutor()
            result = executor.execute(cmd)
            
            status = "SUCCESS" if result.success else "FAILED"
            console.print(f"🚦 Traffic: {sector_id} → "
                         f"{target_reduction:.1%} reduction "
                         f"({reason}) - {status}")
            
            return f"Traffic redirection in sector {sector_id}: {status}"

    return DynamicTrafficRedirectionTool()


def create_route_blocking_tool():
    """Create RouteBlockingTool with dynamic description."""
    actual_ids = get_actual_service_ids()
    available_sectors = actual_ids.get('traffic_sectors', 
                                      ['S001', 'S002', 'S003'])
    
    class DynamicRouteBlockingTool(BaseTool):
        name: str = "block_route"
        description: str = (
            f"Block a route for emergency access.\n"
            f"Available sectors: {', '.join(available_sectors)}"
        )
        
        def _run(self, sector_id: str, duration_minutes: int, 
                reason: str) -> str:
            cmd = Command(
                service=ServiceType.TRAFFIC,
                action="block_route",
                parameters={
                    "sector": sector_id,
                    "reason": reason,
                    "duration_minutes": duration_minutes
                }
            )
            
            executor = CommandExecutor()
            result = executor.execute(cmd)
            
            status = "SUCCESS" if result.success else "FAILED"
            console.print(f"🚧 Route: {sector_id} blocked for "
                         f"{duration_minutes}min ({reason}) - {status}")
            
            return f"Route blocking in sector {sector_id}: {status}"
    
    return DynamicRouteBlockingTool()


# Agent creation functions (copied from morning session)
def create_grid_agent():
    """Create the Grid Management Specialist Agent."""
    actual_ids = get_actual_service_ids()
    available_zones = actual_ids.get('grid_zones', ['Z001', 'Z002', 'Z003'])
    
    try:
        response = requests.get(f"{SERVICE_URLS['grid']}/service/info", 
                               timeout=5)
        if response.status_code == 200:
            available_infrastructure = [
                "hospital", "police", "emergency_services", 
                "water_treatment", "data_center", "emergency_shelter"
            ]
        else:
            available_infrastructure = [
                "hospital", "police", "emergency_services"
            ]
    except Exception:
        available_infrastructure = ["hospital", "police", "emergency_services"]
    
    grid_specialist = Agent(
        role="Power Grid Stability Specialist",
        goal=f"Prevent grid failures through capacity management and "
             f"infrastructure prioritization across {len(available_zones)} "
             f"zones",
        backstory=(
            f"Senior grid engineer specializing in load balancing and "
            f"infrastructure prioritization.\n\n"
            f"Available resources:\n"
            f"• Grid zones: {', '.join(available_zones)}\n"
            f"• Critical infrastructure: "
            f"{', '.join(available_infrastructure)}\n\n"
            f"Decision criteria:\n"
            f"• Reduce capacity for any zone >90% load to 0.8 or lower\n"
            f"• Set all critical infrastructure to 'critical' priority\n"
            f"• Take 6+ actions total (zone adjustments + priorities)\n"
            f"• Use actual resource IDs only"
        ),
        tools=[create_grid_zone_adjustment_tool(), 
               create_infrastructure_priority_tool()],
        verbose=True,
        allow_delegation=False
    )
    
    return grid_specialist


def create_emergency_agent():
    """Create emergency management agent."""
    actual_ids = get_actual_service_ids()
    available_drones = actual_ids.get('drones', 
                                     ['D001', 'D002', 'D003', 'D004'])
    available_incidents = actual_ids.get('incidents', 
                                        ['E-1001', 'E-1002', 'E-1003', 
                                         'E-1004'])
    
    emergency_agent = Agent(
        role="Emergency Response Coordinator",
        goal=f"Optimize drone deployment and incident management across "
             f"{len(available_drones)} drones and "
             f"{len(available_incidents)} incidents",
        backstory=(
            f"Emergency coordinator specializing in resource allocation "
            f"and incident response.\n\n"
            f"Available resources:\n"
            f"• Drones: {', '.join(available_drones)}\n"
            f"• Incidents: {', '.join(available_incidents)}\n\n"
            f"Decision criteria:\n"
            f"• Assign all drones to incidents based on urgency\n"
            f"• Update incident statuses to track progress\n"
            f"• Take 6+ actions total (assignments + status updates)\n"
            f"• Use actual resource IDs only"
        ),
        tools=[create_drone_assignment_tool(), create_incident_update_tool()],
        verbose=True,
        allow_delegation=False
    )
    
    return emergency_agent


def create_traffic_agent():
    """Create traffic management agent."""
    actual_ids = get_actual_service_ids()
    available_sectors = actual_ids.get('traffic_sectors', 
                                      ['S001', 'S002', 'S003'])
    
    traffic_agent = Agent(
        role="Traffic Management Specialist", 
        goal=f"Optimize traffic flow and emergency access across "
             f"{len(available_sectors)} sectors",
        backstory=(
            f"Traffic engineer specializing in congestion management "
            f"and emergency routing.\n\n"
            f"Available resources:\n"
            f"• Traffic sectors: {', '.join(available_sectors)}\n\n"
            f"Decision criteria:\n"
            f"• Redirect traffic in sectors >70% congestion\n"
            f"• Block routes for emergency corridor creation\n"
            f"• Take 4+ actions total (redirections + blockings)\n"
            f"• Use actual resource IDs only"
        ),
        tools=[create_traffic_redirection_tool(), 
               create_route_blocking_tool()],
        verbose=True,
        allow_delegation=False
    )
    
    return traffic_agent


def create_crisis_manager_agent():
    """Create a manager agent that coordinates specialist agents."""
    crisis_manager = Agent(
        role="Crisis Management Coordinator",
        goal="Coordinate specialist agents to achieve comprehensive crisis "
             "response with 15+ total actions",
        backstory="""Senior crisis management coordinator with expertise in 
        multi-agent coordination.
        
        Team specialists:
        - Grid Management: Handles power grid stability and priorities
        - Emergency Response: Manages drone assignments and incidents  
        - Traffic Management: Optimizes traffic flow and emergency access
        
        Coordination requirements:
        • Grid team: Execute 6+ actions (zone adjustments + priorities)
        • Emergency team: Execute 6+ actions (drone assignments + updates)
        • Traffic team: Execute 4+ actions (redirections + route blocks)
        • Total target: 15+ coordinated actions across all services""",
        tools=[],  # Manager agents cannot have tools in hierarchical process
        verbose=True,
        allow_delegation=True,  # Key: Enables hierarchical management
        llm="gpt-4o"  # Use high-capability model for manager
    )
    
    return crisis_manager


# Task creation functions
def create_grid_task(grid_agent):
    """Create a task specifically for the Grid agent."""
    actual_ids = get_actual_service_ids()
    available_zones = actual_ids.get('grid_zones', ['Z001', 'Z002', 'Z003'])
    
    try:
        response = requests.get(f"{SERVICE_URLS['grid']}/service/info", 
                               timeout=5)
        if response.status_code == 200:
            available_infrastructure = [
                "hospital", "police", "emergency_services", 
                "water_treatment", "data_center", "emergency_shelter"
            ]
        else:
            available_infrastructure = [
                "hospital", "police", "emergency_services"
            ]
    except Exception:
        available_infrastructure = ["hospital", "police", "emergency_services"]
    
    grid_task = Task(
        description=(
            f"Heat wave crisis: Grid zones approaching overload thresholds."
            f"\n\nRequired actions:\n"
            f"1. Check all zones: {', '.join(available_zones)}\n"
            f"2. Reduce capacity to 0.8 for any zone >90% load\n"
            f"3. Set all critical infrastructure to 'critical' priority: "
            f"{', '.join(available_infrastructure)}\n\n"
            f"Success criteria: Execute 6+ total actions minimum\n"
            f"Use only the resource IDs listed above"
        ),
        agent=grid_agent,
        expected_output="Grid management plan with capacity adjustments "
                       "and infrastructure priorities",
        output_pydantic=GridManagementPlan
    )
    
    return grid_task


def create_emergency_task(emergency_agent):
    """Create a task specifically for the Emergency agent."""
    actual_ids = get_actual_service_ids()
    available_drones = actual_ids.get('drones', 
                                     ['D001', 'D002', 'D003', 'D004'])
    available_incidents = actual_ids.get('incidents', 
                                        ['E-1001', 'E-1002', 'E-1003', 
                                         'E-1004'])
    
    emergency_task = Task(
        description=(
            f"Heat wave emergency with multiple casualties requiring "
            f"drone response.\n\n"
            f"Available resources:\n"
            f"• Drones: {', '.join(available_drones)}\n"
            f"• Incidents: {', '.join(available_incidents)}\n\n"
            f"Required actions:\n"
            f"1. Assign all drones to incidents by urgency priority\n"
            f"2. Update incident statuses to 'assigned' or 'in_progress'\n\n"
            f"Success criteria: Execute 6+ total actions minimum\n"
            f"Use only the resource IDs listed above"
        ),
        agent=emergency_agent,
        expected_output="Emergency response plan with drone assignments "
                       "and incident tracking",
        output_pydantic=EmergencyResponsePlan
    )
    return emergency_task


def create_traffic_task(traffic_agent):
    """Create a task specifically for the Traffic agent."""
    actual_ids = get_actual_service_ids()
    available_sectors = actual_ids.get('traffic_sectors', 
                                      ['S001', 'S002', 'S003'])
    
    traffic_task = Task(
        description=(
            f"Heat wave crisis: Traffic congestion blocking emergency "
            f"vehicle access.\n\n"
            f"Available sectors: {', '.join(available_sectors)}\n\n"
            f"Required actions:\n"
            f"1. Redirect traffic in sectors >70% congestion "
            f"(reduce by 40-50%)\n"
            f"2. Block 1-2 routes for dedicated emergency corridors "
            f"(30-60 min)\n\n"
            f"Success criteria: Execute 4+ total actions minimum\n"
            f"Use only the sector IDs listed above"
        ),
        agent=traffic_agent,
        expected_output="Traffic management plan with redirections "
                       "and route blocks",
        output_pydantic=TrafficManagementPlan
    )
    return traffic_task


# Baseline agent creation functions
def create_baseline_grid_agent():
    """Create baseline grid agent for comparison."""
    return create_grid_agent()


def create_baseline_emergency_agent():
    """Create baseline emergency agent for comparison."""
    return create_emergency_agent()


def create_baseline_traffic_agent():
    """Create baseline traffic agent for comparison."""
    return create_traffic_agent()


def create_baseline_agent_system():
    """Create baseline agent system for benchmarking."""
    grid_agent = create_baseline_grid_agent()
    emergency_agent = create_baseline_emergency_agent()
    traffic_agent = create_baseline_traffic_agent()
    
    return Crew(
        agents=[grid_agent, emergency_agent, traffic_agent],
        tasks=[
            Task(description="Handle grid stability management", 
                 agent=grid_agent, expected_output="Grid management actions"),
            Task(description="Handle emergency response coordination", 
                 agent=emergency_agent, 
                 expected_output="Emergency response actions"),
            Task(description="Handle traffic flow optimization", 
                 agent=traffic_agent, 
                 expected_output="Traffic management actions")
        ],
        process=Process.sequential,
        verbose=True
    )


# Optimized agent creation functions
def create_optimized_grid_agent():
    """Create optimized grid agent."""
    return create_grid_agent()


def create_optimized_emergency_agent():
    """Create optimized emergency agent."""
    return create_emergency_agent()


def create_optimized_traffic_agent():
    """Create optimized traffic agent."""
    return create_traffic_agent()


def create_optimized_agent_tasks(grid_agent, emergency_agent, traffic_agent, 
                                scenario):
    """Create optimized tasks for agents."""
    return [
        create_grid_task(grid_agent),
        create_emergency_task(emergency_agent),
        create_traffic_task(traffic_agent)
    ]


# Scenario creation functions
def create_evaluation_scenarios() -> Dict[str, ScenarioDefinition]:
    """Create 5 diverse scenarios for comprehensive optimization testing."""
    console.print("🔍 Creating diverse evaluation scenarios...")
    actual_ids = get_actual_service_ids()
    
    grid_zones = actual_ids.get('grid_zones', ["zone_a", "zone_b", "zone_c"])
    available_drones = actual_ids.get('drones', 
                                     ["drone_1", "drone_2", "drone_3", 
                                      "drone_4"])
    incident_ids = actual_ids.get('incidents', 
                                 ["incident_1", "incident_2", "incident_3", 
                                  "incident_4"])
    traffic_sectors = actual_ids.get('traffic_sectors', 
                                    ["S001", "S002", "S003"])
    
    scenarios = {}
    
    # Scenario 1: Heat Wave Crisis (Original)
    scenarios["heat_wave"] = ScenarioDefinition(
        name="Heat Wave Crisis",
        description="An extreme heat wave causing severe grid stress",
        initial_state=ServiceState(
            zones={
                grid_zones[0]: ZoneState(
                    id=grid_zones[0], name="Downtown", capacity=1.0,
                    current_load=0.98, stability=0.4, is_critical=True
                )
            },
            incidents=[
                IncidentState(
                    id=incident_ids[0] if incident_ids else "incident_1",
                    description="Major power outage affecting hospital",
                    location=grid_zones[0], urgency=0.99
                )
            ],
            drones=[
                DroneState(
                    id=available_drones[0] if available_drones else "drone_1",
                    name="Alpha", capabilities=["medical", "surveillance"], 
                    speed=1.5
                )
            ],
            traffic={
                traffic_sectors[0]: TrafficState(
                    zone_id=traffic_sectors[0], congestion=0.9, blocked=False,
                    description="Severe traffic congestion in downtown"
                )
            }
        ),
        success_criteria=SuccessCriteria(
            name="Heat Wave Resolution", 
            description="Resolve heat wave crisis",
            metrics={"grid_stability": 0.8, "incident_response": 0.9},
            thresholds={"max_temperature": 46.0, "min_power": 0.7}
        ),
        optimal_commands=[],
        command_weights={"grid": 0.5, "emergency": 0.4, "traffic": 0.1}
    )
    
    # Scenario 2: Cyber Attack on Grid
    scenarios["cyber_attack"] = ScenarioDefinition(
        name="Cyber Attack on Infrastructure",
        description="Coordinated cyber attack targeting power grid systems",
        initial_state=ServiceState(
            zones={
                grid_zones[0]: ZoneState(
                    id=grid_zones[0], name="Financial District", capacity=1.0,
                    current_load=0.85, stability=0.2, is_critical=True
                ),
                (grid_zones[1] if len(grid_zones) > 1 
                 else "zone_b"): ZoneState(
                    id=grid_zones[1] if len(grid_zones) > 1 else "zone_b",
                    name="Tech Hub", capacity=1.0, current_load=0.90, 
                    stability=0.3
                )
            },
            incidents=[
                IncidentState(
                    id=incident_ids[0] if incident_ids else "incident_1",
                    description="Critical infrastructure systems compromised",
                    location=grid_zones[0], urgency=0.95
                ),
                IncidentState(
                    id=(incident_ids[1] if len(incident_ids) > 1 
                        else "incident_2"),
                    description="Data center power grid under attack",
                    location=(grid_zones[1] if len(grid_zones) > 1 
                             else "zone_b"),
                    urgency=0.9
                )
            ],
            drones=[
                DroneState(
                    id=available_drones[0] if available_drones else "drone_1",
                    name="Security Alpha", 
                    capabilities=["surveillance", "cyber"], speed=1.8
                ),
                DroneState(
                    id=(available_drones[1] if len(available_drones) > 1 
                        else "drone_2"),
                    name="Patrol Beta", 
                    capabilities=["surveillance", "power"], speed=1.4
                )
            ],
            traffic={
                traffic_sectors[0]: TrafficState(
                    zone_id=traffic_sectors[0], congestion=0.6, blocked=False,
                    description="Increased security checkpoints causing delays"
                )
            }
        ),
        success_criteria=SuccessCriteria(
            name="Cyber Attack Mitigation", 
            description="Secure grid and restore stability",
            metrics={"grid_stability": 0.85, "incident_response": 0.95, 
                    "security": 0.9},
            thresholds={"min_power": 0.8, "max_response_time": 180}
        ),
        optimal_commands=[],
        command_weights={"grid": 0.6, "emergency": 0.35, "traffic": 0.05}
    )
    
    # Continue with other scenarios...
    scenarios["earthquake"] = _create_earthquake_scenario(
        grid_zones, incident_ids, available_drones, traffic_sectors)
    scenarios["festival"] = _create_festival_scenario(
        grid_zones, incident_ids, available_drones, traffic_sectors)
    scenarios["complex_crisis"] = _create_complex_crisis_scenario(
        grid_zones, incident_ids, available_drones, traffic_sectors)
    
    console.print(f"✅ Created {len(scenarios)} diverse evaluation scenarios")
    return scenarios


def _create_earthquake_scenario(grid_zones, incident_ids, available_drones, 
                               traffic_sectors):
    """Create earthquake scenario to keep function length manageable."""
    return ScenarioDefinition(
        name="Major Earthquake Response",
        description="7.2 magnitude earthquake causing infrastructure damage",
        initial_state=ServiceState(
            zones={
                grid_zones[0]: ZoneState(
                    id=grid_zones[0], name="Central District", capacity=0.6,
                    current_load=0.95, stability=0.1, is_critical=True
                )
            },
            incidents=[
                IncidentState(
                    id=incident_ids[0] if incident_ids else "incident_1",
                    description="Building collapse with trapped victims",
                    location=grid_zones[0], urgency=1.0
                )
            ],
            drones=[
                DroneState(
                    id=available_drones[0] if available_drones else "drone_1",
                    name="Rescue Alpha", 
                    capabilities=["search_rescue", "medical"], speed=1.2
                )
            ],
            traffic={
                traffic_sectors[0]: TrafficState(
                    zone_id=traffic_sectors[0], congestion=0.95, blocked=True,
                    description="Road closures due to earthquake damage"
                )
            }
        ),
        success_criteria=SuccessCriteria(
            name="Earthquake Recovery", 
            description="Save lives and restore basic services",
            metrics={"incident_response": 0.95, "grid_stability": 0.6},
            thresholds={"max_response_time": 120, "min_power": 0.5}
        ),
        optimal_commands=[],
        command_weights={"emergency": 0.6, "grid": 0.25, "traffic": 0.15}
    )


def _create_festival_scenario(grid_zones, incident_ids, available_drones, 
                             traffic_sectors):
    """Create festival scenario to keep function length manageable."""
    return ScenarioDefinition(
        name="Large Festival Emergency",
        description="Major music festival with multiple emergencies",
        initial_state=ServiceState(
            zones={
                grid_zones[0]: ZoneState(
                    id=grid_zones[0], name="Festival Grounds", capacity=1.0,
                    current_load=0.95, stability=0.7, is_critical=False
                )
            },
            incidents=[
                IncidentState(
                    id=incident_ids[0] if incident_ids else "incident_1",
                    description="Stage collapse with multiple injuries",
                    location=grid_zones[0], urgency=0.98
                )
            ],
            drones=[
                DroneState(
                    id=available_drones[0] if available_drones else "drone_1",
                    name="MedEvac Alpha", 
                    capabilities=["medical", "transport"], speed=2.0
                )
            ],
            traffic={
                traffic_sectors[0]: TrafficState(
                    zone_id=traffic_sectors[0], congestion=0.95, blocked=False,
                    description="Festival evacuees causing congestion"
                )
            }
        ),
        success_criteria=SuccessCriteria(
            name="Festival Emergency Response", 
            description="Manage crowd safety and medical emergencies",
            metrics={"incident_response": 0.9, "crowd_safety": 0.95},
            thresholds={"max_response_time": 60, "max_casualties": 0}
        ),
        optimal_commands=[],
        command_weights={"emergency": 0.7, "traffic": 0.25, "grid": 0.05}
    )


def _create_complex_crisis_scenario(grid_zones, incident_ids, available_drones, 
                                   traffic_sectors):
    """Create complex crisis scenario to keep function length manageable."""
    return ScenarioDefinition(
        name="Multi-Service Complex Crisis",
        description="Simultaneous grid failure, chemical spill, "
                   "traffic system malfunction",
        initial_state=ServiceState(
            zones={
                grid_zones[0]: ZoneState(
                    id=grid_zones[0], name="Industrial Complex", capacity=0.8,
                    current_load=0.95, stability=0.2, is_critical=True
                )
            },
            incidents=[
                IncidentState(
                    id=incident_ids[0] if incident_ids else "incident_1",
                    description="Chemical plant explosion and toxic gas leak",
                    location=grid_zones[0], urgency=1.0
                )
            ],
            drones=[
                DroneState(
                    id=available_drones[0] if available_drones else "drone_1",
                    name="HazMat Alpha", 
                    capabilities=["hazmat", "surveillance"], speed=1.4
                )
            ],
            traffic={
                traffic_sectors[0]: TrafficState(
                    zone_id=traffic_sectors[0], congestion=0.99, blocked=True,
                    description="Traffic system malfunction causing gridlock"
                )
            }
        ),
        success_criteria=SuccessCriteria(
            name="Complex Crisis Management", 
            description="Coordinate response across multiple crises",
            metrics={"incident_response": 0.85, "grid_stability": 0.7},
            thresholds={"max_response_time": 90, "min_power": 0.6}
        ),
        optimal_commands=[],
        command_weights={"emergency": 0.4, "grid": 0.35, "traffic": 0.25}
    )


def create_heat_wave_scenario_for_evaluation():
    """Create heat wave scenario for evaluation."""
    evaluation_scenarios = create_evaluation_scenarios()
    return evaluation_scenarios["heat_wave"]


# Missing workshop utility classes and functions

class BaselinePerformanceMeasurement:
    """Baseline performance measurement system for agent evaluation."""
    
    def __init__(self, evaluation_framework):
        self.evaluation_framework = evaluation_framework
        self.baseline_results = {}
        
    def measure_agent_performance(self, agent_system, scenario, test_name):
        """Measure baseline performance of an agent system."""
        console.print(f"📊 Measuring baseline: {test_name}")
        
        # Measure performance with timing
        result, response_time = self.evaluation_framework.measure_response_time(
            agent_system.kickoff,
            inputs={
                "scenario_name": scenario.name,
                "scenario_description": scenario.description
            }
        )
        
        # Create comprehensive evaluation
        metrics = self.evaluation_framework.create_comprehensive_evaluation(
            result, scenario, response_time
        )
        
        # Store results
        self.baseline_results[test_name] = {
            "metrics": metrics,
            "response_time": response_time,
            "result": result
        }
        
        return metrics, response_time, result
    
    def get_baseline_summary(self):
        """Get summary of all baseline measurements."""
        if not self.baseline_results:
            return {}
            
        avg_score = sum(data["metrics"].overall_score() 
                       for data in self.baseline_results.values()) / len(self.baseline_results)
        avg_time = sum(data["response_time"] 
                      for data in self.baseline_results.values()) / len(self.baseline_results)
        
        return {
            "average_score": avg_score,
            "average_time": avg_time,
            "total_tests": len(self.baseline_results),
            "results": self.baseline_results
        }


# Workshop interaction utilities
def create_workshop_checkpoint(phase_name: str, instructions: str):
    """Create an interactive checkpoint for workshop participants."""
    console.print(Panel(
        f"🎓 **WORKSHOP CHECKPOINT: {phase_name}**\n\n"
        f"**👨‍💻 YOUR TURN TO EXPERIMENT:**\n"
        f"{instructions}\n\n"
        f"**📝 Take notes on:**\n"
        f"• What you observe\n"
        f"• How performance changes\n"
        f"• What you would optimize next\n\n"
        f"**Continue with the workshop...**",
        title="Interactive Workshop",
        border_style="yellow"
    ))


def create_discussion_prompt(topic: str, questions: List[str]):
    """Create a discussion prompt for workshop participants."""
    console.print(Panel(
        f"💬 **DISCUSSION POINT: {topic}**\n\n" +
        "\n".join([f"• {q}" for q in questions]) + "\n\n"
        f"**Take 2-3 minutes to discuss with your team or reflect on these questions.**",
        title="Workshop Discussion",
        border_style="cyan"
    ))


def display_workshop_progress(current_phase: int, total_phases: int, phase_name: str):
    """Display workshop progress indicator."""
    progress = current_phase / total_phases
    filled = int(progress * 20)
    bar = "█" * filled + "░" * (20 - filled)
    
    console.print(f"\n📊 Workshop Progress: [{bar}] {progress:.0%} - {phase_name}\n")


# Additional utility functions
def create_performance_comparison_table(results_dict: Dict[str, Any], title: str):
    """Create a comparison table for different approaches."""
    from rich.table import Table
    
    table = Table(title=title)
    table.add_column("Approach", style="cyan", no_wrap=True)
    table.add_column("Success Rate", style="green")
    table.add_column("Avg Response Time", style="yellow")
    table.add_column("Key Insight", style="blue")
    
    for approach, data in results_dict.items():
        if isinstance(data, dict) and "success_rate" in data:
            table.add_row(
                approach.replace("_", " ").title(),
                f"{data['success_rate']:.1%}",
                f"{data.get('avg_response_time', 0):.1f}ms",
                data.get('insight', 'Performance measured')
            )
    
    console.print(table)


def create_workshop_timer(duration_minutes: int, activity: str):
    """Create a workshop timer for timed activities."""
    console.print(Panel(
        f"⏱️ **TIMED ACTIVITY: {activity}**\n\n"
        f"Duration: {duration_minutes} minutes\n"
        f"Start time: {datetime.now().strftime('%H:%M')}\n"
        f"End time: {(datetime.now() + timedelta(minutes=duration_minutes)).strftime('%H:%M')}\n\n"
        f"Use this time to experiment with the code!",
        title="Workshop Timer",
        border_style="red"
    ))


# Workshop scenario variations for experimentation
def create_workshop_scenario_variations():
    """Create scenario variations for workshop experimentation."""
    base_scenarios = create_evaluation_scenarios()
    
    # Add workshop-specific variations
    variations = {}
    
    for name, scenario in base_scenarios.items():
        # Create an "extreme" variation
        extreme_scenario = ScenarioDefinition(
            name=f"{scenario.name} - EXTREME",
            description=f"EXTREME VERSION: {scenario.description}",
            initial_state=scenario.initial_state,
            success_criteria=scenario.success_criteria,
            optimal_commands=scenario.optimal_commands,
            command_weights=scenario.command_weights
        )
        
        # Modify some parameters to make it more challenging
        if hasattr(extreme_scenario.initial_state, 'zones'):
            for zone in extreme_scenario.initial_state.zones.values():
                if hasattr(zone, 'current_load'):
                    zone.current_load = min(0.99, zone.current_load + 0.1)
                if hasattr(zone, 'stability'):
                    zone.stability = max(0.1, zone.stability - 0.2)
        
        variations[f"{name}_extreme"] = extreme_scenario
    
    return {**base_scenarios, **variations}


def create_hands_on_exercise(exercise_name: str, task_description: str, 
                           expected_outcome: str, difficulty: str = "Medium"):
    """Create a hands-on exercise for workshop participants."""
    difficulty_colors = {
        "Easy": "green",
        "Medium": "yellow", 
        "Hard": "red"
    }
    
    console.print(Panel(
        f"🛠️ **HANDS-ON EXERCISE: {exercise_name}**\n\n"
        f"**Difficulty:** {difficulty}\n\n"
        f"**Your Task:**\n{task_description}\n\n"
        f"**Expected Outcome:**\n{expected_outcome}\n\n"
        f"**💡 Hint:** Look at the code above and modify the parameters!\n"
        f"**🎯 Goal:** Learn by experimenting and observing the results.",
        title="Workshop Exercise",
        border_style=difficulty_colors.get(difficulty, "blue")
    ))


# Workshop completion utilities
def display_workshop_summary(results: Dict[str, Any]):
    """Display comprehensive workshop summary."""
    console.print(Panel(
        f"🎉 **WORKSHOP COMPLETION SUMMARY** 🎉\n\n"
        f"**🏆 Key Achievements:**\n"
        f"• Built comprehensive evaluation frameworks\n"
        f"• Implemented latency optimization strategies\n" 
        f"• Created dynamic model selection systems\n"
        f"• Integrated human feedback loops\n"
        f"• Mastered agent observability techniques\n\n"
        f"**📊 Performance Improvements:**\n"
        f"• Average optimization gain: {results.get('avg_improvement', 'N/A')}\n"
        f"• Best strategy: {results.get('best_strategy', 'N/A')}\n"
        f"• Total scenarios tested: {results.get('scenarios_tested', 'N/A')}\n\n"
        f"**🚀 Production Readiness: {results.get('production_ready', 'ACHIEVED')}**",
        title="Workshop Complete!",
        border_style="green"
    ))


def create_workshop_reflection_questions():
    """Create reflection questions for workshop wrap-up."""
    questions = [
        "What was the most surprising insight about agent performance?",
        "Which optimization strategy would you implement first in production?",
        "How would you modify the evaluation framework for your use case?",
        "What additional metrics would be valuable for your domain?",
        "How does this compare to traditional rule-based approaches you've used?"
    ]
    
    console.print(Panel(
        "🤔 **FINAL REFLECTION**\n\n" +
        "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]) + "\n\n" +
        "Take 5 minutes to reflect on these questions and discuss with your team.",
        title="Workshop Reflection",
        border_style="purple"
    )) 

@weave.op
def execute_rule_commands(commands):
    """
    Execute a list of commands using the CommandExecutor and track their success.
    
    Args:
        commands: List of command dictionaries containing service, action, and parameters
        
    Returns:
        float: Success rate of command execution (0.0 to 1.0)
    """
    executor = CommandExecutor()
    execution_results = []

    for i, command in enumerate(commands, 1):
        console.print(f"\n📏 Command {i}: {command.get('rule', 'No rule description')}")
        
        try:
            cmd = Command(
                service=ServiceType(command["service"]),
                action=command["action"],
                parameters=command.get("parameters", {})
            )
            result = executor.execute(cmd)
            execution_results.append(result.success)
            
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            console.print(f"  {status}: {command['service']}.{command['action']}")
            
            if not result.success:
                console.print(f"    Error: {result.error}")
                
        except Exception as e:
            console.print(f"  ❌ EXECUTION ERROR: {e}")
            execution_results.append(False)

    success_rate = (sum(execution_results) / len(execution_results) 
                   if execution_results else 0)
    return success_rate