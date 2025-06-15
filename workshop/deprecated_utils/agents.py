"""
Agent creation and task management for the SENTINEL GRID workshop.

This module provides functions to create specialized agents and their tasks
for different roles in the SENTINEL GRID system.
"""

import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process

from workshop.deprecated_utils.agent_profiles import AgentProfile
from workshop.deprecated_utils.service_tools import (
    create_weather_tools, create_grid_tools, create_emergency_tools, 
    create_traffic_tools
)

# Configure logging
log = logging.getLogger("agents")

# Create Analyst Agents
def create_weather_analyst():
    """Create a weather analyst agent."""
    profile = AgentProfile.weather_analyst()
    
    return Agent(
        role=profile.role,
        goal="Analyze weather conditions and their impact on city services",
        backstory="""You are an expert meteorologist working for the SENTINEL GRID system.
        Your job is to monitor weather conditions, analyze trends, and identify potential
        impacts on city services. You have a deep understanding of how weather affects
        power grid stability, traffic conditions, and emergency response capabilities.""",
        verbose=True,
        tools=profile.discover_tools(),
        allow_delegation=False
    )

def create_grid_analyst():
    """Create a grid analyst agent."""
    profile = AgentProfile.grid_analyst()
    
    return Agent(
        role=profile.role,
        goal="Monitor power grid status and identify potential issues",
        backstory="""You are a power systems engineer working for the SENTINEL GRID system.
        Your job is to monitor the city's power grid, identify zones approaching capacity
        limits, and analyze load patterns. You understand the complex relationships between
        power distribution, infrastructure priorities, and city services.""",
        verbose=True,
        tools=profile.discover_tools(),
        allow_delegation=False
    )

# Create Executor Agents
def create_grid_executor():
    """Create a grid executor agent."""
    # Create a custom profile for grid executor
    profile = AgentProfile(
        role="Grid Executor",
        capabilities={
            "grid_expertise": True,
            "grid_control": True,
        },
        preferred_tags=["actuation", "grid", "control", "power"],
        required_capabilities={
            "manages_power": True
        }
    )
    
    return Agent(
        role=profile.role,
        goal="Optimize power grid distribution and prevent outages",
        backstory="""You are a power grid operator for the SENTINEL GRID system.
        Your job is to make real-time adjustments to the power grid, manage zone
        capacities, and set infrastructure priorities to prevent outages and ensure
        critical services have power during emergencies.""",
        verbose=True,
        tools=profile.discover_tools(),
        allow_delegation=False
    )

def create_emergency_executor():
    """Create an emergency executor agent."""
    profile = AgentProfile.emergency_executor()
    
    return Agent(
        role=profile.role,
        goal="Efficiently allocate emergency resources to incidents",
        backstory="""You are an emergency response coordinator for the SENTINEL GRID system.
        Your job is to assign drones to incidents, manage response priorities, and ensure
        efficient allocation of limited emergency resources during crisis situations.""",
        verbose=True,
        tools=profile.discover_tools(),
        allow_delegation=False
    )

def create_traffic_manager():
    """Create a traffic manager agent."""
    profile = AgentProfile.traffic_manager()
    
    return Agent(
        role=profile.role,
        goal="Optimize traffic flow and manage routing during emergencies",
        backstory="""You are a traffic control specialist for the SENTINEL GRID system.
        Your job is to monitor traffic conditions, identify congestion points, and
        create efficient routes for emergency vehicles. You understand how weather and
        incidents affect traffic patterns across the city.""",
        verbose=True,
        tools=profile.discover_tools(),
        allow_delegation=False
    )

# Create Coordinator Agent
def create_city_coordinator():
    """Create a city coordinator agent."""
    profile = AgentProfile.city_coordinator()
    
    return Agent(
        role=profile.role,
        goal="Coordinate responses across all city services during emergencies",
        backstory="""You are the chief coordinator for the SENTINEL GRID system.
        Your job is to oversee all city services, coordinate responses during emergencies,
        and make high-level decisions that balance the needs of different systems.
        You understand how actions in one service affect others and can prioritize
        responses based on overall city needs.""",
        verbose=True,
        tools=profile.discover_tools(),
        allow_delegation=True
    )

# Define structured output schemas using Pydantic
class WeatherAnalysisOutput(BaseModel):
    """Structured output for weather analysis."""
    current_conditions: Dict[str, Any] = Field(description="Current weather conditions")
    forecast: List[Dict[str, Any]] = Field(
        description="Weather forecast for the next 24 hours"
    )
    severity: str = Field(description="Weather severity (low, medium, high)")
    affected_services: List[Dict[str, Any]] = Field(
        description="Services affected by the weather"
    )
    recommendations: List[Dict[str, Any]] = Field(
        description="Recommended actions in format: {service, action, parameters}"
    )

class GridAnalysisOutput(BaseModel):
    """Structured output for grid analysis."""
    critical_zones: List[Dict[str, Any]] = Field(
        description="Zones approaching capacity limits"
    )
    stable_zones: List[Dict[str, Any]] = Field(
        description="Zones with stable capacity"
    )
    overall_stability: float = Field(
        description="Overall grid stability (0.0-1.0)"
    )
    recommendations: List[Dict[str, Any]] = Field(
        description="Recommended actions in format: {service, action, parameters}"
    )

class EmergencyResponseOutput(BaseModel):
    """Structured output for emergency response planning."""
    prioritized_incidents: List[Dict[str, Any]] = Field(
        description="Incidents prioritized for response"
    )
    drone_assignments: List[Dict[str, Any]] = Field(
        description="Drone assignments in format: {drone_id, incident_id}"
    )
    unassigned_incidents: List[Dict[str, Any]] = Field(
        description="Incidents without drone coverage"
    )
    alternative_responses: List[Dict[str, Any]] = Field(
        description="Alternative response plans for unassigned incidents"
    )

class CoordinationPlanOutput(BaseModel):
    """Structured output for coordination planning."""
    situation_assessment: Dict[str, Any] = Field(
        description="Overall assessment of the situation"
    )
    service_priorities: Dict[str, float] = Field(
        description="Priority levels for each service (0.0-1.0)"
    )
    cross_service_actions: List[Dict[str, Any]] = Field(
        description="Actions that coordinate multiple services"
    )
    commands: List[Dict[str, Any]] = Field(
        description="List of commands to execute in format: {service, action, parameters}"
    )

# Task creation functions
def create_weather_analysis_task(agent, scenario_description):
    """Create a weather analysis task."""
    return Task(
        description=f"""Analyze current weather conditions in the context of: {scenario_description}
        
        Your analysis should include:
        1. Current weather conditions and forecast
        2. Weather severity assessment (low, medium, high)
        3. Identification of services likely to be affected
        4. Specific recommendations based on your analysis
        
        Use the weather_conditions and weather_forecast tools to gather data.
        Your recommendations should be structured as commands in the format:
        {{
            "service": "service_name",
            "action": "action_name",
            "parameters": {{...}}
        }}
        """,
        expected_output=WeatherAnalysisOutput,
        agent=agent
    )

def create_grid_analysis_task(agent, scenario_description):
    """Create a grid analysis task."""
    return Task(
        description=f"""Analyze current grid status in the context of: {scenario_description}
        
        Your analysis should include:
        1. Identification of critical zones approaching capacity limits
        2. List of stable zones
        3. Assessment of overall grid stability
        4. Specific recommendations for zone capacity adjustments
        
        Use the grid_status tool to gather data.
        Your recommendations should be structured as commands in the format:
        {{
            "service": "service_name",
            "action": "action_name",
            "parameters": {{...}}
        }}
        """,
        expected_output=GridAnalysisOutput,
        agent=agent
    )

def create_emergency_response_task(agent, scenario_description):
    """Create an emergency response task."""
    return Task(
        description=f"""Create an emergency response plan in the context of: {scenario_description}
        
        Your plan should include:
        1. Prioritized list of incidents requiring response
        2. Drone assignments for critical incidents
        3. Identification of incidents without drone coverage
        4. Alternative response plans for incidents without drone coverage
        
        Use the emergency_incidents tool to gather data on current incidents.
        Your drone assignments should be structured as commands in the format:
        {{
            "service": "emergency",
            "action": "assign_drone",
            "parameters": {{
                "drone_id": "drone_X",
                "incident_id": "incident_Y"
            }}
        }}
        """,
        expected_output=EmergencyResponseOutput,
        agent=agent
    )

def create_coordination_task(agent, scenario_description, dependencies=None):
    """Create a coordination task."""
    return Task(
        description=f"""Create a comprehensive response plan for: {scenario_description}
        
        Your plan should include:
        1. Overall assessment of the situation
        2. Priority levels for each service
        3. Cross-service actions needed
        4. Specific commands to execute
        
        Your commands should be structured in the format:
        {{
            "service": "service_name",
            "action": "action_name",
            "parameters": {{...}}
        }}
        """,
        expected_output=CoordinationPlanOutput,
        agent=agent,
        dependencies=dependencies or []
    )

# Crew creation functions
def create_heat_wave_crew(
    scenario_description="A severe heat wave is causing increased power demand across the city"
):
    """
    Create a crew to handle a heat wave (grid surge) scenario.
    
    Args:
        scenario_description: Description of the heat wave scenario
        
    Returns:
        CrewAI Crew object
    """
    # Create agents
    weather_analyst = create_weather_analyst()
    grid_analyst = create_grid_analyst()
    grid_executor = create_grid_executor()
    emergency_executor = create_emergency_executor()
    city_coordinator = create_city_coordinator()
    
    # Create tasks
    weather_task = create_weather_analysis_task(weather_analyst, scenario_description)
    grid_task = create_grid_analysis_task(grid_analyst, scenario_description)
    emergency_task = create_emergency_response_task(
        emergency_executor, scenario_description
    )
    
    # Coordination task depends on previous analyses
    coordination_task = create_coordination_task(
        city_coordinator, 
        scenario_description,
        dependencies=[weather_task, grid_task, emergency_task]
    )
    
    # Create crew
    return Crew(
        agents=[
            weather_analyst,
            grid_analyst,
            grid_executor,
            emergency_executor,
            city_coordinator
        ],
        tasks=[
            weather_task,
            grid_task,
            emergency_task,
            coordination_task
        ],
        verbose=2,
        process=Process.sequential  # Tasks executed in sequence, respecting dependencies
    ) 