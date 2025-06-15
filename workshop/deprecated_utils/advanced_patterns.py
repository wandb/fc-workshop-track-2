"""
Advanced coordination patterns for multi-agent systems in SENTINEL GRID.

This module implements specialized coordination patterns for complex scenarios
such as hierarchical planning, real-time adaptation, and cascading effect modeling.
"""

import logging
from rich.console import Console
from rich.panel import Panel
from crewai import Crew, Task, Process

from workshop.agent_system import ScenarioType
from workshop.scenarios import create_infrastructure_collapse_scenario, run_scenario
from workshop.deprecated_utils.agents import (
    create_city_coordinator, create_weather_analyst, create_grid_analyst,
    create_traffic_manager, create_grid_executor, create_emergency_executor
)

# Initialize console and logging
console = Console()
log = logging.getLogger("advanced_patterns")

def create_hierarchical_crew(scenario_description, constraints=None):
    """
    Create a crew using hierarchical planning for complex scenarios.
    
    This pattern uses a three-level hierarchy:
    1. Strategic Coordinator (city-wide planning)
    2. Tactical Managers (service-specific planning)
    3. Operational Executors (action execution)
    
    Args:
        scenario_description: Description of the scenario
        constraints: Dict of scenario constraints
        
    Returns:
        CrewAI Crew object
    """
    # Create the agents for each level
    # Level 1: Strategic Coordinator
    coordinator = create_city_coordinator()
    
    # Level 2: Tactical Managers
    weather_analyst = create_weather_analyst()
    grid_analyst = create_grid_analyst()
    traffic_manager = create_traffic_manager()
    
    # Level 3: Operational Executors
    grid_executor = create_grid_executor()
    emergency_executor = create_emergency_executor()
    
    # Create tasks with appropriate dependencies
    # Level 1: Initial situation assessment
    situation_assessment = Task(
        description=f"""Perform an initial assessment of the city situation based on: {scenario_description}
        
        Your assessment should include:
        1. Overview of the current emergency
        2. Initial prioritization of city services
        3. Critical areas requiring immediate attention
        4. High-level strategic directives for service managers
        """,
        agent=coordinator
    )
    
    # Level 2: Service-specific analyses
    weather_analysis = Task(
        description=f"""Analyze current weather conditions in the context of: {scenario_description}
        
        Your analysis should include:
        1. Current weather conditions and forecast
        2. Weather severity assessment (low, medium, high)
        3. Identification of services likely to be affected
        4. Specific recommendations based on your analysis
        
        Your recommendations should be structured as commands.
        """,
        agent=weather_analyst,
        dependencies=[situation_assessment]
    )
    
    grid_analysis = Task(
        description=f"""Analyze current grid status in the context of: {scenario_description}
        
        Your analysis should include:
        1. Identification of critical zones approaching capacity limits
        2. List of stable zones
        3. Assessment of overall grid stability
        4. Specific recommendations for zone capacity adjustments
        
        Your recommendations should be structured as commands.
        """,
        agent=grid_analyst,
        dependencies=[situation_assessment]
    )
    
    traffic_analysis = Task(
        description=f"""Analyze current traffic conditions in the context of: {scenario_description}
        
        Your analysis should include:
        1. Identification of high-congestion areas
        2. Critical routes that need to be kept open
        3. Recommendations for traffic management
        
        Your recommendations should be structured as commands.
        """,
        agent=traffic_manager,
        dependencies=[situation_assessment]
    )
    
    # Level 3: Tactical planning
    tactical_plan = Task(
        description=f"""Create a tactical response plan based on service analyses for: {scenario_description}
        
        Incorporate the findings from weather, grid, and traffic analyses to create a comprehensive 
        plan that addresses the most critical issues while balancing needs across services.
        
        Your plan should include:
        1. Prioritized list of actions across all services
        2. Resource allocation directives
        3. Coordination requirements between services
        4. Contingency measures for worsening conditions
        
        Your plan should be structured as a list of commands to execute.
        """,
        agent=coordinator,
        dependencies=[weather_analysis, grid_analysis, traffic_analysis]
    )
    
    # Level 3: Execution tasks
    grid_execution = Task(
        description=f"""Execute grid management actions based on the tactical plan for: {scenario_description}
        
        Focus on:
        1. Implementing grid stability measures
        2. Managing power to critical infrastructure
        3. Preventing cascading failures
        
        Your actions should be structured as commands.
        """,
        agent=grid_executor,
        dependencies=[tactical_plan]
    )
    
    emergency_execution = Task(
        description=f"""Execute emergency response actions based on the tactical plan for: {scenario_description}
        
        Focus on:
        1. Deploying drones to priority incidents
        2. Managing response resources efficiently
        3. Coordinating with other services for access
        
        Your actions should be structured as commands.
        """,
        agent=emergency_executor,
        dependencies=[tactical_plan]
    )
    
    # Create final assessment task
    final_assessment = Task(
        description=f"""Evaluate the effectiveness of the response to: {scenario_description}
        
        Review the actions taken by all services and assess:
        1. Overall effectiveness of the response
        2. Areas that were well-managed
        3. Areas that need improvement
        4. Recommendations for future similar scenarios
        """,
        agent=coordinator,
        dependencies=[grid_execution, emergency_execution]
    )
    
    # Create and return crew
    return Crew(
        agents=[
            coordinator,
            weather_analyst,
            grid_analyst,
            traffic_manager,
            grid_executor,
            emergency_executor
        ],
        tasks=[
            situation_assessment,
            weather_analysis,
            grid_analysis,
            traffic_analysis,
            tactical_plan,
            grid_execution,
            emergency_execution,
            final_assessment
        ],
        verbose=2,
        process=Process.sequential
    )

def run_infrastructure_collapse_scenario():
    """
    Run the infrastructure collapse scenario using hierarchical planning.
    
    This demonstrates advanced coordination patterns for complex scenarios.
    """
    # Get scenario info
    scenario = create_infrastructure_collapse_scenario()
    
    console.print(Panel.fit(
        "[bold yellow]Running Advanced Scenario: Infrastructure Collapse[/bold yellow]\n\n"
        "This scenario uses hierarchical planning to coordinate multiple services\n"
        "during a complex city-wide emergency with cascading failures.",
        title="Advanced Coordination Pattern",
        border_style="yellow"
    ))
    
    # Create hierarchical crew
    crew = create_hierarchical_crew(
        scenario_description=scenario["description"],
        constraints=scenario["constraints"]
    )
    
    # Run the scenario
    return run_scenario(
        scenario_type=scenario["type"],
        scenario_description=scenario["description"],
        crew=crew,
        constraints=scenario["constraints"]
    ) 