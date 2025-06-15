"""
Service tool creation utilities for the SENTINEL GRID workshop.

This module provides functions to create and register tools for interacting 
with the various services of the SENTINEL GRID system.
"""

import time
import logging
from typing import Dict, List, Any, Optional

from langchain.agents import Tool
from rich.logging import RichHandler

from workshop.command import Command, ServiceType, CommandExecutor
from workshop.deprecated_utils.tool_registry import tool_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("service_tools")

# Initialize command executor
command_executor = CommandExecutor()

def create_service_tool(service: str, action: str, description: str, 
                       parameter_description: str = "", 
                       tags=None, capabilities=None, version="1.0"):
    """
    Create a tool for a specific service action and register it with the MCP registry.
    
    Args:
        service: The service name (grid, emergency, weather, traffic)
        action: The action name
        description: Description of what the tool does
        parameter_description: Description of parameters (if any)
        tags: List of tags for categorizing the tool
        capabilities: Dict of tool capabilities (e.g., {"works_in_rain": False})
        version: Tool version string
        
    Returns:
        LangChain Tool object
    """
    def execute_service_action(*args, **kwargs):
        """Execute the service action with provided parameters."""
        parameters = kwargs
        if len(args) == 1 and isinstance(args[0], dict):
            # If a single dictionary was passed, use it as parameters
            parameters = args[0]
        
        try:
            # Create service type
            service_type = ServiceType(service.lower())
            
            # Create command
            command = Command(
                service=service_type,
                action=action,
                parameters=parameters
            )
            
            # Log the command with timing information
            start_time = time.time()
            log.info(f"Executing command: [bold blue]{service}[/bold blue]."
                    f"[bold green]{action}[/bold green] with parameters: {parameters}")
            
            # Execute command
            result = command_executor.execute(command)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Format response
            response = {
                "success": result.success,
                "result": result.result,
                "error": result.error
            }
            
            # Add execution metrics to result
            response["execution_metrics"] = {
                "execution_time_seconds": execution_time,
                "tool_version": version,
                "timestamp": time.time()
            }
            
            # Log the result with timing information
            if response["success"]:
                log.info(f"Command succeeded: [bold green]{service}.{action}[/bold green] "
                        f"in {execution_time:.2f}s")
            else:
                log.error(f"Command failed: [bold red]{service}.{action}[/bold red] - "
                         f"{response['error']} after {execution_time:.2f}s")
            
            return response
        except Exception as e:
            log.exception(f"Error executing command {service}.{action}: {str(e)}")
            return {
                "success": False,
                "result": {},
                "error": str(e)
            }
    
    # Create the tool description
    full_description = description
    if parameter_description:
        full_description += f"\n\nParameters: {parameter_description}"
    
    # Add capability information to description
    if capabilities:
        capability_desc = "\n\nCapabilities:"
        for cap, value in capabilities.items():
            capability_desc += f"\n- {cap}: {value}"
        full_description += capability_desc
    
    # Create the tool
    tool = Tool(
        name=f"{service}_{action}",
        func=execute_service_action,
        description=full_description
    )
    
    # Register the tool with the MCP registry
    if tags is None:
        tags = [service, action, f"{service}_{action}"]
    else:
        tags = tags + [service, action, f"{service}_{action}"]
    
    # Default capabilities based on service if none provided
    if capabilities is None:
        capabilities = {}
        
        # Add default capability values based on service type
        if service == "weather":
            capabilities["provides_forecast"] = True
        elif service == "emergency":
            capabilities["handles_incidents"] = True
        elif service == "grid":
            capabilities["manages_power"] = True
        elif service == "traffic":
            capabilities["manages_routes"] = True
    
    tool_registry.register_tool(
        tool,
        tags=tags,
        metadata={
            "service": service,
            "action": action,
            "parameters": parameter_description
        },
        capabilities=capabilities,
        version=version
    )
    
    return tool

# Grid Service Tools
def create_grid_tools():
    """Create tools for the Grid Service."""
    return [
        create_service_tool(
            "grid", "report_status",
            "Get the current status of all grid zones in the city.",
            "",
            tags=["monitoring", "status", "grid_health"]
        ),
        create_service_tool(
            "grid", "adjust_zone",
            "Adjust the capacity of a specific grid zone to manage load.",
            "zone_id: The ID of the zone to adjust, capacity: The new capacity level (0.0-1.0)",
            tags=["actuation", "load_management", "power_control"]
        ),
        create_service_tool(
            "grid", "set_priority",
            "Set priority level for critical infrastructure.",
            "infrastructure_id: The ID of the infrastructure, level: Priority level (normal, high, critical)",
            tags=["priority", "resource_allocation", "infrastructure"]
        ),
        create_service_tool(
            "grid", "forecast_load",
            "Forecast the grid load for the next X hours.",
            "hours: Number of hours to forecast (default: 24)",
            tags=["prediction", "planning", "forecast"]
        ),
        create_service_tool(
            "grid", "emergency_shutdown",
            "Perform an emergency shutdown of a grid zone.",
            "zone_id: The ID of the zone to shut down, reason: Reason for shutdown",
            tags=["emergency", "safety", "shutdown"]
        )
    ]

# Emergency Service Tools
def create_emergency_tools():
    """Create tools for the Emergency Service."""
    return [
        create_service_tool(
            "emergency", "prioritize_incident",
            "Get a list of incidents with the specified priority.",
            "priority: Priority level (low, medium, high, critical), count: Maximum number of incidents to return",
            tags=["monitoring", "prioritization", "incidents"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "emergency", "assign_drone",
            "Assign a drone to an incident.",
            "drone_id: The ID of the drone to assign, incident_id: The ID of the incident",
            tags=["actuation", "assignment", "drones"],
            capabilities={"requires_available_drone": True, "affected_by_weather": True}
        ),
        create_service_tool(
            "emergency", "recall_drone",
            "Recall a drone from its current assignment.",
            "drone_id: The ID of the drone to recall",
            tags=["actuation", "drone_management"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "emergency", "update_drone_status",
            "Update the status of a drone.",
            "drone_id: The ID of the drone, status: New status (available, deployed, maintenance)",
            tags=["management", "status_update", "drones"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "emergency", "estimate_response",
            "Estimate response time for an incident.",
            "incident_id: The ID of the incident",
            tags=["analysis", "estimation", "planning"],
            capabilities={"considers_traffic": True, "considers_weather": True}
        ),
        create_service_tool(
            "emergency", "update_incident",
            "Update the status of an incident.",
            "incident_id: The ID of the incident, status: New status",
            tags=["management", "status_update", "incidents"],
            capabilities={"works_in_all_conditions": True}
        )
    ]

# Weather Service Tools
def create_weather_tools():
    """Create tools for the Weather Service."""
    return [
        create_service_tool(
            "weather", "report_conditions",
            "Get current weather conditions in the city.",
            "",
            tags=["monitoring", "status", "conditions"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "weather", "forecast",
            "Get weather forecast for the next X hours.",
            "hours: Number of hours to forecast (default: 24)",
            tags=["prediction", "planning", "forecast"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "weather", "get_advisories",
            "Get active weather advisories.",
            "",
            tags=["monitoring", "alerts", "advisories"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "weather", "analyze_trend",
            "Analyze weather trends over the specified number of days.",
            "days: Number of days to analyze",
            tags=["analysis", "trends", "historical"],
            capabilities={"works_in_all_conditions": True}
        ),
        create_service_tool(
            "weather", "report_impact",
            "Report weather impact on city services.",
            "description: Description of the impact being reported",
            tags=["impact", "reporting", "services"],
            capabilities={"works_in_all_conditions": True}
        )
    ]

# Traffic Service Tools
def create_traffic_tools():
    """Create tools for the Traffic Service."""
    return [
        create_service_tool(
            "traffic", "report_conditions",
            "Get current traffic conditions in the city.",
            ""
        ),
        create_service_tool(
            "traffic", "get_congestion",
            "Get information about traffic congestion.",
            "min_level: Minimum congestion level to report (0.0-1.0)"
        ),
        create_service_tool(
            "traffic", "set_route",
            "Set up a priority route between sectors.",
            "from_sector: Starting sector, to_sector: Destination sector, priority: Whether this is a priority route"
        ),
        create_service_tool(
            "traffic", "estimate_travel_time",
            "Estimate travel time between sectors.",
            "from_sector: Starting sector, to_sector: Destination sector, emergency: Whether this is for emergency vehicles"
        ),
        create_service_tool(
            "traffic", "block_route",
            "Block a route due to hazards or maintenance.",
            "sector: The sector to block, reason: Reason for blocking, duration_minutes: Duration of the block"
        ),
        create_service_tool(
            "traffic", "find_alternate_route",
            "Find alternate route avoiding specified sectors.",
            "from_sector: Starting sector, to_sector: Destination sector, avoid_sectors: List of sectors to avoid, emergency: Whether this is for emergency vehicles"
        )
    ]

# Helper functions for tool discovery
def create_analyst_tools():
    """Create tools for analyst agents using the MCP registry."""
    # Use tag-based discovery for monitoring and status tools
    monitoring_tools = tool_registry.discover_tools_by_tag("monitoring")
    status_tools = tool_registry.discover_tools_by_tag("status")
    
    # Use query-based discovery for forecast tools
    forecast_tools = tool_registry.discover_tools_by_query("forecast prediction")
    
    # Combine the tools
    return list(set(monitoring_tools + status_tools + forecast_tools))

def create_executor_tools():
    """Create tools for executor agents using the MCP registry."""
    # Use tag-based discovery for actuation tools
    actuation_tools = tool_registry.discover_tools_by_tag("actuation")
    
    # Use query-based discovery for action tools
    emergency_tools = tool_registry.discover_tools_by_query("emergency response")
    priority_tools = tool_registry.discover_tools_by_query("priority")
    
    # Combine the tools
    return list(set(actuation_tools + emergency_tools + priority_tools))

def create_coordinator_tools():
    """Create tools for coordinator agents using the MCP registry."""
    # Coordinators get access to all tools
    return [tool_info["tool"] for tool_info in tool_registry.tools.values()]

# Function to initialize all tools
def initialize_all_tools():
    """Initialize all service tools and register them with the tool registry."""
    grid_tools = create_grid_tools()
    emergency_tools = create_emergency_tools()
    weather_tools = create_weather_tools()
    traffic_tools = create_traffic_tools()
    
    log.info(f"Initialized {len(grid_tools)} grid tools")
    log.info(f"Initialized {len(emergency_tools)} emergency tools")
    log.info(f"Initialized {len(weather_tools)} weather tools")
    log.info(f"Initialized {len(traffic_tools)} traffic tools")
    
    return {
        "grid": grid_tools,
        "emergency": emergency_tools,
        "weather": weather_tools,
        "traffic": traffic_tools
    } 