"""
Agent Profile definitions for the SENTINEL GRID workshop.

This module provides standardized agent profiles with capability definitions
and tool discovery mechanisms for various agent roles in the smart city.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from workshop.deprecated_utils.tool_registry import tool_registry

class AgentProfile(BaseModel):
    """
    Model for an agent profile with capabilities and preferences.
    
    This helps match agents with appropriate tools using the MCP registry.
    """
    role: str = Field(description="Agent role")
    capabilities: Dict[str, Any] = Field(description="Agent capabilities")
    preferred_tags: List[str] = Field(description="Preferred tool tags")
    required_capabilities: Dict[str, Any] = Field(
        description="Required tool capabilities"
    )
    
    @classmethod
    def weather_analyst(cls):
        """Create a weather analyst profile."""
        return cls(
            role="Weather Analyst",
            capabilities={
                "weather_expertise": True,
                "forecast_analysis": True
            },
            preferred_tags=["monitoring", "forecast", "analysis", "weather"],
            required_capabilities={
                "works_in_all_conditions": True
            }
        )
    
    @classmethod
    def grid_analyst(cls):
        """Create a grid analyst profile."""
        return cls(
            role="Grid Analyst",
            capabilities={
                "grid_expertise": True,
                "load_analysis": True
            },
            preferred_tags=["monitoring", "grid", "stability", "analysis"],
            required_capabilities={
                "manages_power": True
            }
        )
    
    @classmethod
    def emergency_executor(cls):
        """Create an emergency executor profile."""
        return cls(
            role="Emergency Executor",
            capabilities={
                "incident_management": True,
                "drone_control": True
            },
            preferred_tags=["actuation", "emergency", "drones", "assignment"],
            required_capabilities={
                "handles_incidents": True
            }
        )
    
    @classmethod
    def traffic_manager(cls):
        """Create a traffic manager profile."""
        return cls(
            role="Traffic Manager",
            capabilities={
                "traffic_expertise": True,
                "route_optimization": True
            },
            preferred_tags=["monitoring", "routing", "traffic", "analysis"],
            required_capabilities={
                "manages_routes": True
            }
        )
    
    @classmethod
    def city_coordinator(cls):
        """Create a city coordinator profile."""
        return cls(
            role="City Coordinator",
            capabilities={
                "strategic_planning": True,
                "resource_allocation": True,
                "multi_service_coordination": True
            },
            preferred_tags=["coordination", "planning", "priority"],
            required_capabilities={}  # Coordinators can use any tools
        )
    
    def discover_tools(self):
        """Discover suitable tools based on this profile."""
        return tool_registry.discover_tools_for_agent(self.dict()) 