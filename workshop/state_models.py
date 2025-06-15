"""
State models for SENTINEL GRID services.

This module provides Pydantic models for consistent state management
across all services in the SENTINEL GRID system.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ZoneState(BaseModel):
    """Base state for a grid zone."""
    id: str
    name: str
    capacity: float = Field(ge=0.0, le=1.0)
    current_load: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    status: str = Field(default="online")
    is_critical: bool = False

class WeatherState(BaseModel):
    """Base state for weather conditions."""
    condition: str
    temperature: float
    wind_speed: float
    precipitation: float
    severity: float = Field(ge=0.0, le=1.0)
    alerts: List[str] = Field(default_factory=list)

class IncidentState(BaseModel):
    """Base state for an emergency incident."""
    id: str
    description: str
    location: str
    urgency: float = Field(ge=0.0, le=1.0)
    status: str = Field(default="active")
    assigned_drone: Optional[str] = None

class DroneState(BaseModel):
    """Base state for an emergency drone."""
    id: str
    name: str
    status: str = Field(default="available")
    capabilities: List[str]
    location: Optional[str] = None
    weather_resistant: bool = False
    speed: float = Field(ge=0.1, le=2.0)

class TrafficState(BaseModel):
    """Base state for traffic conditions."""
    zone_id: str
    congestion: float = Field(ge=0.0, le=1.0)
    blocked: bool = False
    description: str

class ServiceState(BaseModel):
    """Complete state for all services in the system."""
    # Grid service state
    zones: Dict[str, ZoneState] = Field(default_factory=dict)
    
    # Weather service state
    weather: Optional[WeatherState] = None
    
    # Emergency service state
    incidents: List[IncidentState] = Field(default_factory=list)
    drones: List[DroneState] = Field(default_factory=list)
    
    # Traffic service state
    traffic: Dict[str, TrafficState] = Field(default_factory=dict)
    
    # System-wide state
    timestamp: Optional[float] = None
    scenario_name: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set timestamp if not provided
        if not self.timestamp:
            import time
            self.timestamp = time.time()

class SuccessCriteria(BaseModel):
    """Success criteria for a scenario."""
    name: str
    description: str
    metrics: Dict[str, Any]
    thresholds: Dict[str, float]
    time_limit: Optional[int] = None  # Time limit in minutes

class ScenarioDefinition(BaseModel):
    """Complete definition of a scenario."""
    name: str
    description: str
    initial_state: ServiceState
    success_criteria: SuccessCriteria
    optimal_commands: List[Dict[str, Any]]
    command_weights: Dict[str, float] 