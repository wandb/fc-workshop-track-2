from fastapi import FastAPI, Query, HTTPException, Depends, Path, Request
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import sys
import os
import time
import random
from datetime import datetime
from enum import Enum
import logging
import uuid
import traceback
from fastapi.responses import JSONResponse

# Use relative imports for workshop modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workshop.config import get_verbosity, VerbosityLevel
from workshop.day_seed_generator import DaySeedGenerator

# Configure logger
logger = logging.getLogger("emergency_service")

app = FastAPI(title="NeoCatalis Emergency Response Service - Simplified")

# Global state
current_day = datetime.now().day
seed_generator = DaySeedGenerator(day=current_day)
emergency_incidents = {
    incident["incident_id"]: incident 
    for incident in seed_generator.generate_emergency_incidents(num_incidents=15)
}
drone_fleet = seed_generator.generate_drone_fleet(num_drones=5)
service_health = {
    "status": "healthy",
    "latency": 0.1,
    "error_rate": 0.0,
}

# Log service startup
if get_verbosity() != VerbosityLevel.SILENT:
    logger.info("Simplified Emergency service initialized")
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Initial incidents: {len(emergency_incidents)}, drones: {len(drone_fleet)}")

# Models
class IncidentStatus(str, Enum):
    ACTIVE = "active"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELED = "canceled"

class DroneStatus(str, Enum):
    AVAILABLE = "available"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"
    ON_SITE = "on_site"
    RETURNING = "returning"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"

class ServiceHealthResponse(BaseModel):
    status: str
    latency: float
    error_rate: float

# Dependency to simulate service degradation
def get_service_status():
    if service_health["status"] != "healthy":
        # Simulate random errors
        if random.random() < service_health["error_rate"]:
            raise HTTPException(status_code=500, detail="Emergency service temporarily unavailable")
        
        # Simulate latency
        time.sleep(service_health["latency"])
    
    return service_health

# Middleware to catch and log exceptions
@app.middleware("http")
async def log_exceptions(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # Log the full exception traceback
        error_msg = f"Internal error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Request path: {request.url.path}")
        logger.error(f"Request method: {request.method}")
        logger.error(traceback.format_exc())
        
        # Return a meaningful error response
        return JSONResponse(
            status_code=500,
            content={"detail": error_msg, "traceback": traceback.format_exc()}
        )

# ============================================================================
# SIMPLIFIED CORE ENDPOINTS - Only 3 essential actions
# ============================================================================

@app.post("/emergency/drones/{drone_id}/assign", response_model=Dict[str, Any])
async def assign_drone(
    drone_id: str = Path(..., description="The ID of the drone to assign"),
    service_status: Dict = Depends(get_service_status),
    incident_id: Optional[str] = None,
    request: Request = None
):
    """
    CORE ACTION: Assign a drone to respond to an incident.
    
    This is one of the 3 essential emergency actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Assigning drone {drone_id}")
    
    # Handle both query parameter and request body
    if incident_id is None and request:
        try:
            body = await request.json()
            incident_id = body.get("incident_id")
        except:
            pass
    
    if incident_id is None:
        raise HTTPException(status_code=400, detail="incident_id is required")
    
    # Validate drone exists
    if drone_id not in drone_fleet:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    # Validate incident exists
    if incident_id not in emergency_incidents:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    
    drone = drone_fleet[drone_id]
    incident = emergency_incidents[incident_id]
    
    # Check if drone is available
    if drone["status"] != DroneStatus.AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail=f"Drone {drone_id} is not available (current status: {drone['status']})"
        )
    
    # Check if incident is active
    if incident["status"] not in [IncidentStatus.ACTIVE]:
        raise HTTPException(
            status_code=400,
            detail=f"Incident {incident_id} is not active (current status: {incident['status']})"
        )
    
    # Assign drone to incident
    drone["status"] = DroneStatus.ASSIGNED
    drone["assigned_incident"] = incident_id
    incident["status"] = IncidentStatus.ASSIGNED
    incident["assigned_drone"] = drone_id
    
    # Calculate estimated arrival time based on drone speed and distance
    estimated_arrival_minutes = random.randint(5, 20)  # Simplified calculation
    estimated_completion_minutes = estimated_arrival_minutes + incident.get("estimated_resolution_minutes", 30)
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Drone {drone_id} assigned to incident {incident_id}")
    
    return {
        "success": True,
        "drone_id": drone_id,
        "incident_id": incident_id,
        "estimated_arrival_minutes": estimated_arrival_minutes,
        "estimated_completion_minutes": estimated_completion_minutes,
        "drone_capabilities": drone.get("capabilities", []),
        "incident_type": incident.get("type", "unknown"),
        "incident_urgency": incident.get("urgency", "medium")
    }

@app.post("/emergency/incidents/{incident_id}", response_model=Dict[str, Any])
async def update_incident(
    incident_id: str = Path(..., description="The ID of the incident to update"),
    status: Optional[str] = Query(None, description="New status for the incident"),
    request: Request = None,
    service_status: Dict = Depends(get_service_status)
):
    """
    CORE ACTION: Update incident status.
    
    This is one of the 3 essential emergency actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Updating incident {incident_id}")
    
    # Handle both query parameter and request body
    if status is None and request:
        try:
            body = await request.json()
            status = body.get("status")
        except:
            pass
    
    if status is None:
        raise HTTPException(status_code=400, detail="status is required")
    
    # Validate incident exists
    if incident_id not in emergency_incidents:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    
    # Validate status
    valid_statuses = ["active", "assigned", "in_progress", "resolved", "canceled"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Status must be one of: {valid_statuses}"
        )
    
    incident = emergency_incidents[incident_id]
    old_status = incident["status"]
    incident["status"] = IncidentStatus(status)
    
    # Handle status transitions
    if status in ["resolved", "canceled"]:
        # Free up assigned drone
        assigned_drone_id = incident.get("assigned_drone")
        if assigned_drone_id and assigned_drone_id in drone_fleet:
            drone_fleet[assigned_drone_id]["status"] = DroneStatus.AVAILABLE
            drone_fleet[assigned_drone_id]["assigned_incident"] = None
            incident["assigned_drone"] = None
    
    elif status == "in_progress":
        # Update drone status if assigned
        assigned_drone_id = incident.get("assigned_drone")
        if assigned_drone_id and assigned_drone_id in drone_fleet:
            drone_fleet[assigned_drone_id]["status"] = DroneStatus.ON_SITE
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Incident {incident_id} status updated from {old_status} to {status}")
    
    return {
        "success": True,
        "incident_id": incident_id,
        "old_status": old_status,
        "new_status": status,
        "assigned_drone": incident.get("assigned_drone"),
        "type": incident.get("type", "unknown"),
        "urgency": incident.get("urgency", "medium"),
        "location": incident.get("zone", "unknown")
    }

@app.get("/emergency/report_status", response_model=Dict[str, Any])
async def report_status(
    service_status: Dict = Depends(get_service_status)
):
    """
    CORE ACTION: Get current emergency response status.
    
    This is one of the 3 essential emergency actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Reporting emergency status")
    
    # Count incidents by status
    incident_counts = {}
    for status in IncidentStatus:
        incident_counts[status.value] = sum(
            1 for incident in emergency_incidents.values() 
            if incident["status"] == status.value
        )
    
    # Count drones by status
    drone_counts = {}
    for status in DroneStatus:
        drone_counts[status.value] = sum(
            1 for drone in drone_fleet.values() 
            if drone["status"] == status.value
        )
    
    # Calculate response metrics
    total_incidents = len(emergency_incidents)
    active_incidents = incident_counts.get("active", 0)
    assigned_incidents = incident_counts.get("assigned", 0) + incident_counts.get("in_progress", 0)
    resolved_incidents = incident_counts.get("resolved", 0)
    
    available_drones = drone_counts.get("available", 0)
    active_drones = drone_counts.get("assigned", 0) + drone_counts.get("en_route", 0) + drone_counts.get("on_site", 0)
    
    # Calculate response rate
    response_rate = (assigned_incidents + resolved_incidents) / total_incidents if total_incidents > 0 else 0
    
    # Calculate drone utilization
    total_operational_drones = sum(count for status, count in drone_counts.items() 
                                  if status != "disabled")
    drone_utilization = active_drones / total_operational_drones if total_operational_drones > 0 else 0
    
    return {
        "total_incidents": total_incidents,
        "incident_counts": incident_counts,
        "active_incidents": active_incidents,
        "response_rate": response_rate,
        "total_drones": len(drone_fleet),
        "drone_counts": drone_counts,
        "available_drones": available_drones,
        "drone_utilization": drone_utilization,
        "incidents": {iid: {
            "status": incident["status"],
            "type": incident.get("type", "unknown"),
            "urgency": incident.get("urgency", "medium"),
            "assigned_drone": incident.get("assigned_drone"),
            "location": incident.get("zone", "unknown")
        } for iid, incident in emergency_incidents.items()},
        "drones": {did: {
            "status": drone["status"],
            "capabilities": drone.get("capabilities", []),
            "assigned_incident": drone.get("assigned_incident"),
            "location": drone.get("current_location", "base")
        } for did, drone in drone_fleet.items()}
    }

# ============================================================================
# ESSENTIAL SUPPORT ENDPOINTS - Health, State Management
# ============================================================================

@app.get("/service/health", response_model=ServiceHealthResponse)
async def get_service_health():
    """Get service health status."""
    return service_health

@app.post("/service/health")
async def set_service_health(
    status: str = "healthy", 
    latency: float = Query(0.0, ge=0.0, le=10.0),
    error_rate: float = Query(0.0, ge=0.0, le=1.0)
):
    """Set service health parameters for testing."""
    service_health["status"] = status
    service_health["latency"] = latency
    service_health["error_rate"] = error_rate
    return service_health

def normalize_emergency_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize emergency data to ensure consistent structure."""
    normalized = {"incidents": {}, "drones": {}}
    
    # Handle incidents - could be dict or list
    if "incidents" in state:
        incidents = state["incidents"]
        if isinstance(incidents, list):
            for incident in incidents:
                incident_id = incident.get("id") or incident.get("incident_id")
                if incident_id:
                    normalized["incidents"][incident_id] = {
                        "incident_id": incident_id,
                        "status": incident.get("status", "active"),
                        "type": incident.get("type", "unknown"),
                        "urgency": incident.get("urgency", "medium"),
                        "zone": incident.get("location", incident.get("zone", "unknown")),
                        "assigned_drone": incident.get("assigned_drone"),
                        "estimated_resolution_minutes": incident.get("estimated_resolution_minutes", 30)
                    }
        elif isinstance(incidents, dict):
            normalized["incidents"] = incidents
    
    # Handle drones - could be dict or list
    if "drones" in state:
        drones = state["drones"]
        if isinstance(drones, list):
            for drone in drones:
                drone_id = drone.get("id") or drone.get("drone_id")
                if drone_id:
                    normalized["drones"][drone_id] = {
                        "drone_id": drone_id,
                        "status": drone.get("status", "available"),
                        "capabilities": drone.get("capabilities", []),
                        "speed_kph": drone.get("speed", drone.get("speed_kph", 50)),
                        "current_location": drone.get("current_location", "base"),
                        "assigned_incident": drone.get("assigned_incident")
                    }
        elif isinstance(drones, dict):
            normalized["drones"] = drones
    
    return normalized

@app.post("/state/set", response_model=Dict[str, Any])
async def set_emergency_state(state: Dict[str, Any]):
    """Set the emergency state (used by scenario activation)."""
    global emergency_incidents, drone_fleet
    
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Setting emergency state from scenario")
    
    normalized = normalize_emergency_data(state)
    
    if normalized["incidents"]:
        emergency_incidents = normalized["incidents"]
    
    if normalized["drones"]:
        drone_fleet = normalized["drones"]
    
    return {
        "success": True, 
        "incidents_updated": len(emergency_incidents),
        "drones_updated": len(drone_fleet)
    }

@app.get("/state/get", response_model=Dict[str, Any])
async def get_emergency_state():
    """Get the current emergency state."""
    return {
        "incidents": emergency_incidents,
        "drones": drone_fleet
    }

@app.post("/state/reset", response_model=Dict[str, Any])
async def reset_emergency_state():
    """Reset emergency state to initial values."""
    global emergency_incidents, drone_fleet
    
    emergency_incidents = {
        incident["incident_id"]: incident 
        for incident in seed_generator.generate_emergency_incidents(num_incidents=15)
    }
    drone_fleet = seed_generator.generate_drone_fleet(num_drones=5)
    
    return {"success": True, "message": "Emergency state reset to initial values"}

# ============================================================================
# MAIN - Start the service
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port) 