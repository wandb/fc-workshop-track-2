from fastapi import FastAPI, Query, HTTPException, Depends, Path, Request, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os
import time
import random
import logging
from datetime import datetime
from enum import Enum
import traceback
from fastapi.responses import JSONResponse
import requests
import json

# Use relative imports for workshop modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workshop.config import get_verbosity, VerbosityLevel, should_show
from workshop.day_seed_generator import DaySeedGenerator

# Configure logger
logger = logging.getLogger("grid_service")

app = FastAPI(title="NeoCatalis Power Grid Service - Simplified")

# Global state
current_day = datetime.now().day
grid_generator = DaySeedGenerator(day=current_day)
grid_zones = grid_generator.generate_grid_data(num_zones=10)
service_health = {
    "status": "healthy",
    "latency": 0.1,
    "error_rate": 0.0,
}

# Log service startup
if get_verbosity() != VerbosityLevel.SILENT:
    logger.info("Simplified Grid service initialized")
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Initial zones: {len(grid_zones)}")

# Models
class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ServiceHealthResponse(BaseModel):
    status: str
    latency: float
    error_rate: float

# Track critical infrastructure
critical_infrastructure = {
    "hospital": {
        "infrastructure_id": "hospital",
        "level": PriorityLevel.HIGH,
        "estimated_load_kw": 5000
    },
    "police": {
        "infrastructure_id": "police",
        "level": PriorityLevel.MEDIUM,
        "estimated_load_kw": 3000
    },
    "water_treatment": {
        "infrastructure_id": "water_treatment",
        "level": PriorityLevel.MEDIUM,
        "estimated_load_kw": 4000
    },
    "data_center": {
        "infrastructure_id": "data_center",
        "level": PriorityLevel.LOW,
        "estimated_load_kw": 8000
    },
    "emergency_shelter": {
        "infrastructure_id": "emergency_shelter",
        "level": PriorityLevel.LOW,
        "estimated_load_kw": 2000
    }
}

# Dependency to simulate service degradation
def get_service_status():
    if service_health["status"] != "healthy":
        # Simulate random errors
        if random.random() < service_health["error_rate"]:
            if get_verbosity() != VerbosityLevel.SILENT:
                logger.error("Grid service temporarily unavailable (simulated error)")
            raise HTTPException(status_code=500, detail="Grid service temporarily unavailable")
        
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

@app.put("/grid/zones/{zone_id}/capacity", response_model=Dict[str, Any])
async def adjust_zone_capacity(
    zone_id: str = Path(..., description="The ID of the zone to adjust"),
    service_status: Dict = Depends(get_service_status),
    capacity: Optional[float] = None,
    request: Request = None
):
    """
    CORE ACTION: Adjust power capacity of a specific zone.
    
    This is one of the 3 essential grid actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Adjusting capacity for zone: {zone_id}")
    
    if zone_id not in grid_zones:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    
    # Handle both query parameter and request body
    if capacity is None and request:
        try:
            body = await request.json()
            capacity = body.get("capacity")
        except:
            pass
    
    if capacity is None:
        raise HTTPException(status_code=400, detail="Capacity parameter is required")
    
    if not (0.0 <= capacity <= 1.0):
        raise HTTPException(status_code=400, detail="Capacity must be between 0.0 and 1.0")
    
    zone = grid_zones[zone_id]
    old_capacity = zone.get("capacity_kw", 0)
    
    # Calculate new capacity based on percentage
    max_capacity = zone.get("max_capacity_kw", old_capacity)
    new_capacity = max_capacity * capacity  # Keep as float, don't convert to int
    
    # Update zone capacity
    zone["capacity_kw"] = new_capacity
    
    # Recalculate stability based on new capacity
    load_ratio = zone["current_load_kw"] / new_capacity if new_capacity > 0 else 1.0
    if load_ratio > 1.0:
        zone["stability"] = max(0.1, zone.get("stability", 0.5) - 0.3)
        zone["status"] = "overloaded"
    elif load_ratio > 0.9:
        zone["stability"] = max(0.3, zone.get("stability", 0.5) - 0.1)
        zone["status"] = "degraded"
    else:
        zone["stability"] = min(1.0, zone.get("stability", 0.5) + 0.2)
        zone["status"] = "online"
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Zone {zone_id} capacity adjusted to {capacity:.1%} ({new_capacity}kW)")
    
    return {
        "success": True,
        "zone_id": zone_id,
        "old_capacity_kw": old_capacity,
        "new_capacity_kw": new_capacity,
        "capacity_percentage": capacity,
        "current_load_kw": zone["current_load_kw"],
        "load_ratio": load_ratio,
        "stability": zone["stability"],
        "status": zone["status"]
    }

@app.post("/grid/infrastructure/{infrastructure_id}/priority", response_model=Dict[str, Any])
async def set_priority(
    infrastructure_id: str = Path(..., description="The ID of the infrastructure to prioritize"),
    service_status: Dict = Depends(get_service_status),
    level: Optional[str] = None,
    request: Request = None
):
    """
    CORE ACTION: Set priority level for critical infrastructure.
    
    This is one of the 3 essential grid actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Setting priority for infrastructure: {infrastructure_id}")
    
    # Handle both query parameter and request body
    if level is None and request:
        try:
            body = await request.json()
            level = body.get("level")
        except:
            pass
    
    if level is None:
        raise HTTPException(status_code=400, detail="Priority level is required")
    
    # Validate priority level
    valid_levels = ["critical", "high", "medium", "low"]
    if level not in valid_levels:
        raise HTTPException(status_code=400, detail=f"Priority level must be one of: {valid_levels}")
    
    # Create infrastructure entry if it doesn't exist
    if infrastructure_id not in critical_infrastructure:
        critical_infrastructure[infrastructure_id] = {
            "infrastructure_id": infrastructure_id,
            "level": PriorityLevel.MEDIUM,
            "estimated_load_kw": 1000
        }
    
    old_level = critical_infrastructure[infrastructure_id]["level"]
    critical_infrastructure[infrastructure_id]["level"] = PriorityLevel(level)
    
    # Simulate impact on grid zones based on priority change
    impact_zones = []
    for zone_id, zone in grid_zones.items():
        if zone.get("is_critical", False):
            if level == "critical":
                # Critical priority improves stability
                zone["stability"] = min(1.0, zone.get("stability", 0.5) + 0.1)
                impact_zones.append(zone_id)
            elif old_level == "critical" and level != "critical":
                # Downgrading from critical reduces stability
                zone["stability"] = max(0.1, zone.get("stability", 0.5) - 0.1)
                impact_zones.append(zone_id)
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Infrastructure {infrastructure_id} priority set to {level}")
    
    return {
        "success": True,
        "infrastructure_id": infrastructure_id,
        "old_level": old_level,
        "new_level": level,
        "estimated_load_kw": critical_infrastructure[infrastructure_id]["estimated_load_kw"],
        "affected_zones": impact_zones
    }

@app.get("/grid/report_status", response_model=Dict[str, Any])
async def report_status(
    zone_id: Optional[str] = Query(None, description="Optional specific zone to report on"),
    service_status: Dict = Depends(get_service_status)
):
    """
    CORE ACTION: Get current grid status across all zones or a specific zone.
    
    This is one of the 3 essential grid actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Reporting grid status for zone: {zone_id or 'all zones'}")
    
    if zone_id:
        # Report on specific zone
        if zone_id not in grid_zones:
            raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
        
        zone = grid_zones[zone_id]
        return {
            "zone_id": zone_id,
            "status": zone["status"],
            "capacity_kw": zone["capacity_kw"],
            "current_load_kw": zone["current_load_kw"],
            "load_ratio": zone["current_load_kw"] / zone["capacity_kw"] if zone["capacity_kw"] > 0 else 0,
            "stability": zone.get("stability", 0.5),
            "is_critical": zone.get("is_critical", False)
        }
    else:
        # Report on all zones
        total_capacity = sum(zone["capacity_kw"] for zone in grid_zones.values())
        total_load = sum(zone["current_load_kw"] for zone in grid_zones.values())
        
        critical_zones = [zid for zid, zone in grid_zones.items() if zone.get("is_critical", False)]
        offline_zones = [zid for zid, zone in grid_zones.items() if zone["status"] == "offline"]
        overloaded_zones = [zid for zid, zone in grid_zones.items() if zone["status"] == "overloaded"]
        
        avg_stability = sum(zone.get("stability", 0.5) for zone in grid_zones.values()) / len(grid_zones)
        
        return {
            "total_zones": len(grid_zones),
            "total_capacity_kw": total_capacity,
            "total_load_kw": total_load,
            "overall_load_ratio": total_load / total_capacity if total_capacity > 0 else 0,
            "average_stability": avg_stability,
            "critical_zones": critical_zones,
            "offline_zones": offline_zones,
            "overloaded_zones": overloaded_zones,
            "critical_infrastructure": list(critical_infrastructure.keys()),
            "zones": {zid: {
                "status": zone["status"],
                "capacity_kw": zone["capacity_kw"],
                "current_load_kw": zone["current_load_kw"],
                "stability": zone.get("stability", 0.5),
                "is_critical": zone.get("is_critical", False)
            } for zid, zone in grid_zones.items()}
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

def normalize_zone_data(zone_data):
    """Normalize zone data to ensure consistent structure."""
    normalized = {}
    for zone_id, zone in zone_data.items():
        normalized[zone_id] = {
            "zone_id": zone_id,
            "status": zone.get("status", "online"),
            "capacity_kw": zone.get("capacity", zone.get("capacity_kw", 1000)),
            "current_load_kw": zone.get("current_load", zone.get("current_load_kw", 500)),
            "is_critical": zone.get("is_critical", False),
            "stability": zone.get("stability", 0.5)
        }
        
        # Ensure max_capacity_kw is set for capacity adjustments
        if "max_capacity_kw" not in normalized[zone_id]:
            normalized[zone_id]["max_capacity_kw"] = normalized[zone_id]["capacity_kw"]
    
    return normalized

@app.post("/state/set", response_model=Dict[str, Any])
async def set_grid_state(state: Dict[str, Any]):
    """Set the grid state (used by scenario activation)."""
    global grid_zones, critical_infrastructure
    
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Setting grid state from scenario")
    
    if "zones" in state:
        grid_zones = normalize_zone_data(state["zones"])
    
    if "infrastructure" in state:
        critical_infrastructure.update(state["infrastructure"])
    
    return {"success": True, "zones_updated": len(grid_zones)}

@app.get("/state/get", response_model=Dict[str, Any])
async def get_grid_state():
    """Get the current grid state."""
    return {
        "zones": grid_zones,
        "infrastructure": critical_infrastructure
    }

@app.post("/state/reset", response_model=Dict[str, Any])
async def reset_grid_state():
    """Reset grid state to initial values."""
    global grid_zones, critical_infrastructure
    
    grid_zones = grid_generator.generate_grid_data(num_zones=10)
    
    # Reset infrastructure to defaults
    critical_infrastructure = {
        "hospital": {
            "infrastructure_id": "hospital",
            "level": PriorityLevel.HIGH,
            "estimated_load_kw": 5000
        },
        "police": {
            "infrastructure_id": "police",
            "level": PriorityLevel.MEDIUM,
            "estimated_load_kw": 3000
        },
        "water_treatment": {
            "infrastructure_id": "water_treatment",
            "level": PriorityLevel.MEDIUM,
            "estimated_load_kw": 4000
        },
        "data_center": {
            "infrastructure_id": "data_center",
            "level": PriorityLevel.LOW,
            "estimated_load_kw": 8000
        },
        "emergency_shelter": {
            "infrastructure_id": "emergency_shelter",
            "level": PriorityLevel.LOW,
            "estimated_load_kw": 2000
        }
    }
    
    return {"success": True, "message": "Grid state reset to initial values"}

# ============================================================================
# MAIN - Start the service
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port) 