from fastapi import FastAPI, Query, HTTPException, Depends, Path, Body, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os
import time
import random
from datetime import datetime
from enum import Enum
import logging

# Use relative imports for workshop modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workshop.config import get_verbosity, VerbosityLevel
from workshop.day_seed_generator import DaySeedGenerator

# Configure logger
logger = logging.getLogger("traffic_service")

app = FastAPI(title="NeoCatalis Traffic Service - Simplified")

# Global state
current_day = datetime.now().day
traffic_generator = DaySeedGenerator(day=current_day)
traffic_sectors = traffic_generator.generate_traffic_data(num_sectors=10)
service_health = {
    "status": "healthy",
    "latency": 0.1,
    "error_rate": 0.0,
}

# Log service startup
if get_verbosity() != VerbosityLevel.SILENT:
    logger.info("Simplified Traffic service initialized")
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Initial sectors: {len(traffic_sectors)}")

# Models
class TrafficStatus(str, Enum):
    CLEAR = "clear"
    MODERATE = "moderate"
    HEAVY = "heavy"
    GRIDLOCK = "gridlock"
    BLOCKED = "blocked"

class ServiceHealthResponse(BaseModel):
    status: str
    latency: float
    error_rate: float

# Dependency to simulate service degradation
def get_service_status():
    if service_health["status"] != "healthy":
        # Simulate random errors
        if random.random() < service_health["error_rate"]:
            raise HTTPException(status_code=500, detail="Traffic service temporarily unavailable")
        
        # Simulate latency
        time.sleep(service_health["latency"])
    
    return service_health

# ============================================================================
# SIMPLIFIED CORE ENDPOINTS - Only 3 essential actions
# ============================================================================

@app.post("/traffic/redirect", response_model=Dict[str, Any])
async def redirect_traffic(
    sector_id: str = Query(None),
    target_reduction: float = Query(0.5, ge=0.0, le=1.0),
    body: Dict[str, Any] = Body(None),
    service_status: Dict = Depends(get_service_status)
):
    """
    CORE ACTION: Redirect traffic to reduce congestion.
    
    This is one of the 3 essential traffic actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Redirecting traffic in sector {sector_id}")
    
    # Handle both query parameter and request body
    if sector_id is None and body:
        sector_id = body.get("sector_id")
    
    if target_reduction is None and body:
        target_reduction = body.get("target_reduction", 0.5)
    
    if sector_id is None:
        raise HTTPException(status_code=400, detail="sector_id is required")
    
    # Validate sector exists
    if sector_id not in traffic_sectors:
        raise HTTPException(status_code=404, detail=f"Sector {sector_id} not found")
    
    sector = traffic_sectors[sector_id]
    
    # Check if sector is blocked
    if sector.get("is_blocked", False):
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot redirect traffic in blocked sector {sector_id}"
        )
    
    # Calculate current congestion
    current_congestion = sector.get("congestion_level", 0.0)
    old_congestion = current_congestion
    
    # Apply traffic reduction
    new_congestion = max(0.0, current_congestion * (1.0 - target_reduction))
    sector["congestion_level"] = new_congestion
    
    # Update travel time multiplier based on new congestion
    sector["travel_time_multiplier"] = 1.0 + (new_congestion * 2.0)
    
    # Update status based on new congestion level
    if new_congestion < 0.3:
        sector["status"] = TrafficStatus.CLEAR
    elif new_congestion < 0.6:
        sector["status"] = TrafficStatus.MODERATE
    elif new_congestion < 0.9:
        sector["status"] = TrafficStatus.HEAVY
    else:
        sector["status"] = TrafficStatus.GRIDLOCK
    
    # Calculate actual reduction achieved
    actual_reduction = (old_congestion - new_congestion) / old_congestion if old_congestion > 0 else 0
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Traffic redirected in sector {sector_id}: {old_congestion:.2f} -> {new_congestion:.2f}")
    
    return {
        "success": True,
        "sector_id": sector_id,
        "old_congestion": old_congestion,
        "new_congestion": new_congestion,
        "target_reduction": target_reduction,
        "actual_reduction": actual_reduction,
        "new_status": sector["status"],
        "travel_time_multiplier": sector["travel_time_multiplier"]
    }

@app.post("/traffic/block_route", response_model=Dict[str, Any])
async def block_route(
    service_status: Dict = Depends(get_service_status),
    sector: Optional[str] = None,
    reason: Optional[str] = None,
    duration_minutes: Optional[int] = None,
    request: Request = None
):
    """
    CORE ACTION: Block a route in a specific sector.
    
    This is one of the 3 essential traffic actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Blocking route in sector: {sector}")
    
    # Handle both query parameter and request body
    if sector is None and request:
        try:
            body = await request.json()
            sector = body.get("sector")
            reason = body.get("reason")
            duration_minutes = body.get("duration_minutes")
        except:
            pass
    
    if sector is None:
        raise HTTPException(status_code=400, detail="Sector parameter is required")
    
    if reason is None:
        raise HTTPException(status_code=400, detail="Reason parameter is required")
    
    # Validate sector exists
    if sector not in traffic_sectors:
        raise HTTPException(status_code=404, detail=f"Sector {sector} not found")
    
    sector_data = traffic_sectors[sector]
    old_status = sector_data["status"]
    
    # Block the sector
    sector_data["status"] = TrafficStatus.BLOCKED
    sector_data["is_blocked"] = True
    sector_data["block_reason"] = reason
    sector_data["travel_time_multiplier"] = 5.0  # Significantly increased travel time
    
    if duration_minutes:
        sector_data["block_duration_minutes"] = duration_minutes
        # In a real system, you'd set up a timer to unblock after duration
        sector_data["block_end_time"] = datetime.now().timestamp() + (duration_minutes * 60)
    
    # Increase congestion in neighboring sectors (simplified simulation)
    congestion_spillover = sector_data.get("congestion_level", 0.0) * 0.3
    for other_sector_id, other_sector in traffic_sectors.items():
        if other_sector_id != sector and not other_sector.get("is_blocked", False):
            # Add some of the blocked sector's congestion to neighbors
            old_congestion = other_sector.get("congestion_level", 0.0)
            new_congestion = min(1.0, old_congestion + congestion_spillover * 0.1)
            other_sector["congestion_level"] = new_congestion
            
            # Update travel time multiplier
            other_sector["travel_time_multiplier"] = 1.0 + (new_congestion * 2.0)
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Route blocked in sector {sector}: {reason}")
    
    return {
        "success": True,
        "sector": sector,
        "old_status": old_status,
        "new_status": TrafficStatus.BLOCKED,
        "reason": reason,
        "duration_minutes": duration_minutes,
        "travel_time_multiplier": 5.0,
        "spillover_effect": f"Increased congestion in neighboring sectors by {congestion_spillover * 0.1:.2f}"
    }

@app.post("/traffic/report_conditions", response_model=Dict[str, Any])
async def report_conditions(
    description: Optional[str] = Query(None, description="Optional description of traffic conditions"),
    service_status: Dict = Depends(get_service_status)
):
    """
    CORE ACTION: Get current traffic conditions across all sectors.
    
    This is one of the 3 essential traffic actions.
    """
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Reporting traffic conditions")
    
    # Calculate overall traffic metrics
    total_sectors = len(traffic_sectors)
    blocked_sectors = sum(1 for sector in traffic_sectors.values() if sector.get("is_blocked", False))
    
    # Count sectors by status
    status_counts = {}
    for status in TrafficStatus:
        status_counts[status.value] = sum(
            1 for sector in traffic_sectors.values() 
            if sector["status"] == status.value
        )
    
    # Calculate average congestion (excluding blocked sectors)
    active_sectors = [s for s in traffic_sectors.values() if not s.get("is_blocked", False)]
    avg_congestion = sum(s.get("congestion_level", 0.0) for s in active_sectors) / len(active_sectors) if active_sectors else 0
    
    # Find most congested sectors
    congested_sectors = [
        {"sector_id": sid, "congestion": sector.get("congestion_level", 0.0)}
        for sid, sector in traffic_sectors.items()
        if sector.get("congestion_level", 0.0) > 0.7 and not sector.get("is_blocked", False)
    ]
    congested_sectors.sort(key=lambda x: x["congestion"], reverse=True)
    
    # Calculate average travel time multiplier
    avg_travel_multiplier = sum(s.get("travel_time_multiplier", 1.0) for s in active_sectors) / len(active_sectors) if active_sectors else 1.0
    
    # Determine overall traffic flow efficiency
    if avg_congestion < 0.3:
        overall_flow = "excellent"
    elif avg_congestion < 0.5:
        overall_flow = "good"
    elif avg_congestion < 0.7:
        overall_flow = "moderate"
    elif avg_congestion < 0.9:
        overall_flow = "poor"
    else:
        overall_flow = "critical"
    
    return {
        "total_sectors": total_sectors,
        "blocked_sectors": blocked_sectors,
        "active_sectors": len(active_sectors),
        "status_counts": status_counts,
        "average_congestion": avg_congestion,
        "average_travel_multiplier": avg_travel_multiplier,
        "overall_flow": overall_flow,
        "most_congested": congested_sectors[:5],  # Top 5 most congested
        "description": description or f"Traffic conditions as of {datetime.now().strftime('%H:%M')}",
        "sectors": {sid: {
            "status": sector["status"],
            "congestion_level": sector.get("congestion_level", 0.0),
            "is_blocked": sector.get("is_blocked", False),
            "travel_time_multiplier": sector.get("travel_time_multiplier", 1.0),
            "block_reason": sector.get("block_reason")
        } for sid, sector in traffic_sectors.items()}
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

def normalize_traffic_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize traffic data to ensure consistent structure."""
    normalized = {}
    
    if "traffic" in state:
        traffic_data = state["traffic"]
    elif "sectors" in state:
        traffic_data = state["sectors"]
    else:
        return {}
    
    # Handle both dict and list formats
    if isinstance(traffic_data, dict):
        for sector_id, sector in traffic_data.items():
            normalized[sector_id] = {
                "sector_id": sector_id,
                "status": sector.get("status", "clear"),
                "congestion_level": sector.get("congestion", sector.get("congestion_level", 0.0)),
                "is_blocked": sector.get("blocked", sector.get("is_blocked", False)),
                "travel_time_multiplier": sector.get("travel_time_multiplier", 1.0)
            }
    elif isinstance(traffic_data, list):
        for sector in traffic_data:
            sector_id = sector.get("id") or sector.get("sector_id") or sector.get("zone_id")
            if sector_id:
                normalized[sector_id] = {
                    "sector_id": sector_id,
                    "status": sector.get("status", "clear"),
                    "congestion_level": sector.get("congestion", sector.get("congestion_level", 0.0)),
                    "is_blocked": sector.get("blocked", sector.get("is_blocked", False)),
                    "travel_time_multiplier": sector.get("travel_time_multiplier", 1.0)
                }
    
    return normalized

@app.post("/state/set", response_model=Dict[str, Any])
async def set_traffic_state(state: Dict[str, Any]):
    """Set the traffic state (used by scenario activation)."""
    global traffic_sectors
    
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Setting traffic state from scenario")
    
    normalized = normalize_traffic_data(state)
    if normalized:
        traffic_sectors = normalized
    
    return {"success": True, "sectors_updated": len(traffic_sectors)}

@app.get("/state/get", response_model=Dict[str, Any])
async def get_traffic_state():
    """Get the current traffic state."""
    return {"sectors": traffic_sectors}

@app.post("/state/reset", response_model=Dict[str, Any])
async def reset_traffic_state():
    """Reset traffic state to initial values."""
    global traffic_sectors
    
    traffic_sectors = traffic_generator.generate_traffic_data(num_sectors=10)
    
    return {"success": True, "message": "Traffic state reset to initial values"}

# ============================================================================
# MAIN - Start the service
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port) 