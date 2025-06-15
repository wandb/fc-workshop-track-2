from fastapi import FastAPI, Path, HTTPException, Body, Query
from pydantic import BaseModel
import sys
import os
import time
import uuid
import logging
import json
import requests
from typing import Dict, List, Any, Optional

# Use relative imports for workshop modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workshop.config import get_verbosity, VerbosityLevel, should_show, get_service_url
from workshop.state_models import ScenarioDefinition, ServiceState

# Configure logger
logger = logging.getLogger("scenario_service")

app = FastAPI(title="Sentinel Grid Scenario Service")

# Global state
scenarios = {}

# Log service startup
if get_verbosity() != VerbosityLevel.SILENT:
    logger.info("Scenario service initialized")

# Helper functions
def get_service_endpoints() -> Dict[str, str]:
    """Get URLs for all other services."""
    return {
        "grid": get_service_url("grid"),
        "emergency": get_service_url("emergency"),
        "traffic": get_service_url("traffic")
    }

def is_service_available(service_url: str) -> bool:
    """Check if a service is available."""
    try:
        # Use health check endpoint
        response = requests.get(f"{service_url}/service/health", timeout=2)
        return response.status_code >= 200 and response.status_code < 300
    except:
        return False

def convert_model_to_dict(obj):
    """Convert a Pydantic model to a dict, or return the object if it's not a model."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: convert_model_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_model_to_dict(item) for item in obj]
    else:
        return obj

# Routes
@app.get("/")
async def root():
    """Root endpoint for health check."""
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Health check request received")
    
    return {"status": "ok", "service": "scenario"}

@app.get("/scenarios", response_model=Dict[str, Any])
async def list_scenarios():
    """List all available scenarios."""
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Listing {len(scenarios)} scenarios")
    
    return {
        "scenarios": [
            {
                "id": scenario_id,
                "name": scenario.name,
                "description": scenario.description
            }
            for scenario_id, scenario in scenarios.items()
        ]
    }

@app.get("/scenarios/{scenario_id}", response_model=Dict[str, Any])
async def get_scenario(scenario_id: str = Path(..., description="The ID of the scenario to get")):
    """Get a specific scenario by ID."""
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Getting scenario: {scenario_id}")
    
    if scenario_id not in scenarios:
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.warning(f"Scenario not found: {scenario_id}")
        
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
    
    scenario = scenarios[scenario_id]
    
    # Convert to dict for response
    if hasattr(scenario, 'model_dump'):
        scenario_data = scenario.model_dump()
    else:
        scenario_data = scenario.dict()
    
    # Add scenario ID to response
    scenario_data["id"] = scenario_id
    
    return scenario_data

@app.post("/scenarios", response_model=Dict[str, Any])
async def create_scenario(scenario: ScenarioDefinition = Body(...)):
    """Create a new scenario."""
    # Generate unique ID
    scenario_id = str(uuid.uuid4())
    scenarios[scenario_id] = scenario
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Created scenario: {scenario.name} (ID: {scenario_id})")
    
    # Return the scenario with its ID
    if hasattr(scenario, 'model_dump'):
        scenario_data = scenario.model_dump()
    else:
        scenario_data = scenario.dict()
    
    scenario_data["id"] = scenario_id
    
    return scenario_data

@app.post("/scenarios/{scenario_id}/activate", response_model=Dict[str, Any])
async def activate_scenario(scenario_id: str = Path(..., description="The ID of the scenario to activate")):
    """
    Activate a scenario by initializing all services with the scenario state.
    """
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info(f"Activating scenario: {scenario_id}")
    
    # Check if scenario exists
    if scenario_id not in scenarios:
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.warning(f"Scenario not found: {scenario_id}")
        
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
    
    scenario = scenarios[scenario_id]
    initial_state = scenario.initial_state
    
    # Get service endpoints
    endpoints = get_service_endpoints()
    
    # Check if all services are available
    for service, url in endpoints.items():
        if not is_service_available(url):
            error_msg = f"Service {service} is not available at {url}"
            
            if get_verbosity() != VerbosityLevel.SILENT:
                logger.error(error_msg)
            
            raise HTTPException(status_code=503, detail=error_msg)
    
    # Distribute initial state to all services
    results = {}
    
    try:
        # Initialize grid service
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            logger.debug("Initializing grid service")
        
        # Convert zones to dict for serialization
        grid_state = {"zones": convert_model_to_dict(initial_state.zones)}
        grid_response = requests.post(f"{endpoints['grid']}/state/set", json=grid_state, timeout=10)
        
        if grid_response.status_code < 300:
            results["grid"] = {"status": "success", "message": "Grid state initialized"}
            if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                logger.debug("Grid service initialized successfully")
        else:
            results["grid"] = {"error": f"HTTP {grid_response.status_code}: {grid_response.text}"}
            logger.error(f"Grid service initialization failed: {grid_response.status_code}")
        
        # Initialize emergency service
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            logger.debug("Initializing emergency service")
        
        # Convert incidents and drones to dict for serialization
        emergency_state = {
            "incidents": convert_model_to_dict(initial_state.incidents),
            "drones": convert_model_to_dict(initial_state.drones)
        }
        emergency_response = requests.post(f"{endpoints['emergency']}/state/set", json=emergency_state, timeout=10)
        
        if emergency_response.status_code < 300:
            results["emergency"] = {"status": "success", "message": "Emergency state initialized"}
            if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                logger.debug("Emergency service initialized successfully")
        else:
            results["emergency"] = {"error": f"HTTP {emergency_response.status_code}: {emergency_response.text}"}
            logger.error(f"Emergency service initialization failed: {emergency_response.status_code}")
        
        # Initialize traffic service
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            logger.debug("Initializing traffic service")
        
        # Convert traffic to dict for serialization
        traffic_state = {"sectors": convert_model_to_dict(initial_state.traffic)}
        traffic_response = requests.post(f"{endpoints['traffic']}/state/set", json=traffic_state, timeout=10)
        
        if traffic_response.status_code < 300:
            results["traffic"] = {"status": "success", "message": "Traffic state initialized"}
            if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                logger.debug("Traffic service initialized successfully")
        else:
            results["traffic"] = {"error": f"HTTP {traffic_response.status_code}: {traffic_response.text}"}
            logger.error(f"Traffic service initialization failed: {traffic_response.status_code}")
        
        # Check if all services initialized successfully
        errors = [service for service, result in results.items() if "error" in result]
        
        if errors:
            error_msg = f"Failed to initialize services: {', '.join(errors)}"
            
            if get_verbosity() != VerbosityLevel.SILENT:
                logger.error(error_msg)
            
            # Return partial success with details instead of raising exception
            return {
                "status": "partial_success",
                "message": f"Scenario {scenario_id} partially activated. Some services failed.",
                "results": results,
                "errors": errors,
                "successful_services": [s for s in results.keys() if "error" not in results[s]]
            }
        
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.info(f"Scenario {scenario_id} activated successfully")
        
        return {
            "status": "success",
            "message": f"Scenario {scenario_id} activated successfully",
            "results": results,
            "services_initialized": list(results.keys())
        }
    
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout during scenario activation: {str(e)}"
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.error(error_msg)
        
        return {
            "status": "timeout",
            "message": error_msg,
            "results": results
        }
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error during scenario activation: {str(e)}"
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.error(error_msg)
        
        return {
            "status": "connection_error", 
            "message": error_msg,
            "results": results
        }
    
    except Exception as e:
        error_msg = f"Unexpected error during scenario activation: {str(e)}"
        
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.error(error_msg)
        
        return {
            "status": "error",
            "message": error_msg,
            "results": results,
            "exception_type": type(e).__name__
        }

@app.get("/service/health")
async def health_check():
    """Health check endpoint."""
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug("Health check request received")
    
    return {"status": "healthy"}

@app.post("/state/reset", response_model=Dict[str, Any])
async def reset_state():
    """Reset the scenario service state."""
    global scenarios
    scenarios = {}
    
    if get_verbosity() != VerbosityLevel.SILENT:
        logger.info("Scenario service state reset")
    
    return {"status": "success", "message": "Scenario service state reset"}

@app.get("/state/get", response_model=Dict[str, Any])
async def get_state():
    """Get current scenario service state."""
    if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
        logger.debug(f"Getting scenario service state: {len(scenarios)} scenarios")
    
    return {
        "scenarios": scenarios,
        "total_scenarios": len(scenarios),
        "scenario_ids": list(scenarios.keys())
    }

@app.get("/service/info", response_model=Dict[str, Any])
async def service_info():
    """Get service information and available actions."""
    return {
        "service": "scenario",
        "version": "1.0.0",
        "description": "Sentinel Grid Scenario Management Service",
        "available_actions": [
            "create_scenario",
            "list_scenarios", 
            "get_scenario",
            "activate_scenario",
            "reset_state",
            "get_state"
        ],
        "endpoints": {
            "POST /scenarios": "Create a new scenario",
            "GET /scenarios": "List all scenarios",
            "GET /scenarios/{id}": "Get specific scenario",
            "POST /scenarios/{id}/activate": "Activate scenario across all services",
            "GET /state/get": "Get current service state",
            "POST /state/reset": "Reset service state",
            "GET /service/health": "Health check",
            "GET /service/info": "Service information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sentinel Grid Scenario Service")
    parser.add_argument("--port", type=int, default=8005, help="Port to run the service on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the service on")
    parser.add_argument("--verbosity", choices=["silent", "minimal", "normal", "verbose", "debug"], 
                        default="normal", help="Verbosity level")
    args = parser.parse_args()
    
    # Configure verbosity
    from workshop.config import set_verbosity, VerbosityLevel
    set_verbosity(VerbosityLevel(args.verbosity))
    
    # Start the service
    uvicorn.run(app, host=args.host, port=args.port) 