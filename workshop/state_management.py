"""
State Management Utilities for Workshop
=====================================

This module contains all the state management functions needed for the workshop
to properly handle scenario initialization and service state coordination.
"""

import requests
import time
from rich.console import Console

from .state_models import ScenarioDefinition

console = Console()

# Service URLs - will be set by main workshop
SERVICE_URLS = {
    "grid": "http://localhost:8002",
    "emergency": "http://localhost:8003", 
    "traffic": "http://localhost:8004",
    "scenario": "http://localhost:8005"
}


def reset_all_service_states():
    """Reset state across all services to ensure clean test environment."""
    console.print("🔄 Resetting all service states...")
    
    reset_results = {}
    for service, url in SERVICE_URLS.items():
        try:
            response = requests.post(f"{url}/state/reset", timeout=10)
            if response.status_code == 200:
                reset_results[service] = "✅ Reset successful"
                console.print(f"  ✅ {service.upper()} state reset")
            else:
                error_msg = f"❌ Reset failed: HTTP {response.status_code}"
                reset_results[service] = error_msg
                console.print(
                    f"  ❌ {service.upper()} reset failed: "
                    f"HTTP {response.status_code}")
        except Exception as e:
            reset_results[service] = f"❌ Reset error: {str(e)}"
            console.print(f"  ❌ {service.upper()} reset error: {e}")
    
    # Wait a moment for states to stabilize
    time.sleep(2)
    
    return reset_results


def activate_scenario(scenario: ScenarioDefinition, scenario_name: str = None):
    """
    Activate a scenario by setting initial state across all services.
    This ensures all services start with the correct scenario state.
    """
    if scenario_name is None:
        scenario_name = scenario.name
        
    console.print(f"🎯 Activating scenario: {scenario_name}")
    
    try:
        # First reset all states
        reset_all_service_states()
        
        # Create scenario in scenario service
        scenario_response = requests.post(
            f"{SERVICE_URLS['scenario']}/scenarios", 
            json=scenario.dict(),
            timeout=10
        )
        
        if scenario_response.status_code != 200:
            console.print(
                "[yellow]Warning: Could not create scenario in scenario "
                "service[/yellow]")
            scenario_id = "manual_activation"
        else:
            scenario_data = scenario_response.json()
            scenario_id = scenario_data.get("id", "manual_activation")
            console.print(f"  📋 Scenario created with ID: {scenario_id}")
        
        # Activate scenario across all services
        activation_response = requests.post(
            f"{SERVICE_URLS['scenario']}/scenarios/{scenario_id}/activate",
            timeout=15
        )
        
        if activation_response.status_code == 200:
            console.print(
                f"  ✅ Scenario '{scenario_name}' activated successfully")
            activation_data = activation_response.json()
            
            # Show activation results
            for service, result in activation_data.get("results", {}).items():
                if "error" in result:
                    console.print(
                        f"    ❌ {service.upper()}: {result['error']}")
                else:
                    console.print(
                        f"    ✅ {service.upper()}: State initialized")
                    
            return True
        else:
            console.print(
                "[yellow]Scenario activation failed, trying manual state "
                "setting...[/yellow]")
            return manual_state_activation(scenario, scenario_name)
            
    except Exception as e:
        console.print(f"[yellow]Scenario service activation failed: {e}[/yellow]")
        console.print("[yellow]Attempting manual state setting...[/yellow]")
        return manual_state_activation(scenario, scenario_name)


def manual_state_activation(scenario: ScenarioDefinition, scenario_name: str):
    """Manually set state across services if scenario service activation fails."""
    console.print(f"  🔧 Manually setting state for: {scenario_name}")
    
    activation_results = {}
    
    try:
        # Set grid state
        grid_state = {
            "zones": {k: v.dict() for k, v in 
                      scenario.initial_state.zones.items()}
        }
        grid_response = requests.post(
            f"{SERVICE_URLS['grid']}/state/set", 
            json=grid_state, timeout=10)
        activation_results["grid"] = grid_response.status_code == 200
        status_icon = '✅' if activation_results['grid'] else '❌'
        console.print(f"    {status_icon} Grid state set")
        
        # Set emergency state  
        emergency_state = {
            "incidents": [incident.dict() for incident in 
                          scenario.initial_state.incidents],
            "drones": [drone.dict() for drone in 
                       scenario.initial_state.drones]
        }
        emergency_response = requests.post(
            f"{SERVICE_URLS['emergency']}/state/set", 
            json=emergency_state, timeout=10)
        activation_results["emergency"] = emergency_response.status_code == 200
        status_icon = '✅' if activation_results['emergency'] else '❌'
        console.print(f"    {status_icon} Emergency state set")
        
        # Set traffic state
        traffic_state = {
            "sectors": {k: v.dict() for k, v in 
                        scenario.initial_state.traffic.items()}
        }
        traffic_response = requests.post(
            f"{SERVICE_URLS['traffic']}/state/set", 
            json=traffic_state, timeout=10)
        activation_results["traffic"] = traffic_response.status_code == 200
        status_icon = '✅' if activation_results['traffic'] else '❌'
        console.print(f"    {status_icon} Traffic state set")
        
        success_count = sum(activation_results.values())
        total_services = len(activation_results)
        
        if success_count == total_services:
            console.print(
                f"  ✅ Manual state activation complete: "
                f"{success_count}/{total_services} services")
            return True
        else:
            console.print(
                f"  ⚠️ Partial state activation: "
                f"{success_count}/{total_services} services successful")
            return success_count > 0
            
    except Exception as e:
        console.print(f"  ❌ Manual state activation failed: {e}")
        return False


def verify_scenario_state(scenario: ScenarioDefinition, 
                         scenario_name: str = None):
    """Verify that services have the correct scenario state activated."""
    if scenario_name is None:
        scenario_name = scenario.name
        
    console.print(f"🔍 Verifying scenario state: {scenario_name}")
    
    verification_results = {}
    
    try:
        # Check grid state
        grid_response = requests.get(
            f"{SERVICE_URLS['grid']}/state/get", timeout=5)
        if grid_response.status_code == 200:
            grid_data = grid_response.json()
            expected_zones = len(scenario.initial_state.zones)
            actual_zones = len(grid_data.get("zones", {}))
            verification_results["grid"] = actual_zones >= expected_zones
            status_icon = '✅' if verification_results['grid'] else '❌'
            console.print(
                f"  {status_icon} Grid: {actual_zones} zones "
                f"(expected ≥{expected_zones})")
        else:
            verification_results["grid"] = False
            console.print("  ❌ Grid state check failed")
            
        # Check emergency state
        emergency_response = requests.get(
            f"{SERVICE_URLS['emergency']}/state/get", timeout=5)
        if emergency_response.status_code == 200:
            emergency_data = emergency_response.json()
            expected_incidents = len(scenario.initial_state.incidents)
            expected_drones = len(scenario.initial_state.drones)
            actual_incidents = len(emergency_data.get("incidents", {}))
            actual_drones = len(emergency_data.get("drones", {}))
            verification_results["emergency"] = (
                actual_incidents >= expected_incidents and 
                actual_drones >= expected_drones)
            status_icon = '✅' if verification_results['emergency'] else '❌'
            console.print(
                f"  {status_icon} Emergency: {actual_incidents} incidents, "
                f"{actual_drones} drones")
        else:
            verification_results["emergency"] = False
            console.print("  ❌ Emergency state check failed")
            
        # Check traffic state
        traffic_response = requests.get(
            f"{SERVICE_URLS['traffic']}/state/get", timeout=5)
        if traffic_response.status_code == 200:
            traffic_data = traffic_response.json()
            expected_sectors = len(scenario.initial_state.traffic)
            actual_sectors = len(traffic_data.get("sectors", {}))
            verification_results["traffic"] = actual_sectors >= expected_sectors
            status_icon = '✅' if verification_results['traffic'] else '❌'
            console.print(
                f"  {status_icon} Traffic: {actual_sectors} sectors "
                f"(expected ≥{expected_sectors})")
        else:
            verification_results["traffic"] = False
            console.print("  ❌ Traffic state check failed")
            
        success_count = sum(verification_results.values())
        total_services = len(verification_results)
        
        if success_count == total_services:
            console.print(
                f"  ✅ Scenario verification complete: "
                f"All {total_services} services ready")
            return True
        else:
            console.print(
                f"  ⚠️ Partial verification: "
                f"{success_count}/{total_services} services verified")
            return success_count > 0
            
    except Exception as e:
        console.print(f"  ❌ Scenario verification failed: {e}")
        return False


def get_actual_service_ids():
    """Get actual IDs from running services to avoid hardcoded scenario IDs."""
    try:
        # Get actual grid zones
        grid_response = requests.get(
            f"{SERVICE_URLS['grid']}/grid/report_status", timeout=5)
        grid_zones = []
        if grid_response.status_code == 200:
            grid_data = grid_response.json()
            grid_zones = list(grid_data.get("zones", {}).keys())
        
        # Get actual emergency drones and incidents  
        emergency_response = requests.get(
            f"{SERVICE_URLS['emergency']}/emergency/report_status", 
            timeout=5)
        drones = []
        incidents = []
        if emergency_response.status_code == 200:
            emergency_data = emergency_response.json()
            drones = list(emergency_data.get("drones", {}).keys())
            incidents = list(emergency_data.get("incidents", {}).keys())
        
        # Get actual traffic sectors (default fallback)
        traffic_sectors = ["S001", "S002", "S003", "S004", "S005"]
        
        return {
            "grid_zones": grid_zones,
            "drones": drones, 
            "incidents": incidents,
            "traffic_sectors": traffic_sectors
        }
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not get actual service IDs: {e}[/yellow]")
        # Return fallback IDs
        return {
            "grid_zones": ["Z001", "Z002", "Z003", "Z004", "Z005"],
            "drones": ["D001", "D002", "D003", "D004", "D005"],
            "incidents": ["E-1001", "E-1002", "E-1003", "E-1004", "E-1005"],
            "traffic_sectors": ["S001", "S002", "S003", "S004", "S005"]
        }


def get_system_status():
    """Get current status from all services."""
    status = {}
    for service, url in SERVICE_URLS.items():
        try:
            response = requests.get(f"{url}/state/get", timeout=5)
            if response.status_code == 200:
                status[service] = response.json()
            else:
                status[service] = {"error": "unavailable"}
        except Exception:
            status[service] = {"error": "connection failed"}
    return status 