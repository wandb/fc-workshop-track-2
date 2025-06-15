"""
Service Management Utilities for Workshop
=======================================

This module handles starting, stopping, and managing the workshop services.
"""

import os
import sys
import time
import signal
import subprocess
import requests
from rich.console import Console

console = Console()

# Global variables
service_processes = {}
SERVICE_URLS = {
    "grid": "http://localhost:8002",
    "emergency": "http://localhost:8003", 
    "traffic": "http://localhost:8004",
    "scenario": "http://localhost:8005"
}


def check_environment():
    """Check if all required components are available."""
    console.print("🔍 Checking environment setup...")
    
    # Check API keys
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ OpenAI API Key missing![/red]")
        console.print("Add to .env file: OPENAI_API_KEY=your_key_here")
        return False
    
    # Check packages
    try:
        import crewai  # noqa: F401
        from crewai import Agent, Task, Crew, Process  # noqa: F401
        from crewai.tools import BaseTool  # noqa: F401
        from pydantic import BaseModel, Field  # noqa: F401
        console.print("✅ All packages installed")
    except ImportError as e:
        console.print(f"[red]❌ Missing package: {e}[/red]")
        console.print(
            "Run: pip install crewai crewai-tools openai "
            "python-dotenv rich pydantic")
        return False
    
    console.print("✅ Environment setup complete!")
    return True


def start_services():
    """Start all SENTINEL GRID services."""
    global service_processes
    
    services = {
        "grid": 8002, 
        "emergency": 8003, 
        "traffic": 8004, 
        "scenario": 8005
    }
    
    console.print("🚀 Starting SENTINEL GRID Services...")
    
    for service, port in services.items():
        try:
            api_script = f"workshop/services/{service}_api.py"
            if not os.path.exists(api_script):
                console.print(f"[red]Missing:[/red] {api_script}")
                continue
                
            process = subprocess.Popen(
                [sys.executable, api_script, "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            service_processes[service] = process
            console.print(f"✅ {service.upper()} started on port {port}")
        except Exception as e:
            console.print(f"[red]Failed to start {service}:[/red] {e}")
    
    time.sleep(3)  # Wait for services to initialize
    
    # Verify services
    all_healthy = True
    for service, url in SERVICE_URLS.items():
        try:
            response = requests.get(f"{url}/service/health", timeout=5)
            if response.status_code == 200:
                console.print(f"✅ {service.upper()} healthy")
            else:
                console.print(f"❌ {service.upper()} unhealthy")
                all_healthy = False
        except Exception:
            console.print(f"❌ {service.upper()} not responding")
            all_healthy = False
    
    return all_healthy


def stop_services():
    """Stop all running services."""
    global service_processes
    if not service_processes:
        return
    
    console.print("🛑 Stopping services...")
    for service, process in service_processes.items():
        try:
            if sys.platform == "win32":
                subprocess.run([
                    "taskkill", "/F", "/T", "/PID", str(process.pid)
                ])
            else:
                os.kill(process.pid, signal.SIGTERM)
            console.print(f"✅ {service.upper()} service stopped")
        except Exception as e:
            console.print(f"⚠️ Error stopping {service}: {e}")
    service_processes = {}


def init_workshop_environment():
    """Initialize the complete workshop environment."""
    from .state_management import reset_all_service_states
    
    # Check environment
    environment_ok = check_environment()
    if not environment_ok:
        console.print(
            "[red]Please fix environment issues before continuing[/red]")
        return False
    
    # Start services
    services_running = start_services()
    if not services_running:
        console.print("[yellow]Some services failed to start[/yellow]")
        return False
    
    # Reset all service states for clean environment
    console.print("\n🔄 Initializing workshop environment...")
    reset_all_service_states()
    console.print("✅ Workshop environment ready")
    
    return True 


def save_experiment_results(workshop_results: dict, 
                            results_file: str = None):
    """Save current experiment results to file, overwriting previous results."""
    import json
    from datetime import datetime
    
    if results_file is None:
        results_file = "workshop_experiment_results.json"
    
    # Calculate command counts
    llm_data = workshop_results.get("llm_chain", {})
    rule_data = workshop_results.get("rule_based", {})
    agent_data = workshop_results.get("agent_system", {})
    
    llm_commands = len(llm_data.get("commands", []))
    rule_commands = len(rule_data.get("commands", []))
    agent_commands = len(agent_data.get("commands", []))
    total_commands = llm_commands + rule_commands + agent_commands
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_results": workshop_results,
        "summary": {
            "llm_chain_success": llm_data.get("success_rate", 0),
            "rule_based_success": rule_data.get("success_rate", 0), 
            "agent_system_success": agent_data.get("success_rate", 0),
            "total_commands_executed": total_commands
        }
    }
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        console.print(f"📊 Results saved to {results_file}")
        
        # Show more accurate progress summary
        summary = results_data['summary']
        
        # Check if we have service investigation data
        service_data = workshop_results.get("service_investigation", {})
        if service_data:
            # Show service-by-service progress if available
            grid_data = service_data.get("grid", {})
            emergency_data = service_data.get("emergency", {})
            traffic_data = service_data.get("traffic", {})
            
            if grid_data or emergency_data or traffic_data:
                console.print("    Service Investigation Progress:")
                if grid_data:
                    grid_rules = grid_data.get("rule_success_rate", 0)
                    grid_agents = grid_data.get("agent_success_rate", 0)
                    console.print(f"      Grid: Rules {grid_rules:.1%}, Agents {grid_agents:.1%}")
                if emergency_data:
                    emg_rules = emergency_data.get("rule_success_rate", 0)
                    emg_agents = emergency_data.get("agent_success_rate", 0)
                    console.print(f"      Emergency: Rules {emg_rules:.1%}, Agents {emg_agents:.1%}")
                if traffic_data:
                    traf_rules = traffic_data.get("rule_success_rate", 0)
                    traf_agents = traffic_data.get("agent_success_rate", 0)
                    console.print(f"      Traffic: Rules {traf_rules:.1%}, Agents {traf_agents:.1%}")
        else:
            # Show overall summary
            console.print(
                f"    LLM: {summary['llm_chain_success']:.1%}, "
                f"Rules: {summary['rule_based_success']:.1%}, "
                f"Agents: {summary['agent_system_success']:.1%}")
    except Exception as e:
        console.print(f"[red]Failed to save results: {e}[/red]")
        # Try to save to backup file
        try:
            backup_file = f"{results_file}.backup"
            with open(backup_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            console.print(
                f"[yellow]Results saved to backup: {backup_file}[/yellow]")
        except Exception as backup_error:
            console.print(
                f"[red]Backup save also failed: {backup_error}[/red]") 