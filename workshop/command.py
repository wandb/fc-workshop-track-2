from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import time
import logging
import traceback

from workshop.config import get_verbosity, VerbosityLevel, should_show

logger = logging.getLogger("command")

class ServiceType(str, Enum):
    """Types of services available in the system."""
    GRID = "grid"
    EMERGENCY = "emergency"
    TRAFFIC = "traffic"

class CommandStatus(str, Enum):
    """Status of a command execution."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"

class Command(BaseModel):
    """
    Standardized command structure for agents to interact with services.
    
    Commands follow a standardized format that can be validated, logged,
    and evaluated against optimal solutions.
    """
    service: ServiceType
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: Optional[int] = None
    command_id: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = time.time()
    
    @validator('action')
    def validate_action(cls, action, values):
        """Validate that the action is allowed for the given service."""
        service = values.get('service')
        if not service:
            return action
            
        # Define allowed actions for each service
        allowed_actions = {
            ServiceType.GRID: [
                "adjust_zone",      # Core: Adjust power capacity of zones
                "set_priority",     # Core: Set priority for critical infrastructure
                "report_status"     # Core: Get current grid status
            ],
            ServiceType.EMERGENCY: [
                "assign_drone",     # Core: Assign drone to incident
                "update_incident",  # Core: Update incident status
                "report_status"     # Core: Get emergency status
            ],
            ServiceType.TRAFFIC: [
                "redirect",         # Core: Redirect traffic between zones
                "report_conditions", # Core: Get traffic conditions
                "block_route"       # Core: Block problematic routes
            ]
        }
        
        if service not in allowed_actions:
            raise ValueError(f"Unknown service '{service}'")
            
        if action not in allowed_actions[service]:
            raise ValueError(f"Action '{action}' is not allowed for service '{service}'")
            
        return action

    def to_state_compatible_format(self) -> Dict[str, Any]:
        """
        Convert command to a format compatible with state models.
        
        This ensures that commands can be properly evaluated against
        the state models defined in utils/state_models.py.
        
        Returns:
            Dictionary format compatible with state models
        """
        result = {
            "service": self.service,
            "action": self.action,
            "parameters": self.parameters.copy()
        }
        
        # Log the conversion if verbosity is high enough
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            logger.debug(f"Converting command to state compatible format: {result}")
        
        # Standardize grid parameters
        if self.service == ServiceType.GRID:
            if self.action == "adjust_zone" and "zone_id" in self.parameters:
                # Ensure zone IDs are standardized
                zone_id = self.parameters["zone_id"]
                if not zone_id.startswith("zone_"):
                    result["parameters"]["zone_id"] = f"zone_{zone_id.lower()}"
                
                # Ensure capacity is a float between 0 and 1
                if "capacity" in self.parameters:
                    try:
                        result["parameters"]["capacity"] = float(self.parameters["capacity"])
                    except (ValueError, TypeError):
                        result["parameters"]["capacity"] = 0.8  # Default
        
        # Standardize emergency parameters
        elif self.service == ServiceType.EMERGENCY:
            if self.action == "assign_drone":
                # Ensure drone IDs are standardized
                if "drone_id" in self.parameters:
                    drone_id = self.parameters["drone_id"]
                    if not drone_id.startswith("drone_"):
                        result["parameters"]["drone_id"] = f"drone_{drone_id.lower()}"
                
                # Ensure incident IDs are standardized
                if "incident_id" in self.parameters:
                    incident_id = self.parameters["incident_id"]
                    if not incident_id.startswith("incident_"):
                        result["parameters"]["incident_id"] = f"incident_{incident_id.lower()}"
        
        # Standardize traffic parameters
        elif self.service == ServiceType.TRAFFIC:
            # Ensure sector IDs are standardized
            for param in ["sector", "from_sector", "to_sector"]:
                if param in self.parameters:
                    sector = self.parameters[param]
                    if not sector.startswith("S"):
                        result["parameters"][param] = f"S{sector.zfill(3)}"
        
        return result

class CommandResult(BaseModel):
    """Result of executing a command."""
    command: Command
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float  # Time in seconds to execute
    status: CommandStatus = CommandStatus.SUCCESS
    
    @property
    def service(self) -> ServiceType:
        """Get the service the command was executed against."""
        return self.command.service
    
    @property
    def action(self) -> str:
        """Get the action that was executed."""
        return self.command.action
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the parameters used for execution."""
        return self.command.parameters
    
    @property
    def is_successful(self) -> bool:
        """Check if the command execution was successful."""
        return self.success and self.status == CommandStatus.SUCCESS

class CommandPlan(BaseModel):
    """A plan consisting of multiple commands to be executed in sequence."""
    commands: List[Command]
    name: str
    description: Optional[str] = None
    
    def add_command(self, command: Command):
        """Add a command to the plan."""
        self.commands.append(command)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "commands": [cmd.dict() for cmd in self.commands]
        }
    
    def to_json(self) -> str:
        """Convert the plan to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

# Service-specific command parameter validators
class GridCommandParams:
    """Parameter validators for grid service commands."""
    
    @staticmethod
    def validate_adjust_zone(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for adjust_zone command."""
        required = ["zone_id", "capacity"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for adjust_zone command")
        
        if not 0.0 <= float(params["capacity"]) <= 1.0:
            raise ValueError(f"Capacity must be between 0.0 and 1.0, got {params['capacity']}")
        
        return params

    @staticmethod
    def validate_set_priority(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for set_priority command."""
        required = ["infrastructure_id", "level"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for set_priority command")
        
        allowed_levels = ["critical", "high", "medium", "low"]
        if params["level"] not in allowed_levels:
            raise ValueError(f"Level must be one of {allowed_levels}, got {params['level']}")
        
        return params
        
    @staticmethod
    def validate_emergency_shutdown(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for emergency_shutdown command."""
        required = ["zone_id", "reason"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for emergency_shutdown command")
        
        return params
        
    @staticmethod
    def validate_report_status(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for report_status command."""
        # Optional zone_id parameter
        return params
        
    @staticmethod
    def validate_forecast_load(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for forecast_load command."""
        # Optional zone_id and hours parameters
        if "hours" in params and not (1 <= int(params["hours"]) <= 24):
            raise ValueError(f"Hours must be between 1 and 24, got {params['hours']}")
            
        return params

class EmergencyCommandParams:
    """Parameter validators for emergency service commands."""
    
    @staticmethod
    def validate_assign_drone(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for assign_drone command."""
        required = ["drone_id", "incident_id"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for assign_drone command")
        
        return params
    
    @staticmethod
    def validate_update_incident(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for update_incident command."""
        required = ["incident_id", "status"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for update_incident command")
                
        allowed_statuses = ["active", "assigned", "in_progress", "resolved", "canceled"]
        if params["status"] not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}, got {params['status']}")
        
        return params
        
    @staticmethod
    def validate_report_status(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for report_status command."""
        # No required parameters
        return params

class TrafficCommandParams:
    """Parameter validators for traffic service commands."""
    
    @staticmethod
    def validate_redirect(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for redirect command."""
        required = ["sector_id", "target_reduction"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for redirect command")
                
        # Validate target_reduction is between 0 and 1
        target_reduction = float(params["target_reduction"])
        if not (0.0 <= target_reduction <= 1.0):
            raise ValueError(f"Target reduction must be between 0.0 and 1.0, got {target_reduction}")
            
        return params
    
    @staticmethod
    def validate_report_conditions(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for report_conditions command."""
        # No required parameters
        return params
        
    @staticmethod
    def validate_block_route(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for block_route command."""
        required = ["sector", "reason"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for block_route command")
                
        # Optional duration_minutes parameter
        if "duration_minutes" in params:
            duration = int(params["duration_minutes"])
            if not (5 <= duration <= 1440):  # 5 min to 24 hours
                raise ValueError(f"Duration must be between 5 and 1440 minutes, got {duration}")
                
        return params

# Command executor
class CommandExecutor:
    """Execute commands against services."""
    
    def __init__(self, base_url: str = "http://localhost"):
        """
        Initialize the command executor.
        
        Args:
            base_url: Base URL for service API endpoints
        """
        self.base_url = base_url
        self.service_ports = {
            ServiceType.GRID: 8002,
            ServiceType.EMERGENCY: 8003,
            ServiceType.TRAFFIC: 8004
        }
        self.validators = {
            ServiceType.GRID: {
                "adjust_zone": GridCommandParams.validate_adjust_zone,
                "set_priority": GridCommandParams.validate_set_priority,
                "report_status": GridCommandParams.validate_report_status,
                "forecast_load": GridCommandParams.validate_forecast_load,
                "emergency_shutdown": GridCommandParams.validate_emergency_shutdown
            },
            ServiceType.EMERGENCY: {
                "assign_drone": EmergencyCommandParams.validate_assign_drone,
                "update_incident": EmergencyCommandParams.validate_update_incident,
                "report_status": EmergencyCommandParams.validate_report_status
            },
            ServiceType.TRAFFIC: {
                "redirect": TrafficCommandParams.validate_redirect,
                "report_conditions": TrafficCommandParams.validate_report_conditions,
                "block_route": TrafficCommandParams.validate_block_route
            }
        }
    
    def execute(self, command: Command) -> CommandResult:
        """
        Execute a command against the appropriate service.
        
        Args:
            command: The command to execute
            
        Returns:
            Result of the command execution
        """
        import requests
        import time
        
        service = command.service
        action = command.action
        parameters = command.parameters
        
        # Get service URL
        port = self.service_ports.get(service)
        if not port:
            return CommandResult(
                command=command,
                success=False,
                error=f"Unknown service '{service}'",
                execution_time=0.0,
                status=CommandStatus.FAILURE
            )
        
        service_url = f"{self.base_url}:{port}"
        
        # Validate parameters based on service and action
        try:
            validator = self.validators.get(service, {}).get(action)
            if validator:
                parameters = validator(parameters)
        except ValueError as e:
            return CommandResult(
                command=command,
                success=False,
                error=str(e),
                execution_time=0.0,
                status=CommandStatus.FAILURE
            )
        
        # Map to correct endpoint and parameters
        endpoint, method, data = self._map_to_endpoint(service, action, parameters)
        if not endpoint:
            return CommandResult(
                command=command,
                success=False,
                error=f"Unknown action '{action}' for service '{service}'",
                execution_time=0.0,
                status=CommandStatus.FAILURE
            )
        
        # Full URL
        url = f"{service_url}{endpoint}"
        
        # Execute the request
        start_time = time.time()
        try:
            # Log based on verbosity level
            if should_show("api_calls"):
                logger.info(f"Executing {method} request to {url} with data {data}")
            elif get_verbosity() == VerbosityLevel.NORMAL:
                logger.info(f"Executing {action} on {service} service")
            
            if method == "GET":
                response = requests.get(url, params=data, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method == "PUT":
                response = requests.put(url, json=data, timeout=10)
            else:
                return CommandResult(
                    command=command,
                    success=False,
                    error=f"Unsupported method '{method}'",
                    execution_time=time.time() - start_time,
                    status=CommandStatus.FAILURE
                )
            
            # Check if request was successful
            if response.status_code >= 200 and response.status_code < 300:
                try:
                    result = response.json()
                except ValueError:
                    result = {"response": response.text}
                
                # Log success based on verbosity
                if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                    logger.debug(f"Command successful: {result}")
                
                return CommandResult(
                    command=command,
                    success=True,
                    result=result,
                    execution_time=time.time() - start_time,
                    status=CommandStatus.SUCCESS
                )
            else:
                error_message = f"Request failed with status code {response.status_code}"
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        error_message = f"{response.status_code} - {error_data}"
                except Exception:
                    if response.text:
                        error_message = f"{response.status_code} - {response.text}"
                
                # Enhanced error logging
                if get_verbosity() != VerbosityLevel.SILENT:
                    logger.error(f"Command failed: {error_message}")
                    logger.error(f"Request details: {method} {url}")
                    logger.error(f"Request data: {data}")
                    if get_verbosity() == VerbosityLevel.DEBUG:
                        logger.debug(f"Response headers: {dict(response.headers)}")
                
                return CommandResult(
                    command=command,
                    success=False,
                    error=error_message,
                    execution_time=time.time() - start_time,
                    status=CommandStatus.FAILURE
                )
        
        except requests.Timeout:
            if get_verbosity() != VerbosityLevel.SILENT:
                logger.error(f"Request timed out for {service}.{action}")
                logger.error(f"Request details: {method} {url}")
                logger.error(f"Request data: {data}")
            
            return CommandResult(
                command=command,
                success=False,
                error="Request timed out",
                execution_time=time.time() - start_time,
                status=CommandStatus.TIMEOUT
            )
        
        except Exception as e:
            if get_verbosity() != VerbosityLevel.SILENT:
                logger.error(f"Error executing command: {str(e)}")
                logger.error(f"Request details: {method} {url}")
                logger.error(f"Request data: {data}")
                if get_verbosity() == VerbosityLevel.DEBUG:
                    logger.error(traceback.format_exc())
            
            return CommandResult(
                command=command,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                status=CommandStatus.FAILURE
            )
    
    def _map_to_endpoint(self, service: ServiceType, action: str, parameters: Dict[str, Any]) -> tuple:
        """Map a command to its corresponding API endpoint and HTTP method."""
        # Define endpoint mappings for each service and action
        endpoint_mappings = {
            ServiceType.GRID: {
                "adjust_zone": ("/grid/zones/{zone_id}/capacity", "PUT"),
                "set_priority": ("/grid/infrastructure/{infrastructure_id}/priority", "POST"),
                "emergency_shutdown": ("/grid/emergency_shutdown", "POST"),
                "report_status": ("/grid/report_status", "GET"),
                "forecast_load": ("/grid/forecast_load", "POST")
            },
            ServiceType.EMERGENCY: {
                "assign_drone": ("/emergency/drones/{drone_id}/assign", "POST"),
                "update_incident": ("/emergency/incidents/{incident_id}", "POST"),
                "report_status": ("/emergency/report_status", "GET")
            },
            ServiceType.TRAFFIC: {
                "redirect": ("/traffic/redirect", "POST"),
                "report_conditions": ("/traffic/report_conditions", "POST"),
                "block_route": ("/traffic/block_route", "POST")
            }
        }
        
        if service not in endpoint_mappings:
            raise ValueError(f"Unknown service '{service}'")
            
        if action not in endpoint_mappings[service]:
            raise ValueError(f"Unknown action '{action}' for service '{service}'")
            
        endpoint, method = endpoint_mappings[service][action]
        
        # Create a copy of parameters for modification
        params_copy = parameters.copy()
        
        # Format the endpoint with path parameters and prepare body data
        try:
            endpoint = endpoint.format(**parameters)
            
            # For certain endpoints, remove path parameters from body data
            if service == ServiceType.EMERGENCY and action == "assign_drone":
                # drone_id is in path, incident_id goes in body
                params_copy.pop("drone_id", None)
            elif service == ServiceType.EMERGENCY and action == "update_incident":
                # incident_id is in path, status goes in body
                params_copy.pop("incident_id", None)
            elif service == ServiceType.GRID and action == "adjust_zone":
                # zone_id is in path, capacity goes in body
                params_copy.pop("zone_id", None)
            elif service == ServiceType.GRID and action == "set_priority":
                # infrastructure_id is in path, level goes in body
                params_copy.pop("infrastructure_id", None)
                
        except KeyError as e:
            raise ValueError(f"Missing required path parameter: {e}")
        
        # Return endpoint, method, and body parameters
        return endpoint, method, params_copy