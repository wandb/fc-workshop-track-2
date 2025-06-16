"""
Configuration settings for the SENTINEL GRID system.

This module provides centralized configuration for all components
of the SENTINEL GRID workshop, including logging verbosity and
display options.
"""

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional

# Verbosity levels
class VerbosityLevel(str, Enum):
    """Verbosity levels for system output."""
    SILENT = "silent"      # No output except errors
    MINIMAL = "minimal"    # Minimal output (only important info)
    NORMAL = "normal"      # Standard output
    VERBOSE = "verbose"    # Detailed output
    DEBUG = "debug"        # Full debug output


# Default configuration
DEFAULT_CONFIG = {
    "verbosity": VerbosityLevel.NORMAL,
    "log_to_file": False,
    "log_file": "sentinel_grid.log",
    "show_commands": False,
    "show_api_calls": False,
    "show_progress_bars": True,
    "show_panels": True,
    "show_states": False,  # Default to not showing states
    "use_rich_formatting": True,
    "service_ports": {
        "grid": 8002,
        "weather": 8001,
        "emergency": 8003,
        "traffic": 8004,
        "scenario": 8005
    },
    "base_url": "http://localhost"
}

# Current configuration (starts with defaults)
_config = DEFAULT_CONFIG.copy()

def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return _config.copy()

def update_config(new_config: Dict[str, Any]) -> None:
    """
    Update the configuration with new values.
    
    Args:
        new_config: Dictionary with configuration values to update
    """
    global _config
    _config.update(new_config)
    
    # Apply logging configuration changes
    _configure_logging()

# Set verbosity level (can be overridden with environment variable)
VERBOSITY_LEVEL = os.environ.get("VERBOSITY_LEVEL", "normal").lower()  # Ensure lowercase
try:
    verbosity = VerbosityLevel(VERBOSITY_LEVEL)
except ValueError:
    # Default to DEBUG if invalid value provided
    verbosity = VerbosityLevel.DEBUG
    print(f"Warning: Invalid verbosity level '{VERBOSITY_LEVEL}'. Using {verbosity} instead.")

# Configure logging
def configure_logging():
    """Configure logging based on verbosity level."""
    log_level = logging.INFO
    
    if verbosity == VerbosityLevel.DEBUG:
        log_level = logging.DEBUG
    elif verbosity == VerbosityLevel.SILENT:
        log_level = logging.ERROR
        
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Make third-party loggers less noisy
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Call this at import time
configure_logging()

def get_verbosity() -> VerbosityLevel:
    """Get the current verbosity level."""
    return verbosity

def set_verbosity(level: VerbosityLevel) -> None:
    """
    Set the verbosity level.
    
    Args:
        level: New verbosity level
    """
    global verbosity
    verbosity = level
    _configure_logging()

def _configure_logging() -> None:
    """Configure logging based on current settings."""
    # Map verbosity levels to logging levels
    log_levels = {
        VerbosityLevel.SILENT: logging.ERROR,
        VerbosityLevel.MINIMAL: logging.WARNING,
        VerbosityLevel.NORMAL: logging.INFO,
        VerbosityLevel.VERBOSE: logging.DEBUG,
        VerbosityLevel.DEBUG: logging.DEBUG
    }
    
    level = log_levels.get(verbosity, logging.INFO)
    
    # Reset existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Add file handler if enabled
    if _config["log_to_file"]:
        file_handler = logging.FileHandler(_config["log_file"])
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    # Add console handler (this will be replaced by Rich handler in main.py)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Set the logging level
    root_logger.setLevel(level)

def should_show(feature: str) -> bool:
    """
    Check if a specific feature should be shown based on verbosity.
    
    Args:
        feature: Feature to check (e.g., 'commands', 'api_calls')
        
    Returns:
        True if the feature should be shown, False otherwise
    """
    # Always show everything in DEBUG mode
    if verbosity == VerbosityLevel.DEBUG:
        return True
    
    # Check specific feature flags
    feature_map = {
        "commands": _config["show_commands"],
        "api_calls": _config["show_api_calls"],
        "progress": _config["show_progress_bars"],
        "panels": _config["show_panels"],
        "states": _config["show_states"]
    }
    
    # Restrict based on verbosity
    if verbosity == VerbosityLevel.SILENT:
        return False
    
    if verbosity == VerbosityLevel.MINIMAL:
        if feature in ["api_calls", "commands", "states"]:
            return False
    
    return feature_map.get(feature, True)

def get_service_url(service: str) -> str:
    """
    Get the URL for a specific service.
    
    Args:
        service: Service name (grid, weather, emergency, traffic, scenario)
        
    Returns:
        Full URL for the service
    """
    base = _config["base_url"]
    port = _config["service_ports"].get(service)
    if not port:
        raise ValueError(f"Unknown service: {service}")
    
    return f"{base}:{port}"

# Initialize logging configuration
_configure_logging()

# Load environment-based configuration
if "SENTINEL_VERBOSITY" in os.environ:
    try:
        level = VerbosityLevel(os.environ["SENTINEL_VERBOSITY"].lower())
        set_verbosity(level)
    except ValueError:
        # Invalid verbosity level in environment, ignore
        pass 