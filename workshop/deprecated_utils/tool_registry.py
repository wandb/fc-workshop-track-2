"""
Model Context Protocol (MCP) Tool Registry for agent tool discovery.

This module implements a registry system for tools with tagging, discovery,
and capability management based on the Model Context Protocol principles.
"""

import logging
from typing import Dict, List, Any, Optional
from rich.table import Table
from rich.console import Console

# Initialize console
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tool_registry")

class ToolRegistry:
    """
    A registry for tools with MCP-like discovery capabilities.
    
    This registry allows tools to be registered with tags and metadata,
    and provides methods for discovering tools based on their capabilities.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools = {}
        self.tags = {}
        self.capabilities = {}
        self.versions = {}
        log.info("Tool Registry initialized")
    
    def register_tool(self, tool, tags=None, metadata=None, capabilities=None, 
                     version="1.0"):
        """
        Register a tool with tags, metadata, capabilities, and version.
        
        Args:
            tool: The tool to register
            tags: List of tags for categorizing the tool
            metadata: Additional metadata about the tool
            capabilities: Dict of capabilities (e.g., {"works_in_rain": False})
            version: Tool version string
        """
        if not tags:
            tags = []
        if not metadata:
            metadata = {}
        if not capabilities:
            capabilities = {}
        
        self.tools[tool.name] = {
            "tool": tool,
            "tags": tags,
            "metadata": metadata,
            "capabilities": capabilities,
            "version": version
        }
        
        # Index by tags for faster lookup
        for tag in tags:
            if tag not in self.tags:
                self.tags[tag] = []
            self.tags[tag].append(tool.name)
        
        # Index by capabilities
        for capability, value in capabilities.items():
            if capability not in self.capabilities:
                self.capabilities[capability] = {}
            if value not in self.capabilities[capability]:
                self.capabilities[capability][value] = []
            self.capabilities[capability][value].append(tool.name)
        
        # Index by version
        if version not in self.versions:
            self.versions[version] = []
        self.versions[version].append(tool.name)
        
        log.info(f"Registered tool: [bold blue]{tool.name}[/bold blue] "
                f"v{version} with tags: {tags}")
    
    def get_tool(self, tool_name):
        """Get a tool by name."""
        if tool_name in self.tools:
            return self.tools[tool_name]["tool"]
        return None
    
    def get_tool_info(self, tool_name):
        """Get complete tool info including metadata, capabilities, etc."""
        if tool_name in self.tools:
            return self.tools[tool_name]
        return None
    
    def discover_tools_by_tag(self, tag):
        """Discover tools that have a specific tag."""
        if tag in self.tags:
            return [self.tools[tool_name]["tool"] 
                   for tool_name in self.tags[tag]]
        return []
    
    def discover_tools_by_capability(self, capability, value=None):
        """
        Discover tools based on specific capability requirements.
        
        Args:
            capability: Capability name to filter by
            value: Optional specific value of the capability
        
        Returns:
            List of tools matching the capability criteria
        """
        if capability not in self.capabilities:
            return []
        
        if value is not None:
            # Return tools with exact capability value
            if value in self.capabilities[capability]:
                return [self.tools[tool_name]["tool"] 
                       for tool_name in self.capabilities[capability][value]]
            return []
        else:
            # Return all tools with the capability regardless of value
            tool_names = []
            for val_tools in self.capabilities[capability].values():
                tool_names.extend(val_tools)
            return [self.tools[tool_name]["tool"] for tool_name in tool_names]
    
    def discover_tools_by_query(self, query):
        """
        Discover tools based on a semantic query.
        
        This would normally use embeddings or semantic search,
        but we'll use a simple keyword match for demonstration.
        """
        matching_tools = []
        query_terms = query.lower().split()
        
        for tool_name, tool_info in self.tools.items():
            # Check if any query term is in the tool name or tags
            if any(term in tool_name.lower() for term in query_terms):
                matching_tools.append(tool_info["tool"])
                continue
            
            # Check tags
            if any(any(term in tag.lower() for term in query_terms) 
                  for tag in tool_info["tags"]):
                matching_tools.append(tool_info["tool"])
                continue
            
            # Check description
            if any(term in tool_info["tool"].description.lower() 
                  for term in query_terms):
                matching_tools.append(tool_info["tool"])
        
        return matching_tools
    
    def discover_tools_for_agent(self, agent_profile):
        """
        Discover suitable tools for an agent based on its profile.
        
        Args:
            agent_profile: Dict with agent capabilities and preferences
        
        Returns:
            List of tools suitable for the agent
        """
        matching_tools = []
        required_capabilities = agent_profile.get("required_capabilities", {})
        preferred_tags = agent_profile.get("preferred_tags", [])
        
        # First check for tools that meet required capabilities
        for capability, value in required_capabilities.items():
            capability_tools = self.discover_tools_by_capability(capability, value)
            for tool in capability_tools:
                if tool not in matching_tools:
                    matching_tools.append(tool)
        
        # Then add tools with preferred tags that weren't already included
        for tag in preferred_tags:
            tag_tools = self.discover_tools_by_tag(tag)
            for tool in tag_tools:
                if tool not in matching_tools:
                    matching_tools.append(tool)
        
        return matching_tools
    
    def list_available_tools(self):
        """List all available tools with their tags and capabilities."""
        table = Table(title="Available Tools")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Version", style="blue")
        table.add_column("Tags", style="green")
        table.add_column("Capabilities", style="yellow")
        table.add_column("Description", style="magenta")
        
        for tool_name, tool_info in self.tools.items():
            table.add_row(
                tool_name,
                tool_info["version"],
                ", ".join(tool_info["tags"]),
                str(tool_info["capabilities"]),
                tool_info["tool"].description.split("\n")[0]  # First line of description
            )
        
        console.print(table)
        return list(self.tools.keys())

# Create a global instance of the tool registry
tool_registry = ToolRegistry() 