"""
Command evaluator for assessing agent command quality.

This module provides tools for evaluating the quality and effectiveness
of commands issued by agents in the SENTINEL GRID system.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
import logging
from rich.console import Console

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workshop.agent_system import ScenarioType
from workshop.scenarios import ScenarioEvaluation, MetricType
from workshop.config import get_verbosity, VerbosityLevel

# Configure logger
logger = logging.getLogger("command_evaluator")


class CommandEvaluator:
    """
    Evaluates commands issued by agents based on their appropriateness,
    effectiveness, and alignment with scenario metrics.
    """
    
    def __init__(self, evaluation_config: ScenarioEvaluation):
        """
        Initialize the command evaluator.
        
        Args:
            evaluation_config: Configuration for scenario evaluation
        """
        self.config = evaluation_config
        self.metric_calculators = self._initialize_metric_calculators()
        
        # Log initialization based on verbosity
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            logger.debug(f"CommandEvaluator initialized with config: {evaluation_config.name}")
    
    def _initialize_metric_calculators(self) -> Dict[str, callable]:
        """Initialize functions to calculate each metric."""
        calculators = {
            # Grid metrics
            "grid_stability": self._calculate_grid_stability,
            "power_conservation": self._calculate_power_conservation,
            "critical_infrastructure": self._calculate_critical_infrastructure,
            
            # Emergency metrics
            "incident_response": self._calculate_incident_response,
            "incident_resolution": self._calculate_incident_resolution,
            "drone_utilization": self._calculate_drone_utilization,
            "response_time": self._calculate_response_time,
            
            # Traffic metrics
            "traffic_flow": self._calculate_traffic_flow,
            "emergency_routing": self._calculate_emergency_routing,
            "congestion_management": self._calculate_congestion_management,
            
            # System-wide metrics
            "overall_coordination": self._calculate_overall_coordination,
            "resource_efficiency": self._calculate_resource_efficiency
        }
        
        if get_verbosity() == VerbosityLevel.DEBUG:
            logger.debug(f"Initialized {len(calculators)} metric calculators")
            
        return calculators
    
    def evaluate_commands(
        self, 
        commands: List[Dict[str, Any]], 
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate commands against scenario metrics.
        
        Args:
            commands: List of commands to evaluate
            current_state: Current state of all services
            
        Returns:
            Evaluation results
        """
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            logger.debug(f"Evaluating {len(commands)} commands")
            
        # Calculate current metric values
        current_metrics = self._calculate_current_metrics(current_state)
        
        # Evaluate each command's impact
        command_impacts = []
        for cmd in commands:
            impact = self._evaluate_command_impact(cmd, current_metrics)
            command_impacts.append(impact)
        
        # Calculate overall progress towards metrics
        metric_progress = self._calculate_metric_progress(current_metrics)
        
        # Compare against optimal commands
        optimal_comparison = self._compare_with_optimal(commands)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_progress, optimal_comparison)
        
        if get_verbosity() != VerbosityLevel.SILENT:
            logger.info(f"Evaluation complete - Overall score: {overall_score:.2f}")
            
        return {
            "current_metrics": current_metrics,
            "command_impacts": command_impacts,
            "metric_progress": metric_progress,
            "optimal_comparison": optimal_comparison,
            "overall_score": overall_score
        }
    
    def _calculate_current_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate current values for all metrics with verbose logging."""
        console = Console()
        
        if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            console.print("\n[bold yellow]📊 DETAILED METRIC CALCULATION[/bold yellow]")
        
        metrics = {}
        for metric in self.config.metrics:
            calculator = self.metric_calculators.get(metric.calculation)
            if calculator:
                metrics[metric.name] = calculator(state)
                
                if get_verbosity() in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                    progress = min(1.0, metrics[metric.name] / metric.target if metric.target != 0 else 0)
                    color = "green" if progress >= 0.8 else "yellow" if progress >= 0.5 else "red"
                    console.print(f"  • {metric.name}: [{color}]{metrics[metric.name]:.3f}[/{color}] (target: {metric.target}, progress: {progress:.1%})")
                    
                    # Add detailed breakdown for key metrics
                    if metric.name == "grid_stability":
                        self._log_grid_stability_details(state, console)
                    elif metric.name == "power_conservation":
                        self._log_power_conservation_details(state, console)
                    elif metric.name == "incident_response":
                        self._log_incident_response_details(state, console)
                        
                if get_verbosity() == VerbosityLevel.DEBUG:
                    logger.debug(f"Metric {metric.name}: {metrics[metric.name]:.2f}")
            else:
                if get_verbosity() != VerbosityLevel.SILENT:
                    logger.warning(f"No calculator found for metric: {metric.name}")
                metrics[metric.name] = 0.0
        return metrics
    
    def _log_grid_stability_details(self, state: Dict[str, Any], console):
        """Log detailed grid stability breakdown."""
        zones = state.get("zones", {})
        if not zones:
            console.print("    [red]⚠️  No zones found in state[/red]")
            return
            
        console.print("    [dim]Grid Stability Breakdown:[/dim]")
        total_stability = 0.0
        total_weight = 0.0
        
        for zone_id, zone in zones.items():
            stability = zone.get("stability", 0.0)
            is_critical = zone.get("is_critical", False)
            weight = 2.0 if is_critical else 1.0
            
            status = zone.get("status", "unknown")
            capacity = zone.get("capacity_kw", 0)
            load = zone.get("current_load_kw", 0)
            
            total_stability += stability * weight
            total_weight += weight
            
            status_color = "red" if status == "overloaded" else "yellow" if status == "stressed" else "green"
            critical_marker = "🔴" if is_critical else "⚪"
            
            console.print(f"      {critical_marker} {zone_id}: [{status_color}]{stability:.2f}[/{status_color}] "
                         f"(status: {status}, load: {load:.2f}/{capacity:.2f}kW, weight: {weight})")
        
        final_score = total_stability / total_weight if total_weight > 0 else 0.0
        console.print(f"    [bold]Final Score: {final_score:.3f}[/bold] (weighted average)")
    
    def _log_power_conservation_details(self, state: Dict[str, Any], console):
        """Log detailed power conservation breakdown."""
        zones = state.get("zones", {})
        if not zones:
            console.print("    [red]⚠️  No zones found in state[/red]")
            return
            
        console.print("    [dim]Power Conservation Breakdown:[/dim]")
        
        total_capacity = sum(zone.get("capacity_kw", 0.0) for zone in zones.values())
        total_load = sum(zone.get("current_load_kw", 0.0) for zone in zones.values())
        
        console.print(f"      Total Capacity: {total_capacity:.2f}kW")
        console.print(f"      Total Load: {total_load:.2f}kW")
        
        if total_capacity == 0:
            console.print("    [red]⚠️  Zero total capacity![/red]")
            return
        
        load_ratio = total_load / total_capacity
        console.print(f"      Load Ratio: {load_ratio:.3f} ({load_ratio:.1%})")
        
        # Show scoring logic with correct calculation
        if load_ratio <= 0.6:
            score = 0.6 + (0.6 - load_ratio) * 0.5
            console.print(f"      [green]Under-utilized[/green] (≤60%): score = 0.6 + (0.6 - {load_ratio:.3f}) * 0.5 = {score:.3f}")
        elif load_ratio <= 0.85:
            score = 0.9 + (0.85 - load_ratio) * 0.4
            console.print(f"      [green]Optimal range[/green] (60-85%): score = 0.9 + (0.85 - {load_ratio:.3f}) * 0.4 = {score:.3f}")
        elif load_ratio <= 0.95:
            score = 0.9 - (load_ratio - 0.85) * 3.0
            console.print(f"      [yellow]High utilization[/yellow] (85-95%): score = 0.9 - ({load_ratio:.3f} - 0.85) * 3.0 = {score:.3f}")
        else:
            score = max(0.0, 0.6 - (load_ratio - 0.95) * 6.0)
            console.print(f"      [red]Overloaded[/red] (>95%): score = max(0, 0.6 - ({load_ratio:.3f} - 0.95) * 6.0) = {score:.3f}")
        
        # Apply bounds and show final score
        final_score = max(0.0, min(1.0, score))
        if final_score != score:
            console.print(f"      [dim]Bounded to [0,1]: {final_score:.3f}[/dim]")
    
    def _log_incident_response_details(self, state: Dict[str, Any], console):
        """Log detailed incident response breakdown."""
        incidents = state.get("incidents", {})
        
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        if not incident_list:
            console.print("    [green]✅ No incidents (perfect score)[/green]")
            return
            
        console.print("    [dim]Incident Response Breakdown:[/dim]")
        console.print(f"      Total Incidents: {len(incident_list)}")
        
        assigned = sum(1 for inc in incident_list if inc.get("assigned_drone") is not None)
        in_progress = sum(1 for inc in incident_list if inc.get("status") == "in_progress")
        
        console.print(f"      Assigned Drones: {assigned}/{len(incident_list)} ({assigned/len(incident_list):.1%})")
        console.print(f"      In Progress: {in_progress}/{len(incident_list)} ({in_progress/len(incident_list):.1%})")
        
        base_score = assigned / len(incident_list)
        progress_bonus = (in_progress / len(incident_list)) * 0.2
        
        # Check traffic penalty
        traffic = state.get("traffic", {})
        traffic_penalty = 0.0
        if traffic:
            blocked_sectors = sum(1 for sector in traffic.values() if sector.get("is_blocked", False))
            traffic_penalty = (blocked_sectors / len(traffic)) * 0.1
            console.print(f"      Traffic Penalty: {blocked_sectors}/{len(traffic)} blocked sectors = -{traffic_penalty:.3f}")
        
        final_score = min(1.0, base_score + progress_bonus - traffic_penalty)
        console.print(f"      [bold]Final Score: {base_score:.3f} + {progress_bonus:.3f} - {traffic_penalty:.3f} = {final_score:.3f}[/bold]")
        
        # Show if score was bounded
        unbounded_score = base_score + progress_bonus - traffic_penalty
        if final_score != unbounded_score:
            console.print(f"      [dim]Bounded to [0,1]: {final_score:.3f}[/dim]")
    
    def _evaluate_command_impact(
        self,
        command: Dict[str, Any],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate the impact of a single command."""
        service = command.get("service", "unknown")
        action = command.get("action", "unknown")
        
        if get_verbosity() == VerbosityLevel.DEBUG:
            logger.debug(f"Evaluating impact of {service}.{action}")
            
        # Find matching optimal command
        optimal = next(
            (opt for opt in self.config.optimal_commands 
             if self._commands_match(command, opt.command)),
            None
        )
        
        if optimal:
            if get_verbosity() in [VerbosityLevel.NORMAL, VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                logger.info(f"Command {service}.{action} matches optimal command")
                
            return {
                "command": command,
                "optimal_match": True,
                "affected_metrics": optimal.affected_metrics,
                "expected_impact": optimal.expected_impact,
                "score": 1.0  # Perfect match
            }
        
        # Evaluate non-optimal command
        impact_analysis = self._analyze_command_impact(command, current_metrics)
        
        if get_verbosity() == VerbosityLevel.DEBUG:
            logger.debug(f"Command impact analysis: {impact_analysis}")
            
        return {
            "command": command,
            "optimal_match": False,
            "impact_analysis": impact_analysis
        }
    
    def _calculate_metric_progress(
        self, 
        current_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate progress towards each metric's target."""
        progress = {}
        for metric in self.config.metrics:
            current = current_metrics.get(metric.name, 0.0)
            if metric.type == MetricType.THRESHOLD:
                progress[metric.name] = {
                    "current": current,
                    "target": metric.target,
                    "achieved": current >= metric.target,
                    "progress": min(1.0, current / metric.target if metric.target != 0 else 0)
                }
            elif metric.type == MetricType.RANGE:
                min_val, max_val = metric.target
                in_range = min_val <= current <= max_val
                progress[metric.name] = {
                    "current": current,
                    "target": metric.target,
                    "achieved": in_range,
                    "progress": 1.0 if in_range else 0.0
                }
            # Additional metric types can be handled here if needed
        return progress
    
    def _compare_with_optimal(self, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare executed commands with optimal commands."""
        optimal_commands = [opt.command for opt in self.config.optimal_commands]
        matches = []
        
        for optimal in optimal_commands:
            matched = False
            for cmd in commands:
                if self._commands_match(cmd, optimal):
                    matches.append({
                        "optimal": optimal,
                        "executed": cmd,
                        "match": True
                    })
                    matched = True
                    break
            
            if not matched:
                matches.append({
                    "optimal": optimal,
                    "executed": None,
                    "match": False
                })
        
        match_count = sum(1 for m in matches if m["match"])
        score = match_count / len(optimal_commands) if optimal_commands else 0.0
        
        return {
            "matches": matches,
            "match_count": match_count,
            "total_optimal": len(optimal_commands),
            "score": score
        }
    
    def _calculate_overall_score(
        self,
        metric_progress: Dict[str, Dict[str, Any]],
        optimal_comparison: Dict[str, Any]
    ) -> float:
        """Calculate overall scenario score with reduced agent bias."""
        # Calculate weighted metric progress
        metric_score = 0.0
        total_weight = 0.0
        
        for metric in self.config.metrics:
            if metric.name in metric_progress:
                progress = metric_progress[metric.name]["progress"]
                metric_score += progress * metric.weight
                total_weight += metric.weight
        
        # Normalize by total weight
        if total_weight > 0:
            metric_score /= total_weight
        
        # FIXED: Reduce optimal command bias - agents shouldn't be penalized 
        # for taking different but effective approaches
        # Changed from 70/30 to 85/15 split to favor actual results over command matching
        return 0.85 * metric_score + 0.15 * optimal_comparison["score"]
    
    def _commands_match(self, cmd1: Dict[str, Any], cmd2: Dict[str, Any]) -> bool:
        """
        Check if two commands match based on service, action, and parameters.
        Enhanced to handle commands with minimal or missing parameters.
        
        Args:
            cmd1: First command (could be from agent with minimal params)
            cmd2: Second command (optimal command with full params)
            
        Returns:
            True if commands match, False otherwise
        """
        # Service and action must match exactly
        if cmd1.get("service") != cmd2.get("service") or cmd1.get("action") != cmd2.get("action"):
            return False
        
        # For parameters, handle cases where one might have minimal parameters
        params1 = cmd1.get("parameters", {})
        params2 = cmd2.get("parameters", {})
        
        # If either command has no parameters, consider it a match if service/action match
        if not params1 or not params2:
            logger.debug(f"Parameter-less match: {cmd1.get('service')}.{cmd1.get('action')}")
            return True
        
        # Check if key parameters match (be more lenient)
        key_params_match = True
        for key in params2.keys():
            if key in params1:
                # Allow some flexibility in parameter values
                val1, val2 = params1[key], params2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # For numeric values, allow 20% variance
                    if abs(val1 - val2) / max(abs(val2), 0.1) > 0.2:
                        key_params_match = False
                        break
                elif str(val1).lower() != str(val2).lower():
                    key_params_match = False
                    break
        
        if key_params_match:
            logger.debug(f"Parameter match: {cmd1.get('service')}.{cmd1.get('action')} - {params1} vs {params2}")
        
        return key_params_match

    def _analyze_command_impact(
        self,
        command: Dict[str, Any],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze the impact of a command with enhanced scoring for agent commands.
        """
        service = command.get("service", "")
        action = command.get("action", "")
        
        # Get metrics related to this service
        service_metrics = [
            m.name for m in self.config.metrics if m.service == service
        ]
        
        # Enhanced analysis with better scoring for common agent actions
        analysis = {
            "affected_metrics": service_metrics,
            "expected_impact": {m: 0.15 for m in service_metrics},  # Increased from 0.1 to 0.15
            "relevance_score": 0.75  # Increased from 0.6 to 0.75 - agents deserve more credit
        }
        
        # Specific analysis based on service and action with improved scoring
        if service == "grid":
            if action == "adjust_zone":
                # Grid zone adjustments are highly impactful
                if "grid_stability" in service_metrics:
                    analysis["expected_impact"]["grid_stability"] = 0.25  # Increased from 0.2
                    analysis["relevance_score"] = 0.95  # Increased from 0.9
                if "power_conservation" in service_metrics:
                    analysis["expected_impact"]["power_conservation"] = 0.3  # Increased from 0.25
                    
            elif action == "set_priority":
                # Infrastructure priority is also highly impactful
                if "critical_infrastructure" in service_metrics:
                    analysis["expected_impact"]["critical_infrastructure"] = 0.35  # Increased from 0.3
                if "grid_stability" in service_metrics:
                    analysis["expected_impact"]["grid_stability"] = 0.2  # Increased from 0.15
                analysis["relevance_score"] = 0.9  # Increased from 0.85
                
        elif service == "emergency":
            if action == "assign_drone":
                # Drone assignments are critical for emergency response
                if "incident_response" in service_metrics:
                    analysis["expected_impact"]["incident_response"] = 0.35  # Increased from 0.3
                if "response_time" in service_metrics:
                    analysis["expected_impact"]["response_time"] = 0.25  # Increased from 0.2
                if "drone_utilization" in service_metrics:
                    analysis["expected_impact"]["drone_utilization"] = 0.4  # Increased from 0.35
                analysis["relevance_score"] = 0.98  # Increased from 0.95
                
            elif action == "update_incident":
                # Incident updates show good coordination
                if "incident_response" in service_metrics:
                    analysis["expected_impact"]["incident_response"] = 0.25  # Increased from 0.2
                if "incident_resolution" in service_metrics:
                    analysis["expected_impact"]["incident_resolution"] = 0.3  # Increased from 0.25
                analysis["relevance_score"] = 0.85  # Increased from 0.8
                
        elif service == "traffic":
            if action == "redirect":
                # Traffic redirection helps with flow and emergency access
                if "traffic_flow" in service_metrics:
                    analysis["expected_impact"]["traffic_flow"] = 0.3  # Increased from 0.25
                if "emergency_routing" in service_metrics:
                    analysis["expected_impact"]["emergency_routing"] = 0.25  # Increased from 0.2
                if "congestion_management" in service_metrics:
                    analysis["expected_impact"]["congestion_management"] = 0.35  # Increased from 0.3
                analysis["relevance_score"] = 0.95  # Increased from 0.9
                
            elif action == "block_route":
                # Route blocking can help emergency access
                if "emergency_routing" in service_metrics:
                    analysis["expected_impact"]["emergency_routing"] = 0.3  # Increased from 0.25
                if "congestion_management" in service_metrics:
                    analysis["expected_impact"]["congestion_management"] = 0.25  # Increased from 0.2
                analysis["relevance_score"] = 0.8  # Increased from 0.75
        
        return analysis
    
    # Metric calculator functions
    
    # Grid metrics
    def _calculate_grid_stability(self, state: Dict[str, Any]) -> float:
        """Calculate overall grid stability metric."""
        zones = state.get("zones", {})
        if not zones:
            return 0.0
        
        # Average stability across all zones, weighted by criticality
        total_stability = 0.0
        total_weight = 0.0
        
        for zone in zones.values():
            stability = zone.get("stability", 0.0)
            # Critical zones get higher weight in stability calculation
            weight = 2.0 if zone.get("is_critical", False) else 1.0
            total_stability += stability * weight
            total_weight += weight
        
        return total_stability / total_weight if total_weight > 0 else 0.0
    
    def _calculate_power_conservation(self, state: Dict[str, Any]) -> float:
        """Calculate power conservation metric with realistic scoring."""
        zones = state.get("zones", {})
        if not zones:
            return 0.0
        
        # Calculate load efficiency using correct field names
        total_capacity = sum(zone.get("capacity_kw", 0.0) for zone in zones.values())
        total_load = sum(zone.get("current_load_kw", 0.0) for zone in zones.values())
        
        if total_capacity == 0:
            return 0.0
        
        load_ratio = total_load / total_capacity
        
        # FIXED: Realistic power conservation scoring with clear logic
        # Good conservation means using power efficiently without waste or overload
        if load_ratio <= 0.6:
            # Under-utilization - some waste but safe
            score = 0.6 + (0.6 - load_ratio) * 0.5  # 0.6 to 0.9 range
        elif load_ratio <= 0.85:
            # Optimal range - efficient use without overload
            score = 0.9 + (0.85 - load_ratio) * 0.4  # 0.9 to 1.0 range
        elif load_ratio <= 0.95:
            # High utilization - still acceptable but risky
            score = 0.9 - (load_ratio - 0.85) * 3.0  # 0.9 to 0.6 range
        else:
            # Overloaded - poor conservation due to inefficiency and risk
            score = max(0.0, 0.6 - (load_ratio - 0.95) * 6.0)  # 0.6 to 0.0 range
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _calculate_critical_infrastructure(self, state: Dict[str, Any]) -> float:
        """Calculate critical infrastructure support with realistic cross-service impact."""
        zones = state.get("zones", {})
        infrastructure = state.get("infrastructure", {})
        
        if not zones:
            return 0.0
            
        # Base score from critical zone stability
        critical_zones = [z for z in zones.values() if z.get("is_critical", False)]
        if critical_zones:
            zone_stability = sum(zone.get("stability", 0.0) for zone in critical_zones) / len(critical_zones)
        else:
            zone_stability = 0.8  # Default if no critical zones
        
        # Infrastructure priority scoring (more realistic)
        if infrastructure:
            critical_count = sum(1 for infra in infrastructure.values() 
                               if infra.get("level") == "critical")
            high_count = sum(1 for infra in infrastructure.values() 
                           if infra.get("level") == "high")
            total_count = len(infrastructure)
            
            # Score based on proportion of critical/high priority infrastructure
            priority_score = (critical_count * 1.0 + high_count * 0.7) / total_count
        else:
            priority_score = 0.5
        
        # Cross-service impact: emergency response affects infrastructure support
        incidents = state.get("incidents", {})
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        emergency_impact = 1.0
        if incident_list:
            # Active incidents in critical areas reduce infrastructure support
            critical_zone_ids = [z.get("zone_id") for z in critical_zones]
            critical_incidents = [inc for inc in incident_list 
                                if inc.get("zone") in critical_zone_ids and 
                                   inc.get("status") in ["active", "assigned"]]
            
            if critical_incidents:
                # Reduce score based on unresolved critical incidents
                unresolved_ratio = len(critical_incidents) / len(incident_list)
                emergency_impact = max(0.5, 1.0 - unresolved_ratio * 0.4)
        
        # Combine factors with realistic weighting
        return (zone_stability * 0.4 + priority_score * 0.4 + emergency_impact * 0.2)
    
    # Emergency metrics
    def _calculate_incident_response(self, state: Dict[str, Any]) -> float:
        """Calculate incident response effectiveness with cross-service considerations."""
        incidents = state.get("incidents", {})
        
        # If there are no incidents, return perfect score
        if not incidents:
            return 1.0
        
        # Handle both dictionary and list formats
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
        
        if not incident_list:
            return 1.0
        
        # Calculate base assignment rate
        assigned = sum(1 for inc in incident_list 
                      if inc.get("assigned_drone") is not None)
        base_score = assigned / len(incident_list)
        
        # Bonus for incidents in progress (shows active response)
        in_progress = sum(1 for inc in incident_list 
                         if inc.get("status") == "in_progress")
        progress_bonus = (in_progress / len(incident_list)) * 0.2
        
        # Consider traffic conditions for emergency access
        traffic = state.get("traffic", {})
        if traffic:
            # Penalty for blocked routes that impede emergency response
            blocked_sectors = sum(1 for sector in traffic.values() 
                                if sector.get("is_blocked", False))
            traffic_penalty = (blocked_sectors / len(traffic)) * 0.1
        else:
            traffic_penalty = 0.0
        
        # Calculate final score and ensure it's between 0 and 1
        final_score = base_score + progress_bonus - traffic_penalty
        return max(0.0, min(1.0, final_score))
    
    def _calculate_incident_resolution(self, state: Dict[str, Any]) -> float:
        """Calculate incident resolution metric."""
        incidents = state.get("incidents", {})
        if not incidents:
            return 1.0
        
        # Handle both dictionary and list formats
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        if not incident_list:
            return 1.0
            
        # Calculate percentage of resolved incidents
        resolved = sum(1 for inc in incident_list if inc.get("status") == "resolved")
        in_progress = sum(1 for inc in incident_list if inc.get("status") == "in_progress")
        
        # Give partial credit for incidents in progress
        return (resolved + (in_progress * 0.5)) / len(incident_list)
    
    def _calculate_drone_utilization(self, state: Dict[str, Any]) -> float:
        """Calculate drone utilization metric."""
        drones = state.get("drones", {})
        if not drones:
            return 0.0
        
        # Handle both dictionary and list formats
        if isinstance(drones, dict):
            drone_list = list(drones.values())
        else:
            drone_list = drones
            
        if not drone_list:
            return 0.0
            
        # Calculate percentage of drones in active use
        active = sum(1 for drone in drone_list 
                    if drone.get("status") in ["assigned", "en_route", "on_site"])
        available = sum(1 for drone in drone_list 
                       if drone.get("status") != "disabled")
        
        if available == 0:
            return 0.0
            
        return active / available
    
    def _calculate_response_time(self, state: Dict[str, Any]) -> float:
        """Calculate response time metric considering traffic conditions."""
        incidents = state.get("incidents", {})
        traffic = state.get("traffic", {})
        
        if not incidents:
            return 1.0
        
        # Handle both dictionary and list formats
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        # Only consider incidents with drones assigned
        assigned_incidents = [inc for inc in incident_list 
                             if inc.get("assigned_drone") is not None]
        if not assigned_incidents:
            return 0.0
        
        # Base response time score
        base_score = 0.8
        
        # Adjust based on traffic conditions
        if traffic:
            avg_congestion = sum(sector.get("congestion_level", 0.0) 
                               for sector in traffic.values()) / len(traffic)
            # Higher congestion reduces response time effectiveness
            traffic_impact = avg_congestion * 0.3
            base_score = max(0.2, base_score - traffic_impact)
        
        return base_score
    
    # Traffic metrics
    def _calculate_traffic_flow(self, state: Dict[str, Any]) -> float:
        """Calculate traffic flow metric with realistic cross-service considerations."""
        traffic = state.get("traffic", {})
        if not traffic:
            return 0.7  # Neutral score if no traffic data
        
        # Calculate flow based on congestion levels and blocked status
        total_flow = 0.0
        for sector in traffic.values():
            congestion = sector.get("congestion_level", 0.0)
            is_blocked = sector.get("is_blocked", False)
            
            if is_blocked:
                flow = 0.1  # Minimal flow if blocked (not zero - emergency access)
            else:
                # Flow decreases with congestion but not linearly
                flow = max(0.1, 1.0 - (congestion ** 1.5))  # Non-linear penalty
            
            total_flow += flow
        
        base_score = total_flow / len(traffic)
        
        # Cross-service bonus: good traffic flow helps emergency response
        incidents = state.get("incidents", {})
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        if incident_list:
            active_incidents = [inc for inc in incident_list 
                              if inc.get("status") in ["assigned", "in_progress"]]
            if active_incidents:
                # Bonus for maintaining flow during emergencies
                emergency_bonus = min(0.1, base_score * 0.2)
                base_score += emergency_bonus
        
        return min(1.0, base_score)
    
    def _calculate_emergency_routing(self, state: Dict[str, Any]) -> float:
        """Calculate emergency routing with improved realism."""
        traffic = state.get("traffic", {})
        incidents = state.get("incidents", {})
        
        if not traffic:
            return 0.5  # Neutral if no traffic data
        
        # Handle incidents format
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        if not incident_list:
            return 0.9  # Good routing if no emergencies
        
        # Calculate routing effectiveness based on traffic conditions
        avg_congestion = sum(sector.get("congestion_level", 0.0) 
                           for sector in traffic.values()) / len(traffic)
        blocked_ratio = sum(1 for sector in traffic.values() 
                          if sector.get("is_blocked", False)) / len(traffic)
        
        # Base routing score (lower congestion = better routing)
        routing_score = max(0.2, 1.0 - avg_congestion * 0.8)
        
        # Penalty for blocked routes (but not too harsh - some blocking may be strategic)
        blocking_penalty = blocked_ratio * 0.3
        routing_score = max(0.1, routing_score - blocking_penalty)
        
        # Bonus for active emergency response
        active_emergencies = sum(1 for inc in incident_list 
                               if inc.get("status") in ["assigned", "in_progress"])
        if active_emergencies > 0:
            response_effectiveness = active_emergencies / len(incident_list)
            # Good emergency response improves routing score
            routing_score += response_effectiveness * 0.2
        
        return min(1.0, routing_score)
    
    def _calculate_congestion_management(self, state: Dict[str, Any]) -> float:
        """Calculate congestion management with realistic thresholds."""
        traffic = state.get("traffic", {})
        if not traffic:
            return 0.5  # Neutral if no traffic data
        
        # More realistic congestion thresholds
        excellent = sum(1 for sector in traffic.values() 
                       if sector.get("congestion_level", 0.0) <= 0.2)
        good = sum(1 for sector in traffic.values() 
                  if 0.2 < sector.get("congestion_level", 0.0) <= 0.4)
        moderate = sum(1 for sector in traffic.values() 
                      if 0.4 < sector.get("congestion_level", 0.0) <= 0.6)
        poor = sum(1 for sector in traffic.values() 
                  if 0.6 < sector.get("congestion_level", 0.0) <= 0.8)
        critical = sum(1 for sector in traffic.values() 
                      if sector.get("congestion_level", 0.0) > 0.8)
        
        total_sectors = len(traffic)
        
        # Weighted scoring with realistic expectations
        score = (excellent * 1.0 + good * 0.8 + moderate * 0.6 + 
                poor * 0.3 + critical * 0.0) / total_sectors
        
        # Cross-service consideration: emergency incidents affect acceptable congestion
        incidents = state.get("incidents", {})
        if isinstance(incidents, dict):
            incident_list = list(incidents.values())
        else:
            incident_list = incidents
            
        if incident_list:
            active_incidents = [inc for inc in incident_list 
                              if inc.get("status") in ["active", "assigned", "in_progress"]]
            if active_incidents:
                # During emergencies, higher congestion tolerance is acceptable
                # if it's due to strategic traffic management
                emergency_adjustment = min(0.1, len(active_incidents) / len(incident_list) * 0.15)
                score += emergency_adjustment
        
        return min(1.0, score)
    
    # System-wide metrics
    def _calculate_overall_coordination(self, state: Dict[str, Any]) -> float:
        """Calculate realistic cross-service coordination."""
        zones = state.get("zones", {})
        incidents = state.get("incidents", {})
        traffic = state.get("traffic", {})
        
        coordination_score = 0.0
        coordination_factors = 0
        
        # Factor 1: Grid-Emergency coordination
        if zones and incidents:
            if isinstance(incidents, dict):
                incident_list = list(incidents.values())
            else:
                incident_list = incidents
                
            critical_zones = [z for z in zones.values() if z.get("is_critical", False)]
            
            if critical_zones and incident_list:
                # Check if critical zones have adequate power during emergencies
                critical_zone_ids = [z.get("zone_id") for z in critical_zones]
                critical_incidents = [inc for inc in incident_list 
                                    if inc.get("zone") in critical_zone_ids]
                
                if critical_incidents:
                    # Coordination score based on grid stability in emergency zones
                    emergency_zones = [z for z in critical_zones 
                                     if z.get("zone_id") in critical_zone_ids]
                    if emergency_zones:
                        avg_stability = sum(z.get("stability", 0.0) for z in emergency_zones) / len(emergency_zones)
                        coordination_score += avg_stability
                    else:
                        coordination_score += 0.7
                else:
                    coordination_score += 0.9  # No emergencies in critical zones
                coordination_factors += 1
        
        # Factor 2: Traffic-Emergency coordination
        if traffic and incidents:
            if isinstance(incidents, dict):
                incident_list = list(incidents.values())
            else:
                incident_list = incidents
                
            active_incidents = [inc for inc in incident_list 
                              if inc.get("status") in ["assigned", "in_progress"]]
            
            if active_incidents:
                # Good coordination = low congestion during active emergencies
                avg_congestion = sum(sector.get("congestion_level", 0.0) 
                                   for sector in traffic.values()) / len(traffic)
                # Inverse relationship: lower congestion = better coordination
                traffic_coordination = max(0.3, 1.0 - avg_congestion * 0.7)
                coordination_score += traffic_coordination
            else:
                coordination_score += 0.8  # Neutral if no active emergencies
            coordination_factors += 1
        
        # Factor 3: Grid-Traffic coordination
        if zones and traffic:
            # Stable grid should support traffic management systems
            avg_stability = sum(zone.get("stability", 0.0) for zone in zones.values()) / len(zones)
            avg_congestion = sum(sector.get("congestion_level", 0.0) 
                               for sector in traffic.values()) / len(traffic)
            
            # Good coordination = high stability enables good traffic management
            grid_traffic_score = (avg_stability * 0.6) + ((1.0 - avg_congestion) * 0.4)
            coordination_score += grid_traffic_score
            coordination_factors += 1
        
        if coordination_factors == 0:
            return 0.5  # Default neutral score
        
        return coordination_score / coordination_factors
    
    def _calculate_resource_efficiency(self, state: Dict[str, Any]) -> float:
        """Calculate realistic resource efficiency across all services."""
        efficiency_score = 0.0
        efficiency_factors = 0
        
        # Drone utilization efficiency (realistic expectations)
        drones = state.get("drones", {})
        if drones:
            if isinstance(drones, dict):
                drone_list = list(drones.values())
            else:
                drone_list = drones
                
            if drone_list:
                active = sum(1 for drone in drone_list 
                           if drone.get("status") in ["assigned", "en_route", "on_site"])
                available = sum(1 for drone in drone_list 
                              if drone.get("status") != "disabled")
                
                if available > 0:
                    utilization = active / available
                    # Realistic scoring: 70-90% utilization is optimal
                    if utilization <= 0.7:
                        drone_efficiency = utilization / 0.7 * 0.8  # Scale to 0.8 max
                    elif utilization <= 0.9:
                        drone_efficiency = 0.8 + (utilization - 0.7) * 1.0  # 0.8 to 1.0
                    else:
                        drone_efficiency = max(0.7, 1.0 - (utilization - 0.9) * 2.0)  # Penalty for overuse
                    
                    efficiency_score += drone_efficiency
                    efficiency_factors += 1
        
        # Power efficiency (from power conservation)
        power_efficiency = self._calculate_power_conservation(state)
        efficiency_score += power_efficiency
        efficiency_factors += 1
        
        # Traffic efficiency (realistic flow expectations)
        traffic_efficiency = self._calculate_traffic_flow(state)
        efficiency_score += traffic_efficiency
        efficiency_factors += 1
        
        return efficiency_score / efficiency_factors if efficiency_factors > 0 else 0.0


def evaluate_scenario_commands(
    commands: List[Dict[str, Any]],
    scenario_type: ScenarioType,
    constraints: Optional[Dict[str, Any]] = None,
    current_state: Optional[Dict[str, Any]] = None,
    scenario_definition: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate commands for a specific scenario using the actual scenario definition.
    
    Args:
        commands: List of commands to evaluate
        scenario_type: Type of scenario being evaluated
        constraints: Optional scenario constraints
        current_state: Optional current state of all services
        scenario_definition: The actual scenario definition to use for evaluation
        
    Returns:
        Evaluation results
    """
    # Load scenario evaluation configuration
    from workshop.scenarios import (
        MetricDefinition,
        MetricType,
        CommandImpact,
        ScenarioEvaluation
    )
    
    # If we have a scenario definition, use it to create proper evaluation config
    if scenario_definition:
        # Extract metrics from scenario success criteria
        metrics = []
        success_criteria = scenario_definition.success_criteria
        
        for metric_name, target_value in success_criteria.metrics.items():
            # Map metric names to appropriate calculators and services
            metric_service = "system"  # Default
            calculator = metric_name
            
            if "grid" in metric_name or "power" in metric_name or "stability" in metric_name:
                metric_service = "grid"
            elif "incident" in metric_name or "emergency" in metric_name or "drone" in metric_name:
                metric_service = "emergency"
            elif "traffic" in metric_name or "routing" in metric_name or "congestion" in metric_name:
                metric_service = "traffic"
            
            # Get weight from command_weights or use equal weighting
            weight = scenario_definition.command_weights.get(metric_service, 1.0)
            if metric_service == "system":
                weight = 0.2  # System metrics get lower weight
            
            metrics.append(MetricDefinition(
                name=metric_name,
                description=f"Evaluation of {metric_name} for {scenario_definition.name}",
                type=MetricType.THRESHOLD,
                target=target_value,
                weight=weight,
                service=metric_service,
                calculation=calculator
            ))
        
        # Create command impacts from optimal commands
        command_impacts = []
        for optimal_cmd in scenario_definition.optimal_commands:
            # Determine which metrics this command affects based on service
            service = optimal_cmd.get("service", "unknown")
            affected_metrics = []
            expected_impact = {}
            
            # Map service to likely affected metrics
            for metric in metrics:
                if (metric.service == service or 
                    (service == "grid" and "grid" in metric.name) or
                    (service == "emergency" and ("incident" in metric.name or "emergency" in metric.name)) or
                    (service == "traffic" and ("traffic" in metric.name or "routing" in metric.name))):
                    affected_metrics.append(metric.name)
                    # Base impact proportional to metric weight
                    expected_impact[metric.name] = min(0.3, metric.weight * 0.5)
            
            if affected_metrics:
                command_impacts.append(CommandImpact(
                    command=optimal_cmd,
                    affected_metrics=affected_metrics,
                    expected_impact=expected_impact
                ))
        
        # Create evaluation config from scenario definition
        evaluation_config = ScenarioEvaluation(
            name=scenario_definition.name,
            description=scenario_definition.description,
            metrics=metrics,
            optimal_commands=command_impacts,
            constraints=constraints or {}
        )
    
    else:
        # Fallback: create basic evaluation based on scenario type
        if scenario_type == ScenarioType.GRID_SURGE:
            metrics = [
                MetricDefinition(
                    name="grid_stability",
                    description="Overall grid stability during crisis",
                    type=MetricType.THRESHOLD,
                    target=0.7,
                    weight=0.3,
                    service="grid",
                    calculation="grid_stability"
                ),
                MetricDefinition(
                    name="incident_response",
                    description="Emergency incident response effectiveness",
                    type=MetricType.THRESHOLD,
                    target=0.8,
                    weight=0.4,
                    service="emergency",
                    calculation="incident_response"
                ),
                MetricDefinition(
                    name="traffic_flow",
                    description="Maintaining traffic flow during crisis",
                    type=MetricType.THRESHOLD,
                    target=0.6,
                    weight=0.3,
                    service="traffic",
                    calculation="traffic_flow"
                )
            ]
            command_impacts = []
            
        elif scenario_type == ScenarioType.MEDICAL_EMERGENCY:
            metrics = [
                MetricDefinition(
                    name="incident_response",
                    description="Response to medical incidents",
                    type=MetricType.THRESHOLD,
                    target=0.9,
                    weight=0.5,
                    service="emergency",
                    calculation="incident_response"
                ),
                MetricDefinition(
                    name="emergency_routing",
                    description="Clear routes for medical response",
                    type=MetricType.THRESHOLD,
                    target=0.8,
                    weight=0.5,
                    service="traffic",
                    calculation="emergency_routing"
                )
            ]
            command_impacts = []
            
        else:
            # Default metrics for unknown scenario types
            metrics = [
                MetricDefinition(
                    name="overall_coordination",
                    description="General system coordination",
                    type=MetricType.THRESHOLD,
                    target=0.7,
                    weight=1.0,
                    service="system",
                    calculation="overall_coordination"
                )
            ]
            command_impacts = []
        
        evaluation_config = ScenarioEvaluation(
            name=f"{scenario_type.value} Scenario",
            description=f"Evaluation for {scenario_type.value}",
            metrics=metrics,
            optimal_commands=command_impacts,
            constraints=constraints or {}
        )
    
    # Create evaluator
    evaluator = CommandEvaluator(evaluation_config)
    
    # Normalize state structure to work with our evaluator
    state = _normalize_state(current_state or {})
    
    # Evaluate commands
    return evaluator.evaluate_commands(commands, state)

def _normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize state structure to ensure consistent access patterns.
    Handle both raw service states and structured scenario states.
    
    Args:
        state: Raw state dictionary from services or scenario
        
    Returns:
        Normalized state dictionary
    """
    normalized_state = {}
    
    # Handle zones - could be dict, list, or nested in 'grid'
    if "zones" in state:
        zones = state["zones"]
    elif "grid" in state and "zones" in state["grid"]:
        zones = state["grid"]["zones"]
    else:
        zones = {}
        
    if isinstance(zones, list):
        normalized_zones = {}
        for zone in zones:
            zone_id = zone.get("id") or zone.get("zone_id")
            if zone_id:
                # Normalize zone field names
                normalized_zone = {
                    "id": zone_id,
                    "stability": zone.get("stability", 0.0),
                    "capacity_kw": zone.get("capacity_kw") or zone.get("capacity", 0.0),
                    "current_load_kw": zone.get("current_load_kw") or zone.get("current_load", 0.0),
                    "is_critical": zone.get("is_critical", False),
                    "status": zone.get("status", "normal"),
                    "zone_id": zone_id
                }
                normalized_zones[zone_id] = normalized_zone
        normalized_state["zones"] = normalized_zones
    else:
        # Already a dict, just ensure consistent field names
        normalized_zones = {}
        for zone_id, zone in zones.items():
            if isinstance(zone, dict):
                normalized_zone = {
                    "id": zone_id,
                    "stability": zone.get("stability", 0.0),
                    "capacity_kw": zone.get("capacity_kw") or zone.get("capacity", 0.0),
                    "current_load_kw": zone.get("current_load_kw") or zone.get("current_load", 0.0),
                    "is_critical": zone.get("is_critical", False),
                    "status": zone.get("status", "normal"),
                    "zone_id": zone_id
                }
                normalized_zones[zone_id] = normalized_zone
        normalized_state["zones"] = normalized_zones
    
    # Handle incidents - could be dict, list, or nested in 'emergency'
    if "incidents" in state:
        incidents = state["incidents"]
    elif "emergency" in state and "incidents" in state["emergency"]:
        incidents = state["emergency"]["incidents"]
    else:
        incidents = []
        
    if isinstance(incidents, list):
        normalized_incidents = {}
        for incident in incidents:
            incident_id = incident.get("id") or incident.get("incident_id")
            if incident_id:
                # Normalize incident field names
                normalized_incident = {
                    "id": incident_id,
                    "status": incident.get("status", "active"),
                    "assigned_drone": incident.get("assigned_drone"),
                    "zone": incident.get("zone") or incident.get("location"),
                    "urgency": incident.get("urgency", 0.0),
                    "description": incident.get("description", "")
                }
                normalized_incidents[incident_id] = normalized_incident
        normalized_state["incidents"] = normalized_incidents
    else:
        # Already a dict format
        normalized_state["incidents"] = incidents
    
    # Handle drones - could be dict, list, or nested in 'emergency'
    if "drones" in state:
        drones = state["drones"]
    elif "emergency" in state and "drones" in state["emergency"]:
        drones = state["emergency"]["drones"]
    else:
        drones = []
        
    if isinstance(drones, list):
        normalized_drones = {}
        for drone in drones:
            drone_id = drone.get("id") or drone.get("drone_id")
            if drone_id:
                # Normalize drone field names
                normalized_drone = {
                    "id": drone_id,
                    "status": drone.get("status", "available"),
                    "capabilities": drone.get("capabilities", []),
                    "speed": drone.get("speed", 1.0),
                    "assigned_incident": drone.get("assigned_incident")
                }
                normalized_drones[drone_id] = normalized_drone
        normalized_state["drones"] = normalized_drones
    else:
        # Already a dict format
        normalized_state["drones"] = drones
    
    # Handle traffic - could be dict, list, or nested in 'traffic'
    if "traffic" in state:
        traffic = state["traffic"]
    elif "traffic_management" in state:
        traffic = state["traffic_management"]
    else:
        traffic = {}
        
    if isinstance(traffic, list):
        normalized_traffic = {}
        for sector in traffic:
            sector_id = sector.get("id") or sector.get("zone_id") or sector.get("sector_id")
            if sector_id:
                # Normalize traffic field names
                normalized_sector = {
                    "id": sector_id,
                    "zone_id": sector_id,
                    "congestion_level": sector.get("congestion_level") or sector.get("congestion", 0.0),
                    "is_blocked": sector.get("is_blocked") or sector.get("blocked", False),
                    "description": sector.get("description", "")
                }
                normalized_traffic[sector_id] = normalized_sector
        normalized_state["traffic"] = normalized_traffic
    else:
        # Already a dict, normalize field names
        normalized_traffic = {}
        for sector_id, sector in traffic.items():
            if isinstance(sector, dict):
                normalized_sector = {
                    "id": sector_id,
                    "zone_id": sector_id,
                    "congestion_level": sector.get("congestion_level") or sector.get("congestion", 0.0),
                    "is_blocked": sector.get("is_blocked") or sector.get("blocked", False),
                    "description": sector.get("description", "")
                }
                normalized_traffic[sector_id] = normalized_sector
        normalized_state["traffic"] = normalized_traffic
    
    # Handle infrastructure if present
    if "infrastructure" in state:
        normalized_state["infrastructure"] = state["infrastructure"]
    elif "grid" in state and "infrastructure" in state["grid"]:
        normalized_state["infrastructure"] = state["grid"]["infrastructure"]
    else:
        normalized_state["infrastructure"] = {}
    
    return normalized_state


if __name__ == "__main__":
    # Test the evaluator with some sample commands
    from workshop.scenarios import create_heat_wave_scenario_evaluation
    
    sample_commands = [
        {
            "service": "grid",
            "action": "adjust_zone",
            "parameters": {"zone_id": "zone_a", "capacity": 0.75}
        },
        {
            "service": "grid",
            "action": "set_priority",
            "parameters": {"infrastructure_id": "hospital", "level": "critical"}
        },
        {
            "service": "emergency",
            "action": "assign_drone",
            "parameters": {"drone_id": "drone_1", "incident_id": "incident_1"}
        }
    ]
    
    # Create a sample state
    sample_state = {
        "zones": {
            "zone_a": {"id": "zone_a", "stability": 0.7, "capacity": 1.0, "current_load": 0.8},
            "zone_b": {"id": "zone_b", "stability": 0.6, "capacity": 1.0, "current_load": 0.7}
        },
        "incidents": [
            {"id": "incident_1", "assigned_drone": None},
            {"id": "incident_2", "assigned_drone": None}
        ],
        "traffic": {
            "zone_a": {"congestion": 0.8},
            "zone_b": {"congestion": 0.5}
        }
    }
    
    # Create evaluator
    evaluation_config = create_heat_wave_scenario_evaluation()
    evaluator = CommandEvaluator(evaluation_config)
    
    # Evaluate commands
    result = evaluator.evaluate_commands(sample_commands, sample_state)
    
    print(json.dumps(result, indent=2)) 