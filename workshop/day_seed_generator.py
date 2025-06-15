import random
import datetime
from typing import Optional, Any, Dict, List, Union

class DaySeedGenerator:
    """
    Utility class to generate deterministic random data based on a day seed.
    This ensures that simulation data is consistent for a given day, but varies between days.
    """
    
    def __init__(self, day: Optional[int] = None):
        """
        Initialize the generator with a specific day seed.
        
        Args:
            day: Day number to use as seed. If None, uses the current day of month.
        """
        if day is None:
            day = datetime.datetime.now().day
        
        self.day = day
        self.base_seed = f"neocatalis-{day}"
        self._reset_seed()
    
    def _reset_seed(self) -> None:
        """Reset the random seed to the base day seed."""
        random.seed(self.base_seed)
    
    def get_random_int(self, min_val: int, max_val: int, context: str = "") -> int:
        """
        Get a deterministic random integer based on the day seed and optional context.
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            context: Optional context string to create different sequences for the same day
            
        Returns:
            Random integer between min_val and max_val
        """
        if context:
            random.seed(f"{self.base_seed}-{context}")
        
        result = random.randint(min_val, max_val)
        self._reset_seed()
        return result
    
    def get_random_float(self, min_val: float, max_val: float, context: str = "") -> float:
        """
        Get a deterministic random float based on the day seed and optional context.
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            context: Optional context string to create different sequences for the same day
            
        Returns:
            Random float between min_val and max_val
        """
        if context:
            random.seed(f"{self.base_seed}-{context}")
        
        result = min_val + random.random() * (max_val - min_val)
        self._reset_seed()
        return result
    
    def get_random_choice(self, options: List[Any], context: str = "") -> Any:
        """
        Get a deterministic random choice from a list based on the day seed and optional context.
        
        Args:
            options: List of options to choose from
            context: Optional context string to create different sequences for the same day
            
        Returns:
            Random element from options
        """
        if context:
            random.seed(f"{self.base_seed}-{context}")
        
        result = random.choice(options)
        self._reset_seed()
        return result
    
    def get_random_sample(self, options: List[Any], k: int, context: str = "") -> List[Any]:
        """
        Get a deterministic random sample from a list based on the day seed and optional context.
        
        Args:
            options: List of options to sample from
            k: Number of elements to sample
            context: Optional context string to create different sequences for the same day
            
        Returns:
            List of k random elements from options
        """
        if context:
            random.seed(f"{self.base_seed}-{context}")
        
        result = random.sample(options, k)
        self._reset_seed()
        return result
    
    def get_weighted_choice(self, options: List[Any], weights: List[float], context: str = "") -> Any:
        """
        Get a deterministic weighted random choice based on the day seed and optional context.
        
        Args:
            options: List of options to choose from
            weights: List of weights corresponding to options
            context: Optional context string to create different sequences for the same day
            
        Returns:
            Random element from options based on weights
        """
        if context:
            random.seed(f"{self.base_seed}-{context}")
        
        result = random.choices(options, weights=weights, k=1)[0]
        self._reset_seed()
        return result

    def generate_grid_data(self, num_zones: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Generate power grid data for the city's zones.
        
        Args:
            num_zones: Number of power zones to generate
            
        Returns:
            Dictionary mapping zone IDs to their power status data
        """
        zones = {}
        
        # Different days have different base load patterns
        base_load_factor = self.get_random_float(0.6, 0.9, "grid-base-load")
        
        for i in range(1, num_zones + 1):
            zone_id = f"Z{i:03d}"
            
            # Each zone has its own random seed based on zone ID
            capacity = self.get_random_int(80, 120, f"capacity-{zone_id}")
            current_load = int(capacity * base_load_factor * self.get_random_float(0.7, 1.3, f"load-{zone_id}"))
            
            # Some zones might be offline or critical based on the day
            status = self.get_random_choice(
                ["online", "online", "online", "online", "degraded", "offline"], 
                f"status-{zone_id}"
            )
            
            # Critical zones are more likely on certain days
            is_critical = self.get_random_float(0, 1, f"critical-{zone_id}") < 0.3
            
            zones[zone_id] = {
                "zone_id": zone_id,
                "status": status,
                "capacity_kw": capacity, 
                "current_load_kw": current_load,
                "is_critical": is_critical
            }
        
        return zones
    
    def generate_weather_data(self) -> Dict[str, Any]:
        """
        Generate weather data for the city.
        
        Returns:
            Dictionary with current weather conditions
        """
        # Different days have different weather patterns
        weather_types = ["clear", "cloudy", "rainy", "stormy", "heat_wave"]
        weights = [0.4, 0.3, 0.15, 0.05, 0.1]
        
        weather_type = self.get_weighted_choice(weather_types, weights, "weather-type")
        
        # Temperature depends on weather type
        base_temp = {
            "clear": self.get_random_float(15, 25, "temp-clear"),
            "cloudy": self.get_random_float(10, 20, "temp-cloudy"),
            "rainy": self.get_random_float(5, 15, "temp-rainy"),
            "stormy": self.get_random_float(5, 15, "temp-stormy"),
            "heat_wave": self.get_random_float(30, 40, "temp-heat")
        }[weather_type]
        
        # Wind speed depends on weather type
        base_wind = {
            "clear": self.get_random_float(0, 10, "wind-clear"),
            "cloudy": self.get_random_float(5, 15, "wind-cloudy"),
            "rainy": self.get_random_float(10, 25, "wind-rainy"),
            "stormy": self.get_random_float(25, 45, "wind-stormy"),
            "heat_wave": self.get_random_float(0, 5, "wind-heat")
        }[weather_type]
        
        # Precipitation depends on weather type
        base_precip = {
            "clear": 0,
            "cloudy": self.get_random_float(0, 0.1, "precip-cloudy"),
            "rainy": self.get_random_float(0.3, 0.7, "precip-rainy"),
            "stormy": self.get_random_float(0.7, 1.0, "precip-stormy"),
            "heat_wave": 0
        }[weather_type]
        
        # Generate advisories based on conditions
        advisories = []
        if weather_type == "stormy":
            advisories.append("thunderstorm_warning")
            if self.get_random_float(0, 1, "advisory-flood") > 0.7:
                advisories.append("flood_warning")
        elif weather_type == "heat_wave":
            advisories.append("heat_advisory")
            if self.get_random_float(0, 1, "advisory-power") > 0.6:
                advisories.append("power_conservation_notice")
        elif weather_type == "rainy" and base_precip > 0.5:
            if self.get_random_float(0, 1, "advisory-flood") > 0.7:
                advisories.append("flood_watch")
        
        return {
            "conditions": weather_type,
            "temperature_c": round(base_temp, 1),
            "wind_speed_kph": round(base_wind, 1),
            "precipitation_mm": round(base_precip * 10, 1),
            "advisories": advisories
        }
    
    def generate_emergency_incidents(self, num_incidents: int = 15) -> List[Dict[str, Any]]:
        """
        Generate emergency incidents across the city.
        
        Args:
            num_incidents: Number of incidents to generate
            
        Returns:
            List of emergency incident dictionaries
        """
        incident_types = ["medical", "fire", "police", "infrastructure", "hazmat"]
        urgency_levels = ["low", "medium", "high", "critical"]
        urgency_weights = [0.2, 0.4, 0.3, 0.1]
        
        city_zones = [f"Z{i:03d}" for i in range(1, 11)]
        
        incidents = []
        
        for i in range(1, num_incidents + 1):
            incident_id = f"E-{1000 + i}"
            
            incident_type = self.get_random_choice(incident_types, f"incident-type-{incident_id}")
            urgency = self.get_weighted_choice(urgency_levels, urgency_weights, f"incident-urgency-{incident_id}")
            zone = self.get_random_choice(city_zones, f"incident-zone-{incident_id}")
            
            # Estimated resolution time depends on type and urgency
            base_resolution_time = {
                "medical": 20,
                "fire": 45,
                "police": 30,
                "infrastructure": 60,
                "hazmat": 90
            }[incident_type]
            
            # Adjust based on urgency
            urgency_factor = {
                "low": 0.8,
                "medium": 1.0,
                "high": 1.2,
                "critical": 1.5
            }[urgency]
            
            estimated_resolution_minutes = int(base_resolution_time * urgency_factor * 
                                              self.get_random_float(0.8, 1.2, f"resolution-{incident_id}"))
            
            # Some incidents require specific drone capabilities
            required_capabilities = []
            if incident_type == "medical":
                required_capabilities.append("medical_kit")
            elif incident_type == "fire":
                required_capabilities.append("thermal_imaging")
                if self.get_random_float(0, 1, f"water-{incident_id}") > 0.7:
                    required_capabilities.append("water_dispersal")
            elif incident_type == "hazmat":
                required_capabilities.append("hazmat_sensors")
            
            if urgency in ["high", "critical"]:
                required_capabilities.append("high_speed")
            
            incidents.append({
                "incident_id": incident_id,
                "type": incident_type,
                "urgency": urgency,
                "zone": zone,
                "estimated_resolution_minutes": estimated_resolution_minutes,
                "required_capabilities": required_capabilities,
                "status": "active"
            })
        
        return incidents
    
    def generate_traffic_data(self, num_sectors: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Generate traffic data for city sectors.
        
        Args:
            num_sectors: Number of traffic sectors to generate
            
        Returns:
            Dictionary mapping sector IDs to traffic status data
        """
        sectors = {}
        
        # Different days have different base congestion levels
        base_congestion = self.get_random_float(0.3, 0.7, "traffic-base")
        
        for i in range(1, num_sectors + 1):
            sector_id = f"S{i:03d}"
            
            # Each sector has its own random factor
            congestion_level = base_congestion * self.get_random_float(0.5, 1.5, f"congestion-{sector_id}")
            congestion_level = min(1.0, congestion_level)  # Cap at 1.0
            
            # Traffic status based on congestion level
            if congestion_level < 0.3:
                status = "clear"
            elif congestion_level < 0.6:
                status = "moderate"
            elif congestion_level < 0.9:
                status = "heavy"
            else:
                status = "gridlock"
            
            # Some sectors might be blocked
            is_blocked = self.get_random_float(0, 1, f"blocked-{sector_id}") > 0.9
            
            # Calculate travel time multiplier
            travel_time_multiplier = 1.0 + (congestion_level * 2.0)
            if is_blocked:
                travel_time_multiplier = 5.0
                status = "blocked"
            
            sectors[sector_id] = {
                "sector_id": sector_id,
                "status": status,
                "congestion_level": round(congestion_level, 2),
                "is_blocked": is_blocked,
                "travel_time_multiplier": round(travel_time_multiplier, 1)
            }
        
        return sectors
    
    def generate_drone_fleet(self, num_drones: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Generate a fleet of drones with various capabilities.
        
        Args:
            num_drones: Number of drones to generate
            
        Returns:
            Dictionary mapping drone IDs to their specifications
        """
        drones = {}
        
        drone_versions = ["v1", "v2", "v3"]
        
        capabilities_by_version = {
            "v1": ["basic_camera", "medical_kit"],
            "v2": ["basic_camera", "medical_kit", "thermal_imaging", "high_speed"],
            "v3": ["advanced_camera", "medical_kit", "thermal_imaging", "water_dispersal", 
                  "hazmat_sensors", "high_speed", "weather_resistant"]
        }
        
        speed_by_version = {
            "v1": 30,  # km/h
            "v2": 50,
            "v3": 70
        }
        
        for i in range(1, num_drones + 1):
            drone_id = f"D{i:03d}"
            
            # Each drone has a version that determines its capabilities
            version = self.get_random_choice(drone_versions, f"drone-version-{drone_id}")
            
            # Base speed with some randomness
            speed = speed_by_version[version] * self.get_random_float(0.9, 1.1, f"drone-speed-{drone_id}")
            
            drones[drone_id] = {
                "drone_id": drone_id,
                "version": version,
                "capabilities": capabilities_by_version[version],
                "speed_kph": round(speed, 1),
                "status": "available",
                "current_location": self.get_random_choice([f"S{j:03d}" for j in range(1, 11)], f"drone-location-{drone_id}")
            }
        
        return drones

# Example usage
if __name__ == "__main__":
    generator = DaySeedGenerator(day=5)
    
    # Generate example data
    grid_data = generator.generate_grid_data()
    weather_data = generator.generate_weather_data()
    incidents = generator.generate_emergency_incidents()
    traffic_data = generator.generate_traffic_data()
    drone_fleet = generator.generate_drone_fleet()
    
    # Print some examples
    print(f"Weather: {weather_data}")
    print(f"Sample Grid Zone: {list(grid_data.values())[0]}")
    print(f"Sample Incident: {incidents[0]}")
    print(f"Sample Traffic Sector: {list(traffic_data.values())[0]}")
    print(f"Sample Drone: {list(drone_fleet.values())[0]}") 