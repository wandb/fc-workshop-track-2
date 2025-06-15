# NeoCatalis - Fully Connected Workshop Track 2

## Operation SENTINEL GRID

NeoCatalis, a fully connected smart city, has experienced a systemic AI blackout. This project simulates rebuilding the city's autonomy using modern agentic principles.

## Scoring and Evaluation

The workshop evaluates agent performance using several metrics:

1. **Incident Coverage**: Percentage of emergency incidents successfully dispatched
2. **Average ETA**: Average estimated time of arrival for emergency responders
3. **Tool Failure Handling**: Percentage of tool failures with successful fallback
4. **Capability Match**: Percentage of incidents with correctly matched drone capabilities
5. **Latency**: Time taken to complete each scenario
6. **Steps Taken**: Number of LLM calls needed to reach a conclusion

Results are saved to the `results/` directory with individual scenario files and a summary file containing the overall score.

## Key Scenarios

The system is designed to handle various scenarios:

1. **Grid Surges During Heat Waves**: Managing power consumption during extreme heat
2. **Medical Emergencies**: Dispatching appropriate drones with medical capabilities
3. **Drone Capacity Crisis**: Prioritizing incidents when drone resources are limited
4. **Flood Disruptions**: Rerouting traffic and managing power during flooding
5. **City-wide Cascading Failures**: Coordinating across all systems during major events

## Learning Objectives

1. **Design modular, agentic systems** using CrewAI that coordinate across multiple independent services
2. **Build and orchestrate agents** that communicate, plan, and take actions across diverse environments
3. **Utilize tool discovery and invocation protocols** (like MCP) to enable agent adaptability
4. **Coordinate actions under constraints** (e.g. limited drones, partial system failure)
5. **Integrate real-time feedback** to inform planning and agent decision-making
6. **Model uncertainty and cascading dependencies**, handling unpredictable changes
7. **Experiment with different multiagent architectures** to optimize for various metrics

## Workshop Challenges

1. **Dynamic Power Management**: Balance grid load during extreme weather
2. **Emergency Response Orchestration**: Optimize drone assignments with limited resources
3. **Routing in a Congested City**: Find optimal paths for emergency vehicles
4. **Planning with Weather Interference**: Adapt to changing weather conditions
5. **Modular Agent Design Using MCP**: Create robust fallback mechanisms for tool failures
6. **Multiagent System Architecture**: Implement and compare different agent organization approaches

