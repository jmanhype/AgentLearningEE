"""
Synthetic Dataset Generation for Testing

Provides utilities to generate expert demonstrations of varying sizes
for parameterized testing with different dataset scales.
"""

import random
from typing import List, Dict, Any


# Template patterns for synthetic demo generation
STATE_TEMPLATES = [
    "Vehicle approaching intersection with {light} light",
    "Pedestrian crossing at {location}",
    "Vehicle in {lane} lane with {obstacle} ahead",
    "Traffic signal changing from {from_color} to {to_color}",
    "Vehicle speed {speed}mph in {zone} zone",
    "Weather condition: {weather}, visibility: {visibility}",
    "Vehicle merging from {from_lane} to {to_lane}",
    "Emergency vehicle approaching from {direction}",
    "School zone active, children {activity}",
    "Construction zone ahead, {lanes} lanes closed",
]

ACTION_TEMPLATES = [
    "stop",
    "proceed with caution",
    "accelerate",
    "decelerate",
    "change lane",
    "yield",
    "maintain speed",
    "brake",
    "signal turn",
    "activate hazard lights",
]

# Mappings for state-action-next_state logic
STATE_ACTION_OUTCOMES = {
    ("red light", "stop"): "Vehicle stopped at intersection; light still red",
    ("red light", "proceed with caution"): "Vehicle proceeds through red light; potential violation",
    ("green light", "proceed with caution"): "Vehicle proceeds through intersection safely",
    ("yellow light", "decelerate"): "Vehicle slows down; preparing to stop",
    ("pedestrian", "yield"): "Vehicle yields; pedestrian crosses safely",
    ("obstacle", "change lane"): "Vehicle changes lane; obstacle avoided",
    ("speed limit", "maintain speed"): "Vehicle maintains safe speed",
    ("emergency vehicle", "yield"): "Vehicle yields; emergency vehicle passes",
}

# Substitution values for templates
SUBSTITUTIONS = {
    "light": ["red", "yellow", "green"],
    "location": ["crosswalk", "intersection", "mid-block"],
    "lane": ["left", "right", "center"],
    "obstacle": ["stopped vehicle", "debris", "construction"],
    "from_color": ["green", "yellow"],
    "to_color": ["yellow", "red"],
    "speed": ["25", "35", "45", "55"],
    "zone": ["school", "residential", "highway"],
    "weather": ["rain", "fog", "snow", "clear"],
    "visibility": ["low", "moderate", "high"],
    "from_lane": ["right", "left"],
    "to_lane": ["center", "left", "right"],
    "direction": ["behind", "ahead", "left side", "right side"],
    "activity": ["present", "crossing", "playing nearby"],
    "lanes": ["one", "two"],
}


def generate_synthetic_demos(
    num_demos: int,
    seed: int = 42,
    include_base_demos: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic expert demonstrations for testing.

    Creates deterministic synthetic demos by expanding templates with
    different substitution patterns.

    Args:
        num_demos: Total number of demos to generate
        seed: Random seed for reproducibility
        include_base_demos: If True, include the 10 base demos from deterministic_seeds

    Returns:
        List of demo dictionaries with state, action, next_state fields

    Example:
        >>> demos = generate_synthetic_demos(50, seed=42)
        >>> len(demos)
        50
        >>> demos[0].keys()
        dict_keys(['state', 'action', 'next_state'])
    """
    random.seed(seed)
    demos = []

    # Include base demos if requested
    if include_base_demos:
        from .deterministic_seeds import SAMPLE_EXPERT_DEMOS
        demos.extend(SAMPLE_EXPERT_DEMOS)

    # Generate additional synthetic demos to reach target count
    remaining = num_demos - len(demos)

    for i in range(remaining):
        # Select random template
        state_template = random.choice(STATE_TEMPLATES)

        # Fill in template placeholders
        state = state_template
        for placeholder, values in SUBSTITUTIONS.items():
            if f"{{{placeholder}}}" in state:
                state = state.replace(f"{{{placeholder}}}", random.choice(values))

        # Select action based on state keywords
        if "red light" in state or "stop" in state.lower():
            action = random.choice(["stop", "brake"])
        elif "pedestrian" in state or "crossing" in state:
            action = random.choice(["yield", "stop"])
        elif "emergency" in state:
            action = random.choice(["yield", "change lane"])
        elif "obstacle" in state or "construction" in state:
            action = random.choice(["change lane", "decelerate"])
        else:
            action = random.choice(ACTION_TEMPLATES)

        # Generate next_state based on state-action logic
        next_state = generate_next_state(state, action)

        demo = {
            "state": state,
            "action": action,
            "next_state": next_state,
        }

        demos.append(demo)

    return demos[:num_demos]  # Ensure exact count


def generate_next_state(state: str, action: str) -> str:
    """
    Generate logical next_state given state and action.

    Uses keyword matching and simple rules to create plausible transitions.

    Args:
        state: Current state description
        action: Action taken

    Returns:
        Next state description
    """
    state_lower = state.lower()
    action_lower = action.lower()

    # Rule-based next state generation
    if "red light" in state_lower:
        if action_lower in ["stop", "brake"]:
            return f"{state}; vehicle stopped safely"
        else:
            return f"{state}; vehicle proceeds through red light"

    elif "pedestrian" in state_lower or "crossing" in state_lower:
        if action_lower in ["yield", "stop"]:
            return "Vehicle yields; pedestrian crosses safely"
        else:
            return "Vehicle proceeds; pedestrian waits"

    elif "obstacle" in state_lower or "construction" in state_lower:
        if "change lane" in action_lower:
            return "Vehicle changes lane; obstacle avoided"
        elif action_lower in ["decelerate", "brake"]:
            return "Vehicle slows down; maintaining safe distance from obstacle"
        else:
            return f"{state}; vehicle maintains current trajectory"

    elif "emergency vehicle" in state_lower:
        if action_lower == "yield":
            return "Vehicle yields; emergency vehicle passes safely"
        else:
            return "Vehicle attempts to move aside for emergency vehicle"

    elif "speed" in state_lower:
        if action_lower in ["decelerate", "brake"]:
            return "Vehicle speed reduced to safe level"
        elif action_lower == "accelerate":
            return "Vehicle speed increased within safe limits"
        else:
            return "Vehicle maintains current speed"

    # Default fallback
    return f"After {action}, vehicle continues in new state"


def get_expected_accuracy_range(num_demos: int) -> tuple[float, float]:
    """
    Get expected accuracy range for given dataset size.

    Provides guidance on what accuracy metrics to expect based on
    dataset size and statistical significance.

    Args:
        num_demos: Number of demonstrations in dataset

    Returns:
        Tuple of (min_expected_accuracy, max_expected_accuracy)

    Example:
        >>> get_expected_accuracy_range(10)
        (0.0, 1.0)  # Too small for statistical significance
        >>> get_expected_accuracy_range(100)
        (0.3, 0.9)  # Should show meaningful accuracy
    """
    # With 80/20 split, calculate test set size
    test_size = int(num_demos * 0.2)

    if test_size < 5:
        # Too small for statistical significance
        return (0.0, 1.0)
    elif test_size < 10:
        # Minimal validity - wide range expected
        return (0.0, 0.8)
    elif test_size < 20:
        # Basic validity - expect some positive accuracy
        return (0.1, 0.9)
    else:
        # Good statistical power - expect meaningful accuracy
        return (0.3, 0.9)


def get_dataset_size_recommendation(purpose: str) -> int:
    """
    Get recommended dataset size for different testing purposes.

    Args:
        purpose: Testing purpose - one of:
            - "smoke": Quick smoke test
            - "unit": Unit test validation
            - "integration": Integration testing
            - "validation": Thorough validation
            - "production": Production-ready evaluation

    Returns:
        Recommended number of demonstrations
    """
    recommendations = {
        "smoke": 10,       # Quick validation that code runs
        "unit": 30,        # Basic unit test coverage
        "integration": 50, # Integration test with minimal validity
        "validation": 100, # Thorough validation with good confidence
        "production": 200, # Production-ready evaluation
    }

    return recommendations.get(purpose, 100)
