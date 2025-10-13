"""
Deterministic seed fixtures for reproducible testing.

Provides consistent random seeds for training and inference tests
to ensure reproducibility across test runs.
"""

import random
import numpy as np
import torch


# Standard seed for all tests
DEFAULT_SEED = 42
TRAINING_SEED = 1337
VALIDATION_SEED = 2024


def set_seed(seed: int = DEFAULT_SEED):
    """
    Set deterministic seeds for all random number generators.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_deterministic_config():
    """
    Get configuration dict for deterministic training.

    Returns:
        Dict with seed settings for DSPy training
    """
    return {
        "seed": DEFAULT_SEED,
        "deterministic": True,
        "num_workers": 0,  # Disable multiprocessing for determinism
    }


# Sample expert demonstrations for testing (10 examples minimum per SC-001)
SAMPLE_EXPERT_DEMOS = [
    {
        "state": "Vehicle approaching intersection with red light",
        "action": "stop",
        "next_state": "Vehicle stopped at intersection; light still red"
    },
    {
        "state": "Green light ahead; no obstacles",
        "action": "proceed",
        "next_state": "Vehicle passes through intersection safely"
    },
    {
        "state": "Yellow light; vehicle 50 feet away",
        "action": "slow down",
        "next_state": "Vehicle decelerates to stop at intersection"
    },
    {
        "state": "Pedestrian waiting at crosswalk; green light",
        "action": "yield",
        "next_state": "Pedestrian crosses safely; vehicle waits"
    },
    {
        "state": "Emergency vehicle approaching with sirens",
        "action": "pull over",
        "next_state": "Vehicle stopped on shoulder; emergency vehicle passes"
    },
    {
        "state": "Stop sign ahead; no other vehicles",
        "action": "come to complete stop",
        "next_state": "Vehicle stopped at stop sign"
    },
    {
        "state": "Merging onto highway; traffic gap available",
        "action": "accelerate to merge",
        "next_state": "Vehicle successfully merged into highway traffic"
    },
    {
        "state": "School zone; children present",
        "action": "reduce speed to 15 mph",
        "next_state": "Vehicle traveling at safe school zone speed"
    },
    {
        "state": "Icy road conditions; slight curve ahead",
        "action": "reduce speed and use gentle steering",
        "next_state": "Vehicle navigates curve safely at reduced speed"
    },
    {
        "state": "Vehicle ahead braking suddenly",
        "action": "brake firmly and maintain distance",
        "next_state": "Vehicle maintains safe following distance"
    }
]
