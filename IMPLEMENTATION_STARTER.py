"""
Agent Learning via Early Experience - DSPy Implementation Starter
==================================================================

This file contains ready-to-use code for implementing the core components.
Copy/paste sections as needed into your project structure.

Based on research findings in research.md
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFinetune


# =============================================================================
# SECTION 1: SIGNATURE DEFINITIONS
# =============================================================================

class WorldModelSig(dspy.Signature):
    """
    Predict next state given current state and action.

    This signature models state transitions for the agent's internal
    simulation and planning. It should predict deterministic outcomes
    based on action effects.
    """
    state: str = dspy.InputField(
        desc="current environment state description"
    )
    action: str = dspy.InputField(
        desc="action to execute in current state"
    )
    next_state: str = dspy.OutputField(
        desc="predicted next state after action execution"
    )


class PolicySig(dspy.Signature):
    """
    EE-style policy decision with structured alternatives comparison.

    This signature enforces Engineering Excellence decision-making:
    - Compare at least 2 plausible actions
    - Predict outcomes for each alternative
    - Justify chosen action with evidence
    """
    state: str = dspy.InputField(
        desc="current environment state"
    )

    reasoning: str = dspy.OutputField(
        desc="""
        EE Reflection Format (REQUIRED):

        State: <concise description of current situation>

        Alternatives:
        - Action A: <what> → Expected Outcome: <prediction, evidence>
        - Action B: <what> → Expected Outcome: <prediction, evidence>
        - (Optional) Action C: <what> → Expected Outcome: <prediction, evidence>

        Analysis: Compare outcomes, risks, reversibility, testability.

        Conclusion: Therefore, best action is <Expert Action> because <reasons>.
        """
    )

    action: str = dspy.OutputField(
        desc="single chosen action (must match conclusion)"
    )


# =============================================================================
# SECTION 2: LLM CONFIGURATION
# =============================================================================

def configure_lm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    seed: int = 42
):
    """
    Configure DSPy global LLM.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20240620")
        temperature: Sampling temperature (0.0-1.0)
        seed: Random seed for reproducibility

    Environment variables:
        OPENAI_API_KEY: For OpenAI models
        ANTHROPIC_API_KEY: For Claude models
    """
    if "claude" in model.lower():
        lm = dspy.Anthropic(model=model, max_tokens=4000, temperature=temperature)
    else:
        lm = dspy.OpenAI(model=model, max_tokens=4000, temperature=temperature, seed=seed)

    dspy.configure(lm=lm)
    print(f"Configured DSPy with {model}")


# =============================================================================
# SECTION 3: TRAINING DATA UTILITIES
# =============================================================================

def load_training_examples(path: str) -> List[dspy.Example]:
    """
    Load training examples from JSONL file.

    Expected formats:
        World model: {"state": "...", "action": "...", "next_state": "..."}
        Policy: {"state": "...", "reasoning": "...", "action": "..."}

    Args:
        path: Path to JSONL file

    Returns:
        List of dspy.Example objects with correct input fields marked
    """
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Determine type based on fields
                if "next_state" in data:
                    # World model example
                    ex = dspy.Example(**data).with_inputs("state", "action")
                else:
                    # Policy example
                    ex = dspy.Example(**data).with_inputs("state")

                examples.append(ex)

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue

    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def create_example_training_data():
    """
    Create example training data files for demonstration.
    This creates minimal working examples - expand with real data.
    """
    os.makedirs("data/training", exist_ok=True)

    # World model examples
    world_examples = [
        {
            "state": "at intersection, light=red, vehicles=stopped",
            "action": "wait",
            "next_state": "at intersection, light=red, vehicles=stopped, waited=1_cycle"
        },
        {
            "state": "at intersection, light=green, vehicles=moving",
            "action": "proceed",
            "next_state": "through intersection, vehicles=moving, crossed=true"
        },
    ]

    with open("data/training/world_model_examples.jsonl", "w") as f:
        for ex in world_examples:
            f.write(json.dumps(ex) + "\n")

    # Policy examples
    policy_examples = [
        {
            "state": "at intersection, light=red, vehicles=stopped",
            "reasoning": """
State: At red light with stopped traffic

Alternatives:
- Action A: wait → Expected Outcome: remain safe, legal, light will change in 30-60s
- Action B: run red light → Expected Outcome: traffic violation, high collision risk

Analysis: Action A is zero-risk and legal. Action B violates traffic law and creates
collision hazard with cross traffic. No benefit to rushing.

Conclusion: Therefore, best action is wait because it ensures safety and legal compliance.
""",
            "action": "wait"
        },
        {
            "state": "at yellow light, 50 feet away, speed=35mph",
            "reasoning": """
State: Approaching yellow light at 50ft traveling 35mph

Alternatives:
- Action A: brake hard → Expected Outcome: stop before intersection, safe but abrupt
- Action B: maintain speed → Expected Outcome: enter on yellow/red transition, risky
- Action C: accelerate → Expected Outcome: clear intersection, dangerous and illegal

Analysis: At 35mph with 50ft distance, partial braking achieves safe stop. Action B/C
create legal liability and collision risks. Braking is reversible and testable.

Conclusion: Therefore, best action is brake because stopping is the safest legal option.
""",
            "action": "brake"
        },
    ]

    with open("data/training/policy_examples.jsonl", "w") as f:
        for ex in policy_examples:
            f.write(json.dumps(ex) + "\n")

    print("Created example training data in data/training/")


# =============================================================================
# SECTION 4: METRIC FUNCTIONS
# =============================================================================

def world_model_metric(example, prediction, trace=None):
    """
    Metric for world model training.

    Returns:
        1.0 if prediction is reasonable, 0.0 otherwise
    """
    # Check if next_state field exists and is non-empty
    if not prediction.next_state or len(prediction.next_state) < 5:
        return 0.0

    # Could add semantic similarity check here
    return 1.0


def policy_metric(example, prediction, trace=None):
    """
    Metric for policy training.

    Scores:
        - 0.3 for having EE structure (Alternatives + Conclusion)
        - 0.7 for correct action

    Returns:
        Float between 0.0 and 1.0
    """
    # Check reasoning quality
    has_alternatives = (
        "Alternatives:" in prediction.reasoning or
        "Action A:" in prediction.reasoning
    )
    has_conclusion = (
        "Conclusion:" in prediction.reasoning or
        "best action" in prediction.reasoning.lower()
    )

    reasoning_score = 0.3 if (has_alternatives and has_conclusion) else 0.0

    # Check action correctness
    action_score = 0.7 if example.action.lower() == prediction.action.lower() else 0.0

    return reasoning_score + action_score


def ee_compliant_metric(example, prediction, trace=None):
    """
    Strict EE-compliant metric.

    Enforces:
        - EE structure present
        - At least 2 alternatives
        - Correct action

    Returns:
        Float between 0.0 and 1.0
    """
    # Check EE structure
    required_sections = ["State:", "Alternatives:", "Analysis:", "Conclusion:"]
    has_structure = all(section in prediction.reasoning for section in required_sections)

    if not has_structure:
        return 0.0

    # Check alternative count
    alt_count = len(re.findall(r"Action [A-Z]:", prediction.reasoning))
    if alt_count < 2:
        return 0.0

    # Weight: 40% structure + 60% correctness
    structure_score = 0.4
    action_score = 0.6 if example.action.lower() == prediction.action.lower() else 0.0

    return structure_score + action_score


# =============================================================================
# SECTION 5: TRAINING FUNCTIONS
# =============================================================================

def train_world_model(
    training_data_path: str,
    output_path: str = "models/trained/world_model.json"
) -> dspy.Module:
    """
    Train world model.

    Args:
        training_data_path: Path to JSONL training data
        output_path: Where to save trained model

    Returns:
        Trained DSPy module
    """
    print("\n" + "="*60)
    print("Training World Model")
    print("="*60)

    # Load training data
    train_examples = load_training_examples(training_data_path)

    # Create student module
    world_model_student = dspy.Predict(WorldModelSig)

    # Initialize optimizer
    optimizer = BootstrapFinetune(
        metric=world_model_metric,
        num_threads=4
    )

    # Train
    print("Compiling world model...")
    world_model_compiled = optimizer.compile(
        student=world_model_student,
        trainset=train_examples
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    world_model_compiled.save(output_path)
    print(f"✓ Saved trained world model to {output_path}")

    return world_model_compiled


def train_policy(
    training_data_path: str,
    output_path: str = "models/trained/policy.json",
    use_strict_metric: bool = True
) -> dspy.Module:
    """
    Train EE-compliant policy.

    Args:
        training_data_path: Path to JSONL training data
        output_path: Where to save trained model
        use_strict_metric: If True, use ee_compliant_metric

    Returns:
        Trained DSPy module
    """
    print("\n" + "="*60)
    print("Training Policy")
    print("="*60)

    # Load training data
    train_examples = load_training_examples(training_data_path)

    # Create student module
    policy_student = dspy.ChainOfThought(PolicySig)

    # Choose metric
    metric = ee_compliant_metric if use_strict_metric else policy_metric

    # Initialize optimizer
    optimizer = BootstrapFinetune(
        metric=metric,
        num_threads=4
    )

    # Train
    print("Compiling policy...")
    policy_compiled = optimizer.compile(
        student=policy_student,
        trainset=train_examples
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    policy_compiled.save(output_path)
    print(f"✓ Saved trained policy to {output_path}")

    return policy_compiled


# =============================================================================
# SECTION 6: AGENT COMPOSITION
# =============================================================================

class AgentWithWorldModel(dspy.Module):
    """
    Agent that uses world model to predict outcomes before deciding action.

    Flow:
    1. Receive current state
    2. Query world model for each candidate action
    3. Pass predictions to policy for final decision
    """

    def __init__(
        self,
        world_model_path: Optional[str] = None,
        policy_path: Optional[str] = None
    ):
        super().__init__()

        # Initialize modules
        self.world_model = dspy.Predict(WorldModelSig)
        self.policy = dspy.ChainOfThought(PolicySig)

        # Load trained models if paths provided
        if world_model_path:
            self.world_model.load(world_model_path)
            print(f"Loaded world model from {world_model_path}")

        if policy_path:
            self.policy.load(policy_path)
            print(f"Loaded policy from {policy_path}")

    def forward(
        self,
        state: str,
        candidate_actions: Optional[List[str]] = None
    ) -> dspy.Prediction:
        """
        Execute agent decision-making.

        Args:
            state: Current environment state
            candidate_actions: List of actions to consider

        Returns:
            dspy.Prediction with reasoning, action, and world_model_predictions
        """
        if candidate_actions is None:
            candidate_actions = ["stop", "slow", "proceed"]

        # Step 1: Get world model predictions for each action
        predictions = {}
        for action in candidate_actions:
            pred = self.world_model(state=state, action=action)
            predictions[action] = pred.next_state

        # Step 2: Format predictions for policy
        prediction_text = "\n".join([
            f"- {action}: {next_state}"
            for action, next_state in predictions.items()
        ])

        enhanced_state = f"""{state}

World Model Predictions:
{prediction_text}"""

        # Step 3: Get policy decision with world model context
        decision = self.policy(state=enhanced_state)

        # Step 4: Return comprehensive result
        return dspy.Prediction(
            reasoning=decision.reasoning,
            action=decision.action,
            world_model_predictions=predictions
        )


# =============================================================================
# SECTION 7: UTILITIES
# =============================================================================

def parse_ee_reasoning(reasoning: str) -> Dict:
    """
    Parse EE-structured reasoning into components.

    Returns:
        {
            "state": str,
            "alternatives": List[dict],
            "analysis": str,
            "conclusion": str
        }
    """
    components = {}

    # Extract state
    state_match = re.search(r"State:\s*(.+?)(?:\n|$)", reasoning)
    if state_match:
        components["state"] = state_match.group(1).strip()

    # Extract alternatives
    alternatives = []
    alt_pattern = r"- Action ([A-Z]):\s*(.+?)\s*→\s*Expected(?:\s+Outcome)?:\s*(.+?)(?:\n|$)"
    for match in re.finditer(alt_pattern, reasoning):
        alternatives.append({
            "label": match.group(1),
            "action": match.group(2).strip(),
            "expected": match.group(3).strip()
        })
    components["alternatives"] = alternatives

    # Extract analysis
    analysis_match = re.search(
        r"Analysis:\s*(.+?)(?:\n\n|Conclusion:)",
        reasoning,
        re.DOTALL
    )
    if analysis_match:
        components["analysis"] = analysis_match.group(1).strip()

    # Extract conclusion
    conclusion_match = re.search(r"Conclusion:\s*(.+?)(?:\n|$)", reasoning)
    if conclusion_match:
        components["conclusion"] = conclusion_match.group(1).strip()

    return components


def validate_ee_format(reasoning: str) -> Tuple[bool, List[str]]:
    """
    Validate that reasoning follows EE format.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required sections
    if "State:" not in reasoning:
        errors.append("Missing 'State:' section")

    if "Alternatives:" not in reasoning and "Action A:" not in reasoning:
        errors.append("Missing 'Alternatives:' section")

    if "Analysis:" not in reasoning:
        errors.append("Missing 'Analysis:' section")

    if "Conclusion:" not in reasoning:
        errors.append("Missing 'Conclusion:' section")

    # Check for at least 2 alternatives
    alt_count = len(re.findall(r"Action [A-Z]:", reasoning))
    if alt_count < 2:
        errors.append(f"Only {alt_count} alternative(s) found, need at least 2")

    return len(errors) == 0, errors


# =============================================================================
# SECTION 8: MAIN TRAINING SCRIPT
# =============================================================================

def main_train():
    """
    Main training script - trains both world model and policy.
    """
    print("\n" + "="*60)
    print("Agent Learning via EE - Training Script")
    print("="*60)

    # Step 1: Configure LLM
    configure_lm(
        model=os.getenv("DSPY_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        seed=42
    )

    # Step 2: Create example data (remove this if you have real data)
    create_example_training_data()

    # Step 3: Train world model
    world_model = train_world_model(
        training_data_path="data/training/world_model_examples.jsonl",
        output_path="models/trained/world_model_v1.0.0.json"
    )

    # Step 4: Train policy
    policy = train_policy(
        training_data_path="data/training/policy_examples.jsonl",
        output_path="models/trained/policy_v1.0.0.json",
        use_strict_metric=True
    )

    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print("\nTrained models saved to:")
    print("  - models/trained/world_model_v1.0.0.json")
    print("  - models/trained/policy_v1.0.0.json")


# =============================================================================
# SECTION 9: MAIN INFERENCE SCRIPT
# =============================================================================

def main_inference():
    """
    Main inference script - loads models and runs example.
    """
    print("\n" + "="*60)
    print("Agent Learning via EE - Inference Script")
    print("="*60)

    # Step 1: Configure LLM
    configure_lm(
        model=os.getenv("DSPY_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        seed=42
    )

    # Step 2: Create agent with trained models
    agent = AgentWithWorldModel(
        world_model_path="models/trained/world_model_v1.0.0.json",
        policy_path="models/trained/policy_v1.0.0.json"
    )

    # Step 3: Run example
    test_state = "at yellow light, 50 feet away, speed=35mph"

    print(f"\n{'='*60}")
    print(f"Test State: {test_state}")
    print(f"{'='*60}\n")

    result = agent(state=test_state)

    print("World Model Predictions:")
    for action, next_state in result.world_model_predictions.items():
        print(f"  {action}: {next_state}")

    print(f"\n{'-'*60}")
    print("Policy Reasoning:")
    print(f"{'-'*60}")
    print(result.reasoning)

    print(f"\n{'-'*60}")
    print(f"Chosen Action: {result.action}")
    print(f"{'-'*60}")

    # Validate EE format
    is_valid, errors = validate_ee_format(result.reasoning)
    if is_valid:
        print("\n✓ Reasoning follows EE format")
    else:
        print(f"\n✗ EE format errors: {errors}")

    # Parse reasoning components
    parsed = parse_ee_reasoning(result.reasoning)
    print(f"\nParsed {len(parsed['alternatives'])} alternatives:")
    for alt in parsed["alternatives"]:
        print(f"  {alt['label']}: {alt['action']} → {alt['expected']}")


# =============================================================================
# SECTION 10: CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python IMPLEMENTATION_STARTER.py train     # Train models")
        print("  python IMPLEMENTATION_STARTER.py infer     # Run inference")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        main_train()
    elif command == "infer":
        main_inference()
    else:
        print(f"Unknown command: {command}")
        print("Use 'train' or 'infer'")
        sys.exit(1)
