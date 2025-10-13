# DSPy Implementation Research: Agent Learning via Early Experience

**Date**: 2025-10-12
**Version**: 1.0.0
**DSPy Version**: 3.0.3

## Executive Summary

This document provides comprehensive technical research on DSPy implementation patterns for the Agent Learning via Early Experience (EE) feature. It covers signature design, training workflows, module composition, and integration with the Engineering Excellence constitution.

---

## 1. DSPy Module Signatures

### 1.1 World Model Signature

**Purpose**: Predict next state given current state and action for planning/simulation.

```python
class WorldModelSig(dspy.Signature):
    """Predict next state given current state and action.

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
```

**Usage Pattern**:
```python
# Create world model module
world_model = dspy.Predict(WorldModelSig)

# Use for prediction
result = world_model(
    state="at intersection, light=red, vehicles=stopped",
    action="wait"
)
print(f"Next state: {result.next_state}")
# Output: "at intersection, light=red, vehicles=stopped, waited=true"
```

**File Location**: `src/models/world_model.py`

**Key Design Decisions**:
- Use `dspy.Predict` (not ChainOfThought) for world model - no reasoning needed
- State should be structured string (JSON-like for complex states)
- Action should be atomic/discrete for clear causality
- next_state format must match state format for chaining

---

### 1.2 Policy Signature with Chain-of-Thought

**Purpose**: Select action with structured EE-style reasoning comparing alternatives.

```python
class PolicySig(dspy.Signature):
    """EE-style policy decision with structured alternatives comparison.

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
```

**Usage Pattern**:
```python
# Create policy with ChainOfThought for reasoning
policy = dspy.ChainOfThought(PolicySig)

# Execute policy decision
result = policy(state="at yellow light, 50 feet away, speed=35mph")

print(f"Reasoning:\n{result.reasoning}")
print(f"\nAction: {result.action}")
```

**Example Output**:
```
Reasoning:
State: Approaching yellow light at 50ft traveling 35mph

Alternatives:
- Action A: brake hard → Expected: stop before intersection, safe but abrupt
- Action B: maintain speed → Expected: enter on yellow/red, risky, illegal
- Action C: accelerate → Expected: clear intersection, dangerous, illegal

Analysis: Action A has best safety profile. Action B/C risk traffic violations
and collisions. Braking is reversible and testable via simulation.

Conclusion: Therefore, best action is brake because it ensures safety and legal compliance.

Action: brake
```

**File Location**: `src/models/policy.py`

**Key Design Decisions**:
- Use `dspy.ChainOfThought` to enforce reasoning before action
- Embed EE template in `reasoning` field description
- Multiple output fields capture both reasoning AND final action
- Description acts as implicit prompt engineering
- ChainOfThought guarantees reasoning is generated first

---

### 1.3 Alternative: Separate Reasoning and Action

**Purpose**: Explicit two-stage process with maximum control.

```python
class ReasoningSig(dspy.Signature):
    """Generate structured reasoning about action alternatives."""
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField(
        desc="EE-style structured reflection comparing alternatives"
    )

class ActionSelectionSig(dspy.Signature):
    """Select final action based on completed reasoning."""
    state: str = dspy.InputField()
    reasoning: str = dspy.InputField(desc="completed EE reflection")
    action: str = dspy.OutputField()

class TwoStagePolicy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(ReasoningSig)
        self.actor = dspy.Predict(ActionSelectionSig)

    def forward(self, state):
        # Stage 1: Generate reasoning
        reasoning_result = self.reasoner(state=state)

        # Stage 2: Select action given reasoning
        action_result = self.actor(
            state=state,
            reasoning=reasoning_result.reasoning
        )

        return dspy.Prediction(
            reasoning=reasoning_result.reasoning,
            action=action_result.action
        )
```

**File Location**: `src/models/two_stage_policy.py`

**Tradeoffs**:
- ✅ More control over reasoning/action separation
- ✅ Can validate reasoning structure before action selection
- ✅ Easier to debug reasoning independently
- ❌ Two LLM calls instead of one (slower, more expensive)
- ❌ More complex training (need separate datasets)
- **Recommendation**: Start with ChainOfThought, use two-stage if needed

---

## 2. Training with dspy.BootstrapFinetune

### 2.1 Training Data Format

**Structure**: Use `dspy.Example` with `.with_inputs()` to mark input fields.

```python
# World Model Training Examples
world_training_data = [
    dspy.Example(
        state="at intersection, light=red, vehicles=stopped",
        action="wait",
        next_state="at intersection, light=red, vehicles=stopped, waited=1_cycle"
    ).with_inputs("state", "action"),

    dspy.Example(
        state="at intersection, light=green, vehicles=moving",
        action="proceed",
        next_state="through intersection, vehicles=moving, crossed=true"
    ).with_inputs("state", "action"),

    # Add 20-50 examples covering key state transitions
]

# Policy Training Examples (with reasoning)
policy_training_data = [
    dspy.Example(
        state="at intersection, light=red, vehicles=stopped",
        reasoning="""
State: At red light with stopped traffic

Alternatives:
- Action A: wait → Expected: remain safe, legal, light will change
- Action B: run red light → Expected: traffic violation, collision risk

Analysis: Action A is zero-risk and legal. Action B violates traffic law
and creates collision hazard. No benefit to rushing.

Conclusion: Best action is wait because it ensures safety and compliance.
""",
        action="wait"
    ).with_inputs("state"),

    dspy.Example(
        state="at yellow light, 50 feet away, speed=35mph",
        reasoning="""
State: Approaching yellow light at 50ft traveling 35mph

Alternatives:
- Action A: brake hard → Expected: stop before intersection, safe
- Action B: maintain speed → Expected: enter on yellow/red, risky
- Action C: accelerate → Expected: clear yellow, dangerous

Analysis: At 35mph, stopping distance is ~80ft but partial braking achieves
safe stop. Action B/C create legal and safety risks.

Conclusion: Best action is brake because stopping is safest option.
""",
        action="brake"
    ).with_inputs("state"),

    # Add 30-100 examples with diverse scenarios and full reasoning
]
```

**File Location**: `data/training/world_model_examples.jsonl`, `data/training/policy_examples.jsonl`

**JSONL Format** (for easy storage):
```json
{"state": "at intersection, light=red", "action": "wait", "next_state": "at intersection, light=red, waited=1_cycle"}
{"state": "at yellow light, 50ft, 35mph", "reasoning": "State: Approaching yellow...", "action": "brake"}
```

**Loading from JSONL**:
```python
def load_training_examples(path: str) -> List[dspy.Example]:
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Determine input fields based on data structure
            if "next_state" in data:
                # World model example
                ex = dspy.Example(**data).with_inputs("state", "action")
            else:
                # Policy example
                ex = dspy.Example(**data).with_inputs("state")

            examples.append(ex)

    return examples
```

---

### 2.2 Training Workflow

```python
import dspy
from dspy.teleprompt import BootstrapFinetune

# Step 1: Configure LLM
def configure_lm():
    """Configure DSPy with your LLM backend."""
    model = "gpt-4o-mini"  # or "claude-3-5-sonnet-20240620"

    if "claude" in model.lower():
        lm = dspy.Anthropic(model=model, max_tokens=4000, temperature=0.2)
    else:
        lm = dspy.OpenAI(model=model, max_tokens=4000, temperature=0.2, seed=42)

    dspy.configure(lm=lm)

configure_lm()

# Step 2: Define metric functions
def world_model_metric(example, prediction, trace=None):
    """
    Metric for world model training.
    Returns 1.0 if prediction is reasonable, 0.0 otherwise.
    """
    # Check if next_state field exists and is non-empty
    if not prediction.next_state:
        return 0.0

    # Optional: Add semantic similarity check
    # from difflib import SequenceMatcher
    # similarity = SequenceMatcher(None, example.next_state, prediction.next_state).ratio()
    # return similarity

    return 1.0

def policy_metric(example, prediction, trace=None):
    """
    Metric for policy training.
    Returns 1.0 if action matches, 0.5 if reasoning is good but action differs, 0.0 otherwise.
    """
    # Check reasoning quality (must contain "Alternatives" and "Conclusion")
    has_alternatives = "Alternatives:" in prediction.reasoning or "Action A:" in prediction.reasoning
    has_conclusion = "Conclusion:" in prediction.reasoning or "best action" in prediction.reasoning.lower()

    reasoning_score = 0.3 if has_alternatives and has_conclusion else 0.0

    # Check action correctness
    action_score = 0.7 if example.action.lower() == prediction.action.lower() else 0.0

    return reasoning_score + action_score

# Step 3: Load training data
world_examples = load_training_examples("data/training/world_model_examples.jsonl")
policy_examples = load_training_examples("data/training/policy_examples.jsonl")

# Step 4: Create student modules
world_model_student = dspy.Predict(WorldModelSig)
policy_student = dspy.ChainOfThought(PolicySig)

# Step 5: Initialize optimizers
world_optimizer = BootstrapFinetune(
    metric=world_model_metric,
    num_threads=4  # Parallel training
)

policy_optimizer = BootstrapFinetune(
    metric=policy_metric,
    num_threads=4
)

# Step 6: Compile (train) modules
print("Training world model...")
world_model_compiled = world_optimizer.compile(
    student=world_model_student,
    trainset=world_examples
)

print("Training policy...")
policy_compiled = policy_optimizer.compile(
    student=policy_student,
    trainset=policy_examples
)

# Step 7: Save trained modules
import os
os.makedirs("models/trained", exist_ok=True)

world_model_compiled.save("models/trained/world_model.json")
policy_compiled.save("models/trained/policy.json")

print("Training complete! Saved to models/trained/")
```

**File Location**: `scripts/train_modules.py`

---

### 2.3 Best Practices for Training

**Data Quality**:
1. **Diversity**: Cover edge cases, normal cases, and rare scenarios
2. **Consistency**: Use consistent state/action formats across examples
3. **Quality over Quantity**: 50 high-quality examples > 500 low-quality
4. **Balance**: Ensure all action types are represented proportionally

**Metric Design**:
1. **Strict for Safety**: Use hard checks for safety-critical behaviors
2. **Flexible for Style**: Allow variation in reasoning format
3. **Partial Credit**: Reward partially correct outputs (0.0-1.0 scale)
4. **Fast Computation**: Keep metrics simple (avoid heavy computation)

**Training Configuration**:
```python
# For better convergence
optimizer = BootstrapFinetune(
    metric=policy_metric,
    num_threads=4,              # Parallel processing
    multitask=True,             # Multi-task learning if multiple modules
    exclude_demos=False,        # Include demonstration examples
)

# For faster iteration during development
optimizer = BootstrapFinetune(
    metric=policy_metric,
    num_threads=1,              # Sequential for debugging
    multitask=False
)
```

**Validation**:
```python
# Hold out 20% for validation
from sklearn.model_selection import train_test_split

train_examples, val_examples = train_test_split(
    policy_examples,
    test_size=0.2,
    random_state=42
)

# Train on training set
policy_compiled = policy_optimizer.compile(
    student=policy_student,
    trainset=train_examples
)

# Evaluate on validation set
def evaluate(module, examples, metric):
    total_score = 0.0
    for ex in examples:
        pred = module(**ex.inputs())
        score = metric(ex, pred)
        total_score += score
    return total_score / len(examples)

val_score = evaluate(policy_compiled, val_examples, policy_metric)
print(f"Validation score: {val_score:.2f}")
```

---

### 2.4 Persisting and Loading Modules

**Save Formats**:
```python
# Option 1: JSON (recommended for inspection/debugging)
module.save("models/policy.json")

# Option 2: Pickle (smaller, faster, binary)
module.save("models/policy.pkl")

# What gets saved:
# - Module configuration
# - Learned demonstrations (few-shot examples)
# - Signature structure
# - Metadata (timestamps, version, etc.)
```

**Load and Use**:
```python
import dspy

# Configure LM (required before loading)
lm = dspy.OpenAI(model="gpt-4o-mini", temperature=0.2)
dspy.configure(lm=lm)

# Load trained modules
world_model = dspy.Predict(WorldModelSig)
world_model.load("models/trained/world_model.json")

policy = dspy.ChainOfThought(PolicySig)
policy.load("models/trained/policy.json")

# Use in production
result = policy(state="at yellow light, 30 feet away, 40mph")
print(f"Decision: {result.action}")
print(f"Reasoning:\n{result.reasoning}")
```

**File Location**: `src/agents/agent_loader.py`

**Version Management**:
```python
# Include version in filename
version = "v1.2.0"
policy.save(f"models/policy_{version}.json")

# Load specific version
policy.load("models/policy_v1.2.0.json")

# Metadata tracking
import json
metadata = {
    "version": "1.2.0",
    "trained_date": "2025-10-12",
    "training_examples": len(train_examples),
    "val_score": val_score,
    "model_backend": "gpt-4o-mini"
}

with open("models/policy_v1.2.0_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## 3. ChainOfThought Module Deep Dive

### 3.1 How ChainOfThought Works

**Mechanism**:
- `ChainOfThought` is a specialized `Module` wrapper around `Predict`
- It automatically adds a `rationale_field` (default name: `reasoning`) to the signature
- The LLM is prompted to generate reasoning BEFORE the final outputs
- Output order is guaranteed: reasoning → other outputs

**Signature Transformation**:
```python
# Original signature
class SimplePolicySig(dspy.Signature):
    state: str = dspy.InputField()
    action: str = dspy.OutputField()

# What ChainOfThought does internally:
# 1. Adds reasoning field BEFORE action field
# 2. Prompts LLM to fill reasoning first
# 3. Then generates action conditioned on reasoning

# Equivalent expanded signature:
class ExpandedPolicySig(dspy.Signature):
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField()  # Added by ChainOfThought
    action: str = dspy.OutputField()
```

**Usage**:
```python
# Simple usage - reasoning field added automatically
policy = dspy.ChainOfThought(SimplePolicySig)
result = policy(state="at red light")
print(result.reasoning)  # Automatically available
print(result.action)

# Custom reasoning field name
policy = dspy.ChainOfThought(
    SimplePolicySig,
    rationale_field=dspy.OutputField(name="analysis")
)
result = policy(state="at red light")
print(result.analysis)  # Custom name
```

---

### 3.2 Ensuring Reasoning Before Action

**Guaranteed Order**: ChainOfThought enforces reasoning generation before action through prompt engineering and output ordering.

**Verification Pattern**:
```python
def verify_reasoning_first(result):
    """
    Verify that reasoning was generated and contains required structure.
    """
    # Check reasoning exists and is substantial
    if not result.reasoning or len(result.reasoning) < 50:
        raise ValueError("Reasoning too short or missing")

    # Check for EE structure
    required_keywords = ["Alternatives:", "Conclusion:", "Action"]
    for keyword in required_keywords:
        if keyword not in result.reasoning:
            raise ValueError(f"Missing required keyword: {keyword}")

    # Check action matches conclusion
    conclusion_lower = result.reasoning.lower()
    action_lower = result.action.lower()

    if action_lower not in conclusion_lower:
        print(f"WARNING: Action '{result.action}' not mentioned in conclusion")

    return True

# Use in production
result = policy(state="at intersection")
verify_reasoning_first(result)
```

---

### 3.3 Extracting Reasoning and Action

**Standard Extraction**:
```python
result = policy(state="at yellow light, 50ft, 35mph")

# Access fields directly
reasoning_text = result.reasoning
chosen_action = result.action

# Convert to dict
result_dict = {
    "reasoning": result.reasoning,
    "action": result.action,
    "state": "at yellow light, 50ft, 35mph"  # Original input
}

# Save to log
import json
with open("logs/decisions.jsonl", "a") as f:
    f.write(json.dumps(result_dict) + "\n")
```

**Structured Parsing** (extract EE components):
```python
import re

def parse_ee_reasoning(reasoning: str) -> dict:
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
    analysis_match = re.search(r"Analysis:\s*(.+?)(?:\n\n|Conclusion:)", reasoning, re.DOTALL)
    if analysis_match:
        components["analysis"] = analysis_match.group(1).strip()

    # Extract conclusion
    conclusion_match = re.search(r"Conclusion:\s*(.+?)(?:\n|$)", reasoning)
    if conclusion_match:
        components["conclusion"] = conclusion_match.group(1).strip()

    return components

# Usage
result = policy(state="at yellow light")
parsed = parse_ee_reasoning(result.reasoning)

print(f"Considered {len(parsed['alternatives'])} alternatives")
for alt in parsed["alternatives"]:
    print(f"  {alt['label']}: {alt['action']} → {alt['expected']}")
```

**File Location**: `src/utils/reasoning_parser.py`

---

### 3.4 Custom Reasoning Patterns (EE Template Injection)

**Method 1: Via Field Description** (Recommended)
```python
class EEPolicySig(dspy.Signature):
    """Policy with embedded EE reasoning template."""
    state: str = dspy.InputField()

    reasoning: str = dspy.OutputField(
        desc="""
        REQUIRED FORMAT:

        State: <current situation>

        Alternatives:
        - Action A: <what> → Expected Outcome: <prediction, evidence>
        - Action B: <what> → Expected Outcome: <prediction, evidence>

        Analysis: <compare risks, benefits, reversibility>

        Conclusion: Best action is <X> because <reasons>.
        """
    )

    action: str = dspy.OutputField()

policy = dspy.ChainOfThought(EEPolicySig)
```

**Method 2: Via Instruction in Signature Docstring**
```python
class EEPolicySig(dspy.Signature):
    """
    Policy decision with Engineering Excellence reasoning.

    REASONING REQUIREMENTS:
    1. State the current situation briefly
    2. List at least 2 alternative actions with predicted outcomes
    3. Analyze trade-offs (safety, legality, efficiency)
    4. Conclude with best action and justification

    Format each alternative as: "Action X: <what> → Expected: <outcome>"
    """
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    action: str = dspy.OutputField()
```

**Method 3: Via Few-Shot Examples in Training**
```python
# Training examples with perfect EE structure
examples = [
    dspy.Example(
        state="at red light",
        reasoning="""
State: Stopped at red light with clear intersection ahead

Alternatives:
- Action A: wait → Expected: remain safe, light changes in 30-60s
- Action B: proceed → Expected: traffic violation, collision risk high

Analysis: Action A has zero risk and complies with law. Action B violates
traffic code and creates T-bone collision hazard. No urgency justifies risk.

Conclusion: Best action is wait because safety and legality are paramount.
""",
        action="wait"
    ).with_inputs("state"),
    # ... more examples with consistent format
]

# Train with these examples
optimizer = BootstrapFinetune(metric=policy_metric)
compiled_policy = optimizer.compile(
    student=dspy.ChainOfThought(PolicySig),
    trainset=examples
)

# Compiled policy will learn the EE format from examples
```

**Validation Function**:
```python
def validate_ee_format(reasoning: str) -> tuple[bool, list[str]]:
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

# Use in training metric
def strict_policy_metric(example, prediction, trace=None):
    is_valid, errors = validate_ee_format(prediction.reasoning)

    if not is_valid:
        print(f"EE format errors: {errors}")
        return 0.0

    # If format is valid, check action correctness
    return 1.0 if example.action == prediction.action else 0.5
```

---

## 4. Module Composition

### 4.1 Sequential Chaining (World Model → Policy)

**Pattern**: Use world model predictions to inform policy decisions.

```python
class AgentWithWorldModel(dspy.Module):
    """
    Agent that uses world model to predict outcomes before deciding action.

    Flow:
    1. Receive current state
    2. Query world model for each candidate action
    3. Pass predictions to policy for final decision
    """

    def __init__(self):
        super().__init__()
        self.world_model = dspy.Predict(WorldModelSig)
        self.policy = dspy.ChainOfThought(PolicySig)

    def forward(self, state: str, candidate_actions: list[str] = None):
        """
        Args:
            state: Current environment state
            candidate_actions: List of actions to consider (default: ["stop", "slow", "proceed"])

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
            world_model_predictions=predictions,
            enhanced_state=enhanced_state
        )

# Usage
agent = AgentWithWorldModel()
result = agent(
    state="at yellow light, 50ft, 35mph",
    candidate_actions=["brake", "maintain", "accelerate"]
)

print("World Model Predictions:")
for action, next_state in result.world_model_predictions.items():
    print(f"  {action}: {next_state}")

print(f"\nReasoning:\n{result.reasoning}")
print(f"\nChosen Action: {result.action}")
```

**File Location**: `src/agents/agent_with_world_model.py`

---

### 4.2 World Model as Tool (Integrated in Reasoning)

**Pattern**: Policy signature explicitly includes world model predictions as input.

```python
class EnhancedPolicySig(dspy.Signature):
    """
    Policy that receives world model predictions as context.
    """
    state: str = dspy.InputField(desc="current environment state")
    predictions: str = dspy.InputField(
        desc="world model predictions for candidate actions"
    )
    reasoning: str = dspy.OutputField(desc="EE-style reasoning")
    action: str = dspy.OutputField(desc="chosen action")

class PolicyWithEmbeddedWorldModel(dspy.Module):
    """
    Policy that uses world model as an internal tool.
    World model predictions are passed as explicit input to policy reasoning.
    """

    def __init__(self):
        super().__init__()
        self.world_model = dspy.Predict(WorldModelSig)
        self.policy = dspy.ChainOfThought(EnhancedPolicySig)

    def forward(self, state: str, num_alternatives: int = 3):
        """
        Generate world model predictions and pass to policy.

        Args:
            state: Current state
            num_alternatives: Number of actions to simulate (default: 3)
        """
        # Define action candidates (could be learned or fixed)
        action_library = ["stop", "slow", "proceed", "brake", "accelerate", "wait"]
        candidates = action_library[:num_alternatives]

        # Simulate outcomes
        predictions_list = []
        for action in candidates:
            pred = self.world_model(state=state, action=action)
            predictions_list.append(f"{action} → {pred.next_state}")

        predictions_text = "\n".join(predictions_list)

        # Policy makes decision with predictions as context
        decision = self.policy(
            state=state,
            predictions=predictions_text
        )

        return decision

# Usage
policy = PolicyWithEmbeddedWorldModel()
result = policy(state="at red light, cars waiting")

print(f"Reasoning:\n{result.reasoning}")
print(f"Action: {result.action}")
```

**File Location**: `src/agents/policy_with_embedded_world_model.py`

---

### 4.3 Parallel Evaluation (Ensemble)

**Pattern**: Run multiple policies/world models in parallel and aggregate.

```python
class EnsembleAgent(dspy.Module):
    """
    Agent that runs multiple policies and aggregates their decisions.
    Useful for robustness and uncertainty estimation.
    """

    def __init__(self, num_policies: int = 3):
        super().__init__()
        # Create multiple policy instances (could use different training data)
        self.policies = [
            dspy.ChainOfThought(PolicySig)
            for _ in range(num_policies)
        ]

        # Optional: Load different trained versions
        for i, policy in enumerate(self.policies):
            policy.load(f"models/policy_v{i+1}.json")

    def forward(self, state: str):
        """
        Run all policies and aggregate results.
        """
        # Get decisions from all policies
        results = [policy(state=state) for policy in self.policies]

        # Count action votes
        from collections import Counter
        action_votes = Counter([r.action for r in results])
        majority_action = action_votes.most_common(1)[0][0]

        # Aggregate reasoning (concatenate or select best)
        aggregated_reasoning = "\n\n---\n\n".join([
            f"Policy {i+1}:\n{r.reasoning}"
            for i, r in enumerate(results)
        ])

        # Calculate confidence based on agreement
        confidence = action_votes[majority_action] / len(self.policies)

        return dspy.Prediction(
            action=majority_action,
            reasoning=aggregated_reasoning,
            confidence=confidence,
            individual_results=results
        )

# Usage
ensemble = EnsembleAgent(num_policies=3)
result = ensemble(state="at yellow light")

print(f"Consensus Action: {result.action}")
print(f"Confidence: {result.confidence:.1%}")
```

**File Location**: `src/agents/ensemble_agent.py`

---

### 4.4 Hierarchical (Meta-Policy)

**Pattern**: High-level policy selects which low-level policy to use.

```python
class MetaPolicySig(dspy.Signature):
    """Select which policy to use based on state characteristics."""
    state: str = dspy.InputField()
    policy_choice: str = dspy.OutputField(
        desc="one of: conservative, aggressive, balanced"
    )

class HierarchicalAgent(dspy.Module):
    """
    Agent with multiple specialized policies and a meta-policy selector.
    """

    def __init__(self):
        super().__init__()

        # Meta-policy chooses which policy to use
        self.meta_policy = dspy.Predict(MetaPolicySig)

        # Specialized policies
        self.conservative_policy = dspy.ChainOfThought(PolicySig)
        self.aggressive_policy = dspy.ChainOfThought(PolicySig)
        self.balanced_policy = dspy.ChainOfThought(PolicySig)

        # Load trained versions
        self.conservative_policy.load("models/policy_conservative.json")
        self.aggressive_policy.load("models/policy_aggressive.json")
        self.balanced_policy.load("models/policy_balanced.json")

        self.policy_map = {
            "conservative": self.conservative_policy,
            "aggressive": self.aggressive_policy,
            "balanced": self.balanced_policy
        }

    def forward(self, state: str):
        """
        Select and execute appropriate policy.
        """
        # Step 1: Meta-policy selects policy type
        selection = self.meta_policy(state=state)
        chosen_policy_name = selection.policy_choice

        # Step 2: Execute selected policy
        policy = self.policy_map.get(
            chosen_policy_name,
            self.balanced_policy  # Fallback
        )
        result = policy(state=state)

        # Step 3: Annotate with meta-decision
        return dspy.Prediction(
            action=result.action,
            reasoning=result.reasoning,
            policy_used=chosen_policy_name
        )

# Usage
agent = HierarchicalAgent()
result = agent(state="at red light with ambulance behind")

print(f"Policy Used: {result.policy_used}")
print(f"Action: {result.action}")
```

**File Location**: `src/agents/hierarchical_agent.py`

---

## 5. Serialization and Model Management

### 5.1 File Formats

**JSON Format** (.json):
```json
{
  "predict": {
    "lm": null,
    "demos": [
      {
        "state": "at red light",
        "reasoning": "State: At red light...",
        "action": "stop"
      }
    ],
    "signature": "PolicySig"
  },
  "metadata": {
    "timestamp": "2025-10-12T10:30:00Z",
    "dspy_version": "3.0.3",
    "training_examples": 50
  }
}
```

**Pickle Format** (.pkl):
- Binary serialization of entire module state
- Faster to load but not human-readable
- Smaller file size

**Recommendation**: Use JSON for development/debugging, Pickle for production.

---

### 5.2 Versioning Strategy

```python
# Version naming convention
VERSION_FORMAT = "v{major}.{minor}.{patch}"

# Example: v1.2.3
# - major: Breaking changes to signature or behavior
# - minor: New features, additional training data
# - patch: Bug fixes, parameter tuning

def save_versioned_model(module, base_path: str, version: str, metadata: dict):
    """
    Save model with version information.

    Args:
        module: DSPy module to save
        base_path: Base path (e.g., "models/policy")
        version: Version string (e.g., "v1.2.3")
        metadata: Additional metadata to save
    """
    import json
    from datetime import datetime

    # Save model
    model_path = f"{base_path}_{version}.json"
    module.save(model_path)

    # Save metadata
    metadata_full = {
        "version": version,
        "saved_at": datetime.now().isoformat(),
        "model_path": model_path,
        **metadata
    }

    metadata_path = f"{base_path}_{version}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_full, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")

    return model_path, metadata_path

# Usage
save_versioned_model(
    module=policy_compiled,
    base_path="models/policy",
    version="v1.2.0",
    metadata={
        "training_examples": len(train_examples),
        "val_score": 0.87,
        "model_backend": "gpt-4o-mini",
        "notes": "Added emergency vehicle scenarios"
    }
)
```

**File Location**: `src/utils/model_versioning.py`

---

### 5.3 Model Registry

```python
import json
from pathlib import Path
from typing import Dict, Optional

class ModelRegistry:
    """
    Centralized registry for tracking trained models.
    """

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register(self, name: str, version: str, model_path: str, metadata: Dict):
        """Register a new model version."""
        if name not in self.registry["models"]:
            self.registry["models"][name] = {"versions": {}}

        self.registry["models"][name]["versions"][version] = {
            "path": model_path,
            "metadata": metadata
        }

        # Set as latest
        self.registry["models"][name]["latest"] = version

        self._save_registry()

    def get_model_path(self, name: str, version: Optional[str] = None) -> str:
        """Get path to model (latest if version not specified)."""
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")

        if version is None:
            version = self.registry["models"][name]["latest"]

        return self.registry["models"][name]["versions"][version]["path"]

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        if name not in self.registry["models"]:
            return []
        return list(self.registry["models"][name]["versions"].keys())

# Usage
registry = ModelRegistry()

# Register after training
registry.register(
    name="policy",
    version="v1.2.0",
    model_path="models/policy_v1.2.0.json",
    metadata={"val_score": 0.87, "training_date": "2025-10-12"}
)

# Load latest version
policy = dspy.ChainOfThought(PolicySig)
policy.load(registry.get_model_path("policy"))

# Load specific version
policy.load(registry.get_model_path("policy", version="v1.1.0"))

# List available versions
versions = registry.list_versions("policy")
print(f"Available versions: {versions}")
```

**File Location**: `src/utils/model_registry.py`

---

## 6. Integration with EE Constitution

### 6.1 Constitution Compliance Check

```python
def check_ee_compliance(reasoning: str) -> dict:
    """
    Check if reasoning complies with EE constitution requirements.

    Returns:
        {
            "compliant": bool,
            "score": float (0-1),
            "violations": List[str],
            "recommendations": List[str]
        }
    """
    violations = []
    recommendations = []
    score_components = {}

    # Requirement 1: Compare at least 2 alternatives
    alt_count = len(re.findall(r"Action [A-Z]:", reasoning))
    if alt_count < 2:
        violations.append("EE Principle I: Must compare at least 2 alternatives")
        score_components["alternatives"] = 0.0
    else:
        score_components["alternatives"] = min(1.0, alt_count / 3.0)

    # Requirement 2: Predict outcomes for each alternative
    has_expected = reasoning.count("Expected") >= alt_count
    if not has_expected:
        violations.append("EE Principle I: Must predict outcomes for each alternative")
        score_components["outcomes"] = 0.0
    else:
        score_components["outcomes"] = 1.0

    # Requirement 3: Risk/assumption consideration
    has_risk_analysis = any(
        keyword in reasoning.lower()
        for keyword in ["risk", "assumption", "uncertainty", "trade-off"]
    )
    if not has_risk_analysis:
        recommendations.append("Consider adding risk analysis to strengthen reasoning")
        score_components["risk"] = 0.5
    else:
        score_components["risk"] = 1.0

    # Requirement 4: Justified expert choice
    has_justification = "because" in reasoning.lower() or "therefore" in reasoning.lower()
    if not has_justification:
        violations.append("EE Principle I: Must justify chosen action")
        score_components["justification"] = 0.0
    else:
        score_components["justification"] = 1.0

    # Calculate total score
    total_score = sum(score_components.values()) / len(score_components)

    return {
        "compliant": len(violations) == 0,
        "score": total_score,
        "score_components": score_components,
        "violations": violations,
        "recommendations": recommendations
    }

# Usage in metric
def ee_compliant_metric(example, prediction, trace=None):
    """Metric that enforces EE constitution compliance."""
    compliance = check_ee_compliance(prediction.reasoning)

    # Log violations
    if not compliance["compliant"]:
        print(f"EE violations: {compliance['violations']}")

    # Weight: 70% compliance + 30% action correctness
    compliance_score = compliance["score"] * 0.7
    action_score = (1.0 if example.action == prediction.action else 0.0) * 0.3

    return compliance_score + action_score
```

**File Location**: `src/utils/constitution_checker.py`

---

### 6.2 EE-Aware Training Pipeline

```python
from typing import List
import dspy
from dspy.teleprompt import BootstrapFinetune

def train_ee_compliant_policy(
    training_examples: List[dspy.Example],
    val_examples: List[dspy.Example],
    min_compliance_score: float = 0.7
) -> dspy.Module:
    """
    Train policy with EE constitution compliance checks.

    Args:
        training_examples: Training data (must include EE-formatted reasoning)
        val_examples: Validation data
        min_compliance_score: Minimum EE compliance score (0-1)

    Returns:
        Trained and validated DSPy module
    """

    # Metric combines EE compliance + action correctness
    def ee_metric(example, prediction, trace=None):
        compliance = check_ee_compliance(prediction.reasoning)

        # Fail if below minimum compliance
        if compliance["score"] < min_compliance_score:
            return 0.0

        # Weight compliance and correctness
        compliance_weight = 0.6
        action_weight = 0.4

        action_correct = 1.0 if example.action == prediction.action else 0.0

        return (compliance["score"] * compliance_weight +
                action_correct * action_weight)

    # Train with EE-aware metric
    policy_student = dspy.ChainOfThought(PolicySig)
    optimizer = BootstrapFinetune(metric=ee_metric, num_threads=4)

    print("Training EE-compliant policy...")
    policy_compiled = optimizer.compile(
        student=policy_student,
        trainset=training_examples
    )

    # Validate on holdout set
    print("\nValidating on holdout set...")
    val_scores = []
    compliance_scores = []

    for ex in val_examples:
        pred = policy_compiled(**ex.inputs())
        score = ee_metric(ex, pred)
        val_scores.append(score)

        compliance = check_ee_compliance(pred.reasoning)
        compliance_scores.append(compliance["score"])

    avg_score = sum(val_scores) / len(val_scores)
    avg_compliance = sum(compliance_scores) / len(compliance_scores)

    print(f"\nValidation Results:")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Average EE Compliance: {avg_compliance:.2f}")

    # Fail if validation doesn't meet threshold
    if avg_compliance < min_compliance_score:
        raise ValueError(
            f"Model failed validation: compliance {avg_compliance:.2f} "
            f"< threshold {min_compliance_score}"
        )

    print("✓ Model passed EE compliance validation")
    return policy_compiled

# Usage
policy = train_ee_compliant_policy(
    training_examples=policy_train,
    val_examples=policy_val,
    min_compliance_score=0.75
)

policy.save("models/ee_compliant_policy_v1.0.0.json")
```

**File Location**: `scripts/train_ee_compliant.py`

---

## 7. Recommended Project Structure

```
agent-learning-ee/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── world_model.py          # WorldModelSig definition
│   │   ├── policy.py                # PolicySig definitions
│   │   └── signatures.py            # All signature definitions
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent_loader.py          # Load/configure agents
│   │   ├── agent_with_world_model.py
│   │   ├── policy_with_embedded_world_model.py
│   │   ├── ensemble_agent.py
│   │   └── hierarchical_agent.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── reasoning_parser.py      # Parse EE reasoning
│       ├── constitution_checker.py  # EE compliance checks
│       ├── model_versioning.py      # Version management
│       └── model_registry.py        # Model registry
│
├── scripts/
│   ├── train_modules.py             # Basic training script
│   ├── train_ee_compliant.py        # EE-compliant training
│   └── evaluate_models.py           # Evaluation scripts
│
├── data/
│   ├── training/
│   │   ├── world_model_examples.jsonl
│   │   └── policy_examples.jsonl
│   └── validation/
│       ├── world_model_val.jsonl
│       └── policy_val.jsonl
│
├── models/
│   ├── trained/
│   │   ├── world_model_v1.0.0.json
│   │   ├── policy_v1.0.0.json
│   │   └── *.pkl
│   ├── registry.json
│   └── README.md                    # Model documentation
│
├── tests/
│   ├── test_world_model.py
│   ├── test_policy.py
│   ├── test_agents.py
│   └── test_ee_compliance.py
│
├── .env.example                     # Example environment variables
├── requirements.txt                 # dspy-ai>=3.0.3, etc.
└── README.md
```

---

## 8. Key Technical Decisions

### Decision 1: ChainOfThought vs Two-Stage

**Recommendation**: Use `ChainOfThought` for initial implementation.

**Reasoning**:
- State: Need reasoning before action in policy module
- Alternatives:
  - A: ChainOfThought → Expected: One LLM call, automatic reasoning
  - B: Two-stage → Expected: More control, two LLM calls, slower
- Analysis: ChainOfThought provides sufficient control via signature description. Two-stage adds complexity without clear benefit for initial version. Can migrate later if needed.
- Conclusion: Best action is ChainOfThought because it balances simplicity and effectiveness.

**Action**: Implement PolicySig with ChainOfThought and EE template in description.

---

### Decision 2: JSON vs Pickle for Serialization

**Recommendation**: Use JSON for development, Pickle for production.

**Reasoning**:
- State: Need to persist trained models
- Alternatives:
  - A: JSON → Expected: Human-readable, larger files, slower load
  - B: Pickle → Expected: Faster, smaller, not inspectable
- Analysis: JSON enables debugging and version control inspection. Pickle optimizes production performance. No reason to choose only one.
- Conclusion: Best action is dual-format because different stages have different needs.

**Action**: Save both formats, document when to use each.

---

### Decision 3: World Model Integration Pattern

**Recommendation**: Use Pattern 2 (world model as tool with predictions as input).

**Reasoning**:
- State: Need to integrate world model predictions into policy
- Alternatives:
  - A: Sequential chaining → Expected: Simple, less integrated reasoning
  - B: Embedded predictions → Expected: LLM sees predictions directly, more holistic
  - C: Hierarchical → Expected: Most complex, overkill for MVP
- Analysis: Pattern B enables LLM to reason about predictions directly in EE format. Pattern A requires additional prompt engineering to inject context.
- Conclusion: Best action is embedded predictions because it naturally fits EE alternative comparison.

**Action**: Implement PolicyWithEmbeddedWorldModel as primary agent class.

---

### Decision 4: Training Data Format

**Recommendation**: JSONL format with full reasoning chains.

**Reasoning**:
- State: Need format for training data storage and versioning
- Alternatives:
  - A: JSONL → Expected: Line-by-line, git-friendly, appendable
  - B: Single JSON → Expected: Easier parsing, not appendable
  - C: CSV → Expected: Limited for nested structures like reasoning
- Analysis: JSONL allows easy appending and git diff. Each example is self-contained. Supports streaming loading.
- Conclusion: Best action is JSONL because it optimizes for iteration and version control.

**Action**: Use JSONL for all training/validation data.

---

## 9. Next Steps

### Immediate Actions (Week 1):
1. ✅ Complete research.md (this document)
2. ⬜ Implement signature definitions in `src/models/signatures.py`
3. ⬜ Create basic training examples (10-20 per module)
4. ⬜ Write training script with BootstrapFinetune
5. ⬜ Implement EE compliance checker

### Short-term (Week 2-3):
6. ⬜ Expand training data to 50+ examples per module
7. ⬜ Train and validate initial models
8. ⬜ Implement PolicyWithEmbeddedWorldModel agent
9. ⬜ Write comprehensive tests
10. ⬜ Create model registry and versioning system

### Medium-term (Month 1-2):
11. ⬜ Implement ensemble/hierarchical patterns
12. ⬜ Build evaluation framework
13. ⬜ Optimize metrics based on real usage
14. ⬜ Create production deployment pipeline
15. ⬜ Document lessons learned

---

## 10. References

### DSPy Documentation:
- Main repo: https://github.com/stanfordnlp/dspy
- Paper: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
- Examples: https://github.com/stanfordnlp/dspy/tree/main/examples

### Related Work:
- JTBD Idea Validator (this repository): Uses GEPA optimizer with JudgeScoreSig
- Engineering Excellence Constitution: `.specify/memory/constitution.md`

### Internal Files Referenced:
- `plugins/llm_dspy.py` - DSPy configuration and modules (from JTBD project)
- `tools/optimize_judge.py` - GEPA optimizer example (from JTBD project)
- `.specify/memory/constitution.md` - EE principles and decision framework

---

## Appendix A: Complete Working Example

```python
"""
Complete working example: Train and use EE-compliant policy with world model.

Run this script to see end-to-end workflow.
"""

import dspy
from dspy.teleprompt import BootstrapFinetune
import json

# 1. Configure LLM
lm = dspy.OpenAI(model="gpt-4o-mini", temperature=0.2, seed=42)
dspy.configure(lm=lm)

# 2. Define signatures
class WorldModelSig(dspy.Signature):
    """Predict next state."""
    state: str = dspy.InputField()
    action: str = dspy.InputField()
    next_state: str = dspy.OutputField()

class PolicySig(dspy.Signature):
    """EE policy with reasoning."""
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField(
        desc="""
        EE Format:
        State: <situation>
        Alternatives:
        - Action A: <what> → Expected: <outcome>
        - Action B: <what> → Expected: <outcome>
        Analysis: <compare>
        Conclusion: Best is <X> because <why>.
        """
    )
    action: str = dspy.OutputField()

# 3. Create training data
world_examples = [
    dspy.Example(
        state="at red light",
        action="wait",
        next_state="at red light, waited"
    ).with_inputs("state", "action"),
]

policy_examples = [
    dspy.Example(
        state="at red light",
        reasoning="""
State: At red light

Alternatives:
- Action A: wait → Expected: safe, legal
- Action B: proceed → Expected: violation, collision risk

Analysis: Action A is zero-risk. B violates law.

Conclusion: Best is wait because safety is paramount.
""",
        action="wait"
    ).with_inputs("state"),
]

# 4. Train modules
world_model = dspy.Predict(WorldModelSig)
world_optimizer = BootstrapFinetune(
    metric=lambda ex, pred, trace: 1.0 if pred.next_state else 0.0
)
world_trained = world_optimizer.compile(
    student=world_model,
    trainset=world_examples
)

policy = dspy.ChainOfThought(PolicySig)
policy_optimizer = BootstrapFinetune(
    metric=lambda ex, pred, trace: 1.0 if ex.action == pred.action else 0.0
)
policy_trained = policy_optimizer.compile(
    student=policy,
    trainset=policy_examples
)

# 5. Save models
world_trained.save("world_model.json")
policy_trained.save("policy.json")

# 6. Use in production
world_loaded = dspy.Predict(WorldModelSig)
world_loaded.load("world_model.json")

policy_loaded = dspy.ChainOfThought(PolicySig)
policy_loaded.load("policy.json")

# Test
result = policy_loaded(state="at yellow light")
print(f"Reasoning:\n{result.reasoning}\n")
print(f"Action: {result.action}")
```

---

**End of Research Document**

**Next Steps**: Review with team, implement signature definitions, begin training data collection.
