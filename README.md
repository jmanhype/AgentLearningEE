# Agent Learning via Early Experience (EE)

**DSPy-Powered Agent with Engineering Excellence Decision-Making**

This repository implements an agent learning system using DSPy that embeds Engineering Excellence (EE) principles into its decision-making process. The agent learns from early experiences through supervised fine-tuning on structured reasoning examples.

---

## Overview

This project demonstrates how to:
- Define DSPy signatures for world models and EE-compliant policies
- Train modules using BootstrapFinetune with custom metrics
- Compose modules for multi-stage reasoning (world model → policy)
- Enforce Engineering Excellence structured reasoning format
- Version and manage trained models

**Key Features**:
- 🧠 **World Model**: Predicts state transitions for planning
- 🎯 **EE-Style Policy**: Makes decisions with structured alternatives comparison
- 🔄 **Module Composition**: Integrates world model predictions into policy reasoning
- ✅ **EE Compliance**: Validates reasoning follows Engineering Excellence format
- 📦 **Model Management**: Versioned storage with metadata tracking

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install dspy-ai>=3.0.3

# Set API keys
export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY for Claude
```

### Training

```bash
python IMPLEMENTATION_STARTER.py train
```

This will:
1. Create example training data in `data/training/`
2. Train world model and policy with EE metrics
3. Save trained models to `models/trained/`

### Inference

```bash
python IMPLEMENTATION_STARTER.py infer
```

This will:
1. Load trained models
2. Run example scenario: "at yellow light, 50 feet away, speed=35mph"
3. Show world model predictions, policy reasoning, and chosen action

---

## Repository Structure

```
agent-learning-ee/
├── README.md                      # This file
├── research.md                    # Comprehensive DSPy research (60+ pages)
├── QUICK_REFERENCE.md             # Quick-start patterns and recipes
├── IMPLEMENTATION_STARTER.py      # Ready-to-use training/inference code
│
├── data/
│   └── training/                  # Training data (JSONL format)
│       ├── world_model_examples.jsonl
│       └── policy_examples.jsonl
│
├── models/
│   └── trained/                   # Trained models
│       ├── world_model_v1.0.0.json
│       └── policy_v1.0.0.json
│
└── .specify/
    └── memory/
        └── constitution.md        # Engineering Excellence principles
```

---

## Documentation

### 📚 [research.md](research.md)
**Comprehensive DSPy implementation research** (60+ pages)

Covers:
- Signature design patterns for world models and policies
- Training with BootstrapFinetune (data format, metrics, best practices)
- ChainOfThought deep dive (reasoning before action)
- Module composition patterns (sequential, ensemble, hierarchical)
- Serialization and model management
- Integration with EE constitution

### 📖 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Quick-start guide with code snippets**

Contains:
- Basic signature patterns
- 5-step training workflow
- Loading and using models
- Module composition examples
- Common gotchas and debugging tips

### 💻 [IMPLEMENTATION_STARTER.py](IMPLEMENTATION_STARTER.py)
**Ready-to-use implementation** (500+ lines)

Includes:
- Complete signature definitions
- Training data utilities (load JSONL, create examples)
- Metric functions (world model, policy, EE-compliant)
- Training functions for both modules
- AgentWithWorldModel composition class
- EE parsing and validation utilities
- CLI for training and inference

---

## Key Concepts

### 1. World Model

Predicts next state given current state and action:

```python
class WorldModelSig(dspy.Signature):
    """Predict next state given current state and action."""
    state: str = dspy.InputField()
    action: str = dspy.InputField()
    next_state: str = dspy.OutputField()

world_model = dspy.Predict(WorldModelSig)
result = world_model(state="at red light", action="wait")
```

### 2. EE-Style Policy

Makes decisions with structured reasoning comparing alternatives:

```python
class PolicySig(dspy.Signature):
    """EE-style decision with alternatives comparison."""
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="EE format: State, Alternatives, Analysis, Conclusion")
    action: str = dspy.OutputField()

policy = dspy.ChainOfThought(PolicySig)
result = policy(state="at yellow light")
```

**Example reasoning output**:
```
State: Approaching yellow light at 50ft traveling 35mph

Alternatives:
- Action A: brake → Expected: stop safely before intersection
- Action B: maintain → Expected: enter on yellow/red, risky
- Action C: accelerate → Expected: clear yellow, dangerous

Analysis: Action A ensures safety and legal compliance. B/C create
collision risks and traffic violations.

Conclusion: Best action is brake because safety is paramount.
```

### 3. Module Composition

Integrate world model predictions into policy reasoning:

```python
class AgentWithWorldModel(dspy.Module):
    def __init__(self):
        super().__init__()
        self.world_model = dspy.Predict(WorldModelSig)
        self.policy = dspy.ChainOfThought(PolicySig)

    def forward(self, state):
        # Simulate outcomes for candidate actions
        predictions = {}
        for action in ["stop", "slow", "proceed"]:
            pred = self.world_model(state=state, action=action)
            predictions[action] = pred.next_state

        # Policy decides with predictions as context
        enhanced_state = f"{state}\n\nPredictions:\n{predictions}"
        decision = self.policy(state=enhanced_state)

        return decision
```

---

## Training Data Format

**JSONL** (JSON Lines) format with one example per line:

### World Model Examples
```json
{"state": "at red light", "action": "wait", "next_state": "at red light, waited"}
{"state": "at green light", "action": "proceed", "next_state": "through intersection"}
```

### Policy Examples (with full reasoning)
```json
{"state": "at red light", "reasoning": "State: At red light...\n\nAlternatives:\n- Action A: wait → ...\n\nConclusion: ...", "action": "wait"}
```

**Load with**:
```python
from IMPLEMENTATION_STARTER import load_training_examples

examples = load_training_examples("data/training/policy_examples.jsonl")
```

---

## Engineering Excellence (EE) Format

All policy reasoning must follow this structure:

```
State: <concise situation description>

Alternatives:
- Action A: <what> → Expected Outcome: <prediction, evidence>
- Action B: <what> → Expected Outcome: <prediction, evidence>
- (Optional) Action C: <what> → Expected Outcome: <prediction, evidence>

Analysis: <compare risks, benefits, reversibility, testability>

Conclusion: Therefore, best action is <X> because <reasons>.
```

**Validation**:
```python
from IMPLEMENTATION_STARTER import validate_ee_format

is_valid, errors = validate_ee_format(reasoning)
if not is_valid:
    print(f"EE format errors: {errors}")
```

**Compliance Metrics**:
- Checks for required sections (State, Alternatives, Analysis, Conclusion)
- Validates at least 2 alternatives present
- Enforces action matches conclusion
- Used in training metrics to reinforce correct format

---

## Metrics

### World Model Metric
```python
def world_model_metric(example, prediction, trace=None):
    # 1.0 if predicted next_state is reasonable, 0.0 otherwise
    return 1.0 if prediction.next_state else 0.0
```

### Policy Metric (Basic)
```python
def policy_metric(example, prediction, trace=None):
    # 0.3 for EE structure + 0.7 for correct action
    has_structure = "Alternatives:" in prediction.reasoning
    structure_score = 0.3 if has_structure else 0.0
    action_score = 0.7 if example.action == prediction.action else 0.0
    return structure_score + action_score
```

### EE-Compliant Metric (Strict)
```python
def ee_compliant_metric(example, prediction, trace=None):
    # Enforces full EE format + at least 2 alternatives
    is_valid, _ = validate_ee_format(prediction.reasoning)
    if not is_valid:
        return 0.0
    # 0.4 for structure + 0.6 for correct action
    return 0.4 + (0.6 if example.action == prediction.action else 0.0)
```

---

## Model Management

### Saving
```python
# JSON (recommended for development)
model.save("models/policy.json")

# Pickle (recommended for production)
model.save("models/policy.pkl")
```

### Loading
```python
import dspy

# Configure LLM first
dspy.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

# Load model
policy = dspy.ChainOfThought(PolicySig)
policy.load("models/policy.json")
```

### Versioning
```python
# Save with version
version = "v1.2.0"
model.save(f"models/policy_{version}.json")

# Include metadata
metadata = {
    "version": version,
    "trained_date": "2025-10-12",
    "training_examples": 50,
    "val_score": 0.87
}

with open(f"models/policy_{version}_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## Usage Examples

### Basic Policy Usage
```python
from IMPLEMENTATION_STARTER import configure_lm, PolicySig
import dspy

configure_lm()
policy = dspy.ChainOfThought(PolicySig)
policy.load("models/trained/policy_v1.0.0.json")

result = policy(state="at intersection, light=red")
print(result.reasoning)  # Full EE reasoning
print(result.action)     # Chosen action
```

### Agent with World Model
```python
from IMPLEMENTATION_STARTER import configure_lm, AgentWithWorldModel

configure_lm()
agent = AgentWithWorldModel(
    world_model_path="models/trained/world_model_v1.0.0.json",
    policy_path="models/trained/policy_v1.0.0.json"
)

result = agent(state="at yellow light, 50ft, 35mph")

print("World Model Predictions:")
for action, next_state in result.world_model_predictions.items():
    print(f"  {action}: {next_state}")

print(f"\nChosen Action: {result.action}")
print(f"\nReasoning:\n{result.reasoning}")
```

### Parsing EE Reasoning
```python
from IMPLEMENTATION_STARTER import parse_ee_reasoning

parsed = parse_ee_reasoning(result.reasoning)

print(f"State: {parsed['state']}")
print(f"Alternatives: {len(parsed['alternatives'])}")
for alt in parsed["alternatives"]:
    print(f"  {alt['label']}: {alt['action']} → {alt['expected']}")
print(f"Conclusion: {parsed['conclusion']}")
```

---

## Extending the System

### Add New Signatures

Create custom signatures in your own module:

```python
class RiskAssessmentSig(dspy.Signature):
    """Assess risks for a given state and action."""
    state: str = dspy.InputField()
    action: str = dspy.InputField()
    risk_level: str = dspy.OutputField(desc="low/medium/high")
    risk_factors: str = dspy.OutputField(desc="identified risk factors")

risk_assessor = dspy.Predict(RiskAssessmentSig)
```

### Custom Metrics

Define domain-specific metrics:

```python
def safety_first_metric(example, prediction, trace=None):
    """Penalize unsafe actions heavily."""
    unsafe_actions = ["run_red_light", "speed", "tailgate"]

    # Check if predicted action is safe
    if prediction.action in unsafe_actions:
        return 0.0  # Zero score for unsafe actions

    # Normal scoring
    return 1.0 if example.action == prediction.action else 0.5
```

### Ensemble Models

Run multiple policies and aggregate:

```python
class EnsembleAgent(dspy.Module):
    def __init__(self, num_policies=3):
        super().__init__()
        self.policies = [dspy.ChainOfThought(PolicySig) for _ in range(num_policies)]

        # Load different trained versions
        for i, policy in enumerate(self.policies):
            policy.load(f"models/policy_v{i+1}.json")

    def forward(self, state):
        # Get all decisions
        results = [p(state=state) for p in self.policies]

        # Majority vote
        from collections import Counter
        votes = Counter([r.action for r in results])
        action = votes.most_common(1)[0][0]

        return dspy.Prediction(action=action, confidence=votes[action]/len(self.policies))
```

---

## Testing

### Unit Tests
```python
import unittest
from IMPLEMENTATION_STARTER import validate_ee_format, parse_ee_reasoning

class TestEEFormat(unittest.TestCase):
    def test_valid_format(self):
        reasoning = """
State: Test state

Alternatives:
- Action A: test → Expected Outcome: result

Analysis: comparison

Conclusion: Best action is test because reasons.
"""
        is_valid, errors = validate_ee_format(reasoning)
        self.assertTrue(is_valid)

    def test_missing_alternatives(self):
        reasoning = "State: Test\nConclusion: Done"
        is_valid, errors = validate_ee_format(reasoning)
        self.assertFalse(is_valid)
        self.assertIn("Missing 'Alternatives:' section", errors)
```

### Integration Tests
```python
def test_agent_with_world_model():
    agent = AgentWithWorldModel(
        world_model_path="models/trained/world_model_v1.0.0.json",
        policy_path="models/trained/policy_v1.0.0.json"
    )

    result = agent(state="test state")

    # Check outputs exist
    assert result.action
    assert result.reasoning
    assert result.world_model_predictions

    # Check EE format
    is_valid, _ = validate_ee_format(result.reasoning)
    assert is_valid
```

---

## Troubleshooting

### Common Issues

**Problem**: `ValueError: path must end with .json or .pkl`
**Solution**: Use `.json` or `.pkl` extension when saving

**Problem**: `AttributeError: 'ChainOfThought' object has no attribute 'signature'`
**Solution**: Access via `module.predict.signature` or use `str(module)`

**Problem**: Training metric always returns 0.0
**Solution**: Check that output field names in metric match signature fields

**Problem**: Reasoning doesn't follow EE format
**Solution**:
- Ensure training data has consistent EE format
- Use `ee_compliant_metric` instead of basic `policy_metric`
- Add more training examples with perfect EE structure

### Debugging

**Enable DSPy tracing**:
```python
dspy.configure(lm=lm, trace=True)
result = policy(state="test")
policy.inspect_history(n=1)  # See last LLM call
```

**Check saved model contents**:
```python
import json
with open("models/policy.json", "r") as f:
    data = json.load(f)
    print("Demos:", len(data['predict']['demos']))
    print("First demo:", data['predict']['demos'][0])
```

**Validate training data**:
```python
from IMPLEMENTATION_STARTER import load_training_examples

examples = load_training_examples("data/training/policy_examples.jsonl")
print(f"Loaded {len(examples)} examples")

# Check first example
ex = examples[0]
print(f"Inputs: {ex.inputs()}")
print(f"Has reasoning: {'reasoning' in ex}")
print(f"Has action: {'action' in ex}")
```

---

## Performance Optimization

### Batch Processing
```python
# Process multiple states in parallel
states = ["state1", "state2", "state3"]
results = policy.batch([{"state": s} for s in states], num_threads=4)
```

### Temperature Tuning
```python
# Lower temperature for more deterministic output
configure_lm(model="gpt-4o-mini", temperature=0.0, seed=42)

# Higher temperature for more diverse training examples
configure_lm(model="gpt-4o-mini", temperature=0.8)
```

### Training Optimization
```python
# Increase threads for faster training
optimizer = BootstrapFinetune(metric=metric, num_threads=8)

# Use fewer examples for faster iteration during development
train_subset = train_examples[:10]  # Use 10 examples for quick testing
```

---

## Contributing

### Adding Training Examples

1. Write examples in JSONL format
2. Follow EE structure for policy examples
3. Ensure state/action formats are consistent
4. Include diverse scenarios (edge cases, normal cases, rare events)

### Improving Metrics

1. Start with existing metrics as templates
2. Add domain-specific checks (safety, legality, efficiency)
3. Weight components appropriately (structure vs correctness)
4. Test metrics on validation set before full training

### Extending Signatures

1. Keep input/output fields minimal and clear
2. Use field descriptions to guide LLM behavior
3. Test with zero-shot before training
4. Document expected output format in signature docstring

---

## References

### Papers
- **DSPy**: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
- **Engineering Excellence**: See `.specify/memory/constitution.md`

### Links
- DSPy GitHub: https://github.com/stanfordnlp/dspy
- DSPy Examples: https://github.com/stanfordnlp/dspy/tree/main/examples

### Related Projects
- **JTBD Idea Validator**: Example DSPy project using GEPA optimizer (inspiration for this repo)

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Documentation**: See `research.md` for comprehensive details
- **Quick Reference**: See `QUICK_REFERENCE.md` for recipes
- **Code Examples**: See `IMPLEMENTATION_STARTER.py` for ready-to-use code
- **Constitution**: See `.specify/memory/constitution.md` for EE principles

---

**Built with DSPy 3.0.3** | **Engineering Excellence Compliant**
