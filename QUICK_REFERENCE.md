# DSPy Quick Reference - Agent Learning via EE

**TL;DR**: This is a quick-start guide for implementing DSPy modules. See `research.md` for comprehensive details.

---

## 1. Basic Signature Patterns

### World Model (State Prediction)
```python
class WorldModelSig(dspy.Signature):
    """Predict next state given state and action."""
    state: str = dspy.InputField()
    action: str = dspy.InputField()
    next_state: str = dspy.OutputField()

# Usage
world_model = dspy.Predict(WorldModelSig)
result = world_model(state="at red light", action="wait")
```

### Policy with Reasoning (EE-Style)
```python
class PolicySig(dspy.Signature):
    """EE-style decision with alternatives comparison."""
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="EE format: State, Alternatives, Analysis, Conclusion")
    action: str = dspy.OutputField()

# Usage - ChainOfThought ensures reasoning before action
policy = dspy.ChainOfThought(PolicySig)
result = policy(state="at yellow light")
print(result.reasoning)  # Full EE reflection
print(result.action)     # Chosen action
```

---

## 2. Training in 5 Steps

```python
import dspy
from dspy.teleprompt import BootstrapFinetune

# Step 1: Configure LLM
lm = dspy.OpenAI(model="gpt-4o-mini", temperature=0.2, seed=42)
dspy.configure(lm=lm)

# Step 2: Create training data
examples = [
    dspy.Example(state="...", action="...").with_inputs("state"),
    # ... more examples
]

# Step 3: Define metric
def metric(example, prediction, trace=None):
    return 1.0 if example.action == prediction.action else 0.0

# Step 4: Train
policy = dspy.ChainOfThought(PolicySig)
optimizer = BootstrapFinetune(metric=metric)
compiled = optimizer.compile(student=policy, trainset=examples)

# Step 5: Save
compiled.save("models/policy.json")
```

---

## 3. Loading and Using

```python
# Configure LLM first
dspy.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

# Load
policy = dspy.ChainOfThought(PolicySig)
policy.load("models/policy.json")

# Use
result = policy(state="approaching intersection")
```

---

## 4. Module Composition (World Model + Policy)

```python
class AgentWithWorldModel(dspy.Module):
    def __init__(self):
        super().__init__()
        self.world_model = dspy.Predict(WorldModelSig)
        self.policy = dspy.ChainOfThought(PolicySig)

    def forward(self, state: str):
        # Simulate outcomes for candidate actions
        actions = ["stop", "slow", "proceed"]
        predictions = []

        for action in actions:
            pred = self.world_model(state=state, action=action)
            predictions.append(f"{action} → {pred.next_state}")

        # Policy decides with predictions as context
        decision = self.policy(
            state=f"{state}\\n\\nPredictions:\\n" + "\\n".join(predictions)
        )

        return decision

# Usage
agent = AgentWithWorldModel()
result = agent(state="at yellow light")
```

---

## 5. File Formats

**JSON** (recommended for development):
```python
module.save("model.json")  # Human-readable, larger
module.load("model.json")
```

**Pickle** (recommended for production):
```python
module.save("model.pkl")   # Binary, smaller, faster
module.load("model.pkl")
```

---

## 6. EE Compliance Check

```python
import re

def has_ee_structure(reasoning: str) -> bool:
    """Quick check for EE format."""
    required = ["State:", "Alternatives:", "Analysis:", "Conclusion:"]
    has_structure = all(r in reasoning for r in required)

    alt_count = len(re.findall(r"Action [A-Z]:", reasoning))
    has_alternatives = alt_count >= 2

    return has_structure and has_alternatives

# Use in metric
def ee_metric(example, prediction, trace=None):
    if not has_ee_structure(prediction.reasoning):
        return 0.0

    return 1.0 if example.action == prediction.action else 0.5
```

---

## 7. Training Data Format (JSONL)

**File: data/training/policy_examples.jsonl**
```json
{"state": "at red light", "reasoning": "State: At red light...", "action": "wait"}
{"state": "at yellow light", "reasoning": "State: Approaching yellow...", "action": "brake"}
```

**Loading:**
```python
import json

def load_jsonl(path: str):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ex = dspy.Example(**data).with_inputs("state")
            examples.append(ex)
    return examples

train_data = load_jsonl("data/training/policy_examples.jsonl")
```

---

## 8. Common Patterns

### Pattern: Two-Stage Reasoning
```python
class ReasoningSig(dspy.Signature):
    state: str = dspy.InputField()
    reasoning: str = dspy.OutputField()

class ActionSig(dspy.Signature):
    state: str = dspy.InputField()
    reasoning: str = dspy.InputField()
    action: str = dspy.OutputField()

class TwoStagePolicy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(ReasoningSig)
        self.actor = dspy.Predict(ActionSig)

    def forward(self, state):
        reasoning = self.reasoner(state=state)
        action = self.actor(state=state, reasoning=reasoning.reasoning)
        return dspy.Prediction(reasoning=reasoning.reasoning, action=action.action)
```

### Pattern: Ensemble
```python
class Ensemble(dspy.Module):
    def __init__(self, num_policies=3):
        super().__init__()
        self.policies = [dspy.ChainOfThought(PolicySig) for _ in range(num_policies)]

    def forward(self, state):
        results = [p(state=state) for p in self.policies]

        # Majority vote
        from collections import Counter
        votes = Counter([r.action for r in results])
        action = votes.most_common(1)[0][0]

        # Aggregate reasoning
        reasoning = "\\n---\\n".join([r.reasoning for r in results])

        return dspy.Prediction(action=action, reasoning=reasoning)
```

---

## 9. Key Gotchas

**❌ DON'T:**
- Use `.dspy` extension (use `.json` or `.pkl`)
- Call `load()` before `dspy.configure(lm=...)`
- Mix input/output fields in `.with_inputs()`
- Skip field descriptions (they guide LLM behavior)

**✅ DO:**
- Configure LLM before loading models
- Use descriptive field names and descriptions
- Include both reasoning and action in training data
- Validate EE format in metrics
- Version your models with metadata

---

## 10. Debugging Tips

**Check what's in a saved model:**
```python
import json
with open("model.json", "r") as f:
    data = json.load(f)
    print(data.keys())  # ['predict', 'metadata']
    print(data['predict']['demos'])  # Training examples
```

**Print LLM calls during execution:**
```python
dspy.configure(lm=lm, trace=True)  # Enable tracing
result = policy(state="test")
policy.inspect_history(n=1)  # See last LLM call
```

**Test signature without training:**
```python
policy = dspy.ChainOfThought(PolicySig)
result = policy(state="at red light")
# Will use zero-shot prompting
```

---

## 11. File Locations

```
src/models/
  ├── world_model.py         # WorldModelSig definition
  ├── policy.py              # PolicySig definitions
  └── signatures.py          # All signatures

src/agents/
  ├── agent_with_world_model.py
  └── ensemble_agent.py

scripts/
  ├── train_modules.py       # Training scripts
  └── evaluate_models.py

data/training/
  ├── world_model_examples.jsonl
  └── policy_examples.jsonl

models/trained/
  ├── world_model_v1.0.0.json
  └── policy_v1.0.0.json
```

---

**For detailed explanations, see `research.md`**
