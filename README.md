# Agent Learning via Early Experience (EE)

**Reward-Free Agent Learning through Expert Demonstration and Self-Reflection**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DSPy 2.0](https://img.shields.io/badge/dspy-2.0-green.svg)](https://github.com/stanfordnlp/dspy)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](#releases)

This repository implements a complete agent learning system that learns from expert demonstrations without reward signals. The agent develops structured reasoning capabilities through a 4-stage pipeline that generates self-reflective decision-making with explicit alternatives analysis.

---

## 🎯 Overview

**Agent Learning EE** demonstrates reward-free learning where an agent:
1. Learns state transitions from expert demonstrations (World Model)
2. Generates exploratory rollouts with alternative actions (Exploration)
3. Creates structured reasoning comparing expert vs alternatives (Reflection)
4. Trains a policy with self-reflective decision-making (Policy)

**Key Innovation**: Instead of reward-based RL, the agent learns by analyzing *why* expert actions are better than alternatives through explicit comparative reasoning.

### Key Features

- 🧠 **Implicit World Model**: Learns state transitions from (state, action, next_state) demonstrations
- 🔄 **Exploratory Data Generation**: Generates 2.5x more training data through alternative action rollouts
- 💭 **Structured Reflection**: Creates 4-section EE-style reasoning (Situation → Expert → Alternatives → Conclusion)
- 🎯 **Self-Reflective Policy**: Makes decisions by explicitly comparing alternatives
- 📊 **Comprehensive Testing**: Parameterized tests across dataset scales (10, 50, 75+ demos)
- ✅ **Quality Validation**: Ensures expansion ratio, alternative coverage, and reasoning structure
- 🔄 **ACE Integration (Optional)**: Adaptive Code Evolution playbook for continuous self-improvement

### ACE Integration (Adaptive Code Evolution)

**Optional** integration with [ACE playbook system](https://github.com/yourusername/ace-playbook) for continuous learning:

- **Self-Reflection → Playbook**: Converts EE reflections into ACE insights with semantic deduplication
- **Playbook → Policy Context**: Injects accumulated knowledge at inference time
- **FAISS Semantic Search**: 0.80 similarity threshold prevents duplicate insights
- **Multi-Stage Deployment**: Shadow → Staging → Prod promotion gates
- **Append-Only Knowledge**: No forgetting, no context collapse

The system works standalone without ACE. When enabled, it creates a continuous learning loop where agent reflections improve future decisions.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jmanhype/AgentLearningEE.git
cd AgentLearningEE

# Install dependencies
pip install -e .

# For development with tests
pip install -e ".[dev]"

# Optional: Install ACE dependencies for playbook integration
pip install sentence-transformers faiss-cpu sqlalchemy alembic structlog
```

### Optional: ACE Integration Setup

To enable continuous learning through ACE playbook:

```bash
# 1. Clone ACE repository (or use your ACE installation)
git clone https://github.com/yourusername/ace-playbook.git /path/to/ace-playbook

# 2. Configure environment
source .env.ace

# Or manually set:
export ACE_ENABLED=1
export ACE_DOMAIN_ID="agent-learning"
export ACE_TARGET_STAGE="shadow"
export ACE_SIMILARITY_THRESHOLD="0.80"
export ACE_TOKEN_BUDGET="3500"
export DATABASE_URL="sqlite:///ace_playbook.db"
export PYTHONPATH="/path/to/ace-playbook:$PYTHONPATH"

# 3. Initialize ACE database
python -c "
from sqlalchemy import create_engine
from ace.models.base import Base
engine = create_engine('sqlite:///ace_playbook.db')
Base.metadata.create_all(engine)
"

# 4. Test integration
python examples/ace_integration_demo.py
```

**Result**: Training pipeline will now seed ACE playbook with reflections, and policy will inject playbook context at inference.

### Prerequisites

```bash
# Python 3.11+
python --version

# Set OpenRouter API key (we use OpenRouter for access to multiple models)
export OPENAI_API_KEY="sk-or-v1-your-key-here"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
```

### Running the Complete Pipeline

```python
from agent_learning.pipeline import run_complete_pipeline
from agent_learning.utils import setup_logger

# Configure logger
logger = setup_logger("training")

# Run complete 4-stage pipeline
result = run_complete_pipeline(
    expert_demos_path="data/expert_demos.jsonl",
    output_dir="artifacts/",
    logger=logger,
)

# Check results
if result["success"]:
    print(f"Pipeline completed: {result['stage_completed']}")
    print(f"World Model Accuracy: {result['metrics']['world_model']['accuracy']:.2%}")
    print(f"Expansion Ratio: {result['metrics']['exploration']['expansion_ratio']:.2f}x")
    print(f"Policy Accuracy: {result['metrics']['policy']['accuracy']:.2%}")
else:
    print(f"Pipeline failed: {result['error']}")
```

### Using Trained Models

```python
from agent_learning.policy import load_trained_policy, generate_decision

# Load trained policy
policy = load_trained_policy("artifacts/policy.pkl")

# Generate decision with reasoning
reasoning, action = generate_decision(
    policy,
    state="Vehicle approaching intersection with red light"
)

print(f"Action: {action}")
print(f"\nReasoning:\n{reasoning}")
```

---

## 📁 Repository Structure

```
agent-learning-ee/
├── README.md                           # This file
├── pyproject.toml                      # Package configuration
├── .gitignore                          # Git ignore rules
├── .env.ace                            # ACE configuration template
│
├── src/agent_learning/                 # Core implementation
│   ├── __init__.py
│   ├── world_model.py                  # State transition prediction
│   ├── exploration.py                  # Alternative action generation
│   ├── reflection.py                   # Structured reasoning generation
│   ├── policy.py                       # Self-reflective decision-making (ACE-aware)
│   ├── pipeline.py                     # End-to-end orchestration
│   └── utils.py                        # JSONL, logging, metrics, serialization
│
├── src/ee_ace_bridge/                  # ACE integration (optional)
│   ├── __init__.py                     # Feature flags, exports
│   ├── translate.py                    # EE → ACE schema translation
│   ├── config_extra.py                 # Extended ACE configuration
│   ├── ace_client.py                   # InProcessAceClient, InMemoryAceClient
│   └── contracts.py                    # Type definitions, protocols
│
├── tests/                              # Test suite
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── deterministic_seeds.py      # Base expert demonstrations
│   │   └── generate_demos.py           # Synthetic demo generation
│   │
│   ├── unit/                           # Unit tests for each module
│   │   ├── test_world_model.py
│   │   ├── test_exploration.py
│   │   ├── test_reflection.py
│   │   ├── test_policy.py
│   │   └── test_pipeline_module.py
│   │
│   ├── integration/
│   │   └── test_pipeline.py            # End-to-end integration tests
│   │
│   └── ee_ace_bridge/                  # ACE integration tests
│       ├── conftest.py                 # Test configuration
│       └── test_ace_integration.py     # Wire compatibility tests
│
├── examples/                           # Example scripts
│   └── ace_integration_demo.py         # Full ACE integration demo
│
├── data/                               # Training data (generated)
│   ├── expert_demos.jsonl
│   ├── exploratory_rollouts.jsonl
│   └── reflection_data.jsonl
│
├── artifacts/                          # Trained models (generated)
│   ├── world_model.pkl
│   ├── policy.pkl
│   └── *.meta.json                     # Model metadata
│
├── contracts/                          # Technical contracts
│   └── success_criteria.yaml           # Module success criteria
│
└── docs/                               # Additional documentation
    ├── research.md                     # Comprehensive DSPy research
    └── QUICK_REFERENCE.md              # Quick-start patterns
```

---

## 🏗️ System Architecture

### 4-Stage Learning Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Expert Demos   │ --> │  World Model     │ --> │  Exploration    │ --> │  Reflection  │
│  (state, action,│     │  (IWM Training)  │     │  (Rollouts)     │     │  (Reasoning) │
│   next_state)   │     │                  │     │                 │     │              │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                               │                         │                       │
                               │                         │                       │
                               v                         v                       v
                        Trained Model           2.5x Expanded Data      Structured Reasoning
                        artifacts/              exploratory_            reflection_
                        world_model.pkl         rollouts.jsonl          data.jsonl
                                                                              │
                                                                              │
                                                                              v
                                                                    ┌─────────────────┐
                                                                    │  Policy         │
                                                                    │  (Self-Reflect) │
                                                                    │                 │
                                                                    └─────────────────┘
                                                                              │
                                                                              v
                                                                      Trained Policy
                                                                      artifacts/
                                                                      policy.pkl
                                                                              │
                                                                              │
                                        ┌─────────────────────────────────────┘
                                        │
                                        v
                            ┌──────────────────────────┐
                            │  ACE Playbook (Optional) │ <──┐
                            │  - Semantic Dedup (FAISS)│    │
                            │  - Multi-stage (Shadow)  │    │
                            │  - Append-only Storage   │    │
                            └──────────────────────────┘    │
                                        │                   │
                                        │ Playbook Context  │ Reflections
                                        v                   │
                            ┌──────────────────────────┐    │
                            │  Policy Inference        │ ───┘
                            │  (With Playbook Context) │
                            └──────────────────────────┘
                                        │
                                        v
                                  Better Decisions
```

### Stage Details

**1. World Model (Implicit World Model - IWM)**
- **Input**: Expert demonstrations (state, action, next_state)
- **Training**: DSPy BootstrapFewShot with state transition prediction
- **Output**: Trained model predicting next_state given (state, action)
- **Success Criteria**: >70% accuracy on held-out transitions

**2. Exploration**
- **Input**: Expert demos + trained world model
- **Process**: For each demo, generate 1-3 alternative actions
- **Rollouts**: Use world model to predict outcomes for alternatives
- **Output**: Exploratory rollouts (state, action, next_state, expert_action)
- **Success Criteria**: Expansion ratio ≥ 2.0x, alternative coverage ≥ 50%

**3. Reflection**
- **Input**: Exploratory rollouts with expert and alternative actions
- **Process**: Generate structured 4-section reasoning comparing actions
- **Output**: Reflection data (state, reasoning, action)
- **Success Criteria**: Valid 4-section structure (Situation → Expert → Alternatives → Conclusion)

**4. Policy**
- **Input**: Reflection data with structured reasoning
- **Training**: DSPy BootstrapFewShot with ChainOfThought
- **Output**: Trained policy making decisions with self-reflection
- **Success Criteria**: >70% accuracy, reasoning quality >75%

---

## 🧪 Testing

### Environment Setup for Tests

Integration tests require an OpenRouter API key:

```bash
# Set API key for testing
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Alternative: Create .env file in project root
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key-here' > .env
```

⚠️ **Security Note**: Never commit API keys to version control. The test suite reads keys from environment variables only.

### Run All Tests

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src/agent_learning

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only (includes slow tests)
pytest tests/integration/

# Specific module tests
pytest tests/unit/test_world_model.py -v
pytest tests/unit/test_policy.py -v
```

### Parameterized Integration Tests

The integration test suite includes parameterized tests across dataset sizes:

```bash
# Run parameterized tests (10, 50, 75 demo scales)
pytest tests/integration/test_pipeline.py::TestCompletePipelineIntegration::test_pipeline_with_variable_dataset_sizes -v
```

**Test Scales**:
- **10 demos** (smoke): Quick validation that pipeline runs
- **50 demos** (integration): Integration test with basic validity
- **75 demos** (validation): Thorough validation with good statistical confidence

**Validation Results (v0.2.0)**:
- ✅ 10-demo test: PASSED (expansion ≥ 2.0x, coverage ≥ 50%, reasoning quality > 0%)
- ✅ 50-demo test: PASSED (all quality thresholds met)

---

## 📊 Training Data Format

All data uses **JSONL** (JSON Lines) format - one JSON object per line.

### Expert Demonstrations (Input)

```jsonl
{"state": "Vehicle approaching intersection with red light", "action": "stop", "next_state": "Vehicle stopped at intersection; light still red"}
{"state": "Pedestrian crossing at crosswalk", "action": "yield", "next_state": "Vehicle yields; pedestrian crosses safely"}
{"state": "Vehicle in left lane with obstacle ahead", "action": "change lane", "next_state": "Vehicle changes lane; obstacle avoided"}
```

### Exploratory Rollouts (Generated by Exploration Stage)

```jsonl
{"state": "Vehicle approaching intersection with red light", "action": "proceed with caution", "next_state": "Vehicle proceeds through red light; potential violation", "expert_action": "stop"}
{"state": "Vehicle approaching intersection with red light", "action": "stop", "next_state": "Vehicle stopped at intersection; light still red", "expert_action": "stop"}
```

### Reflection Data (Generated by Reflection Stage)

```jsonl
{"state": "Vehicle approaching intersection with red light", "reasoning": "Section 1 - Situation Analysis:\n...\n\nSection 2 - Expert Action Evaluation:\n...\n\nSection 3 - Alternative Actions Analysis:\n...\n\nSection 4 - Conclusion:\n...", "action": "stop"}
```

---

## 🎓 Key Concepts

### Implicit World Model (IWM)

Unlike explicit world models that learn reward functions, IWM learns only state transitions:

```python
from agent_learning.world_model import WorldModelModule, train_world_model

# Train world model from demonstrations
world_model, metrics = train_world_model(
    expert_demos_path="data/expert_demos.jsonl",
    output_path="artifacts/world_model.pkl",
    test_split=0.2,
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Exploratory Data Generation

Expansion through alternative action exploration:

```python
from agent_learning.exploration import generate_exploratory_rollouts

# Generate 2.5x more data through alternatives
rollouts = generate_exploratory_rollouts(
    expert_demos_path="data/expert_demos.jsonl",
    world_model_path="artifacts/world_model.pkl",
    output_path="data/exploratory_rollouts.jsonl",
)

print(f"Expansion: {rollouts['expansion_ratio']:.2f}x")
print(f"Alternative Coverage: {rollouts['alternative_coverage']:.2%}")
```

### Structured Reflection (EE-Style)

4-section reasoning comparing expert vs alternatives:

```python
from agent_learning.reflection import generate_reflection_data

# Generate structured reasoning
reflection = generate_reflection_data(
    rollouts_path="data/exploratory_rollouts.jsonl",
    output_path="data/reflection_data.jsonl",
)

# Example reasoning structure:
"""
Section 1 - Situation Analysis:
Vehicle approaching intersection at red light. Traffic signal indicates stop required.
Key factors: Legal compliance, safety, visibility.

Section 2 - Expert Action Evaluation:
Expert chose: stop
Rationale: Ensures legal compliance and safety at controlled intersection.
Strengths: Prevents violations, avoids collision risk.

Section 3 - Alternative Actions Analysis:
Alternative 1: proceed with caution
  Benefits: Saves time if intersection clear
  Drawbacks: Traffic violation, collision risk, legal penalty
  Differs: Prioritizes time over safety/legality

Alternative 2: accelerate
  Benefits: Clears intersection quickly
  Drawbacks: High collision risk, severe violation
  Differs: Disregards safety and law entirely

Section 4 - Conclusion:
Best action: stop
Justification: Safety and legal compliance paramount at controlled intersections.
Confidence: High (expert demonstrates correct protocol)
"""
```

### Self-Reflective Policy

Policy trained on structured reasoning learns to explicitly compare alternatives:

```python
from agent_learning.policy import train_policy, generate_decision

# Train policy from reflection data
policy, metrics = train_policy(
    reflection_data_path="data/reflection_data.jsonl",
    output_path="artifacts/policy.pkl",
)

# Generate decision with explicit reasoning
reasoning, action = generate_decision(
    policy,
    state="Vehicle speed 45mph in 35mph residential zone"
)

# Policy reasons about why to decelerate vs maintain vs accelerate
print(f"Decision: {action}")
print(f"Reasoning shows alternatives analysis: {reasoning}")
```

### ACE Integration (Optional)

Enable continuous learning through playbook accumulation:

```python
from agent_learning.policy import train_policy

# ACE automatically activates if ACE_ENABLED=1 in environment
policy, metrics = train_policy(
    reflection_data_path="data/reflection_data.jsonl",
    output_path="artifacts/policy.pkl",
)

# Training automatically seeds ACE playbook with reflections
# Inference automatically injects playbook context before decisions
```

**ACE Flow**:
1. **Training**: Reflections → ACE insights → Playbook database (shadow stage)
2. **Promotion**: Shadow → Staging → Prod (manual or automated gates)
3. **Inference**: Load playbook context → Inject into policy prompt → Better decisions

**Key Features**:
- **Semantic Deduplication**: FAISS cosine similarity (0.80 threshold) prevents duplicates
- **Counter Tracking**: Helpful/harmful votes track insight reliability
- **Stage Isolation**: New insights start in "shadow", promoting validates safety
- **Append-Only**: No forgetting, all historical insights preserved
- **Domain Isolation**: Multi-tenant support via domain_id

**Demo**:
```bash
# Test full ACE integration
python examples/ace_integration_demo.py

# Expected output:
# ✓ Schema translation working
# ✓ Deduplication working (3 duplicates detected)
# ✓ Playbook rendering: 5 insights
# ✓ Health check: healthy
```

**Monitoring**:
```python
from ee_ace_bridge.ace_client import InProcessAceClient

client = InProcessAceClient(domain_id="agent-learning")

# Get health status
health = client.get_health()
print(f"Status: {health['status']}")
print(f"Insights Ingested: {health['insights_ingested']}")
print(f"Stage Counts: {health['stage_counts']}")

# Get insight counts
section_count = client.get_section_count()
insight_count = client.get_insight_count()
print(f"Sections: {section_count}")
print(f"Total Insights: {insight_count}")

# Render current playbook
playbook = client.render_playbook(token_budget=3500)
print(playbook)
```

---

## 🔧 API Reference

### Pipeline Module

```python
from agent_learning.pipeline import run_complete_pipeline

result = run_complete_pipeline(
    expert_demos_path: str,           # Path to expert_demos.jsonl
    output_dir: str = "artifacts/",   # Output directory for trained models
    logger: Optional[Logger] = None,  # Optional logger instance
) -> Dict[str, Any]

# Returns:
{
    "success": bool,                  # Pipeline success status
    "stage_completed": str,           # Last completed stage
    "artifacts": {                    # Paths to generated artifacts
        "world_model": str,
        "exploratory_rollouts": str,
        "reflection_data": str,
        "policy": str,
    },
    "metrics": {                      # Performance metrics
        "world_model": {...},
        "exploration": {...},
        "policy": {...},
    },
    "error": Optional[str],           # Error message if failed
}
```

### World Model Module

```python
from agent_learning.world_model import (
    train_world_model,
    predict_next_state,
    load_trained_world_model,
)

# Training
world_model, metrics = train_world_model(
    expert_demos_path: str,
    output_path: str = "artifacts/world_model.pkl",
    test_split: float = 0.2,
    random_seed: int = 42,
    max_bootstrapped_demos: int = 8,
    max_labeled_demos: int = 16,
    metric_threshold: Optional[float] = 0.70,
)

# Inference
next_state = predict_next_state(
    world_model=world_model,
    state="current state",
    action="action to take",
)
```

### Exploration Module

```python
from agent_learning.exploration import generate_exploratory_rollouts

metrics = generate_exploratory_rollouts(
    expert_demos_path: str,
    world_model_path: str,
    output_path: str = "data/exploratory_rollouts.jsonl",
    alternatives_per_demo: int = 2,
    expansion_target: float = 2.5,
    alternative_coverage_target: float = 0.5,
)

# Returns:
{
    "expansion_ratio": float,         # Achieved expansion ratio
    "alternative_coverage": float,    # % demos with alternatives
    "total_rollouts": int,
    "alternative_rollouts": int,
}
```

### Reflection Module

```python
from agent_learning.reflection import generate_reflection_data

metrics = generate_reflection_data(
    rollouts_path: str,
    output_path: str = "data/reflection_data.jsonl",
    reasoning_quality_target: float = 0.75,
)

# Returns:
{
    "reasoning_quality": float,       # Avg reasoning structure score
    "total_reflections": int,
}
```

### Policy Module

```python
from agent_learning.policy import (
    train_policy,
    generate_decision,
    load_trained_policy,
)

# Training
policy, metrics = train_policy(
    reflection_data_path: str,
    output_path: str = "artifacts/policy.pkl",
    test_split: float = 0.2,
    random_seed: int = 42,
    max_bootstrapped_demos: int = 8,
    max_labeled_demos: int = 16,
    metric_threshold: Optional[float] = 0.70,
)

# Inference
reasoning, action = generate_decision(
    policy=policy,
    state="current state description",
)
```

---

## 🎯 Performance Metrics

### Success Criteria (from contracts/success_criteria.yaml)

**World Model (SC-001)**:
- Accuracy > 70% on held-out state transitions
- Validated on 20% test split

**Exploration (SC-002, SC-003)**:
- Expansion ratio ≥ 2.0x (generate at least 2x more data)
- Alternative coverage ≥ 50% (alternatives for at least half of demos)

**Reflection (SC-005)**:
- Reasoning structure: All 4 sections present (Situation, Expert, Alternatives, Conclusion)
- Reasoning quality ≥ 75%

**Policy (SC-004)**:
- Accuracy > 70% on held-out decisions
- Reasoning quality > 75% (4-section structure)

---

## 🚧 Extending the System

### Custom Alternative Actions

Provide domain-specific alternative actions:

```python
from agent_learning.exploration import generate_exploratory_rollouts

# Custom alternatives function
def custom_alternatives(state: str, expert_action: str) -> List[str]:
    """Generate domain-specific alternatives."""
    # Your custom logic here
    return ["alternative1", "alternative2"]

# Use custom alternatives
generate_exploratory_rollouts(
    expert_demos_path="data/expert_demos.jsonl",
    world_model_path="artifacts/world_model.pkl",
    output_path="data/exploratory_rollouts.jsonl",
    alternatives_generator=custom_alternatives,
)
```

### Custom Metrics

Add domain-specific validation:

```python
from agent_learning.world_model import train_world_model

def custom_world_model_metric(example, prediction, trace=None):
    """Custom metric with domain constraints."""
    predicted = prediction.next_state.lower()
    expected = example.next_state.lower()

    # Exact match
    if predicted == expected:
        return 1.0

    # Custom partial credit logic
    if "safe" in predicted and "safe" in expected:
        return 0.7

    return 0.0

# Use custom metric
train_world_model(
    expert_demos_path="data/expert_demos.jsonl",
    custom_metric=custom_world_model_metric,
)
```

---

## 🐛 Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError: expert_demos.jsonl not found`
**Solution**: Create training data or use synthetic demo generator:
```python
from tests.fixtures.generate_demos import generate_synthetic_demos
from agent_learning.utils import save_jsonl

demos = generate_synthetic_demos(num_demos=50, seed=42)
save_jsonl(demos, "data/expert_demos.jsonl")
```

**Problem**: `ValueError: Insufficient expert demonstrations: need at least 10`
**Solution**: Generate more demos or reduce test_split ratio

**Problem**: Low accuracy (<70%) on world model or policy
**Solution**:
- Increase training examples (aim for 50+ demos)
- Check data quality and consistency
- Ensure expert demonstrations are high quality
- Adjust `max_bootstrapped_demos` and `max_labeled_demos`

**Problem**: Tests timeout on large datasets
**Solution**: Use smaller dataset scales or increase pytest timeout:
```python
@pytest.mark.timeout(7200)  # 2 hours for large datasets
```

### Debugging

**Enable verbose logging**:
```python
from agent_learning.utils import setup_logger
import logging

logger = setup_logger("debug", level=logging.DEBUG)
```

**Check generated data**:
```python
from agent_learning.utils import load_jsonl

data = load_jsonl("data/exploratory_rollouts.jsonl")
print(f"Loaded {len(data)} rollouts")
print(f"First rollout: {data[0]}")
```

**Inspect trained model metadata**:
```python
from agent_learning.utils import load_metadata

metadata = load_metadata("artifacts/policy.pkl")
print(f"Training accuracy: {metadata['accuracy']:.2%}")
print(f"Training date: {metadata['timestamp']}")
```

---

## 📚 Additional Resources

- **research.md**: Comprehensive DSPy implementation research (60+ pages)
- **QUICK_REFERENCE.md**: Quick-start patterns and recipes
- **contracts/**: Technical contracts defining module interfaces and success criteria
- **tests/**: Comprehensive test suite with examples

---

## 📝 Releases

### v0.3.0 (Current)
- ✅ **ACE Bridge Integration**: Optional continuous learning via ACE playbook
- ✅ **InProcessAceClient**: Direct integration with ACE CuratorService
- ✅ **FAISS Semantic Deduplication**: 0.80 similarity threshold for insight management
- ✅ **Schema Translation**: EE reflections → ACE insights (Helpful/Harmful/Neutral)
- ✅ **Multi-Stage Deployment**: Shadow → Staging → Prod promotion gates
- ✅ **Integration Tests**: Full wire compatibility validation
- ✅ **Demo Script**: Comprehensive ACE integration demonstration

### v0.2.0
- ✅ Complete 4-stage pipeline implementation
- ✅ Comprehensive unit and integration tests
- ✅ Parameterized testing infrastructure
- ✅ Synthetic demo generation
- ✅ All quality thresholds validated

### v0.1.0
- Initial release with world model
- Basic training infrastructure

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional alternative action generators
- Domain-specific metrics
- Enhanced reasoning structure validation
- Performance optimizations
- Additional test coverage

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **DSPy Framework**: Stanford NLP Group
- **Engineering Excellence Principles**: Structured reasoning methodology
- Built with Claude Code

---

**Built with DSPy 2.0** | **Python 3.11+** | **Reward-Free Learning**
