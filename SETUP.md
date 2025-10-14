# Setup Guide - Agent Learning EE

This guide helps you set up and run the Agent Learning via Early Experience system with ACE integration.

## Prerequisites

### 1. Python Environment
```bash
# Requires Python 3.11+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys

You need an LLM API key for training and inference. Supported providers:

**Option A: OpenRouter** (Recommended - supports multiple models)
```bash
export OPENAI_API_KEY='your-openrouter-api-key-here'
export OPENAI_API_BASE='https://openrouter.ai/api/v1'
```

**Option B: OpenAI**
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
# No need to set OPENAI_API_BASE
```

⚠️ **NEVER commit API keys to git!** Use environment variables only.

### 3. ACE Playbook (Optional but Recommended)

The live exploration loop works with or without ACE, but semantic deduplication requires it.

**Option A: Full ACE Integration** (Recommended)
```bash
# Clone ACE repository
git clone https://github.com/YourOrg/ace-playbook.git ../ace-playbook

# Initialize database
cd ../ace-playbook
python -c "from ace.utils.database import init_database; init_database()"

# Add to PYTHONPATH when running demos
export PYTHONPATH=/path/to/ace-playbook:$PYTHONPATH
```

**Option B: In-Memory Stub** (For quick testing)
```bash
# No setup needed - will use in-memory implementation
# Set ACE_ENABLED=0 to disable playbook features entirely
export ACE_ENABLED=0
```

## Quick Start

### 1. Generate Training Data
```bash
# Generate synthetic expert demonstrations
python -c "
from tests.fixtures.generate_demos import generate_synthetic_demos
from agent_learning.utils import save_jsonl

demos = generate_synthetic_demos(num_demos=50, seed=42)
save_jsonl(demos, 'data/expert_demos.jsonl')
print(f'✓ Generated {len(demos)} demonstrations')
"
```

### 2. Train Pipeline
```bash
# Ensure API key is set
echo $OPENAI_API_KEY  # Should print your key

# Run complete 4-stage training pipeline
python -m agent_learning.pipeline \
    --expert-demos data/expert_demos.jsonl \
    --output-dir artifacts/
```

This runs:
- **Stage 1**: World model training (predicts next states)
- **Stage 2**: Exploration rollouts (generates alternatives)
- **Stage 3**: Reflection generation (structured reasoning)
- **Stage 4**: Policy training (learns decision-making with reasoning)

Expected duration: ~15-20 minutes
Expected output: `artifacts/policy.pkl`

### 3. Run Live Loop Demo
```bash
# With ACE (full semantic deduplication)
PYTHONPATH=/path/to/ace-playbook:$PYTHONPATH python examples/live_loop_demo.py

# Without ACE (in-memory stub)
ACE_ENABLED=0 python examples/live_loop_demo.py
```

The demo:
- Runs 50 episodes of driving simulation
- Generates reflections every 10 episodes
- Updates ACE playbook with semantic dedup
- Shows final metrics and playbook content

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | API key for LLM provider |
| `OPENAI_API_BASE` | No | `https://api.openai.com/v1` | API endpoint URL |
| `ACE_ENABLED` | No | `1` | Enable ACE integration (1/0) |
| `ACE_DOMAIN_ID` | No | `default` | ACE playbook domain identifier |
| `ACE_TARGET_STAGE` | No | `shadow` | ACE deployment stage (shadow/staging/prod) |

### ACE Configuration

Edit `src/ee_ace_bridge/config.py` for advanced settings:
```python
ACE_ENABLED = os.getenv("ACE_ENABLED", "1") == "1"
ACE_SECTIONS = ["Helpful"]  # Playbook sections to use
ACE_TOKEN_BUDGET = 1000  # Max tokens for playbook context
SIMILARITY_THRESHOLD = 0.80  # FAISS deduplication threshold
```

## Troubleshooting

### "No LM is loaded"
**Problem**: DSPy not configured with API key
**Solution**: Ensure `OPENAI_API_KEY` environment variable is set before running

### "no such table: playbook_bullets"
**Problem**: ACE database not initialized
**Solution**:
```bash
cd /path/to/ace-playbook
python -c "from ace.utils.database import init_database; init_database()"
```

### "'dict' object is not callable"
**Problem**: Old policy.pkl format
**Solution**: Delete `artifacts/policy.pkl` and re-run training pipeline

### ImportError: ace_playbook
**Problem**: ACE repository not in PYTHONPATH
**Solution**: Add `PYTHONPATH=/path/to/ace-playbook:$PYTHONPATH` when running

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Test with coverage
pytest --cov=agent_learning --cov-report=html
```

## Project Structure

```
AgentLearningEE/
├── src/agent_learning/     # Core pipeline modules
│   ├── world_model.py      # Stage 1: State prediction
│   ├── exploration.py      # Stage 2: Rollout generation
│   ├── reflection.py       # Stage 3: Reasoning generation
│   ├── policy.py           # Stage 4: Policy training
│   ├── live_loop.py        # Continuous learning loop
│   ├── pipeline.py         # Orchestrates all stages
│   └── utils.py            # Shared utilities
├── src/ee_ace_bridge/      # ACE integration bridge
│   ├── ace_client.py       # InProcessAceClient + stub
│   ├── schema_mapping.py   # EE → ACE translation
│   └── config.py           # ACE configuration
├── examples/               # Demo scripts
│   └── live_loop_demo.py   # Driving simulator demo
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── data/                  # Training data
│   └── expert_demos.jsonl
└── artifacts/             # Trained models
    ├── policy.pkl
    ├── world_model.pkl
    └── reflection_data.jsonl
```

## Next Steps

1. **Customize Environment**: Implement your own `Environment` class (see `examples/live_loop_demo.py` for DrivingSimulator example)

2. **Adjust Config**: Modify `LiveLoopConfig` parameters:
   ```python
   config = LiveLoopConfig(
       episode_batch_size=10,      # Episodes per batch
       max_episodes=100,            # Total episodes to run
       reflection_interval=10,      # Reflect every N episodes
       ace_enabled=True,            # Use ACE playbook
       ace_update_interval=10,      # Update ACE every N reflections
   )
   ```

3. **Monitor Metrics**: Track loop performance:
   - Total episodes collected
   - Total reflections generated
   - Total ACE playbook updates
   - Episodes per minute (throughput)

4. **Promote Insights**: Move insights through deployment stages:
   - Shadow → Staging → Production
   - See ACE documentation for promotion workflows

## Support

- Documentation: `README.md`, `README_LIVE_LOOP.md`
- Issues: File bug reports on GitHub
- Contracts: See `contracts/` for detailed specifications

## License

[Your License Here]
