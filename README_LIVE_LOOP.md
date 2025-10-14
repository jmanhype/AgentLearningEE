# Live Exploration Loop - Continuous Learning Architecture

## Overview

Designed and prototyped a continuous learning system that wires together:
1. **Policy execution** with ACE playbook context
2. **Experience collection** from environment interactions  
3. **Online reflection** generation on experiences
4. **ACE playbook updates** with semantic deduplication
5. **Continuous improvement** through feedback loop

## Architecture

```
┌──────────────┐
│ Environment  │
│  (reset/step)│
└──────┬───────┘
       │ state
       ↓
┌──────────────────────────┐
│ Policy + Playbook Context│  ← Playbook loaded before each decision
│  (generate_decision)     │
└──────┬───────────────────┘
       │ action
       ↓
┌─────────────────────┐
│  Execute in Env     │
│  (state, action,    │
│   next_state)       │
└──────┬──────────────┘
       │ episode
       ↓
┌──────────────────────┐
│  Episode Buffer      │  ← Collect experiences
│  (deque, batched)    │
└──────┬───────────────┘
       │ batch
       ↓
┌──────────────────────┐
│  Reflection          │  ← Generate structured reasoning
│  (ChainOfThought)    │
└──────┬───────────────┘
       │ reflections
       ↓
┌──────────────────────┐
│  ACE Insights        │  ← Schema translation (EE → ACE)
│  (bridge_to_ace)     │
└──────┬───────────────┘
       │ insights
       ↓
┌──────────────────────────┐
│  CuratorService          │  ← Semantic dedup + persist
│  (FAISS, SQLite)         │
└──────┬───────────────────┘
       │ updated playbook
       └────────────────────────┐
                                ↓
                      ┌─────────────────────┐
                      │  Next Decision      │
                      │  (improved context) │
                      └─────────────────────┘
```

## Key Components

### LiveExplorationLoop
- **Episode collection**: Batched experience gathering from environment
- **Reflection triggers**: Configurable intervals (every N episodes)
- **ACE updates**: Automatic playbook ingestion with deduplication
- **Health monitoring**: Real-time metrics and status checks
- **Graceful shutdown**: Clean termination with final reflection pass

### Configuration (LiveLoopConfig)
```python
episode_batch_size: int = 10      # Buffer size
max_episodes: Optional[int] = None  # Run limit (None = infinite)
reflection_interval: int = 10      # Reflect every N episodes
ace_enabled: bool = True           # Enable ACE integration
ace_update_interval: int = 10      # Update ACE every N reflections
save_episodes: bool = True         # Persist experiences
save_reflections: bool = True      # Persist reasoning
```

### Metrics Tracked
- Total episodes collected
- Total reflections generated
- Total ACE playbook updates
- Episodes per minute (throughput)
- Runtime duration

## Integration Points

### 1. Environment Protocol
```python
class Environment(Protocol):
    def reset(self) -> str:
        """Return initial state"""
    
    def step(self, action: str) -> tuple[str, bool]:
        """Execute action, return (next_state, done)"""
```

### 2. Policy Integration
- Loads trained policy via `load_trained_policy()`
- Augments state with playbook context (if ACE enabled)
- Generates decisions via `generate_decision(policy, state, playbook)`

### 3. Reflection Generation
- Uses DSPy ChainOfThought with ReflectionSig
- Generates structured 4-section reasoning
- Converts episodes to (state, expert_action, alternatives) format

### 4. ACE Integration
- Translates reflections to ACE insights via `reflection_to_insight()`
- Ingests batches via `ace_client.ingest_insights_batch()`
- FAISS semantic deduplication (0.80 threshold)
- Shadow → Staging → Prod promotion gates

## Demo Script

See `examples/live_loop_demo.py` for complete working example with:
- DrivingSimulator environment (8 realistic scenarios)
- Configurable loop parameters
- ACE status monitoring
- Playbook rendering
- Artifact persistence

## Benefits

1. **Continuous Learning**: Agent improves from own experiences
2. **No Manual Labeling**: Self-supervised through reflection
3. **Semantic Deduplication**: FAISS prevents redundant insights
4. **Graceful Degradation**: Works with/without ACE
5. **Observable**: Rich metrics and health checks
6. **Resumable**: Persists episodes and reflections

## Next Steps

1. Implement live_loop.py core orchestrator
2. Add episode replay for offline learning
3. Add A/B testing between policy versions
4. Add automatic promotion gates (shadow → prod)
5. Add distributed execution for high-throughput environments

## Research Foundation

This implements the **Generator-Reflector-Curator loop** from Early Experience research:
- **Generator**: Policy generates actions in environment
- **Reflector**: Structured reasoning on experiences
- **Curator**: ACE CuratorService with semantic deduplication

The continuous loop enables **online adaptation** where playbook knowledge compounds over time, improving decision quality without model retraining.
