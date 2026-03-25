# AgentLearningEE

Implements the Early Experience (EE) agent training pipeline combined with Stanford's Agentic Context Engineering (ACE). An agent bootstraps from expert demonstrations, explores via self-generated rollouts, reflects on outcomes, and enforces hard guarantees through deterministic guardrails. Lessons feed into an evolving ACE playbook.

## What it does

1. Load expert demonstrations (JSONL)
2. Train a world model from those demos
3. Explore: generate new rollouts using the world model
4. Reflect: critique rollouts and extract insights
5. Update policy based on reflections
6. (Optional) Run a live loop with domain-specific guardrails and ACE integration

No reward signals are used. The agent learns from demonstrations and self-critique.

## Pipeline stages

| Stage | Module | Purpose |
|-------|--------|---------|
| World model | `agent_learning.world_model` | Learn environment dynamics from demos |
| Exploration | `agent_learning.exploration` | Generate new rollouts |
| Reflection | `agent_learning.reflection` | Critique and extract insights |
| Policy | `agent_learning.policy` | Update decision-making |
| Live loop | `agent_learning.live_loop` | Online execution with guardrails |

## Included domains

| Domain | Guardrail | Data |
|--------|-----------|------|
| SWE-bench | Patch applies + tests pass | 23 samples from princeton-nlp/SWE-bench_Lite |
| MagicBrush | MSE <= 1500, SSIM >= 0.60 | 50 samples from osunlp/MagicBrush |
| Finance | Domain-specific checks | 20 samples |
| Claims processing | Domain-specific checks | 20 samples |

Guardrails are deterministic (no LLM in the loop). They clamp agent outputs to canonical pass/fail results.

## Setup

```bash
git clone https://github.com/jmanhype/AgentLearningEE.git
cd AgentLearningEE

python -m pip install -e .[dev]

# For dataset generation
python -m pip install datasets pillow numpy scikit-image
```

LLM credentials (any one works):
```bash
export OPENROUTER_API_KEY=sk-or-v1-...  # preferred
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### Optional: ACE playbook

```bash
git clone https://github.com/jmanhype/ace-playbook.git
python -m pip install -e ace-playbook
export ACE_ENABLED=1
export ACE_DOMAIN_ID=swe-bench
export DATABASE_URL=sqlite:///ace_playbook.db
```

## Usage

### Offline benchmark (no LLM needed)

```bash
PYTHONPATH=src python scripts/run_benchmark.py data/swe_bench_samples/swe_bench_50.jsonl \
    --domain swe-bench --offline --output results/swe_bench_offline.json
```

### Online benchmark

```bash
PYTHONPATH=src python scripts/run_benchmark.py data/swe_bench_samples/swe_bench_50.jsonl \
    --domain swe-bench --output results/swe_bench_online.json
```

### Live loop with ACE

```bash
ACE_ENABLED=1 ACE_DOMAIN_ID=swe-bench ACE_TARGET_STAGE=shadow \
DATABASE_URL=sqlite:///ace_playbook.db \
PYTHONPATH=src python examples/live_loop_swe_magic.py --domain swe-bench --episodes 50 --ace
```

### Full pipeline

```python
from agent_learning.pipeline import run_complete_pipeline
result = run_complete_pipeline(
    expert_demos_path="data/expert_demos.jsonl",
    output_dir="artifacts",
)
print(result["success"], result["metrics"])
```

## Commands

| Task | Command |
|------|---------|
| Install | `python -m pip install -e .[dev]` |
| Scaffold guardrails | `python scripts/scaffold_domain.py <domain> --from-benchmark <jsonl>` |
| Offline benchmark | `PYTHONPATH=src python scripts/run_benchmark.py ... --offline` |
| Online benchmark | `PYTHONPATH=src python scripts/run_benchmark.py ...` |
| Live loop + ACE | `ACE_ENABLED=1 ... python examples/live_loop_swe_magic.py --domain <d> --episodes N --ace` |
| Train pipeline | `python -m agent_learning.pipeline --expert-demos data/expert_demos.jsonl` |

## Project layout

```
src/
  agent_learning/     # EE pipeline (world_model, exploration, reflection, policy, live_loop)
  guardrails/         # Deterministic domain guardrails (swe_bench, magicbrush, finance, claims)
  ee_ace_bridge/      # ACE playbook integration
data/
  swe_bench_samples/  # Generated benchmark slice
  magicbrush_samples/ # Generated benchmark slice
  claims_samples/     # Claims processing samples
  finance_samples/    # Finance QA samples
scripts/              # Benchmarks, scaffolding, metrics
examples/             # Live loop demos
docs/                 # Domain ETL docs, onboarding guide
```

## Dependencies

Core: DSPy (>= 2.0), PyTorch (>= 2.0), Transformers (>= 4.30).
Optional: datasets, pillow, scikit-image (for sample generation).

Requires Python 3.11+.

## Limitations

- Sample datasets are generated from external sources (HuggingFace) and require downloading at setup time.
- The SWE-bench guardrail checks patch application but does not actually run test suites in this repo.
- ACE integration requires the separate ace-playbook repo and a SQLite/PostgreSQL database.
- The world model and policy modules use DSPy but their actual learning effectiveness is not benchmarked.
- No CI pipeline is configured.
- Version 0.3.0. API surface may change.

## License

MIT
