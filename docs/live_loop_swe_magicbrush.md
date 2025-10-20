# Live Loop & ACE Integration Guide (SWE-bench + MagicBrush)

This note explains how to feed the new deterministic guardrails into the continuous EE→ACE loop. The goal is to let the agent learn from actual patch/test feedback (SWE-bench) or perceptual validation (MagicBrush) while ACE accumulates the resulting lessons.

---

## Common Preparation

1. **Import guardrails before running loops**
   ```python
   import guardrails.swe_bench  # registers patch+test runner
   import guardrails.magicbrush  # registers MSE/SSIM guardrails
   ```

2. **Load benchmark records**
   - SWE-bench: `data/swe_bench_samples/swe_bench_50.jsonl`
   - MagicBrush: `data/magicbrush_samples/magicbrush_50.jsonl`
   Each line contains `state`, `action`, `next_state`, `ground_truth`, and guardrail metadata.

3. **Configure ACE**
   Ensure `ACE_ENABLED=1` and import `ee_ace_bridge` so delta updates reach the playbook. When guardrail auto-corrections fire, ACE will ingest insights such as “ensure test suite exits with code 0” or “maintain SSIM ≥ 0.60”.

---

## SWE-bench: Live Loop Skeleton

```python
import json
from pathlib import Path

import dspy
from agent_learning.live_loop import LiveExplorationLoop, LiveLoopConfig
from guardrails import get_guardrail
import guardrails.swe_bench  # side effect: register guardrails


class SweBenchEnvironment:
    def __init__(self, dataset_path: str):
        self.records = json.loads(Path(dataset_path).read_text().splitlines())
        self.index = 0

    def reset(self):
        record = self.records[self.index % len(self.records)]
        self.current = record
        state = record["state"]
        return json.dumps({
            "task_id": record["task_id"],
            "repo": state["repo"],
            "issue": state["issue"],
            "tests": state.get("failing_tests", [])
        }, indent=2)

    def step(self, action: str):
        record = self.current
        guardrail = get_guardrail(record["task_id"], domain="swe-bench")
        guardrail._result = None  # force re-evaluation
        guardrail.validate(action, record.get("ground_truth", "pass"))
        info = {
            "task_id": record["task_id"],
            "canonical": guardrail.canonical_answer()
        }
        self.index += 1
        return json.dumps(info, indent=2), True


config = LiveLoopConfig(
    max_episodes=50,
    reflection_interval=5,
    ace_enabled=True,
    default_guardrail_domain="swe-bench",
)

env = SweBenchEnvironment("data/swe_bench_samples/swe_bench_50.jsonl")
loop = LiveExplorationLoop(env, policy_path="artifacts/policy.pkl", config=config)
loop.run()
```

- The guardrail’s canonical answer (`pass` or `fail`) is injected into live loop logs and propagated to ACE when reflections run.
- Extend `_run_tests` in `SweBenchGuardrail` to include reproduction commands or environment variables as needed.
- In a production setup, cache clones under `repo_cache` to avoid re-fetching on every episode.

---

## MagicBrush: Live Loop Skeleton

```python
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
import dspy
from agent_learning.live_loop import LiveExplorationLoop, LiveLoopConfig
from guardrails import get_guardrail
import guardrails.magicbrush


class MagicBrushEnvironment:
    def __init__(self, dataset_path: str):
        self.records = [json.loads(line) for line in Path(dataset_path).read_text().splitlines()]
        self.index = 0

    def reset(self):
        record = self.records[self.index % len(self.records)]
        self.current = record
        state = record["state"]
        preview = {
            "task_id": record["task_id"],
            "instruction": state["instruction"],
            "size": state["metadata"]
        }
        return json.dumps(preview, indent=2)

    def step(self, action: str):
        record = self.current
        guardrail = get_guardrail(record["task_id"], domain="magicbrush")
        guardrail._result = None
        guardrail.validate(action, record.get("ground_truth", "pass"))
        info = {
            "task_id": record["task_id"],
            "canonical": guardrail.canonical_answer(),
            "metrics": record["next_state"]["metrics"]
        }
        self.index += 1
        return json.dumps(info, indent=2), True


config = LiveLoopConfig(
    max_episodes=50,
    reflection_interval=5,
    ace_enabled=True,
    default_guardrail_domain="magicbrush",
)

env = MagicBrushEnvironment("data/magicbrush_samples/magicbrush_50.jsonl")
loop = LiveExplorationLoop(env, policy_path="artifacts/policy.pkl", config=config)
loop.run()
```

- `MagicBrushGuardrail` recomputes MSE/SSIM deterministically before returning “pass”.
- To reduce payloads for live prompts, swap base64 fields for object-store URIs and have the guardrail fetch from storage.
- Guardrail auto-corrections (e.g., SSIM too low) become ACE insights that tighten future instructions.

---

## Feeding ACE & Playbook Updates

- Enable ACE (`ACE_ENABLED=1`) and let `LiveExplorationLoop` run with reflections enabled. Guardrail metrics (pass/fail) are logged via `guardrail_passed`, `guardrail_auto_corrections`, and surfaced in ACE ingestion.
- After each reflection batch, ACE’s creator appends delta updates like “Always run `pytest` after modifying lint rules” or “Keep SSIM ≥ 0.60 when replacing skies”.
- Inspect ACE playbook via `ee_ace_bridge.augment_state_with_playbook` or the ACE client to verify new lessons.

---

## Tips

- **Caching repos**: For SWE-bench, persist clone dirs to avoid repeated git operations. Update `SweBenchGuardrail._clone_repo` to reuse a local mirror.
- **Sandboxing**: Run the guardrail calculations (tests, LPIPS/SSIM) in a controlled environment; ensure commands are safe before enabling in CI.
- **ACE insights**: In `SweBenchGuardrail._run_tests`, log stdout/stderr to feed richer context into reflections and ACE.
- **Live Loop metrics**: Check `live_loop_artifacts/episodes.jsonl` for guardrail pass/fail counts, auto-corrections, and canonical answers.

With these hooks, SWE-bench and MagicBrush demos participate fully in the reward-free EE → ACE loop: policy acts, guardrails enforce deterministic feedback, reflections distill lessons, and ACE evolves the playbook for subsequent episodes.

