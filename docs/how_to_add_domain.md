# How to Add a Domain in 5 Steps

1. **Scaffold the domain**
   - Quick stub: `python scripts/scaffold_domain.py your-domain-name`.
   - From an existing suite: `python scripts/scaffold_domain.py your-domain-name --from-benchmark benchmarks/your-suite.jsonl`.
   - Generated assets always include:
     - `benchmarks/your-domain-name.jsonl`
     - `src/guardrails/your-domain-name.py`
     - `docs/domains/your-domain-name.md`

2. **Populate the benchmark suite**
   - Add newline-delimited JSON tasks to `benchmarks/your-domain-name.jsonl`.
   - Each record should include `task_id`, `description` (or `state`), `ground_truth`, and optionally a `guardrail` block:
     ```json
     {
       "task_id": "fin-001",
       "description": "Compute net income from revenue 1200 and expenses 450",
       "ground_truth": "750",
       "guardrail": {
         "instructions": "Return only revenue minus expenses with no prose.",
         "value": "750",
         "format": "number",
         "decimals": 0
       }
     }
     ```
 - Supplying the `guardrail` block lets the scaffolder generate deterministic calculators automatically.

   **Ready-made recipes:**

   - Coding (SWE-bench) ETL plan: see `docs/etl_swe_bench.md`
   - Artistry (MagicBrush) ETL plan: see `docs/etl_magicbrush.md`

3. **Implement guardrail calculators**
   - If you used `--from-benchmark`, review the generated module and replace any TODO stubs with real logic.
   - Provide deterministic calculators (or use `constant_guardrail`) and set `auto_correct=True` when safe.
   - Importing the module automatically registers the domain with the guardrail registry.

4. **Train and benchmark the policy**
   - Produce a policy via the standard pipeline (`run_complete_pipeline`).
   - Validate it with `python scripts/run_benchmark.py benchmarks/your-domain-name.jsonl --domain your-domain-name`.
   - Review `results/benchmark_metrics.json` for guardrail matches, auto-corrections, and failures.

5. **Wire the live loop**
   - Update your environment to return metadata (`task_id`, `domain`, `ground_truth`) from `reset()`.
   - `LiveExplorationLoop` will auto-augment prompts, apply guardrail corrections, and log pass/fail stats during continuous runs.
