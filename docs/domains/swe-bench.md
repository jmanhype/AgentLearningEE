# Swe Bench

## Overview

- Benchmark file: `benchmarks/swe-bench.jsonl`
- Guardrails module: `src/guardrails/swe-bench.py`
- Results: `results/swe-bench_benchmark.json`

## Setup Checklist

1. Populate the benchmark JSONL with representative tasks.
2. Implement guardrail calculators and set `auto_correct=True` when safe.
3. Register the domain (handled automatically by the generated module).
4. Run `python scripts/run_benchmark.py benchmarks/swe-bench.jsonl --domain swe-bench`.
5. Capture findings and iterate on guardrails as needed.
