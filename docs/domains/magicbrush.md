# Magicbrush

## Overview

- Benchmark file: `benchmarks/magicbrush.jsonl`
- Guardrails module: `src/guardrails/magicbrush.py`
- Results: `results/magicbrush_benchmark.json`

## Setup Checklist

1. Populate the benchmark JSONL with representative tasks.
2. Implement guardrail calculators and set `auto_correct=True` when safe.
3. Register the domain (handled automatically by the generated module).
4. Run `python scripts/run_benchmark.py benchmarks/magicbrush.jsonl --domain magicbrush`.
5. Capture findings and iterate on guardrails as needed.
