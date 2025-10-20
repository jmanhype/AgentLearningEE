# SWE-bench ETL Plan

## Goal

Transform `princeton-nlp/SWE-bench` records into Early Experience (EE) trajectories plus guardrail metadata so the coding agent can train, benchmark, and self-correct via deterministic tests.

## Source Layout

Each SWE-bench sample provides:

- `repo`: GitHub repository identifier
- `base_commit`: commit SHA before the fix
- `issue`: natural-language issue description (title + body)
- `patch`: unified diff implementing the fix
- `test_commands`: shell commands that verify the bug
- Optional supporting snippets (stack traces, reproducer files)

## Target Schema (per JSONL line)

```json
{
  "task_id": "swe-<repo>-<issue_id>",
  "state": {
    "repo": "...",
    "base_commit": "...",
    "issue": "...",
    "failing_tests": "pytest tests/...",
    "files": {
      "path/to/file.py": "<pre-context snippet>"
    }
  },
  "action": {
    "patch": "diff --git ..."
  },
  "next_state": {
    "repo": "...",
    "applied_patch": true,
    "tests_pass": true
  },
  "ground_truth": "tests pass",
  "guardrail": {
    "instructions": "Apply the patch and ensure test suite exits with status 0 (no warnings ignored).",
    "value": "pass",
    "format": "string"
  }
}
```

### Notes

- `state.files` can be limited to key files mentioned in the issue. Include enough context to reproduce the bug before the patch.
- `action.patch` should be the exact unified diff (ensure newline escapes are preserved).
- `next_state` records canonical success indicators after applying the patch.
- `guardrail.value` is intentionally simple (`"pass"`); the deterministic evaluator will run the test commands and substitute `"pass"`/`"fail"` as needed.

## Deterministic Guardrails

1. Clone repo at `base_commit`.
2. Apply patch (`git apply`). If it fails, canonical answer is `fail`.
3. Run `test_commands` with timeout; canonical answer is `pass` only when exit code == 0.
4. Log artifacts (stdout/stderr) for ACE insights.

## ETL Steps

1. **Extract**: iterate through SWE-bench samples, fetch repo snapshot at `base_commit` (use cached git mirror).
2. **Transform**:
   - Capture pre-patch context snippets (optional for compact JSON).
   - Assemble JSONL record using schema above.
   - Add guardrail metadata (instructions/value/format).
3. **Load**: write newline-delimited JSON to `benchmarks/swe-bench.jsonl`.
4. **Scaffold**: `python scripts/scaffold_domain.py swe-bench --from-benchmark benchmarks/swe-bench.jsonl` (auto-generates guardrail module).
5. **Verify**: run `python scripts/run_benchmark.py benchmarks/swe-bench.jsonl --domain swe-bench --offline` (tests all guardrails without executing tests) then optional online run with real repos in a workspace that allows git + pytest.

## ACE Integration

- During benchmarking or live loop, guardrail corrections (e.g., test failures) are fed into ACE via `guardrail_auto_corrections` metrics.
- ACE insights can capture rules: “Run `pytest tests/foo.py` after editing `foo.py`” or “Ensure regression test `test_issue_123` passes,” reinforcing future behavior.
