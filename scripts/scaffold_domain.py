#!/usr/bin/env python3
"""Scaffold helper for creating guardrail-enabled domains.

Supports two workflows:
1. Create blank stubs for a new domain (benchmark, guardrails module, docs).
2. Generate guardrails directly from an existing benchmark JSONL with metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


BENCHMARK_TEMPLATE = Template(
    dedent(
        """\
        {
            "task_id": "${domain}-001",
            "description": "TODO: describe the input state",
            "ground_truth": "TODO"
        }
        """
    ).strip()
)


GUARDRAIL_TEMPLATE = Template(
    dedent(
        '''\
        """Guardrails for ${domain} domain."""

        from __future__ import annotations

        from typing import Dict, Optional

        from guardrails.base import NumericGuardrail
        from guardrails import register_domain


        DOMAIN_GUARDRAILS: Dict[str, NumericGuardrail] = {
            "${domain}-001": NumericGuardrail(
                instructions="TODO: describe how to validate the canonical answer.",
                calculator=None,
                auto_correct=False,
            ),
        }


        def get_guardrail(task_id: str) -> Optional[NumericGuardrail]:
            return DOMAIN_GUARDRAILS.get(task_id)


        register_domain("${domain}", DOMAIN_GUARDRAILS)
        '''
    ).strip()
)


DOCS_TEMPLATE = Template(
    dedent(
        """# ${title}

## Overview

- Benchmark file: `benchmarks/${domain}.jsonl`
- Guardrails module: `src/guardrails/${domain}.py`
- Results: `results/${domain}_benchmark.json`

## Setup Checklist

1. Populate the benchmark JSONL with representative tasks.
2. Implement guardrail calculators and set `auto_correct=True` when safe.
3. Register the domain (handled automatically by the generated module).
4. Run `python scripts/run_benchmark.py benchmarks/${domain}.jsonl --domain ${domain}`.
5. Capture findings and iterate on guardrails as needed.

"""
    ).strip()
)


def write_file(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def scaffold_domain(domain: str) -> None:
    normalized = domain.lower().replace(" ", "-")

    benchmark_path = PROJECT_ROOT / "benchmarks" / f"{normalized}.jsonl"
    guardrail_path = PROJECT_ROOT / "src" / "guardrails" / f"{normalized}.py"
    docs_path = PROJECT_ROOT / "docs" / "domains" / f"{normalized}.md"

    write_file(benchmark_path, BENCHMARK_TEMPLATE.substitute(domain=normalized))
    write_file(guardrail_path, GUARDRAIL_TEMPLATE.substitute(domain=normalized))

    title = normalized.replace("-", " ").title()
    write_file(docs_path, DOCS_TEMPLATE.substitute(domain=normalized, title=title))

    print(f"Created benchmark stub: {benchmark_path.relative_to(PROJECT_ROOT)}")
    print(f"Created guardrail module: {guardrail_path.relative_to(PROJECT_ROOT)}")
    print(f"Created docs stub: {docs_path.relative_to(PROJECT_ROOT)}")


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def guardrail_literal(task_id: str, meta: Dict[str, Any], fallback_value: str) -> str:
    instructions = meta.get(
        "instructions",
        f"Return only the canonical answer for {task_id} with no additional prose.",
    )
    value = meta.get("value", fallback_value)
    format_hint = meta.get("format")
    if not format_hint:
        format_hint = "percent" if str(value).strip().endswith("%") else "number"
    decimals = meta.get("decimals")

    lines = [
        f"    \"{task_id}\": constant_guardrail(",
        f"        instructions={instructions!r},",
        f"        value={value!r},",
        f"        format={format_hint!r},",
    ]
    if decimals is not None:
        lines.append(f"        decimals={decimals},")
    lines.append("    ),")
    return "\n".join(lines)


def stub_literal(task_id: str, description: str) -> str:
    desc = description or task_id
    return dedent(
        f"""    \"{task_id}\": NumericGuardrail(
        instructions="TODO: define guardrail instructions for: {desc}",
        calculator=None,
        auto_correct=False,
    ),"""
    )


def generate_guardrails_from_benchmark(domain: str, suite_path: Path) -> str:
    records = load_benchmark(suite_path)
    if not records:
        raise ValueError(f"No tasks found in benchmark: {suite_path}")

    entries: List[str] = []
    missing: List[str] = []
    for record in records:
        task_id = record.get("task_id")
        if not task_id:
            continue
        ground_truth = str(record.get("ground_truth", ""))
        guardrail_meta = record.get("guardrail") or {}
        if guardrail_meta:
            entries.append(guardrail_literal(task_id, guardrail_meta, ground_truth))
        else:
            entries.append(stub_literal(task_id, record.get("description", "")))
            missing.append(task_id)

    body = "\n".join(entries)

    module = dedent(
        f'''"""Guardrails for {domain} domain generated from benchmark."""

from __future__ import annotations

from typing import Dict, Optional

from guardrails.base import NumericGuardrail, constant_guardrail
from guardrails import register_domain


DOMAIN_GUARDRAILS: Dict[str, NumericGuardrail] = {{
{body}
}}


def get_guardrail(task_id: str) -> Optional[NumericGuardrail]:
    return DOMAIN_GUARDRAILS.get(task_id)


register_domain({domain!r}, DOMAIN_GUARDRAILS)
'''
    )

    if missing:
        print(
            "Warning: the following task_ids were scaffolded with TODO guardrails: "
            + ", ".join(missing)
        )

    return module


def scaffold_from_benchmark(domain: str, suite: Path) -> None:
    normalized = domain.lower().replace(" ", "-")
    suite_path = suite.resolve()
    if not suite_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {suite}")

    guardrail_path = PROJECT_ROOT / "src" / "guardrails" / f"{normalized}.py"
    docs_path = PROJECT_ROOT / "docs" / "domains" / f"{normalized}.md"

    guardrail_module = generate_guardrails_from_benchmark(normalized, suite_path)
    write_file(guardrail_path, guardrail_module)

    title = normalized.replace("-", " ").title()
    write_file(docs_path, DOCS_TEMPLATE.substitute(domain=normalized, title=title))

    print(f"Created guardrail module from benchmark: {guardrail_path.relative_to(PROJECT_ROOT)}")
    print(f"Created docs stub: {docs_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a guardrail-enabled domain")
    parser.add_argument("domain", help="Domain name (e.g., finance-lite)")
    parser.add_argument(
        "--from-benchmark",
        type=Path,
        help="Generate guardrails directly from an existing benchmark JSONL",
    )
    args = parser.parse_args()

    if args.from_benchmark:
        scaffold_from_benchmark(args.domain, args.from_benchmark)
    else:
        scaffold_domain(args.domain)


if __name__ == "__main__":
    main()
