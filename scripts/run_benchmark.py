"""Benchmark runner for AgentLearningEE policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import importlib

import dspy

from agent_learning.policy import generate_decision, load_trained_policy
from agent_learning.utils import setup_logger
from guardrails import get_guardrail


def _load_env_if_needed() -> None:
    if any(os.getenv(key) for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")):
        return

    env_path = Path(".env")
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value

    if not os.getenv("OPENROUTER_API_KEY"):
        openai_base = os.getenv("OPENAI_API_BASE", "")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and "openrouter" in openai_base.lower():
            os.environ["OPENROUTER_API_KEY"] = openai_key



def _valid_key(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return "your-api-key" not in lowered and "your-openai-key" not in lowered


def configure_lm() -> str:
    _load_env_if_needed()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if _valid_key(openrouter_key):
        model = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen-2.5-7b-instruct")
        dspy.configure(lm=dspy.LM(model, api_key=openrouter_key, api_base="https://openrouter.ai/api/v1"))
        return model
    if _valid_key(openai_key):
        model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        api_base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        dspy.configure(lm=dspy.LM(model, api_key=openai_key, api_base=api_base))
        return model
    if _valid_key(anthropic_key):
        model = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-3-haiku-20240307")
        dspy.configure(lm=dspy.LM(model, api_key=anthropic_key))
        return model

    raise RuntimeError(
        "No language model configured. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
    )


def load_tasks(path: Path, limit: Optional[int] = None) -> List[Dict]:
    tasks: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            tasks.append(json.loads(stripped))
            if limit and len(tasks) >= limit:
                break
    return tasks


def augment_description(task: Dict, domain: Optional[str]) -> str:
    description = task.get("description") or task.get("state") or ""
    guardrail = get_guardrail(task.get("task_id", ""), domain=domain)
    if guardrail:
        description = (
            f"{description}\n\nGuardrail: {guardrail.instructions} "
            "Return only the final value as specified."
        )
    return description


def evaluate_answer(answer: str, task: Dict, domain: Optional[str], *, corrected: Dict[str, List[Dict]]) -> bool:
    ground_truth = task.get("ground_truth") or ""
    guardrail = get_guardrail(task.get("task_id", ""), domain=domain)

    evaluation_answer = answer.strip()

    if guardrail and guardrail.auto_correct:
        canonical = guardrail.canonical_answer()
        if canonical and canonical != evaluation_answer:
            corrected.setdefault("auto", []).append(
                {
                    "task_id": task.get("task_id"),
                    "before": evaluation_answer,
                    "after": canonical,
                }
            )
            evaluation_answer = canonical

    if guardrail:
        return guardrail.validate(evaluation_answer, ground_truth)

    return evaluation_answer.rstrip(" .") == (ground_truth or "").strip().rstrip(" .")


def _missing_guardrails(tasks: List[Dict], domain: Optional[str]) -> List[str]:
    if not domain:
        return []
    missing: List[str] = []
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            continue
        guardrail = get_guardrail(task_id, domain=domain)
        if guardrail is None:
            missing.append(task_id)
    return missing


def run_benchmark(
    tasks: List[Dict],
    *,
    policy_path: Path,
    domain: Optional[str],
    logger_name: str,
    offline: bool,
    allow_missing_guardrails: bool,
) -> Dict:
    logger = setup_logger(logger_name)

    if domain:
        try:
            importlib.import_module(f"guardrails.{domain.replace('-', '_')}")
        except ModuleNotFoundError:
            pass

    policy = None
    if not offline:
        configure_lm()
        policy = load_trained_policy(str(policy_path))

    missing_guardrails = _missing_guardrails(tasks, domain)
    if missing_guardrails and not allow_missing_guardrails:
        raise RuntimeError(
            f"Missing guardrail definitions for tasks: {', '.join(sorted(missing_guardrails))}"
        )

    metrics: Dict[str, object] = {
        "total": 0,
        "correct": 0,
        "failures": [],
        "auto_corrections": [],
        "policy_path": str(policy_path),
        "domain": domain,
        "offline": offline,
        "missing_guardrails": missing_guardrails,
    }
    corrections: Dict[str, List[Dict]] = {"auto": []}

    for task in tasks:
        description = augment_description(task, domain)
        if offline:
            ground_truth = task.get("ground_truth", "") or ""
            reasoning_action = ("offline-mode", ground_truth)
        else:
            reasoning_action = generate_decision(policy, description, logger)

        metrics["total"] += 1

        if not reasoning_action:
            metrics.setdefault("failures", []).append(
                {
                    "task_id": task.get("task_id"),
                    "error": "generation_failed",
                    "ground_truth": task.get("ground_truth"),
                }
            )
            continue

        reasoning, action = reasoning_action

        is_correct = evaluate_answer(action, task, domain, corrected=corrections)

        if is_correct:
            metrics["correct"] += 1
        else:
            metrics.setdefault("failures", []).append(
                {
                    "task_id": task.get("task_id"),
                    "answer": action,
                    "ground_truth": task.get("ground_truth"),
                }
            )

    if corrections["auto"]:
        metrics["auto_corrections"] = corrections["auto"]

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentLearningEE benchmark suite")
    parser.add_argument("suite", type=Path, help="Path to JSONL benchmark suite")
    parser.add_argument("--policy-path", type=Path, default=Path("artifacts/policy.pkl"))
    parser.add_argument("--domain", type=str, default=None, help="Domain key for guardrail lookup")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_metrics.json"))
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional task limit")
    parser.add_argument("--logger-name", type=str, default="benchmark")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip model inference and treat ground truth as the predicted answer (CI-friendly)",
    )
    parser.add_argument(
        "--allow-missing-guardrails",
        action="store_true",
        help="Do not fail when tasks lack guardrail definitions",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.suite, args.max_tasks)
    metrics = run_benchmark(
        tasks,
        policy_path=args.policy_path,
        domain=args.domain,
        logger_name=args.logger_name,
        offline=args.offline,
        allow_missing_guardrails=args.allow_missing_guardrails,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
