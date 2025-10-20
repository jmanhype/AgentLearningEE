"""Guardrails for SWE-bench samples with deterministic patch application."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from guardrails import register_domain

logger = logging.getLogger(__name__)


class SweBenchGuardrail:
    """Guardrail that verifies patches by applying them to upstream repositories."""

    auto_correct = True
    format = "string"
    decimals = None
    CACHE_ROOT = Path.home() / ".cache" / "swe_guardrail"

    def __init__(
        self,
        *,
        task_id: str,
        repo: str,
        base_commit: str,
        patch: str,
        instructions: str,
        test_commands: Optional[List[str]] = None,
    ) -> None:
        self.task_id = task_id
        self.repo = repo
        self.base_commit = base_commit
        self.patch = patch
        self.instructions = instructions
        self.test_commands = test_commands or []
        self._result: Optional[str] = None

    def canonical_answer(self) -> str:
        self._ensure_evaluated()
        return self._result or "fail"

    def validate(self, answer: str, ground_truth: str) -> bool:
        self._ensure_evaluated()
        return (self._result or "fail") == "pass"

    def parse_numeric(self, answer: str) -> Optional[str]:  # compatibility hook
        return None

    def reset(self) -> None:
        self._result = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_evaluated(self) -> None:
        if self._result is not None:
            return

        repo_url = f"https://github.com/{self.repo}.git"
        SweBenchGuardrail.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="swe_guardrail_"))
        repo_dir = work_dir / "repo"

        try:
            mirror = self._ensure_mirror(repo_url)
            self._checkout_from_mirror(mirror, repo_url, repo_dir, self.base_commit)
            if not self._apply_patch(repo_dir, self.patch):
                self._result = "fail"
                return

            if self.test_commands:
                if not self._run_tests(repo_dir, self.test_commands):
                    self._result = "fail"
                    return

            self._result = "pass"
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("swe_guardrail_failure", extra={"task_id": self.task_id, "error": str(exc)})
            self._result = "fail"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _ensure_mirror(self, repo_url: str) -> Path:
        mirror_dir = self.CACHE_ROOT / repo_url.replace("https://github.com/", "").replace("/", "_")
        if not mirror_dir.exists():
            subprocess.run([
                "git",
                "clone",
                "--mirror",
                repo_url,
                str(mirror_dir),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run([
                "git",
                "-C",
                str(mirror_dir),
                "fetch",
                "origin",
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mirror_dir

    def _checkout_from_mirror(self, mirror: Path, repo_url: str, repo_dir: Path, commit: str) -> None:
        subprocess.run([
            "git",
            "clone",
            "--reference",
            str(mirror),
            repo_url,
            str(repo_dir),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "checkout", commit], check=True, cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _apply_patch(self, repo_dir: Path, patch: str) -> bool:
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            input=patch.encode("utf-8"),
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            logger.warning(
                "swe_guardrail_patch_failed",
                extra={"task_id": self.task_id, "stderr": result.stderr.decode("utf-8", "ignore")},
            )
            return False
        return True

    def _run_tests(self, repo_dir: Path, commands: List[str]) -> bool:
        for command in commands:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                logger.warning(
                    "swe_guardrail_tests_failed",
                    extra={
                        "task_id": self.task_id,
                        "command": command,
                        "stdout": proc.stdout.decode("utf-8", "ignore"),
                        "stderr": proc.stderr.decode("utf-8", "ignore"),
                    },
                )
                return False
        return True


def _load_guardrails() -> Dict[str, SweBenchGuardrail]:
    domain_guardrails: Dict[str, SweBenchGuardrail] = {}
    root = Path(__file__).resolve().parents[2]
    benchmark_path = root / "data" / "swe_bench_samples" / "swe_bench_50.jsonl"
    if not benchmark_path.exists():
        logger.warning("swe_guardrail_benchmark_missing", extra={"path": str(benchmark_path)})
        return domain_guardrails

    with benchmark_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task_id = record["task_id"]
            state = record["state"]
            action = record["action"]
            guardrail_meta = record.get("guardrail", {})

            domain_guardrails[task_id] = SweBenchGuardrail(
                task_id=task_id,
                repo=state["repo"],
                base_commit=state["base_commit"],
                patch=action["patch"],
                instructions=guardrail_meta.get(
                    "instructions",
                    "Apply the patch and ensure test_commands exit with code 0.",
                ),
                test_commands=state.get("failing_tests", []),
            )

    return domain_guardrails


DOMAIN_GUARDRAILS = _load_guardrails()


def get_guardrail(task_id: str) -> Optional[SweBenchGuardrail]:
    return DOMAIN_GUARDRAILS.get(task_id)


register_domain("swe-bench", DOMAIN_GUARDRAILS)


__all__ = ["SweBenchGuardrail", "get_guardrail", "DOMAIN_GUARDRAILS"]
