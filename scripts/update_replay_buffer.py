"""Merge live loop episodes into an experience replay buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_episode_files(episodes_dir: Path) -> List[Path]:
    if not episodes_dir.exists():
        return []
    return sorted(episodes_dir.glob("episodes*.jsonl"))


def merge_replay_buffer(
    existing: List[Dict],
    new_records: List[Dict],
) -> List[Dict]:
    seen: Set[Tuple] = set()
    merged: List[Dict] = []

    def make_key(record: Dict) -> Tuple:
        return (
            record.get("task_id"),
            record.get("timestamp"),
            record.get("domain"),
            record.get("action"),
        )

    for record in existing:
        key = make_key(record)
        seen.add(key)
        merged.append(record)

    for record in new_records:
        key = make_key(record)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Update experience replay buffer from live loop episodes")
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path("live_loop_artifacts"),
        help="Directory containing live loop episode JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/replay_buffer.jsonl"),
        help="Path to the aggregated replay buffer JSONL",
    )
    args = parser.parse_args()

    episode_files = collect_episode_files(args.episodes_dir)
    if not episode_files:
        print(f"No episode files found under {args.episodes_dir}, skipping replay buffer update.")
        return

    new_records: List[Dict] = []
    for file_path in episode_files:
        new_records.extend(load_jsonl(file_path))

    if not new_records:
        print("Episode files were empty, skipping replay buffer update.")
        return

    existing_records = load_jsonl(args.output)
    merged_records = merge_replay_buffer(existing_records, new_records)

    write_jsonl(args.output, merged_records)
    print(
        "Replay buffer updated",
        json.dumps(
            {
                "output_path": str(args.output),
                "existing_records": len(existing_records),
                "new_records": len(new_records),
                "merged_records": len(merged_records),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
