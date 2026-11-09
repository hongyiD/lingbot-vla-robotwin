"""Minimal readers for LeRobot dataset metadata (v2.1 and v3.0), no lerobot dependency."""
from __future__ import annotations

import json
from pathlib import Path

from .tasks import TASKS

_TASKS_BY_LENGTH = sorted(TASKS, key=len, reverse=True)


def read_info(ds: Path) -> dict | None:
    info = ds / "meta" / "info.json"
    if not info.is_file():
        return None
    return json.loads(info.read_text(encoding="utf-8"))


def read_tasks(ds: Path) -> list[str] | None:
    """Task (instruction) strings of a dataset; None if no tasks file can be read."""
    jsonl = ds / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        out = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(str(json.loads(line)["task"]))
        return out
    parquet = ds / "meta" / "tasks.parquet"
    if parquet.is_file():
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(parquet)
        for col in ("task", "__index_level_0__"):
            if col in table.column_names:
                return [str(x) for x in table.column(col).to_pylist()]
        for name, typ in zip(table.column_names, table.schema.types):
            if pa.types.is_string(typ) or pa.types.is_large_string(typ):
                return [str(x) for x in table.column(name).to_pylist()]
    return None


def infer_task_from_name(name: str) -> str | None:
    """Map a dataset directory name such as 'click_bell-aloha-agilex_clean_50-50' or
    'shake_bottle_horizontally_joint_v30' to its task (longest prefix match)."""
    for task in _TASKS_BY_LENGTH:
        if name == task or name.startswith(task + "-") or name.startswith(task + "_"):
            return task
    return None
