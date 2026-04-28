"""Parquet storage backend — single .parquet file for fast filtering.

Requires pyarrow. Falls back gracefully if not installed.
Recommended once run count exceeds ~100.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storage.run import Run, _json_default

_PARQUET_PATH = Path(__file__).parent.parent / "runs" / "runs.parquet"

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False


def _run_to_row(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "model": run.model,
        "params_hash": run.params_hash,
        "params_json": json.dumps(run.params, default=_json_default),
        "results_json": json.dumps(run.results, default=_json_default),
        "convergence_json": json.dumps(run.convergence, default=_json_default),
        "tags": json.dumps(run.tags),
        "notes": run.notes,
    }


def _row_to_run(row: dict) -> Run:
    return Run(
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        model=row["model"],
        params_hash=row["params_hash"],
        params=json.loads(row["params_json"]),
        results=json.loads(row["results_json"]),
        convergence=json.loads(row["convergence_json"]),
        tags=json.loads(row.get("tags", "[]")),
        notes=row.get("notes", ""),
    )


class ParquetBackend:
    """Append-only Parquet backend.  Reads are fully vectorised via pyarrow."""

    def __init__(self, parquet_path: Path | str | None = None):
        if not _HAVE_PYARROW:
            raise ImportError(
                "pyarrow is required for ParquetBackend. "
                "Install with: pip install pyarrow>=15.0"
            )
        self.path = Path(parquet_path) if parquet_path else _PARQUET_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_table(self):
        if self.path.exists():
            return pq.read_table(str(self.path))
        return pa.table({k: pa.array([], type=pa.string())
                         for k in _run_to_row(Run.create("_", {}, {})).keys()})

    def save(self, run: Run) -> None:
        existing = self._read_table()
        new_row = {k: [v] for k, v in _run_to_row(run).items()}
        new_table = pa.table(new_row)
        combined = pa.concat_tables([existing, new_table], promote_options="default")
        pq.write_table(combined, str(self.path), compression="snappy")

    def load_all(self) -> list[Run]:
        table = self._read_table()
        return [_row_to_run({k: table[k][i].as_py() for k in table.column_names})
                for i in range(table.num_rows)]

    def delete(self, run_id: str) -> bool:
        table = self._read_table()
        mask = [table["run_id"][i].as_py() != run_id
                for i in range(table.num_rows)]
        if all(mask):
            return False
        import pyarrow.compute as pc
        filtered = table.filter(pa.array(mask))
        pq.write_table(filtered, str(self.path), compression="snappy")
        return True

    def filter(self, model: str | None = None,
               date_from: str | None = None,
               date_to: str | None = None,
               tag: str | None = None) -> list[Run]:
        runs = self.load_all()
        if model:
            runs = [r for r in runs if r.model == model]
        if date_from:
            runs = [r for r in runs if r.timestamp >= date_from]
        if date_to:
            runs = [r for r in runs if r.timestamp <= date_to]
        if tag:
            runs = [r for r in runs if tag in r.tags]
        return runs
