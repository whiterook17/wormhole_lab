"""JSON storage backend — one .json file per run in runs/.

Default backend; human-readable, zero extra dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from storage.run import Run

_RUNS_DIR = Path(__file__).parent.parent / "runs"


class JSONBackend:
    def __init__(self, runs_dir: Path | str | None = None):
        self.runs_dir = Path(runs_dir) if runs_dir else _RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(self, run: Run) -> Path:
        path = self.runs_dir / f"{run.timestamp[:10]}_{run.run_id}.json"
        path.write_text(run.to_json(), encoding="utf-8")
        return path

    # ── Read ───────────────────────────────────────────────────────────────────

    def load(self, run_id: str) -> Run | None:
        for f in self.runs_dir.glob("*.json"):
            if run_id in f.name:
                return Run.from_json(f.read_text(encoding="utf-8"))
        return None

    def load_all(self) -> list[Run]:
        runs = []
        for f in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                runs.append(Run.from_json(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return runs

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete(self, run_id: str) -> bool:
        for f in self.runs_dir.glob("*.json"):
            if run_id in f.name:
                f.unlink()
                return True
        return False

    # ── Filter ─────────────────────────────────────────────────────────────────

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

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_json(self, run_ids: list[str]) -> str:
        runs = [r for r in self.load_all() if r.run_id in run_ids]
        return json.dumps([json.loads(r.to_json()) for r in runs], indent=2)

    def export_csv(self, run_ids: list[str]) -> str:
        import io, csv
        runs = [r for r in self.load_all() if r.run_id in run_ids]
        if not runs:
            return ""
        rows = [r.flat_dict() for r in runs]
        keys = list(rows[0].keys())
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
