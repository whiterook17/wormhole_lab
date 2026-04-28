"""Run data structure — one Run per Solve/Check invocation.

Each Run is an immutable snapshot of inputs + outputs + solver metadata.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _short_uuid() -> str:
    return str(uuid.uuid4())[:8]


def _params_hash(params: dict) -> str:
    serialised = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


@dataclass
class Run:
    """Complete record of one physics computation session."""

    run_id: str
    timestamp: str           # ISO-8601
    model: str               # "GR", "fR", "Throat", "Verification", …
    params: dict             # all user inputs
    params_hash: str         # SHA256[:16] of params for dedup
    results: dict            # all outputs (NEC, tau, f0, regime, …)
    convergence: dict        # solver method, residual, n_steps, success
    plots: list[str] = field(default_factory=list)   # saved PNG paths
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, model: str, params: dict, results: dict,
               convergence: dict | None = None) -> "Run":
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(
            run_id=_short_uuid(),
            timestamp=now,
            model=model,
            params=params,
            params_hash=_params_hash(params),
            results=results,
            convergence=convergence or {},
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=_json_default, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Run":
        d = json.loads(s)
        return cls(**d)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def flat_dict(self) -> dict:
        """Flat representation for DataFrame display."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "tags": ", ".join(self.tags),
            "notes": self.notes,
            **{f"p_{k}": v for k, v in self.params.items()},
            **{f"r_{k}": v for k, v in self.results.items()
               if isinstance(v, (int, float, bool, str))},
        }


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for numpy scalars etc."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
