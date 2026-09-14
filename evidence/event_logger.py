"""
Evidence Event Logger
=====================

One append-only JSONL writer, shared by every phase of RoboGuardian
(baseline, disturbance injection, detection, recovery). This is the
single source of truth for "what happened" that G3–G6 evidence and the
killer-demo overlay both read from.

Deliberately NOT an evidence platform: no DB, no schema migrations,
just append-only JSONL per run, exactly as specified in the build plan
(section 22/23) — "ไฟล์ไม่ต้องซับซ้อน แค่ทำให้ทุก PASS ตรวจย้อนหลังได้".

Usage:
    logger = EventLogger(run_id="RUN-0042")

    logger.log_step(
        step_id="STEP-04",
        action_id="ACT-017",
        action="pick_plate",
        expected_pose=[0.42, 0.18, 0.03],
        observed_pose=[0.54, 0.19, 0.03],
        pose_error=0.12,
        state="RECOVERABLE",
        decision="PAUSE_AND_REOBSERVE",
        executed=False,
        reason="OBJECT_DISPLACED",
    )

    logger.log_recovery(
        step_id="STEP-04-R1",
        state="NORMAL",
        decision="RESUME",
        recovery_attempt=1,
        target_updated=True,
        executed=True,
        success=True,
    )

    logger.close()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EventLogger:
    """Append-only JSONL logger for one run.

    One file per run_id under evidence_dir, named `<run_id>.jsonl`.
    Every log_* call appends exactly one line and flushes immediately —
    a crash mid-trial should never lose the trace leading up to it.
    """

    def __init__(self, run_id: str, evidence_dir: str = "results/events"):
        self.run_id = run_id
        self._dir = Path(evidence_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_id}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")
        self._t0 = time.perf_counter()

    # -- internal ------------------------------------------------------

    def _elapsed(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    def _write(self, event: dict[str, Any]) -> None:
        event.setdefault("run_id", self.run_id)
        event.setdefault("timestamp", self._elapsed())
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()

    # -- public API ------------------------------------------------------

    def log_step(
        self,
        *,
        step_id: str,
        action_id: str,
        action: str,
        expected_pose: list[float],
        observed_pose: list[float],
        pose_error: float,
        state: str,
        decision: str,
        executed: bool,
        reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record one plan/observe/decide cycle at a critical action
        boundary — the primary event type described in section 22."""
        event = {
            "step_id": step_id,
            "action_id": action_id,
            "action": action,
            "expected_pose": expected_pose,
            "observed_pose": observed_pose,
            "pose_error": pose_error,
            "state": state,
            "decision": decision,
            "executed": executed,
        }
        if reason is not None:
            event["reason"] = reason
        if extra:
            event.update(extra)
        self._write(event)

    def log_recovery(
        self,
        *,
        step_id: str,
        state: str,
        decision: str,
        recovery_attempt: int,
        target_updated: bool,
        executed: bool,
        success: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record one recovery attempt's outcome (section 22, second
        event type)."""
        event = {
            "step_id": step_id,
            "state": state,
            "decision": decision,
            "recovery_attempt": recovery_attempt,
            "target_updated": target_updated,
            "executed": executed,
            "success": success,
        }
        if extra:
            event.update(extra)
        self._write(event)

    def log_safe_stop(
        self,
        *,
        step_id: str,
        reason: str,
        recovery_attempts_used: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a SAFE_STOP terminal event — always executed=False,
        per the safe-stop semantics in section 11."""
        event = {
            "step_id": step_id,
            "state": "SAFE_STOP",
            "decision": "NOT_EXECUTED",
            "reason": reason,
            "recovery_attempts_used": recovery_attempts_used,
            "executed": False,
        }
        if extra:
            event.update(extra)
        self._write(event)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_run(run_id: str, evidence_dir: str = "results/events") -> list[dict[str, Any]]:
    """Read back every event for a run, in order — used by RESULTS.md
    generation and by gate evidence scripts (g4/g5/g6) to reconstruct
    what happened without re-running the sim."""
    path = Path(evidence_dir) / f"{run_id}.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    # Smoke test / usage example — run directly to sanity-check the logger.
    with EventLogger(run_id="RUN-SMOKETEST") as logger:
        logger.log_step(
            step_id="STEP-04",
            action_id="ACT-017",
            action="pick_plate",
            expected_pose=[0.42, 0.18, 0.03],
            observed_pose=[0.54, 0.19, 0.03],
            pose_error=0.12,
            state="RECOVERABLE",
            decision="PAUSE_AND_REOBSERVE",
            executed=False,
            reason="OBJECT_DISPLACED",
        )
        logger.log_recovery(
            step_id="STEP-04-R1",
            state="NORMAL",
            decision="RESUME",
            recovery_attempt=1,
            target_updated=True,
            executed=True,
            success=True,
        )

    events = read_run("RUN-SMOKETEST")
    print(f"Wrote and read back {len(events)} events for RUN-SMOKETEST.")
