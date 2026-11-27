"""Wall-clock training budgets without overriding the epoch/lr schedule."""
from __future__ import annotations

import time


def phase_hours(started: float, total_hours: float, reserve_hours: float, cap_hours: float,
                now: float | None = None) -> float:
    if total_hours <= 0 or reserve_hours < 0 or cap_hours < 0:
        raise ValueError("Invalid run budget")
    current = time.monotonic() if now is None else now
    remaining = total_hours - max(0., current - started) / 3600. - reserve_hours
    return max(0., min(cap_hours, remaining))


class StopAtDeadline:
    """Single-GPU callback: request normal validation/checkpoint save at a batch boundary.

    This bounds training, not network/data preparation, GPU kernel latency, or validation/export.
    """
    def __init__(self, hours: float, clock=time.monotonic):
        if hours <= 0:
            raise ValueError("Training budget must be positive")
        self.clock = clock
        self.deadline = clock() + hours * 3600.
        self.triggered = False

    def __call__(self, trainer):
        if self.clock() >= self.deadline:
            trainer.stop = True
            if not self.triggered:
                print("[BUDGET] Finishing current training step; saving checkpoint and validating.", flush=True)
            self.triggered = True
