"""Small dependency-free terminal progress display for long experiment stages."""

from __future__ import annotations

import sys
import time


def _duration(seconds: float) -> str:
    if not seconds or seconds < 0.0 or seconds == float("inf"):
        return "--:--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TerminalProgress:
    """Render an in-place bar in a terminal and sparse lines when redirected."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        width: int = 20,
        minimum_interval: float = 0.25,
        initial_completed: int = 0,
        prior_elapsed: float = 0.0,
    ) -> None:
        self.label = label
        self.total = max(int(total), 0)
        self.width = max(int(width), 5)
        self.minimum_interval = float(minimum_interval)
        self.started = time.perf_counter()
        self.initial_completed = int(initial_completed)
        self.prior_elapsed = float(prior_elapsed)
        self.last_update = 0.0
        self.last_length = 0
        self.last_logged_tenth = -1
        self.stream = sys.stderr
        self.interactive = bool(self.stream.isatty())

    def update(self, completed: int, *, detail: str = "", force: bool = False) -> None:
        completed = min(max(int(completed), 0), self.total) if self.total else 0
        now = time.perf_counter()
        if not force and completed < self.total and now - self.last_update < self.minimum_interval:
            return
        fraction = completed / self.total if self.total else 1.0
        session_elapsed = now - self.started
        elapsed = session_elapsed + self.prior_elapsed
        session_completed = completed - self.initial_completed
        rate = session_completed / session_elapsed if session_completed > 0 and session_elapsed > 0 else 0.0
        eta = (self.total - completed) / rate if rate > 0.0 else float("inf")
        filled = int(round(self.width * fraction))
        bar = "█" * filled + "░" * (self.width - filled)
        line = (
            f"[{self.label}] [{bar}] {completed}/{self.total} "
            f"{100.0 * fraction:5.1f}% elapsed={_duration(elapsed)} "
            f"ETA={_duration(eta)}"
        )
        if detail:
            line += f" {detail}"

        if self.interactive:
            print(f"\r\033[2K{line}", end="", file=self.stream, flush=True)
            self.last_length = len(line)
            if completed >= self.total:
                print(file=self.stream, flush=True)
        else:
            tenth = int(fraction * 10)
            if force or completed >= self.total or tenth > self.last_logged_tenth:
                print(line, file=self.stream, flush=True)
                self.last_logged_tenth = tenth
        self.last_update = now

    def update_nested(
        self,
        *,
        outer_completed: int,
        outer_detail: str,
        inner_completed: int,
        inner_total: int,
        valid: int,
        failed: int = 0,
    ) -> None:
        """Update one shared line for an outer task and its current inner loop."""
        now = time.perf_counter()
        if now - self.last_update < self.minimum_interval and inner_completed < inner_total:
            return
        inner_total = max(int(inner_total), 1)
        inner_completed = min(max(int(inner_completed), 0), inner_total)
        outer_completed = min(max(int(outer_completed), 0), self.total)
        equivalent_completed = outer_completed + inner_completed / inner_total
        fraction = equivalent_completed / self.total if self.total else 1.0
        session_elapsed = now - self.started
        elapsed = session_elapsed + self.prior_elapsed
        session_completed = equivalent_completed - self.initial_completed
        rate = (
            session_completed / session_elapsed
            if session_completed > 0.0 and session_elapsed > 0.0
            else 0.0
        )
        eta = (
            (self.total - equivalent_completed) / rate
            if rate > 0.0
            else float("inf")
        )
        filled = int(round(self.width * fraction))
        bar = "█" * filled + "░" * (self.width - filled)
        outer_index = min(outer_completed + 1, self.total)
        line = (
            f"[{self.label}] [{bar}] overall={100.0 * fraction:5.1f}% "
            f"outer={outer_index}/{self.total} {outer_detail} "
            f"inner={inner_completed}/{inner_total} ({100 * inner_completed / inner_total:.1f}%) valid={int(valid)} "
            f"failed={int(failed)} "
            f"elapsed={_duration(elapsed)} ETA={_duration(eta)}"
        )

        if self.interactive:
            print(f"\r\033[2K{line}", end="", file=self.stream, flush=True)
            self.last_length = len(line)
        else:
            tenth = int(fraction * 10)
            if tenth > self.last_logged_tenth:
                print(line, file=self.stream, flush=True)
                self.last_logged_tenth = tenth
        self.last_update = now
