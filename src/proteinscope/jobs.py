"""Analyses started from the browser, run one at a time in a background thread.

An analysis is CPU-bound and takes seconds to a minute, so the web server hands it to a
single worker and answers at once; the page then polls for progress. One worker keeps a
laptop responsive and keeps the RCSB, UniProt and ChEMBL request rate polite.

Messages shown in the browser never carry a traceback or a local path.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .pipeline import Progress

# runner(identifier, out_dir, progress) analyses one structure and writes its report.
Runner = Callable[[str, Path, Progress], None]

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


class QueueFull(RuntimeError):
    """Too many analyses are waiting; the caller should try again later."""


@dataclass
class Job:
    identifier: str
    state: str = QUEUED
    stage: str = "Waiting"
    fraction: float = 0.0
    error: str | None = None
    submitted: float = field(default_factory=time.time)
    finished: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "state": self.state,
            "stage": self.stage,
            "fraction": round(self.fraction, 3),
            "error": self.error,
            "report": f"{self.identifier}/report.html" if self.state == DONE else None,
        }


def default_runner(identifier: str, out_dir: Path, progress: Progress) -> None:
    from .pipeline import run_analysis
    from .report.builder import build_report

    # allow_files=False: an identifier from the browser is never read as a local path.
    result = run_analysis(identifier, progress=progress, allow_files=False)
    progress("Writing report and figures", 0.8)
    build_report(result, out_dir)


def public_message(exc: BaseException) -> str:
    """An error message fit for the browser: first line only, no local paths, bounded."""
    from .rcsb import RCSBError
    from .webdata import DataSourceError

    if isinstance(exc, (ValueError, RCSBError, DataSourceError, FileNotFoundError)):
        text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    else:
        text = f"The analysis stopped with an internal error ({type(exc).__name__})."
    for secret in sorted({str(Path.home()), str(Path.cwd())}, key=len, reverse=True):
        text = text.replace(secret, "~" if secret == str(Path.home()) else ".")
    return text[:240]


class JobQueue:
    """One job per identifier; a resubmitted identifier returns its live job."""

    def __init__(self, out_dir: Path, runner: Runner | None = None, max_pending: int = 8) -> None:
        self.out_dir = Path(out_dir)
        self.runner = runner or default_runner
        self.max_pending = max_pending
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._work, name="proteinscope-jobs", daemon=True)
        self._worker.start()

    def submit(self, identifier: str) -> Job:
        with self._lock:
            current = self._jobs.get(identifier)
            if current and current.state in (QUEUED, RUNNING):
                return current
            pending = sum(1 for job in self._jobs.values() if job.state in (QUEUED, RUNNING))
            if pending >= self.max_pending:
                raise QueueFull(f"{pending} analyses are already waiting. Try again when one has finished.")
            job = Job(identifier)
            self._jobs[identifier] = job
        self._queue.put(identifier)
        return job

    def get(self, identifier: str) -> Job | None:
        with self._lock:
            return self._jobs.get(identifier)

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [job.to_dict() for job in sorted(self._jobs.values(), key=lambda j: j.submitted)]

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until nothing is queued or running; for tests and orderly shutdown."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if not any(job.state in (QUEUED, RUNNING) for job in self._jobs.values()):
                    return True
            time.sleep(0.02)
        return False

    def _update(self, job: Job, **changes: object) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(job, name, value)

    def _work(self) -> None:
        while True:
            identifier = self._queue.get()
            job = self.get(identifier)
            if job is None or job.state != QUEUED:
                continue
            self._update(job, state=RUNNING, stage="Starting", fraction=0.01)

            def progress(stage: str, fraction: float, job: Job = job) -> None:
                self._update(job, stage=stage, fraction=max(job.fraction, min(fraction, 0.99)))

            try:
                self.runner(identifier, self.out_dir, progress)
            except Exception as exc:  # Any failure must end the job, not the worker.
                message = public_message(exc)
                print(f"Analysis of {identifier} failed: {message}", file=sys.stderr, flush=True)
                self._update(job, state=FAILED, stage="Failed", error=message, finished=time.time())
            else:
                self._update(job, state=DONE, stage="Report ready", fraction=1.0, finished=time.time())
