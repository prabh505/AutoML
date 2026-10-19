"""Run records; MLflow for local/benchmark only (PRD §30).

Writes events, stage timings, trials, resources, experiment records.

Must not: Raw cell values in records.
"""

from tabpilot.tracking.records import RECORD_FILES, RunRecorder, dumps, to_jsonable

__all__ = ["RECORD_FILES", "RunRecorder", "dumps", "to_jsonable"]
