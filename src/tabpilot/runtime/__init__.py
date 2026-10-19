"""Run lifecycle: run manager, worker, events, cancel, slot lock, cleanup (PRD §28, FR-RUN-*).

Interface: RunManager.start/cancel/status/cleanup. Depends on orchestration.

Must not: ML logic.
"""
