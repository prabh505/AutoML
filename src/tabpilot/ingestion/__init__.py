"""File → typed frame + metadata (PRD §8, FR-ING-*, FR-TGT-01).

Interface: load_dataset, validate_target, estimate_feasibility. Depends on schemas, resources.

Must not: LLM, training; dropping rows other than missing-target rows; network access.
"""

from tabpilot.ingestion.loader import LoadedDataset, load_dataset
from tabpilot.ingestion.normalize import normalize_columns, normalize_name
from tabpilot.ingestion.target import TargetCheck, apply_target, validate_target
from tabpilot.ingestion.upload import StoredUpload, accept_upload, sanitize_display_name
from tabpilot.resources.estimator import estimate_feasibility, raise_if_infeasible

__all__ = [
    "LoadedDataset", "StoredUpload", "TargetCheck", "accept_upload", "apply_target",
    "estimate_feasibility", "load_dataset", "normalize_columns", "normalize_name",
    "raise_if_infeasible", "sanitize_display_name", "validate_target",
]  # fmt: skip
