"""Bundle, manifest, pipeline.py, parity (PRD §22, FR-ART-*).

Interface: export_bundle(state, final_model, report_dir, settings) -> BundleResult.

Must not: llm; raw dataset rows in the bundle.
"""

from tabpilot.export.bundle import (
    BundleResult,
    BundleTooLargeError,
    ExportSettings,
    RawRowError,
    dataset_slug,
    export_bundle,
    find_raw_rows,
    preprocessing_spec,
    sha256_file,
)
from tabpilot.export.code import render_pipeline
from tabpilot.export.codegen import CodegenError, estimator_source
from tabpilot.export.csvsafe import csv_text, safe_cell, write_csv
from tabpilot.export.parity import (
    R1Result,
    column_rules,
    fixture_frame,
    known_categories,
    row_hashes,
    serialize_and_verify,
)
from tabpilot.export.requirements import model_libraries, pinned, requirements_for

__all__ = [
    "BundleResult",
    "BundleTooLargeError",
    "CodegenError",
    "ExportSettings",
    "R1Result",
    "RawRowError",
    "column_rules",
    "csv_text",
    "dataset_slug",
    "estimator_source",
    "export_bundle",
    "find_raw_rows",
    "fixture_frame",
    "known_categories",
    "model_libraries",
    "pinned",
    "preprocessing_spec",
    "render_pipeline",
    "requirements_for",
    "row_hashes",
    "safe_cell",
    "serialize_and_verify",
    "sha256_file",
    "write_csv",
]
