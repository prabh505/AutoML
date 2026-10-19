"""File → typed frame + metadata (PRD §8, FR-ING-*, FR-TGT-01).

Interface: load_dataset, validate_target, estimate_feasibility. Depends on schemas, resources.

Must not: LLM, training; dropping rows other than missing-target rows; network access.
"""
