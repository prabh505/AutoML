"""Fixture datasets prepared up to training inputs (load → target → split → profile → schema).

Cached per (file, target, profile) because profiling is the slow step; callers must not mutate
the returned objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tabpilot.config import Settings, load_settings
from tabpilot.evaluation.metrics import Problem
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.ingestion.loader import LoadedDataset
from tabpilot.ingestion.task import detect_task
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import PipelineSchema, schema_from
from tabpilot.profiling import profile
from tabpilot.schemas import ExecutionProfileName, SplitSpec, Task
from tabpilot.splitting import build_split
from tabpilot.training import DevData

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
FIX = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"
TARGETS = {"binary_basic.csv": "churn", "multiclass_basic.csv": "species",
           "regression_basic.csv": "price", "binary_small.csv": "target_",
           "regression_skewed.csv": "claim_amount", "multiclass_rare.csv": "color"}  # fmt: skip


@dataclass(frozen=True)
class Prepared:
    loaded: LoadedDataset
    target: str
    task: Task
    split: SplitSpec
    schema: PipelineSchema
    problem: Problem
    dev: DevData
    settings: Settings
    registry: ModelRegistry


@cache
def prepared(
    name: str, profile_name: ExecutionProfileName = ExecutionProfileName.LOCAL
) -> Prepared:
    settings = load_settings({"TABPILOT_PROFILE": profile_name.value})
    target = TARGETS[name]
    ld = load_dataset(FIX / name, name, settings, RID)
    det = detect_task(ld.frame[target].dropna())
    ld = apply_target(ld, validate_target(ld, target, det.task))
    split = build_split(ld.frame[target], det.task, features=ld.frame.drop(columns=[target])).spec
    summary = profile(ld.frame, split, ld.metadata, det.task, execution_profile=profile_name,
                      time_budget_s=960, positive_label=det.positive_label)  # fmt: skip
    if det.task == Task.REGRESSION:
        problem = Problem(Task.REGRESSION)
    else:
        classes = tuple(sorted(str(v) for v in ld.frame[target].dropna().unique()))
        positive = classes.index(det.positive_label) if det.task == Task.BINARY else None
        problem = Problem(det.task, classes, positive)
    dev = DevData.from_frame(ld.frame, target, problem)
    return Prepared(ld, target, det.task, split, schema_from(ld.metadata, summary), problem, dev,
                    settings, ModelRegistry(settings))  # fmt: skip
