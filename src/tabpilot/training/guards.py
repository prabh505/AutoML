"""Pre-fit resource guards (PRD §16.3, §29.4, §29.6)."""

from __future__ import annotations

from collections.abc import Callable

from tabpilot.resources.estimator import RF_TREES, rf_model_mb
from tabpilot.schemas import CandidateSpec, ExecutionProfileName, ModelName

RF_BUDGET_SHARE = 0.40


def rf_resource_guard(
    profile: ExecutionProfileName, train_rows: int, rss_budget_mb: float
) -> Callable[[CandidateSpec], str | None]:
    """On hosted, skip random forest when its model estimate exceeds 40% of the RSS budget
    (§16.3). The returned callable gives the skip reason, or None to proceed.
    """

    def guard(candidate: CandidateSpec) -> str | None:
        if (
            profile != ExecutionProfileName.HOSTED
            or candidate.model.name != ModelName.RANDOM_FOREST
        ):
            return None
        trees = candidate.model.params.get("n_estimators", RF_TREES)
        estimate = rf_model_mb(
            train_rows, int(trees if isinstance(trees, int | float) else RF_TREES)
        )
        if estimate > RF_BUDGET_SHARE * rss_budget_mb:
            return (f"random_forest estimate {estimate:.0f} MB exceeds "
                    f"{RF_BUDGET_SHARE:.0%} of the {rss_budget_mb:.0f} MB budget")  # fmt: skip
        return None

    return guard
