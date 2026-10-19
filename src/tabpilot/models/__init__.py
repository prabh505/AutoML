"""Model zoo registry and factories (PRD §16).

Interface: registry.available(profile), make_estimator(name, params, task).

Must not: —
"""

from tabpilot.models.registry import (
    FACTORIES,
    ModelRegistry,
    class_weight_params,
    default_class_weight,
)

__all__ = ["FACTORIES", "ModelRegistry", "class_weight_params", "default_class_weight"]
