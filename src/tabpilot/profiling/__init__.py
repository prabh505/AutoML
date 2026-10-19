"""Statistics + compact profile (PRD §9, FR-PRF-*).

Interface: profile(dev, meta) -> ProfileSummary; compact(profile) -> CompactProfile. Depends on
schemas.

Must not: HoldoutGuard, LLM; reading HOLD rows; raw values in CompactProfile.
"""

from tabpilot.profiling.compact import build_compact_profile, canonical_json
from tabpilot.profiling.leakage import apply_leakage_exclusions
from tabpilot.profiling.profile import profile

__all__ = ["apply_leakage_exclusions", "build_compact_profile", "canonical_json", "profile"]
