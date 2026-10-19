"""Statistics + compact profile (PRD §9, FR-PRF-*).

Interface: profile(dev, meta) -> ProfileSummary; compact(profile) -> CompactProfile. Depends on
schemas.

Must not: HoldoutGuard, LLM; reading HOLD rows; raw values in CompactProfile.
"""
