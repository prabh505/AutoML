"""Reflection proposals and paired-fold acceptance (PRD §19, FR-RFL-01).

Interface: reflect(...), accept(...). Depends on llm, planning (validator).

Must not: HoldoutGuard; exceeding round/time/trial budgets.
"""
