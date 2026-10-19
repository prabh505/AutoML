"""Standalone runtime package for every custom class pickled into model.joblib (amendment A-001,
ADR-0002).

Shipped inside export bundles; imports only numpy, pandas, scikit-learn and model libraries.

Must not: Importing tabpilot (import contract R5).
"""
