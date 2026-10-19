"""Provider access (PRD §23, FR-LLM-*).

Interface: LLMRouter.complete(req, budget). Depends on schemas, config.

Must not: DataFrames / pandas (payloads are dicts); logging prompt contents on hosted.
"""
