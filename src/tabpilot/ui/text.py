"""Safe text for Streamlit (PRD §27, §24.3; P16 "never render a user string as HTML").

Streamlit renders labels, `st.markdown`, `st.error`/`st.info` bodies and captions as Markdown.
`unsafe_allow_html` is never used, so raw HTML is shown as text, but Markdown itself can still
create links, images and formatting from a column name or an LLM rationale. Every user- or
LLM-derived string that reaches a Markdown context goes through `md_escape`; tables, `st.text`
and `st.code` show text verbatim and need no escaping.
"""

from __future__ import annotations

import re

_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$:&=])")


def md_escape(text: object) -> str:
    """Backslash-escape every Markdown-significant character and flatten newlines."""
    flat = " ".join(str(text).split())
    return _SPECIAL.sub(r"\\\1", flat)


def quote(text: object) -> str:
    """An escaped Markdown blockquote (LLM rationales are shown as quoted text, §27.4)."""
    return "> " + md_escape(text)


def duration(seconds: float | None) -> str:
    """`m:ss` for banners and meters (§27.3 "Completed in m:ss")."""
    if seconds is None:
        return "–"
    total = max(0, round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def size(n_bytes: int) -> str:
    """Human-readable byte count (downloads, §27.2)."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024**2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024**2:.1f} MB"
