"""P01 UI shell: the entrypoint renders the title and build status (replaced in P16)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tabpilot.ui.app import STATUS, TITLE

APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"


@pytest.mark.ui
def test_entrypoint_renders_shell() -> None:
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    assert [t.value for t in at.title] == [TITLE]
    assert STATUS in [m.value for m in at.markdown]
