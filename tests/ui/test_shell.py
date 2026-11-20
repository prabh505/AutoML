"""App shell (P01 test, updated in P16): the entrypoint renders the Upload page without errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tabpilot.ui.app import TITLE

APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"


@pytest.mark.ui
def test_entrypoint_renders_upload_page(make_ui: object) -> None:
    ui = make_ui()  # type: ignore[operator]
    at = ui.run()
    assert not at.exception
    assert [t.value for t in at.title] == [TITLE]
    assert sum(b.label == "Use this sample" for b in at.button) == 6
    assert AppTest  # the shell test uses AppTest through the shared harness
