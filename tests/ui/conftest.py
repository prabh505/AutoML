"""AppTest harness for the Streamlit UI (P16).

Each test gets its own `RunManager` (thread mode, temporary data dir, local profile: in thread
mode the worker shares the pytest process, whose RSS would trip the hosted 750 MB limit late in
a full run) injected in place of the cached app manager, the offline stub LLM option, and fast
worker options. Only
external boundaries are replaced: the browser upload widget (AppTest cannot drive
`st.file_uploader`), and, where a test says so, the worker.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from tabpilot.config import load_settings
from tabpilot.orchestration.context import WorkerOptions
from tabpilot.runtime import RunManager
from tabpilot.ui import services

APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"
FAST = WorkerOptions(tuning_mode="trials", trials_per_model=1, models=("linear", "lightgbm"))


class FakeUpload(io.BytesIO):
    """What `st.file_uploader` returns: a binary buffer with a name and a file id."""

    def __init__(self, data: bytes, name: str, file_id: str) -> None:
        super().__init__(data)
        self.name = name
        self.file_id = file_id


@dataclass
class UploadSlot:
    """The file the patched uploader currently 'holds' (None: nothing chosen)."""

    file: FakeUpload | None = None
    count: int = 0

    def put(self, data: bytes, name: str) -> None:
        self.count += 1
        self.file = FakeUpload(data, name, f"file-{self.count}")

    def __call__(self, *args: Any, **kwargs: Any) -> FakeUpload | None:
        if self.file is not None:
            self.file.seek(0)
        return self.file


@dataclass
class UI:
    """One app under test with its manager and upload slot."""

    at: AppTest
    manager: RunManager
    upload: UploadSlot
    root: Path
    captured: list[Any] = field(default_factory=list)

    def run(self) -> AppTest:
        return self.at.run()

    def page(self) -> str:
        return str(self.at.session_state["page"])

    def wait_until(self, done: Callable[[], bool], timeout: float = 240.0) -> None:
        end = time.monotonic() + timeout
        while not done():
            if time.monotonic() > end:
                raise TimeoutError("the app did not reach the expected state")
            time.sleep(1.0)
            self.at.run()

    def texts(self) -> str:
        """Every text-like element value on the page with Markdown escapes removed."""
        parts: list[str] = []
        for kind in ("markdown", "caption", "text", "error", "warning", "info", "success",
                     "code", "header", "subheader", "title"):  # fmt: skip
            parts += [str(e.value) for e in getattr(self.at, kind)]
        return re.sub(r"\\(.)", r"\1", "\n".join(parts))  # undo Markdown escaping


@pytest.fixture
def make_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., UI]]:
    managers: list[RunManager] = []

    def build(worker: Callable[..., None] | None = None, **kwargs: Any) -> UI:
        settings = load_settings({"TABPILOT_PROFILE": "local"}).model_copy(
            update={"data_dir": tmp_path / "data"}
        )
        extra = {"worker": worker} if worker is not None else {}
        manager = RunManager(tmp_path / "data" / "runs", settings, mode="thread",
                             cancel_grace_s=2.0, term_grace_s=1.0, **extra, **kwargs)  # fmt: skip
        managers.append(manager)
        slot = UploadSlot()
        monkeypatch.setattr(services, "manager", lambda: manager)
        monkeypatch.setattr(services, "worker_options", FAST)
        monkeypatch.setattr(streamlit, "file_uploader", slot)
        monkeypatch.setenv("TABPILOT_LLM_STUB", "1")
        at = AppTest.from_file(str(APP), default_timeout=120)
        return UI(at, manager, slot, tmp_path / "data")

    yield build
    for manager in managers:
        manager.shutdown()
