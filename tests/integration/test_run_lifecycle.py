"""P15 lifecycle tests with the real worker: cleanup per outcome, process cancel, log hygiene.

T-SEC-06 (real runs) · T-RUN-02 (real process) · T-PRIV-04 (end-to-end log scan) · T-SEC key scan.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import pytest
from tests.helpers.run_harness import FIXTURES, go

from tabpilot.cli import RunArgs, build_config
from tabpilot.config import load_settings
from tabpilot.ingestion import accept_upload
from tabpilot.llm.stub import StubProvider
from tabpilot.orchestration.context import WorkerOptions
from tabpilot.runtime import RunManager
from tabpilot.runtime.manager import pid_exists
from tabpilot.schemas import ErrorCode, ExecutionProfileName, ModelName

LOCAL = load_settings({"TABPILOT_PROFILE": "local"})
SECRET = "sk-live-SUPERSECRET-0123456789"  # noqa: S105 - a canary, not a real key


def real_run(tmp_path: Path, mode: str, file: str = "binary_small.csv", target: str = "target",
             **manager_kw: Any) -> tuple[RunManager, str]:  # fmt: skip
    """Start a real worker (thread or process) through the manager; returns (manager, run_id)."""
    source = FIXTURES / file
    args = RunArgs(file=source, target=target, out=tmp_path / "unused", tuning_mode="trials",
                   trials=2, models=(ModelName.LINEAR, ModelName.LIGHTGBM))  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="tabpilot-test-upload-") as staging:
        with source.open("rb") as handle:
            stored = accept_upload(handle, source.name,
                                   LOCAL.model_copy(update={"data_dir": Path(staging)}))  # fmt: skip
        config = build_config(args, LOCAL, stored)
        mgr = RunManager(tmp_path / "runs", LOCAL, mode=mode, **manager_kw)
        input_path = mgr.run_dir(config.run_id) / "input" / stored.path.name  # as the UI stores it
        input_path.parent.mkdir(parents=True)
        shutil.copyfile(stored.path, input_path)
    options = WorkerOptions(tuning_mode="trials", trials_per_model=2, models=args.models)
    mgr.start(config, input_path=input_path, display_name=stored.display_name, options=options)
    return mgr, config.run_id


def wait_for(predicate: Any, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------- T-SEC-06 with real runs
@pytest.mark.integration
def test_T_SEC_06_completed_real_run_keeps_the_bundle_and_drops_the_input(tmp_path: Path) -> None:
    mgr, run_id = real_run(tmp_path, "thread")
    result = mgr.wait(run_id, timeout=300)
    assert result is not None
    assert result.status == "completed"
    directory = tmp_path / "runs" / run_id
    assert not list(directory.rglob("raw.*"))  # raw input deleted right after export (§24.7)
    assert not (directory / "input").exists()
    assert not list(directory.rglob("normalized.parquet"))
    assert result.final is not None
    assert (directory / result.final.bundle_path).is_file()  # type: ignore[arg-type]
    assert (directory / "report" / "report.html").is_file()
    mgr.delete(run_id)
    assert not directory.exists()


@pytest.mark.integration
def test_T_SEC_06_failed_real_run_leaves_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyRam:
        total = 64 * 2**20

    monkeypatch.setattr(psutil, "virtual_memory", TinyRam)
    mgr, run_id = real_run(tmp_path, "thread")
    result = mgr.wait(run_id, timeout=120)
    assert result is not None
    assert result.status == "failed"
    assert result.error_code == ErrorCode.RESOURCE_INFEASIBLE
    assert "even after simplifications" in (result.error_message or "")
    assert not (tmp_path / "runs" / run_id).exists()  # whole run directory deleted
    assert mgr.events(run_id)  # the sanitized record stays in memory for display


@pytest.mark.integration
def test_T_SEC_06_cancelled_real_run_leaves_no_directory(tmp_path: Path) -> None:
    mgr, run_id = real_run(tmp_path, "thread")
    assert wait_for(lambda: any(e.type.value == "candidate_started" for e in mgr.events(run_id)),
                    120)  # fmt: skip
    mgr.cancel(run_id)
    result = mgr.result(run_id)
    assert result is not None
    assert result.status == "cancelled"
    assert not (tmp_path / "runs" / run_id).exists()


# ---------------------------------------------------------------- T-RUN-02 with a real process
@pytest.mark.integration
def test_T_RUN_02_real_worker_process_is_gone_within_15_seconds_of_cancel(tmp_path: Path) -> None:
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    mgr, run_id = real_run(tmp_path, "process")
    try:
        pid = mgr.worker_pid(run_id)
        assert pid is not None
        assert pid_exists(pid)
        assert wait_for(lambda: any(e.type.value == "candidate_started" for e in mgr.events(run_id)),
                        180)  # fmt: skip
        started = time.monotonic()
        mgr.cancel(run_id)
        assert time.monotonic() - started <= 15
        assert not pid_exists(pid)
        result = mgr.result(run_id)
        assert result is not None
        assert result.status in ("cancelled", "failed")
        if result.status == "failed":  # killed before it could say so: still a coded failure
            assert result.error_code == ErrorCode.WORKER_LOST
        assert not (tmp_path / "runs" / run_id).exists()
        workers = [p for p in psutil.Process().children(recursive=True)
                   if p.pid not in before and "multiprocessing.spawn" in " ".join(p.cmdline())
                   and p.status() != psutil.STATUS_ZOMBIE]  # fmt: skip
        assert workers == []  # (Python's own resource-tracker helper is not a worker)
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------- T-PRIV-04 and key hygiene
def data_markers() -> tuple[set[str], set[str]]:
    """Category labels (≥ 6 chars), identifiers and distinctive cell values of the fixture."""
    frame = pd.read_csv(FIXTURES / "binary_basic.csv", dtype=str)
    labels = {v for col in ("Contract", "Region") for v in frame[col].dropna().unique()}
    cells = set(frame["Customer ID"]) | set(frame["Income"].dropna().head(200))
    return {v for v in labels if len(v) >= 6}, cells


@pytest.mark.integration
def test_T_PRIV_04_logs_records_and_llm_traffic_hold_no_cell_values_or_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    sent: list[str] = []
    real = StubProvider.complete

    def spy(self: StubProvider, req: Any) -> Any:
        sent.append(req.system + "\n" + str(req.user_json))
        return real(self, req)

    monkeypatch.setattr(StubProvider, "complete", spy)
    with caplog.at_level(logging.DEBUG):
        outcome = go(tmp_path, "binary_basic.csv", "Churn", llm="stub")
    assert outcome.result.status == "completed"
    labels, cells = data_markers()
    streams = {"logs": caplog.text + "".join(capsys.readouterr()), "llm traffic": "\n".join(sent)}
    streams |= {p.name: p.read_text("utf-8") for p in (outcome.run_dir / "records").glob("*")}
    for name in ("leaderboard.json", "selection.json", "holdout.json", "split_spec.json"):
        streams[name] = (outcome.run_dir / name).read_text("utf-8")
    assert streams["llm traffic"]  # the LLM path really ran
    for where, text in streams.items():
        assert SECRET not in text, where
        assert not [v for v in labels if v in text], (where, [v for v in labels if v in text][:3])
        found = [v for v in cells if re.search(rf"(?<![\w.]){re.escape(v)}(?![\w.])", text)]
        assert not found, (where, found[:3])
    # the bundle carries no key either (text members only; the model is binary)
    bundle = outcome.run_dir / outcome.result.final.bundle_path  # type: ignore[union-attr]
    with zipfile.ZipFile(bundle) as zf:
        for member in zf.namelist():
            if member.endswith((".json", ".csv", ".md", ".html", ".py", ".txt", ".jsonl")):
                assert SECRET not in zf.read(member).decode("utf-8", "replace"), member


@pytest.mark.integration
def test_llm_call_records_hold_hashes_and_sizes_never_bodies(tmp_path: Path) -> None:
    outcome = go(tmp_path, llm="stub")
    text = (outcome.run_dir / "records" / "llm_calls.jsonl").read_text("utf-8")
    assert "request_sha256" in text
    assert "response_sha256" in text
    for body_marker in ("available_models", "paragraphs", "dsl_catalog", "leaderboard_top"):
        assert body_marker not in text  # no prompt or response content (§30.3)
    assert ExecutionProfileName.LOCAL.value  # profile enum is importable for hosted variants
