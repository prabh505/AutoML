"""P15 runtime tests: slots, cleanup, sweeper, `RunManager` lifecycle (PRD §13.7, §24.7, §28).

T-RUN-02 cancel · T-RUN-04 worker crash · T-RUN-05 slot lock · T-SEC-06 cleanup · T-SEC-07 sweeper.
Workers are the controlled fakes of `tests/helpers/fake_workers.py` (real protocol, no ML).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import pytest
from tests.helpers import fake_workers as fw
from ulid import ULID

from tabpilot.config import load_settings
from tabpilot.runtime import InvalidConfigError, RunManager, SlotBusyError, UnknownRunError
from tabpilot.runtime.cleanup import delete_run_dir, sweep
from tabpilot.runtime.manager import pid_exists
from tabpilot.runtime.slots import acquire_slot, slots_in_use
from tabpilot.schemas import ErrorCode, ExecutionProfileName, MetricName, RunConfig, Task

LOCAL = load_settings({"TABPILOT_PROFILE": "local"})
HOSTED = load_settings({"TABPILOT_PROFILE": "hosted"})


def config(profile: ExecutionProfileName = ExecutionProfileName.LOCAL, **kw: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_id": str(ULID()), "target": "t", "task": Task.BINARY,
        "primary_metric": MetricName.ROC_AUC, "time_budget_s": 240, "llm_mode": "off",
        "execution_profile": profile,
    }  # fmt: skip
    return RunConfig(**{**base, **kw})


def manager(tmp_path: Path, worker: Callable[..., None], **kw: Any) -> RunManager:
    mode = kw.pop("mode", "process")
    settings = kw.pop("settings", LOCAL)
    return RunManager(tmp_path / "runs", settings, mode=mode, worker=worker,
                      cancel_grace_s=kw.pop("cancel_grace_s", 1.0),
                      term_grace_s=kw.pop("term_grace_s", 1.0),
                      heartbeat_timeout_s=kw.pop("heartbeat_timeout_s", 20.0), **kw)  # fmt: skip


def start(mgr: RunManager, tmp_path: Path, cfg: RunConfig | None = None) -> str:
    cfg = cfg or config(settings_profile(mgr))
    source = tmp_path / "data.csv"
    source.write_text("a,t\n1,0\n", encoding="utf-8")
    return mgr.start(cfg, input_path=source, display_name="data.csv")


def settings_profile(mgr: RunManager) -> ExecutionProfileName:
    return mgr.settings.profile


def wait_for(predicate: Callable[[], bool], timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------- slots (T-RUN-05)
@pytest.mark.unit
def test_T_RUN_05_slot_lock_allows_only_the_configured_number(tmp_path: Path) -> None:
    first = acquire_slot(tmp_path, 1)
    assert slots_in_use(tmp_path, 1) == 1
    with pytest.raises(SlotBusyError, match="Another analysis is running"):
        acquire_slot(tmp_path, 1)
    first.release()
    first.release()  # idempotent
    assert slots_in_use(tmp_path, 1) == 0
    two = [acquire_slot(tmp_path, 2), acquire_slot(tmp_path, 2)]
    assert {s.index for s in two} == {0, 1}
    with pytest.raises(SlotBusyError):
        acquire_slot(tmp_path, 2)
    for slot in two:
        slot.release()


@pytest.mark.unit
def test_slot_is_released_by_the_os_when_the_holder_dies(tmp_path: Path) -> None:
    code = (
        "import sys, time; from pathlib import Path; "
        "from tabpilot.runtime.slots import acquire_slot; "
        "acquire_slot(Path(sys.argv[1]), 1); print('held', flush=True); time.sleep(120)"
    )
    holder = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                              stdout=subprocess.PIPE, text=True)  # fmt: skip
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(SlotBusyError):
            acquire_slot(tmp_path, 1)
    finally:
        holder.kill()
        holder.wait()
    acquire_slot(tmp_path, 1).release()  # a crashed holder never leaves the slot stuck


@pytest.mark.integration
def test_T_RUN_05_second_start_is_busy_and_same_run_id_is_idempotent(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.polite)
    try:
        cfg = config()
        run_id = start(mgr, tmp_path, cfg)
        assert start(mgr, tmp_path, cfg) == run_id  # idempotent: no second worker
        pid = mgr.worker_pid(run_id)
        assert start(mgr, tmp_path, cfg) == run_id
        assert mgr.worker_pid(run_id) == pid
        with pytest.raises(SlotBusyError):
            start(mgr, tmp_path, config())
        mgr.cancel(run_id)
        start(mgr, tmp_path, config())  # the slot is free again after the first run ended
    finally:
        mgr.shutdown()


@pytest.mark.unit
def test_invalid_configuration_is_rejected_before_any_worker_starts(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.quick_ok, settings=HOSTED)
    with pytest.raises(InvalidConfigError, match="outside 240–1800"):
        start(mgr, tmp_path, config(ExecutionProfileName.HOSTED, time_budget_s=2000))
    with pytest.raises(InvalidConfigError, match="does not match the running profile"):
        start(mgr, tmp_path, config(ExecutionProfileName.LOCAL))
    assert slots_in_use(tmp_path / "runs", 1) == 0
    with pytest.raises(ValueError, match="unknown run mode"):
        RunManager(tmp_path / "x", LOCAL, mode="fork")
    with pytest.raises(UnknownRunError):
        mgr.events("nope")
    with pytest.raises(UnknownRunError):
        mgr.delete("nope")


# ---------------------------------------------------------------- results and events
@pytest.mark.integration
def test_events_result_and_status_of_a_finished_run(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.quick_ok)
    run_id = start(mgr, tmp_path)
    result = mgr.wait(run_id, timeout=30)
    assert result is not None
    assert result.status == "completed"
    assert mgr.result(run_id) == result
    assert mgr.status(run_id).state == "completed"
    seqs = [e.seq for e in mgr.events(run_id)]
    assert seqs == [0, 1]
    assert [e.seq for e in mgr.events(run_id, since_seq=1)] == [1]
    assert mgr.events(run_id, since_seq=5) == []
    assert slots_in_use(tmp_path / "runs", 1) == 0  # slot released at the end


# ---------------------------------------------------------------- T-RUN-02 cancel
@pytest.mark.integration
def test_T_RUN_02_cooperative_cancel_stops_the_worker_and_deletes_the_directory(
    tmp_path: Path,
) -> None:
    mgr = manager(tmp_path, fw.polite)
    run_id = start(mgr, tmp_path)
    pid = mgr.worker_pid(run_id)
    assert pid is not None
    assert wait_for(lambda: mgr.events(run_id) != [])  # the worker is running
    assert (tmp_path / "runs" / run_id).is_dir()
    started = time.monotonic()
    mgr.cancel(run_id)
    assert time.monotonic() - started <= 15  # PRD: stopped within 15 s
    mgr.cancel(run_id)  # repeated cancel is a no-op
    result = mgr.result(run_id)
    assert result is not None
    assert result.status == "cancelled"
    assert not pid_exists(pid)
    assert not (tmp_path / "runs" / run_id).exists()  # §24.7: cancelled → directory gone
    assert mgr.status(run_id).state == "cancelled"
    assert mgr.events(run_id)  # events are kept in memory for the UI


@pytest.mark.integration
def test_cancel_escalates_to_sigterm_then_sigkill_and_leaves_no_orphans(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.stubborn, cancel_grace_s=1.0, term_grace_s=1.0)
    run_id = start(mgr, tmp_path)
    pid = mgr.worker_pid(run_id)
    assert pid is not None
    assert pid_exists(pid)
    assert wait_for(lambda: mgr.events(run_id) != [])
    started = time.monotonic()
    mgr.cancel(run_id)
    waited = time.monotonic() - started
    assert 1.5 < waited <= 15  # the cooperative grace (1 s) and the SIGTERM grace (1 s) passed
    assert not pid_exists(pid)
    assert mgr.worker_exit_code(run_id) == -9  # SIGTERM was ignored, so SIGKILL ended it
    result = mgr.result(run_id)
    assert result is not None
    assert result.status == "failed"  # killed without a result message
    assert result.error_code == ErrorCode.WORKER_LOST


@pytest.mark.integration
def test_sigterm_ends_a_worker_that_ignores_the_cancel_event(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.ignores_cancel, cancel_grace_s=1.0, term_grace_s=3.0)
    run_id = start(mgr, tmp_path)
    pid = mgr.worker_pid(run_id)
    assert wait_for(lambda: mgr.events(run_id) != [])
    mgr.cancel(run_id)
    assert not pid_exists(pid)
    assert mgr.worker_exit_code(run_id) == -15  # ended by SIGTERM, before any SIGKILL
    assert mgr.result(run_id) is not None


@pytest.mark.integration
def test_T_RUN_02_no_orphan_child_processes_after_a_kill(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.spawner, cancel_grace_s=0.5, term_grace_s=0.5)
    run_id = start(mgr, tmp_path)
    assert wait_for(lambda: mgr.events(run_id) != [])
    child_pid = int(mgr.events(run_id)[0].data["child_pid"])  # type: ignore[arg-type]
    worker = mgr.worker_pid(run_id)
    assert pid_exists(child_pid)
    mgr.cancel(run_id)
    assert wait_for(lambda: not pid_exists(child_pid), timeout=10)
    assert not pid_exists(worker)
    survivors = [p for p in psutil.Process().children(recursive=True)
                 if p.pid in (child_pid, worker)]  # fmt: skip
    assert survivors == []


@pytest.mark.integration
def test_thread_mode_cancels_cooperatively(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.polite, mode="thread")
    run_id = start(mgr, tmp_path)
    assert mgr.worker_pid(run_id) is None
    assert wait_for(lambda: mgr.events(run_id) != [])
    mgr.cancel(run_id)
    result = mgr.result(run_id)
    assert result is not None
    assert result.status == "cancelled"


# ---------------------------------------------------------------- T-RUN-04 crash and hangs
@pytest.mark.integration
def test_T_RUN_04_worker_crash_is_e_worker_lost_and_cleaned_up(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.crash)
    run_id = start(mgr, tmp_path)
    result = mgr.wait(run_id, timeout=30)
    assert result is not None
    assert result.status == "failed"
    assert result.error_code == ErrorCode.WORKER_LOST
    assert "stopped unexpectedly" in (result.error_message or "")
    assert not (tmp_path / "runs" / run_id).exists()
    assert [e.type.value for e in mgr.events(run_id)] == ["stage_started"]
    assert slots_in_use(tmp_path / "runs", 1) == 0
    start(mgr, tmp_path, config())  # the slot is usable again
    mgr.shutdown()


@pytest.mark.integration
def test_missing_heartbeats_mark_the_run_lost(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.silent, heartbeat_timeout_s=1.5)
    run_id = start(mgr, tmp_path)
    pid = mgr.worker_pid(run_id)
    result = mgr.wait(run_id, timeout=30)
    assert result is not None
    assert result.error_code == ErrorCode.WORKER_LOST
    assert not pid_exists(pid)


@pytest.mark.integration
def test_hard_wall_clock_kill_at_130_percent(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.stubborn, hard_kill_factor=0.005)  # 240 s × 0.005 = 1.2 s
    run_id = start(mgr, tmp_path)
    pid = mgr.worker_pid(run_id)
    started = time.monotonic()
    result = mgr.wait(run_id, timeout=30)
    assert result is not None
    assert result.error_code == ErrorCode.WORKER_LOST
    assert time.monotonic() - started < 15
    assert not pid_exists(pid)


# ---------------------------------------------------------------- T-SEC-06 cleanup lifecycle
@pytest.mark.integration
def test_T_SEC_06_directory_state_per_outcome(tmp_path: Path) -> None:
    ok = manager(tmp_path / "ok", fw.quick_ok)
    run_id = start(ok, tmp_path)
    ok.wait(run_id, timeout=30)
    directory = tmp_path / "ok" / "runs" / run_id
    assert (directory / "bundle.zip").is_file()  # completed: bundle kept
    failed = manager(tmp_path / "bad", fw.quick_fail)
    bad_id = start(failed, tmp_path)
    bad = failed.wait(bad_id, timeout=30)
    assert bad is not None
    assert bad.error_code == ErrorCode.NO_VALID_MODEL
    assert not (tmp_path / "bad" / "runs" / bad_id).exists()  # failed: gone
    keep = manager(tmp_path / "keep", fw.quick_fail, delete_failed=False)
    kept_id = start(keep, tmp_path)
    keep.wait(kept_id, timeout=30)
    assert (tmp_path / "keep" / "runs" / kept_id).is_dir()  # the CLI keeps records for debugging


@pytest.mark.integration
def test_delete_now_removes_the_directory_and_is_repeatable(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.quick_ok)
    run_id = start(mgr, tmp_path)
    mgr.wait(run_id, timeout=30)
    assert (tmp_path / "runs" / run_id).is_dir()
    mgr.delete(run_id)
    mgr.delete(run_id)  # no-op
    assert not (tmp_path / "runs" / run_id).exists()
    assert mgr.status(run_id).state == "deleted"


@pytest.mark.integration
def test_delete_of_a_running_run_cancels_it_first(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.polite)
    run_id = start(mgr, tmp_path)
    pid = mgr.worker_pid(run_id)
    assert wait_for(lambda: mgr.events(run_id) != [])
    mgr.delete(run_id)
    assert not pid_exists(pid)
    assert not (tmp_path / "runs" / run_id).exists()


@pytest.mark.unit
def test_delete_run_dir_reports_failure_for_the_sweeper_to_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "run"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "f").write_text("x", encoding="utf-8")
    real = os.rmdir

    def refuse(path: Any, *a: Any, **k: Any) -> None:
        if Path(path).name == "sub":
            raise PermissionError("busy")
        real(path, *a, **k)

    monkeypatch.setattr(os, "rmdir", refuse)
    assert delete_run_dir(target) is False  # logged, left for the sweeper
    monkeypatch.setattr(os, "rmdir", real)
    assert delete_run_dir(target) is True
    assert delete_run_dir(target) is True  # already gone


# ---------------------------------------------------------------- T-SEC-07 sweeper
def aged(root: Path, name: str, age_s: float, now: float) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "f").write_text("x", encoding="utf-8")
    os.utime(path, (now - age_s, now - age_s))
    return path


@pytest.mark.unit
def test_T_SEC_07_sweeper_removes_old_directories_only(tmp_path: Path) -> None:
    now = 1_000_000.0
    aged(tmp_path, "old", 3 * 3600, now)
    aged(tmp_path, "fresh", 60, now)
    (tmp_path / ".locks").mkdir()
    (tmp_path / "stray.txt").write_text("keep", encoding="utf-8")
    removed = sweep(tmp_path, 2 * 3600, now=lambda: now)
    assert removed == ["old"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [".locks", "fresh", "stray.txt"]
    assert sweep(tmp_path / "missing", 10, now=lambda: now) == []


@pytest.mark.integration
def test_T_SEC_07_manager_sweep_uses_the_profile_age_and_spares_active_runs(
    tmp_path: Path,
) -> None:
    now = [time.time()]
    mgr = manager(tmp_path, fw.polite, settings=HOSTED, wall=lambda: now[0])
    old = aged(tmp_path / "runs", "stale", 3 * 3600, now[0])
    cfg = config(ExecutionProfileName.HOSTED)
    run_id = start(mgr, tmp_path, cfg)
    active = tmp_path / "runs" / run_id
    assert wait_for(lambda: active.is_dir())
    os.utime(active, (now[0] - 5 * 3600, now[0] - 5 * 3600))  # old by mtime, but running
    assert mgr.sweep() == ["stale"]
    assert not old.exists()
    assert active.is_dir()
    mgr.cancel(run_id)
    local = manager(tmp_path / "l", fw.quick_ok, wall=lambda: now[0])
    aged(tmp_path / "l" / "runs", "week_old", 6 * 86400, now[0])
    aged(tmp_path / "l" / "runs", "eight_days", 8 * 86400, now[0])
    assert local.sweep() == ["eight_days"]  # local keeps runs for 7 days


@pytest.mark.integration
def test_idle_ttl_deletes_finished_hosted_runs_after_30_minutes(tmp_path: Path) -> None:
    now = [1_000.0]
    mgr = manager(tmp_path, fw.quick_ok, settings=HOSTED, wall=lambda: now[0])
    run_id = start(mgr, tmp_path, config(ExecutionProfileName.HOSTED))
    mgr.wait(run_id, timeout=30)
    directory = tmp_path / "runs" / run_id
    now[0] += 29 * 60
    assert mgr.reap_idle() == []
    assert directory.is_dir()
    now[0] += 2 * 60
    assert mgr.reap_idle() == [run_id]
    assert not directory.exists()
    local = manager(tmp_path / "local", fw.quick_ok, wall=lambda: now[0])
    local_id = start(local, tmp_path)
    local.wait(local_id, timeout=30)
    now[0] += 10 * 3600
    assert local.reap_idle() == []  # no idle TTL on the local profile


@pytest.mark.integration
def test_start_sweeper_sweeps_immediately_and_reschedules(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.quick_ok)
    aged(tmp_path / "runs", "ancient", 30 * 86400, time.time())
    mgr.start_sweeper()
    try:
        assert not (tmp_path / "runs" / "ancient").exists()
        assert mgr._sweeper is not None
        assert mgr._sweeper.is_alive()
    finally:
        mgr.shutdown()


@pytest.mark.unit
def test_validate_upload_stores_the_file_under_a_new_run_id(tmp_path: Path) -> None:
    import io  # noqa: PLC0415

    mgr = manager(tmp_path, fw.quick_ok)
    first = mgr.validate_upload(io.BytesIO(b"a,t\n1,0\n2,1\n"), "my data.csv")
    second = mgr.validate_upload(io.BytesIO(b"a,t\n1,0\n"), "../../evil.csv")
    assert first.run_id != second.run_id
    assert first.display_name == "my data.csv"
    assert second.display_name == "evil.csv"
    (stored,) = mgr.run_dir(first.run_id).glob("input/raw.*")
    assert stored.read_bytes() == b"a,t\n1,0\n2,1\n"
    assert first.metadata.n_rows_raw == 2


# ---------------------------------------------------------------- U2 KI-034: §28.2 contract
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def upload(mgr: RunManager, relative: str) -> Any:
    path = FIXTURE_DIR / relative
    with path.open("rb") as handle:
        return mgr.validate_upload(handle, path.name)


@pytest.mark.integration
def test_U2_KI_034_validate_upload_parses_and_returns_metadata_and_warnings(tmp_path: Path) -> None:
    mgr = manager(tmp_path / "custom_root_name", fw.quick_ok)  # not named "runs"
    result = upload(mgr, "datasets/binary_basic.csv")
    assert result.metadata.n_rows_raw == 2500
    assert result.metadata.run_id == result.run_id
    assert result.display_name == "binary_basic.csv"
    assert isinstance(result.warnings, list)
    stored = list(mgr.run_dir(result.run_id).glob("input/raw.*"))
    assert len(stored) == 1  # stored under the manager's own root, not <root.parent>/runs
    cfg = config(run_id=result.run_id, target="churn")
    assert mgr.start(cfg, display_name=result.display_name) == result.run_id  # found its input
    mgr.wait(result.run_id, timeout=30)
    assert upload(mgr, "datasets/binary_basic.csv").run_id != result.run_id  # new id per call


@pytest.mark.integration
@pytest.mark.parametrize(
    ("relative", "code"),
    [("malicious/zip_renamed.csv", ErrorCode.ARCHIVE_UNSUPPORTED),
     ("malicious/corrupt.parquet", ErrorCode.FILE_CORRUPT)],
)  # fmt: skip
def test_U2_KI_034_rejected_upload_leaves_nothing_on_disk(
    tmp_path: Path, relative: str, code: ErrorCode
) -> None:
    from tabpilot.schemas import TabPilotError  # noqa: PLC0415

    mgr = manager(tmp_path, fw.quick_ok)
    with pytest.raises(TabPilotError) as err:
        upload(mgr, relative)
    assert err.value.code in (code, ErrorCode.FILE_CORRUPT, ErrorCode.PARSE_MALFORMED)
    assert [p for p in (tmp_path / "runs").iterdir() if p.name != ".locks"] == []


@pytest.mark.integration
def test_U2_KI_034_detect_task_and_estimate_are_pure_calls(tmp_path: Path) -> None:
    mgr = manager(tmp_path, fw.quick_ok)
    up = upload(mgr, "datasets/binary_basic.csv")
    detected = mgr.detect_task(up.run_id, "Churn")  # the original column name works
    assert detected.task == Task.BINARY
    assert mgr.detect_task(up.run_id, "churn").task == Task.BINARY
    with pytest.raises(ValueError, match="not in the file"):
        mgr.detect_task(up.run_id, "nope")
    report = mgr.estimate(up.run_id, config(run_id=up.run_id, target="churn"))
    assert report.status in ("ok", "simplified")
    assert report.estimated_peak_mb > 0
    assert mgr.estimate(up.run_id, config(run_id=up.run_id, target="churn")) == report


@pytest.mark.integration
def test_U2_KI_034_preview_planner_input_is_cached_and_byte_identical_to_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-PRIV-02 through the manager: the previewed JSON is what the planner sends."""
    from tabpilot.llm.stub import StubProvider  # noqa: PLC0415
    from tabpilot.orchestration.context import WorkerOptions  # noqa: PLC0415
    from tabpilot.runtime import worker as run_worker  # noqa: PLC0415
    from tabpilot.tracking import dumps  # noqa: PLC0415

    sent: list[Any] = []
    real = StubProvider.complete

    def spy(self: StubProvider, req: Any) -> Any:
        if req.purpose == "plan":
            sent.append(req.user_json)
        return real(self, req)

    monkeypatch.setattr(StubProvider, "complete", spy)
    mgr = manager(tmp_path, run_worker.worker_main, mode="thread")
    up = upload(mgr, "datasets/binary_basic.csv")
    cfg = config(run_id=up.run_id, target="churn", llm_mode="stub")
    preview = mgr.preview_planner_input(up.run_id, cfg)
    assert mgr.preview_planner_input(up.run_id, cfg) is preview  # cached by (run, config)
    assert not [p for p in mgr.root.iterdir() if p.name.startswith(".preview")]  # scratch removed
    mgr.start(cfg, display_name=up.display_name,
              options=WorkerOptions(tuning_mode="trials", trials_per_model=1))  # fmt: skip
    mgr.wait(up.run_id, timeout=300)
    assert dumps(sent[0], indent=None) == dumps(__import__("json").loads(preview), indent=None)
    assert "Churn" not in preview  # the target's raw label never leaves (names are normalized)
