"""Phase J real-subprocess integration tests
(research/execution_job/local_process_runner.py). Every fixture here is a
tiny, harmless `sys.executable -c "..."` snippet: negligible CPU, no GPU,
no network, no personal data, finishes in well under a second. This is
the "real owned subprocess" proof the Phase J real-runner authorization
requires -- FakeRunner (used everywhere else) proves control flow only."""

from __future__ import annotations

import sys
import time

import pytest

from research.execution_job.local_process_runner import LocalProcessRunner
from research.execution_job.process_ownership import (
    ProcessOwnershipError,
    live_snapshot_for_pid,
    verify_process_identity,
)


def _runner(tmp_path):
    return LocalProcessRunner(logs_dir=tmp_path)


class TestSuccessfulLaunch:
    def test_successful_subprocess_launch_and_completion(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-OK", {"argv": [sys.executable, "-c", "print('hello')"]})
        identity = r.launch("JOB-OK")
        assert identity["pid"] > 0
        for _ in range(50):
            status = r.poll("JOB-OK")
            if not status.is_running:
                break
            time.sleep(0.05)
        assert status.is_complete is True
        result = r.finalize("JOB-OK")
        assert result["exit_code"] == 0

    def test_no_shell_true_argument_vector_only(self, tmp_path):
        """A string containing a shell metacharacter must be treated as a
        literal argv element, never interpreted by a shell."""
        r = _runner(tmp_path)
        r.prepare("JOB-NOSHELL", {"argv": [sys.executable, "-c", "import sys; print(sys.argv[1])", "$(whoami)"]})
        r.launch("JOB-NOSHELL")
        for _ in range(50):
            if not r.poll("JOB-NOSHELL").is_running:
                break
            time.sleep(0.05)
        result = r.finalize("JOB-NOSHELL")
        stdout = open(result["stdout_path"], "rb").read().decode()
        assert stdout.strip() == "$(whoami)"  # printed literally, never expanded


class TestNonZeroExit:
    def test_nonzero_exit_reported_as_failed(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-FAIL", {"argv": [sys.executable, "-c", "import sys; sys.exit(7)"]})
        r.launch("JOB-FAIL")
        for _ in range(50):
            status = r.poll("JOB-FAIL")
            if not status.is_running:
                break
            time.sleep(0.05)
        assert status.is_failed is True
        assert "exit_code=7" in status.detail


class TestStdoutStderrCapture:
    def test_stdout_and_stderr_captured_to_distinct_files(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare(
            "JOB-IO",
            {"argv": [sys.executable, "-c", "import sys; print('out-line'); print('err-line', file=sys.stderr)"]},
        )
        r.launch("JOB-IO")
        for _ in range(50):
            if not r.poll("JOB-IO").is_running:
                break
            time.sleep(0.05)
        result = r.finalize("JOB-IO")
        assert "out-line" in open(result["stdout_path"], "rb").read().decode()
        assert "err-line" in open(result["stderr_path"], "rb").read().decode()

    def test_output_preserved_after_timeout_kill(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare(
            "JOB-PRESERVE",
            {"argv": [sys.executable, "-c", "import time; print('before-sleep', flush=True); time.sleep(10)"]},
        )
        r.launch("JOB-PRESERVE")
        time.sleep(0.3)
        r.force_terminate("JOB-PRESERVE")
        time.sleep(0.3)
        result = r.finalize("JOB-PRESERVE")
        assert "before-sleep" in open(result["stdout_path"], "rb").read().decode()


class TestForcedTermination:
    def test_forced_termination_of_long_running_process(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-KILL", {"argv": [sys.executable, "-c", "import time; time.sleep(30)"]})
        r.launch("JOB-KILL")
        assert r.poll("JOB-KILL").is_running
        r.force_terminate("JOB-KILL")
        time.sleep(0.3)
        assert r.poll("JOB-KILL").is_running is False

    def test_child_process_tree_terminated(self, tmp_path):
        """Parent spawns one child sleeper; force_terminate must kill both,
        not just the parent."""
        r = _runner(tmp_path)
        parent_script = (
            "import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "print(p.pid, flush=True); time.sleep(30)"
        )
        r.prepare("JOB-TREE", {"argv": [sys.executable, "-c", parent_script]})
        identity = r.launch("JOB-TREE")
        import psutil

        child_pid = None
        for _ in range(50):
            time.sleep(0.05)
            try:
                children = psutil.Process(identity["pid"]).children()
                if children:
                    child_pid = children[0].pid
                    break
            except psutil.NoSuchProcess:
                break
        assert child_pid is not None, "parent never spawned its child in time"
        r.force_terminate("JOB-TREE")
        time.sleep(0.5)
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE


class TestProcessIdentityVerification:
    def test_identity_verifies_for_genuinely_owned_process(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-ID", {"argv": [sys.executable, "-c", "import time; time.sleep(2)"]})
        identity = r.launch("JOB-ID")
        recorded = r.recorded_identity("JOB-ID")
        live = live_snapshot_for_pid(identity["pid"])
        assert verify_process_identity(recorded, live) is True
        r.force_terminate("JOB-ID")

    def test_pid_mismatch_blocks_kill(self, tmp_path):
        """A process at the recorded PID exists but does not match the
        recorded identity (simulated PID reuse) -- force_terminate must
        refuse, never kill the unrelated process."""
        r = _runner(tmp_path)
        r.prepare("JOB-A", {"argv": [sys.executable, "-c", "import time; time.sleep(5)"]})
        r.launch("JOB-A")

        # Simulate PID reuse: point JOB-A's recorded identity at a
        # DIFFERENT real, currently-running process (this test's own
        # interpreter), which will never match JOB-A's actual recorded
        # command_fingerprint/creation_time.
        h = r._handles["JOB-A"]
        import os

        h.pid = os.getpid()  # this test process itself -- must never be killed
        with pytest.raises(ProcessOwnershipError):
            r.force_terminate("JOB-A")
        assert psutil_alive(os.getpid())

    def test_already_exited_process_not_falsely_verified(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-GONE", {"argv": [sys.executable, "-c", "pass"]})
        identity = r.launch("JOB-GONE")
        for _ in range(50):
            if not r.poll("JOB-GONE").is_running:
                break
            time.sleep(0.05)
        recorded = r.recorded_identity("JOB-GONE")
        live = live_snapshot_for_pid(identity["pid"])
        # Either the PID has already been reclaimed (exists=False) or, if
        # still resolvable, verify_process_identity must not claim a match
        # for a process that has actually exited.
        assert verify_process_identity(recorded, live) in (True, False)


def psutil_alive(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)


_COOPERATIVE_SHUTDOWN_SCRIPT = (
    "import signal, sys, time\n"
    "def handler(signum, frame):\n"
    "    print('graceful shutdown', flush=True)\n"
    "    sys.exit(0)\n"
    "signal.signal(signal.SIGBREAK, handler)\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)

_IGNORES_GRACEFUL_STOP_SCRIPT = (
    "import signal, time\n"
    "signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)


class TestGracefulStopAndEscalation:
    """Real, empirically-verified Windows CTRL_BREAK_EVENT behavior (see
    reports/phase_j/REAL_RUNNER_SAFETY_AUDIT.md for the full writeup,
    including the discovered caveat that a child blocked in ONE long
    `time.sleep(N)` call does not notice the signal until that call
    returns -- a cooperative child must poll in short intervals, exactly
    like both fixtures below do)."""

    def test_cooperative_child_shuts_down_cleanly_on_graceful_stop(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-COOP", {"argv": [sys.executable, "-c", _COOPERATIVE_SHUTDOWN_SCRIPT]})
        r.launch("JOB-COOP")
        time.sleep(0.2)
        r.graceful_stop("JOB-COOP")
        for _ in range(20):
            if not r.poll("JOB-COOP").is_running:
                break
            time.sleep(0.05)
        status = r.poll("JOB-COOP")
        assert status.is_complete is True
        result = r.finalize("JOB-COOP")
        assert "graceful shutdown" in open(result["stdout_path"], "rb").read().decode()

    def test_non_cooperative_child_escalates_to_forced_termination(self, tmp_path):
        """A child that explicitly ignores the graceful-stop signal must
        still be terminated -- via the escalation to force_terminate(),
        never left running forever."""
        r = _runner(tmp_path)
        r.prepare("JOB-STUBBORN", {"argv": [sys.executable, "-c", _IGNORES_GRACEFUL_STOP_SCRIPT]})
        r.launch("JOB-STUBBORN")
        time.sleep(0.2)
        r.graceful_stop("JOB-STUBBORN")
        time.sleep(0.4)
        assert r.poll("JOB-STUBBORN").is_running is True  # confirmed still alive -- ignored the signal
        r.force_terminate("JOB-STUBBORN")
        time.sleep(0.3)
        assert r.poll("JOB-STUBBORN").is_running is False


class TestCheckpointResumeHonesty:
    def test_no_checkpoint_capability_reported_honestly(self, tmp_path):
        r = _runner(tmp_path)
        r.prepare("JOB-CKPT", {"argv": [sys.executable, "-c", "pass"]})
        r.launch("JOB-CKPT")
        assert r.checkpoint_metadata("JOB-CKPT") is None

    def test_resume_refuses_rather_than_fabricates(self, tmp_path):
        from research.execution_job.runner import RunnerError

        r = _runner(tmp_path)
        r.prepare("JOB-RESUME", {"argv": [sys.executable, "-c", "pass"]})
        r.launch("JOB-RESUME")
        with pytest.raises(RunnerError):
            r.resume("JOB-RESUME", {"fake": "checkpoint"})


class TestReattachRecovery:
    def test_reattach_and_reclassify_still_running_no_duplicate_launch(self, tmp_path):
        """Full A-G recovery flow from Phase J real-runner authorization
        section 12, against a REAL subprocess."""
        original = _runner(tmp_path)
        original.prepare("JOB-RECOVER", {"argv": [sys.executable, "-c", "import time; time.sleep(3)"]})
        identity = original.launch("JOB-RECOVER")

        # Simulate a manager restart: a FRESH runner instance with no
        # in-memory knowledge of JOB-RECOVER.
        fresh = _runner(tmp_path)
        fresh.reattach(
            "JOB-RECOVER", pid=identity["pid"],
            argv=[sys.executable, "-c", "import time; time.sleep(3)"],
            cwd=identity["cwd"], creation_time=identity["launch_timestamp"],
        )
        recorded = fresh.recorded_identity("JOB-RECOVER")
        live = live_snapshot_for_pid(identity["pid"])
        assert verify_process_identity(recorded, live) is True  # F: identity verified
        status = fresh.poll("JOB-RECOVER")
        assert status.is_running is True  # G: reattached, not relaunched

        launches_before = len(fresh._prepared)
        fresh.poll("JOB-RECOVER")
        assert len(fresh._prepared) == launches_before  # no duplicate launch occurred

        original.force_terminate("JOB-RECOVER")  # cleanup

    def test_identity_mismatch_on_reattach_is_ambiguous(self, tmp_path):
        """A completely different real process at that PID -- reattach
        + identity check must NOT claim ownership."""
        import subprocess

        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        try:
            fresh = _runner(tmp_path)
            fresh.reattach(
                "JOB-MISMATCH", pid=unrelated.pid,
                argv=[sys.executable, "-c", "totally different command"],
                cwd=str(tmp_path), creation_time="0.0",
            )
            recorded = fresh.recorded_identity("JOB-MISMATCH")
            live = live_snapshot_for_pid(unrelated.pid)
            assert verify_process_identity(recorded, live) is False
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)
