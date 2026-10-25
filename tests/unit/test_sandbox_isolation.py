"""Process-isolation guarantees for the sandbox.

These tests exist because the AST validator alone never contained untrusted
code: `pd`/`np` are bound into the execution globals, and real library
objects expose real modules by ordinary attribute access (`pd.compat.os`,
`pd.read_pickle`, `pd.read_csv('http://…')`), which no import allowlist can
see. Containment therefore has to come from the *process*, not the parser.

Each test below drives one of those escapes end to end and asserts it fails.
They are deliberately written against the attack, not the implementation, so
they keep their meaning if the hardening is rewritten.
"""
import os
import sys

import pandas as pd
import pytest

from app.models.chat import SandboxConfig
from app.services.ai_chat.sandbox import SandboxService


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "ClassKey": ["PC", "PE", "PC"]})


def run(code, df, **kw):
    return SandboxService.execute(code, df, **kw)


class TestStillWorks:
    """Isolation must not cost normal functionality."""

    def test_dataframe_and_stdout(self, df):
        res = run("print('rows', len(df))", df)
        assert res.success, res.error_message
        assert res.stdout.strip() == "rows 3"

    def test_result_df_crosses_the_boundary(self, df):
        res = run("result_df = df.head(2)", df)
        assert res.success, res.error_message
        assert len(res.result_df) == 2

    def test_figure_crosses_the_boundary(self, df):
        res = run("fig = px.bar(df, x='ClassKey', y='a')", df)
        assert res.success, res.error_message
        assert res.figure is not None
        assert len(res.figure.data) > 0

    def test_runtime_errors_are_reported_not_raised(self, df):
        res = run("1/0", df)
        assert not res.success
        assert "ZeroDivisionError" in res.error_message


class TestSecretsAreNotInherited:
    def test_api_key_is_not_visible_to_generated_code(self, df, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-leak")
        res = run(
            "print(pd.io.common.os.environ.get('ANTHROPIC_API_KEY'))", df
        )
        assert res.success, res.error_message
        assert "sk-must-not-leak" not in res.stdout
        assert res.stdout.strip() == "None"


class TestFilesCannotBeRead:
    """Regression: the child inherited no environment but could still read
    files, so `pd.read_csv` on a secrets file baked into the image, or on
    the parent's /proc/<pid>/environ (where ECS puts the API key), printed
    the secret straight back."""

    @pytest.fixture
    def secret_file(self, tmp_path):
        path = tmp_path / "secrets.csv"
        path.write_text("key\nsk-must-not-leak\n")
        return str(path)

    @pytest.mark.parametrize("reader", [
        "pd.read_csv({p!r})",
        "pd.read_csv({p!r}, engine='pyarrow')",
        "pd.read_csv({p!r}, engine='python')",
        "pd.read_table({p!r})",
        "pd.read_json({p!r})",
        "pd.read_parquet({p!r})",
        "np.loadtxt({p!r}, dtype=str, delimiter='|')",
        "np.fromfile({p!r}, dtype='uint8')",
        "np.memmap({p!r}, dtype='uint8', mode='r')",
        "pd.io.common.get_handle({p!r}, 'r').handle.read()",
    ])
    def test_library_readers_cannot_open_files(self, df, secret_file, reader):
        res = run(f"print({reader.format(p=secret_file)})", df)
        assert not res.success
        assert "sk-must-not-leak" not in res.stdout + res.error_message

    def test_parent_environment_cannot_be_read(self, df, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-leak")
        res = run(
            f"print(pd.read_csv('/proc/{os.getpid()}/environ', sep='|'))", df,
        )
        assert not res.success
        assert "sk-must-not-leak" not in res.stdout + res.error_message

    def test_directories_cannot_be_listed(self, df, secret_file):
        res = run(
            f"print(pd.compat.os.listdir({os.path.dirname(secret_file)!r}))", df,
        )
        assert not res.success
        assert "secrets.csv" not in res.stdout + res.error_message

    def test_dot_dot_cannot_climb_out_of_the_libraries(self, df, secret_file):
        res = run(
            f"print(pd.read_csv(np.__file__ + '/../../../../../..' + {secret_file!r}))",
            df,
        )
        assert not res.success
        assert "sk-must-not-leak" not in res.stdout + res.error_message


class TestFilesystemIsBlocked:
    def test_cannot_write_a_file_through_pandas(self, df, tmp_path):
        target = tmp_path / "escaped.csv"
        res = run(f"df.to_csv({str(target)!r})", df)
        assert not res.success
        # The size rlimit lets an empty file be created but permits no
        # bytes into it, so no data can be exfiltrated to disk.
        if target.exists():
            assert target.stat().st_size == 0


class TestProcessExecutionIsBlocked:
    def test_os_system_cannot_spawn(self, df, tmp_path):
        marker = tmp_path / "spawned"
        run(f"pd.compat.os.system('touch {marker}')", df)
        # os.system reports failure rather than raising, so the assertion
        # that matters is that no process actually ran.
        assert not marker.exists(), "sandbox spawned a process"


class TestNetworkIsBlocked:
    def test_cannot_reach_the_network_through_pandas(self, df):
        res = run("print(pd.read_csv('http://example.com/data.csv'))", df)
        assert not res.success
        assert "example.com" not in (res.stdout or "")


class TestTimeoutKills:
    def test_infinite_loop_is_terminated_and_reported(self, df):
        res = run("while True: pass", df, config=SandboxConfig(
            execution_timeout=3,
        ))
        assert not res.success
        assert "timed out" in res.error_message.lower()

    def test_parent_survives_a_child_that_hard_exits(self, df):
        before = os.getpid()
        res = run("pd.io.common.os._exit(1)", df)
        assert not res.success          # reported, not fatal
        assert os.getpid() == before    # app process untouched
        # and the sandbox still works afterwards
        assert run("print('alive')", df).stdout.strip() == "alive"


class TestValidatorStillApplies:
    """AST validation remains as defense-in-depth in front of the process."""

    def test_blocked_import_is_rejected_before_spawning(self, df):
        res = run("import os", df)
        assert not res.success
        assert "not allowed" in res.error_message


class TestParentNeverUnpicklesChildOutput:
    """Regression: the parent unpickled the child's response, so code that
    controlled those bytes ran in the app process. The response is JSON now;
    these tests pin that and the fidelity the JSON form has to keep."""

    def test_response_is_decoded_without_pickle(self, df, monkeypatch):
        import pickle

        def _refuse(*args, **kwargs):
            raise AssertionError("parent unpickled the child's output")

        monkeypatch.setattr(pickle, "loads", _refuse)
        res = run("fig = px.bar(df, x='ClassKey', y='a')\nresult_df = df", df)
        assert res.success, res.error_message
        assert res.figure is not None
        assert res.result_df is not None

    def test_figure_comes_back_as_a_plotly_figure(self, df):
        import plotly.graph_objects as go

        res = run("fig = px.bar(df, x='ClassKey', y='a', title='T')", df)
        assert isinstance(res.figure, go.Figure)
        assert res.figure.layout.title.text == "T"
        assert list(res.figure.data[0].y) == [1, 2, 3]

    def test_result_df_keeps_dtypes_labels_and_exact_floats(self, df):
        code = (
            "result_df = pd.DataFrame({"
            "'Lipid': ['PC 1', '02', None],"
            "'s1': [1.2345678901234567e-20, 123456789.12345678, float('nan')],"
            "'n': [1, 2, 3], 'flag': [True, False, True]},"
            " index=pd.Index(['a', 'b', 'c'], name='k'))"
        )
        res = run(code, df)
        assert res.success, res.error_message
        out = res.result_df
        assert out.index.name == "k"
        assert list(out.index) == ["a", "b", "c"]
        assert out["Lipid"].dtype == object
        assert out["Lipid"].tolist()[:2] == ["PC 1", "02"]  # "02" stays text
        assert out["Lipid"].isna().tolist() == [False, False, True]
        assert out["s1"].dtype == "float64"
        assert out["s1"].iloc[0] == 1.2345678901234567e-20
        assert out["s1"].iloc[1] == 123456789.12345678
        assert pd.isna(out["s1"].iloc[2])
        assert out["n"].dtype == "int64"
        assert out["flag"].dtype == bool

    def test_result_df_keeps_multiindex_columns(self, df):
        res = run(
            "result_df = df.groupby('ClassKey')[['a']].agg(['mean', 'sum'])", df,
        )
        assert res.success, res.error_message
        out = res.result_df
        assert list(out.columns) == [("a", "mean"), ("a", "sum")]
        assert out.index.name == "ClassKey"
        assert out.loc["PC", ("a", "sum")] == 4


def _run_child_script(script):
    """Run ``script`` in a fresh interpreter, so limits never hit pytest."""
    import subprocess

    from app.services.ai_chat.sandbox import _child_env

    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=_child_env(), timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


_HARDEN = """
import os, resource
{patch}
from app.services.ai_chat import sandbox_runner
try:
    sandbox_runner._harden(5, 1024 ** 3)
except Exception as e:
    print('refused:', e)
else:
    print('applied')
"""


class TestHardeningFailsClosed:
    """Regression: every control in ``_harden`` swallowed its own failure,
    so the deployed container (root, which Linux exempts from
    RLIMIT_NPROC) ran generated code with no process limit and nothing
    said so. The child now checks the limits took effect and otherwise
    refuses to run the code."""

    def test_limits_apply_for_a_normal_user(self):
        assert _run_child_script(_HARDEN.format(patch="")) == "applied"

    def test_refuses_to_run_as_root(self):
        out = _run_child_script(_HARDEN.format(patch="os.geteuid = lambda: 0"))
        assert out.startswith("refused:") and "root" in out

    def test_refuses_when_a_limit_cannot_be_set(self):
        patch = (
            "_set = resource.setrlimit\n"
            "def _fail_nproc(which, value):\n"
            "    if which == resource.RLIMIT_NPROC:\n"
            "        raise ValueError('not permitted')\n"
            "    _set(which, value)\n"
            "resource.setrlimit = _fail_nproc"
        )
        out = _run_child_script(_HARDEN.format(patch=patch))
        assert out.startswith("refused:")

    def test_refuses_when_a_limit_is_silently_ignored(self):
        patch = "resource.setrlimit = lambda which, value: None"
        out = _run_child_script(_HARDEN.format(patch=patch))
        assert out.startswith("refused:") and "did not take effect" in out

    def test_code_is_not_run_when_hardening_fails(self):
        script = (
            "import io, json, os, pickle, sys\n"
            "import pandas as pd\n"
            "os.geteuid = lambda: 0\n"
            "from app.services.ai_chat import sandbox_runner\n"
            "r, w = os.pipe()\n"
            "sys.argv = ['runner', str(w)]\n"
            "request = {'code': \"print('ran')\", 'df': pd.DataFrame({'a': [1]})}\n"
            "sys.stdin = io.TextIOWrapper(io.BytesIO(pickle.dumps(request)))\n"
            "sandbox_runner.main()\n"
            "print(os.read(r, 1 << 20).decode())\n"
        )
        response = __import__("json").loads(_run_child_script(script))
        assert response["ok"] is False
        assert response["stdout"] == ""
        assert "was not run" in response["error"]


_FAKE_PRCTL = """
class _Libc:
    def prctl(self, *args):
        return {result}
sandbox_runner.ctypes.CDLL = lambda *a, **k: _Libc()
sandbox_runner._syscall_filter_applies = lambda: True
"""


class TestMatplotlibLoadsBeforeTheLimits:
    """Regression: on Linux, threads count toward RLIMIT_NPROC, and
    matplotlib's font manager starts a thread while it builds its font
    cache (always, in a fresh container). It was loaded after the limits,
    failed silently, and every later matplotlib import in the sandbox
    failed ("can't start new thread"): lc.volcano() and any plot needing
    matplotlib broke on the server, never on macOS."""

    def test_font_manager_is_loaded_before_any_limit(self):
        patch = (
            "import sys\n"
            "_set = resource.setrlimit\n"
            "def _record(which, value):\n"
            "    print('font manager loaded:', 'matplotlib.font_manager' in sys.modules)\n"
            "    _set(which, value)\n"
            "resource.setrlimit = _record"
        )
        out = _run_child_script(_HARDEN.format(patch=patch)).splitlines()
        assert out[0] == "font manager loaded: True"


class TestNetworkIsBlockedByTheKernel:
    """S2 option B: on Linux x86_64 the child installs a seccomp filter, so
    the kernel refuses sockets (and ptrace / process_vm_*) whatever route
    the code takes to libc, not only the Python socket module."""

    _PROBE = """
import _socket
from app.services.ai_chat import sandbox_runner
sandbox_runner._harden(5, 1024 ** 3)
for name, make in (('socket', _socket.socket), ('socketpair', _socket.socketpair)):
    try:
        make()
    except PermissionError:
        print(name, 'refused')
    else:
        print(name, 'opened')
"""

    @pytest.mark.skipif(
        not (sys.platform.startswith("linux") and __import__("platform").machine() == "x86_64"),
        reason="the filter is written for Linux x86_64",
    )
    def test_the_kernel_refuses_sockets(self):
        assert _run_child_script(self._PROBE).splitlines() == [
            "socket refused", "socketpair refused",
        ]

    def test_refuses_to_run_if_the_filter_cannot_be_installed(self):
        script = _HARDEN.format(patch="").replace(
            "try:", "exec(" + repr(_FAKE_PRCTL.format(result=-1)) + ")\ntry:", 1,
        )
        out = _run_child_script(script)
        assert out.startswith("refused:") and "no_new_privs" in out

    def test_refuses_to_run_if_the_filter_does_not_take(self):
        """prctl reports success but sockets still open: refuse."""
        script = _HARDEN.format(patch="").replace(
            "try:", "exec(" + repr(_FAKE_PRCTL.format(result=0)) + ")\ntry:", 1,
        )
        out = _run_child_script(script)
        assert out.startswith("refused:") and "did not take effect" in out


class TestMemoryIsBounded:
    """Regression: no memory limit was set, so one large allocation in
    generated code could exhaust the shared server."""

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="macOS cannot lower RLIMIT_AS; the limit applies on Linux only",
    )
    def test_allocation_beyond_the_cap_fails(self, df):
        # np.zeros maps without touching pages, so the old code "succeeded"
        # here without actually using 3 GiB.
        res = run(
            "a = np.zeros(3 * 2**30 // 8)\nprint(a.nbytes)", df,
            config=SandboxConfig(max_memory_bytes=2**30),
        )
        assert not res.success
        assert "MemoryError" in res.error_message

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="macOS cannot lower RLIMIT_AS; the limit applies on Linux only",
    )
    def test_allocation_under_the_cap_works(self, df):
        res = run(
            "a = np.zeros(2**28 // 8)\nprint(a.nbytes)", df,
            config=SandboxConfig(max_memory_bytes=2**30),
        )
        assert res.success, res.error_message


class TestOutputIsBounded:
    """Regression: the parent buffered everything the child wrote, for as
    long as the timeout allowed, so one oversized result could exhaust the
    app process."""

    def test_oversized_result_is_stopped_and_reported(self, df):
        res = run(
            "result_df = pd.DataFrame({'x': ['y' * 1000] * 200})", df,
            config=SandboxConfig(max_output_bytes=50_000),
        )
        assert not res.success
        assert res.result_df is None
        assert "larger than" in res.error_message

    def test_oversized_printed_output_counts_too(self, df):
        res = run(
            "for i in range(200): print('y' * 1000)", df,
            config=SandboxConfig(max_output_bytes=50_000),
        )
        assert not res.success
        assert "larger than" in res.error_message

    def test_result_under_the_cap_is_returned(self, df):
        res = run(
            "result_df = pd.DataFrame({'x': ['y' * 1000] * 20})", df,
            config=SandboxConfig(max_output_bytes=50_000),
        )
        assert res.success, res.error_message
        assert len(res.result_df) == 20


class TestConcurrencyIsBounded:
    """Regression: nothing limited how many sandbox children ran at once,
    so simultaneous requests could start enough interpreters to exhaust
    the shared server."""

    @pytest.fixture
    def one_slot(self, monkeypatch):
        import threading

        from app.services.ai_chat import sandbox

        slots = threading.BoundedSemaphore(1)
        monkeypatch.setattr(sandbox, "_SLOTS", slots)
        return slots

    def test_run_reports_busy_when_no_slot_frees_up(self, df, one_slot):
        one_slot.acquire()  # another request holds the only slot
        try:
            res = run("print('ran')", df, config=SandboxConfig(execution_timeout=1))
        finally:
            one_slot.release()
        assert not res.success
        assert "busy" in res.error_message
        assert res.stdout == ""

    def test_slot_is_released_after_success_and_failure(self, df, one_slot):
        assert run("1/0", df).error_message
        assert run("print('ran')", df).stdout.strip() == "ran"
        assert run("print('again')", df).stdout.strip() == "again"


class TestParentHandlesTheChildsPipesAndWorkdir:
    """The parent drains the result pipe but, until this, read stderr only
    after the child exited: a child that wrote more than a pipe buffer of
    library chatter blocked on stderr before it could write its result and
    was reported as a timeout. The timeout branch also left the child's
    temp workdir behind."""

    def test_large_stderr_does_not_stall_the_result(self, df):
        # pd.compat exposes sys, so generated code can reach stderr without
        # importing anything the allowlist refuses; a warnings flood from a
        # library has the same effect.
        code = (
            "pd.compat.sys.stderr.write('x' * 300_000)\n"
            "pd.compat.sys.stderr.flush()\n"
            "print('done')\n"
        )
        res = run(code, df, config=SandboxConfig(execution_timeout=5))
        assert res.success, res.error_message
        assert res.stdout.strip() == "done"

    def test_timeout_removes_the_workdir(self, df):
        import tempfile
        from pathlib import Path

        def workdirs():
            return set(Path(tempfile.gettempdir()).glob("lc_sandbox_*"))

        before = workdirs()
        res = run("while True:\n    pass", df, config=SandboxConfig(execution_timeout=1))
        assert not res.success
        assert "timed out" in res.error_message
        assert workdirs() - before == set()


class TestDockerImage:
    def test_image_does_not_run_as_root(self):
        from pathlib import Path

        dockerfile = Path(__file__).parents[2] / "Dockerfile"
        lines = [ln.split() for ln in dockerfile.read_text().splitlines()]
        users = [ln[1] for ln in lines if ln and ln[0] == "USER"]
        assert users, "Dockerfile has no USER line, so the app runs as root"
        assert users[-1] not in ("root", "0")

    def test_r_is_opt_in(self):
        """R + IsoCorrectoR is built only with --build-arg WITH_R=1."""
        from pathlib import Path

        dockerfile = Path(__file__).parents[2] / "Dockerfile"
        assert "ARG WITH_R=0" in dockerfile.read_text().splitlines()


class TestAppProcessIsPrivate:
    """S2 option A: the app process holds the API key in its environment,
    which Linux shows any process of the same user in /proc/<pid>/environ.
    The sandbox marks the app process non-dumpable before running code, so
    the kernel refuses that read even if the Python-level guard is beaten."""

    def _holder(self, private):
        import os
        import subprocess
        from pathlib import Path

        src = Path(__file__).parents[2] / "src"
        code = (
            "from app.services.ai_chat.sandbox import make_this_process_private\n"
            f"assert make_this_process_private() if {private} else True\n"
            "print('ready', flush=True)\n"
            "import time; time.sleep(60)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(src),
                 "LC_TEST_SECRET": "not-a-real-key"},
        )
        assert proc.stdout.readline().strip() == "ready"
        return proc

    def _read_environ(self, pid):
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return fh.read()

    @pytest.mark.skipif(not sys.platform.startswith("linux"),
                        reason="/proc and prctl are Linux-only")
    def test_environment_of_a_private_process_cannot_be_read(self):
        proc = self._holder(private=True)
        try:
            with pytest.raises(PermissionError):
                self._read_environ(proc.pid)
        finally:
            proc.kill()

    @pytest.mark.skipif(not sys.platform.startswith("linux"),
                        reason="/proc and prctl are Linux-only")
    def test_control_an_ordinary_process_is_readable(self):
        proc = self._holder(private=False)
        try:
            assert b"LC_TEST_SECRET=not-a-real-key" in self._read_environ(proc.pid)
        finally:
            proc.kill()

    def test_code_is_not_run_if_the_app_process_cannot_be_made_private(
        self, df, monkeypatch,
    ):
        from app.services.ai_chat import sandbox
        monkeypatch.setattr(sandbox, "make_this_process_private", lambda: False)
        res = run("print('ran')", df)
        assert not res.success
        assert "could not protect the app process" in res.error_message
        assert res.stdout == ""
