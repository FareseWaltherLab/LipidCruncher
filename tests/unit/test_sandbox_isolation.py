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
