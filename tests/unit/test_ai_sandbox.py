"""
Unit tests for the AI sandbox — security-critical.

Tests that forbidden operations are blocked at the AST level
and that safe code executes correctly in the restricted environment.
"""
import signal

import pandas as pd
import pytest

from app.services.ai_chat.sandbox import (
    SandboxService,
    validate_code,
    SandboxError,
)
from app.models.chat import SandboxConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small DataFrame for sandbox testing."""
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1", "PE 36:2", "SM 42:1"],
        "ClassKey": ["PC", "PE", "SM"],
        "intensity[s1]": [100.0, 200.0, 300.0],
        "intensity[s2]": [110.0, 190.0, 310.0],
    })


@pytest.fixture
def default_config():
    return SandboxConfig()


# ---------------------------------------------------------------------------
# AST Validation — Blocked Operations
# ---------------------------------------------------------------------------

class TestASTValidation:
    """Tests that forbidden code patterns are rejected at parse time."""

    def test_import_os_blocked(self):
        with pytest.raises(SandboxError, match="Import of 'os'"):
            validate_code("import os")

    def test_import_subprocess_blocked(self):
        with pytest.raises(SandboxError, match="Import of 'subprocess'"):
            validate_code("import subprocess")

    def test_import_sys_blocked(self):
        with pytest.raises(SandboxError, match="Import of 'sys'"):
            validate_code("import sys")

    def test_from_os_import_blocked(self):
        with pytest.raises(SandboxError, match="Import from 'os'"):
            validate_code("from os import path")

    def test_from_pathlib_import_blocked(self):
        with pytest.raises(SandboxError, match="Import from 'pathlib'"):
            validate_code("from pathlib import Path")

    def test_import_socket_blocked(self):
        with pytest.raises(SandboxError, match="Import of 'socket'"):
            validate_code("import socket")

    def test_import_requests_blocked(self):
        with pytest.raises(SandboxError, match="Import of 'requests'"):
            validate_code("import requests")

    def test_whitelisted_import_math_allowed(self):
        tree = validate_code("import math\nx = math.sqrt(4)")
        assert tree is not None

    def test_whitelisted_import_statistics_allowed(self):
        tree = validate_code("import statistics\nx = statistics.mean([1, 2, 3])")
        assert tree is not None

    def test_whitelisted_import_re_allowed(self):
        tree = validate_code("import re\nx = re.findall(r'\\d+', 'PE 16:0_18:1')")
        assert tree is not None

    def test_whitelisted_import_pandas_allowed(self):
        tree = validate_code("import pandas as pd\nx = pd.DataFrame({'a': [1]})")
        assert tree is not None

    def test_whitelisted_import_numpy_allowed(self):
        tree = validate_code("import numpy as np\nx = np.array([1, 2])")
        assert tree is not None

    def test_whitelisted_import_plotly_allowed(self):
        tree = validate_code("import plotly.express as px\nx = px.bar")
        assert tree is not None

    def test_whitelisted_from_plotly_allowed(self):
        tree = validate_code("from plotly.graph_objects import Figure\nx = Figure")
        assert tree is not None

    def test_dunder_class_blocked(self):
        with pytest.raises(SandboxError, match="__class__"):
            validate_code("x = ().__class__")

    def test_dunder_subclasses_blocked(self):
        with pytest.raises(SandboxError, match="__subclasses__"):
            validate_code("x = object.__subclasses__()")

    def test_dunder_bases_blocked(self):
        with pytest.raises(SandboxError, match="__bases__"):
            validate_code("x = int.__bases__")

    def test_dunder_globals_blocked(self):
        with pytest.raises(SandboxError, match="__globals__"):
            validate_code("x = f.__globals__")

    def test_dunder_code_blocked(self):
        with pytest.raises(SandboxError, match="__code__"):
            validate_code("x = f.__code__")

    def test_dunder_builtins_blocked(self):
        with pytest.raises(SandboxError, match="__builtins__"):
            validate_code("x = __builtins__")

    def test_dunder_import_blocked(self):
        with pytest.raises(SandboxError, match="__import__"):
            validate_code("__import__('os')")

    def test_dunder_traversal_blocked(self):
        """The classic sandbox escape: ().__class__.__bases__[0].__subclasses__()"""
        with pytest.raises(SandboxError, match="is not allowed"):
            validate_code("().__class__.__bases__[0].__subclasses__()")

    def test_open_blocked(self):
        with pytest.raises(SandboxError, match="'open'"):
            validate_code("f = open('/etc/passwd')")

    def test_exec_blocked(self):
        with pytest.raises(SandboxError, match="'exec'"):
            validate_code("exec('print(1)')")

    def test_eval_blocked(self):
        with pytest.raises(SandboxError, match="'eval'"):
            validate_code("eval('1+1')")

    def test_compile_blocked(self):
        with pytest.raises(SandboxError, match="'compile'"):
            validate_code("compile('pass', '', 'exec')")

    def test_getattr_blocked(self):
        with pytest.raises(SandboxError, match="'getattr'"):
            validate_code("getattr(obj, 'secret')")

    def test_globals_function_blocked(self):
        with pytest.raises(SandboxError, match="'globals'"):
            validate_code("globals()")

    def test_locals_function_blocked(self):
        with pytest.raises(SandboxError, match="'locals'"):
            validate_code("locals()")

    def test_global_statement_blocked(self):
        with pytest.raises(SandboxError, match="'global'"):
            validate_code("global x")

    def test_nonlocal_statement_blocked(self):
        with pytest.raises(SandboxError, match="'nonlocal'"):
            validate_code("def f():\n  x = 1\n  def g():\n    nonlocal x")

    def test_syntax_error_caught(self):
        with pytest.raises(SandboxError, match="Syntax error"):
            validate_code("def f(")

    def test_breakpoint_blocked(self):
        with pytest.raises(SandboxError, match="'breakpoint'"):
            validate_code("breakpoint()")

    def test_input_blocked(self):
        with pytest.raises(SandboxError, match="'input'"):
            validate_code("x = input('enter: ')")


# ---------------------------------------------------------------------------
# Safe Code Execution
# ---------------------------------------------------------------------------

class TestSafeExecution:
    """Tests that legitimate code runs correctly in the sandbox."""

    def test_simple_assignment(self, sample_df, default_config):
        result = SandboxService.execute("x = 1 + 1", sample_df, default_config)
        assert result.success is True
        assert result.error_message == ""

    def test_dataframe_access(self, sample_df, default_config):
        code = "row_count = len(df)\nprint(row_count)"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "3" in result.stdout

    def test_dataframe_manipulation(self, sample_df, default_config):
        code = (
            "result_df = df.copy()\n"
            "result_df['total'] = result_df['intensity[s1]'] + result_df['intensity[s2]']\n"
        )
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.result_df is not None
        assert "total" in result.result_df.columns
        assert result.result_df["total"].iloc[0] == 210.0

    def test_plotly_figure(self, sample_df, default_config):
        code = (
            "import plotly.express as px\n"  # This would fail — px is already in globals
        )
        # Instead, use the pre-injected px
        code = "fig = px.bar(df, x='LipidMolec', y='intensity[s1]')"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.figure is not None

    def test_plotly_go_figure(self, sample_df, default_config):
        code = (
            "fig = go.Figure()\n"
            "fig.add_trace(go.Bar(x=df['LipidMolec'], y=df['intensity[s1]']))\n"
        )
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.figure is not None

    def test_numpy_operations(self, sample_df, default_config):
        code = "mean_val = np.mean(df['intensity[s1]'].values)\nprint(mean_val)"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "200" in result.stdout

    def test_math_module(self, sample_df, default_config):
        code = "import math\nresult = math.sqrt(144)\nprint(result)"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "12" in result.stdout

    def test_re_module(self, sample_df, default_config):
        code = "chains = re.findall(r'(\\d+):\\d+', 'PE 16:0_18:1')\nprint(chains)"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "16" in result.stdout and "18" in result.stdout

    def test_redundant_pandas_import_is_no_op(self, sample_df, default_config):
        # AI sometimes adds defensive `import pandas as pd` even though pd
        # is already pre-loaded. Should run without error.
        code = (
            "import pandas as pd\n"
            "result_df = pd.DataFrame({'a': [1, 2, 3]})\n"
        )
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.result_df is not None
        assert list(result.result_df["a"]) == [1, 2, 3]

    def test_redundant_numpy_import_is_no_op(self, sample_df, default_config):
        code = "import numpy as np\nprint(np.array([1, 2]).sum())"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "3" in result.stdout

    def test_redundant_plotly_import_is_no_op(self, sample_df, default_config):
        code = (
            "import plotly.graph_objects as go\n"
            "fig = go.Figure()\n"
        )
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.figure is not None

    def test_print_capture(self, sample_df, default_config):
        code = "print('hello')\nprint('world')"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_df_is_copy(self, sample_df, default_config):
        """Sandbox code cannot modify the original DataFrame."""
        original_val = sample_df["intensity[s1]"].iloc[0]
        code = "df['intensity[s1]'] = 0"
        SandboxService.execute(code, sample_df, default_config)
        assert sample_df["intensity[s1]"].iloc[0] == original_val

    def test_builtin_functions_available(self, sample_df, default_config):
        code = (
            "nums = [3, 1, 4, 1, 5]\n"
            "print(len(nums))\n"
            "print(sum(nums))\n"
            "print(sorted(nums))\n"
            "print(min(nums))\n"
            "print(max(nums))\n"
        )
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert "5" in result.stdout
        assert "14" in result.stdout


# ---------------------------------------------------------------------------
# Runtime Error Handling
# ---------------------------------------------------------------------------

class TestRuntimeErrors:
    """Tests that runtime errors produce clean error messages."""

    def test_division_by_zero(self, sample_df, default_config):
        result = SandboxService.execute("x = 1 / 0", sample_df, default_config)
        assert result.success is False
        assert "ZeroDivisionError" in result.error_message

    def test_key_error(self, sample_df, default_config):
        result = SandboxService.execute("x = df['nonexistent']", sample_df, default_config)
        assert result.success is False
        assert "KeyError" in result.error_message

    def test_type_error(self, sample_df, default_config):
        result = SandboxService.execute("x = 'a' + 1", sample_df, default_config)
        assert result.success is False
        assert "TypeError" in result.error_message

    def test_name_error(self, sample_df, default_config):
        result = SandboxService.execute("x = undefined_var", sample_df, default_config)
        assert result.success is False
        assert "NameError" in result.error_message


# ---------------------------------------------------------------------------
# Output Limits
# ---------------------------------------------------------------------------

class TestOutputLimits:
    """Tests that output size constraints are enforced."""

    def test_large_dataframe_truncated(self, sample_df):
        config = SandboxConfig(max_output_rows=2)
        code = "result_df = df.copy()"  # 3 rows, limit is 2
        result = SandboxService.execute(code, sample_df, config)
        assert result.success is True
        assert result.result_df is not None
        assert len(result.result_df) == 2

    def test_non_figure_in_fig_ignored(self, sample_df, default_config):
        """If code sets fig to a non-Figure, it should be ignored."""
        code = "fig = 'not a figure'"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.figure is None

    def test_non_dataframe_in_result_df_ignored(self, sample_df, default_config):
        """If code sets result_df to a non-DataFrame, it should be ignored."""
        code = "result_df = [1, 2, 3]"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is True
        assert result.result_df is None


# ---------------------------------------------------------------------------
# Blocked at Execution (belt-and-suspenders with AST)
# ---------------------------------------------------------------------------

class TestExecutionBlocked:
    """Tests that forbidden operations fail even if they somehow pass AST."""

    def test_ast_blocks_import_os(self, sample_df, default_config):
        """import os is caught at AST phase — result is failure, not exception."""
        result = SandboxService.execute("import os\nos.system('ls')", sample_df, default_config)
        assert result.success is False
        assert "not allowed" in result.error_message

    def test_ast_blocks_open(self, sample_df, default_config):
        result = SandboxService.execute("f = open('/etc/passwd')", sample_df, default_config)
        assert result.success is False

    def test_ast_blocks_eval(self, sample_df, default_config):
        result = SandboxService.execute("eval('1+1')", sample_df, default_config)
        assert result.success is False

    def test_ast_blocks_dunder_escape(self, sample_df, default_config):
        code = "().__class__.__bases__[0].__subclasses__()"
        result = SandboxService.execute(code, sample_df, default_config)
        assert result.success is False


# ---------------------------------------------------------------------------
# Timeout (only on platforms with SIGALRM)
# ---------------------------------------------------------------------------

class TestTimeout:
    """Tests execution timeout enforcement."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_infinite_loop_times_out(self, sample_df):
        config = SandboxConfig(execution_timeout=1)
        code = "while True:\n    pass"
        result = SandboxService.execute(code, sample_df, config)
        assert result.success is False
        assert "timed out" in result.error_message
