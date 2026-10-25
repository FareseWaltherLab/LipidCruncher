"""
Sandboxed code execution for AI-generated Python code.

Security-critical module. AI-generated code is validated via AST
analysis before execution in a restricted environment with curated
builtins, whitelisted globals, and a timeout.

Defense model: the sandbox is the security boundary, not AI compliance.
All AI output is treated as untrusted.
"""
import ast
import ctypes
import json
import math
import os
import pickle
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from ...models.chat import SandboxConfig
from .exceptions import SandboxError


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandboxResult:
    """Result of a sandboxed code execution.

    Attributes:
        success: Whether execution completed without error.
        figure: Plotly figure produced by the code, if any.
        result_df: DataFrame produced by the code, if any.
        stdout: Captured print output.
        error_message: Error description if execution failed.
        execution_time: Wall-clock seconds the code took to run.
        methods: One line per statistical analysis the code ran through
            ``lc``, saying which choices (test, correction, ...) were made.
    """
    success: bool
    figure: Optional[go.Figure] = None
    result_df: Optional[pd.DataFrame] = None
    stdout: str = ""
    error_message: str = ""
    execution_time: float = 0.0
    methods: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# AST Validator
# ---------------------------------------------------------------------------

# Analysis modules generated code may import, for questions the pipeline
# doesn't already cover (prefer the `lc` namespace for anything LipidCruncher
# implements). Submodules, not whole packages: `import scipy` or
# `from scipy import io` is refused, so readers and dataset downloaders such
# as `scipy.io`, `sklearn.datasets` and `statsmodels.api` (`sm.load` unpickles)
# are not an import away. Listed in the prompts as written here.
ANALYSIS_MODULES: Tuple[str, ...] = (
    "scipy.stats", "scipy.cluster.hierarchy", "scipy.spatial.distance",
    "statsmodels.stats", "statsmodels.formula.api",
    "sklearn.decomposition", "sklearn.preprocessing", "sklearn.cluster",
    "sklearn.manifold", "sklearn.metrics",
)

# Every importable module (and its submodules). pandas/numpy/plotly are
# pre-bound as pd/np/px/go; the AI sometimes adds defensive imports
# (`import pandas as pd`) after an error, which stay harmless no-ops.
#
# This narrows what an import can reach; it does not contain the code.
# `import scipy.stats` still binds `scipy`, whose other submodules load on
# attribute access, and `pd`/`np` reach the same capabilities by ordinary
# attribute chains (`pd.read_pickle`, `pd.compat.os`), which the AST
# validator does not restrict. The child process is the boundary.
_ALLOWED_MODULES: frozenset = frozenset({
    "math", "statistics", "re", "pandas", "numpy", "plotly",
    *ANALYSIS_MODULES,
})


def _module_allowed(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in _ALLOWED_MODULES)


def _import_allowed(module: str, names=()) -> bool:
    """Whether ``import module`` / ``from module import names`` is allowed.

    ``from scipy.stats import ttest_ind`` imports from an allowed module;
    ``from scipy import stats`` imports allowed modules from their parent.
    """
    if _module_allowed(module):
        return True
    return bool(names) and all(
        _module_allowed(f"{module}.{name}") for name in names
    )


# Dunder attributes that must never be accessed
_BLOCKED_DUNDERS: frozenset = frozenset({
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__code__", "__builtins__", "__import__",
    "__loader__", "__spec__", "__qualname__",
})

# Builtin names that must never be called
_BLOCKED_BUILTINS: frozenset = frozenset({
    "exec", "eval", "compile", "__import__", "__builtins__", "open",
    "breakpoint", "exit", "quit", "input",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "memoryview", "bytearray",
})


class _ASTValidator(ast.NodeVisitor):
    """Walks the AST to detect forbidden operations.

    Raises SandboxError on the first violation found.
    """

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _import_allowed(alias.name):
                raise SandboxError(f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            names = [alias.name for alias in node.names]
            if not _import_allowed(node.module, names):
                raise SandboxError(f"Import from '{node.module}' is not allowed")
        else:
            raise SandboxError("Relative imports are not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_DUNDERS:
            raise SandboxError(f"Access to '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise SandboxError("'global' statements are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise SandboxError("'nonlocal' statements are not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_BUILTINS:
            raise SandboxError(f"Use of '{node.id}' is not allowed")
        self.generic_visit(node)


def validate_code(code: str) -> ast.Module:
    """Parse and validate code via AST analysis.

    Args:
        code: Python source code string.

    Returns:
        Parsed AST module node (for subsequent exec).

    Raises:
        SandboxError: If the code contains forbidden operations or
            has syntax errors.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"Syntax error: {e}") from e

    _ASTValidator().visit(tree)
    return tree


# ---------------------------------------------------------------------------
# Execution environment
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "print": print,  # replaced at runtime with capturing print
    "isinstance": isinstance,
    "type": type,
    "True": True,
    "False": False,
    "None": None,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "reversed": reversed,
    "frozenset": frozenset,
    "KeyError": KeyError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
}


def _restricted_import(name, *args, **kwargs):
    """Import function that only allows the modules in ``_ALLOWED_MODULES``.

    Applies the AST validator's rule again at run time (``from X import a``
    arrives as ``name=X, fromlist=('a',)``).
    """
    fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist")
    if not _import_allowed(name, tuple(fromlist or ())):
        raise ImportError(f"Import of '{name}' is not allowed")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)


class _UnavailablePipelineAPI:
    """Stands in for ``lc`` when no instance was supplied.

    Binding ``None`` instead would surface as a bare ``'NoneType' object has
    no attribute 'volcano'``, which tells neither the model nor the user
    what actually went wrong.
    """

    def __getattr__(self, name):
        raise RuntimeError(
            f"lc.{name} is unavailable: LipidCruncher's pipeline was not "
            f"supplied to this execution context. Compute the answer "
            f"directly from `df` instead."
        )


def _build_globals(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    lc: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the restricted globals dict for exec().

    Includes whitelisted libraries and the user's DataFrame. When
    ``numeric_columns`` is provided, those columns are coerced to numeric
    (with ``errors='coerce'``) on the DataFrame copy before exposure, so
    AI-generated code can do arithmetic on sample intensities without
    worrying about lingering string dtypes from the raw CSV read.

    Also exposes ``lc`` — LipidCruncher's own pipeline capabilities — so
    generated code defers to the app's documented methods (statistics,
    plots, summaries) instead of re-deriving them with ad-hoc formulas.
    ``lc`` is supplied by the caller rather than constructed here: it lives
    in the workflows layer (it delegates to ``AnalysisWorkflow``), and a
    service must not import a workflow.
    """
    captured_output: list = []

    def capturing_print(*args, **kwargs):
        captured_output.append(" ".join(str(a) for a in args))

    builtins = {**_SAFE_BUILTINS, "print": capturing_print, "__import__": _restricted_import}

    df_copy = df.copy()
    if numeric_columns:
        present = [c for c in numeric_columns if c in df_copy.columns]
        if present:
            df_copy[present] = df_copy[present].apply(
                pd.to_numeric, errors='coerce',
            )

    return {
        "__builtins__": builtins,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "math": math,
        "statistics": statistics,
        "re": re,
        "df": df_copy,
        # Rebind to the prepared copy so lc and df see identical data.
        "lc": lc.with_dataframe(df_copy) if lc is not None
              else _UnavailablePipelineAPI(),
        "_captured_output": captured_output,
    }


def build_sandbox_globals(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    lc: Optional[Any] = None,
) -> Dict[str, Any]:
    """Public entry to ``_build_globals`` for the child runner process."""
    return _build_globals(
        df, numeric_columns=numeric_columns, lc=lc,
    )


# ---------------------------------------------------------------------------
# Child -> parent wire format
# ---------------------------------------------------------------------------
# Everything the child sends back is untrusted: the code it ran could have
# replaced any function the encoder calls. So the parent only json-decodes
# it and rebuilds plain pandas/plotly objects from the values. It must never
# unpickle it — unpickling calls whatever the bytes name, in the app process.

def _index_to_wire(index: pd.Index) -> Dict[str, Any]:
    return {"values": index.tolist(), "names": list(index.names)}


def _index_from_wire(wire: Dict[str, Any]) -> pd.Index:
    names = wire["names"]
    if len(names) > 1:
        tuples = [tuple(v) for v in wire["values"]]
        if not tuples:
            return pd.MultiIndex.from_arrays([[]] * len(names), names=names)
        return pd.MultiIndex.from_tuples(tuples, names=names)
    return pd.Index(wire["values"], name=names[0])


def frame_to_wire(df: pd.DataFrame) -> Dict[str, Any]:
    """JSON-safe form of ``df`` that keeps labels, dtypes and exact floats.

    Column-oriented so each column keeps its own dtype; missing values go
    as ``null``. ``json`` writes floats with ``repr``, which round-trips
    exactly (pandas' own ``to_json`` rounds to 15 digits).
    """
    return {
        "columns": _index_to_wire(df.columns),
        "index": _index_to_wire(df.index),
        "dtypes": [str(t) for t in df.dtypes],
        "data": [
            col.astype(object).where(col.notna(), None).tolist()
            for _, col in df.items()
        ],
    }


def _wire_dtype(name: str) -> Any:
    """Numeric and boolean dtypes are restored; anything else is object."""
    try:
        dtype = pd.api.types.pandas_dtype(name)
    except TypeError:
        return object
    return dtype if dtype.kind in "biuf" else object


def _frame_from_wire(wire: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(
        {i: pd.array(values, dtype=_wire_dtype(dtype))
         for i, (values, dtype) in enumerate(zip(wire["data"], wire["dtypes"]))},
        index=_index_from_wire(wire["index"]),
    )
    df.columns = _index_from_wire(wire["columns"])
    return df


def _decode_response(payload: bytes) -> Dict[str, Any]:
    response = json.loads(payload)
    if response.get("figure") is not None:
        response["figure"] = pio.from_json(response["figure"])
    if response.get("result_df") is not None:
        response["result_df"] = _frame_from_wire(response["result_df"])
    return response


# ---------------------------------------------------------------------------
# Out-of-process execution
# ---------------------------------------------------------------------------

# Environment handed to the child. Deliberately minimal: secrets such as
# ANTHROPIC_API_KEY must never be readable by AI-generated code, so the
# child is given a fresh environment rather than an inherited one.
def _child_env() -> Dict[str, str]:
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": src_root,
        "PYTHONHASHSEED": "0",
        "HOME": tempfile.gettempdir(),
        "MPLCONFIGDIR": tempfile.gettempdir(),
    }


# How much of the child's stderr the parent keeps for the crash message.
_STDERR_TAIL_BYTES = 64 * 1024


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group (it is the group leader)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone
    except PermissionError:
        proc.kill()


def _run_in_subprocess(
    code: str,
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]],
    config: SandboxConfig,
    lc: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute code in a separate, hardened process and return its response.

    The child cannot outlive ``config.execution_timeout``: if the result
    pipe has not closed by then the process is killed outright. Unlike the
    previous in-thread approach, a hung or runaway execution cannot survive
    in the app process. Nor can its output: at most
    ``config.max_output_bytes`` are read back before the child is killed.

    Raises:
        _SandboxTimeout: if the child overran its timeout.
        _SandboxCrash: if the child died without returning a response, or
            its response was too large.
    """
    read_fd, write_fd = os.pipe()
    workdir = tempfile.mkdtemp(prefix="lc_sandbox_")

    request = pickle.dumps({
        "code": code,
        "df": df,
        "numeric_columns": numeric_columns,
        "lc": lc,
        "max_output_rows": config.max_output_rows,
        "cpu_seconds": max(1, config.execution_timeout),
        "memory_bytes": config.max_memory_bytes,
    }, protocol=pickle.HIGHEST_PROTOCOL)

    proc = subprocess.Popen(
        [sys.executable, "-m", "app.services.ai_chat.sandbox_runner",
         str(write_fd)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        env=_child_env(),
        cwd=workdir,
        close_fds=True,
        # Its own session and process group, so a kill reaches anything it
        # managed to fork as well (Linux forbids that via RLIMIT_NPROC; macOS
        # does not), instead of leaving an orphan holding the result pipe.
        start_new_session=True,
    )
    os.close(write_fd)

    chunks: List[bytes] = []
    overflow = threading.Event()

    def _drain():
        received = 0
        with os.fdopen(read_fd, "rb") as pipe:
            while True:
                block = pipe.read(65536)
                if not block:
                    break
                received += len(block)
                if received > config.max_output_bytes:
                    overflow.set()  # stop buffering; the child is killed below
                    break
                chunks.append(block)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    # Drain stderr as it is written, keeping only the tail for the crash
    # message. Left unread until the child exits, a flood of library
    # warnings filled the pipe and blocked the child before it could write
    # its result, which the parent then reported as a timeout.
    stderr_tail = bytearray()

    def _drain_stderr():
        for block in iter(lambda: proc.stderr.read(65536), b""):
            stderr_tail.extend(block)
            del stderr_tail[:-_STDERR_TAIL_BYTES]

    stderr_reader = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_reader.start()

    try:
        proc.stdin.write(request)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass  # child died early; handled below

    reader.join(timeout=config.execution_timeout)

    if reader.is_alive():
        _kill_group(proc)
        proc.wait(timeout=5)
        shutil.rmtree(workdir, ignore_errors=True)
        raise _SandboxTimeout(
            f"Execution timed out after {config.execution_timeout} seconds"
        )

    if overflow.is_set():
        _kill_group(proc)
        proc.wait(timeout=5)
        shutil.rmtree(workdir, ignore_errors=True)
        raise _SandboxCrash(
            f"the result is larger than the "
            f"{config.max_output_bytes / 1024 ** 2:g} MB limit; return a "
            f"smaller table or figure"
        )

    proc.wait(timeout=5)
    _kill_group(proc)  # stragglers it forked, if any; nothing to do normally
    stderr_reader.join(timeout=5)
    shutil.rmtree(workdir, ignore_errors=True)

    if not chunks:
        # A negative return code means the child was killed by a signal.
        # The CPU rlimit (SIGXCPU) fires slightly before the parent's
        # wall-clock deadline, so report that as a timeout rather than a
        # crash — to the user it is the same situation.
        if proc.returncode is not None and proc.returncode < 0:
            raise _SandboxTimeout(
                f"Execution timed out after {config.execution_timeout} seconds"
            )
        stderr = bytes(stderr_tail).decode("utf-8", "replace")
        raise _SandboxCrash(stderr.strip()[-500:] or "no output")

    return _decode_response(b"".join(chunks))


# How many sandbox children may run at once in this app process. Each is a
# full interpreter with pandas/numpy/plotly loaded plus up to
# ``max_memory_bytes`` of its own, on a server every user shares; without a
# cap, simultaneous requests could start enough of them to exhaust it.
MAX_CONCURRENT_EXECUTIONS = 2
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_EXECUTIONS)


_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


def make_this_process_private() -> bool:
    """Stop other processes of this user reading this process's secrets.

    The app process holds ANTHROPIC_API_KEY in its environment. Linux lets
    any process of the same user read ``/proc/<pid>/environ`` and
    ``/proc/<pid>/mem``, so a sandbox child that got past the Python-level
    file guard could read the key there. Marking this process non-dumpable
    makes the kernel refuse those reads to anything without
    CAP_SYS_PTRACE (the child is never root). Side effect: no core dumps
    and no debugger attaching to the app process. The child starts
    dumpable again after exec, which is harmless: its environment is empty.

    Returns:
        True when the process is private (or there is no /proc to protect:
        not Linux), False if the setting did not take.
    """
    if not sys.platform.startswith("linux"):
        return True
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
        return libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


class _SandboxTimeout(Exception):
    pass


class _SandboxCrash(Exception):
    pass


class SandboxService:
    """Executes AI-generated code in a restricted environment.

    All methods are stateless static methods.
    """

    @staticmethod
    def execute(
        code: str,
        df: pd.DataFrame,
        config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
        lc: Optional[Any] = None,
    ) -> SandboxResult:
        """Validate and execute code in the sandbox.

        Args:
            code: Python source code to execute.
            df: DataFrame available to the code as ``df``.
            config: Sandbox configuration. Uses defaults if None.
            numeric_columns: Optional list of column names to coerce to
                numeric (``errors='coerce'``) on the DataFrame copy
                before exposing it to user code. Use this to make sample
                intensity columns reliably numeric so AI-generated
                arithmetic doesn't fail with ``TypeError: unsupported
                operand type(s) for /: 'str' and 'float'`` when the raw
                CSV inferred string dtypes.
            lc: Optional ``PipelineAPI`` bound into the code's globals as
                ``lc`` (rebound to the prepared copy of ``df``). Without it,
                every ``lc.*`` call explains that the pipeline is
                unavailable.

        Returns:
            SandboxResult with execution outcome.
        """
        if config is None:
            config = SandboxConfig()

        # Phase 1: AST validation (raises SandboxError on violation)
        try:
            validate_code(code)
        except SandboxError as e:
            return SandboxResult(
                success=False,
                error_message=str(e),
            )

        # Phase 2: Execute out-of-process. The AST validation above is
        # defense-in-depth, not the boundary — the boundary is the
        # hardened child process (see sandbox_runner._harden: OS limits on
        # writes, processes, CPU and memory; Python-level hooks only for
        # reads and network; no inherited secrets; killable).
        start_time = time.monotonic()

        # Fail closed, like the child's own limits: never start untrusted
        # code while the key is readable from this process.
        if not make_this_process_private():
            return SandboxResult(
                success=False,
                error_message=(
                    "The sandbox could not protect the app process, so the "
                    "code was not run."
                ),
            )

        if not _SLOTS.acquire(timeout=config.execution_timeout):
            return SandboxResult(
                success=False,
                error_message=(
                    "The sandbox is busy with other requests; "
                    "please try again in a moment."
                ),
                execution_time=time.monotonic() - start_time,
            )
        try:
            response = _run_in_subprocess(
                code, df, numeric_columns, config, lc=lc,
            )
        except _SandboxTimeout as e:
            return SandboxResult(
                success=False,
                error_message=str(e),
                execution_time=time.monotonic() - start_time,
            )
        except _SandboxCrash as e:
            return SandboxResult(
                success=False,
                error_message=f"Execution failed: {e}",
                execution_time=time.monotonic() - start_time,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error_message=f"Runtime error: {type(e).__name__}: {e}",
                execution_time=time.monotonic() - start_time,
            )
        finally:
            _SLOTS.release()

        elapsed = time.monotonic() - start_time

        if not response.get("ok"):
            return SandboxResult(
                success=False,
                error_message=f"Runtime error: {response.get('error', '')}",
                stdout=response.get("stdout", ""),
                execution_time=elapsed,
            )

        return SandboxResult(
            success=True,
            figure=response.get("figure"),
            result_df=response.get("result_df"),
            stdout=response.get("stdout", ""),
            execution_time=elapsed,
            methods=tuple(
                m for m in response.get("methods") or () if isinstance(m, str)
            ),
        )
