"""
Sandboxed code execution for AI-generated Python code.

Security-critical module. AI-generated code is validated via AST
analysis before execution in a restricted environment with curated
builtins, whitelisted globals, and a timeout.

Defense model: the sandbox is the security boundary, not AI compliance.
All AI output is treated as untrusted.
"""
import ast
import math
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    """
    success: bool
    figure: Optional[go.Figure] = None
    result_df: Optional[pd.DataFrame] = None
    stdout: str = ""
    error_message: str = ""
    execution_time: float = 0.0


# ---------------------------------------------------------------------------
# AST Validator
# ---------------------------------------------------------------------------

# Imports that are safe to allow in sandbox code
_WHITELISTED_IMPORTS: frozenset = frozenset({
    "math", "statistics", "re",
    # Already pre-loaded in globals as pd/np/px/go — but the AI sometimes
    # adds defensive imports (e.g. `import pandas as pd`) after an error.
    # Whitelist them so those imports are no-ops instead of fatal errors.
    "pandas", "numpy", "plotly",
})

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
            module_name = alias.name.split(".")[0]
            if module_name not in _WHITELISTED_IMPORTS:
                raise SandboxError(f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name not in _WHITELISTED_IMPORTS:
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
    """Import function that only allows whitelisted modules.

    Matches the AST validator's policy: the top-level package name must
    be in ``_WHITELISTED_IMPORTS``. Submodules of whitelisted packages
    (e.g. ``plotly.graph_objects``) are allowed.
    """
    top_level = name.split(".", 1)[0]
    if top_level not in _WHITELISTED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)


def _build_globals(df: pd.DataFrame) -> Dict[str, Any]:
    """Build the restricted globals dict for exec().

    Includes whitelisted libraries and the user's DataFrame.
    """
    captured_output: list = []

    def capturing_print(*args, **kwargs):
        captured_output.append(" ".join(str(a) for a in args))

    builtins = {**_SAFE_BUILTINS, "print": capturing_print, "__import__": _restricted_import}

    return {
        "__builtins__": builtins,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "math": math,
        "statistics": statistics,
        "re": re,
        "df": df.copy(),
        "_captured_output": captured_output,
    }


# ---------------------------------------------------------------------------
# Thread-based timeout execution
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _exec_with_timeout(compiled, exec_globals, timeout_seconds):
    """Execute compiled code in a thread with a timeout.

    Args:
        compiled: Compiled code object.
        exec_globals: Globals dict for exec().
        timeout_seconds: Maximum wall-clock seconds to allow.

    Raises:
        _TimeoutError: If execution exceeds the timeout.
        Exception: Any exception raised by the executed code.
    """
    result = {"error": None}

    def _target():
        try:
            exec(compiled, exec_globals)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running — we can't forcibly kill it, but
        # the daemon flag ensures it won't block process exit.
        raise _TimeoutError(f"Code execution timed out after {timeout_seconds} seconds")

    if result["error"] is not None:
        raise result["error"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SandboxService:
    """Executes AI-generated code in a restricted environment.

    All methods are stateless static methods.
    """

    @staticmethod
    def execute(
        code: str,
        df: pd.DataFrame,
        config: Optional[SandboxConfig] = None,
    ) -> SandboxResult:
        """Validate and execute code in the sandbox.

        Args:
            code: Python source code to execute.
            df: DataFrame available to the code as ``df``.
            config: Sandbox configuration. Uses defaults if None.

        Returns:
            SandboxResult with execution outcome.
        """
        if config is None:
            config = SandboxConfig()

        # Phase 1: AST validation (raises SandboxError on violation)
        try:
            tree = validate_code(code)
        except SandboxError as e:
            return SandboxResult(
                success=False,
                error_message=str(e),
            )

        # Phase 2: Execute in restricted environment
        exec_globals = _build_globals(df)
        start_time = time.monotonic()

        try:
            compiled = compile(tree, "<sandbox>", "exec")
            _exec_with_timeout(compiled, exec_globals, config.execution_timeout)
        except _TimeoutError:
            return SandboxResult(
                success=False,
                error_message=f"Execution timed out after {config.execution_timeout} seconds",
                execution_time=time.monotonic() - start_time,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error_message=f"Runtime error: {type(e).__name__}: {e}",
                execution_time=time.monotonic() - start_time,
            )

        elapsed = time.monotonic() - start_time
        stdout = "\n".join(exec_globals.get("_captured_output", []))

        # Extract results
        figure = exec_globals.get("fig")
        if figure is not None and not isinstance(figure, go.Figure):
            figure = None

        result_df = exec_globals.get("result_df")
        if result_df is not None and isinstance(result_df, pd.DataFrame):
            if len(result_df) > config.max_output_rows:
                result_df = result_df.head(config.max_output_rows)
        else:
            result_df = None

        return SandboxResult(
            success=True,
            figure=figure,
            result_df=result_df,
            stdout=stdout,
            execution_time=elapsed,
        )
