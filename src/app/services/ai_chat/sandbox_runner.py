"""
Child-process entry point for sandboxed execution.

Run as ``python -m app.services.ai_chat.sandbox_runner <result_fd>``:

1. reads a pickled request from **stdin**,
2. applies process-level hardening (see ``_harden``),
3. executes the AI-generated code against the restricted globals,
4. writes a pickled response to **<result_fd>**.

The response goes to a dedicated pipe rather than stdout because stdout can
be polluted by library chatter, and because ``RLIMIT_FSIZE`` is set to 0 —
a pipe is not a file, so the result can still be written while the code
itself is unable to create any file.

Nothing here trusts the code it runs. The parent (``sandbox.py``) owns the
timeout and kills this process if it overruns.
"""
import os
import pickle
import sys


def _harden(cpu_seconds: int) -> None:
    """Reduce what this process can do before running untrusted code.

    Applied in the child, after the heavy imports are already resolved, so
    the limits constrain the AI's code rather than our own start-up.

    Best-effort by design: each control is independent, and a control that
    is unavailable on the host must not prevent the others from applying.
    """
    # --- No filesystem writes -------------------------------------------
    # RLIMIT_FSIZE caps the size of files the process may create; 0 means
    # it may create none. This is what stops `df.to_csv('/…')`,
    # `scipy.io.savemat`, and friends. Pipes are unaffected, so the result
    # channel still works.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except Exception:
        pass

    # --- Bounded CPU ----------------------------------------------------
    # Backstop for the parent's wall-clock timeout: a busy loop that
    # ignores signals still dies here.
    try:
        import resource
        resource.setrlimit(
            resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds),
        )
    except Exception:
        pass

    # --- No new processes ------------------------------------------------
    # Blocks os.system / subprocess even if the code reaches them through
    # a library's module attributes (e.g. pd.compat.os), which the AST
    # validator cannot see.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except Exception:
        pass

    # --- No network ------------------------------------------------------
    # Neuter the already-imported socket module in place, so every existing
    # reference to it (including ones held inside pandas/urllib) sees the
    # blocked version. Covers `pd.read_csv('http://…')` exfiltration.
    try:
        import socket

        def _blocked(*args, **kwargs):
            raise OSError("Network access is disabled in the sandbox")

        socket.socket = _blocked
        socket.create_connection = _blocked
        socket.socketpair = _blocked
        for _name in ("gethostbyname", "getaddrinfo", "create_server"):
            if hasattr(socket, _name):
                setattr(socket, _name, _blocked)
    except Exception:
        pass


def main() -> int:
    result_fd = int(sys.argv[1])

    request = pickle.loads(sys.stdin.buffer.read())

    # Imported after the request is read so the cost is paid once, and so
    # a malformed request fails before any heavy work.
    from app.services.ai_chat.sandbox import build_sandbox_globals

    exec_globals = build_sandbox_globals(
        df=request["df"],
        numeric_columns=request.get("numeric_columns"),
        experiment=request.get("experiment"),
        lc=request.get("lc"),
    )

    _harden(cpu_seconds=int(request.get("cpu_seconds", 10)))

    response = {
        "ok": False,
        "error": "",
        "figure": None,
        "result_df": None,
        "stdout": "",
    }

    try:
        exec(compile(request["code"], "<sandbox>", "exec"), exec_globals)
        response["ok"] = True
    except BaseException as e:  # noqa: BLE001 - report, never leak upward
        response["error"] = f"{type(e).__name__}: {e}"

    response["stdout"] = "\n".join(exec_globals.get("_captured_output", []))

    if response["ok"]:
        figure = exec_globals.get("fig")
        result_df = exec_globals.get("result_df")
        max_rows = request.get("max_output_rows")

        try:
            import pandas as pd
            import plotly.graph_objects as go

            if isinstance(figure, go.Figure):
                response["figure"] = figure
            if isinstance(result_df, pd.DataFrame):
                if max_rows is not None and len(result_df) > max_rows:
                    result_df = result_df.head(max_rows)
                response["result_df"] = result_df
        except Exception:
            pass

    payload = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
    with os.fdopen(result_fd, "wb") as out:
        out.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
