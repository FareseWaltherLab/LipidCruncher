"""
Child-process entry point for sandboxed execution.

Run as ``python -m app.services.ai_chat.sandbox_runner <result_fd>``:

1. reads a pickled request from **stdin**,
2. applies process-level hardening (see ``_harden``),
3. executes the AI-generated code against the restricted globals,
4. writes a JSON response to **<result_fd>**.

The response goes to a dedicated pipe rather than stdout because stdout can
be polluted by library chatter, and because ``RLIMIT_FSIZE`` is set to 0 —
a pipe is not a file, so the result can still be written while the code
itself is unable to create any file.

The response is JSON, never pickle: the parent decodes it, and unpickling
bytes that untrusted code could have shaped would run code in the parent.

Nothing here trusts the code it runs. The parent (``sandbox.py``) owns the
timeout and kills this process if it overruns.
"""
import ctypes
import errno
import importlib
import json
import os
import pickle
import platform
import sys


_READABLE_DEVICES = ('/dev/null', '/dev/urandom', '/dev/random')


def _readable_roots() -> list:
    """Directories the sandboxed code may read from: library code only."""
    import site
    import app

    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix,
             os.path.dirname(app.__file__)}
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except Exception:
        pass
    return [os.path.realpath(r) + os.sep for r in roots if r]


def _install_read_guard() -> None:
    """Refuse file opens and directory listings outside the readable roots."""
    roots = _readable_roots()

    def _allowed(path) -> bool:
        if isinstance(path, int):
            return True  # an already-open descriptor, e.g. the result pipe
        if path is None:
            path = os.getcwd()
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        real = os.path.realpath(os.fspath(path))
        if real in _READABLE_DEVICES:
            return True
        return any((real + os.sep).startswith(root) for root in roots)

    def _guard(event, args):
        if event in ('open', 'os.listdir', 'os.scandir') and not _allowed(args[0]):
            raise PermissionError(
                "Reading files is disabled in the sandbox; use the provided data."
            )

    sys.addaudithook(_guard)


# --- Kernel syscall filter (seccomp) ----------------------------------------
# x86_64 syscall numbers the filter refuses. The child needs none of them:
# its result goes over an inherited pipe.
_DENIED_SYSCALLS = {
    "socket": 41,
    "socketpair": 53,
    "ptrace": 101,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "io_uring_setup": 425,  # io_uring can open sockets without socket()
}
_AUDIT_ARCH_X86_64 = 0xC000003E
_X32_SYSCALL_BIT = 0x40000000
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
# Classic BPF opcodes.
_LD_W_ABS = 0x20     # BPF_LD | BPF_W | BPF_ABS
_JEQ_K = 0x15        # BPF_JMP | BPF_JEQ | BPF_K
_JGE_K = 0x35        # BPF_JMP | BPF_JGE | BPF_K
_RET_K = 0x06        # BPF_RET | BPF_K


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _syscall_filter_applies() -> bool:
    """The filter is written for Linux x86_64, what Fargate runs."""
    return sys.platform.startswith("linux") and platform.machine() == "x86_64"


def _syscall_filter() -> list:
    """The BPF program: refuse _DENIED_SYSCALLS with EPERM, allow the rest.

    Anything other than the native x86_64 ABI (i386 via int 0x80, x32) is
    refused outright: there the numbers above mean other calls.
    """
    deny = _SECCOMP_RET_ERRNO | errno.EPERM
    checks = [(_JGE_K, _X32_SYSCALL_BIT)] + [
        (_JEQ_K, nr) for nr in _DENIED_SYSCALLS.values()
    ]
    # 0: arch, 1: arch == x86_64 ? 3 : 2, 2: deny, 3: syscall number,
    # 4..: one check each (match -> deny), then allow, then deny.
    program = [
        (_LD_W_ABS, 0, 0, 4),
        (_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),
        (_RET_K, 0, 0, deny),
        (_LD_W_ABS, 0, 0, 0),
    ]
    deny_at = len(program) + len(checks) + 1
    for code, value in checks:
        program.append((code, deny_at - len(program) - 1, 0, value))
    program += [(_RET_K, 0, 0, _SECCOMP_RET_ALLOW), (_RET_K, 0, 0, deny)]
    return program


def _install_syscall_filter() -> None:
    """Have the kernel refuse network sockets and reads of other processes.

    The Python-level socket patch below sits in the same interpreter as the
    untrusted code, which can reach libc through the bound modules. A
    seccomp filter cannot be removed by the process, and needs no privilege
    once no_new_privs is set. Raises if it does not take.
    """
    program = _syscall_filter()
    filters = (_SockFilter * len(program))(*(_SockFilter(*ins) for ins in program))
    fprog = _SockFprog(len(program), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise RuntimeError(f"no_new_privs failed (errno {ctypes.get_errno()})")
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise RuntimeError(f"seccomp filter failed (errno {ctypes.get_errno()})")
    import _socket
    try:
        _socket.socket()
    except OSError:
        return
    raise RuntimeError("seccomp filter did not take effect")


def _mapped_bytes() -> int:
    """Address space this process maps now (Linux)."""
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[0])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _harden(cpu_seconds: int, memory_bytes: int) -> None:
    """Reduce what this process can do before running untrusted code.

    Applied in the child, after the heavy imports are already resolved, so
    the limits constrain the AI's code rather than our own start-up.

    Fails closed: raises if any control cannot be put in place, and
    ``main`` then refuses to run the code. These used to be best-effort
    with every error swallowed, which hid that the deployed container ran
    as root and so had no process limit at all.
    """
    import resource

    # Load matplotlib's font list before any limit. On first use it reads
    # (or rebuilds, scanning the system font folders) a cache that the
    # sandbox keeps in the system temp dir, which the read guard must not
    # open up; and the rebuild starts a thread, which RLIMIT_NPROC forbids on
    # Linux, where threads count as processes. Drawing afterwards only reads
    # matplotlib's own fonts.
    import matplotlib.font_manager  # noqa: F401

    # Root (CAP_SYS_RESOURCE) is exempt from RLIMIT_NPROC on Linux, so the
    # "no new processes" limit below would be set and silently ignored.
    if os.geteuid() == 0:
        raise RuntimeError("running as root, which ignores the process limit")

    limits = {
        # --- No filesystem writes -----------------------------------------
        # RLIMIT_FSIZE caps the size of files the process may create; 0
        # means it may create none. This is what stops `df.to_csv('/…')`,
        # `scipy.io.savemat`, and friends. Pipes are unaffected, so the
        # result channel still works.
        "RLIMIT_FSIZE": (0, 0),
        # --- Bounded CPU --------------------------------------------------
        # Backstop for the parent's wall-clock timeout: a busy loop that
        # ignores signals still dies here.
        "RLIMIT_CPU": (cpu_seconds, cpu_seconds),
        # --- No new processes ---------------------------------------------
        # Blocks os.system / subprocess even if the code reaches them
        # through a library's module attributes (e.g. pd.compat.os), which
        # the AST validator cannot see.
        "RLIMIT_NPROC": (0, 0),
    }
    # --- Bounded memory ---------------------------------------------------
    # Without it one `np.ones(10**10)` takes down the whole task and every
    # user's session with it. The cap is what the code may map on top of
    # what the process maps after its imports, so the base footprint (which
    # varies with platform and BLAS build) does not eat into it. macOS
    # cannot lower RLIMIT_AS, so there it is not applied.
    if sys.platform.startswith("linux"):
        cap = _mapped_bytes() + memory_bytes
        limits["RLIMIT_AS"] = (cap, cap)
    for name, value in limits.items():
        which = getattr(resource, name)
        resource.setrlimit(which, value)
        if resource.getrlimit(which) != value:
            raise RuntimeError(f"{name} did not take effect")

    # --- No reading files outside the libraries -------------------------
    # The child inherits no environment, but it could still *read*:
    # `pd.read_csv('/proc/<parent>/environ')` printed the parent's
    # environment (where ECS puts the API key), and any file baked into the
    # image was readable. An audit hook cannot be removed by Python code; it
    # refuses every open or directory listing outside the interpreter's own
    # install, the site-packages, the app package (for lazy imports) and
    # matplotlib's cache.
    _install_read_guard()

    # --- No network, no reading other processes (kernel) ----------------
    # On Linux x86_64 the kernel itself refuses socket(), ptrace() and
    # process_vm_readv/writev(), whatever path the code finds to libc.
    if _syscall_filter_applies():
        _install_syscall_filter()

    # --- No network ------------------------------------------------------
    # Neuter the already-imported socket module in place, so every existing
    # reference to it (including ones held inside pandas/urllib) sees the
    # blocked version. Covers `pd.read_csv('http://…')` exfiltration.
    import socket

    def _blocked(*args, **kwargs):
        raise OSError("Network access is disabled in the sandbox")

    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.socketpair = _blocked
    for _name in ("gethostbyname", "getaddrinfo", "create_server"):
        if hasattr(socket, _name):
            setattr(socket, _name, _blocked)


def main() -> int:
    result_fd = int(sys.argv[1])

    request = pickle.loads(sys.stdin.buffer.read())

    # Imported after the request is read so the cost is paid once, and so
    # a malformed request fails before any heavy work.
    from app.services.ai_chat.sandbox import (
        ANALYSIS_MODULES, build_sandbox_globals, frame_to_wire,
    )

    # Resolve the modules the code may import before _harden: imported after
    # it, a library that subclasses socket.socket (ssl, which scikit-learn
    # reaches through joblib and asyncio) fails, because _harden replaces
    # socket.socket with a function.
    for module in ANALYSIS_MODULES:
        importlib.import_module(module)

    exec_globals = build_sandbox_globals(
        df=request["df"],
        numeric_columns=request.get("numeric_columns"),
        lc=request.get("lc"),
    )

    response = {
        "ok": False,
        "error": "",
        "figure": None,
        "result_df": None,
        "stdout": "",
        "methods": [],
    }

    try:
        _harden(
            cpu_seconds=int(request.get("cpu_seconds", 10)),
            memory_bytes=int(request.get("memory_bytes", 1024 ** 3)),
        )
    except Exception as e:  # noqa: BLE001 - never run the code unconfined
        response["error"] = (
            f"The sandbox could not apply its safety limits ({e}), "
            f"so the code was not run."
        )
    else:
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
                response["figure"] = figure.to_json()
            if isinstance(result_df, pd.DataFrame):
                if max_rows is not None and len(result_df) > max_rows:
                    result_df = result_df.head(max_rows)
                response["result_df"] = frame_to_wire(result_df)
        except Exception:
            pass
        try:
            # The statistical choices `lc` made, for the app to show.
            response["methods"] = list(exec_globals["lc"].methods_used)
        except Exception:
            pass  # no pipeline in this run (lc unavailable)

    payload = json.dumps(response, default=str).encode("utf-8")
    with os.fdopen(result_fd, "wb") as out:
        out.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
