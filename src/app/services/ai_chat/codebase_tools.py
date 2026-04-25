"""
Codebase exploration tools for AI tool use.

Provides tool definitions (for the Anthropic API tools parameter) and
execution logic that lets the AI search, read, and list project files
at query time. All tools are read-only with path security validation.

No Streamlit imports — pure service layer.
"""
import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Optional

from .exceptions import AIChatError

# Project root: four levels up from this file
# (this file is at src/app/services/ai_chat/codebase_tools.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Directories the AI is allowed to read
_ALLOWED_DIRS = [
    _PROJECT_ROOT / "src" / "app",
    _PROJECT_ROOT / "tests",
    _PROJECT_ROOT / "docs",
]

# Individual files the AI is allowed to read
_ALLOWED_FILES = [
    _PROJECT_ROOT / "CODEBASE.md",
]

# Patterns that are always blocked (case-insensitive match on path components)
_BLOCKED_PATTERNS = [
    ".env",
    "secrets.toml",
    "__pycache__",
    ".git",
    ".pyc",
]

# Maximum characters returned by any single tool call
_MAX_OUTPUT_CHARS = 8000

# Higher limit for documentation reads (papers, supplementary methods)
_MAX_DOC_CHARS = 40000

# Maximum number of search results returned
_MAX_SEARCH_RESULTS = 15

# Context lines to show above and below each search match
_SEARCH_CONTEXT_LINES = 10

# Documentation name → file path mapping
_DOC_MAP = {
    "CODEBASE": _PROJECT_ROOT / "CODEBASE.md",
    "paper": _PROJECT_ROOT / "docs" / "lipidcruncher_paper.md",
    "supplementary_methods": _PROJECT_ROOT / "docs" / "supplementary_methods.md",
    "release_notes": _PROJECT_ROOT / "src" / "app" / "ui" / "landing_page.py",
    "analysis_docs": _PROJECT_ROOT / "src" / "app" / "ui" / "content" / "analysis_docs.py",
    "normalization_docs": _PROJECT_ROOT / "src" / "app" / "ui" / "content" / "normalization_docs.py",
    "processing_docs": _PROJECT_ROOT / "src" / "app" / "ui" / "content" / "processing_docs.py",
    "standards_help": _PROJECT_ROOT / "src" / "app" / "ui" / "content" / "standards_help.py",
}

# ---------------------------------------------------------------------------
# Tool definitions for the Anthropic API
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_codebase",
        "description": (
            "Search for keywords or patterns across the LipidCruncher codebase. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find where specific functions, classes, or concepts "
            "are implemented."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term or pattern to look for in the codebase.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": (
                        "Optional glob pattern to filter files "
                        "(e.g. '*.py', 'test_*.py'). Defaults to all files."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file_section",
        "description": (
            "Read a specific file or section of a file from the project. "
            "Returns file contents with line numbers. Use this after "
            "search_codebase to read the full context around a match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the file, relative to the project root "
                        "(e.g. 'src/app/services/ai_chat/ai_client.py')."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-based). Defaults to 1.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (inclusive). Defaults to end of file.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in the project directory tree. Use this to explore "
            "the project structure before searching or reading specific files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory to list, relative to the project root "
                        "(e.g. 'src/app/services'). Defaults to 'src/app'."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Optional glob pattern to filter results "
                        "(e.g. '*.py'). Defaults to all files."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_documentation",
        "description": (
            "Read project documentation by name. Available documents: "
            "paper (LipidCruncher publication — authoritative description "
            "of features, workflow, and scientific context), "
            "supplementary_methods (detailed technical specifications "
            "for all processing steps, normalization, statistics, and "
            "visualizations), "
            "release_notes (What's New changelogs for each version — "
            "features added after the paper was written), "
            "CODEBASE (project architecture overview), "
            "analysis_docs (visualization & analysis features), "
            "normalization_docs (normalization methods), "
            "processing_docs (data processing pipeline), "
            "standards_help (internal standards help). "
            "Prefer 'paper' and 'supplementary_methods' for answering "
            "questions about LipidCruncher features and methods. Use "
            "'release_notes' for features not covered in the paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": (
                        "Document name. Prefer 'paper' or "
                        "'supplementary_methods' for feature questions."
                    ),
                    "enum": list(_DOC_MAP.keys()),
                },
            },
            "required": ["doc_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Dispatch a tool call and return its text result.

    Args:
        name: Tool name (must match a name in TOOL_DEFINITIONS).
        tool_input: Input dict matching the tool's input_schema.

    Returns:
        Tool result as a string, truncated to _MAX_OUTPUT_CHARS.

    Raises:
        AIChatError: If the tool name is unknown or execution fails.
    """
    dispatchers = {
        "search_codebase": _search_codebase,
        "read_file_section": _read_file_section,
        "list_files": _list_files,
        "read_documentation": _read_documentation,
    }
    if name not in dispatchers:
        raise AIChatError(f"Unknown tool: {name}")

    try:
        result = dispatchers[name](**tool_input)
    except AIChatError:
        raise
    except Exception as e:
        raise AIChatError(f"Tool '{name}' failed: {e}") from e

    max_chars = _MAX_DOC_CHARS if name == "read_documentation" else _MAX_OUTPUT_CHARS
    return _truncate(result, max_chars)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _search_codebase(query: str, file_pattern: Optional[str] = None) -> str:
    """Search for a pattern across allowed project files.

    Returns matching lines with surrounding context so the AI can
    understand the match without needing a follow-up read_file_section.
    """
    if not query.strip():
        return "Error: query must not be empty."

    results: list[str] = []
    n_matches = 0
    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    except re.error:
        return f"Error: invalid search pattern: {query}"

    for allowed_dir in _ALLOWED_DIRS:
        if not allowed_dir.exists():
            continue
        for root, _dirs, files in os.walk(allowed_dir):
            for fname in files:
                fpath = Path(root) / fname
                if not _is_path_safe(fpath):
                    continue
                if file_pattern and not fnmatch.fnmatch(fname, file_pattern):
                    continue
                try:
                    text = fpath.read_text(errors="replace")
                except OSError:
                    continue
                rel = fpath.relative_to(_PROJECT_ROOT)
                all_lines = text.splitlines()
                for lineno, line in enumerate(all_lines):
                    if pattern.search(line):
                        n_matches += 1
                        # Include context lines around the match
                        start = max(0, lineno - _SEARCH_CONTEXT_LINES)
                        end = min(len(all_lines), lineno + _SEARCH_CONTEXT_LINES + 1)
                        context_block = []
                        for i in range(start, end):
                            marker = ">>>" if i == lineno else "   "
                            context_block.append(
                                f"{marker} {i + 1}: {all_lines[i].rstrip()}"
                            )
                        results.append(
                            f"--- {rel}:{lineno + 1} ---\n"
                            + "\n".join(context_block)
                        )
                        if n_matches >= _MAX_SEARCH_RESULTS:
                            break
                if n_matches >= _MAX_SEARCH_RESULTS:
                    break
            if n_matches >= _MAX_SEARCH_RESULTS:
                break
        if n_matches >= _MAX_SEARCH_RESULTS:
            break

    if not results:
        return f"No matches found for '{query}'."

    header = f"Found {n_matches} match(es) for '{query}':\n\n"
    return header + "\n\n".join(results)


def _read_file_section(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read a file (or section) after validating the path."""
    resolved = _resolve_and_validate(file_path)
    if not resolved.is_file():
        return f"Error: file not found: {file_path}"

    try:
        lines = resolved.read_text(errors="replace").splitlines()
    except OSError as e:
        return f"Error reading file: {e}"

    start = max(1, start_line or 1)
    end = min(len(lines), end_line or len(lines))

    if start > len(lines):
        return f"Error: start_line {start} exceeds file length ({len(lines)} lines)."

    selected = lines[start - 1 : end]
    numbered = [f"{i}: {line}" for i, line in enumerate(selected, start)]
    header = f"File: {file_path} (lines {start}-{end} of {len(lines)})\n"
    return header + "\n".join(numbered)


def _list_files(
    directory: Optional[str] = None,
    pattern: Optional[str] = None,
) -> str:
    """List files under an allowed directory."""
    rel_dir = directory or "src/app"
    target = (_PROJECT_ROOT / rel_dir).resolve()

    # Ensure target is within an allowed directory
    if not _is_within_allowed(target):
        return f"Error: access denied to directory '{rel_dir}'."
    if not target.is_dir():
        return f"Error: directory not found: {rel_dir}"

    file_list: list[str] = []
    for root, _dirs, files in os.walk(target):
        for fname in files:
            fpath = Path(root) / fname
            if not _is_path_safe(fpath):
                continue
            if pattern and not fnmatch.fnmatch(fname, pattern):
                continue
            file_list.append(str(fpath.relative_to(_PROJECT_ROOT)))

    file_list.sort()
    if not file_list:
        return f"No files found in '{rel_dir}'" + (f" matching '{pattern}'" if pattern else "") + "."

    header = f"{len(file_list)} file(s) in '{rel_dir}'" + (f" matching '{pattern}'" if pattern else "") + ":\n"
    return header + "\n".join(file_list)


def _read_documentation(doc_name: str) -> str:
    """Read a named documentation file."""
    if doc_name not in _DOC_MAP:
        available = ", ".join(_DOC_MAP.keys())
        return f"Error: unknown document '{doc_name}'. Available: {available}"

    doc_path = _DOC_MAP[doc_name]
    if not doc_path.is_file():
        return f"Error: document file not found: {doc_path.name}"

    try:
        content = doc_path.read_text(errors="replace")
    except OSError as e:
        return f"Error reading document: {e}"

    return content


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _resolve_and_validate(rel_path: str) -> Path:
    """Resolve a relative path and validate it is within allowed areas.

    Args:
        rel_path: Path relative to the project root.

    Returns:
        Resolved absolute Path.

    Raises:
        AIChatError: If the path is outside allowed directories or matches
            a blocked pattern.
    """
    resolved = (_PROJECT_ROOT / rel_path).resolve()

    if not _is_within_allowed(resolved):
        raise AIChatError(f"Access denied: '{rel_path}' is outside allowed directories.")

    if not _is_path_safe(resolved):
        raise AIChatError(f"Access denied: '{rel_path}' matches a blocked pattern.")

    return resolved


def _is_within_allowed(path: Path) -> bool:
    """Check if a path is within an allowed directory or is an allowed file."""
    resolved = path.resolve()
    for allowed_dir in _ALLOWED_DIRS:
        try:
            resolved.relative_to(allowed_dir.resolve())
            return True
        except ValueError:
            continue
    for allowed_file in _ALLOWED_FILES:
        if resolved == allowed_file.resolve():
            return True
    return False


def _is_path_safe(path: Path) -> bool:
    """Check that no path component matches a blocked pattern."""
    path_str = str(path).lower()
    for blocked in _BLOCKED_PATTERNS:
        if blocked in path_str:
            return False
    return True


def _truncate(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate text to max_chars, appending a notice if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Output truncated — use start_line/end_line to read specific sections]"
