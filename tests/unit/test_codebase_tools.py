"""
Unit tests for codebase exploration tools.

Tests path security validation, search functionality, file reading,
directory listing, documentation reading, and output truncation.
"""
import pytest

from app.services.ai_chat.codebase_tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    _is_path_safe,
    _is_within_allowed,
    _resolve_and_validate,
    _truncate,
    _MAX_OUTPUT_CHARS,
    _MAX_DOC_CHARS,
    _MAX_SEARCH_RESULTS,
    _PROJECT_ROOT,
)
from app.services.ai_chat.exceptions import AIChatError


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    """Verify the tool definitions are well-formed for the Anthropic API."""

    def test_four_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == {"search_codebase", "read_file_section", "list_files", "read_documentation"}

    def test_each_tool_has_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

class TestPathSecurity:
    """Verify that path validation blocks unauthorized access."""

    def test_allowed_src_app_path(self):
        path = _PROJECT_ROOT / "src" / "app" / "services" / "ai_chat" / "ai_client.py"
        assert _is_within_allowed(path) is True

    def test_allowed_tests_path(self):
        path = _PROJECT_ROOT / "tests" / "unit" / "test_codebase_tools.py"
        assert _is_within_allowed(path) is True

    def test_allowed_codebase_md(self):
        path = _PROJECT_ROOT / "CODEBASE.md"
        assert _is_within_allowed(path) is True

    def test_blocked_parent_directory(self):
        path = _PROJECT_ROOT.parent / "some_file.py"
        assert _is_within_allowed(path) is False

    def test_blocked_root_level_file(self):
        path = _PROJECT_ROOT / "setup.py"
        assert _is_within_allowed(path) is False

    def test_is_path_safe_blocks_env(self):
        path = _PROJECT_ROOT / ".env"
        assert _is_path_safe(path) is False

    def test_is_path_safe_blocks_secrets(self):
        path = _PROJECT_ROOT / "secrets.toml"
        assert _is_path_safe(path) is False

    def test_is_path_safe_blocks_pycache(self):
        path = _PROJECT_ROOT / "src" / "app" / "__pycache__" / "foo.pyc"
        assert _is_path_safe(path) is False

    def test_is_path_safe_blocks_git(self):
        path = _PROJECT_ROOT / ".git" / "config"
        assert _is_path_safe(path) is False

    def test_is_path_safe_allows_normal_file(self):
        path = _PROJECT_ROOT / "src" / "app" / "models" / "chat.py"
        assert _is_path_safe(path) is True

    def test_resolve_and_validate_rejects_traversal(self):
        with pytest.raises(AIChatError, match="outside allowed"):
            _resolve_and_validate("../../../etc/passwd")

    def test_resolve_and_validate_rejects_env(self):
        with pytest.raises(AIChatError, match="outside allowed"):
            _resolve_and_validate(".env")

    def test_resolve_and_validate_accepts_valid_path(self):
        result = _resolve_and_validate("src/app/services/ai_chat/ai_client.py")
        assert result.is_file()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    """Verify output truncation works correctly."""

    def test_short_text_not_truncated(self):
        text = "short"
        assert _truncate(text) == text

    def test_exact_limit_not_truncated(self):
        text = "x" * _MAX_OUTPUT_CHARS
        assert _truncate(text) == text

    def test_over_limit_truncated(self):
        text = "x" * (_MAX_OUTPUT_CHARS + 100)
        result = _truncate(text)
        assert len(result) < len(text)
        assert result.startswith("x" * 100)
        assert "[Output truncated" in result

    def test_truncated_output_ends_with_notice(self):
        text = "x" * (_MAX_OUTPUT_CHARS + 1)
        result = _truncate(text)
        assert result.endswith("[Output truncated — use start_line/end_line to read specific sections]")


# ---------------------------------------------------------------------------
# search_codebase tool
# ---------------------------------------------------------------------------

class TestSearchCodebase:
    """Test the search_codebase tool."""

    def test_finds_known_class(self):
        result = execute_tool("search_codebase", {"query": "class AIClientService"})
        assert "ai_client.py" in result
        assert "class AIClientService" in result

    def test_returns_line_numbers(self):
        result = execute_tool("search_codebase", {"query": "class AIClientService"})
        # Format is file:lineno: content
        assert ":" in result

    def test_no_matches(self):
        # Use a query that won't appear anywhere (not even in this file)
        result = execute_tool("search_codebase", {
            "query": "zzz999qqq888",
            "file_pattern": "*.md",
        })
        assert "No matches found" in result

    def test_empty_query_returns_error(self):
        result = execute_tool("search_codebase", {"query": "   "})
        assert "Error" in result or "empty" in result.lower()

    def test_file_pattern_filter(self):
        result = execute_tool("search_codebase", {
            "query": "import pytest",
            "file_pattern": "test_*.py",
        })
        # Search output now includes context lines around each match, so we
        # only check the file-header lines (--- path:lineno ---) to confirm
        # the pattern filter was honored.
        if "No matches" not in result:
            header_lines = [
                line for line in result.split("\n")
                if line.startswith("--- ") and line.endswith(" ---")
            ]
            assert header_lines, "Expected at least one match header line"
            for line in header_lines:
                assert "test_" in line, f"Pattern filter violated: {line!r}"

    def test_results_capped(self):
        # Search for a very common term — capped at _MAX_SEARCH_RESULTS.
        result = execute_tool("search_codebase", {"query": "import"})
        assert f"Found {_MAX_SEARCH_RESULTS} match(es)" in result


# ---------------------------------------------------------------------------
# read_file_section tool
# ---------------------------------------------------------------------------

class TestReadFileSection:
    """Test the read_file_section tool."""

    def test_read_full_file(self):
        result = execute_tool("read_file_section", {
            "file_path": "src/app/services/ai_chat/exceptions.py",
        })
        assert "class AIChatError" in result
        assert "class AIClientError" in result

    def test_read_line_range(self):
        result = execute_tool("read_file_section", {
            "file_path": "src/app/services/ai_chat/ai_client.py",
            "start_line": 1,
            "end_line": 5,
        })
        assert "lines 1-5" in result
        # Should have exactly 5 numbered lines of content
        content_lines = [line for line in result.split("\n") if line and not line.startswith("File:")]
        assert len(content_lines) == 5

    def test_nonexistent_file(self):
        result = execute_tool("read_file_section", {
            "file_path": "src/app/nonexistent_file.py",
        })
        assert "not found" in result.lower()

    def test_blocked_path_raises(self):
        with pytest.raises(AIChatError):
            execute_tool("read_file_section", {"file_path": "../../../etc/passwd"})

    def test_start_line_beyond_file(self):
        result = execute_tool("read_file_section", {
            "file_path": "src/app/services/ai_chat/exceptions.py",
            "start_line": 99999,
        })
        assert "exceeds" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# list_files tool
# ---------------------------------------------------------------------------

class TestListFiles:
    """Test the list_files tool."""

    def test_default_lists_src_app(self):
        result = execute_tool("list_files", {})
        assert "src/app/" in result

    def test_specific_directory(self):
        result = execute_tool("list_files", {"directory": "src/app/services/ai_chat"})
        assert "ai_client.py" in result
        assert "codebase_tools.py" in result

    def test_pattern_filter(self):
        result = execute_tool("list_files", {
            "directory": "src/app/services/ai_chat",
            "pattern": "*.py",
        })
        assert "ai_client.py" in result
        # __pycache__ files should be filtered out by safety check
        assert "__pycache__" not in result

    def test_blocked_directory(self):
        result = execute_tool("list_files", {"directory": ".git"})
        assert "denied" in result.lower() or "error" in result.lower()

    def test_nonexistent_directory(self):
        result = execute_tool("list_files", {"directory": "src/app/nonexistent"})
        assert "not found" in result.lower()

    def test_pycache_excluded(self):
        result = execute_tool("list_files", {"directory": "src/app/services/ai_chat"})
        assert ".pyc" not in result


# ---------------------------------------------------------------------------
# read_documentation tool
# ---------------------------------------------------------------------------

class TestReadDocumentation:
    """Test the read_documentation tool."""

    def test_read_codebase_doc(self):
        result = execute_tool("read_documentation", {"doc_name": "CODEBASE"})
        assert "LipidCruncher" in result

    def test_read_analysis_docs(self):
        result = execute_tool("read_documentation", {"doc_name": "analysis_docs"})
        assert len(result) > 0

    def test_unknown_doc_name(self):
        result = execute_tool("read_documentation", {"doc_name": "nonexistent"})
        assert "unknown" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------

class TestExecuteTool:
    """Test the tool dispatcher."""

    def test_unknown_tool_raises(self):
        with pytest.raises(AIChatError, match="Unknown tool"):
            execute_tool("nonexistent_tool", {})

    def test_result_is_string(self):
        result = execute_tool("list_files", {"directory": "src/app/services/ai_chat"})
        assert isinstance(result, str)

    def test_result_truncated_if_large(self):
        # read_documentation uses the higher _MAX_DOC_CHARS cap (papers and
        # supplementary methods need full context); other tools use
        # _MAX_OUTPUT_CHARS. The result must respect the per-tool cap, with
        # a small slack for the truncation notice when it triggers.
        result = execute_tool("read_documentation", {"doc_name": "CODEBASE"})
        assert len(result) <= _MAX_DOC_CHARS + 200


class TestUntrackedDocsStayPrivate:
    """Regression (S9): all of docs/ was readable, and on a local checkout
    that folder also holds untracked notes (deployment details, runbooks).
    Only the two tracked papers may be read."""

    @pytest.fixture
    def private_note(self):
        import uuid

        # Neither appears in any tracked file; the content token is kept
        # out of the file name, which error messages echo back.
        token = uuid.uuid4().hex
        path = _PROJECT_ROOT / "docs" / f"_private_note_{uuid.uuid4().hex}.md"
        path.write_text(f"account id {token}\n")
        try:
            yield path, token
        finally:
            path.unlink()

    def test_untracked_doc_cannot_be_read(self, private_note):
        path, _ = private_note
        with pytest.raises(AIChatError, match="Access denied"):
            execute_tool("read_file_section", {
                "file_path": f"docs/{path.name}",
            })

    def test_untracked_doc_is_not_searched(self, private_note):
        _, token = private_note
        result = execute_tool("search_codebase", {"query": token})
        assert "No matches found" in result

    def test_docs_folder_cannot_be_listed(self, private_note):
        path, _ = private_note
        result = execute_tool("list_files", {"directory": "docs"})
        assert path.name not in result

    def test_papers_are_still_readable_and_searchable(self):
        result = execute_tool("read_file_section", {
            "file_path": "docs/supplementary_methods.md",
        })
        assert "Error" not in result.splitlines()[0]
        found = execute_tool("search_codebase", {
            "query": "Alexander-Govern", "file_pattern": "supplementary_methods.md",
        })
        assert "docs/supplementary_methods.md" in found
